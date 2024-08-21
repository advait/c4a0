use std::{
    collections::{BTreeMap, HashMap, HashSet},
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use crossbeam::thread;
use crossbeam_channel::{bounded, Receiver, RecvError, Sender};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{
    c4r::Pos,
    mcts::MctsGame,
    types::{EvalPosResult, EvalPosT, GameMetadata, GameResult, ModelID},
};

/// Generate training samples with self play and MCTS.
///
/// We use a batched NN forward pass to expand a given node (to determine the initial policy values
/// based on the NN's output policy). Because we want to batch these NN calls for performance, we
/// partially compute many MCTS traversals simultaneously (via [MctsThread]), pausing each until we
/// reach the node expansion phase. Then we are able to batch several NN calls simultaneously
/// (via [NNThread]). This process ping-pongs until the game reaches a terminal state after which
/// it is added to `done_queue`.
///
/// We use one [NNThread] and (n-1) [MctsThread]s (where n=core count).
/// The thread termination mechanism is as follows:
/// 1. [MctsThread] check whether we have finished all games via `n_games_remaining` atomic. When
///    the first thread detects all work is complete, it sends a [MctsJob::PoisonPill] to all
///    remaining [MctsThread]s, resulting in all of these threads completing.
/// 2. When the last [MctsThread] completes, it drops the last `nn_queue_tx`
///    [crossbeam_channel::Sender], causing the `nn_queue_rx` [crossbeam_channel::Receiver] to
///    close. This notifies the [NNThread], allowing it to close.
/// 3. The main thread simply waits for all threads to complete and then returns the results
///    from the `done_queue`.
pub fn self_play<E: EvalPosT + Send + Sync>(
    eval_pos: E,
    reqs: Vec<GameMetadata>,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
) -> Vec<GameResult> {
    let n_games = reqs.len();
    let (pb_game_done, pb_nn_eval, pb_mcts_iter) = init_progress_bars(n_games);
    let (nn_queue_tx, nn_queue_rx) = bounded::<MctsGame>(n_games);
    let (mcts_queue_tx, mcts_queue_rx) = bounded::<MctsJob>(n_games);
    let (done_queue_tx, done_queue_rx) = bounded::<GameResult>(n_games);
    let n_games_remaining = Arc::new(AtomicUsize::new(n_games));

    // Create initial games
    for req in reqs {
        let game = MctsGame::new_from_pos(Pos::default(), req);
        nn_queue_tx.send(game).unwrap();
    }

    thread::scope(|s| {
        // NN batch inference thread
        let mcts_queue = mcts_queue_tx.clone();
        s.builder()
            .name("nn_thread".into())
            .spawn(move |_| {
                NNThread::new(
                    nn_queue_rx,
                    mcts_queue,
                    max_nn_batch_size,
                    eval_pos,
                    pb_nn_eval,
                )
                .loop_until_close()
            })
            .unwrap();

        // MCTS threads
        let n_mcts_threads = usize::max(1, num_cpus::get() - 1);
        for i in 0..n_mcts_threads {
            let nn_queue_tx = nn_queue_tx.clone();
            let mcts_queue_tx = mcts_queue_tx.clone();
            let mcts_queue_rx = mcts_queue_rx.clone();
            let done_queue_tx = done_queue_tx.clone();
            let n_games_remaining = Arc::clone(&n_games_remaining);
            let pb_game_done = pb_game_done.clone();
            let pb_mcts_iter = pb_mcts_iter.clone();
            s.builder()
                .name(format!("mcts_thread {}", i))
                .spawn(move |_| {
                    MctsThread {
                        nn_queue_tx,
                        mcts_queue_tx,
                        mcts_queue_rx,
                        done_queue_tx,
                        n_games_remaining,
                        n_mcts_iterations,
                        n_mcts_threads,
                        c_exploration,
                        c_ply_penalty,
                        pb_game_done,
                        pb_mcts_iter,
                    }
                    .loop_until_close()
                })
                .unwrap();
        }

        // The main thread doesn't tx on any channels. Explicitly drop the txs so the zero reader
        // channel close mechanism enables all threads to terminate.
        drop(nn_queue_tx);
        drop(mcts_queue_tx);
        drop(done_queue_tx);
    })
    .unwrap();

    let ret: Vec<_> = done_queue_rx.into_iter().collect();

    let unique_pos: HashSet<_> = ret
        .iter()
        .flat_map(|result| result.samples.iter().map(|s| s.pos.clone()))
        .collect();
    println!(
        "Generated {} games with {} unique positions",
        ret.len(),
        unique_pos.len()
    );

    ret
}

/// Performs NN batch inference by reading from the [NNThread::nn_queue].
/// Performs a batch inference of [Pos]s using [NNThread::eval_pos] with up to
/// [NNThread::max_nn_batch_size] positions for the [ModelID] that has the most positions to
/// evaluate.
///
/// After the batch inference returns its evaluation, we send the evaluated positions back to the
/// MCTS threads via [NNThread::mcts_queue].
///
/// [NNThread::loop_until_close] will continue to loop until the [NNThread::nn_queue] is closed and
/// there are no more pending games to evaluate.
struct NNThread<E: EvalPosT> {
    nn_queue: Receiver<MctsGame>,
    mcts_queue: Sender<MctsJob>,
    max_nn_batch_size: usize,
    eval_pos: E,
    pb_nn_eval: ProgressBar,
    pending_games: Vec<MctsGame>,
    chan_closed: bool,
}

impl<E: EvalPosT> NNThread<E> {
    fn new(
        nn_queue: Receiver<MctsGame>,
        mcts_queue: Sender<MctsJob>,
        max_nn_batch_size: usize,
        eval_pos: E,
        pb_nn_eval: ProgressBar,
    ) -> Self {
        Self {
            nn_queue,
            mcts_queue,
            max_nn_batch_size,
            eval_pos,
            pb_nn_eval,
            pending_games: Vec::default(),
            chan_closed: false,
        }
    }

    /// Drains any items in the [NNThread::nn_queue] into the [NNThread::pending_games] vector,
    /// blocking if we have no pending games yet.
    /// Sets [NNThread::chan_closed] when the queue closes.
    fn drain_queue(&mut self) {
        if self.pending_games.is_empty() {
            match self.nn_queue.recv() {
                Ok(game) => {
                    self.pending_games.push(game);
                }
                Err(RecvError) => {
                    self.chan_closed = true;
                    return;
                }
            }
        }

        // Optimistically drain additional games from the queue.
        while let Ok(game) = self.nn_queue.try_recv() {
            self.pending_games.push(game);
        }
    }

    /// Main [NNThread] logic. Optimistically drain items from the queue, call [NNThread::eval_pos]
    /// for the [ModelID] with the most queued positions, send the evaluated positions back to the
    /// [NNThread::mcts_queue], and update [NNThread::pending_games] with all games that were not
    /// processed in this tick.
    fn loop_once(&mut self) {
        self.drain_queue();
        if self.pending_games.is_empty() {
            // pending_games can be empty if the channel closes
            return;
        }

        let mut model_pos = BTreeMap::<ModelID, HashSet<Pos>>::new();
        for game in self.pending_games.iter() {
            let model_id = game.leaf_model_id_to_play();
            let entry = model_pos.entry(model_id).or_default();
            entry.insert(game.leaf_pos());
        }

        // Select the model with the most positions and evaluate
        let model_id = model_pos
            .iter()
            .max_by_key(|(_, positions)| positions.len())
            .map(|(model_id, _)| *model_id)
            .unwrap();
        let pos = model_pos[&model_id]
            .iter()
            .take(self.max_nn_batch_size)
            .cloned()
            .collect::<Vec<_>>();
        self.pb_nn_eval.inc(pos.len() as u64);
        let evals = self.eval_pos.eval_pos(model_id, pos.clone());
        let eval_map = pos.into_iter().zip(evals).collect::<HashMap<_, _>>();

        let mut games = Vec::<MctsGame>::default();
        mem::swap(&mut self.pending_games, &mut games);
        for game in games.into_iter() {
            let pos = game.leaf_pos();
            if game.leaf_model_id_to_play() != model_id || !eval_map.contains_key(&pos) {
                self.pending_games.push(game);
                continue;
            }

            let nn_result = eval_map[&pos].clone();
            self.mcts_queue.send(MctsJob::Job(game, nn_result)).unwrap();
        }
    }

    /// Continuously loops until the [NNThread::chan_closed] flag is set and there are no more
    /// pending games to evaluate.
    fn loop_until_close(&mut self) {
        while !self.chan_closed || !self.pending_games.is_empty() {
            self.loop_once();
        }
    }
}

/// Performs MCTS iterations by reading from the [Self::mcts_queue_rx].
/// If we reach the requisite number of iterations, we probabalistically make a move with
/// [MctsGame::make_move]. Then, if the game reaches a terminal position, pass the game to
/// [Self::done_queue]. Otherwise, we pass back to the nn via [Self::nn_queue].
struct MctsThread {
    nn_queue_tx: Sender<MctsGame>,
    mcts_queue_tx: Sender<MctsJob>,
    mcts_queue_rx: Receiver<MctsJob>,
    done_queue_tx: Sender<GameResult>,
    n_games_remaining: Arc<AtomicUsize>,
    n_mcts_iterations: usize,
    n_mcts_threads: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    pb_game_done: ProgressBar,
    pb_mcts_iter: ProgressBar,
}

impl MctsThread {
    /// Main [MctsThread] logic. Returns [Loop] whether we should continue or break from the loop.
    fn loop_once(&mut self) -> Loop {
        match self.mcts_queue_rx.recv() {
            Ok(MctsJob::PoisonPill) => Loop::Break,
            Ok(MctsJob::Job(mut game, nn_result)) => {
                self.pb_mcts_iter.inc(1);
                game.on_received_policy(
                    nn_result.policy,
                    nn_result.q_penalty,
                    nn_result.q_no_penalty,
                    self.c_exploration,
                    self.c_ply_penalty,
                );

                // If we haven't reached the requisite number of MCTS iterations, send back to NN
                // to evaluate the next leaf.
                if game.root_visit_count() < self.n_mcts_iterations {
                    self.nn_queue_tx.send(game).unwrap();
                    return Loop::Continue;
                }

                // We have reached the sufficient number of MCTS iterations to make a move.
                let root_pos = game.root_pos();
                if root_pos.is_terminal_state().is_none() {
                    // Make a random move according to the MCTS policy.
                    // If we are in the early game, use a higher temperature to encourage
                    // generating more diverse (but suboptimal) games.
                    let ply = root_pos.ply();
                    let temperature = match () {
                        _ if ply < 4 => 4.0,
                        _ if ply < 8 => 2.0,
                        _ => 1.0,
                    };
                    game.make_random_move(self.c_exploration, temperature);
                    self.nn_queue_tx.send(game).unwrap();
                } else {
                    // Game is over. Send to done_queue.
                    self.n_games_remaining.fetch_sub(1, Ordering::Relaxed);
                    self.done_queue_tx
                        .send(game.to_result(self.c_ply_penalty))
                        .unwrap();
                    self.pb_game_done.inc(1);

                    if self.n_games_remaining.load(Ordering::Relaxed) == 0 {
                        // We wrote the last game. Send poison pills to remaining threads.
                        self.terminate_and_poison_other_threads();
                        return Loop::Break;
                    }
                }

                Loop::Continue
            }
            Err(RecvError) => {
                panic!("mcts_thread: mcts_queue unexpectedly closed")
            }
        }
    }

    fn terminate_and_poison_other_threads(&self) {
        self.pb_mcts_iter
            .finish_with_message("MCTS iterations complete");
        self.pb_game_done.finish_with_message("All games generated");
        for _ in 0..(self.n_mcts_threads - 1) {
            self.mcts_queue_tx.send(MctsJob::PoisonPill).unwrap();
        }
    }

    fn loop_until_close(&mut self) {
        while let Loop::Continue = self.loop_once() {}
    }
}

/// A piece of work for [mcts_thread]s. [MctsJob::PoisonPill] indicates the thread should terminate.
enum MctsJob {
    Job(MctsGame, EvalPosResult),
    PoisonPill,
}

/// Indicates whether we should continue or break from the loop.
enum Loop {
    Break,
    Continue,
}

/// Initialize progress bars for monitoring.
fn init_progress_bars(n_games: usize) -> (ProgressBar, ProgressBar, ProgressBar) {
    let multi_pb = MultiProgress::new();

    let pb_game_done = multi_pb.add(ProgressBar::new(n_games as u64));
    pb_game_done.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} games ({per_sec} games)")
        .unwrap()
        .progress_chars("#>-"));
    multi_pb.add(pb_game_done.clone());

    let pb_nn_eval = multi_pb.add(ProgressBar::new_spinner());
    pb_nn_eval.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] NN evals: {pos} ({per_sec} pos)")
            .unwrap()
            .progress_chars("#>-"),
    );
    multi_pb.add(pb_nn_eval.clone());

    let pb_mcts_iter = multi_pb.add(ProgressBar::new_spinner());
    pb_mcts_iter.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] MCTS iterations: {pos} ({per_sec} it)")
            .unwrap()
            .progress_chars("#>-"),
    );
    multi_pb.add(pb_mcts_iter.clone());

    (pb_game_done, pb_nn_eval, pb_mcts_iter)
}

#[cfg(test)]
pub mod tests {
    use more_asserts::{assert_ge, assert_le};

    use super::*;

    const MAX_NN_BATCH_SIZE: usize = 10;

    pub struct UniformEvalPos {}
    impl EvalPosT for UniformEvalPos {
        fn eval_pos(&self, _model_id: ModelID, pos: Vec<Pos>) -> Vec<EvalPosResult> {
            assert_le!(pos.len(), MAX_NN_BATCH_SIZE);
            pos.into_iter()
                .map(|_| EvalPosResult {
                    policy: MctsGame::UNIFORM_POLICY,
                    q_penalty: 0.0,
                    q_no_penalty: 0.0,
                })
                .collect()
        }
    }

    #[test]
    fn test_self_play() {
        let n_games = 1;
        let mcts_iterations = 50;
        let c_exploration = 1.0;
        let c_ply_penalty = 0.01;
        let results = self_play(
            UniformEvalPos {},
            (0..n_games)
                .map(|game_id| GameMetadata {
                    game_id,
                    player0_id: 0,
                    player1_id: 0,
                })
                .collect(),
            MAX_NN_BATCH_SIZE,
            mcts_iterations,
            c_exploration,
            c_ply_penalty,
        );

        for result in results {
            assert_ge!(result.samples.len(), 7);
            assert_eq!(
                result
                    .samples
                    .iter()
                    .filter(|sample| sample.pos == Pos::default())
                    .count(),
                1,
                "game {:?} should have a single starting position",
                result
            );

            let terminal_positions = result
                .samples
                .iter()
                .filter(|sample| sample.pos.is_terminal_state().is_some())
                .collect::<Vec<_>>();
            assert_eq!(
                terminal_positions.len(),
                1,
                "game {:?} should have a single terminal position",
                result
            );
            let terminal_value = terminal_positions[0].q_no_penalty;
            if terminal_value != -1.0 && terminal_value != 0.0 && terminal_value != 1.0 {
                assert!(
                    false,
                    "expected terminal value {} to be -1, 0, or 1",
                    terminal_value
                );
            }
        }
    }
}
