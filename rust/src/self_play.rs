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
use rand::{rngs::StdRng, seq::IteratorRandom, RngExt, SeedableRng};

use crate::{
    c4r::Pos,
    mcts::MctsGame,
    types::{EvalPosResult, EvalPosT, GameMetadata, GameResult, ModelID},
};

/// Options that affect self-play move selection and training targets.
#[derive(Clone, Debug)]
pub struct SelfPlayOptions {
    pub root_noise_alpha: Option<f32>,
    pub root_noise_frac: f32,
    pub value_target_q_weight: f32,
    pub random_start_frac: f32,
    pub random_start_max_ply: usize,
    /// See [MctsGame::with_leaf_minimax_depth]; 0 disables leaf search.
    pub leaf_minimax_depth: usize,
}

impl Default for SelfPlayOptions {
    fn default() -> Self {
        Self {
            root_noise_alpha: None,
            root_noise_frac: 0.0,
            value_target_q_weight: 0.0,
            random_start_frac: 0.0,
            random_start_max_ply: 12,
            leaf_minimax_depth: 0,
        }
    }
}

impl SelfPlayOptions {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.random_start_frac.is_finite() || !(0.0..=1.0).contains(&self.random_start_frac) {
            return Err("random_start_frac must be finite and between 0 and 1");
        }
        if self.random_start_max_ply == 0 {
            return Err("random_start_max_ply must be at least 1");
        }
        if self
            .root_noise_alpha
            .is_some_and(|a| !a.is_finite() || a <= 0.0)
        {
            return Err("root_noise_alpha must be finite and positive");
        }
        if !self.root_noise_frac.is_finite() || !(0.0..=1.0).contains(&self.root_noise_frac) {
            return Err("root_noise_frac must be between 0 and 1");
        }
        if self.root_noise_frac > 0.0 && self.root_noise_alpha.is_none() {
            return Err("root_noise_alpha is required when root_noise_frac > 0");
        }
        if !self.value_target_q_weight.is_finite()
            || !(0.0..=1.0).contains(&self.value_target_q_weight)
        {
            return Err("value_target_q_weight must be finite and between 0 and 1");
        }
        Ok(())
    }

    fn start_pos(&self, game_id: u64) -> Pos {
        if self.random_start_frac == 0.0 {
            return Pos::default();
        }
        // A separate stream leaves move sampling and root noise untouched.
        let mut rng = StdRng::seed_from_u64(game_id ^ 0xa0761d6478bd642f);
        if !rng.random_bool(self.random_start_frac as f64) {
            return Pos::default();
        }
        // A full board is terminal, so lengths above 41 cannot produce a start.
        let max_ply = self.random_start_max_ply.min(Pos::N_ROWS * Pos::N_COLS - 1);
        let length = rng.random_range(1..=max_ply);
        // Keep L fixed on retries so rejecting terminal walks does not bias start plies.
        'retry: loop {
            let mut pos = Pos::default();
            for _ in 0..length {
                let col = pos
                    .legal_moves()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(col, legal)| legal.then_some(col))
                    .choose(&mut rng)
                    .unwrap();
                pos = pos.make_move(col).unwrap();
                if pos.is_terminal_state().is_some() {
                    continue 'retry;
                }
            }
            return pos;
        }
    }

    fn add_root_noise(&self, game: &mut MctsGame, c_exploration: f32) {
        if let Some(alpha) = self.root_noise_alpha {
            if self.root_noise_frac > 0.0 {
                game.add_root_noise(alpha, self.root_noise_frac, c_exploration);
            }
        }
    }

    fn temperature(&self, ply: usize) -> f32 {
        if ply < 4 {
            4.0
        } else if ply < 8 {
            2.0
        } else {
            1.0
        }
    }
}

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
    options: SelfPlayOptions,
) -> Vec<GameResult> {
    let n_games = reqs.len();
    let (pb_game_done, pb_nn_eval, pb_mcts_iter) = init_progress_bars(n_games);
    let (nn_queue_tx, nn_queue_rx) = bounded::<MctsGame>(n_games);
    let (mcts_queue_tx, mcts_queue_rx) = bounded::<MctsJob>(n_games);
    let (done_queue_tx, done_queue_rx) = bounded::<GameResult>(n_games);
    let n_games_remaining = Arc::new(AtomicUsize::new(n_games));

    // Create initial games
    for req in reqs {
        let game = MctsGame::new_from_pos(options.start_pos(req.game_id), req)
            .with_leaf_minimax_depth(options.leaf_minimax_depth);
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
            let options = options.clone();
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
                        options,
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
    // Models are fixed during one self_play call; discard these results with the thread.
    eval_cache: HashMap<(ModelID, Pos), EvalPosResult>,
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
            eval_cache: HashMap::new(),
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
            let pos = game.leaf_pos();
            if !self.eval_cache.contains_key(&(model_id, pos.clone())) {
                model_pos.entry(model_id).or_default().insert(pos);
            }
        }

        // Select the model with the most uncached positions and evaluate only its misses.
        if let Some((&model_id, positions)) = model_pos
            .iter()
            .max_by_key(|(_, positions)| positions.len())
        {
            let pos = positions
                .iter()
                .take(self.max_nn_batch_size)
                .cloned()
                .collect::<Vec<_>>();
            self.pb_nn_eval.inc(pos.len() as u64);
            let evals = self.eval_pos.eval_pos(model_id, pos.clone());
            self.eval_cache.extend(
                pos.into_iter()
                    .zip(evals)
                    .map(|(pos, result)| ((model_id, pos), result)),
            );
        }

        let mut games = Vec::<MctsGame>::default();
        mem::swap(&mut self.pending_games, &mut games);
        for game in games.into_iter() {
            let key = (game.leaf_model_id_to_play(), game.leaf_pos());
            if let Some(nn_result) = self.eval_cache.get(&key) {
                self.mcts_queue
                    .send(MctsJob::Job(game, nn_result.clone()))
                    .unwrap();
            } else {
                self.pending_games.push(game);
            }
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
    options: SelfPlayOptions,
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

                self.options.add_root_noise(&mut game, self.c_exploration);

                // If we haven't reached the requisite number of MCTS iterations, send back to NN
                // to evaluate the next leaf.
                if game.needs_search(self.n_mcts_iterations) {
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
                    let temperature = self.options.temperature(ply);
                    game.make_random_move(self.c_exploration, self.c_ply_penalty, temperature);
                    self.options.add_root_noise(&mut game, self.c_exploration);
                    self.nn_queue_tx.send(game).unwrap();
                } else {
                    // Game is over. Send to done_queue.
                    let remaining_before = self.n_games_remaining.fetch_sub(1, Ordering::AcqRel);
                    self.done_queue_tx
                        .send(
                            game.to_result(self.c_ply_penalty, self.options.value_target_q_weight),
                        )
                        .unwrap();
                    self.pb_game_done.inc(1);

                    // Only the thread that finished the last game shuts down. A separate load()
                    // let two threads both see zero and both send poison pills, overfilling the
                    // bounded job queue and blocking forever.
                    if remaining_before == 1 {
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

    #[test]
    fn zero_random_start_fraction_matches_pre_random_start_games() {
        // Captured at 291c5ee with the empty-board constructor, before random starts. Recaptured
        // with proven-value backup: each game matched the old bytes in position and policy up to
        // the first move whose search tree held a proof, and diverged exactly there.
        for max_ply in [1, 12, 20, usize::MAX] {
            let mut games = self_play(
                UniformEvalPos {},
                [0, 1, 7, 99]
                    .into_iter()
                    .map(|game_id| GameMetadata {
                        game_id,
                        ..GameMetadata::default()
                    })
                    .collect(),
                MAX_NN_BATCH_SIZE,
                8,
                2.0,
                0.01,
                SelfPlayOptions {
                    random_start_max_ply: max_ply,
                    ..SelfPlayOptions::default()
                },
            );
            games.sort_by_key(|game| game.metadata.game_id);
            assert_eq!(
                serde_cbor::to_vec(&games).unwrap(),
                include_bytes!("self_play_empty_start.cbor"),
            );
        }
    }

    #[test]
    fn random_starts_are_non_terminal_deterministic_and_varied() {
        for max_ply in [1, 12, 20] {
            let options = SelfPlayOptions {
                random_start_frac: 1.0,
                random_start_max_ply: max_ply,
                ..SelfPlayOptions::default()
            };
            let mut positions = HashSet::new();
            let mut plies = HashSet::new();
            for game_id in 0..256 {
                let pos = options.start_pos(game_id);
                assert_eq!(pos, options.start_pos(game_id));
                assert!((1..=max_ply).contains(&pos.ply()));
                assert_eq!(pos.is_terminal_state(), None);
                positions.insert(pos.clone());
                plies.insert(pos.ply());
            }
            assert!(positions.len() > 1);
            assert_eq!(plies.len(), max_ply);
        }
    }

    #[test]
    fn random_start_samples_begin_at_start_without_prefix() {
        let options = SelfPlayOptions {
            random_start_frac: 1.0,
            ..SelfPlayOptions::default()
        };
        let games = self_play(
            UniformEvalPos {},
            (0..16)
                .map(|game_id| GameMetadata {
                    game_id,
                    ..GameMetadata::default()
                })
                .collect(),
            MAX_NN_BATCH_SIZE,
            8,
            2.0,
            0.01,
            options.clone(),
        );
        for game in games {
            let start = options.start_pos(game.metadata.game_id);
            assert_eq!(game.samples[0].pos, start);
            for (offset, sample) in game.samples.iter().enumerate() {
                assert_eq!(sample.pos.ply(), start.ply() + offset);
            }
            assert!(game
                .samples
                .last()
                .unwrap()
                .pos
                .is_terminal_state()
                .is_some());
        }
    }

    #[test]
    fn fractional_random_starts_mix_empty_and_random_positions() {
        let options = SelfPlayOptions {
            random_start_frac: 0.5,
            ..SelfPlayOptions::default()
        };
        let empty = (0..256)
            .filter(|&id| options.start_pos(id) == Pos::default())
            .count();
        assert!((80..176).contains(&empty));
    }

    #[test]
    fn random_start_length_is_uniform_even_when_walks_are_rejected() {
        for max_ply in [20, 41, usize::MAX] {
            let options = SelfPlayOptions {
                random_start_frac: 1.0,
                random_start_max_ply: max_ply,
                ..SelfPlayOptions::default()
            };
            for game_id in 0..64 {
                let mut rng = StdRng::seed_from_u64(game_id ^ 0xa0761d6478bd642f);
                assert!(rng.random_bool(1.0));
                let length = rng.random_range(1..=max_ply.min(41));
                let pos = options.start_pos(game_id);
                assert_eq!(pos.ply(), length);
                assert_eq!(pos.is_terminal_state(), None);
            }
        }
    }

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

    #[derive(Default)]
    struct CountingEvalPos {
        counts: std::cell::RefCell<HashMap<(ModelID, Pos), usize>>,
    }

    impl EvalPosT for CountingEvalPos {
        fn eval_pos(&self, model_id: ModelID, positions: Vec<Pos>) -> Vec<EvalPosResult> {
            assert!(!positions.is_empty());
            assert_le!(positions.len(), MAX_NN_BATCH_SIZE);
            positions
                .into_iter()
                .map(|pos| {
                    *self
                        .counts
                        .borrow_mut()
                        .entry((model_id, pos.clone()))
                        .or_default() += 1;
                    EvalPosResult {
                        policy: std::array::from_fn(|col| {
                            -(((col + pos.ply() + model_id as usize) % Pos::N_COLS) as f32)
                        }),
                        q_penalty: model_id as f32 / 100.0,
                        q_no_penalty: pos.ply() as f32 / 100.0,
                    }
                })
                .collect()
        }
    }

    // Drive complete games synchronously to exercise NNThread without the worker shutdown race.
    fn deterministic_game(
        metadata: GameMetadata,
        mut evaluate: impl FnMut(MctsGame) -> (MctsGame, EvalPosResult),
    ) -> GameResult {
        let mut game = MctsGame::new_from_pos(Pos::default(), metadata);
        loop {
            loop {
                let (mut evaluated_game, result) = evaluate(game);
                evaluated_game.on_received_policy(
                    result.policy,
                    result.q_penalty,
                    result.q_no_penalty,
                    1.0,
                    0.01,
                );
                game = evaluated_game;
                if game.root_visit_count() >= 8 {
                    break;
                }
            }
            if game.root_pos().is_terminal_state().is_some() {
                return game.to_result(0.01, 0.0);
            }
            let col = game
                .root_pos()
                .legal_moves()
                .iter()
                .position(|legal| *legal)
                .unwrap();
            game.make_move(col as _, 1.0, 0.01);
        }
    }

    #[test]
    fn test_nn_cache_reuses_evaluations_and_preserves_game_results() {
        let (nn_tx, nn_rx) = bounded(1);
        let (mcts_tx, mcts_rx) = bounded(1);
        let mut nn = NNThread::new(
            nn_rx,
            mcts_tx,
            MAX_NN_BATCH_SIZE,
            CountingEvalPos::default(),
            ProgressBar::hidden(),
        );
        let uncached = CountingEvalPos::default();
        let mut first_game_evals = 0;
        // The first two games repeat exactly. Swapping the models also requires separate
        // evaluations of the same positions, including the starting position.
        for (game_id, (player0_id, player1_id)) in
            [(7, 13), (7, 13), (13, 7)].into_iter().enumerate()
        {
            let metadata = GameMetadata {
                game_id: game_id as u64,
                player0_id,
                player1_id,
            };
            let expected = deterministic_game(metadata.clone(), |game| {
                let result = uncached
                    .eval_pos(game.leaf_model_id_to_play(), vec![game.leaf_pos()])
                    .pop()
                    .unwrap();
                (game, result)
            });
            let actual = deterministic_game(metadata, |game| {
                let key = (game.leaf_model_id_to_play(), game.leaf_pos());
                nn_tx.send(game).unwrap();
                nn.loop_once();
                let MctsJob::Job(game, result) = mcts_rx.try_recv().unwrap() else {
                    panic!("expected an evaluated game");
                };
                let expected = uncached.eval_pos(key.0, vec![key.1]).pop().unwrap();
                assert_eq!(result.policy, expected.policy);
                assert_eq!(result.q_penalty, expected.q_penalty);
                assert_eq!(result.q_no_penalty, expected.q_no_penalty);
                (game, result)
            });
            assert_eq!(actual.samples.len(), expected.samples.len());
            for (actual, expected) in actual.samples.iter().zip(&expected.samples) {
                assert_eq!(actual.pos, expected.pos);
                assert_eq!(actual.policy, expected.policy);
                assert_eq!(actual.q_penalty, expected.q_penalty);
                assert_eq!(actual.q_no_penalty, expected.q_no_penalty);
            }
            let counts = nn.eval_pos.counts.borrow();
            assert!(counts.values().all(|count| *count == 1));
            if game_id == 0 {
                first_game_evals = counts.len();
                assert!(first_game_evals > 1);
            } else if game_id == 1 {
                assert_eq!(counts.len(), first_game_evals);
            }
        }
        let counts = nn.eval_pos.counts.borrow();
        assert_eq!(counts.get(&(7, Pos::default())), Some(&1));
        assert_eq!(counts.get(&(13, Pos::default())), Some(&1));
        assert_eq!(nn.pb_nn_eval.position() as usize, counts.len());
        assert!(uncached.counts.borrow().values().any(|count| *count > 1));
    }

    #[test]
    fn test_nn_cache_handles_mixed_batches() {
        let (nn_tx, nn_rx) = bounded(4);
        let (mcts_tx, mcts_rx) = bounded(4);
        let mut nn = NNThread::new(
            nn_rx,
            mcts_tx,
            1,
            CountingEvalPos::default(),
            ProgressBar::hidden(),
        );
        let new_game = |pos, model_id| {
            MctsGame::new_from_pos(
                pos,
                GameMetadata {
                    player0_id: model_id,
                    player1_id: model_id,
                    ..GameMetadata::default()
                },
            )
        };
        nn_tx.send(new_game(Pos::default(), 7)).unwrap();
        nn.loop_once();
        mcts_rx.try_recv().unwrap();

        // Mix an existing hit, a duplicate miss and a distinct miss, with a batch limit of one.
        for (pos, model_id) in [
            (Pos::default(), 7),
            (Pos::default(), 13),
            (Pos::default(), 13),
            (Pos::from_moves(&[3]), 13),
        ] {
            nn_tx.send(new_game(pos, model_id)).unwrap();
        }
        nn.loop_once();
        assert!(!nn.pending_games.is_empty());
        nn.loop_once();
        assert!(nn.pending_games.is_empty());
        let jobs: Vec<_> = mcts_rx.try_iter().collect();
        assert_eq!(jobs.len(), 4);
        for job in jobs {
            let MctsJob::Job(game, result) = job else {
                panic!("expected an evaluated game");
            };
            assert_eq!(
                result.q_penalty,
                game.leaf_model_id_to_play() as f32 / 100.0
            );
            assert_eq!(result.q_no_penalty, game.leaf_pos().ply() as f32 / 100.0);
        }
        assert_eq!(nn.pb_nn_eval.position(), 3);
        assert!(nn
            .eval_pos
            .counts
            .borrow()
            .values()
            .all(|count| *count == 1));
    }

    #[test]
    fn fixed_temperature_at_every_ply() {
        for ply in 0..42 {
            let expected = if ply < 4 {
                4.0
            } else if ply < 8 {
                2.0
            } else {
                1.0
            };
            assert_eq!(SelfPlayOptions::default().temperature(ply), expected);
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
            SelfPlayOptions::default(),
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
