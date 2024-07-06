use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use crossbeam::thread;
use crossbeam_channel::{bounded, unbounded, Receiver, RecvError, Sender};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{
    c4r::Pos,
    mcts::{MctsGame, Policy, PosValue, Sample},
};

/// Evaluate a batch of positions with an NN forward pass.
/// The ordering of the results corresponds to the ordering of the input positions.
pub trait EvalPosT {
    fn eval_pos(&self, pos: &Vec<Pos>) -> Vec<EvalPosResult>;
}

/// The returned output from the forward pass of the NN.
#[derive(Debug, Clone)]
pub struct EvalPosResult {
    pub policy: Policy,  // Probability distribution over moves from the position.
    pub value: PosValue, // Lucrativeness [-1, 1] of the position.
}

/// Generate training samples with self play and MCTS.
///
/// We use a batched NN forward pass to expand a given node (to determine the initial policy values
/// based on the NN's output policy). Because we want to batch these NN calls for performance, we
/// partially compute many MCTS traversals simultaneously (via [mcts_thread]), pausing each until we
/// reach the node expansion phase. Then we are able to batch several NN calls simultaneously
/// (via [nn_thread]). This process ping-pongs until the game reaches a terminal state after which
/// it is added to `done_queue`.
///
/// We use one [nn_thread] and (n-1) [mcts_thread]s (where n=core count).
/// The thread termination mechanism is as follows:
/// 1. [mcts_thread]s whether we have finished all games via `n_games_remaining` atomic. When the
///    first thread detects all work is complete, it sends a [MctsJob::PoisonPill] to all remaining
///    [mcts_thread]s, resulting in all of these threads completing.
/// 2. When the last [mcts_thread] completes, it drops the last `nn_queue_tx`
///    [crossbeam_channel::Sender], causing the `nn_queue_rx` [crossbeam_channel::Receiver] to
///    close. This notifies the [nn_thread], allowing it to close.
/// 3. The main thread uses a [WaitGroup] to block on all of the above threads. When the wg
///    completes, we are able to drain all [Sample]s from the `done_queue` and return.
pub fn self_play<E: EvalPosT + Send + Sync>(
    eval_pos: E,
    n_games: usize,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    exploration_constant: f32,
) -> Vec<Sample> {
    let multi_pb = MultiProgress::new();
    let pb_games = multi_pb.add(ProgressBar::new(n_games as u64));
    pb_games.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} games ({per_sec} games/sec)")
        .unwrap()
        .progress_chars("#>-"));
    multi_pb.add(pb_games.clone());
    let pb_nn = multi_pb.add(ProgressBar::new_spinner());
    pb_nn.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] NN evals: {pos} ({per_sec} pos/sec)")
            .unwrap()
            .progress_chars("#>-"),
    );
    multi_pb.add(pb_nn.clone());

    let eval_pos = Arc::new(eval_pos);
    let (nn_queue_tx, nn_queue_rx) = bounded::<MctsGame>(n_games);
    let (mcts_queue_tx, mcts_queue_rx) = bounded::<MctsJob>(n_games);
    let (done_queue_tx, done_queue_rx) = unbounded::<Sample>();
    let n_games_remaining = Arc::new(AtomicUsize::new(n_games));

    // Create initial games
    for i in 0..n_games {
        let game = MctsGame::new_with_id(i as u64, exploration_constant);
        nn_queue_tx.send(game).unwrap();
    }

    thread::scope(|s| {
        // NN batch inference thread
        let eval_pos = Arc::clone(&eval_pos);
        let mcts_queue = mcts_queue_tx.clone();
        let pb_nn = pb_nn.clone();
        s.builder()
            .name("nn_thread".into())
            .spawn(move |_| {
                while nn_thread(
                    &nn_queue_rx,
                    &mcts_queue,
                    max_nn_batch_size,
                    &eval_pos,
                    pb_nn.clone(),
                ) {}
                pb_nn.finish_with_message("NN batch inference complete");
            })
            .unwrap();

        // MCTS threads
        let mcts_thread_count = usize::max(1, num_cpus::get() - 1);
        for i in 0..mcts_thread_count {
            let nn_queue_tx = nn_queue_tx.clone();
            let mcts_queue_tx = mcts_queue_tx.clone();
            let mcts_queue_rx = mcts_queue_rx.clone();
            let done_queue_tx = done_queue_tx.clone();
            let done_pb = pb_games.clone();
            let n_games_remaining = Arc::clone(&n_games_remaining);
            s.builder()
                .name(format!("mcts_thread {}", i))
                .spawn(move |_| {
                    while mcts_thread(
                        &nn_queue_tx,
                        &mcts_queue_tx,
                        &mcts_queue_rx,
                        &done_queue_tx,
                        done_pb.clone(),
                        &n_games_remaining,
                        n_mcts_iterations,
                        mcts_thread_count,
                    ) {}
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

    done_queue_rx.into_iter().collect()
}

/// Performs NN batch inference by reading from the `nn_queue_rx`. Performas a batch inference
/// using `eval_pos` with up to `max_nn_batch_size` positions. Passes the resulting position
/// evaluations to the `mcts_queue`.
/// Returns whether the thread should continue looping.
fn nn_thread<E: EvalPosT>(
    nn_queue: &Receiver<MctsGame>,
    mcts_queue: &Sender<MctsJob>,
    max_nn_batch_size: usize,
    eval_pos: &Arc<E>,
    pb_nn: ProgressBar,
) -> bool {
    // Read games from queue until we have max_nn_batch_size unique positions or queue is empty
    let mut games = Vec::with_capacity(max_nn_batch_size);
    let mut position_set = HashSet::<Pos>::with_capacity(max_nn_batch_size);

    // Block on reading the first game
    match nn_queue.recv() {
        Ok(game) => {
            position_set.insert(game.leaf_pos().clone());
            games.push(game);
        }
        Err(RecvError) => {
            return false;
        }
    }

    // Optimistically pull additional games from the queue until we reach `max_nn_batch_size` unique
    // positions or the queue is empty.
    while let Ok(game) = nn_queue.try_recv() {
        position_set.insert(game.leaf_pos().clone());
        games.push(game);
        if position_set.len() >= max_nn_batch_size {
            break;
        }
    }

    let positions: Vec<_> = position_set.into_iter().collect();
    let all_evals = eval_pos.eval_pos(&positions);
    pb_nn.inc(positions.len() as u64);
    let eval_map: HashMap<_, _> = positions.into_iter().zip(all_evals).collect();

    for game in games.into_iter() {
        let pos = game.leaf_pos();
        let nn_result = eval_map[pos].clone();
        mcts_queue.send(MctsJob::Job(game, nn_result)).unwrap();
    }
    true
}

/// Performs MCTS iterations by reading from the `mcts_queue`.
/// If we reach the requisite number of iterations, we probabalistically make a move with
/// [MctsGame::make_move]. Then, if the game reaches a terminal position, pass the game to
/// `done_queue`. Otherwise, we pass back to the nn via `nn_queue`.
/// Returns whether the thread should continue looping.
fn mcts_thread(
    nn_queue: &Sender<MctsGame>,
    mcts_queue_tx: &Sender<MctsJob>,
    mcts_queue_rx: &Receiver<MctsJob>,
    done_queue: &Sender<Sample>,
    done_pb: ProgressBar,
    n_games_remaining: &Arc<AtomicUsize>,
    n_mcts_iterations: usize,
    n_mcts_threads: usize,
) -> bool {
    match mcts_queue_rx.recv() {
        Ok(MctsJob::PoisonPill) => false,
        Ok(MctsJob::Job(mut game, nn_result)) => {
            game.on_received_policy(nn_result.policy, nn_result.value);

            if game.root_visit_count() >= n_mcts_iterations {
                // We have reached the sufficient number of MCTS iterations to make a move.
                game.make_random_move();
                if game.root_pos().is_terminal_state().is_some() {
                    n_games_remaining.fetch_sub(1, Ordering::Relaxed);
                    for sample in game.to_training_samples() {
                        done_queue.send(sample).unwrap();
                        done_pb.inc(1);
                    }
                } else {
                    nn_queue.send(game).unwrap();
                }
            } else {
                nn_queue.send(game).unwrap();
            }

            if n_games_remaining.load(Ordering::Relaxed) == 0 {
                // We wrote the last game. Send poison pills to remaining threads.
                done_pb.finish_with_message("All games generated");
                for _ in 0..(n_mcts_threads - 1) {
                    mcts_queue_tx.send(MctsJob::PoisonPill).unwrap();
                }
                false
            } else {
                true
            }
        }
        Err(RecvError) => {
            panic!("mcts_thread: mcts_queue unexpectedly closed")
        }
    }
}

/// A piece of work for [mcts_thread]s. [MctsJob::PoisonPill] indicates the thread should terminate.
enum MctsJob {
    Job(MctsGame, EvalPosResult),
    PoisonPill,
}

#[cfg(test)]
mod tests {
    use more_asserts::{assert_ge, assert_le};

    use super::*;

    const MAX_NN_BATCH_SIZE: usize = 10;

    struct UniformEvalPos {}
    impl EvalPosT for UniformEvalPos {
        fn eval_pos(&self, pos: &Vec<Pos>) -> Vec<EvalPosResult> {
            assert_le!(pos.len(), MAX_NN_BATCH_SIZE);
            pos.into_iter()
                .map(|_| EvalPosResult {
                    policy: MctsGame::UNIFORM_POLICY,
                    value: 0.0,
                })
                .collect()
        }
    }

    #[test]
    fn test_self_play() {
        let n_games = 100;
        let mcts_iterations = 50;
        let exploration_constant = 1.0;
        let samples = self_play(
            UniformEvalPos {},
            n_games,
            MAX_NN_BATCH_SIZE,
            mcts_iterations,
            exploration_constant,
        );

        for g in 0..n_games {
            let game_samples = samples
                .iter()
                .filter(|Sample { game_id, .. }| *game_id == (g as u64))
                .collect::<Vec<_>>();

            assert_ge!(game_samples.len(), 7);
            assert_eq!(
                game_samples
                    .iter()
                    .filter(|Sample { pos, .. }| *pos == Pos::new())
                    .count(),
                1,
                "game {} should have a single starting position",
                g
            );

            let terminal_positions = game_samples
                .iter()
                .filter(|Sample { pos, .. }| pos.is_terminal_state().is_some())
                .collect::<Vec<_>>();
            assert_eq!(
                terminal_positions.len(),
                1,
                "game {} should have a single terminal position",
                g
            );
            let terminal_value = terminal_positions[0].value;
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
