use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use crossbeam::{
    channel,
    channel::{Receiver, RecvError, Sender, TryRecvError},
    sync::WaitGroup,
};
use crossbeam_queue::ArrayQueue;

use crate::{
    c4r::Pos,
    mcts::{MctsGame, Policy, PosValue, Sample},
};

/// Evaluate a batch of positions with an NN forward pass.
/// The ordering of the results corresponds to the ordering of the input positions.
pub type EvalPosFn = fn(&Vec<Pos>) -> Vec<EvalPosResult>;

/// The returned output from the forward pass of the NN.
#[derive(Debug, Clone)]
pub struct EvalPosResult {
    pub policy: Policy,
    pub value: PosValue,
}

/// Generate training samples with self play and MCTS.
/// We use a batched NN forward pass to expand a given node (to determine the initial policy values
/// based on the NN's output policy). Because we want to batch these NN calls for performance, we
/// partially compute many MCTS traversals simultaneously (via [mcts_thread]), pausing each until we
/// reach the node expansion phase. Then we are able to batch several NN calls simultaneously
/// (via [nn_thread]). This process ping-pongs until the game reaches a terminal state after which
/// it is added to `done_queue`.
pub fn self_play(
    eval_pos: EvalPosFn,
    n_games: usize,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    exploration_constant: f64,
) -> Vec<Sample> {
    let (nn_queue_tx, mut nn_queue_rx) = channel::bounded(n_games);
    let mcts_queue = Arc::new(ArrayQueue::<(MctsGame, EvalPosResult)>::new(n_games));
    let (done_queue_tx, done_queue_rx) = channel::unbounded::<Sample>();
    let n_games_remaining = Arc::new(AtomicUsize::new(n_games));

    // Create initial games
    for i in 0..n_games {
        let game = MctsGame::new_with_id(i as u64, exploration_constant);
        nn_queue_tx.send(game).unwrap();
    }

    let wg = WaitGroup::new();

    // NN batch inference thread
    {
        let mcts_queue = Arc::clone(&mcts_queue);
        let wg = wg.clone();
        thread::Builder::new()
            .name("nn_thread".into())
            .spawn(move || {
                while nn_thread(&mut nn_queue_rx, &mcts_queue, max_nn_batch_size, eval_pos) {}
                drop(wg);
            })
            .unwrap();
    }

    // MCTS threads
    let mcts_thread_count = usize::max(1, num_cpus::get() - 1);
    for i in 0..mcts_thread_count {
        let nn_queue_tx = nn_queue_tx.clone();
        let mcts_queue = Arc::clone(&mcts_queue);
        let done_queue_tx = done_queue_tx.clone();
        let n_games_remaining = Arc::clone(&n_games_remaining);
        let wg = wg.clone();
        thread::Builder::new()
            .name(format!("mcts_thread {}", i))
            .spawn(move || {
                while mcts_thread(
                    &nn_queue_tx,
                    &mcts_queue,
                    &done_queue_tx,
                    &n_games_remaining,
                    n_mcts_iterations,
                ) {}
                drop(wg);
            })
            .unwrap();
    }

    // Only `mcts_thread`s txs to the `nn_queue` and `done_queue`, so we explicitly drop here so
    // the readers appropriately receive done signals after the last `mcts_thread` finishes.
    drop(done_queue_tx);
    drop(nn_queue_tx);

    wg.wait();
    done_queue_rx.into_iter().collect()
}

/// Performs NN batch inference by reading from the `nn_queue_rx`. Performas a batch inference
/// using `eval_pos` with up to `max_nn_batch_size` positions. Passes the resulting position
/// evaluations to the `mcts_queue`.
/// Returns whether the thread should continue looping.
fn nn_thread(
    nn_queue: &mut Receiver<MctsGame>,
    mcts_queue: &Arc<ArrayQueue<(MctsGame, EvalPosResult)>>,
    max_nn_batch_size: usize,
    eval_pos: EvalPosFn,
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
    loop {
        match nn_queue.try_recv() {
            Ok(game) => {
                position_set.insert(game.leaf_pos().clone());
                games.push(game);
                if position_set.len() >= max_nn_batch_size {
                    break;
                }
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                return false;
            }
        }
    }

    let positions: Vec<_> = position_set.into_iter().collect();
    let all_evals = (eval_pos)(&positions);
    let eval_map: HashMap<_, _> = positions.into_iter().zip(all_evals).collect();

    for game in games.into_iter() {
        let pos = game.leaf_pos();
        let nn_result = eval_map[pos].clone();
        mcts_queue.push((game, nn_result)).unwrap();
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
    mcts_queue: &Arc<ArrayQueue<(MctsGame, EvalPosResult)>>,
    done_queue: &Sender<Sample>,
    n_games_remaining: &Arc<AtomicUsize>,
    n_mcts_iterations: usize,
) -> bool {
    match mcts_queue.pop() {
        Some((mut game, nn_result)) => {
            game.on_received_policy(nn_result.policy, nn_result.value);
            if game.root_visit_count() >= n_mcts_iterations {
                // We have reached the sufficient number of MCTS iterations to make a move.
                game.make_random_move();
                if game.root_pos().is_terminal_state().is_some() {
                    n_games_remaining.fetch_sub(1, Ordering::Relaxed);
                    for sample in game.to_training_samples() {
                        done_queue.send(sample).unwrap();
                    }
                } else {
                    nn_queue.send(game).unwrap();
                }
            } else {
                nn_queue.send(game).unwrap();
            }
            true
        }
        None if n_games_remaining.load(Ordering::Relaxed) == 0 => false,
        None => {
            // If there are no MCTS games to process, yield to other threads
            // TODO: replace this with a blocking mechanism
            thread::sleep(Duration::from_millis(1));
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use more_asserts::assert_ge;

    use super::*;

    fn uniform_eval_pos(pos: &Vec<Pos>) -> Vec<EvalPosResult> {
        pos.into_iter()
            .map(|_| EvalPosResult {
                policy: MctsGame::UNIFORM_POLICY,
                value: 0.0,
            })
            .collect()
    }

    #[test]
    fn test_self_play() {
        let n_games = 10;
        let mcts_iterations = 5;
        let max_nn_batch_size = 2;
        let exploration_constant = 1.0;
        let samples = self_play(
            uniform_eval_pos,
            n_games,
            max_nn_batch_size,
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
