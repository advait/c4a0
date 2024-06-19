use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    thread,
    time::Duration,
};

use crossbeam::sync::WaitGroup;
use crossbeam_queue::ArrayQueue;

use crate::{
    c4r::Pos,
    mcts::{MctsGame, Policy, PosValue, Sample},
};

/// Evaluate a batch of positions with an NN forward pass.
pub type EvalPosFn = fn(&Vec<Pos>) -> Vec<EvalPosResult>;

/// The returned output from the forward pass of the NN.
#[derive(Debug, Clone)]
pub struct EvalPosResult {
    pub policy: Policy,
    pub value: PosValue,
}

/// Generate training samples with self play and MCTS.
/// We a pytorch NN forward pass to expand a given node (to determine the initial policy values
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
    let nn_queue = Arc::new(ArrayQueue::<MctsGame>::new(n_games));
    let mcts_queue = Arc::new(ArrayQueue::<(MctsGame, EvalPosResult)>::new(n_games));
    let done_queue = Arc::new(ArrayQueue::<MctsGame>::new(n_games));

    // Create initial games
    for i in 0..n_games {
        let game = MctsGame::new_with_id(i as u64, exploration_constant);
        nn_queue.push(game).unwrap();
    }

    let wg = WaitGroup::new();

    // NN batch inference thread
    {
        let nn_queue = Arc::clone(&nn_queue);
        let mcts_queue = Arc::clone(&mcts_queue);
        let done_queue = Arc::clone(&done_queue);
        let wg = wg.clone();
        thread::Builder::new()
            .name("nn_thread".into())
            .spawn(move || {
                while nn_thread(
                    &nn_queue,
                    &mcts_queue,
                    &done_queue,
                    max_nn_batch_size,
                    n_games,
                    eval_pos,
                ) {}
                drop(wg);
            })
            .unwrap();
    }

    // MCTS thread
    for i in 0..(num_cpus::get() - 1) {
        let nn_queue = Arc::clone(&nn_queue);
        let mcts_queue = Arc::clone(&mcts_queue);
        let done_queue = Arc::clone(&done_queue);
        let wg = wg.clone();
        thread::Builder::new()
            .name(format!("mcts_thread {}", i))
            .spawn(move || {
                while mcts_thread(
                    &nn_queue,
                    &mcts_queue,
                    &done_queue,
                    n_mcts_iterations,
                    n_games,
                ) {}
                drop(wg);
            })
            .unwrap();
    }

    wg.wait();

    let mut ret = Vec::<Sample>::with_capacity(n_games * 8);
    while let Some(game) = done_queue.pop() {
        ret.append(&mut game.to_training_samples())
    }
    ret
}

/// Performs NN batch inference by reading from the `nn_queue`. Performas a batch inference
/// using `eval_pos` with up to `max_nn_batch_size` positions. Passes the resulting position
/// evaluations to the `mcts_queue`.
/// Returns whether the thread should continue looping.
fn nn_thread(
    nn_queue: &Arc<ArrayQueue<MctsGame>>,
    mcts_queue: &Arc<ArrayQueue<(MctsGame, EvalPosResult)>>,
    done_queue: &Arc<ArrayQueue<MctsGame>>,
    max_nn_batch_size: usize,
    n_games: usize,
    eval_pos: EvalPosFn,
) -> bool {
    // Read games from queue until we have max_nn_batch_size unique positions or queue is empty
    let mut games = Vec::with_capacity(max_nn_batch_size);
    let mut position_set = HashSet::<Pos>::with_capacity(max_nn_batch_size);
    loop {
        if let Some(game) = nn_queue.pop() {
            position_set.insert(game.leaf_pos().clone());
            games.push(game);
            if position_set.len() >= max_nn_batch_size {
                break;
            }
        } else if done_queue.len() == n_games {
            return false;
        } else if position_set.is_empty() {
            // Waiting for more positions to enter into the queue
            thread::sleep(Duration::from_millis(1));
            return true;
        } else {
            break;
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
    nn_queue: &Arc<ArrayQueue<MctsGame>>,
    mcts_queue: &Arc<ArrayQueue<(MctsGame, EvalPosResult)>>,
    done_queue: &Arc<ArrayQueue<MctsGame>>,
    n_mcts_iterations: usize,
    n_games: usize,
) -> bool {
    if let Some((mut game, nn_result)) = mcts_queue.pop() {
        game.on_received_policy(nn_result.policy, nn_result.value);
        if game.root_visit_count() >= n_mcts_iterations {
            // We are done with one round of MCTS. Make a random move. If the game is over
            // store in the done queue. If it's not, continue MCTS.
            game.make_random_move();
            if game.root_pos().is_terminal_state().is_some() {
                done_queue.push(game).unwrap();
            } else {
                nn_queue.push(game).unwrap();
            }
        } else {
            nn_queue.push(game).unwrap();
        }
        true
    } else if done_queue.len() == n_games {
        false
    } else {
        // If there are no MCTS games to process, yield to other threads
        thread::sleep(Duration::from_millis(1));
        true
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
