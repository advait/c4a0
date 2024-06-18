use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    thread,
};

use crate::mcts::{EvalPosResult, MctsGame};
use crossbeam::sync::WaitGroup;
use crossbeam_queue::ArrayQueue;

use crate::{
    c4r::Pos,
    mcts::{EvalPosFn, Policy, PosValue},
};

/// A training sample generated via self-play.
pub struct Sample {
    game_id: u64,
    pos: Pos,
    policy: Policy,
    value: PosValue,
}

pub fn generate_samples(
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
    for _ in 0..n_games {
        let game = MctsGame::new();
        nn_queue.push(game).unwrap();
    }

    let wg = WaitGroup::new();

    // NN batch inference thread
    {
        let nn_queue = Arc::clone(&nn_queue);
        let mcts_queue = Arc::clone(&mcts_queue);
        let done_queue = Arc::clone(&done_queue);
        let wg = wg.clone();
        thread::spawn(move || {
            while nn_thread(
                &nn_queue,
                &mcts_queue,
                &done_queue,
                max_nn_batch_size,
                n_games,
                eval_pos,
            ) {}
            drop(wg);
        });
    }

    // MCTS thread
    for _ in 0..(num_cpus::get() - 1) {
        let nn_queue = Arc::clone(&nn_queue);
        let mcts_queue = Arc::clone(&mcts_queue);
        let done_queue = Arc::clone(&done_queue);
        let wg = wg.clone();
        thread::spawn(move || {
            while mcts_thread(
                &nn_queue,
                &mcts_queue,
                &done_queue,
                n_mcts_iterations,
                exploration_constant,
                n_games,
            ) {}
            drop(wg);
        });
    }

    wg.wait();

    todo!()
}

fn nn_thread(
    nn_queue: &Arc<ArrayQueue<MctsGame>>,
    mcts_queue: &Arc<ArrayQueue<(MctsGame, EvalPosResult)>>,
    done_queue: &Arc<ArrayQueue<MctsGame>>,
    max_nn_batch_size: usize,
    n_games: usize,
    eval_pos: EvalPosFn,
) -> bool {
    let mut games = Vec::with_capacity(max_nn_batch_size);
    let mut position_set = HashSet::<Pos>::new();
    loop {
        if let Some(game) = nn_queue.pop() {
            position_set.insert(game.get_leaf_pos().clone());
            games.push(game);
            if position_set.len() >= max_nn_batch_size {
                break;
            }
        } else if done_queue.len() == n_games {
            return false;
        } else if position_set.is_empty() {
            // Waiting for more positions to enter into the queue
            thread::yield_now();
            return true;
        } else {
            break;
        }
    }

    let positions: Vec<_> = position_set.into_iter().collect();
    let all_evals = (eval_pos)(&positions);
    let eval_map: HashMap<_, _> = positions.into_iter().zip(all_evals).collect();

    for game in games.into_iter() {
        let pos = game.get_leaf_pos();
        let nn_result = eval_map[pos].clone();
        mcts_queue.push((game, nn_result)).unwrap();
    }
    true
}

fn mcts_thread(
    nn_queue_producer: &Arc<ArrayQueue<MctsGame>>,
    mcts_queue_consumer: &Arc<ArrayQueue<(MctsGame, EvalPosResult)>>,
    done_queue_producer: &Arc<ArrayQueue<MctsGame>>,
    n_mcts_iterations: usize,
    exploration_constant: f64,
    n_games: usize,
) -> bool {
    match mcts_queue_consumer.pop() {
        None => {
            if done_queue_producer.len() == n_games {
                false
            } else {
                // If there are no MCTS games to process, yield to other threads
                thread::yield_now();
                true
            }
        }
        Some((mut game, nn_result)) => {
            game.on_received_policy(nn_result.policy, nn_result.value, exploration_constant);
            if game.root_visit_count() >= n_mcts_iterations {
                // We are done with one round of MCTS. Make a move and continue.
                done_queue_producer.push(game).unwrap();
            } else {
                nn_queue_producer.push(game).unwrap();
            }
            true
        }
    }
}
