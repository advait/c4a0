//! MCTS over caller-supplied roots, for evaluation. Single-threaded and in lockstep: no queues,
//! no workers, so no shutdown protocol and evaluator errors return to the caller.

use std::collections::HashMap;

use crate::{
    c4r::Pos,
    mcts::MctsGame,
    types::{EvalPosResult, GameMetadata, Policy},
};

/// Runs `n_iterations` MCTS iterations from every (non-terminal) root and returns each root's
/// visit policy. A root stops early once the search proves it, and its policy is then all mass
/// on the proven-best moves. Each round takes one leaf per unfinished root, dedups the unproven
/// leaves, makes ONE `eval` call on them and applies the results; proven leaves (terminal or
/// proven by search) back up their exact value without an evaluation. Deterministic for a
/// deterministic evaluator. No root noise and no temperature. `leaf_minimax_depth` > 0 searches
/// each leaf that exactly before evaluating it ([MctsGame::with_leaf_minimax_depth]).
pub fn search_positions<F, E>(
    mut eval: F,
    roots: &[Pos],
    n_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    leaf_minimax_depth: usize,
) -> Result<Vec<Policy>, E>
where
    F: FnMut(Vec<Pos>) -> Result<Vec<EvalPosResult>, E>,
{
    let mut games: Vec<MctsGame> = roots
        .iter()
        .map(|pos| {
            debug_assert!(pos.is_terminal_state().is_none(), "terminal root");
            MctsGame::new_from_pos(pos.clone(), GameMetadata::default())
                .with_leaf_minimax_depth(leaf_minimax_depth)
        })
        .collect();

    loop {
        let active: Vec<usize> = (0..games.len())
            .filter(|&i| games[i].needs_search(n_iterations))
            .collect();
        if active.is_empty() {
            break;
        }

        let mut batch_index = HashMap::<Pos, usize>::new();
        let mut batch = Vec::new();
        for &i in &active {
            let leaf = games[i].leaf_pos();
            if games[i].leaf_needs_eval() && !batch_index.contains_key(&leaf) {
                batch_index.insert(leaf.clone(), batch.len());
                batch.push(leaf);
            }
        }
        let results = if batch.is_empty() {
            Vec::new()
        } else {
            eval(batch)?
        };

        for &i in &active {
            let game = &mut games[i];
            let (policy, q_penalty, q_no_penalty) = match batch_index.get(&game.leaf_pos()) {
                Some(&j) => (
                    results[j].policy,
                    results[j].q_penalty,
                    results[j].q_no_penalty,
                ),
                // Proven leaf: on_received_policy ignores these and backs up the exact value.
                None => (MctsGame::UNIFORM_POLICY, 0.0, 0.0),
            };
            game.on_received_policy(
                policy,
                q_penalty,
                q_no_penalty,
                c_exploration,
                c_ply_penalty,
            );
        }
    }

    Ok(games.iter().map(|g| g.root_policy()).collect())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;

    fn uniform(pos: Vec<Pos>) -> Result<Vec<EvalPosResult>, Infallible> {
        Ok(pos
            .iter()
            .map(|_| EvalPosResult {
                policy: MctsGame::UNIFORM_POLICY,
                q_penalty: 0.0,
                q_no_penalty: 0.0,
            })
            .collect())
    }

    /// A deterministic, position-dependent evaluator so batching mistakes change results.
    fn hashed(pos: Vec<Pos>) -> Result<Vec<EvalPosResult>, Infallible> {
        Ok(pos
            .iter()
            .map(|p| {
                let (mask, value) = p.bits();
                let h = mask.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ value;
                let policy = std::array::from_fn(|c| ((h >> (c * 8)) & 0xff) as f32 / 64.0);
                let q = ((h >> 56) as f32 / 255.0) * 2.0 - 1.0;
                EvalPosResult {
                    policy,
                    q_penalty: q * 0.9,
                    q_no_penalty: q,
                }
            })
            .collect())
    }

    fn roots() -> Vec<Pos> {
        // Includes a duplicate and transpositions so leaves are shared across roots.
        [vec![], vec![3], vec![3, 3], vec![3], vec![2, 4], vec![4, 2]]
            .iter()
            .map(|m| Pos::from_moves(m))
            .collect()
    }

    #[test]
    fn batched_search_equals_per_root_search() {
        let roots = roots();
        let batched = search_positions(hashed, &roots, 64, 4.0, 0.01, 0).unwrap();
        for (root, got) in roots.iter().zip(&batched) {
            let single = search_positions(hashed, std::slice::from_ref(root), 64, 4.0, 0.01, 0);
            assert_eq!(&single.unwrap()[0], got);
        }
    }

    #[test]
    fn immediate_win_is_the_argmax() {
        // Player to move has three in a row on the bottom at columns 1-3; 0 and 4 win.
        let pos = Pos::from_moves(&[1, 1, 2, 2, 3, 3]);
        let policy = search_positions(uniform, &[pos], 512, 4.0, 0.01, 0).unwrap()[0];
        let best = policy.iter().cloned().fold(f32::MIN, f32::max);
        assert!(policy[0] == best || policy[4] == best, "{:?}", policy);
        assert!(policy[0] + policy[4] > 0.8, "{:?}", policy);
    }

    #[test]
    fn root_visits_match_iterations() {
        // N root visits are the root expansion plus N-1 child visits.
        let policy = search_positions(uniform, &[Pos::default()], 15, 4.0, 0.01, 0).unwrap()[0];
        let total: f32 = policy.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        let visits: Vec<f32> = policy.iter().map(|p| (p * 14.0).round()).collect();
        assert_eq!(visits.iter().sum::<f32>(), 14.0);
    }

    #[test]
    fn evaluator_errors_propagate() {
        let err = search_positions(|_| Err("boom"), &[Pos::default()], 4, 4.0, 0.01, 0);
        assert_eq!(err.unwrap_err(), "boom");
    }
}
