//! Depth-limited exact negamax for MCTS leaves (an MCTS-minimax hybrid after Baier and Winands).
//! It reads only the rules through [Board]'s move generation and threat masks; it never consults
//! an opening book or a solver.
//!
//! # Exactness
//!
//! Scores use [Proof::rank]: a win ending at ply e scores 100 - e, a loss ending at ply e scores
//! e - 100, a draw 0. Searching `depth` plies from ply q fixes the horizon H = q + depth. The
//! search computes, with plain alpha-beta over a full root window, the exact negamax value of the
//! game cut at H where every position still undecided at H scores 0. Call that value V'.
//!
//! - V' > 0 (a win at e = 100 - V'): the mover's maximizing strategy scores at least V' > 0, so it
//!   never reaches a cut position (those score 0) and wins by ply e <= H against every defense.
//!   The defender's minimizing strategy holds every line to at most V' before H, and any line it
//!   lets past H can only be won later than H >= e, which ranks lower. So the true value is
//!   exactly a win at ply e, the fastest win. Losses are symmetric: the slowest loss.
//! - V' = 0 with H >= 42: no position is undecided at ply 42, so nothing was cut and 0 is a draw.
//! - V' = 0 with H < 42: draws need ply 42, so 0 came from a cut and the result is unknown.
//!
//! Conversely, if the true result is decided by ply H (a win or loss ending at e <= H, or any
//! result when H >= 42), the same two strategies show V' equals it, so the search proves it.
//! Every proof therefore lies within the horizon, and every result within the horizon is proved.
//! The pruning below only skips moves whose cut-game value cannot exceed what was already found,
//! so it keeps V' exact.

use crate::{c4r::Pos, mcts::Proof, native_solver::board::Board};

/// Columns from the center out: central moves join more lines, so they tend to cut off sooner.
const ORDER: [usize; 7] = [3, 2, 4, 1, 5, 0, 6];
const FULL_PLY: u8 = 42;

/// The exact outcome of the non-terminal `pos` if it is decided within `depth` plies, else None.
/// See the module comment for why proven distances are the fastest win or slowest loss.
pub(crate) fn prove(pos: &Pos, depth: usize) -> Option<Proof> {
    prove_counting(&Board::from(pos), depth, &mut 0)
}

/// [prove] on a [Board], adding the number of searched nodes to `nodes`.
pub(crate) fn prove_counting(board: &Board, depth: usize, nodes: &mut u64) -> Option<Proof> {
    debug_assert!(
        !board.opponent_has_won() && board.ply() < FULL_PLY,
        "terminal"
    );
    let depth = depth.min((FULL_PLY - board.ply()) as usize) as u8;
    let score = negamax(board, depth, -i16::MAX, i16::MAX, nodes);
    if score > 0 {
        Some(Proof::Win {
            end_ply: (100 - score) as u8,
        })
    } else if score < 0 {
        Some(Proof::Loss {
            end_ply: (score + 100) as u8,
        })
    } else if board.ply() + depth >= FULL_PLY {
        Some(Proof::Draw)
    } else {
        None
    }
}

/// The cut game's value of the non-terminal `b` searched `depth` plies, within (alpha, beta):
/// exact inside the window, otherwise a bound on the same side as the true value.
fn negamax(b: &Board, depth: u8, mut alpha: i16, mut beta: i16, nodes: &mut u64) -> i16 {
    *nodes += 1;
    if depth == 0 {
        return 0;
    }
    let ply = b.ply() as i16;
    // Winning next ply is the best any position can score.
    if b.can_win_now() {
        return 100 - (ply + 1);
    }
    // One ply left: every move either fills the board (a draw) or reaches the horizon, and both
    // score 0 because no move wins.
    if depth == 1 {
        return 0;
    }
    let non_losing = b.non_losing();
    if non_losing == 0 {
        // Every move lets the opponent complete four next ply, within the horizon (depth >= 2).
        return (ply + 2) - 100;
    }
    // Without an immediate win, the fastest remaining win ends at ply + 3.
    beta = beta.min(100 - (ply + 3));
    if alpha >= beta {
        return beta;
    }
    let mut best = -i16::MAX;
    for col in ORDER {
        let cell = b.landing(col) & non_losing;
        if cell == 0 {
            continue;
        }
        let child = b.play_cell(cell);
        let score = if child.ply() == FULL_PLY {
            0
        } else {
            -negamax(&child, depth - 1, -beta, -alpha, nodes)
        };
        if score > best {
            best = score;
        }
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        c4r::{Pos, TerminalState},
        native_solver::{brute, live_pos, suite_rows},
        solver::MoveScores,
    };

    /// The true outcome of the non-terminal `pos` from Pons scores (22 - k for a win with the
    /// winner's k-th stone): the first player's k-th stone lands on ply 2k - 1, the second's on 2k.
    fn true_outcome(pos: &Pos, scores: &MoveScores) -> Proof {
        let best = scores.iter().flatten().copied().max().unwrap();
        let mover_offset = pos.ply() as i16 % 2;
        let end_ply = |k: i16, offset: i16| (2 * k - 1 + offset) as u8;
        match best {
            0 => Proof::Draw,
            s if s > 0 => Proof::Win {
                end_ply: end_ply(22 - s as i16, mover_offset),
            },
            s => Proof::Loss {
                end_ply: end_ply(22 + s as i16, 1 - mover_offset),
            },
        }
    }

    /// The proof [prove] must return: the true outcome when it is decided by the horizon.
    fn expected(pos: &Pos, truth: Proof, depth: usize) -> Option<Proof> {
        let horizon = pos.ply() + depth;
        let decided_by = match truth {
            Proof::Win { end_ply } | Proof::Loss { end_ply } => end_ply as usize,
            Proof::Draw => FULL_PLY as usize,
        };
        (decided_by <= horizon).then_some(truth)
    }

    fn assert_exact(pos: &Pos, truth: Proof, depths: impl Iterator<Item = usize>) {
        for depth in depths {
            assert_eq!(
                prove(pos, depth),
                expected(pos, truth, depth),
                "{:?} at depth {}",
                pos.to_moves(),
                depth
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Late positions against exhaustive minimax, at every depth up to the end of the game:
        /// proofs are exactly the outcomes decided within the horizon, with their distances.
        #[test]
        fn proofs_match_brute_force_on_late_positions(pos in live_pos(41)) {
            prop_assume!(pos.ply() >= 31);
            let truth = true_outcome(&pos, &brute::analyze(&pos));
            assert_exact(&pos, truth, 0..=(FULL_PLY as usize - pos.ply() + 1));
        }

        /// Depth 1 proves exactly the immediate wins [Pos] sees, anywhere in the game.
        #[test]
        fn depth_one_proves_exactly_the_immediate_wins(pos in live_pos(41)) {
            let wins_now = (0..Pos::N_COLS).any(|c| {
                pos.make_move(c)
                    .is_some_and(|child| child.is_terminal_state() == Some(TerminalState::OpponentWin))
            });
            let win_next = Proof::Win { end_ply: pos.ply() as u8 + 1 };
            prop_assert_eq!(prove(&pos, 1) == Some(win_next), wins_now);
        }
    }

    /// Every committed dev row has Pons scores for every ply, so this covers the middle game
    /// where the true result usually lies beyond the horizon and the search must say so.
    #[test]
    fn proofs_match_committed_pons_scores_on_dev_suite() {
        let rows = suite_rows("dev");
        assert!(rows.len() > 2000, "{} rows", rows.len());
        let mut proven = 0;
        for r in rows {
            let pos = Pos::from_moves(&r.moves);
            let truth = true_outcome(&pos, &r.scores);
            let depths = [1, 2, 3, 4, 5, 6, 7, 8];
            assert_exact(&pos, truth, depths.into_iter());
            proven += prove(&pos, 8).is_some() as usize;
        }
        // The fixture must exercise proofs as well as unknowns.
        assert!(proven > 300, "{proven} proven at depth 8");
    }

    /// Cost per call on dev rows at plies 12-29, one thread. Run with
    /// `cargo test --release minimax_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn minimax_cost() {
        let boards: Vec<Board> = suite_rows("dev")
            .iter()
            .filter(|r| (12..30).contains(&r.moves.len()))
            .map(|r| Board::from(&Pos::from_moves(&r.moves)))
            .collect();
        println!("{} positions at plies 12-29", boards.len());
        // Leaves deep in a real tree are mostly unproven, and those searches cost the most.
        println!("depth  mean_us  unproven_mean_us  max_us  mean_nodes  proven");
        for depth in [4, 6, 8, 10, 12] {
            let (mut total, mut open_total, mut max, mut nodes, mut proven) =
                (0.0, 0.0, 0.0f64, 0, 0);
            for b in &boards {
                let start = std::time::Instant::now();
                let proof = prove_counting(b, depth, &mut nodes);
                let us = start.elapsed().as_secs_f64() * 1e6;
                total += us;
                max = max.max(us);
                match proof {
                    Some(_) => proven += 1,
                    None => open_total += us,
                }
            }
            let n = boards.len() as f64;
            println!(
                "{depth:5}  {:7.1}  {:16.1}  {max:6.0}  {:10.0}  {:5.1}%",
                total / n,
                open_total / (n - proven as f64),
                nodes as f64 / n,
                100.0 * proven as f64 / n
            );
        }
    }

    #[test]
    fn empty_board_is_unknown_at_shallow_depth() {
        assert_eq!(prove(&Pos::default(), 8), None);
    }
}
