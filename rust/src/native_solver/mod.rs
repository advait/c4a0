//! A clean-room Connect Four solver, written without reference to Pascal Pons's source, to check
//! and eventually replace the external solver in [crate::solver]. Scores follow the Pons `-a`
//! convention so the two compare directly: per column, from the side to move, a win with the
//! winner's k-th stone scores 22 - k for the winner (so faster wins are larger), a draw scores 0,
//! and a full column is None.
//!
//! [board::Board] is the search position, [brute] the exhaustive cross-check over [crate::c4r::Pos].

pub mod bench;
pub mod board;
pub mod book;
pub mod brute;
pub mod diff;
pub mod parallel;
pub mod search;
pub mod tt;

#[cfg(test)]
pub use tests::{live_pos, suite_rows};

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::c4r::{Move, Pos};
    use crate::solver::MoveScores;

    prop_compose! {
        /// A non-terminal position with up to `max_ply` stones: plays each listed column if it is
        /// legal and does not complete four, and stops before filling the board.
        pub fn live_pos(max_ply: usize)(moves in prop::collection::vec(0..Pos::N_COLS, 0..120)) -> Pos {
            let mut pos = Pos::default();
            for &m in &moves {
                if pos.ply() >= max_ply.min(41) {
                    break;
                }
                if !pos.legal_moves()[m] {
                    continue;
                }
                let child = pos.make_move(m).unwrap();
                if child.is_terminal_state().is_none() {
                    pos = child;
                }
            }
            pos
        }
    }

    pub struct SuiteRow {
        pub moves: Vec<Move>,
        pub scores: MoveScores,
    }

    /// Rows of a committed suite (`dev` or `test`): 0-indexed moves and Pons scores.
    pub fn suite_rows(split: &str) -> Vec<SuiteRow> {
        let path = format!(
            "{}/../src/c4a0/suite/v1/{}.tsv",
            env!("CARGO_MANIFEST_DIR"),
            split
        );
        let text = std::fs::read_to_string(&path).unwrap();
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("moves"))
            .map(|l| {
                let (moves, scores) = l.split_once('\t').unwrap();
                let scores: Vec<Option<i8>> = scores
                    .split(',')
                    .map(|s| (s != "x").then(|| s.parse().unwrap()))
                    .collect();
                SuiteRow {
                    moves: moves
                        .chars()
                        .map(|c| c.to_digit(10).unwrap() as Move)
                        .collect(),
                    scores: scores.try_into().unwrap(),
                }
            })
            .collect()
    }
}
