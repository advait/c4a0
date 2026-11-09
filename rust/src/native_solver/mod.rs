//! The project's Connect Four solver: it labels every suite and training position
//! ([label::label_moves]). Written clean-room, without reference to Pascal Pons's source, it was
//! validated against Pons's `c4solver -a` as a black-box oracle on 228,041 positions with zero
//! disagreements before it replaced Pons (docs/native-solver.md). Scores keep that convention, so
//! stored labels are unchanged: per column, from the side to move, a win with the winner's k-th
//! stone scores 22 - k for the winner (so faster wins are larger), a draw scores 0, and a full
//! column is None.
//!
//! [board::Board] is the search position, [brute] the exhaustive cross-check over [crate::c4r::Pos].

use crate::c4r::Pos;

/// Scores for playing each column, from the mover's view: positive wins (larger is faster), zero
/// draws, negative loses. None marks a full column.
pub type MoveScores = [Option<i8>; Pos::N_COLS];

pub mod bench;
pub mod board;
pub mod book;
pub mod brute;
pub mod label;
pub mod parallel;
pub mod sample;
pub mod search;
pub mod tt;

#[cfg(test)]
pub use tests::{live_pos, suite_rows};

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::MoveScores;
    use crate::c4r::{Move, Pos};

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

    /// Rows of a committed suite (`dev` or `test`): 0-indexed moves and their stored scores.
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
