//! Exhaustive minimax over [Pos], the cross-check for late positions. No pruning, no bounds and
//! no [super::board] code: it only memoizes exact values of positions it has fully expanded, and
//! scores a win from the winner's stone count, so it shares no logic with the search.

use std::collections::HashMap;

use crate::c4r::{CellValue, Pos, TerminalState};
use crate::solver::MoveScores;

/// Exact scores of every column of the non-terminal `pos`, in the Pons convention: a win with
/// the winner's k-th stone scores 22 - k for the winner, a draw 0, and a full column None.
pub fn analyze(pos: &Pos) -> MoveScores {
    assert!(pos.is_terminal_state().is_none(), "terminal position");
    let mut memo = HashMap::new();
    let legal = pos.legal_moves();
    std::array::from_fn(|col| legal[col].then(|| move_score(pos, col, &mut memo)))
}

/// Exact score of playing the legal `col` in the non-terminal `pos`.
pub fn score_move(pos: &Pos, col: usize) -> i8 {
    move_score(pos, col, &mut HashMap::new())
}

fn value(pos: &Pos, memo: &mut HashMap<Pos, i8>) -> i8 {
    if let Some(&v) = memo.get(pos) {
        return v;
    }
    let legal = pos.legal_moves();
    let v = (0..Pos::N_COLS)
        .filter(|&c| legal[c])
        .map(|c| move_score(pos, c, memo))
        .max()
        .expect("a non-terminal position has a legal move");
    memo.insert(pos.clone(), v);
    v
}

fn move_score(pos: &Pos, col: usize, memo: &mut HashMap<Pos, i8>) -> i8 {
    let child = pos.make_move(col).expect("legal move");
    match child.is_terminal_state() {
        // The child's opponent is the side that just played `col`.
        Some(TerminalState::OpponentWin) => 22 - stones(&child, CellValue::Opponent),
        Some(TerminalState::Draw) => 0,
        Some(TerminalState::PlayerWin) => unreachable!("the side to move cannot have won"),
        None => -value(&child, memo),
    }
}

fn stones(pos: &Pos, side: CellValue) -> i8 {
    let mut n = 0;
    for row in 0..Pos::N_ROWS {
        for col in 0..Pos::N_COLS {
            n += (pos.get(row, col) == Some(side)) as i8;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_solver::suite_rows;

    #[test]
    fn immediate_win_scores_by_winner_stone_count() {
        // First player completes a vertical four with its 4th stone.
        let pos = Pos::from_moves(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(score_move(&pos, 0), 18);
        // Second player completes a horizontal four with its 4th stone.
        let pos = Pos::from_moves(&[0, 1, 0, 2, 6, 3, 6]);
        assert_eq!(score_move(&pos, 4), 18);
        // First player wins with its 5th stone.
        let pos = Pos::from_moves(&[0, 1, 1, 6, 0, 1, 0, 6]);
        assert_eq!(score_move(&pos, 0), 17);
    }

    #[test]
    fn full_columns_are_none() {
        // A late position where only the last column is open: 41 stones, no four.
        let pos = Pos::from_moves(&[
            0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 4, 5, 4, 5, 4,
            5, 5, 4, 5, 4, 5, 4, 6, 6, 6, 6, 6,
        ]);
        assert!(pos.is_terminal_state().is_none(), "{}", pos);
        assert_eq!(analyze(&pos), [None, None, None, None, None, None, Some(0)]);
    }

    /// The committed suites hold Pons scores; brute force must reproduce every late row.
    #[test]
    fn matches_committed_pons_scores_on_late_suite_rows() {
        let rows: Vec<_> = suite_rows("dev")
            .into_iter()
            .filter(|r| r.moves.len() >= 37)
            .collect();
        assert!(rows.len() >= 300, "{} rows", rows.len());
        for r in rows {
            assert_eq!(
                analyze(&Pos::from_moves(&r.moves)),
                r.scores,
                "{:?}",
                r.moves
            );
        }
    }
}
