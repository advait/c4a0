//! Labeling: exact scores for move sequences, the solver's entry point for the Python side.

use super::board::Board;
use super::book::Book;
use super::parallel::analyze_batch;
use super::tt::Tt;
use super::MoveScores;
use crate::c4r::Move;

/// The shared table for a labeling batch. 1 GiB: twice the cold working set measured for the
/// committed suites, and small next to this machine's memory.
pub const LABEL_TT_BYTES: usize = 1 << 30;

/// Scores every column of every move sequence (0-indexed columns from the empty board) on
/// `threads` workers sharing one table, in input order. `book` only speeds the search; the
/// scores do not depend on it. Errors name the first illegal or finished sequence.
pub fn label_moves(
    moves: &[Vec<Move>],
    book: Option<&Book>,
    threads: usize,
) -> Result<Vec<MoveScores>, String> {
    let boards = moves
        .iter()
        .map(|ms| {
            let b = Board::from_moves(ms).ok_or(format!("illegal move sequence {:?}", ms))?;
            if b.opponent_has_won() || b.ply() == 42 {
                return Err(format!("game is over after {:?}", ms));
            }
            Ok(b)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tt = Tt::with_bytes(LABEL_TT_BYTES);
    let total = boards.len() * super::board::WIDTH;
    let step = (total / 20).max(1);
    let progress = |done: usize| {
        if done % step == 0 || done == total {
            log::info!("solver: {}/{} columns", done, total);
        }
    };
    Ok(analyze_batch(&boards, &tt, book, threads, &progress))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digits(s: &str) -> Vec<Move> {
        s.chars().map(|c| c.to_digit(10).unwrap() as Move).collect()
    }

    /// Late rows of the committed dev suite, including full columns and immediate wins.
    #[test]
    fn labels_match_the_committed_suite_at_any_thread_count() {
        let rows: Vec<_> = crate::native_solver::suite_rows("dev")
            .into_iter()
            .filter(|r| r.moves.len() >= 26)
            .collect();
        assert!(rows.iter().any(|r| r.scores.contains(&None)));
        let moves: Vec<Vec<Move>> = rows.iter().map(|r| r.moves.clone()).collect();
        let want: Vec<MoveScores> = rows.iter().map(|r| r.scores).collect();
        for threads in [1, 8] {
            assert_eq!(label_moves(&moves, None, threads).unwrap(), want);
        }
    }

    #[test]
    fn finished_and_illegal_sequences_are_errors() {
        let err = label_moves(&[digits("0"), digits("0101010")], None, 2).unwrap_err();
        assert!(err.contains("game is over"), "{}", err);
        let err = label_moves(&[digits("0000000")], None, 2).unwrap_err();
        assert!(err.contains("illegal"), "{}", err);
    }
}
