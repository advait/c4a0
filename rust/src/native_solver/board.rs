//! [Board]: the search-side position. Column-major bitboards with a sentinel row, so every line
//! direction is a single shift and no shift carries between columns.

use crate::c4r::{CellValue, Move, Pos};

pub const WIDTH: usize = 7;
pub const HEIGHT: usize = 6;
/// Bits per column: [HEIGHT] cells plus one always-empty sentinel bit on top.
const COL_BITS: usize = HEIGHT + 1;
/// Cells on the board.
pub const CELLS: u8 = (WIDTH * HEIGHT) as u8;

const fn bottom_mask() -> u64 {
    let mut m = 0;
    let mut c = 0;
    while c < WIDTH {
        m |= 1 << (c * COL_BITS);
        c += 1;
    }
    m
}

/// The bottom cell of every column.
pub const BOTTOM: u64 = bottom_mask();
/// Every playable cell (sentinels excluded).
pub const BOARD: u64 = BOTTOM * ((1 << HEIGHT) - 1);

/// Bit of the cell at `row` (0 at the bottom) in `col`.
pub const fn cell(row: usize, col: usize) -> u64 {
    1 << (col * COL_BITS + row)
}

/// Every cell of `col`.
pub const fn column(col: usize) -> u64 {
    ((1 << HEIGHT) - 1) << (col * COL_BITS)
}

const fn top_cell(col: usize) -> u64 {
    cell(HEIGHT - 1, col)
}

/// A non-terminal or terminal Connect Four position from the side to move's view. `mover` holds
/// the side to move's stones, `occupied` every stone. `ply` is the stone count, so the side to
/// move is implied.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Board {
    mover: u64,
    occupied: u64,
    ply: u8,
}

impl Board {
    pub fn ply(&self) -> u8 {
        self.ply
    }

    pub fn occupied(&self) -> u64 {
        self.occupied
    }

    /// The side to move's stones.
    pub fn mover(&self) -> u64 {
        self.mover
    }

    /// The side that just moved.
    pub fn opponent(&self) -> u64 {
        self.mover ^ self.occupied
    }

    pub fn can_play(&self, col: Move) -> bool {
        self.occupied & top_cell(col) == 0
    }

    /// The cell a stone in `col` lands on, or 0 for a full column.
    pub fn landing(&self, col: Move) -> u64 {
        (self.occupied + BOTTOM) & column(col)
    }

    /// Every cell a stone can land on next.
    pub fn playable(&self) -> u64 {
        (self.occupied + BOTTOM) & BOARD
    }

    /// Plays the stone landing on `cell`, which must come from [Self::landing] or
    /// [Self::playable]. The new mover is the old opponent.
    pub fn play_cell(&self, cell: u64) -> Board {
        Board {
            mover: self.mover ^ self.occupied,
            occupied: self.occupied | cell,
            ply: self.ply + 1,
        }
    }

    /// Plays `col`, which must not be full.
    pub fn play(&self, col: Move) -> Board {
        debug_assert!(self.can_play(col));
        self.play_cell(self.landing(col))
    }

    /// Cells where the side to move would complete four, empty or not.
    pub fn mover_threats(&self) -> u64 {
        threats(self.mover)
    }

    /// Cells where the side that just moved would complete four, empty or not.
    pub fn opponent_threats(&self) -> u64 {
        threats(self.opponent())
    }

    /// Whether the side to move wins by playing `col`.
    pub fn wins_with(&self, col: Move) -> bool {
        self.mover_threats() & self.landing(col) != 0
    }

    /// Whether the side to move can complete four this move.
    pub fn can_win_now(&self) -> bool {
        self.mover_threats() & self.playable() != 0
    }

    /// Whether the side that just moved has four in a row.
    pub fn opponent_has_won(&self) -> bool {
        has_four(self.opponent())
    }

    /// Playable cells after which the opponent cannot win at once. Excludes a cell directly under
    /// an opponent threat, and must block an opponent threat that is already playable. Empty when
    /// every move loses at once. Assumes the side to move cannot win now.
    pub fn non_losing(&self) -> u64 {
        let playable = self.playable();
        let theirs = self.opponent_threats() & !self.occupied;
        let forced = playable & theirs;
        let candidates = if forced == 0 {
            playable
        } else if forced & (forced - 1) != 0 {
            return 0;
        } else {
            forced
        };
        candidates & !(theirs >> 1)
    }

    /// A unique key under 2^49: per column, `occupied + mover` lies in a range that only that
    /// column height produces, so it identifies both the height and the colors.
    pub fn key(&self) -> u64 {
        self.occupied + self.mover
    }

    /// The key of the left-right mirror.
    pub fn mirror_key(&self) -> u64 {
        mirror(self.key())
    }

    /// The smaller of the key and its mirror. Mirrors share a solved score.
    pub fn canonical_key(&self) -> u64 {
        self.key().min(self.mirror_key())
    }

    /// The board reached by playing `moves` (0-indexed columns) from empty. None if a move plays a
    /// full or out-of-range column or follows a win.
    pub fn from_moves(moves: &[Move]) -> Option<Board> {
        let mut b = Board::default();
        for &m in moves {
            if m >= WIDTH || !b.can_play(m) || b.opponent_has_won() {
                return None;
            }
            b = b.play(m);
        }
        Some(b)
    }
}

/// Mirrors a column-major bitboard or key left to right.
pub fn mirror(bits: u64) -> u64 {
    (0..WIDTH).fold(0, |acc, c| {
        let col = (bits >> (c * COL_BITS)) & ((1 << COL_BITS) - 1);
        acc | col << ((WIDTH - 1 - c) * COL_BITS)
    })
}

/// Shifts toward higher bits by `d`, or lower bits for negative `d`.
fn shift(bits: u64, d: i32) -> u64 {
    if d >= 0 {
        bits << d
    } else {
        bits >> -d
    }
}

/// Bit offsets of one step along each line: up, right, up-right, down-right.
const DIRECTIONS: [i32; 4] = [1, COL_BITS as i32, COL_BITS as i32 + 1, COL_BITS as i32 - 1];

/// Cells that would complete four with `stones`. A line that would wrap between columns passes
/// through a sentinel, which never holds a stone, so wraps cannot produce false threats.
pub fn threats(stones: u64) -> u64 {
    let mut r = 0;
    for d in DIRECTIONS {
        // For a target cell t, the stones at offsets {-3,-2,-1}, {-2,-1,+1}, {-1,+1,+2} and
        // {+1,+2,+3} along d.
        let back1 = shift(stones, d);
        let back2 = shift(stones, 2 * d);
        let back3 = shift(stones, 3 * d);
        let fwd1 = shift(stones, -d);
        let fwd2 = shift(stones, -2 * d);
        let fwd3 = shift(stones, -3 * d);
        r |= back1 & back2 & back3;
        r |= back1 & back2 & fwd1;
        r |= back1 & fwd1 & fwd2;
        r |= fwd1 & fwd2 & fwd3;
    }
    r & BOARD
}

/// Whether `stones` holds four in a row.
pub fn has_four(stones: u64) -> bool {
    DIRECTIONS.iter().any(|&d| {
        let pairs = stones & (stones >> d);
        pairs & (pairs >> (2 * d)) != 0
    })
}

impl From<&Pos> for Board {
    /// Converts cell by cell, so it relies only on [Pos::get], not on its bit layout.
    fn from(pos: &Pos) -> Board {
        let mut b = Board::default();
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                match pos.get(row, col) {
                    Some(CellValue::Player) => {
                        b.mover |= cell(row, col);
                        b.occupied |= cell(row, col);
                    }
                    Some(CellValue::Opponent) => b.occupied |= cell(row, col),
                    None => {}
                }
            }
        }
        b.ply = b.occupied.count_ones() as u8;
        b
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::c4r::{tests::random_pos, TerminalState};
    use crate::native_solver::live_pos;

    proptest! {
        /// Every move agrees with Pos: same legal columns, same landing cells, and a win exactly
        /// when Pos sees the mover's four.
        #[test]
        fn moves_and_wins_match_pos(pos in random_pos()) {
            let b = Board::from(&pos);
            prop_assert_eq!(b.ply() as usize, pos.ply());
            let legal = pos.legal_moves();
            for col in 0..WIDTH {
                prop_assert_eq!(b.can_play(col), legal[col]);
                if !legal[col] || pos.is_terminal_state().is_some() {
                    continue;
                }
                let child = pos.make_move(col).unwrap();
                prop_assert_eq!(Board::from(&child), b.play(col));
                let pos_win = child.is_terminal_state() == Some(TerminalState::OpponentWin);
                prop_assert_eq!(b.wins_with(col), pos_win);
                prop_assert_eq!(b.play(col).opponent_has_won(), pos_win);
            }
        }

        /// A threat is a cell whose line of four has its other three cells held by the stones.
        #[test]
        fn threats_match_definition(pos in random_pos()) {
            let b = Board::from(&pos);
            for stones in [b.mover(), b.opponent()] {
                let want = lines()
                    .iter()
                    .flat_map(|line| line.iter().map(move |&c| (line, c)))
                    .filter(|(line, c)| line.iter().all(|&o| o == *c || stones & o != 0))
                    .fold(0, |acc, (_, c)| acc | c);
                prop_assert_eq!(threats(stones), want);
                prop_assert_eq!(has_four(stones), lines().iter().any(|l| l.iter().all(|&c| stones & c != 0)));
            }
        }

        #[test]
        fn mirror_key_matches_mirrored_pos(pos in random_pos()) {
            let b = Board::from(&pos);
            prop_assert_eq!(Board::from(&pos.flip_h()).key(), b.mirror_key());
            prop_assert_eq!(mirror(mirror(b.key())), b.key());
        }

        #[test]
        fn non_losing_moves_avoid_immediate_losses(pos in live_pos(41)) {
            let b = Board::from(&pos);
            prop_assume!(!b.can_win_now());
            let safe = b.non_losing();
            for col in (0..WIDTH).filter(|&c| b.can_play(c)) {
                let child = b.play(col);
                let loses = (0..WIDTH).any(|c| child.can_play(c) && child.wins_with(c));
                prop_assert_eq!(safe & b.landing(col) != 0, !loses, "col {}", col);
            }
        }
    }

    /// Every line of four cells on the board, by scanning each start and direction.
    fn lines() -> Vec<[u64; 4]> {
        let mut ret = vec![];
        for r in 0..HEIGHT as i32 {
            for c in 0..WIDTH as i32 {
                for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
                    let (er, ec) = (r + 3 * dr, c + 3 * dc);
                    if (0..HEIGHT as i32).contains(&er) && (0..WIDTH as i32).contains(&ec) {
                        ret.push(std::array::from_fn(|i| {
                            let i = i as i32;
                            cell((r + i * dr) as usize, (c + i * dc) as usize)
                        }));
                    }
                }
            }
        }
        assert_eq!(ret.len(), 69);
        ret
    }

    #[test]
    fn keys_are_unique_over_small_plies() {
        let mut seen = std::collections::HashMap::new();
        let mut frontier = vec![Board::default()];
        for _ in 0..6 {
            let mut next = vec![];
            for b in frontier {
                for col in (0..WIDTH).filter(|&c| b.can_play(c)) {
                    let child = b.play(col);
                    if let Some(prev) = seen.insert(child.key(), child) {
                        assert_eq!(prev, child, "key collision");
                    }
                    next.push(child);
                }
            }
            frontier = next;
        }
        assert!(seen.len() > 10_000);
    }

    #[test]
    fn from_moves_rejects_illegal_sequences() {
        assert!(Board::from_moves(&[0; 6]).is_some());
        assert!(Board::from_moves(&[0; 7]).is_none());
        assert!(Board::from_moves(&[7]).is_none());
        // 0101010 wins for the first player; any further move is after the game ended.
        assert!(Board::from_moves(&[0, 1, 0, 1, 0, 1, 0]).is_some());
        assert!(Board::from_moves(&[0, 1, 0, 1, 0, 1, 0, 1]).is_none());
    }
}
