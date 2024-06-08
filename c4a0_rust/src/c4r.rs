/// Connect Four game logic

/// Represents the value of a cell in a connect four position.
#[derive(Debug, PartialEq, Eq)]
enum CellValue {
    Opponent = 0,
    Player = 1,
}

/// Represents the possible terminal states of a connect four game.
enum TerminalState {
    PlayerWin,
    OpponentWin,
    Draw,
}

/// Represents a connect four position.
/// Internally consists of a u64 mask (bitmask representing whether a piece exists at a given
/// location) and a u64 value (bitmask representing the color of the given piece).
/// Bit indexing is specified by `_idx_mask_unsafe`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pos {
    mask: u64,
    value: u64,
}

impl Pos {
    const N_ROWS: usize = 6;
    const N_COLS: usize = 7;

    pub fn new() -> Pos {
        Pos { mask: 0, value: 0 }
    }

    /// Plays a move in the given column from the perspective of the 1 player.
    /// Returns a new position where the cell values are flipped.
    /// Perofrms bounds and collision checing.
    pub fn make_move(&self, col: usize) -> Option<Pos> {
        if col > Self::N_COLS {
            return None;
        }

        for row in 0..Self::N_ROWS {
            let idx = Self::_idx_mask_unsafe(row, col);
            if (idx & self.mask) == 0 {
                return Some(Self::_set_piece_unsafe(&self, row, col, CellValue::Player)._invert());
            }
        }
        None
    }

    /// Returns the value of cell at the given position.
    /// Performs collision checking.
    pub fn get(&self, row: usize, col: usize) -> Option<CellValue> {
        if col > Self::N_COLS || row > Self::N_ROWS {
            return None;
        }
        let idx = Self::_idx_mask_unsafe(row, col);

        if (self.mask & idx) == 0 {
            return None;
        }

        if (self.value & idx) == 0 {
            Some(CellValue::Opponent)
        } else {
            Some(CellValue::Player)
        }
    }

    /// Sets the given piece without any bounds or collision checking.
    fn _set_piece_unsafe(&self, row: usize, col: usize, piece: CellValue) -> Pos {
        let mut p = self.clone();
        let idx_mask = Self::_idx_mask_unsafe(row, col);
        p.mask |= idx_mask;
        match piece {
            CellValue::Opponent => {
                p.value &= !idx_mask;
            }
            CellValue::Player => {
                p.value |= idx_mask;
            }
        };
        p
    }

    /// Returns a single bit for the given row and column.
    fn _idx_mask_unsafe(row: usize, col: usize) -> u64 {
        let idx = row * Self::N_COLS + col;
        0b1 << idx
    }

    /// Inverts the colors of this position.
    fn _invert(mut self) -> Pos {
        self.value = !self.value;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playing_moves_works() {
        let mut pos = Pos::new();
        for col in 0..Pos::N_COLS {
            for row in 0..Pos::N_ROWS {
                pos = pos.make_move(col).unwrap();
                assert_eq!(pos.get(row, col), Some(CellValue::Opponent));
            }

            // Playing here should overflow column
            assert_eq!(pos.make_move(col), None);
        }
    }
}
