use std::array::from_fn;

/// Connect Four game logic

/// The oponnent/player token within a cell.
#[derive(Debug, PartialEq, Eq)]
pub enum CellValue {
    Opponent = 0,
    Player = 1,
}

/// Possible terminal states of a connect four game.
#[derive(Debug, PartialEq, Eq)]
pub enum TerminalState {
    PlayerWin,
    OpponentWin,
    Draw,
}

/// Connect four position.
/// Internally consists of a u64 mask (bitmask representing whether a piece exists at a given
/// location) and a u64 value (bitmask representing the color of the given piece).
/// Bit indexing is specified by [Pos::_idx_mask_unsafe].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pos {
    mask: u64,
    value: u64,
}

/// The column for a given move (0..[Pos::N_COLS])
pub type Move = usize;

impl Pos {
    pub const N_ROWS: usize = 6;
    pub const N_COLS: usize = 7;

    /// The number of channels in the numpy buffer (one per player)
    pub const BUF_N_CHANNELS: usize = 2;
    /// The length of a single channel (in # of f32s) of the numpy buffer
    pub const BUF_CHANNEL_LEN: usize = Self::N_ROWS * Self::N_COLS;
    /// The required length (in # of f32s) of the numpy buffer
    pub const BUF_LEN: usize = Self::BUF_N_CHANNELS * Self::BUF_CHANNEL_LEN;

    pub fn new() -> Pos {
        Pos { mask: 0, value: 0 }
    }

    /// Plays a move in the given column from the perspective of the [CellValue::Player].
    /// Returns a new position where the cell values are flipped.
    /// Performs bounds and collision checing.
    /// DOES NOT perform win checking.
    pub fn make_move(&self, col: Move) -> Option<Pos> {
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
    /// Performs bounds checking.
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

    /// Returns the ply of the position or the number of moves that have been played.
    /// Ply of 0 is the starting position.
    pub fn ply(&self) -> usize {
        u64::count_ones(self.mask).try_into().unwrap()
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
    const fn _idx_mask_unsafe(row: usize, col: usize) -> u64 {
        let idx = row * Self::N_COLS + col;
        0b1 << idx
    }

    /// Inverts the colors of this position.
    fn _invert(mut self) -> Pos {
        self.value = !self.value;
        self
    }

    /// Generates a horizontal win mask starting from the given cell.
    const fn _gen_h_win_mask(row: usize, col: usize) -> u64 {
        Self::_idx_mask_unsafe(row, col)
            | Self::_idx_mask_unsafe(row, col + 1)
            | Self::_idx_mask_unsafe(row, col + 2)
            | Self::_idx_mask_unsafe(row, col + 3)
    }

    /// Generates a vertical win mask starting from the given cell.
    const fn _gen_v_win_mask(row: usize, col: usize) -> u64 {
        Self::_idx_mask_unsafe(row, col)
            | Self::_idx_mask_unsafe(row + 1, col)
            | Self::_idx_mask_unsafe(row + 2, col)
            | Self::_idx_mask_unsafe(row + 3, col)
    }

    /// Generates a diagonal (top-left to bottom-right) win mask starting from the given cell.
    const fn _gen_d1_win_mask(row: usize, col: usize) -> u64 {
        Self::_idx_mask_unsafe(row, col)
            | Self::_idx_mask_unsafe(row + 1, col + 1)
            | Self::_idx_mask_unsafe(row + 2, col + 2)
            | Self::_idx_mask_unsafe(row + 3, col + 3)
    }

    /// Generates a diagonal (bottom-left to top-right) win mask starting from the given cell.
    const fn _gen_d2_win_mask(row: usize, col: usize) -> u64 {
        Self::_idx_mask_unsafe(row, col)
            | Self::_idx_mask_unsafe(row - 1, col + 1)
            | Self::_idx_mask_unsafe(row - 2, col + 2)
            | Self::_idx_mask_unsafe(row - 3, col + 3)
    }

    /// Represents the set of all possible wins.
    /// Each item is a bitmask representing the required locations of consecutive pieces.
    const WIN_MASKS: [u64; 69] = {
        // Note rust doesn't support for loops in const functions so we have to resort to while:
        // See: https://github.com/rust-lang/rust/issues/87575

        let mut masks = [0u64; 69];
        let mut index = 0;

        // Horizontal wins
        let mut row = 0;
        while row < Self::N_ROWS {
            let mut col = 0;
            while col <= Self::N_COLS - 4 {
                masks[index] = Self::_gen_h_win_mask(row, col);
                index += 1;
                col += 1;
            }
            row += 1;
        }

        // Vertical wins
        let mut col = 0;
        while col < Self::N_COLS {
            let mut row = 0;
            while row <= Self::N_ROWS - 4 {
                masks[index] = Self::_gen_v_win_mask(row, col);
                index += 1;
                row += 1;
            }
            col += 1;
        }

        // Diagonal (top-left to bottom-right) wins
        row = 0;
        while row <= Self::N_ROWS - 4 {
            let mut col = 0;
            while col <= Self::N_COLS - 4 {
                masks[index] = Self::_gen_d1_win_mask(row, col);
                index += 1;
                col += 1;
            }
            row += 1;
        }

        // Diagonal (bottom-left to top-right) wins
        row = 3;
        while row < Self::N_ROWS {
            let mut col = 0;
            while col <= Self::N_COLS - 4 {
                masks[index] = Self::_gen_d2_win_mask(row, col);
                index += 1;
                col += 1;
            }
            row += 1;
        }

        if index != 69 {
            panic!("expected 69 win masks");
        }
        masks
    };

    /// Determines if the game is over, and if so, who won.
    /// If the game is not over, returns None.
    pub fn is_terminal_state(&self) -> Option<TerminalState> {
        if self._is_terminal_for_player() {
            Some(TerminalState::PlayerWin)
        } else if self.clone()._invert()._is_terminal_for_player() {
            Some(TerminalState::OpponentWin)
        } else if self.ply() == Self::N_COLS * Self::N_ROWS {
            Some(TerminalState::Draw)
        } else {
            None
        }
    }

    /// Determines if the current player has won.
    fn _is_terminal_for_player(&self) -> bool {
        let player_tokens = self.mask & self.value;
        for win_mask in Self::WIN_MASKS {
            if u64::count_ones(player_tokens & win_mask) == 4 {
                return true;
            }
        }
        false
    }

    /// Indicates which moves (columns) are legal to play.
    pub fn legal_moves(&self) -> [bool; Self::N_COLS] {
        let top_row = Self::N_ROWS - 1;
        from_fn(|col| self.get(top_row, col).is_none())
    }

    /// Writes the position to a buffer intended to be interpreted as a [numpy] array.
    /// The final array is of shape (2, 6, 7) where the first dim represents player/opponent,
    /// the second dim represents rows, and the final dim represents columns. The data is written
    /// in row-major format.
    pub fn write_numpy_buffer(&self, buf: &mut [f32]) {
        assert_eq!(buf.len(), Self::BUF_LEN);
        (0..Self::BUF_N_CHANNELS).for_each(|player| {
            (0..Self::N_ROWS).for_each(|row| {
                (0..Self::N_COLS).for_each(|col| {
                    let idx = player * Self::BUF_CHANNEL_LEN + row * Self::N_COLS + col;
                    buf[idx] = match self.get(row, col) {
                        Some(CellValue::Player) if player == 0 => 1.0,
                        Some(CellValue::Opponent) if player == 1 => 1.0,
                        _ => 0.0,
                    };
                });
            });
        })
    }
}

impl ToString for Pos {
    fn to_string(&self) -> String {
        let mut ret: Vec<String> = Vec::with_capacity(Self::N_ROWS);
        for row in (0..Self::N_ROWS).rev() {
            let mut s = String::new();
            for col in 0..Self::N_COLS {
                let p = match self.get(row, col) {
                    Some(CellValue::Player) => "🔴",
                    Some(CellValue::Opponent) => "🔵",
                    None => "⚫",
                };
                s.push_str(p);
            }
            ret.push(s);
        }
        ret.join("\n")
    }
}

impl From<&str> for Pos {
    fn from(s: &str) -> Self {
        let mut pos = Pos::new();
        for (row, line) in s.lines().rev().enumerate() {
            for (col, c) in line.chars().enumerate() {
                let cell_value = match c {
                    '🔴' => CellValue::Player,
                    '🔵' => CellValue::Opponent,
                    _ => continue,
                };
                pos = pos._set_piece_unsafe(row, col, cell_value);
            }
        }
        pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helpers for Pos
    impl Pos {
        fn test_move(&self, col: usize) -> Pos {
            self.make_move(col).unwrap()
        }

        fn test_moves(&self, cols: &[usize]) -> Pos {
            let mut pos = self.clone();
            for col in cols {
                pos = pos.test_move(*col)
            }
            pos
        }
    }

    #[test]
    fn playing_moves_works() {
        let mut pos = Pos::new();
        for col in 0..Pos::N_COLS {
            for row in 0..Pos::N_ROWS {
                pos = pos.test_move(col);
                assert_eq!(pos.get(row, col), Some(CellValue::Opponent));
            }

            // Playing here should overflow column
            assert!(!pos.legal_moves()[col]);
            assert_eq!(pos.make_move(col), None);
        }
    }

    #[test]
    fn row_win() {
        let pos = Pos::new().test_moves(&[0, 0, 1, 1, 2, 2, 3]);

        // Because the board is inverted, the last move results in the opponent winning
        assert_eq!(pos.is_terminal_state(), Some(TerminalState::OpponentWin));
    }

    #[test]
    fn col_win() {
        let pos = Pos::new().test_moves(&[6, 0, 6, 0, 6, 0, 6]);
        assert_eq!(pos.is_terminal_state(), Some(TerminalState::OpponentWin));
    }

    #[test]
    fn draw() {
        let pos = Pos::new().test_moves(&[
            // Fill first three rows with alternating moves
            0, 1, 2, 3, 4, 5, // First row
            0, 1, 2, 3, 4, 5, // Second row
            0, 1, 2, 3, 4, 5, // Third row
            // Fill fourth and fifth rows in reverse order to continue pattern
            5, 4, 3, 2, 1, 0, // Fourth row
            5, 4, 3, 2, 1, 0, // Fifth row
            5, 4, 3, 2, 1, 0, // Sixth row
            // Fill the last column (column 6) to complete all rows
            6, 6, 6, 6, 6, 6, // Last column full
        ]);

        // Verify if the terminal state is a draw
        assert_eq!(pos.is_terminal_state(), Some(TerminalState::Draw));
    }

    #[test]
    fn to_str() {
        let pos = Pos::new().test_moves(&[
            0, 1, 2, 3, 4, 5, // First row
            0, 1, 2, 3, 4, 5, // Second row
            0, 1, 2, 3, 4, 5, // Third row
            5, 4, 3, 2, 1, 0, // Fourth row
            5, 4, 3, 2, 1, 0, // Fifth row
            5, 4, 3, 2, 1, 0, // Sixth row
            6, 6, 6, 6, 6, 6, // Last column full
        ]);

        let expected = [
            "🔵🔴🔵🔴🔵🔴🔵",
            "🔵🔴🔵🔴🔵🔴🔴",
            "🔵🔴🔵🔴🔵🔴🔵",
            "🔴🔵🔴🔵🔴🔵🔴",
            "🔴🔵🔴🔵🔴🔵🔵",
            "🔴🔵🔴🔵🔴🔵🔴",
        ]
        .join("\n");

        assert_eq!(pos.to_string(), expected);
        assert_eq!(Pos::from(expected.as_str()), pos);
    }

    #[test]
    fn legal_moves() {
        let mut pos = Pos::new();
        assert_legal_moves(&pos, "OOOOOOO");

        pos = pos.test_moves(&[
            0, 1, 2, 3, 4, 5, // First row
            0, 1, 2, 3, 4, 5, // Second row
            0, 1, 2, 3, 4, 5, // Third row
            5, 4, 3, 2, 1, 0, // Fourth row
            5, 4, 3, 2, 1, 0, // Fifth row
        ]);

        // Fill up top row
        assert_legal_moves(&pos, "OOOOOOO");
        pos = pos.test_move(5);
        assert_legal_moves(&pos, "OOOOOXO");
        pos = pos.test_move(4);
        assert_legal_moves(&pos, "OOOOXXO");
        pos = pos.test_move(3);
        assert_legal_moves(&pos, "OOOXXXO");
        pos = pos.test_move(2);
        assert_legal_moves(&pos, "OOXXXXO");
        pos = pos.test_move(1);
        assert_legal_moves(&pos, "OXXXXXO");
        pos = pos.test_move(0);
        assert_legal_moves(&pos, "XXXXXXO");

        // Fill up last column
        pos = pos.test_moves(&[6, 6, 6, 6, 6, 6]);
        assert_legal_moves(&pos, "XXXXXXX");
    }

    fn assert_legal_moves(pos: &Pos, s: &str) {
        let legal_moves = pos.legal_moves();
        for mov in 0..Pos::N_COLS {
            if legal_moves[mov] && s[mov..mov + 1] != *"O" {
                assert!(
                    false,
                    "expected col {} to be legal in game\n\n{}",
                    mov,
                    pos.to_string()
                );
            } else if !legal_moves[mov] && s[mov..mov + 1] != *"X" {
                assert!(
                    false,
                    "expected col {} to be illegal in game\n\n{}",
                    mov,
                    pos.to_string()
                );
            }
        }
    }
}
