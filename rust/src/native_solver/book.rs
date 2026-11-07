//! [Book]: exact scores of every non-terminal position at one ply, keyed by mirror-canonical
//! [Board::canonical_key]. The search stops at that ply, which bounds the depth of early solves.

use std::collections::HashMap;

use super::board::Board;

pub struct Book {
    ply: u8,
    scores: HashMap<u64, i8>,
}

impl Book {
    pub fn ply(&self) -> u8 {
        self.ply
    }

    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// The exact score of `b` for the side to move, when `b` is at the book's ply.
    pub fn get(&self, b: &Board) -> Option<i8> {
        if b.ply() != self.ply {
            return None;
        }
        self.scores.get(&b.canonical_key()).copied()
    }
}
