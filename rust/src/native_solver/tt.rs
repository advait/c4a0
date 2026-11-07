//! [Tt]: a fixed-size, lock-free transposition table shared by every search thread.
//!
//! Each slot is one `AtomicU64` holding the full 49-bit [super::board::Board::key], so a probe
//! never returns another position's bound, and a single atomic store can never tear. Every
//! stored score is a true bound on the game value (searches run to the end of the game, so there
//! is no depth to qualify it), which makes lost or overwritten entries a speed cost only.

use std::sync::atomic::{AtomicU64, Ordering};

/// What a stored score says about the true value V.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// V >= score.
    Lower,
    /// V <= score.
    Upper,
    /// V == score.
    Exact,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Entry {
    pub score: i8,
    pub bound: Bound,
}

const KEY_BITS: u32 = 49;
const KEY_MASK: u64 = (1 << KEY_BITS) - 1;
const SCORE_SHIFT: u32 = KEY_BITS;
/// Scores are stored offset by 64 in 7 bits.
const SCORE_OFFSET: i32 = 64;
const BOUND_SHIFT: u32 = SCORE_SHIFT + 7;
const OCCUPIED: u64 = 1 << 63;

pub struct Tt {
    slots: Box<[AtomicU64]>,
    index_shift: u32,
}

impl Tt {
    /// 256 MiB: 32M slots. Enough that a cold solve of the hardest mid-game positions keeps most
    /// of its working set, and small next to this machine's memory. See the solver report.
    pub const DEFAULT_BYTES: usize = 256 << 20;

    /// The largest power-of-two table that fits in `bytes`, at least 1024 slots.
    pub fn with_bytes(bytes: usize) -> Tt {
        let slots = (bytes / std::mem::size_of::<AtomicU64>()).max(1024);
        let bits = usize::BITS - 1 - slots.leading_zeros();
        Tt {
            slots: (0..1usize << bits).map(|_| AtomicU64::new(0)).collect(),
            index_shift: u64::BITS - bits,
        }
    }

    pub fn bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<AtomicU64>()
    }

    fn slot(&self, key: u64) -> &AtomicU64 {
        let i = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.index_shift;
        &self.slots[i as usize]
    }

    pub fn get(&self, key: u64) -> Option<Entry> {
        let raw = self.slot(key).load(Ordering::Relaxed);
        if raw & OCCUPIED == 0 || raw & KEY_MASK != key {
            return None;
        }
        let score = (((raw >> SCORE_SHIFT) & 0x7f) as i32 - SCORE_OFFSET) as i8;
        let bound = match (raw >> BOUND_SHIFT) & 0b11 {
            0 => Bound::Lower,
            1 => Bound::Upper,
            _ => Bound::Exact,
        };
        Some(Entry { score, bound })
    }

    /// Stores a bound for `key`, replacing whatever the slot held.
    pub fn put(&self, key: u64, score: i8, bound: Bound) {
        debug_assert!(key <= KEY_MASK);
        let bound_bits = match bound {
            Bound::Lower => 0,
            Bound::Upper => 1,
            Bound::Exact => 2,
        };
        let raw = OCCUPIED
            | bound_bits << BOUND_SHIFT
            | ((score as i32 + SCORE_OFFSET) as u64) << SCORE_SHIFT
            | key;
        self.slot(key).store(raw, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_score_and_bound() {
        let tt = Tt::with_bytes(1 << 16);
        for score in -21..=21 {
            for bound in [Bound::Lower, Bound::Upper, Bound::Exact] {
                // The empty board's key is 0, which must not read as an empty slot.
                for key in [0, 1, KEY_MASK, 0x1234_5678_9abc] {
                    tt.put(key, score, bound);
                    assert_eq!(tt.get(key), Some(Entry { score, bound }), "{}", key);
                }
            }
        }
    }

    #[test]
    fn never_returns_another_keys_entry() {
        let tt = Tt::with_bytes(8 * 1024);
        for key in 0..100_000u64 {
            tt.put(key, (key % 43) as i8 - 21, Bound::Exact);
        }
        for key in 0..100_000u64 {
            if let Some(e) = tt.get(key) {
                assert_eq!(e.score, (key % 43) as i8 - 21);
            }
        }
        assert_eq!(tt.get(200_000), None);
    }

    #[test]
    fn size_is_bounded_by_the_request() {
        assert_eq!(Tt::with_bytes(Tt::DEFAULT_BYTES).bytes(), Tt::DEFAULT_BYTES);
        assert_eq!(Tt::with_bytes(3 << 20).bytes(), 2 << 20);
        assert_eq!(Tt::with_bytes(0).bytes(), 8 * 1024);
    }
}
