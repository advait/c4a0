//! Reproducible benchmark positions: uniform random walks to every ply 0-41, the same mix the
//! labeling sampler draws. A seed means the same positions on every platform and toolchain.

use std::collections::HashSet;

use super::board::{Board, WIDTH};
use crate::c4r::Move;

/// Every canonical non-terminal position at plies 0..=EXHAUSTIVE_PLIES is excluded from the
/// walks, as it was when the walks were one stratum of the Pons comparison (docs/native-solver.md),
/// so a seed still names the benchmark workload recorded there.
const EXHAUSTIVE_PLIES: usize = 4;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Rng {
        Rng(seed ^ stream.wrapping_mul(0xA24B_AED4_963E_E407))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
}

/// Columns playable in `b` that do not complete four.
fn quiet_moves(b: &Board) -> Vec<Move> {
    (0..WIDTH)
        .filter(|&c| b.can_play(c) && !b.wins_with(c))
        .collect()
}

/// A uniform walk to `ply` among quiet moves. None if the walk dead-ends.
fn walk(ply: usize, rng: &mut Rng) -> Option<Vec<Move>> {
    let mut b = Board::default();
    let mut moves = vec![];
    while moves.len() < ply {
        let quiet = quiet_moves(&b);
        if quiet.is_empty() {
            return None;
        }
        let m = quiet[rng.below(quiet.len())];
        b = b.play(m);
        moves.push(m);
    }
    Some(moves)
}

/// The keys of one witness board per canonical non-terminal position at plies
/// 0..=[EXHAUSTIVE_PLIES], reached by playing quiet moves in column order.
fn exhaustive_keys() -> HashSet<u64> {
    let mut canonical = HashSet::new();
    let mut keys = HashSet::new();
    let mut level = vec![Board::default()];
    for _ in 0..=EXHAUSTIVE_PLIES {
        let mut next = vec![];
        for b in level {
            if !canonical.insert(b.canonical_key()) {
                continue;
            }
            keys.insert(b.key());
            next.extend(quiet_moves(&b).into_iter().map(|m| b.play(m)));
        }
        level = next;
    }
    keys
}

/// Up to `per_ply` distinct non-terminal walk positions at every ply 0..=41, from `seed`, in ply
/// order. A mirror is a distinct position.
pub fn walks(per_ply: usize, seed: u64) -> Vec<Vec<Move>> {
    let mut seen = exhaustive_keys();
    let mut ret = vec![];
    for ply in 0..=41 {
        let mut rng = Rng::new(seed, ply as u64);
        let mut found = 0;
        for _ in 0..per_ply * 4 {
            if let Some(m) = walk(ply, &mut rng) {
                if seen.insert(Board::from_moves(&m).unwrap().key()) {
                    ret.push(m);
                    found += 1;
                }
            }
            if found >= per_ply {
                break;
            }
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_are_reproducible_distinct_and_live_at_every_ply() {
        let a = walks(6, 11);
        assert_eq!(a, walks(6, 11));
        assert_ne!(a, walks(6, 12));
        let keys: HashSet<u64> = a
            .iter()
            .map(|m| Board::from_moves(m).unwrap().key())
            .collect();
        assert_eq!(keys.len(), a.len());
        for m in &a {
            let b = Board::from_moves(m).unwrap();
            assert!(!b.opponent_has_won() && b.ply() < 42, "terminal {:?}", m);
        }
        // Early walks mostly repeat excluded positions and late walks can dead-end, so a ply
        // may come up short, but past the excluded plies never empty.
        for ply in 0..=41 {
            let n = a.iter().filter(|m| m.len() == ply).count();
            let min = (ply > EXHAUSTIVE_PLIES) as usize;
            assert!((min..=6).contains(&n), "ply {}: {}", ply, n);
        }
    }
}
