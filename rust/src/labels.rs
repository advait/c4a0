//! Solver-labeled positions: move strings, mirror-canonical identity, split membership and the
//! position sampler. [Split::of] is the single owner of dev/test/train membership.

use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt,
    ops::RangeInclusive,
};

use crate::c4r::{Move, Pos};

/// Mirror-canonical identity of a board. A position and its horizontal mirror share a key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PositionKey(u64, u64);

impl PositionKey {
    pub fn of(pos: &Pos) -> Self {
        let (mask, value) = pos.bits();
        let (m, v) = (mask, value).min((mirror(mask), mirror(value)));
        PositionKey(m, v)
    }
}

/// Mirrors a row-major bitboard (bit `row * 7 + col`) left to right, as [Pos::flip_h] does.
fn mirror(bits: u64) -> u64 {
    (0..Pos::N_ROWS).fold(0, |acc, row| {
        let shift = row * Pos::N_COLS;
        let cols = ((bits >> shift) & 0x7f) as u8;
        acc | (((cols.reverse_bits() >> 1) as u64) << shift)
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Split {
    Train,
    Dev,
    Test,
}

impl Split {
    /// Changing the salt reassigns every position to a new split, which invalidates every
    /// labeled file. Bump the suite version if you do.
    const SALT: u64 = 0xC4A0_5017_5EED_0001;

    /// 10% test, 10% dev, 80% train, by a salted hash of the canonical key.
    pub fn of(key: &PositionKey) -> Split {
        match mix(mix(key.0 ^ Self::SALT) ^ key.1) % 100 {
            0..=9 => Split::Test,
            10..=19 => Split::Dev,
            _ => Split::Train,
        }
    }

    pub fn parse(s: &str) -> Result<Split, String> {
        match s {
            "train" => Ok(Split::Train),
            "dev" => Ok(Split::Dev),
            "test" => Ok(Split::Test),
            _ => Err(format!(
                "unknown split {:?}, expected train, dev or test",
                s
            )),
        }
    }
}

/// Parses a move string of 0-indexed column digits from the empty board. Rejects digits outside
/// 0-6, moves into full columns and moves after the game ended. The final position may be
/// terminal.
pub fn parse_moves(s: &str) -> Result<Pos, String> {
    let mut pos = Pos::default();
    for (i, c) in s.chars().enumerate() {
        let col = c
            .to_digit(10)
            .filter(|&d| (d as usize) < Pos::N_COLS)
            .ok_or_else(|| format!("{:?}: move {} is not a column 0-6", s, i))?
            as usize;
        if pos.is_terminal_state().is_some() {
            return Err(format!("{:?}: move {} is after the game ended", s, i));
        }
        if !pos.legal_moves()[col] {
            return Err(format!("{:?}: move {} plays full column {}", s, i, col));
        }
        pos = pos.make_move(col).expect("legal move");
    }
    Ok(pos)
}

pub fn moves_string(moves: &[Move]) -> String {
    moves.iter().map(|m| m.to_string()).collect()
}

/// A sampled position and the move sequence that reached it, in walked orientation.
#[derive(Clone, Debug)]
pub struct Sampled {
    pub moves: Vec<Move>,
    pub pos: Pos,
}

/// A ply whose quota was not met within its attempt budget.
#[derive(Debug, PartialEq, Eq)]
pub struct Shortfall {
    pub ply: usize,
    pub found: usize,
    pub quota: usize,
    pub attempts: usize,
}

impl fmt::Display for Shortfall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ply {}: found {} of {} positions in {} attempts",
            self.ply, self.found, self.quota, self.attempts
        )
    }
}

impl Error for Shortfall {}

/// Walk attempts per requested row before a ply fails with [Shortfall].
const ATTEMPTS_PER_ROW: usize = 200;

/// Plies whose full level of non-terminal canonical positions has at most this many keys are
/// enumerated exactly, so a split holding fewer keys than the quota takes all of them.
const ENUM_CAP: usize = 100_000;

/// Samples up to `per_ply` positions at every ply in `plies`, all in `split`, non-terminal and with
/// unique keys. Each ply has its own RNG seeded from `(seed, ply)`. Even attempts walk uniform
/// legal moves and restart if the game ends; odd attempts walk uniformly among moves that do not
/// end the game, which reaches late plies. A ply takes fewer than `per_ply` rows only when
/// exact enumeration shows the split has fewer keys at that ply; any other shortfall is an error.
pub fn sample_positions(
    split: Split,
    plies: RangeInclusive<usize>,
    per_ply: usize,
    seed: u64,
) -> Result<Vec<Sampled>, Shortfall> {
    let levels = enumerate_levels(*plies.end(), split, per_ply);
    let mut ret = Vec::new();
    for ply in plies {
        let mut rng = WalkRng::new(seed, ply as u64);
        let exhaustive: Option<Vec<(&PositionKey, &Vec<Move>)>> = levels.get(ply).map(|level| {
            level
                .iter()
                .filter(|(key, _)| Split::of(key) == split)
                .collect()
        });
        match exhaustive {
            Some(keys) if keys.len() <= per_ply => {
                for (_, witness) in keys {
                    let moves = if rng.below(2) == 1 {
                        witness.iter().map(|m| Pos::N_COLS - 1 - m).collect()
                    } else {
                        witness.clone()
                    };
                    let pos = Pos::from_moves(&moves);
                    ret.push(Sampled { moves, pos });
                }
            }
            _ => ret.extend(walk_ply(
                split,
                ply,
                per_ply,
                ATTEMPTS_PER_ROW * per_ply,
                &mut rng,
            )?),
        }
    }
    Ok(ret)
}

fn walk_ply(
    split: Split,
    ply: usize,
    per_ply: usize,
    budget: usize,
    rng: &mut WalkRng,
) -> Result<Vec<Sampled>, Shortfall> {
    let mut seen = HashSet::new();
    let mut ret = Vec::with_capacity(per_ply);
    for attempt in 0..budget {
        if ret.len() == per_ply {
            break;
        }
        let Some(sampled) = walk(ply, attempt % 2 == 1, rng) else {
            continue;
        };
        let key = PositionKey::of(&sampled.pos);
        if Split::of(&key) == split && seen.insert(key) {
            ret.push(sampled);
        }
    }
    if ret.len() < per_ply {
        return Err(Shortfall {
            ply,
            found: ret.len(),
            quota: per_ply,
            attempts: budget,
        });
    }
    Ok(ret)
}

/// One random walk of `ply` moves. Returns None if the walk hit a terminal position or a dead end.
fn walk(ply: usize, avoid_terminal: bool, rng: &mut WalkRng) -> Option<Sampled> {
    let mut pos = Pos::default();
    let mut moves = Vec::with_capacity(ply);
    for _ in 0..ply {
        let candidates: Vec<(Move, Pos)> = (0..Pos::N_COLS)
            .filter(|&m| pos.legal_moves()[m])
            .map(|m| (m, pos.make_move(m).expect("legal move")))
            .filter(|(_, child)| !avoid_terminal || child.is_terminal_state().is_none())
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let (m, child) = candidates[rng.below(candidates.len())].clone();
        if child.is_terminal_state().is_some() {
            return None;
        }
        moves.push(m);
        pos = child;
    }
    Some(Sampled { moves, pos })
}

/// Breadth-first levels of non-terminal positions keyed canonically, each with the first move
/// sequence that reached it. Stops after `max_ply`, after the first level where `split` holds more
/// than `per_ply` keys (later levels only grow), or before a level larger than [ENUM_CAP].
fn enumerate_levels(
    max_ply: usize,
    split: Split,
    per_ply: usize,
) -> Vec<BTreeMap<PositionKey, Vec<Move>>> {
    let mut levels = vec![BTreeMap::from([(PositionKey::of(&Pos::default()), vec![])])];
    while levels.len() <= max_ply {
        let prev = levels.last().expect("non-empty");
        if prev.keys().filter(|k| Split::of(k) == split).count() > per_ply {
            break;
        }
        let mut next = BTreeMap::new();
        for witness in prev.values() {
            let pos = Pos::from_moves(witness);
            for m in (0..Pos::N_COLS).filter(|&m| pos.legal_moves()[m]) {
                let child = pos.make_move(m).expect("legal move");
                if child.is_terminal_state().is_none() {
                    next.entry(PositionKey::of(&child)).or_insert_with(|| {
                        let mut moves = witness.clone();
                        moves.push(m);
                        moves
                    });
                }
            }
        }
        if next.len() > ENUM_CAP {
            break;
        }
        levels.push(next);
    }
    levels
}

/// SplitMix64 finalizer. Stable across Rust and `rand` versions, which keeps labeled files
/// reproducible.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct WalkRng(u64);

impl WalkRng {
    fn new(seed: u64, stream: u64) -> Self {
        WalkRng(mix(mix(seed) ^ stream))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// Uniform in 0..n (multiply-shift, bias below 2^-57 for n <= 7).
    fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use proptest::prelude::*;

    use super::*;
    use crate::c4r::tests::random_pos;

    fn digits(s: &str) -> Vec<Move> {
        s.chars().map(|c| c.to_digit(10).unwrap() as Move).collect()
    }

    proptest! {
        #[test]
        fn key_is_mirror_invariant(pos in random_pos()) {
            prop_assert_eq!(PositionKey::of(&pos), PositionKey::of(&pos.flip_h()));
            let (mask, value) = pos.bits();
            prop_assert_eq!((mirror(mask), mirror(value)), pos.flip_h().bits());
        }
    }

    #[test]
    fn key_separates_player_to_move_and_boards() {
        assert_ne!(
            PositionKey::of(&Pos::from_moves(&[3])),
            PositionKey::of(&Pos::from_moves(&[3, 3]))
        );
        assert_ne!(
            PositionKey::of(&Pos::from_moves(&[0, 1])),
            PositionKey::of(&Pos::from_moves(&[1, 0]))
        );
        assert_eq!(
            PositionKey::of(&Pos::from_moves(&[0, 1])),
            PositionKey::of(&Pos::from_moves(&[6, 5]))
        );
    }

    /// Pins SALT: if this fails, every committed labels file changed split.
    #[test]
    fn golden_splits() {
        let got: Vec<Split> = ["", "3", "23", "04", "33", "32", "142", "012"]
            .iter()
            .map(|s| Split::of(&PositionKey::of(&Pos::from_moves(&digits(s)))))
            .collect();
        use Split::*;
        assert_eq!(got, vec![Train, Train, Dev, Dev, Test, Test, Dev, Test]);
    }

    #[test]
    fn split_shares_are_ten_ten_eighty() {
        let level = &enumerate_levels(6, Split::Train, usize::MAX)[6];
        let mut counts = HashMap::new();
        for key in level.keys() {
            *counts.entry(Split::of(key)).or_insert(0usize) += 1;
        }
        let share = |s| counts[&s] as f64 / level.len() as f64;
        assert!((share(Split::Test) - 0.1).abs() < 0.015, "{:?}", counts);
        assert!((share(Split::Dev) - 0.1).abs() < 0.015, "{:?}", counts);
    }

    #[test]
    fn enumerated_levels_match_brute_force() {
        // Distinct boards per ply from every move sequence (OEIS A212693: 1, 7, 49, 238, 1120;
        // none is terminal before ply 7). Levels hold one key per mirror orbit.
        let mut boards = vec![HashSet::from([Pos::default()])];
        for ply in 1..=5 {
            let next: HashSet<Pos> = boards[ply - 1]
                .iter()
                .flat_map(|p| (0..Pos::N_COLS).filter_map(move |m| p.make_move(m)))
                .collect();
            boards.push(next);
        }
        let sizes: Vec<usize> = boards.iter().map(|b| b.len()).collect();
        assert_eq!(sizes, vec![1, 7, 49, 238, 1120, 4263]);
        let levels = enumerate_levels(5, Split::Train, usize::MAX);
        for (ply, level) in levels.iter().enumerate() {
            let orbits: HashSet<PositionKey> = boards[ply].iter().map(PositionKey::of).collect();
            assert_eq!(level.keys().copied().collect::<HashSet<_>>(), orbits);
            for (key, witness) in level {
                assert_eq!(PositionKey::of(&Pos::from_moves(witness)), *key);
            }
        }
    }

    #[test]
    fn parse_moves_rejects_bad_input() {
        assert_eq!(parse_moves("33").unwrap(), Pos::from_moves(&[3, 3]));
        assert_eq!(parse_moves("").unwrap(), Pos::default());
        assert!(parse_moves("7").is_err());
        assert!(parse_moves("x").is_err());
        assert!(parse_moves("0000000").is_err());
        // First player wins with 0,0,0,0 on the seventh move; an eighth move is rejected.
        assert!(parse_moves("0101010").is_ok());
        assert!(parse_moves("01010102").is_err());
    }

    fn check_sample(split: Split, rows: &[Sampled], plies: RangeInclusive<usize>, per_ply: usize) {
        let mut keys = HashSet::new();
        let mut per = HashMap::new();
        for row in rows {
            assert_eq!(row.pos, Pos::from_moves(&row.moves));
            assert!(row.pos.is_terminal_state().is_none());
            let key = PositionKey::of(&row.pos);
            assert_eq!(Split::of(&key), split);
            assert!(keys.insert(key), "duplicate key");
            *per.entry(row.moves.len()).or_insert(0usize) += 1;
        }
        let levels = enumerate_levels(*plies.end(), split, per_ply);
        for ply in plies {
            let available = levels
                .get(ply)
                .map(|l| l.keys().filter(|k| Split::of(k) == split).count());
            let expected = available.map_or(per_ply, |a| a.min(per_ply));
            assert_eq!(per.get(&ply).copied().unwrap_or(0), expected, "ply {}", ply);
        }
    }

    #[test]
    fn sampling_is_deterministic_and_pure() {
        let a = sample_positions(Split::Dev, 0..=41, 8, 1).unwrap();
        let b = sample_positions(Split::Dev, 0..=41, 8, 1).unwrap();
        let strings = |r: &[Sampled]| r.iter().map(|s| moves_string(&s.moves)).collect::<Vec<_>>();
        assert_eq!(strings(&a), strings(&b));
        check_sample(Split::Dev, &a, 0..=41, 8);
        let c = sample_positions(Split::Dev, 0..=41, 8, 2).unwrap();
        assert_ne!(strings(&a), strings(&c));
    }

    #[test]
    fn small_plies_take_every_key_in_split() {
        let rows = sample_positions(Split::Test, 0..=4, 64, 3).unwrap();
        check_sample(Split::Test, &rows, 0..=4, 64);
    }

    #[test]
    fn unmet_quota_is_an_error() {
        let mut rng = WalkRng::new(0, 20);
        let err = walk_ply(Split::Dev, 20, 50, 10, &mut rng).unwrap_err();
        assert_eq!(err.ply, 20);
        assert_eq!(err.quota, 50);
        assert_eq!(err.attempts, 10);
        assert!(err.found < 50);
    }

    /// The committed suites: rows parse, are non-terminal, belong to their split, have unique keys
    /// shared by no other split, mark exactly the full columns, and fill every ply's quota (or
    /// every key the split has at that ply).
    #[test]
    fn committed_suites_are_split_pure_and_complete() {
        let suites = [
            (Split::Dev, include_str!("../../src/c4a0/suite/v1/dev.tsv")),
            (
                Split::Test,
                include_str!("../../src/c4a0/suite/v1/test.tsv"),
            ),
        ];
        let mut all_keys = HashSet::new();
        for (split, text) in suites {
            let mut rows = Vec::new();
            for line in text.lines().skip(2) {
                let (moves, scores) = line.split_once('\t').unwrap();
                let pos = parse_moves(moves).unwrap();
                assert!(pos.is_terminal_state().is_none(), "{:?}", moves);
                let full: Vec<bool> = scores.split(',').map(|s| s == "x").collect();
                let legal = pos.legal_moves();
                assert_eq!(
                    full,
                    legal.iter().map(|l| !l).collect::<Vec<_>>(),
                    "{:?}",
                    moves
                );
                assert!(
                    all_keys.insert(PositionKey::of(&pos)),
                    "shared key {:?}",
                    moves
                );
                rows.push(Sampled {
                    moves: moves
                        .chars()
                        .map(|c| c.to_digit(10).unwrap() as Move)
                        .collect(),
                    pos,
                });
            }
            check_sample(split, &rows, 0..=41, 64);
        }
    }
}
