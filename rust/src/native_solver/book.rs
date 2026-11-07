//! [Book]: exact scores of every non-terminal position at one ply, keyed by mirror-canonical
//! [Board::canonical_key]. The search stops at that ply, which bounds the depth of early solves.
//! The native solver builds its own book ([build]); it never reads the Pons book.
//!
//! File format, little-endian: the 8-byte magic `C4NBOOK1`, the ply (u8), the entry count
//! (u64), then that many keys (u64, strictly increasing) followed by as many scores (i8).

use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::path::Path;

use super::board::{Board, WIDTH};
use super::parallel::solve_batch;
use super::tt::Tt;

const MAGIC: &[u8; 8] = b"C4NBOOK1";

pub struct Book {
    ply: u8,
    keys: Vec<u64>,
    scores: Vec<i8>,
}

impl Book {
    pub fn ply(&self) -> u8 {
        self.ply
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// The exact score of `b` for the side to move, when `b` is at the book's ply.
    pub fn get(&self, b: &Board) -> Option<i8> {
        if b.ply() != self.ply {
            return None;
        }
        let i = self.keys.binary_search(&b.canonical_key()).ok()?;
        Some(self.scores[i])
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut out = io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&[self.ply])?;
        out.write_all(&(self.keys.len() as u64).to_le_bytes())?;
        for k in &self.keys {
            out.write_all(&k.to_le_bytes())?;
        }
        let scores: Vec<u8> = self.scores.iter().map(|&s| s as u8).collect();
        out.write_all(&scores)?;
        out.flush()
    }

    /// Loads a book and checks its framing and key order. Scores are trusted: the file is this
    /// solver's own output (see [build]).
    pub fn load(path: &Path) -> io::Result<Book> {
        let bad =
            |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{:?}: {}", path, m));
        let mut bytes = vec![];
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        if bytes.len() < 17 || &bytes[..8] != MAGIC {
            return Err(bad("not a native solver book"));
        }
        let ply = bytes[8];
        let n = u64::from_le_bytes(bytes[9..17].try_into().unwrap()) as usize;
        if bytes.len() != 17 + 9 * n {
            return Err(bad("truncated"));
        }
        let keys: Vec<u64> = bytes[17..17 + 8 * n]
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if keys.windows(2).any(|w| w[0] >= w[1]) {
            return Err(bad("keys not strictly increasing"));
        }
        let scores = bytes[17 + 8 * n..].iter().map(|&b| b as i8).collect();
        Ok(Book { ply, keys, scores })
    }
}

/// One board per canonical key among the non-terminal positions at `ply`, in generation order:
/// children of one parent are adjacent, so neighbors share subtrees in a shared table.
pub fn positions_at(ply: u8) -> Vec<Board> {
    let mut level = vec![Board::default()];
    for _ in 0..ply {
        let mut seen = HashSet::new();
        let mut next = vec![];
        for b in &level {
            for col in (0..WIDTH).filter(|&c| b.can_play(c) && !b.wins_with(c)) {
                let child = b.play(col);
                if seen.insert(child.canonical_key()) {
                    next.push(child);
                }
            }
        }
        level = next;
    }
    level
}

impl Book {
    /// Solves `boards`, which must share one ply, into a book. `deeper` may be a book at a later
    /// ply to speed the solve; the scores do not depend on it.
    pub fn solve(
        boards: &[Board],
        tt: &Tt,
        deeper: Option<&Book>,
        threads: usize,
        progress: &(dyn Fn(usize) + Sync),
    ) -> Book {
        let ply = boards.first().map_or(0, |b| b.ply());
        assert!(boards.iter().all(|b| b.ply() == ply), "mixed plies");
        let scores = solve_batch(boards, tt, deeper, threads, progress);
        let mut entries: Vec<(u64, i8)> = boards
            .iter()
            .map(|b| b.canonical_key())
            .zip(scores)
            .collect();
        entries.sort_unstable();
        entries.dedup();
        Book {
            ply,
            keys: entries.iter().map(|e| e.0).collect(),
            scores: entries.iter().map(|e| e.1).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::c4r::Pos;
    use crate::labels::PositionKey;
    use crate::native_solver::search::Searcher;
    use crate::native_solver::suite_rows;

    #[test]
    fn levels_match_an_independent_enumeration_over_pos() {
        let mut level: BTreeSet<PositionKey> = BTreeSet::from([PositionKey::of(&Pos::default())]);
        let mut witnesses = vec![Pos::default()];
        for ply in 0..=7u8 {
            assert_eq!(positions_at(ply).len(), level.len(), "ply {}", ply);
            let mut next_level = BTreeSet::new();
            let mut next_witnesses = vec![];
            for pos in &witnesses {
                for col in (0..Pos::N_COLS).filter(|&c| pos.legal_moves()[c]) {
                    let child = pos.make_move(col).unwrap();
                    if child.is_terminal_state().is_none()
                        && next_level.insert(PositionKey::of(&child))
                    {
                        next_witnesses.push(child);
                    }
                }
            }
            (level, witnesses) = (next_level, next_witnesses);
        }
    }

    /// Every descendant of `b` `depth` plies later that the game reaches.
    fn descendants(b: Board, depth: u8) -> Vec<Board> {
        let mut level = vec![b];
        for _ in 0..depth {
            level = level
                .iter()
                .flat_map(|b| {
                    (0..WIDTH)
                        .filter(|&c| b.can_play(c) && !b.wins_with(c))
                        .map(|c| b.play(c))
                        .collect::<Vec<_>>()
                })
                .collect();
        }
        level
    }

    #[test]
    fn searching_through_a_book_matches_pons_and_reads_the_book() {
        let rows: Vec<_> = suite_rows("dev")
            .into_iter()
            .filter(|r| (18..=20).contains(&r.moves.len()))
            .take(12)
            .collect();
        assert_eq!(rows.len(), 12);
        let mut misread = 0;
        for r in rows {
            let root = Board::from_moves(&r.moves).unwrap();
            let tt = Tt::with_bytes(1 << 20);
            let book = Book::solve(&descendants(root, 4), &tt, None, 4, &|_| {});
            let path = std::env::temp_dir().join(format!("c4a0-book-{}", std::process::id()));
            book.save(&path).unwrap();
            let loaded = Book::load(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(
                (loaded.ply, &loaded.keys, &loaded.scores),
                (book.ply, &book.keys, &book.scores)
            );

            let cold = Tt::with_bytes(1 << 20);
            assert_eq!(
                Searcher::new(&cold, Some(&loaded)).analyze(&root),
                r.scores,
                "{:?}",
                r.moves
            );

            // Shift every book score toward zero: wrong answers prove the search read the book.
            let wrong = Book {
                scores: loaded.scores.iter().map(|&s| s - s.signum()).collect(),
                ..loaded
            };
            let cold = Tt::with_bytes(1 << 20);
            misread += (Searcher::new(&cold, Some(&wrong)).analyze(&root) != r.scores) as usize;
        }
        // Rows decided by forced tactics above the book ply never reach it; most rows do.
        assert!(misread >= 6, "only {} of 12 rows read the book", misread);
    }

    #[test]
    fn load_rejects_foreign_and_truncated_files() {
        let path = std::env::temp_dir().join(format!("c4a0-badbook-{}", std::process::id()));
        std::fs::write(&path, b"not a book at all").unwrap();
        assert!(Book::load(&path).is_err());
        let mut bytes = MAGIC.to_vec();
        bytes.push(8);
        bytes.extend(2u64.to_le_bytes());
        bytes.extend(5u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = Book::load(&path).err().expect("truncated book loads");
        assert!(err.to_string().contains("truncated"), "{}", err);
        std::fs::remove_file(&path).unwrap();
    }
}
