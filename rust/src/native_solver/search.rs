//! [Searcher]: exact negamax with alpha-beta over [Board], a shared [Tt] and an optional
//! [Book], driven by null-window probes.
//!
//! Score algebra at a node with `n` stones: winning now places stone n+1 and scores
//! `(43 - n) / 2`, which equals 22 minus the winner's stone count. Losing to an immediate reply
//! scores `-(42 - n) / 2`. A node the side to move can neither win nor lose at once lies in
//! `[-(40 - n) / 2, (41 - n) / 2]`.

use super::board::{Board, CELLS, WIDTH};
use super::book::Book;
use super::tt::{Bound, Tt};
use crate::solver::MoveScores;

/// Columns from the center out. Central stones take part in more lines, so they cut first.
const COLUMN_ORDER: [usize; WIDTH] = [3, 2, 4, 1, 5, 0, 6];

pub fn win_now_score(ply: u8) -> i8 {
    (43 - ply as i8) / 2
}

/// Move ordering policy. [Order::Threats] is the default; [Order::Columns] is the plain center
/// order, kept as an independent configuration for adjudicating disagreements.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    Threats,
    Columns,
}

pub struct Searcher<'a> {
    tt: &'a Tt,
    book: Option<&'a Book>,
    order: Order,
    /// Nodes visited, for measurement only.
    pub nodes: u64,
}

impl<'a> Searcher<'a> {
    pub fn new(tt: &'a Tt, book: Option<&'a Book>) -> Self {
        Searcher {
            tt,
            book,
            order: Order::Threats,
            nodes: 0,
        }
    }

    pub fn with_order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }

    /// The exact score of a non-terminal `b` for the side to move.
    pub fn solve(&mut self, b: &Board) -> i8 {
        let n = b.ply();
        if n == CELLS {
            return 0;
        }
        if b.can_win_now() {
            return win_now_score(n);
        }
        let n = n as i8;
        let (mut lo, mut hi) = (-(42 - n) / 2, (41 - n) / 2);
        while lo < hi {
            // Probe the midpoint, pulled toward zero: most positions are near a draw, and a probe
            // near the true value refutes cheaply.
            let mut probe = lo + (hi - lo) / 2;
            if probe <= 0 && lo / 2 < probe {
                probe = lo / 2;
            } else if probe >= 0 && hi / 2 > probe {
                probe = hi / 2;
            }
            let r = self.negamax(b, probe, probe + 1);
            if r <= probe {
                hi = r;
            } else {
                lo = r;
            }
        }
        lo
    }

    /// Scores of every column of a non-terminal `b`, like `c4solver -a`.
    pub fn analyze(&mut self, b: &Board) -> MoveScores {
        std::array::from_fn(|col| self.score_move(b, col))
    }

    /// The score of playing `col` in a non-terminal `b`, or None for a full column.
    pub fn score_move(&mut self, b: &Board, col: usize) -> Option<i8> {
        if !b.can_play(col) {
            return None;
        }
        if b.wins_with(col) {
            return Some(win_now_score(b.ply()));
        }
        Some(-self.solve(&b.play(col)))
    }

    /// Searches a node the side to move cannot win at once. Returns r with: r <= alpha means
    /// V <= r, r >= beta means V >= r, otherwise V == r. Every narrowing below comes from a true
    /// bound on V, which keeps that contract and makes every stored entry a true bound.
    fn negamax(&mut self, b: &Board, mut alpha: i8, mut beta: i8) -> i8 {
        debug_assert!(alpha < beta && !b.can_win_now());
        self.nodes += 1;
        let next = b.non_losing();
        let n = b.ply() as i8;
        if next == 0 {
            return -(42 - n) / 2;
        }
        if n >= CELLS as i8 - 2 {
            return 0;
        }
        let lo = -(40 - n) / 2;
        if alpha < lo {
            alpha = lo;
            if alpha >= beta {
                return alpha;
            }
        }
        let hi = (41 - n) / 2;
        if beta > hi {
            beta = hi;
            if alpha >= beta {
                return beta;
            }
        }
        let key = b.key();
        if let Some(e) = self.tt.get(key) {
            match e.bound {
                Bound::Exact => return e.score,
                Bound::Lower if e.score > alpha => {
                    alpha = e.score;
                    if alpha >= beta {
                        return alpha;
                    }
                }
                Bound::Upper if e.score < beta => {
                    beta = e.score;
                    if alpha >= beta {
                        return beta;
                    }
                }
                _ => {}
            }
        }
        if let Some(score) = self.book.and_then(|book| book.get(b)) {
            return score;
        }

        let alpha_in = alpha;
        let moves = self.ordered(b, next);
        for &cell in moves.cells() {
            let s = -self.negamax(&b.play_cell(cell), -beta, -alpha);
            if s >= beta {
                self.tt.put(key, s, Bound::Lower);
                return s;
            }
            if s > alpha {
                alpha = s;
            }
        }
        let bound = if alpha > alpha_in {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.put(key, alpha, bound);
        alpha
    }

    fn ordered(&self, b: &Board, next: u64) -> Moves {
        let mut moves = Moves::default();
        for col in COLUMN_ORDER {
            let cell = next & b.landing(col);
            if cell == 0 {
                continue;
            }
            let rank = match self.order {
                Order::Threats => {
                    let mine = b.mover() | cell;
                    (super::board::threats(mine) & !(b.occupied() | cell)).count_ones()
                }
                Order::Columns => 0,
            };
            moves.insert(cell, rank);
        }
        moves
    }
}

/// Up to [WIDTH] move cells sorted by rank, highest first, stable for equal ranks.
#[derive(Default)]
struct Moves {
    len: usize,
    cells: [u64; WIDTH],
    ranks: [u32; WIDTH],
}

impl Moves {
    fn insert(&mut self, cell: u64, rank: u32) {
        let mut i = self.len;
        while i > 0 && self.ranks[i - 1] < rank {
            self.cells[i] = self.cells[i - 1];
            self.ranks[i] = self.ranks[i - 1];
            i -= 1;
        }
        self.cells[i] = cell;
        self.ranks[i] = rank;
        self.len += 1;
    }

    fn cells(&self) -> &[u64] {
        &self.cells[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::c4r::Pos;
    use crate::native_solver::{brute, live_pos, suite_rows};

    /// Seeded positions at plies 30-41 that brute force solves quickly even in debug builds.
    fn late_positions(n: usize) -> Vec<Pos> {
        let mut state = 0x5eed_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut ret = vec![];
        while ret.len() < n {
            let target = 30 + next() % 12;
            let mut pos = Pos::default();
            for _ in 0..400 {
                if pos.ply() == target {
                    break;
                }
                let m = next() % Pos::N_COLS;
                if let Some(child) = pos.make_move(m) {
                    if child.is_terminal_state().is_none() {
                        pos = child;
                    }
                }
            }
            if pos.ply() == target {
                ret.push(pos);
            }
        }
        ret
    }

    #[test]
    fn matches_brute_force_on_late_positions() {
        let tt = Tt::with_bytes(1 << 20);
        for (i, pos) in late_positions(3000).iter().enumerate() {
            let want = brute::analyze(pos);
            // A tiny shared table forces collisions and overwrites across positions.
            let got = Searcher::new(&tt, None).analyze(&Board::from(pos));
            assert_eq!(got, want, "position {} {:?}", i, pos.to_moves());
            let cold = Tt::with_bytes(0);
            let got = Searcher::new(&cold, None)
                .with_order(Order::Columns)
                .analyze(&Board::from(pos));
            assert_eq!(got, want, "column order, position {}", i);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// The solved score is the best of the negated child scores, and wins now are scored.
        #[test]
        fn solve_is_best_move_score(pos in live_pos(41)) {
            prop_assume!(pos.ply() >= 16);
            let tt = Tt::with_bytes(1 << 22);
            let mut s = Searcher::new(&tt, None);
            let b = Board::from(&pos);
            let scores = s.analyze(&b);
            prop_assert_eq!(scores.iter().flatten().max().copied(), Some(s.solve(&b)));
        }
    }

    #[test]
    fn matches_committed_pons_scores_from_ply_20() {
        let tt = Tt::with_bytes(16 << 20);
        let mut s = Searcher::new(&tt, None);
        let rows: Vec<_> = suite_rows("dev")
            .into_iter()
            .filter(|r| r.moves.len() >= 20)
            .collect();
        assert!(rows.len() > 1000);
        for r in rows {
            let b = Board::from_moves(&r.moves).unwrap();
            assert_eq!(s.analyze(&b), r.scores, "{:?}", r.moves);
        }
    }
}
