//! Parallel analysis. Work is one (position, column) pair, handed out from a shared counter, so a
//! batch spreads across positions and a single position spreads across its columns. Every
//! worker shares one [Tt], so a subtree one worker solved is a probe for the others.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::board::{Board, WIDTH};
use super::book::Book;
use super::search::Searcher;
use super::tt::Tt;
use super::MoveScores;

/// The machine's available parallelism, or 1 if unknown.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Scores every column of every non-terminal board, on `threads` workers. `progress` is
/// called with the number of finished columns after each one, from worker threads.
pub fn analyze_batch(
    boards: &[Board],
    tt: &Tt,
    book: Option<&Book>,
    threads: usize,
    progress: &(dyn Fn(usize) + Sync),
) -> Vec<MoveScores> {
    let scores = map(
        boards.len() * WIDTH,
        tt,
        book,
        threads,
        progress,
        |s, item| s.score_move(&boards[item / WIDTH], item % WIDTH),
    );
    scores
        .chunks(WIDTH)
        .map(|c| c.try_into().expect("WIDTH scores"))
        .collect()
}

/// The exact score of every non-terminal board, one board per work item.
pub fn solve_batch(
    boards: &[Board],
    tt: &Tt,
    book: Option<&Book>,
    threads: usize,
    progress: &(dyn Fn(usize) + Sync),
) -> Vec<i8> {
    map(boards.len(), tt, book, threads, progress, |s, i| {
        s.solve(&boards[i])
    })
}

/// Runs `f` on items `0..n` across `threads` workers, each with its own [Searcher] on the shared
/// table, and returns the results in item order.
fn map<T: Send>(
    n: usize,
    tt: &Tt,
    book: Option<&Book>,
    threads: usize,
    progress: &(dyn Fn(usize) + Sync),
    f: impl Fn(&mut Searcher, usize) -> T + Sync,
) -> Vec<T> {
    par_map(n, threads, progress, || Searcher::new(tt, book), f)
}

/// Runs `f` on items `0..n` across `threads` workers, each with worker state from `init`, and
/// returns the results in item order. Items are handed out one at a time from a shared counter,
/// so uneven items balance.
pub fn par_map<S, T: Send>(
    n: usize,
    threads: usize,
    progress: &(dyn Fn(usize) + Sync),
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize) -> T + Sync,
) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results = Mutex::new((0..n).map(|_| None).collect::<Vec<Option<T>>>());
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            s.spawn(|| {
                let mut state = init();
                let mut local = vec![];
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    local.push((i, f(&mut state, i)));
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1);
                }
                let mut results = results.lock().unwrap();
                for (i, r) in local {
                    results[i] = Some(r);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("every item ran"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_solver::suite_rows;

    #[test]
    fn batch_matches_committed_suite_scores_at_any_thread_count() {
        let rows: Vec<_> = suite_rows("test")
            .into_iter()
            .filter(|r| r.moves.len() >= 18)
            .collect();
        let boards: Vec<Board> = rows
            .iter()
            .map(|r| Board::from_moves(&r.moves).unwrap())
            .collect();
        for threads in [1, 3, 16] {
            let tt = Tt::with_bytes(4 << 20);
            let got = analyze_batch(&boards, &tt, None, threads, &|_| {});
            for (r, g) in rows.iter().zip(got) {
                assert_eq!(g, r.scores, "{:?} threads={}", r.moves, threads);
            }
        }
    }
}
