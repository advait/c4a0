//! Throughput and scaling measurement on a labeling-like workload: [sample::walks], uniform per
//! ply 0-41. Thread counts run interleaved, each with a cold table, and every run's rows are
//! checked against the first run's, so a fast wrong run cannot pass as a speedup.

use std::time::Instant;

use super::board::Board;
use super::book::Book;
use super::parallel::analyze_batch;
use super::sample;
use super::search::Searcher;
use super::tt::Tt;
use super::MoveScores;

pub struct BenchConfig {
    pub per_ply: usize,
    pub seed: u64,
    pub threads: Vec<usize>,
    pub runs: usize,
    pub tt_bytes: usize,
    pub book: Option<Book>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Native(usize),
}

fn median_range(xs: &[f64]) -> (f64, f64, f64) {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

/// Peak resident memory of this process in KiB, from /proc (Linux only).
pub fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn load_average() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

/// Runs the benchmark and returns a plain-text report. Errors if any run's rows differ.
pub fn run(cfg: &BenchConfig) -> Result<String, String> {
    let cases = sample::walks(cfg.per_ply, cfg.seed);
    let boards: Vec<Board> = cases
        .iter()
        .map(|m| Board::from_moves(m).expect("walks are legal"))
        .collect();
    let max_threads = cfg.threads.iter().copied().max().unwrap_or(1);
    let sides: Vec<Side> = cfg.threads.iter().map(|&t| Side::Native(t)).collect();
    let mut report = format!(
        "workload: {} walk positions, {} per ply 0-41, seed {}; tt {} MiB; book ply {:?}\n",
        boards.len(),
        cfg.per_ply,
        cfg.seed,
        cfg.tt_bytes >> 20,
        cfg.book.as_ref().map(|b| b.ply())
    );
    let mut reference: Option<Vec<MoveScores>> = None;
    let mut secs: Vec<Vec<f64>> = vec![vec![]; sides.len()];
    let mut per_position: Vec<f64> = vec![];
    for run in 0..cfg.runs {
        // Rotate the order every run so warmup and drift favor no side.
        for k in 0..sides.len() {
            let s = (k + run) % sides.len();
            let side = sides[s];
            let loadavg = load_average();
            let t = Instant::now();
            let rows = match side {
                Side::Native(1) => {
                    let tt = Tt::with_bytes(cfg.tt_bytes);
                    let mut searcher = Searcher::new(&tt, cfg.book.as_ref());
                    let mut times = vec![];
                    let rows = boards
                        .iter()
                        .map(|b| {
                            let t = Instant::now();
                            let r = searcher.analyze(b);
                            times.push(t.elapsed().as_secs_f64());
                            r
                        })
                        .collect();
                    if per_position.is_empty() {
                        per_position = times;
                    }
                    rows
                }
                Side::Native(n) => {
                    let tt = Tt::with_bytes(cfg.tt_bytes);
                    analyze_batch(&boards, &tt, cfg.book.as_ref(), n, &|_| {})
                }
            };
            let elapsed = t.elapsed().as_secs_f64();
            match &reference {
                None => reference = Some(rows),
                Some(r) if *r != rows => {
                    let bad = r.iter().zip(&rows).filter(|(a, b)| a != b).count();
                    return Err(format!(
                        "{:?} run {}: {} rows differ from the first run",
                        side, run, bad
                    ));
                }
                _ => {}
            }
            secs[s].push(elapsed);
            report += &format!(
                "run {} {:?}: {:.2} s, {:.0} positions/s (load average before: {})\n",
                run,
                side,
                elapsed,
                boards.len() as f64 / elapsed,
                loadavg
            );
        }
    }
    report += "summary (median, range over runs):\n";
    for (side, xs) in sides.iter().zip(&secs) {
        let (med, lo, hi) = median_range(xs);
        report += &format!(
            "  {:?}: {:.2} s ({:.2}-{:.2}), {:.0} positions/s ({:.0}-{:.0}), {} runs\n",
            side,
            med,
            lo,
            hi,
            boards.len() as f64 / med,
            boards.len() as f64 / hi,
            boards.len() as f64 / lo,
            xs.len()
        );
    }
    if !per_position.is_empty() {
        let (i, t1) =
            per_position.iter().enumerate().fold(
                (0, 0.0),
                |acc, (i, &t)| if t > acc.1 { (i, t) } else { acc },
            );
        let moves: String = cases[i].iter().map(|m| m.to_string()).collect();
        report += &format!(
            "hardest position: {:?} (ply {}), {:.2} s in the first 1-thread run\n",
            moves,
            cases[i].len(),
            t1
        );
        for &n in &[1, max_threads] {
            let mut xs = vec![];
            for _ in 0..cfg.runs {
                let tt = Tt::with_bytes(cfg.tt_bytes);
                let t = Instant::now();
                analyze_batch(&boards[i..=i], &tt, cfg.book.as_ref(), n, &|_| {});
                xs.push(t.elapsed().as_secs_f64());
            }
            let (med, lo, hi) = median_range(&xs);
            report += &format!(
                "  alone, cold table, {} threads: {:.2} s ({:.2}-{:.2}), {} runs\n",
                n,
                med,
                lo,
                hi,
                xs.len()
            );
        }
        let by_ply = |lo: usize, hi: usize| -> f64 {
            cases
                .iter()
                .zip(&per_position)
                .filter(|(c, _)| (lo..=hi).contains(&c.len()))
                .map(|(_, t)| t)
                .sum()
        };
        report += "1-thread seconds by ply bucket (first run):";
        for b in 0..7 {
            report += &format!(" {}-{}: {:.1}", b * 6, b * 6 + 5, by_ply(b * 6, b * 6 + 5));
        }
        report += "\n";
    }
    report += &format!(
        "peak RSS of this process: {} MiB\n",
        peak_rss_kib().map_or("?".into(), |k| (k >> 10).to_string())
    );
    Ok(report)
}
