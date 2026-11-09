//! The native Connect Four solver's command line. See `c4native help`.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use c4a0_rust::native_solver::bench::BenchConfig;
use c4a0_rust::native_solver::board::Board;
use c4a0_rust::native_solver::book::{positions_at, Book};
use c4a0_rust::native_solver::parallel::{analyze_batch, default_threads, solve_batch};
use c4a0_rust::native_solver::tt::Tt;

const USAGE: &str = "\
c4native analyze [--book FILE] [--tt-mb N] [--threads N]
    Reads one position per stdin line as 1-indexed column digits (empty line = empty board) and
    prints the line followed by seven scores, -1000 for a full column.
c4native book --ply P --out FILE [--deeper FILE] [--tt-mb N] [--threads N] [--estimate K]
    Solves every non-terminal position at ply P into a book. --estimate K solves K evenly spaced
    positions instead and prints the projected build time.


c4native bench [--book FILE] [--per-ply N] [--seed S] [--threads-list 1,8,32] [--runs R]
        [--tt-mb N]
    Times analysis of the seeded walk positions at each thread count, interleaved, each with a
    cold table, checking every run's rows against the first.

--tt-mb is the shared transposition table size (default 256); --threads defaults to the
available parallelism.";

struct Opts {
    book: Option<PathBuf>,
    deeper: Option<PathBuf>,
    out: Option<PathBuf>,
    ply: Option<u8>,
    tt_mb: usize,
    threads: usize,
    estimate: Option<usize>,
    per_ply: usize,
    seed: u64,
    threads_list: Vec<usize>,
    runs: usize,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        book: None,
        deeper: None,
        out: None,
        ply: None,
        tt_mb: Tt::DEFAULT_BYTES >> 20,
        threads: default_threads(),
        estimate: None,
        per_ply: 200,
        seed: 1,
        threads_list: vec![1, 8, 32],
        runs: 5,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{} needs a value", flag));
        let num = |v: &String| v.parse::<usize>().map_err(|e| format!("{}: {}", flag, e));
        match flag.as_str() {
            "--book" => o.book = Some(value()?.into()),
            "--deeper" => o.deeper = Some(value()?.into()),
            "--out" => o.out = Some(value()?.into()),
            "--ply" => o.ply = Some(num(value()?)? as u8),
            "--tt-mb" => o.tt_mb = num(value()?)?,
            "--threads" => o.threads = num(value()?)?.max(1),
            "--estimate" => o.estimate = Some(num(value()?)?),
            "--per-ply" => o.per_ply = num(value()?)?,
            "--seed" => o.seed = num(value()?)? as u64,
            "--runs" => o.runs = num(value()?)?.max(1),
            "--threads-list" => {
                o.threads_list = value()?
                    .split(',')
                    .map(|t| {
                        t.parse::<usize>()
                            .map_err(|e| format!("--threads-list: {}", e))
                    })
                    .collect::<Result<_, _>>()?
            }
            _ => return Err(format!("unknown flag {}", flag)),
        }
    }
    Ok(o)
}

fn load_book(path: &Option<PathBuf>) -> Result<Option<Book>, String> {
    path.as_ref()
        .map(|p| Book::load(p).map_err(|e| e.to_string()))
        .transpose()
}

/// Parses an `analyze` line: 1-indexed column digits from the empty board.
fn parse_line(line: &str) -> Result<Board, String> {
    let moves = line
        .chars()
        .map(|c| match c.to_digit(10) {
            Some(d @ 1..=7) => Ok(d as usize - 1),
            _ => Err(format!("{:?}: not a column 1-7", line)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let b = Board::from_moves(&moves).ok_or(format!("{:?}: illegal move sequence", line))?;
    if b.opponent_has_won() || b.ply() == 42 {
        return Err(format!("{:?}: game is over", line));
    }
    Ok(b)
}

fn analyze(o: &Opts) -> Result<(), String> {
    let book = load_book(&o.book)?;
    let lines: Vec<String> = std::io::stdin()
        .lock()
        .lines()
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let boards = lines
        .iter()
        .map(|l| parse_line(l.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    let tt = Tt::with_bytes(o.tt_mb << 20);
    let scores = analyze_batch(&boards, &tt, book.as_ref(), o.threads, &|_| {});
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (line, s) in lines.iter().zip(scores) {
        let cols: Vec<String> = s
            .iter()
            .map(|c| c.map_or("-1000".into(), |v| v.to_string()))
            .collect();
        writeln!(out, "{} {}", line.trim(), cols.join(" ")).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn book(o: &Opts) -> Result<(), String> {
    let ply = o.ply.ok_or("--ply is required")?;
    let deeper = load_book(&o.deeper)?;
    let tt = Tt::with_bytes(o.tt_mb << 20);
    let t = Instant::now();
    let all = positions_at(ply);
    eprintln!(
        "ply {}: {} positions, enumerated in {:.1?}",
        ply,
        all.len(),
        t.elapsed()
    );
    let progress = |done: usize, total: usize| {
        if done % 1000 == 0 || done == total {
            eprintln!("{}/{} solved, {:.1?}", done, total, t.elapsed());
        }
    };
    if let Some(k) = o.estimate {
        let step = (all.len() / k.max(1)).max(1);
        let sample: Vec<Board> = all.iter().step_by(step).take(k).copied().collect();
        let t = Instant::now();
        solve_batch(&sample, &tt, deeper.as_ref(), o.threads, &|d| {
            progress(d, sample.len())
        });
        let per = t.elapsed().as_secs_f64() / sample.len() as f64;
        println!(
            "{} positions in {:.2?} on {} threads: {:.1} positions/s, projected {:.0} s for {}",
            sample.len(),
            t.elapsed(),
            o.threads,
            1.0 / per,
            per * all.len() as f64,
            all.len()
        );
        return Ok(());
    }
    let out = o.out.as_ref().ok_or("--out is required")?;
    let book = Book::solve(&all, &tt, deeper.as_ref(), o.threads, &|d| {
        progress(d, all.len())
    });
    book.save(out).map_err(|e| e.to_string())?;
    eprintln!(
        "wrote {} entries to {:?} in {:.1?}",
        book.len(),
        out,
        t.elapsed()
    );
    Ok(())
}

fn bench(o: &Opts) -> Result<(), String> {
    let cfg = BenchConfig {
        per_ply: o.per_ply,
        seed: o.seed,
        threads: o.threads_list.clone(),
        runs: o.runs,
        tt_bytes: o.tt_mb << 20,
        book: load_book(&o.book)?,
    };
    print!("{}", c4a0_rust::native_solver::bench::run(&cfg)?);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some(cmd @ ("analyze" | "book" | "bench")) => parse(&args[1..]).and_then(|o| match cmd {
            "analyze" => analyze(&o),
            "book" => book(&o),
            _ => bench(&o),
        }),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(2);
        }
    }
}
