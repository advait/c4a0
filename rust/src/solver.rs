//! Interface to Pascal Pons's Connect Four solver: https://github.com/PascalPons/connect4
//! Opening book: https://github.com/PascalPons/connect4/releases/tag/book

use std::{
    error::Error,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

use crate::c4r::{Move, Pos};

/// Pons scores for playing each column, from the mover's view: positive wins (larger is faster),
/// zero draws, negative loses. None marks a full column.
pub type MoveScores = [Option<i8>; Pos::N_COLS];

/// Solves every move sequence with Pascal Pons's solver in `-a` (analyze) mode, split across
/// `n_procs` solver processes, each with its own read-only `book`. Strict: every response line
/// must echo its request, carry exactly seven scores, and mark exactly the full columns; the
/// response count must equal the request count. Any violation is an error, never a shifted row.
pub fn solve_parallel(
    solver: &str,
    book: &str,
    moves: &[Vec<Move>],
    n_procs: usize,
) -> Result<Vec<MoveScores>, Box<dyn Error + Send + Sync>> {
    if !Path::new(book).is_file() {
        // Pons silently runs without a book when the file is missing, which is far slower.
        return Err(format!("solver book not found: {}", book).into());
    }
    let chunk_len = moves.len().div_ceil(n_procs.max(1)).max(1);
    let chunks: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = moves
            .chunks(chunk_len)
            .map(|chunk| s.spawn(move || solve_chunk(solver, book, chunk)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("solver thread panicked"))
            .collect()
    });
    let mut ret = Vec::with_capacity(moves.len());
    for chunk in chunks {
        ret.extend(chunk?);
    }
    Ok(ret)
}

fn solve_chunk(
    solver: &str,
    book: &str,
    moves: &[Vec<Move>],
) -> Result<Vec<MoveScores>, Box<dyn Error + Send + Sync>> {
    // Pons reads 1-indexed columns.
    let requests: Vec<String> = moves
        .iter()
        .map(|ms| ms.iter().map(|m| (m + 1).to_string()).collect())
        .collect();
    let mut child = Command::new(solver)
        .arg("-b")
        .arg(book)
        .arg("-a")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to start solver {}: {}", solver, e))?;
    let mut stdin = child.stdin.take().ok_or("failed to open solver stdin")?;
    let stdout = child.stdout.take().ok_or("failed to open solver stdout")?;

    let lines = std::thread::scope(|s| {
        // Write from a separate thread so a full stdout pipe cannot deadlock the writer.
        let writer = s.spawn(|| -> std::io::Result<()> {
            for r in &requests {
                writeln!(stdin, "{}", r)?;
            }
            drop(stdin);
            Ok(())
        });
        let lines: std::io::Result<Vec<String>> = BufReader::new(stdout).lines().collect();
        let written = writer.join().expect("solver writer panicked");
        lines.and_then(|l| written.map(|_| l))
    })?;
    let status = child.wait()?;
    if !status.success() {
        return Err(format!("solver exited with {}", status).into());
    }
    if lines.len() != requests.len() {
        return Err(format!(
            "solver returned {} lines for {} positions (it skips positions it rejects)",
            lines.len(),
            requests.len()
        )
        .into());
    }
    requests
        .iter()
        .zip(moves)
        .zip(&lines)
        .enumerate()
        .map(|(i, ((request, ms), line))| {
            parse_scores(request, &Pos::from_moves(ms), line)
                .map_err(|e| format!("solver line {} ({:?}): {}", i, line, e).into())
        })
        .collect()
}

/// Parses one `-a` response line: the echoed request, then seven scores. -1000 marks a full column.
fn parse_scores(request: &str, pos: &Pos, line: &str) -> Result<MoveScores, String> {
    let mut tokens = line.split(' ');
    let echo = tokens.next().unwrap_or_default();
    if echo != request {
        return Err(format!(
            "echo {:?} does not match request {:?}",
            echo, request
        ));
    }
    let raw: Vec<&str> = tokens.collect();
    if raw.len() != Pos::N_COLS {
        return Err(format!(
            "expected {} scores, got {}",
            Pos::N_COLS,
            raw.len()
        ));
    }
    let legal = pos.legal_moves();
    let mut ret = [None; Pos::N_COLS];
    for (col, tok) in raw.iter().enumerate() {
        let score: i16 = tok.parse().map_err(|_| format!("bad score {:?}", tok))?;
        ret[col] = match (score, legal[col]) {
            (-1000, false) => None,
            (s, true) if (-21..=21).contains(&s) => Some(s as i8),
            _ => {
                return Err(format!(
                    "score {} for column {} (legal: {})",
                    score, col, legal[col]
                ))
            }
        };
    }
    Ok(ret)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// The solver built per the README in `<repo>/solver`, or `C4A0_SOLVER` and `C4A0_BOOK`.
    /// Solver-gated tests return early without it.
    fn solver_paths() -> Option<(String, String)> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../solver");
        let solver = std::env::var("C4A0_SOLVER")
            .unwrap_or_else(|_| dir.join("c4solver").to_string_lossy().into_owned());
        let book = std::env::var("C4A0_BOOK")
            .unwrap_or_else(|_| dir.join("7x6.book").to_string_lossy().into_owned());
        (Path::new(&solver).is_file() && Path::new(&book).is_file()).then_some((solver, book))
    }

    fn digits(s: &str) -> Vec<Move> {
        s.chars().map(|c| c.to_digit(10).unwrap() as Move).collect()
    }

    /// Golden rows from `c4solver -a` on this machine, 0-indexed moves.
    #[test]
    fn solve_parallel_matches_golden_rows() {
        let Some((solver, book)) = solver_paths() else {
            eprintln!("skipping: solver not found");
            return;
        };
        let golden: Vec<(&str, MoveScores)> = vec![
            ("", [-2, -1, 0, 1, 0, -1, -2].map(Some)),
            ("3344", [-4, -3, 18, 2, 2, 18, -3].map(Some)),
            (
                "000000",
                [
                    None,
                    Some(-2),
                    Some(0),
                    Some(1),
                    Some(-2),
                    Some(0),
                    Some(-2),
                ],
            ),
            ("010101", [18, -3, -18, -18, -18, -18, -18].map(Some)),
            ("01010", [1, -18, -18, -18, -18, -18, -18].map(Some)),
        ];
        let moves: Vec<Vec<Move>> = golden.iter().map(|(s, _)| digits(s)).collect();
        for n_procs in [1, 2, 8] {
            let got = solve_parallel(&solver, &book, &moves, n_procs).unwrap();
            let want: Vec<MoveScores> = golden.iter().map(|(_, s)| *s).collect();
            assert_eq!(got, want, "n_procs={}", n_procs);
        }
    }

    /// A stand-in solver that answers every line with `body`, a shell snippet reading `$l`.
    fn fake_solver(name: &str, body: &str) -> (String, String) {
        let dir =
            std::env::temp_dir().join(format!("c4a0-fake-solver-{}-{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("solver.sh");
        // Write the script from a child process. If this process held the write fd, a fork by a
        // concurrently running test could inherit it, and exec would fail with ETXTBSY.
        let mut writer = std::process::Command::new("sh")
            .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
            .arg(&script)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        writer
            .stdin
            .take()
            .unwrap()
            .write_all(format!("#!/bin/sh\nwhile IFS= read -r l; do {}; done\n", body).as_bytes())
            .unwrap();
        assert!(writer.wait().unwrap().success());
        let book = dir.join("book");
        std::fs::write(&book, "").unwrap();
        (
            script.to_string_lossy().into_owned(),
            book.to_string_lossy().into_owned(),
        )
    }

    fn fake_solve(name: &str, body: &str, moves: &[&str]) -> Result<Vec<MoveScores>, String> {
        let (solver, book) = fake_solver(name, body);
        let moves: Vec<Vec<Move>> = moves.iter().map(|s| digits(s)).collect();
        solve_parallel(&solver, &book, &moves, 1).map_err(|e| e.to_string())
    }

    #[test]
    fn well_formed_fake_responses_parse() {
        let got = fake_solve("ok", r#"echo "$l 1 2 3 4 5 6 7""#, &["", "3"]).unwrap();
        assert_eq!(got, vec![[1, 2, 3, 4, 5, 6, 7].map(Some); 2]);
    }

    #[test]
    fn skipped_line_is_an_error_not_a_shift() {
        // Pons drops rejected positions from stdout, which would shift later labels by one.
        let body = r#"[ "$l" = "44" ] || echo "$l 1 2 3 4 5 6 7""#;
        let err = fake_solve("skip", body, &["3", "3", "33"]).unwrap_err();
        assert!(err.contains("2 lines for 3 positions"), "{}", err);
    }

    #[test]
    fn mismatched_echo_is_an_error() {
        let err = fake_solve("echo", r#"echo "1 1 2 3 4 5 6 7""#, &["0", "1"]).unwrap_err();
        assert!(err.contains("does not match"), "{}", err);
    }

    #[test]
    fn wrong_score_count_is_an_error() {
        let err = fake_solve("count", r#"echo "$l 1 2 3 4 5 6""#, &["3"]).unwrap_err();
        assert!(err.contains("expected 7 scores"), "{}", err);
    }

    #[test]
    fn full_column_must_be_marked_exactly() {
        let ok = fake_solve("full", r#"echo "$l -1000 0 0 0 0 0 0""#, &["000000"]).unwrap();
        assert_eq!(ok[0][0], None);
        let err = fake_solve("notfull", r#"echo "$l -1000 0 0 0 0 0 0""#, &["0"]).unwrap_err();
        assert!(err.contains("column 0"), "{}", err);
    }

    #[test]
    fn missing_book_is_an_error() {
        let (solver, _) = fake_solver("nobook", "true");
        let err = solve_parallel(&solver, "/nonexistent/book", &[vec![]], 1).unwrap_err();
        assert!(err.to_string().contains("book not found"));
    }
}
