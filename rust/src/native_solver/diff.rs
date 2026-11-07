//! The differential harness: sample [Case]s reproducibly, score each with both solvers, and
//! adjudicate every disagreement with evidence neither solver supplies alone.
//!
//! A [Case] is a move sequence with the [Stratum] that produced it. A [Verdict] is what the
//! adjudication concluded about one case. Evidence, in decreasing strength:
//! - exhaustive minimax ([super::brute]) for positions at [BRUTE_FROM_PLY] or later,
//! - each solver's own consistency: a column's score must equal minus the best score of the
//!   position it leads to, which the harness asks that same solver for,
//! - an alternative native configuration: plain column ordering, no book, a separate table.

use std::collections::HashSet;
use std::fmt::Write as _;

use super::board::{Board, WIDTH};
use super::book::Book;
use super::brute;
use super::parallel::analyze_batch;
use super::search::{win_now_score, Order, Searcher};
use super::tt::Tt;
use crate::c4r::{Move, Pos};
use crate::solver::MoveScores;

/// Positions at this ply or later are small enough for [brute::analyze]: in release builds about
/// 0.4 s each at ply 22 and 12 ms at ply 26 (5-10 seeded positions per ply, one run).
pub const BRUTE_FROM_PLY: usize = 22;

/// How a case was produced. Each targets a different way a solver could be wrong.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Stratum {
    /// Every canonical position at plies 0..=[EXHAUSTIVE_PLIES].
    Exhaustive,
    /// A uniform random walk to a target ply, rejecting moves that end the game.
    Walk,
    /// A walk that prefers one column until it fills, so the position has a full column.
    FullColumn,
    /// A walk stopped at the first position where the side to move can win or must block.
    NearTerminal,
    /// A row of a committed suite, which also carries the Pons scores recorded when it was built.
    Suite(&'static str),
}

impl Stratum {
    pub fn name(&self) -> String {
        match self {
            Stratum::Exhaustive => "exhaustive".into(),
            Stratum::Walk => "walk".into(),
            Stratum::FullColumn => "full_column".into(),
            Stratum::NearTerminal => "near_terminal".into(),
            Stratum::Suite(s) => format!("suite_{}", s),
        }
    }
}

pub const EXHAUSTIVE_PLIES: usize = 4;

#[derive(Clone, Debug)]
pub struct Case {
    pub moves: Vec<Move>,
    pub stratum: Stratum,
    /// Scores stored with a suite row.
    pub committed: Option<MoveScores>,
}

impl Case {
    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    pub fn board(&self) -> Board {
        Board::from_moves(&self.moves).expect("cases are legal")
    }
}

/// SplitMix64, so a seed means the same cases on every platform and toolchain.
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

/// A walk to `ply` choosing among quiet moves; `prefer` biases toward one column while it is
/// open. None if the walk dead-ends.
fn walk(ply: usize, prefer: Option<Move>, rng: &mut Rng) -> Option<Vec<Move>> {
    let mut b = Board::default();
    let mut moves = vec![];
    while moves.len() < ply {
        let quiet = quiet_moves(&b);
        if quiet.is_empty() {
            return None;
        }
        let m = match prefer {
            Some(p) if quiet.contains(&p) && rng.below(3) != 0 => p,
            _ => quiet[rng.below(quiet.len())],
        };
        b = b.play(m);
        moves.push(m);
    }
    Some(moves)
}

/// A walk that stops at the first position from `min_ply` on where the side to move can win
/// at once or must block an immediate threat.
fn near_terminal_walk(min_ply: usize, rng: &mut Rng) -> Option<Vec<Move>> {
    let mut b = Board::default();
    let mut moves = vec![];
    while moves.len() < 41 {
        let tense = b.can_win_now() || b.opponent_threats() & b.playable() != 0;
        if moves.len() >= min_ply && tense {
            return Some(moves);
        }
        let quiet = quiet_moves(&b);
        if quiet.is_empty() {
            return None;
        }
        let m = quiet[rng.below(quiet.len())];
        b = b.play(m);
        moves.push(m);
    }
    None
}

/// Every canonical non-terminal position at plies 0..=[EXHAUSTIVE_PLIES], with a witness path.
fn exhaustive() -> Vec<Vec<Move>> {
    let mut seen = HashSet::new();
    let mut level = vec![vec![]];
    let mut ret = vec![];
    for _ in 0..=EXHAUSTIVE_PLIES {
        let mut next = vec![];
        for moves in level {
            let b = Board::from_moves(&moves).unwrap();
            if !seen.insert(b.canonical_key()) {
                continue;
            }
            for m in quiet_moves(&b) {
                let mut child = moves.clone();
                child.push(m);
                next.push(child);
            }
            ret.push(moves);
        }
        level = next;
    }
    ret
}

/// Samples `per_ply` walks at every ply 0..=41 plus the targeted strata, from `seed`. Distinct
/// boards only (a mirror is a distinct board: it checks mirror consistency for free).
pub fn sample(per_ply: usize, seed: u64) -> Vec<Case> {
    let mut seen = HashSet::new();
    let mut ret = vec![];
    let mut push = |moves: Vec<Move>, stratum: Stratum, ret: &mut Vec<Case>| {
        if seen.insert(Board::from_moves(&moves).unwrap().key()) {
            ret.push(Case {
                moves,
                stratum,
                committed: None,
            });
        }
    };
    for moves in exhaustive() {
        push(moves, Stratum::Exhaustive, &mut ret);
    }
    for ply in 0..=41 {
        let mut rng = Rng::new(seed, ply as u64);
        for _ in 0..per_ply * 4 {
            if let Some(m) = walk(ply, None, &mut rng) {
                push(m, Stratum::Walk, &mut ret);
            }
            if ret
                .iter()
                .filter(|c| c.ply() == ply && c.stratum == Stratum::Walk)
                .count()
                >= per_ply
            {
                break;
            }
        }
        // Plies 6+ can hold a full column. A quarter of the walk quota, filtered to those that do.
        if ply >= 6 {
            let mut rng = Rng::new(seed, 1000 + ply as u64);
            let mut found = 0;
            for _ in 0..per_ply * 8 {
                if found >= per_ply.div_ceil(4) {
                    break;
                }
                let prefer = Some(rng.below(WIDTH));
                if let Some(m) = walk(ply, prefer, &mut rng) {
                    let b = Board::from_moves(&m).unwrap();
                    if (0..WIDTH).any(|c| !b.can_play(c)) {
                        found += 1;
                        push(m, Stratum::FullColumn, &mut ret);
                    }
                }
            }
        }
    }
    let mut rng = Rng::new(seed, 2000);
    for i in 0..per_ply * 10 {
        if let Some(m) = near_terminal_walk(i % 42, &mut rng) {
            push(m, Stratum::NearTerminal, &mut ret);
        }
    }
    ret
}

/// Rows of a committed suite file's text: `moves<TAB>scores`, 0-indexed moves, `x` for full.
pub fn parse_suite(name: &'static str, text: &str) -> Result<Vec<Case>, String> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("moves") && !l.is_empty())
        .map(|l| {
            let (moves, scores) = l.split_once('\t').ok_or(format!("bad row {:?}", l))?;
            let moves: Vec<Move> = moves
                .chars()
                .map(|c| {
                    c.to_digit(10)
                        .map(|d| d as Move)
                        .ok_or(format!("bad move in {:?}", l))
                })
                .collect::<Result<_, _>>()?;
            let scores: Vec<Option<i8>> = scores
                .split(',')
                .map(|s| match s {
                    "x" => Ok(None),
                    s => s
                        .parse()
                        .map(Some)
                        .map_err(|_| format!("bad score in {:?}", l)),
                })
                .collect::<Result<_, _>>()?;
            let committed: MoveScores = scores
                .try_into()
                .map_err(|_| format!("not 7 scores: {:?}", l))?;
            Board::from_moves(&moves).ok_or(format!("illegal moves {:?}", l))?;
            Ok(Case {
                moves,
                stratum: Stratum::Suite(name),
                committed: Some(committed),
            })
        })
        .collect()
}

/// What the evidence says about one case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Agree,
    /// Pons disagrees with an independent check that the native solver passes.
    PonsBug,
    /// The native solver disagrees with an independent check that Pons passes.
    NativeBug,
    /// The scores disagree only in how they are written (a full column, a range edge).
    Protocol,
    /// The evidence does not decide. The bar is zero of these.
    Unexplained,
}

impl Verdict {
    pub fn name(&self) -> &'static str {
        match self {
            Verdict::Agree => "agree",
            Verdict::PonsBug => "pons_bug",
            Verdict::NativeBug => "native_bug",
            Verdict::Protocol => "protocol",
            Verdict::Unexplained => "unexplained",
        }
    }
}

/// One solver's row checked against its own rows for each child position: a non-winning column
/// must score minus the child's best column. Lists the inconsistent columns.
pub fn inconsistent_columns(
    b: &Board,
    row: &MoveScores,
    child_rows: &[Option<MoveScores>; WIDTH],
) -> Vec<usize> {
    (0..WIDTH)
        .filter(|&c| {
            let want = if !b.can_play(c) {
                None
            } else if b.wins_with(c) {
                Some(win_now_score(b.ply()))
            } else if b.play(c).ply() == 42 {
                Some(0)
            } else {
                child_rows[c]
                    .as_ref()
                    .and_then(|r| r.iter().flatten().max().map(|&m| -m))
            };
            want != row[c]
        })
        .collect()
}

/// The move sequences a consistency check needs scored: each quiet child short of a full board.
pub fn child_moves(case: &Case) -> [Option<Vec<Move>>; WIDTH] {
    let b = case.board();
    std::array::from_fn(|c| {
        (b.can_play(c) && !b.wins_with(c) && b.ply() < 41).then(|| {
            let mut m = case.moves.clone();
            m.push(c);
            m
        })
    })
}

/// Every piece of evidence the harness gathered for one disagreeing case.
#[derive(Clone, Debug)]
pub struct Evidence {
    pub brute: Option<MoveScores>,
    pub alt_native: MoveScores,
    pub pons_inconsistent: Vec<usize>,
    pub native_inconsistent: Vec<usize>,
}

/// Exhaustive scores when the case is late enough, else None.
pub fn brute_scores(case: &Case) -> Option<MoveScores> {
    (case.ply() >= BRUTE_FROM_PLY).then(|| brute::analyze(&Pos::from_moves(&case.moves)))
}

/// The native row under the alternative configuration: column ordering, no book, own table.
pub fn alt_native(b: &Board, tt_bytes: usize) -> MoveScores {
    let tt = Tt::with_bytes(tt_bytes);
    Searcher::new(&tt, None)
        .with_order(Order::Columns)
        .analyze(b)
}

/// Classifies a case from its two rows and the evidence. Brute force is decisive. Below its
/// ply, one side must fail while the other passes, or the case stays unexplained. Native passes
/// when it is self-consistent and its alternative configuration reproduces its row; Pons passes
/// when it is self-consistent.
///
/// When the rows agree, evidence exists only because a check failed: brute force rejecting the
/// shared row leaves it unexplained, and Pons failing its own consistency while native passes
/// puts the bug in a Pons child row.
pub fn classify(native: &MoveScores, pons: &MoveScores, ev: Option<&Evidence>) -> Verdict {
    if native == pons {
        return match ev {
            None => Verdict::Agree,
            Some(ev) if ev.brute.is_some_and(|b| &b != native) => Verdict::Unexplained,
            Some(ev) if !ev.pons_inconsistent.is_empty() && ev.native_inconsistent.is_empty() => {
                Verdict::PonsBug
            }
            Some(ev) if ev.pons_inconsistent.is_empty() && ev.native_inconsistent.is_empty() => {
                Verdict::Agree
            }
            Some(_) => Verdict::Unexplained,
        };
    }
    if (0..WIDTH).any(|c| native[c].is_none() != pons[c].is_none()) {
        return Verdict::Protocol;
    }
    let Some(ev) = ev else {
        return Verdict::Unexplained;
    };
    if let Some(truth) = &ev.brute {
        return match (truth == native, truth == pons) {
            (true, false) => Verdict::PonsBug,
            (false, true) => Verdict::NativeBug,
            _ => Verdict::Unexplained,
        };
    }
    let native_ok = ev.native_inconsistent.is_empty() && &ev.alt_native == native;
    let pons_ok = ev.pons_inconsistent.is_empty();
    match (native_ok, pons_ok) {
        (true, false) => Verdict::PonsBug,
        (false, true) => Verdict::NativeBug,
        _ => Verdict::Unexplained,
    }
}

/// Native rows for every case, analyzed in one parallel batch with a shared table.
pub fn native_rows(
    cases: &[Case],
    tt: &Tt,
    book: Option<&Book>,
    threads: usize,
) -> Vec<MoveScores> {
    let boards: Vec<Board> = cases.iter().map(Case::board).collect();
    analyze_batch(&boards, tt, book, threads, &|_| {})
}

/// A JSON string literal for the ASCII text the report writes (digits, names).
fn js(s: &str) -> String {
    format!("\"{}\"", s)
}

fn js_row(row: &MoveScores) -> String {
    let cols: Vec<String> = row
        .iter()
        .map(|c| c.map_or("null".into(), |v| v.to_string()))
        .collect();
    format!("[{}]", cols.join(","))
}

/// One report line for a case: its identity, both rows, the verdict and the evidence.
pub fn diff_record(
    case: &Case,
    native: &MoveScores,
    pons: &MoveScores,
    ev: Option<&Evidence>,
    verdict: Verdict,
) -> String {
    let moves: String = case.moves.iter().map(|m| m.to_string()).collect();
    let pons_line: String = case.moves.iter().map(|m| (m + 1).to_string()).collect();
    let mut s = String::new();
    write!(
        s,
        "{{\"moves\":{},\"pons_input\":{},\"ply\":{},\"stratum\":{},\"native\":{},\"pons\":{},\"verdict\":{}",
        js(&moves),
        js(&pons_line),
        case.ply(),
        js(&case.stratum.name()),
        js_row(native),
        js_row(pons),
        js(verdict.name())
    )
    .unwrap();
    if let Some(c) = &case.committed {
        write!(s, ",\"committed\":{}", js_row(c)).unwrap();
    }
    if let Some(ev) = ev {
        let cols = |v: &[usize]| {
            format!(
                "[{}]",
                v.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        write!(
            s,
            ",\"brute\":{},\"alt_native\":{},\"pons_inconsistent\":{},\"native_inconsistent\":{}",
            ev.brute.as_ref().map_or("null".into(), js_row),
            js_row(&ev.alt_native),
            cols(&ev.pons_inconsistent),
            cols(&ev.native_inconsistent)
        )
        .unwrap();
    }
    s.push('}');
    s
}

/// What a harness run needs: the Pons oracle, the native book, the sample and the resources.
pub struct Config {
    pub pons: String,
    pub pons_book: String,
    pub book: Option<Book>,
    pub per_ply: usize,
    pub seed: u64,
    pub threads: usize,
    pub tt_bytes: usize,
    /// Committed suites as (name, file text).
    pub suites: Vec<(&'static str, String)>,
    pub out: std::path::PathBuf,
}

/// Ply buckets as the evaluation reports them: 0-5, 6-11, ..., 36-41.
pub const BUCKET: usize = 6;

#[derive(Default, Clone)]
struct Tally {
    compared: usize,
    agree: usize,
    pons_bug: usize,
    native_bug: usize,
    protocol: usize,
    unexplained: usize,
    brute_checked: usize,
    brute_vs_native: usize,
    brute_vs_pons: usize,
    pons_inconsistent: usize,
    committed_vs_pons: usize,
}

impl Tally {
    fn json(&self) -> String {
        format!(
            "{{\"compared\":{},\"agree\":{},\"pons_bug\":{},\"native_bug\":{},\"protocol\":{},\"unexplained\":{},\"brute_checked\":{},\"brute_vs_native\":{},\"brute_vs_pons\":{},\"pons_inconsistent\":{},\"committed_vs_pons\":{}}}",
            self.compared, self.agree, self.pons_bug, self.native_bug, self.protocol, self.unexplained,
            self.brute_checked, self.brute_vs_native, self.brute_vs_pons, self.pons_inconsistent,
            self.committed_vs_pons
        )
    }
}

/// Runs the harness and writes `diffs.jsonl` (one record per disagreement or failed check),
/// `cases.tsv` (every case with both rows) and `summary.json` into `cfg.out`. Returns the
/// number of disagreements the evidence did not explain.
pub fn run(cfg: &Config) -> Result<usize, String> {
    use std::time::Instant;
    let t0 = Instant::now();
    let log = |m: String| eprintln!("[{:>7.1}s] {}", t0.elapsed().as_secs_f64(), m);

    let mut cases = sample(cfg.per_ply, cfg.seed);
    let sampled = cases.len();
    for (name, text) in &cfg.suites {
        cases.extend(parse_suite(name, text)?);
    }
    log(format!(
        "{} cases ({} sampled, {} suite rows)",
        cases.len(),
        sampled,
        cases.len() - sampled
    ));

    let moves: Vec<Vec<Move>> = cases.iter().map(|c| c.moves.clone()).collect();
    let t = Instant::now();
    let pons = crate::solver::solve_parallel(&cfg.pons, &cfg.pons_book, &moves, cfg.threads)
        .map_err(|e| e.to_string())?;
    let pons_secs = t.elapsed().as_secs_f64();
    log(format!("pons: {} rows in {:.1}s", pons.len(), pons_secs));

    let tt = Tt::with_bytes(cfg.tt_bytes);
    let t = Instant::now();
    let native = native_rows(&cases, &tt, cfg.book.as_ref(), cfg.threads);
    let native_secs = t.elapsed().as_secs_f64();
    log(format!(
        "native: {} rows in {:.1}s",
        native.len(),
        native_secs
    ));

    // Pons's own consistency on every case: score every quiet child with Pons too.
    let children: Vec<[Option<Vec<Move>>; WIDTH]> = cases.iter().map(child_moves).collect();
    let flat: Vec<Vec<Move>> = children.iter().flatten().flatten().cloned().collect();
    let t = Instant::now();
    let flat_rows = crate::solver::solve_parallel(&cfg.pons, &cfg.pons_book, &flat, cfg.threads)
        .map_err(|e| e.to_string())?;
    log(format!(
        "pons children: {} rows in {:.1}s",
        flat_rows.len(),
        t.elapsed().as_secs_f64()
    ));
    let mut flat_rows = flat_rows.into_iter();
    let pons_children: Vec<[Option<MoveScores>; WIDTH]> = children
        .iter()
        .map(|cs| std::array::from_fn(|c| cs[c].as_ref().map(|_| flat_rows.next().unwrap())))
        .collect();
    let pons_bad: Vec<Vec<usize>> = cases
        .iter()
        .zip(&pons)
        .zip(&pons_children)
        .map(|((c, row), kids)| inconsistent_columns(&c.board(), row, kids))
        .collect();

    // Exhaustive minimax on every late case, agreeing or not.
    let late: Vec<usize> = (0..cases.len())
        .filter(|&i| cases[i].ply() >= BRUTE_FROM_PLY)
        .collect();
    let t = Instant::now();
    let brute_rows: Vec<Option<MoveScores>> = super::parallel::par_map(
        late.len(),
        cfg.threads,
        &|_| {},
        || (),
        |_, i| brute_scores(&cases[late[i]]),
    );
    log(format!(
        "brute force: {} late cases in {:.1}s",
        late.len(),
        t.elapsed().as_secs_f64()
    ));
    let mut brute: Vec<Option<MoveScores>> = vec![None; cases.len()];
    for (i, row) in late.iter().zip(brute_rows) {
        brute[*i] = row;
    }

    // Every case where the solvers disagree, or either fails an independent check, gets the full
    // evidence. That includes Pons rows that agree with native but fail Pons's own consistency.
    let suspect: Vec<usize> = (0..cases.len())
        .filter(|&i| {
            native[i] != pons[i]
                || !pons_bad[i].is_empty()
                || brute[i].is_some_and(|b| b != native[i] || b != pons[i])
                || cases[i].committed.is_some_and(|c| c != pons[i])
        })
        .collect();
    log(format!("{} suspect cases", suspect.len()));
    let evidence: Vec<Evidence> = super::parallel::par_map(
        suspect.len(),
        cfg.threads,
        &|_| {},
        || Tt::with_bytes(64 << 20),
        |tt, k| {
            let i = suspect[k];
            let b = cases[i].board();
            let mut s = Searcher::new(tt, cfg.book.as_ref());
            let kids: [Option<MoveScores>; WIDTH] =
                std::array::from_fn(|c| children[i][c].as_ref().map(|_| s.analyze(&b.play(c))));
            Evidence {
                brute: brute[i],
                alt_native: alt_native(&b, 256 << 20),
                pons_inconsistent: pons_bad[i].clone(),
                native_inconsistent: inconsistent_columns(&b, &native[i], &kids),
            }
        },
    );
    let evidence_of = |i: usize| suspect.binary_search(&i).ok().map(|k| &evidence[k]);

    std::fs::create_dir_all(&cfg.out).map_err(|e| e.to_string())?;
    let mut diffs = String::new();
    let mut tsv = String::from("moves\tply\tstratum\tnative\tpons\tverdict\n");
    let mut by_bucket = vec![Tally::default(); 42usize.div_ceil(BUCKET)];
    let mut by_stratum: std::collections::BTreeMap<String, Tally> = Default::default();
    let mut unexplained = 0;
    for (i, case) in cases.iter().enumerate() {
        let ev = evidence_of(i);
        let verdict = classify(&native[i], &pons[i], ev);
        unexplained += (verdict == Verdict::Unexplained) as usize;
        for t in [
            &mut by_bucket[case.ply() / BUCKET],
            by_stratum.entry(case.stratum.name()).or_default(),
        ] {
            t.compared += 1;
            match verdict {
                Verdict::Agree => t.agree += 1,
                Verdict::PonsBug => t.pons_bug += 1,
                Verdict::NativeBug => t.native_bug += 1,
                Verdict::Protocol => t.protocol += 1,
                Verdict::Unexplained => t.unexplained += 1,
            }
            if let Some(b) = brute[i] {
                t.brute_checked += 1;
                t.brute_vs_native += (b != native[i]) as usize;
                t.brute_vs_pons += (b != pons[i]) as usize;
            }
            t.pons_inconsistent += !pons_bad[i].is_empty() as usize;
            t.committed_vs_pons += case.committed.is_some_and(|c| c != pons[i]) as usize;
        }
        if ev.is_some() {
            diffs.push_str(&diff_record(case, &native[i], &pons[i], ev, verdict));
            diffs.push('\n');
        }
        let moves: String = case.moves.iter().map(|m| m.to_string()).collect();
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}",
            moves,
            case.ply(),
            case.stratum.name(),
            js_row(&native[i]),
            js_row(&pons[i]),
            verdict.name()
        )
        .unwrap();
    }
    let buckets: Vec<String> = by_bucket
        .iter()
        .enumerate()
        .map(|(b, t)| {
            format!(
                "\"{}-{}\":{}",
                b * BUCKET,
                b * BUCKET + BUCKET - 1,
                t.json()
            )
        })
        .collect();
    let strata: Vec<String> = by_stratum
        .iter()
        .map(|(k, t)| format!("{}:{}", js(k), t.json()))
        .collect();
    let total = by_bucket.iter().fold(Tally::default(), |mut a, t| {
        a.compared += t.compared;
        a.agree += t.agree;
        a.pons_bug += t.pons_bug;
        a.native_bug += t.native_bug;
        a.protocol += t.protocol;
        a.unexplained += t.unexplained;
        a.brute_checked += t.brute_checked;
        a.brute_vs_native += t.brute_vs_native;
        a.brute_vs_pons += t.brute_vs_pons;
        a.pons_inconsistent += t.pons_inconsistent;
        a.committed_vs_pons += t.committed_vs_pons;
        a
    });
    let summary = format!(
        "{{\"seed\":{},\"per_ply\":{},\"threads\":{},\"tt_bytes\":{},\"native_book_ply\":{},\"pons_secs\":{:.2},\"native_secs\":{:.2},\"peak_rss_mib\":{},\"total\":{},\"by_bucket\":{{{}}},\"by_stratum\":{{{}}}}}\n",
        cfg.seed,
        cfg.per_ply,
        cfg.threads,
        cfg.tt_bytes,
        cfg.book.as_ref().map_or("null".into(), |b| b.ply().to_string()),
        pons_secs,
        native_secs,
        super::bench::peak_rss_kib().map_or("null".into(), |k| (k >> 10).to_string()),
        total.json(),
        buckets.join(","),
        strata.join(",")
    );
    let write = |name: &str, body: &str| {
        std::fs::write(cfg.out.join(name), body).map_err(|e| format!("{}: {}", name, e))
    };
    write("diffs.jsonl", &diffs)?;
    write("cases.tsv", &tsv)?;
    write("summary.json", &summary)?;
    log(format!(
        "{} compared, {} agree, {} pons_bug, {} native_bug, {} protocol, {} unexplained; brute checked {} (native wrong {}, pons wrong {}); pons self-inconsistent {}; suite rows where live pons != committed {}",
        total.compared, total.agree, total.pons_bug, total.native_bug, total.protocol, total.unexplained,
        total.brute_checked, total.brute_vs_native, total.brute_vs_pons, total.pons_inconsistent, total.committed_vs_pons
    ));
    Ok(unexplained)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_reproducible_and_covers_every_stratum_and_ply() {
        let a = sample(6, 11);
        let b = sample(6, 11);
        assert_eq!(
            a.iter().map(|c| &c.moves).collect::<Vec<_>>(),
            b.iter().map(|c| &c.moves).collect::<Vec<_>>()
        );
        assert_ne!(
            sample(6, 12)
                .iter()
                .map(|c| c.moves.clone())
                .collect::<Vec<_>>(),
            a.iter().map(|c| c.moves.clone()).collect::<Vec<_>>()
        );
        // Walks at plies 0..=EXHAUSTIVE_PLIES duplicate exhaustive cases and are dropped.
        for ply in 0..=41 {
            let want = if ply <= EXHAUSTIVE_PLIES {
                Stratum::Exhaustive
            } else {
                Stratum::Walk
            };
            let found = a.iter().any(|c| c.ply() == ply && c.stratum == want);
            assert!(found, "ply {}", ply);
        }
        for c in &a {
            let b = c.board();
            assert!(
                !b.opponent_has_won() && b.ply() < 42,
                "terminal case {:?}",
                c.moves
            );
            match c.stratum {
                Stratum::FullColumn => assert!((0..WIDTH).any(|col| !b.can_play(col))),
                Stratum::NearTerminal => {
                    assert!(b.can_win_now() || b.opponent_threats() & b.playable() != 0)
                }
                _ => {}
            }
        }
        for s in [
            Stratum::Exhaustive,
            Stratum::FullColumn,
            Stratum::NearTerminal,
        ] {
            assert!(a.iter().filter(|c| c.stratum == s).count() >= 6, "{:?}", s);
        }
        // Exactly one case per canonical position at plies 0..=EXHAUSTIVE_PLIES; the book test
        // pins those level sizes against an enumeration over Pos.
        let levels: usize = (0..=EXHAUSTIVE_PLIES as u8)
            .map(|p| crate::native_solver::book::positions_at(p).len())
            .sum();
        let exhaustive_cases = a
            .iter()
            .filter(|c| c.stratum == Stratum::Exhaustive)
            .count();
        assert_eq!(exhaustive_cases, levels);
        let keys: HashSet<u64> = a
            .iter()
            .filter(|c| c.stratum == Stratum::Exhaustive)
            .map(|c| c.board().canonical_key())
            .collect();
        assert_eq!(
            keys.len(),
            a.iter()
                .filter(|c| c.stratum == Stratum::Exhaustive)
                .count()
        );
    }

    #[test]
    fn brute_force_decides_late_disagreements() {
        let truth = [Some(1), Some(-2), None, Some(0), Some(0), Some(0), Some(0)];
        let mut wrong = truth;
        wrong[1] = Some(-1);
        let ev = |brute| Evidence {
            brute: Some(brute),
            alt_native: truth,
            pons_inconsistent: vec![],
            native_inconsistent: vec![],
        };
        assert_eq!(classify(&truth, &truth, None), Verdict::Agree);
        // Both rows wrong by brute force: nothing explains it.
        assert_eq!(
            classify(&wrong, &wrong, Some(&ev(truth))),
            Verdict::Unexplained
        );
        assert_eq!(classify(&truth, &wrong, Some(&ev(truth))), Verdict::PonsBug);
        assert_eq!(
            classify(&wrong, &truth, Some(&ev(truth))),
            Verdict::NativeBug
        );
        let mut other = truth;
        other[0] = Some(2);
        assert_eq!(
            classify(&truth, &wrong, Some(&ev(other))),
            Verdict::Unexplained
        );
        assert_eq!(classify(&truth, &wrong, None), Verdict::Unexplained);
    }

    #[test]
    fn early_disagreements_need_a_failed_consistency_check() {
        let native = [Some(1); WIDTH];
        let mut pons = native;
        pons[3] = Some(2);
        let ev = |pons_bad: Vec<usize>, native_bad: Vec<usize>, alt| Evidence {
            brute: None,
            alt_native: alt,
            pons_inconsistent: pons_bad,
            native_inconsistent: native_bad,
        };
        assert_eq!(
            classify(&native, &pons, Some(&ev(vec![3], vec![], native))),
            Verdict::PonsBug
        );
        assert_eq!(
            classify(&native, &pons, Some(&ev(vec![], vec![3], pons))),
            Verdict::NativeBug
        );
        // The native row contradicts its own child rows; the alternative configuration shares the
        // column scoring, so it may repeat the bug.
        assert_eq!(
            classify(&native, &pons, Some(&ev(vec![], vec![3], native))),
            Verdict::NativeBug
        );
        // The native configurations disagree with each other while Pons is consistent.
        assert_eq!(
            classify(&native, &pons, Some(&ev(vec![], vec![], pons))),
            Verdict::NativeBug
        );
        // Both self-consistent: nothing independent decides it.
        assert_eq!(
            classify(&native, &pons, Some(&ev(vec![], vec![], native))),
            Verdict::Unexplained
        );
        // Agreeing rows where only Pons contradicts its own child rows: the bug is in a child row.
        assert_eq!(
            classify(&native, &native, Some(&ev(vec![3], vec![], native))),
            Verdict::PonsBug
        );
        assert_eq!(
            classify(&native, &native, Some(&ev(vec![3], vec![3], native))),
            Verdict::Unexplained
        );
        // The alternative configuration disagrees with the native row: not a clean Pons bug.
        assert_eq!(
            classify(&native, &pons, Some(&ev(vec![3], vec![], pons))),
            Verdict::Unexplained
        );
    }

    #[test]
    fn consistency_flags_a_column_that_misreads_its_child() {
        let rows = crate::native_solver::suite_rows("dev");
        let r = rows.iter().find(|r| r.moves.len() == 26).unwrap();
        let b = Board::from_moves(&r.moves).unwrap();
        let tt = Tt::with_bytes(1 << 20);
        let mut s = Searcher::new(&tt, None);
        let row = s.analyze(&b);
        let child_rows: [Option<MoveScores>; WIDTH] = std::array::from_fn(|c| {
            (b.can_play(c) && !b.wins_with(c)).then(|| s.analyze(&b.play(c)))
        });
        assert!(inconsistent_columns(&b, &row, &child_rows).is_empty());
        let mut bad = row;
        bad[0] = bad[0].map(|v| v + 1);
        assert_eq!(inconsistent_columns(&b, &bad, &child_rows), vec![0]);
    }

    #[test]
    fn committed_suites_parse() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/c4a0/suite/v1/");
        for name in ["dev", "test"] {
            let text = std::fs::read_to_string(format!("{}{}.tsv", dir, name)).unwrap();
            let rows = parse_suite(if name == "dev" { "dev" } else { "test" }, &text).unwrap();
            assert!(rows.len() > 2000, "{}", name);
        }
        assert!(parse_suite("dev", "0\t1,2,3\n").is_err());
    }
}
