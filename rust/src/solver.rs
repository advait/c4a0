use std::{
    collections::{HashMap, HashSet},
    error::Error,
    io::Write,
    ops,
    process::{Command, Stdio},
};

use rocksdb::{Options, DB};
use serde::{Deserialize, Serialize};

use crate::{c4r::Pos, types::Policy, utils::OrdF32};

/// A caching wrapper around [Solver] that caches solutions to positions in [rocksdb].
pub struct CachingSolver {
    solver: Solver,
    db: DB,
}

impl CachingSolver {
    /// path_to_solver: Path to the solver binary.
    /// path_to_book: Path to the solver's book file.
    /// path_to_solution_db: Path to the rocksdb database to cache solutions.
    pub fn new(path_to_solver: String, path_to_book: String, path_to_solution_db: String) -> Self {
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = DB::open(&options, path_to_solution_db).expect("failed to open rocksdb");

        Self {
            solver: Solver::new(path_to_solver, path_to_book),
            db,
        }
    }

    /// Scores the given policies for the given positions based on the solutions from the solver.
    pub fn score_policies(
        &self,
        pos_and_policy: Vec<(Pos, Policy)>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (pos, policy): (Vec<Pos>, Vec<Policy>) = pos_and_policy.into_iter().unzip();
        let solutions = self.solve(pos)?;
        let ret = solutions
            .into_iter()
            .zip(policy.into_iter())
            .map(|(sol, pol)| sol.score_policy(&pol))
            .collect();
        Ok(ret)
    }

    /// Solves the given position, resorting to cached positions if possible, relying on
    /// [Self::solver] to solve missing positions, and then caches resulting solutions.
    fn solve(&self, pos: Vec<Pos>) -> Result<Vec<Solution>, Box<dyn Error>> {
        let missing_pos = pos
            .iter()
            .filter(|p| self.get(&p).is_none())
            .cloned()
            .collect::<HashSet<_>>() // Remove duplicates
            .into_iter()
            .collect::<Vec<_>>();

        log::debug!("Solving {} missing positions", missing_pos.len());
        for chunk in missing_pos.chunks(100) {
            log::debug!("Chunk size {}", chunk.len());
            let chunk_solutions = self.solver.solve(chunk)?;
            for (pos, solution) in chunk.into_iter().zip(chunk_solutions.into_iter()) {
                self.put(&pos, &solution);
            }
        }
        self.db.flush()?;
        log::debug!("Finished solving positions");

        let ret = pos.into_iter().map(|pos| self.get(&pos).unwrap()).collect();
        Ok(ret)
    }

    fn get(&self, pos: &Pos) -> Option<Solution> {
        self.db
            .get(serde_cbor::to_vec(pos).expect("failed to serialize"))
            .expect("failed to get from db")
            .map(|bytes| serde_cbor::from_slice(&bytes).expect("failed to deserialize"))
    }

    fn put(&self, pos: &Pos, solution: &Solution) {
        self.db
            .put(
                serde_cbor::to_vec(pos).expect("failed to serialize"),
                serde_cbor::to_vec(solution).expect("failed to serialize"),
            )
            .expect("failed to put to db");
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct SolutionCache(HashMap<Pos, Solution>);

/// Interface to PascalPons's connect4 solver: https://github.com/PascalPons/connect4
/// Runs the solver in a subprocess, communicating via stdin/out.
struct Solver {
    path_to_solver: String,
    path_to_book: String,
}

impl Solver {
    /// Creates a new solver with the given path to the solver binary and book file.
    /// Book files available here: https://github.com/PascalPons/connect4/releases/tag/book
    fn new(path_to_solver: String, path_to_book: String) -> Self {
        Self {
            path_to_solver,
            path_to_book,
        }
    }

    /// Calls the solver to solve the given positions.
    fn solve(&self, pos: &[Pos]) -> Result<Vec<Solution>, Box<dyn Error>> {
        let mut cmd = Command::new(self.path_to_solver.clone())
            .arg("-b")
            .arg(self.path_to_book.clone())
            .arg("-a")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdin_bytes = pos
            .iter()
            .map(|p| {
                p.to_moves()
                    .iter()
                    .map(|m| (m + 1).to_string())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";

        let mut stdin = cmd.stdin.take().ok_or("failed to open stdin")?;
        stdin.write_all(&stdin_bytes.into_bytes())?;
        drop(stdin); // Close stdin to signal we're done writing

        let output = cmd.wait_with_output()?;
        let stdout_str = String::from_utf8(output.stdout)?;

        let ret = stdout_str
            .split("\n")
            .filter(|l| l.len() > 1)
            .map(|l| {
                let mut nums: Vec<_> = l.trim().split(" ").collect();
                if nums.len() == Pos::N_COLS + 1 {
                    // Remove the first number which is the move sequence.
                    // If there is no first move number, we're playing the starting position.
                    nums.remove(0);
                }

                nums.iter()
                    .map(|num| {
                        num.parse::<i16>()
                            .expect(format!("failed to parse stdout: '{}'", num).as_str())
                    })
                    .collect()
            })
            .collect();
        Ok(ret)
    }
}

/// Solution from the solver. Each index represents a column. Positive values indicate that the
/// current player will win if they play in that column, negative values indicate that the current
/// player will lose if they play in that column. The magnitude of the value indicates the number
/// of tokens remaining at the end of the game for the current player.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Solution([i16; Pos::N_COLS]);

impl FromIterator<i16> for Solution {
    fn from_iter<T: IntoIterator<Item = i16>>(iter: T) -> Self {
        let arr: [i16; Pos::N_COLS] = iter.into_iter().collect::<Vec<_>>().try_into().expect("");
        Solution(arr)
    }
}

impl From<[i16; Pos::N_COLS]> for Solution {
    fn from(arr: [i16; Pos::N_COLS]) -> Self {
        Solution(arr)
    }
}

impl ops::Neg for Solution {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.0.iter().map(|&x| -x).collect()
    }
}

impl Solution {
    /// Given a [Pos] and a [Policy], score the policy relative to this solution. Only considers the
    /// highest probability move in the policy.
    /// Selecting the best move according to this solution will score 1.0.
    /// Selecting a winning move (but not best) will score 0.5.
    /// Selecting a losing move will score 0.0.
    fn score_policy(&self, policy: &Policy) -> f32 {
        let &sol_max = self.0.iter().max().unwrap();
        let best_moves = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, &x)| x == sol_max)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let winning_moves = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, &x)| x > 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        let policy_max = policy.iter().map(|&p| OrdF32(p)).max().unwrap().0;
        let selected_move = policy.iter().position(|&p| p == policy_max).unwrap();

        if best_moves.contains(&selected_move) {
            1.0
        } else if winning_moves.contains(&selected_move) {
            0.5
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use proptest::prelude::*;

    use crate::c4r::{tests::random_pos, Pos};

    use super::*;

    // TODO: Dynamically pull/compile this solver in CI
    const PATH_TO_SOLVER: &str = "/home/advait/connect4/c4solver";
    const PATH_TO_BOOK: &str = "/home/advait/connect4/7x6.book";

    fn paths_exist() -> bool {
        Path::new(PATH_TO_SOLVER).exists() && Path::new(PATH_TO_BOOK).exists()
    }

    fn test_solver() -> Solver {
        Solver {
            path_to_solver: PATH_TO_SOLVER.to_string(),
            path_to_book: PATH_TO_BOOK.to_string(),
        }
    }

    fn test_solve(pos: Pos) -> Solution {
        test_solver()
            .solve(&vec![pos])
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }

    fn one_hot(idx: usize) -> Policy {
        let mut ret = Policy::default();
        ret[idx] = 1.0;
        ret
    }

    #[test]
    fn default_pos() {
        if !paths_exist() {
            eprintln!("Warning: Skipping Solver tests because solver paths do not exist.");
            return;
        }
        let solution = test_solve(Pos::default());
        let expected_scores = &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        for (i, &score) in expected_scores.iter().enumerate() {
            assert_eq!(solution.score_policy(&one_hot(i)), score);
        }
    }

    #[test]
    fn p0_winning_pos() {
        if !paths_exist() {
            return;
        }
        let solution = test_solve(Pos::from_moves(&[2, 2, 3, 3]));
        let expected_scores = &[0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0];
        for (i, &score) in expected_scores.iter().enumerate() {
            assert_eq!(solution.score_policy(&one_hot(i)), score);
        }
    }

    #[test]
    fn p1_winning_pos() {
        if !paths_exist() {
            return;
        }
        let solution = test_solve(Pos::from_moves(&[0]));
        let expected_scores = &[0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.0];
        for (i, &score) in expected_scores.iter().enumerate() {
            assert_eq!(solution.score_policy(&one_hot(i)), score);
        }
    }

    proptest! {
        #[test]
        fn random_solutions(
            pos in random_pos().prop_filter(
                "non-terminal positions",
                |p| p.is_terminal_state().is_none()
            )
        ) {
            if !paths_exist() {
                return Ok(());
            }
            let _solution = test_solve(pos);
        }
    }
}
