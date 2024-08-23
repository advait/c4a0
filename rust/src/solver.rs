use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs,
    io::Write,
    process::{Command, Stdio},
};

use serde::{Deserialize, Serialize};

use crate::c4r::Pos;

/// A caching wrapper around [Solver] that caches solutions to positions.
struct CachingSolver {
    solver: Solver,
    cache_path: String,
}

impl CachingSolver {
    pub fn new(path_to_solver: String, path_to_book: String, cache_path: String) -> Self {
        Self {
            solver: Solver::new(path_to_solver, path_to_book),
            cache_path,
        }
    }

    /// Solves the given position, resorting to cached positions if possible, relying on
    /// [Self::solver] to solve missing positions, and then caches resulting solutions.
    pub fn solve(&self, pos: Vec<Pos>) -> Result<Vec<Solution>, Box<dyn Error>> {
        let mut cache = self.get_cache()?;

        let missing_pos = pos
            .iter()
            .filter(|p| !cache.0.contains_key(p))
            .cloned()
            .collect::<HashSet<_>>() // Remove duplicates
            .into_iter()
            .collect::<Vec<_>>();
        let missing_solutions = self.solver.solve(missing_pos)?;
        for solution in missing_solutions.iter() {
            cache.0.insert(solution.pos.clone(), solution.solution);
        }
        self.write_cache(&cache)?;

        let ret = pos
            .into_iter()
            .map(|pos| {
                let solution = cache.0.get(&pos).unwrap().clone();
                Solution { pos, solution }
            })
            .collect();
        Ok(ret)
    }

    fn get_cache(&self) -> Result<SolutionCache, Box<dyn Error>> {
        if !std::path::Path::new(&self.cache_path).exists() {
            return Ok(SolutionCache::default());
        }

        let bytes = fs::read(&self.cache_path)?;
        Ok(serde_cbor::from_slice(&bytes)?)
    }

    fn write_cache(&self, cache: &SolutionCache) -> Result<(), Box<dyn Error>> {
        let bytes = serde_cbor::to_vec(&cache)?;
        fs::write(&self.cache_path, &bytes)?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct SolutionCache(HashMap<Pos, [i16; Pos::N_COLS]>);

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
    fn solve(&self, pos: Vec<Pos>) -> Result<Vec<Solution>, Box<dyn Error>> {
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

        let solutions: Vec<[i16; Pos::N_COLS]> = stdout_str
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
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect(format!("failed to parse stdout: '{}'", l).as_str())
            })
            .collect();

        Ok(pos
            .into_iter()
            .zip(solutions.into_iter())
            .map(|(pos, solution)| Solution { pos, solution })
            .collect())
    }
}

#[derive(Debug)]
struct Solution {
    pos: Pos,
    solution: [i16; Pos::N_COLS],
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use proptest::prelude::*;

    use crate::{
        c4r::{tests::random_pos, Pos},
        solver::Solver,
    };

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

    #[test]
    fn empty_pos() {
        if !paths_exist() {
            eprintln!("Warning: Skipping SOlver test because solver paths do not exist.");
            return;
        }
        let pos = Pos::default();
        let _solution = &test_solver().solve(vec![pos]).unwrap()[0];
    }

    proptest! {
        #[test]
        fn test_solve(
            pos in random_pos().prop_filter(
                "non-terminal positions",
                |p| p.is_terminal_state().is_none()
            )
        ) {
            if !paths_exist() {
                return Ok(());
            }
            let _solution = &test_solver().solve(vec![pos]).unwrap()[0];
        }
    }
}
