use std::{
    error::Error,
    io::Write,
    process::{Command, Stdio},
};

use crate::c4r::Pos;

/// Interface to PascalPons's connect4 solver: https://github.com/PascalPons/connect4
/// Runs the solver in a subprocess, communicating via stdin/out.
struct Solver {
    path_to_solver: String,
    path_to_book: String,
}

impl Solver {
    fn new(path_to_solver: String, path_to_book: String) -> Self {
        Self {
            path_to_solver,
            path_to_book,
        }
    }

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
    use proptest::prelude::*;

    use crate::{
        c4r::{tests::random_pos, Pos},
        solver::Solver,
    };

    fn test_solver() -> Solver {
        Solver {
            path_to_solver: "/home/advait/connect4/c4solver".to_string(),
            path_to_book: "/home/advait/connect4/7x6.book".to_string(),
        }
    }

    #[test]
    fn empty_pos() {
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
            let _solution = &test_solver().solve(vec![pos]).unwrap()[0];
        }
    }
}
