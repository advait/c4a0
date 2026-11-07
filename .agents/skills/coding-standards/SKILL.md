---
name: coding-standards
description: "Read before writing or committing c4a0 code: Python and Rust rules specific to this repo and the check a diff must pass."
---

# c4a0 coding standards

Status: implemented

These specialize the poteto principle leaves. Ignore **typescript-best-practices**, as there
is no TypeScript here.

**Before commit.** `mise run check` passes (Ruff, Pyright, Rust tests, Python tests, about
22 seconds warm). A change to self-play, training or the CLI also passes `mise run
train:smoke`. Run `cargo fmt` in `rust/` on Rust you touch. CI does not run it yet.

**Python.** 3.11.15 (`project.requires-python` in `pyproject.toml`, managed by uv). Ruff with default rules. Pyright in `basic` mode
(`pyproject.toml`). Existing `# type: ignore` comments mark untyped PyO3 calls. Add a stub to
`src/c4a0_rust/__init__.pyi` instead of adding more.

**Rust.** Stable toolchain. Tests live in a `mod tests` at the bottom of each file, with
`proptest` for properties and `approx` and `more_asserts` for floats.

**Cross-boundary rules.**
- A PyO3 signature change updates `rust/src/pybridge.rs`, `src/c4a0_rust/__init__.pyi`
  and every Python caller in one commit.
- Search constants (`c_exploration`, `c_ply_penalty`, MCTS iterations) have defaults in
  `src/c4a0/main.py`, `src/c4a0/tournament.py` and the Rust tests. Change them together.
- Changing what a `TrainingGen` field means is a migration (see **project-context**).

**Tests.** Search and scoring tests assert on concrete positions with known answers (an
immediate win, a forced loss, solver scores), not on internal counters. A threshold tuned to
one exploration formula is a smell. Assert the move the search should prefer.
