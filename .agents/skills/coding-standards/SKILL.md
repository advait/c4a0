---
name: coding-standards
description: "Read before writing or committing c4a0 code: Python and Rust rules specific to this repo and the check a diff must pass."
---

# c4a0 coding standards

Status: implemented

These specialize the poteto principle leaves. Ignore **typescript-best-practices**, as there
is no TypeScript here.

**Before commit.** `mise run check` passes (Ruff, ty, Rust tests, Python tests, about
22 seconds warm). A change to self-play, training or the CLI also passes `mise run
train:smoke`. Run `cargo fmt` in `rust/` on Rust you touch. CI does not run it yet.

**Python.** 3.11.15 (`project.requires-python` in `pyproject.toml`, managed by uv).
Ruff enforces ANN (including ANN401), UP, I, B, F, E and W, except E501.
`ruff format --check` owns wrapping. RUF100 and PGH003/PGH004 reject unused or
blanket suppressions. All source and test signatures carry annotations. Explicit `Any`
is banned in signatures; use concrete types, protocols or `object`. ty is the only type checker
(`pyproject.toml`). Default ty diagnostics remain enabled. Import, reference, attribute,
argument, assignment, return and override errors fail the check. Missing generic arguments
and invalid or unused suppressions fail too. Warnings fail the task.
Add native types to `src/c4a0_rust/__init__.pyi`; `_native.pyi`
re-exports them for the compiled module. Fix types before suppressing diagnostics.
A necessary suppression uses `# ty: ignore[rule-name]` with a reason on that line.
Blanket and unused suppressions fail typecheck.

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
