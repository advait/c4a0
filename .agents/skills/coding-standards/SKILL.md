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
  `src/c4a0/main.py` and the Rust tests. Keep their intended settings explicit.
- Winning `train` defaults live in `src/c4a0/main.py`: auto device, 2048 games,
  256 MCTS iterations, exploration 2.0, ply penalty 0.01, root noise 1.0/0.25,
  replay 8, Q weight 1.0, random starts 0.5 through ply 12, 8 cosine-annealed
  epochs per generation, an 8-block 64-filter network with conv heads, leaf minimax
  off, mirror inference off, and eval budgets 64/256. `ceiling` matches: 8x64,
  40 annealed epochs.
  `GenConfig` keeps historical fallback values for old metadata that omitted controls.
  Keep `train:smoke` explicitly small and on CPU.
- Self-play temperature is fixed: 4.0 before ply 4, 2.0 before ply 8, then 1.0.
  Terminal policies remain supervised. Keep losses functional so model deepcopy works.
- Changing what a `TrainingGen` field means is a migration (see **project-context**).
- A new training knob is a GenConfig field plus a CLI option; nothing else threads it.
  Consume it from `config` where it takes effect. `TrainingGen` inherits config fields
  to keep metadata flat and includes them automatically for root and child generations.

**Tests.** Search and scoring tests assert on concrete positions with known answers (an
immediate win, a forced loss, solver scores), not on internal counters. A threshold tuned to
one exploration formula is a smell. Assert the move the search should prefer.
