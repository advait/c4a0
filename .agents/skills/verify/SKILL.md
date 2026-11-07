---
name: verify
description: "Read before proving a c4a0 change works: the fast component lanes, the opt-in convergence lane, their costs on this CPU-only container, and known hazards."
---

# Verifying c4a0

Status: implemented

Pick the cheapest lane that exercises the changed behavior. Then go one lane higher before
calling the work done. Timings were measured on the CPU dev container (32 cores, no GPU) on
2026-11-06.

## Lane 1: components (seconds)

- `mise run test:rust` takes about 9 s. It covers rules, MCTS and self-play units. For a
  search or scoring change, add a test on a concrete position (an immediate win, a forced
  block, a terminal node) and watch it fail before the fix.
- `mise run check` takes about 22 s warm. That is lint, typecheck, Rust and Python tests.
  The first run after a source change rebuilds the extension (about 2 minutes).
- A single Python file runs with `timeout 900 uv run --locked pytest -q tests/c4a0_tests/<file>_test.py`, in
  about 1 to 13 s.
- Raw-network dev eval of a default model (2447 positions, suite load through `SuiteEval`)
  took a median 0.15 s (0.148 to 0.166 s, 3 runs, load average 5, 16 torch threads); the
  forward pass is about 0.08 s of it. MCTS-64 on the dev suite took 1.7 s (one run).
- Solver-gated tests (golden rows, suite rebuild) run when `solver/c4solver` and
  `solver/7x6.book` exist and skip otherwise.

## Lane 2: end to end (seconds to minutes)

- `mise run train:smoke` runs one tiny generation (4 games, 4 MCTS iterations) through
  Rust self-play, the PyO3 callback, Lightning training and saving.
- One realistic generation (256 games, 256 MCTS iterations, default 1-block 32-filter
  network, `--device cpu`) took 276 s. Self-play was 229 s and training 43 s. Use a
  `/tmp` base dir, as in `--base-dir /tmp/c4a0-<name>`.

## Lane 3: convergence (opt-in, minutes to hours)

Run it only when the task is to measure a change's effect on convergence, or the user asks.
Its question is how the share of solver-optimal moves changes across generations.

Requirements before a convergence number counts:
1. **The metric.** Every published generation's `metadata.json` holds
   `evals["v1/dev"]` (`SuiteEval` in `src/c4a0/evaluation.py`): the saved child model's top-1
   agreement with solver-best moves on the committed dev suite (`src/c4a0/suite/v1/`), ties
   split evenly, slower wins count as misses, full columns masked, trivial rows excluded,
   tallied per ply bucket. Raw network and value are always recorded. Add MCTS budgets with
   `train --eval-mcts-iterations 64`, or afterwards with `main.py eval --base-dir <run>
   --mcts-iterations 64`. No solver is needed. Report raw and MCTS separately, and read the
   buckets: on a fresh random net MCTS-64 scores 0.33 at plies 0-5 but 0.99 at plies 36-41
   (search nearly solves late tactics), so the aggregate hides the opening. Legacy
   `solver_score` (parent self-play, half credit for slower wins) is never written now; do not
   compare it with `evals`. Keep the test suite for a final number, not for selection.
2. **Ceiling.** `main.py label` then `main.py ceiling` (see `docs/development.md`) shows what
   the architecture reaches with solver labels directly. Needs Pascal Pons's solver built in
   `solver/` with `7x6.book`, per the README.
3. **Variance.** Use at least 3 seeds per arm. A single run is an anecdote. The README's
   9-generation result is one.
4. **Scale.** On CPU, self-play scales roughly with games × MCTS iterations (inferred from
   one measurement). A proxy of 256 games × 128 iterations is a few minutes per generation.
   The autoresearch branch's best config (2048 × 512) is about an hour per generation.
   Compare arms at the same proxy scale, and confirm only the winner at a larger one.

Apply the **benchmark-checklist** skill and **principle-explain-the-number** before you report
a convergence delta.

## Hazards

- **Self-play shutdown deadlock.** `rust/src/self_play.rs` decrements `n_games_remaining`
  then reloads it, so two workers can both start shutdown and overfill the job queue.
  `test:python` hung in 2 of 3 runs before `mise.toml` wrapped it in `timeout 900`. Exit code
  124 from `test:python` means this hang, not a test failure. Wrap any ad hoc self-play or
  training run in `timeout`.
- If the unrestricted Python test retry also hangs, `taskset -c 0 timeout 900 mise run check`
  exercises all tests with one self-play worker. Use an allowed CPU from
  `os.sched_getaffinity(0)` if CPU 0 is unavailable. Report that concurrency was restricted.
  This validates Python behavior but does not prove the concurrent shutdown is correct.
- A Python exception in the NN callback panics the NN thread and leaves workers waiting
  forever (`rust/src/pybridge.rs`). That's a second reason for `timeout`.
- `nn-sweep` fails every trial (`trainer.gen_n` unset in `src/c4a0/sweep.py`). Don't use it
  as evidence until it's fixed.
- `py-spy` cannot attach in this container (ptrace denied). Diagnose hangs by bisecting test
  files with `timeout`.
