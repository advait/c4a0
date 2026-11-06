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
- A single Python file runs with `uv run pytest -q tests/c4a0_tests/<file>_test.py`, in
  about 1 to 13 s.

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
1. **Solver.** Pascal Pons's solver built in `solver/` (gitignored) with `7x6.book`, per
   the README. Pass `--solver-path` and `--book-path`, and give each run its own
   `--solutions-path`.
2. **A metric that measures the right thing.** On master, a generation's `solver_score` is
   computed from its *parent's* self-play games (`src/c4a0/training.py`, scored before
   training). It gives 0.5 credit to slower wins and breaks ties by lowest column
   (`rust/src/solver.rs` `score_policy`). Until that is fixed, report it only as "parent
   self-play policy score". The target metric is the trained model's strict top-1
   agreement with solver-best moves (ties split evenly), on a fixed held-out position suite
   that spans plies, both players and all columns. Report the raw network and MCTS
   separately.
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
  `test:python` hung in 2 of 3 runs before `mise.toml` wrapped it in `timeout 300`. Exit code
  124 from `test:python` means this hang, not a test failure. Wrap any ad hoc self-play or
  training run in `timeout`.
- A Python exception in the NN callback panics the NN thread and leaves workers waiting
  forever (`rust/src/pybridge.rs`). That's a second reason for `timeout`.
- `nn-sweep` fails every trial (`trainer.gen_n` unset in `src/c4a0/sweep.py`). Don't use it
  as evidence until it's fixed.
- `py-spy` cannot attach in this container (ptrace denied). Diagnose hangs by bisecting test
  files with `timeout`.
