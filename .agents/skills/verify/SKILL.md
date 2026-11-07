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
- A single Python file runs with `timeout 300 uv run --locked pytest -q tests/c4a0_tests/<file>_test.py`, in
  about 1 to 13 s.
- PUCT with true-mean Q is the only search; the exploration default is now 2.0.
  Search tests assert preferred moves and visit spread, including the low-prior immediate
  win that the old formula starved. Do not restore old-formula visit-share thresholds.
- For experiment controls, `mcts.rs` tests root-Q blending at W=0, W=0.25 and W=1,
  unchanged terminal targets, signed-zero preservation in draws, and positive targets
  for either player in immediate-win positions. `self_play.rs` checks the default
  temperature at every ply 0-41, custom boundaries and the greedy override.
  `pybridge_test.py` covers greedy moves, root noise, value blending, temperature schedules
  and terminal target masking; `nn_test.py` checks masked policy loss and gradients. `training_test.py`
  checks replay from disk, saved metadata and the saved model's validation loss on the
  replay union. It also compares a fixed two-generation run with the pre-GenConfig
  baseline and checks epoch caps and patience in real Lightning fits.
  `main_test.py` checks CLI defaults, flag forwarding through `GenConfig`, and invalid
  training-length values.
- Raw-network dev eval of a default model (2447 positions, suite load through `SuiteEval`)
  took a median 0.15 s (0.148 to 0.166 s, 3 runs, load average 5, 16 torch threads); the
  forward pass is about 0.08 s of it. MCTS-64 on the dev suite took 1.7 s (one run).
- Solver-gated tests (golden rows, suite rebuild) run when `solver/c4solver` and
  `solver/7x6.book` exist and skip otherwise.

## Lane 2: end to end (seconds to minutes)

- `mise run train:smoke` runs one tiny generation (4 games, 4 MCTS iterations) through
  Rust self-play, the PyO3 callback, Lightning training and saving.
- The smoke task uses default controls. To exercise enabled controls, use its tiny CLI
  command with `--max-gens 2 --greedy-after-ply 8 --root-noise-alpha 0.3
  --root-noise-frac 0.25 --mask-terminal-policy --replay-gens 2
  --value-target-q-weight 0.6 --temperature-schedule 2:1.5
  --train-max-epochs 2 --train-patience 1`. Generation two trains on
  eight games but saves only its four new games. This proves plumbing, not convergence.
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
   compare it with `evals`. Judge convergence by MCTS move selection at a fixed budget;
   raw-network top-1 diagnoses the estimator. Keep the test suite for a final number, not
   for selection.
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

## Lane 4: the GPU machine (alapin)

`alapin` (RTX 3090, 32 cores) runs the heavy lanes, driven from this container with
`scripts/alapin` (see the **workspace** skill). Measured on 2026-11-06, 5 runs, medians:

| Work | alapin GPU | dev container CPU |
|---|---|---|
| Self-play, 1024 games x 128 MCTS iterations | 7.4 s | 29.8 s (with eval cache) |
| Supervised epoch, default net, 1.69M samples | 10 s (batched loader, 2 epochs) | about 3-4 min |
| Supervised epoch, 4 blocks x 64 filters | 41 s (old loader) | over 95 min |

Self-play there is bound by Python dispatch and CPU-side MCTS (GPU about 20% busy), not the
GPU. Keep at least 1024 games in flight. Cut MCTS iterations, not games, to shrink a proxy run.

Supervised ceiling on suite v1 dev (846,552 train labels, patience 10, lr 2e-3 flat): the
default net reaches 91.8-91.9% raw top-1 over 3 seeds (92.1-92.4% with MCTS-64). The 4x64 net
reaches 92.6% raw (93.5% MCTS-64), 1 seed. Plies 0-11 are the weak buckets (73-85%).
The raw ceiling measures estimator fit. MCTS can improve on raw-network move selection;
the architecture's raw top-1 is not a bound on MCTS play.

## Hazards

- Historical MCTS scores used a different search, often with exploration 6.6. PUCT also
  applies to `eval` backfills, which read exploration from each generation's metadata.
  Label old and new search results and match budgets and exploration before comparing.
- All new controls preserve the previous self-play and training behavior by default:
  `temperature_schedule=[(4, 4.0), (8, 2.0)]` keeps temperatures 4 below ply 4,
  2 below ply 8, then 1; `greedy_after_ply=None` leaves that schedule in effect;
  `value_target_q_weight=0.0` keeps the exact final-outcome value targets;
  `root_noise_alpha=None` and `root_noise_frac=0` disable noise;
  `mask_terminal_policy=False` keeps terminal policy supervision; `replay_gens=1` uses
  only new games; `train_max_epochs=100` and `train_patience=10` preserve the previous
  training limits. These are also the defaults when loading old metadata.
- `--train-max-epochs N` caps each generation's Lightning fit at N epochs.
  `--train-patience P` stops after P validation checks without improved `val_loss`.
  Validation runs once per epoch. Both must be at least 1 and are recorded as flat
  metadata fields through `GenConfig`.
- `--value-target-q-weight W` blends each non-terminal head's final-outcome target z
  with that head's root MCTS mean, `(1-W)*z + W*q_root`. Root means are captured before
  advancing the root and use the player-to-move perspective. W must be finite and in
  [0, 1]. Terminal samples always keep z. No convergence benefit is established by
  component or smoke tests.
- `--temperature-schedule SPEC` takes comma-separated `until_ply:temperature` pairs.
  Plies are non-negative integers in strictly increasing order; temperatures are finite
  and positive. Boundaries are exclusive, with 1.0 after the last pair. For example,
  `2:1.5` uses 1.5 at plies 0-1 and 1.0 afterwards. `greedy_after_ply` takes precedence.
  Metadata stores `temperature_schedule` as a list of pairs.
- Root noise affects self-play only, once per move over legal priors, including retained
  subtrees. Evaluation and interactive play never mix noise. Alpha must be positive and
  finite; fraction must be in [0, 1], with alpha supplied when fraction is positive.
  Noise uses a separate RNG stream seeded by game ID and ply.
- Terminal masking changes split copies, not `games.pkl`: zero policy targets mean zero
  policy loss, with both value losses retained. Replay follows parent links and loads up
  to G-1 ancestors. It splits the combined games once, and never saves replay duplicates.
  Old games in the window retain the settings that generated them; allow the window to
  fill after changing self-play settings before interpreting an experiment.
- A Python exception in the NN callback during self-play panics the NN thread
  (`rust/src/pybridge.rs` `PyEvalPos::eval_pos`, `.expect`) and leaves workers waiting
  forever. Wrap ad hoc self-play and training runs in `timeout`. Evaluation search
  (`mcts_policies`) propagates the exception instead.
- Run validation tasks sequentially: overlapping `check` and `train:smoke` builds can race
  while maturin stages the extension. `train --device cpu` selects Lightning's CPU
  accelerator through `lightning_accelerator`; the root generation is scored on CPU.
- `nn-sweep` fails every trial (`trainer.gen_n` unset in `src/c4a0/sweep.py`). Don't use it
  as evidence until it's fixed.
- `py-spy` cannot attach in this container or on alapin (ptrace restricted, no sudo).
  Long runs therefore must log progress themselves; `ceiling` logs one line per epoch.
