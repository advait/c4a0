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
  It proves outcomes from the rules (MCTS-Solver). A proven root stops early, so
  `root_visit_count` can stay below the budget. Its policy is all mass on the proven-best
  moves and its Q is exact. `mcts.rs` checks forced wins in 1, 3 and 5 plies, the slowest
  loss and near-full draws against a negamax oracle in the test module.
  Search tests assert preferred moves and visit spread, including the low-prior immediate
  win that the old formula starved. Do not restore old-formula visit-share thresholds.
- For experiment controls, `mcts.rs` tests root-Q blending at W=0, W=0.25 and W=1,
  unchanged terminal targets, signed-zero preservation in draws, and positive targets
  for either player in immediate-win positions. `self_play.rs` checks the fixed
  temperature at every ply 0-41 and compares default games byte for byte with a
  pre-random-start fixture. It also checks deterministic non-terminal random starts
  and samples without opening prefixes. `pybridge_test.py` covers root noise,
  value blending and deterministic splits. `nn_test.py` checks functional losses and
  deepcopy after training. `training_test.py` checks replay from disk, saved metadata,
  the saved model's loss on the replay union, and epoch caps and patience in real fits.
  Legacy metadata tests include retired fields. `main_test.py` checks winning CLI
  defaults, control forwarding, device selection, baseline play without training data,
  and the error for a missing trained model.
- Head tests in `nn_test.py` pin default and explicit `fc` tensor shapes and bytes to
  `1bccc92` with seed 1337 and one torch thread. The fixture includes BatchNorm buffers.
  Keep `fc_4x64_seed1337.json` and `legacy_model.pkl` from that pre-change code.
  Do not regenerate them from the implementation under test. Tests also check legacy
  metadata defaults, conv output shapes and ranges, singleton batches, gradients, numpy
  callbacks, and pickle round trips. A 4-block 64-filter conv model has 309,470 parameters
  with `value_hidden=256`, below 1M. FC head depth settings have no effect on conv heads.
  CLI tests check head forwarding for `train` and `ceiling`, including ceiling reports.
- `generation_golden_test.py` pins two generations to master `57d3a2a`: game CBOR,
  augmented dataset bytes, every training and validation batch loss, validation scores,
  and every published model tensor. It uses seed 1337, one torch thread, singleton NN
  batches and serial games **in the test only**. Master returns games in worker completion
  order, which otherwise changes replay splits even with fixed random seeds. Both omitted
  and explicitly disabled BF16, compile and fixed-step controls must match the fixture.
  Do not regenerate the fixture from optimized code. The baseline was captured by loading
  master `nn.py` and `training.py` before running the same harness, and checked six times.
- Bulk sample tests compare every float byte, including signed zero, to the legacy split
  and per-sample conversion. Replay concatenation must preserve CBOR order. Callback tests
  cover changing batch sizes on tiny and default 4x64 nets, including singleton batches,
  and compare to the original `no_grad` forward. Device tests
  check dataset storage is reused by loaders. CUDA cases skip in this container.
- Fixed-step tests run real fits at 1, 5 and 11 steps, check final published tensors and
  final validation loss, and prove minibatch sampling uses replacement. BF16 has a real
  CPU fit. Compile tests exercise dynamic Dynamo batches with the eager backend, plus
  immediate and lazy backend failures; they do not measure Inductor performance.
- Raw-network dev eval of the historical 1x32 model (2447 positions, suite load through `SuiteEval`)
  took a median 0.15 s (0.148 to 0.166 s, 3 runs, load average 5, 16 torch threads); the
  forward pass is about 0.08 s of it. MCTS-64 on the dev suite took 1.7 s (one run).
- Solver-gated tests (golden rows, suite rebuild) run when `solver/c4solver` and
  `solver/7x6.book` exist and skip otherwise.

## Lane 2: end to end (seconds to minutes)

- `mise run train:smoke` runs two tiny CPU jobs, one with `--head fc` and one with
  `--head conv`. Each runs one generation with 4 games and 4 MCTS iterations through
  Rust self-play, the PyO3 callback, Lightning training, saving, and dev evaluation.
  FC artifacts go to `training/ci-smoke`, conv artifacts to `training/ci-smoke-conv`.
  The task removes both directories before their runs.
- The smoke task explicitly uses a 1-block 8-filter CPU network, 2 epochs with
  patience 1, and MCTS-4 evaluation. It exercises the winning self-play controls.
  For replay across generations, repeat its tiny command with `--max-gens 2`.
  Generation two saves only its new games. This proves plumbing, not convergence.
- `training_test.py` also runs a conv generation with `value_hidden=17`, ignored FC
  head depths, and MCTS-4 evaluation. It checks optimizer updates and reloads root and
  child models and metadata. `main_test.py` checks the saved ceiling model and report.
- For phase timing after the smoke run, copy its artifacts and append one measured
  generation. `--phase training` reuses the newest games, includes each older ancestor
  once, and still evaluates and publishes:

  ```sh
  run_copy=$(mktemp -d /tmp/c4a0-bench.XXXXXX)
  cp -a training/ci-smoke/. "$run_copy/"
  timeout 900 uv run --locked python scripts/bench_generation.py "$run_copy" --device cpu --torch-threads 1 --eval-mcts-iterations 4 --phase training
  ```

  The final stdout line is JSON with synchronized phase times, epochs and optimizer
  steps. `total_seconds` excludes imports and initial device startup. Compare equal
  work counts from identical snapshots. Use `--phase all` for self-play too.
- `timeout 300 uv run --locked python scripts/bench_samples.py --samples 700000 --runs 5
  --torch-threads 1` alternates the master conversion with Rust bulk arrays and hashes
  all tensor bytes. On 2026-11-08, 699,448 augmented samples took medians 6.16 s and
  0.040 s, respectively, across five runs per arm. Tiny CPU end-to-end timings overlap;
  they establish no generation speedup. Details are in `docs/data/generation-speed-cpu.json`.
- `mise run train -- --max-gens 1 --n-self-play-games 8 --n-mcts-iterations 8
  --base-dir /tmp/c4a0-check --device cpu --n-residual-blocks 1 --conv-filter-size 8`
  exercises the user task with a small CPU run and the default eval budgets 64/256.
- `play` needs a terminal. Check startup with
  `timeout 5 script -q -c 'mise run play -- --model uniform' /dev/null`.
  Confirm that the board renders and termination is due to timeout.
- To check bootstrap, create a temporary worktree at the branch commit, run
  `mise trust && mise install && mise run play -- --model uniform` in a terminal,
  then remove that worktree. No solver or training directory is needed.
- A historical generation with 256 games, 256 MCTS iterations and a 1-block 32-filter
  CPU network took 276 s. It does not measure the current 4-block 64-filter default.

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

When the coordinator owns GPU measurements, provide commands and do not run remote jobs.
The generation benchmark and five-run alternating baseline procedure are in
`docs/development.md` under **Benchmarking**. Use built checkouts at `b5eca94` (instrumented
master behavior) and the candidate, one job at a time, with fresh copies of the same run.
Time both `--phase all` and `--phase training`. CUDA synchronization brackets every phase.
Inspect epochs, optimizer steps, validation loss and all errors before reporting throughput.

First run the CUDA parity cases (they skip on CPU):

```sh
timeout 300 uv run --locked pytest -q tests/c4a0_tests/nn_test.py tests/c4a0_tests/training_test.py -k 'inference_callback or datasets_live'
```

The maintainer's default-generation baseline is self-play 30.1 s (three identical runs),
training 61.9-88.5 s as early stopping varies, and total 98-125 s including startup and eval.
The default-on optimizations have CPU equality evidence, not GPU speedup evidence. Device
datasets use 372 bytes per augmented sample; account for model and activation memory too.

`--bf16`, `--compile`, and `--train-steps N` are results-changing, default-off controls,
recorded in metadata. Test them separately on copied runs. BF16 affects self-play and
training/validation, while published-model evaluation stays FP32. Compile applies only
to self-play; cold compilation counts in its timer and unsupported backends warn and
fall back to eager. Fixed steps sample with replacement, ignore epoch caps and patience,
validate at the end and publish the final model. CPU tests do not prove convergence.
Judge these experiments by strict fastest-win/slowest-loss agreement at MCTS-256 on dev,
with a target of at least 99%, using the variance requirements in lane 3.

`alapin` (RTX 3090, 32 cores) runs the heavy lanes, driven from this container with
`scripts/alapin` (see the **workspace** skill). Measured on 2026-11-06, 5 runs, medians:

| Work | alapin GPU | dev container CPU |
|---|---|---|
| Self-play, 1024 games x 128 MCTS iterations | 7.4 s | 29.8 s (with eval cache) |
| Supervised epoch, historical 1x32 net, 1.69M samples | 10 s (batched loader, 2 epochs) | about 3-4 min |
| Supervised epoch, 4 blocks x 64 filters | 41 s (old loader) | over 95 min |

Self-play there is bound by Python dispatch and CPU-side MCTS (GPU about 20% busy), not the
GPU. Keep at least 1024 games in flight. Cut MCTS iterations, not games, to shrink a proxy run.

Supervised ceiling on suite v1 dev (846,552 train labels, patience 10, lr 2e-3 flat): the
historical 1x32 net reaches 91.8-91.9% raw top-1 over 3 seeds (92.1-92.4% with MCTS-64). The 4x64 net
reaches 92.6% raw (93.5% MCTS-64), 1 seed. Plies 0-11 are the weak buckets (73-85%).
The raw ceiling measures estimator fit. MCTS can improve on raw-network move selection;
the architecture's raw top-1 is not a bound on MCTS play.

## Hazards

- Historical MCTS scores used a different search, often with exploration 6.6. PUCT also
  applies to `eval` backfills, which read exploration from each generation's metadata.
  Label old and new search results and match budgets and exploration before comparing.
- Winning `train` defaults are 2048 games, MCTS-256, exploration 2.0, ply penalty
  0.01, root noise alpha 1.0 and fraction 0.25, replay 8, Q weight 1.0, random starts
  0.5 through ply 20, 20 epochs with patience 3, and a 4-block 64-filter network.
  Device `auto` selects CUDA, then MPS, then CPU. Dev eval budgets are 64 and 256.
  Each control remains a CLI option. `GenConfig` retains historical fallback values
  for omitted fields in legacy metadata; explicit saved values still win.
- Self-play temperature is fixed at 4 below ply 4, 2 below ply 8, then 1.
  Terminal policy supervision stays enabled. Metadata fields for retired controls
  are ignored when loading and omitted when saving.
- `--train-max-epochs N` caps each generation's Lightning fit at N epochs.
  `--train-patience P` stops after P validation checks without improved `val_loss`.
  Validation runs once per epoch. Both must be at least 1 and are recorded as flat
  metadata fields through `GenConfig`.
- `--head fc|conv` defaults to `fc` for both new and legacy configurations. Conv heads
  use `--value-hidden N` (default 256, positive) and ignore `--n-policy-layers` and
  `--n-value-layers`. Both `train` and `ceiling` expose these options. `train` records
  them in flat metadata; `ceiling` records them under `net` in `report.json`.
  A resumed run must use the saved model's head architecture and, for conv, value width.
  Use a fresh `--base-dir` to change either. These options change results. Component
  tests and smoke runs establish no convergence benefit; compare architectures using
  lane 3 at a fixed MCTS-256 budget on dev.
- `--random-start-frac F` starts a fraction of self-play games from random openings.
  F must be finite and in [0, 1]. `--random-start-max-ply K` must be at least 1 and
  defaults to 20 for `train` (legacy metadata falls back to 12). Rust caps K at 41,
  the last possible non-terminal ply, and samples
  an opening length uniformly from 1 through that cap. Each walk chooses uniformly
  among legal moves and restarts at the same length if it reaches a terminal position.
  The coin flip and walks use a separate game-ID-seeded stream. The random prefix
  produces no training samples; temperature uses the board's ply.
  Both settings are flat `GenConfig` metadata fields. Component and smoke tests prove
  plumbing and default compatibility, not improved convergence.
- `--value-target-q-weight W` blends each non-terminal head's final-outcome target z
  with that head's root MCTS mean, `(1-W)*z + W*q_root`. Root means are captured before
  advancing the root and use the player-to-move perspective. W must be finite and in
  [0, 1]. Terminal samples always keep z. No convergence benefit is established by
  component or smoke tests.
- Root noise affects self-play only, once per move over legal priors, including retained
  subtrees. Evaluation and interactive play never mix noise. Alpha must be positive and
  finite; fraction must be in [0, 1], with alpha supplied when fraction is positive.
  Noise uses a separate RNG stream seeded by game ID and ply.
- Replay follows parent links and loads up to G-1 ancestors. It splits the combined
  games once and never saves replay duplicates. Old games retain the settings that
  generated them. Allow the window to fill after changing self-play settings before
  interpreting an experiment.
- A Python exception in the NN callback during self-play panics the NN thread
  (`rust/src/pybridge.rs` `PyEvalPos::eval_pos`, `.expect`) and leaves workers waiting
  forever. Wrap ad hoc self-play and training runs in `timeout`. Evaluation search
  (`mcts_policies`) propagates the exception instead.
- Run validation tasks sequentially: overlapping `check` and `train:smoke` builds can race
  while maturin stages the extension. `train --device cpu` selects Lightning's CPU
  accelerator through `lightning_accelerator`; root and child generations are scored on
  the requested training device, reloading the saved CPU model pickle first.
- `py-spy` cannot attach in this container or on alapin (ptrace restricted, no sudo).
  Long runs therefore must log progress themselves; `ceiling` logs one line per epoch.
