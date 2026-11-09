# Development reference

This page describes the code layout, tasks, tooling and training options. For install and usage, see the [README](../README.md). For what the experiments showed, see [what we learned trying to make self-play converge](experiments.md).

## Layout

| Path | Owns |
|---|---|
| `rust/src/c4r.rs` | Bitboard position, move rules, terminal detection, network input encoding. |
| `rust/src/mcts.rs` | PUCT search with proven-outcome backup (MCTS-Solver), root noise, visit-count policies, training samples. |
| `rust/src/self_play.rs` | Multi-threaded self-play, batched network calls, the per-run evaluation cache, random starts. |
| `rust/src/analysis.rs` | Single-threaded, batched MCTS over caller-supplied positions, used by evaluation. |
| `rust/src/labels.rs` | Mirror-canonical position keys, the dev/test/train split, the position sampler. |
| `rust/src/native_solver/`, `rust/src/bin/c4native.rs` | The clean-room solver that labels every position, its self-generated opening book, and a benchmark. See [native-solver.md](native-solver.md). |
| `rust/src/pybridge.rs` | The PyO3 API. Its stubs are `src/c4a0_rust/__init__.pyi`. |
| `rust/src/tui.rs`, `rust/src/interactive_play.rs` | The terminal UI behind `mise run play`. |
| `src/c4a0/nn.py` | The residual network with AlphaZero-style policy and value heads, its loss and the inference callback. |
| `src/c4a0/training.py` | `GenConfig`, `TrainingGen` metadata, the generation loop, replay. |
| `src/c4a0/evaluation.py` | Suite scoring: strict and keeps-the-result, raw network and MCTS, by ply bucket. |
| `src/c4a0/labels.py` | Solver labels and the committed suites in `src/c4a0/suite/v1/`. |
| `src/c4a0/supervised.py` | The supervised ceiling: training the same network directly on solver labels. |
| `src/c4a0/main.py` | The CLI: `train`, `play`, `eval`, `build-suite`, `label`, `ceiling`. |
| `scripts/converge_report.py` | Score curves per run, read from each generation's `metadata.json`. |
| `scripts/plot_progress.py` | The README progress graph, from `docs/data/training-progress.json`. |
| `scripts/alapin` | Runs jobs on a remote GPU machine: `sync`, `run`, `status`, `stop`, `fetch`. |

## Tasks

`mise.toml` defines every task. Each user-facing task depends on `build`, which depends on `install`, so a fresh checkout needs only `mise trust` and `mise install`.

| Task | Does |
|---|---|
| `train`, `play`, `eval`, `report`, `plot` | Usage tasks. Extra arguments go after `--`. |
| `check` | Ruff lint and format check, ty, Rust tests, Python tests. Run it before every commit. |
| `ci` | `check` plus `train:smoke`. CI runs this on every push. |
| `train:smoke` | One tiny CPU run through self-play, annealed training, saving and evaluation. |
| `suite:build` | Rebuilds the committed suites byte for byte. Needs no setup or book. |
| `solver:build`, `solver:book` | Build the solver CLI and its ply-10 opening book in `native-solver/` (gitignored, about 22 min once). `main.py label` needs the book. |
| `install`, `build` | `uv sync --locked`, then `maturin develop --release`. |

## Tooling

- **uv** owns the Python environment and the interpreter. `.python-version` picks the interpreter and `requires-python` states the supported range. Every task runs with `--locked`, so a stale `uv.lock` fails. Run `mise run install:update` after changing dependencies.
- **ty** is the only type checker. It runs with `--error-on-warning`.
- **Ruff** lints and formats. The ANN rules require annotations on every function signature, and `Any` is not allowed in signatures. Suppress a finding only with a rule-scoped comment and a reason.
- **Rust** uses the stable toolchain from mise. Run `cargo fmt` in `rust/` on code you change.

## Training options

`train` accepts these options. Self-play and training controls are `GenConfig` fields, recorded flat in each generation's `metadata.json`; network shape and `--mirror-inference` live in `ModelConfig` and the pickled model. The defaults are the best configuration measured (see [experiments](experiments.md)). A new training knob is a `GenConfig` field plus a CLI option, and nothing else threads it.

| Option | Default | Effect |
|---|---|---|
| `--n-self-play-games` | 2048 | Games per generation. All play at once, so this also sets the network batch size. |
| `--n-mcts-iterations` | 256 | Search iterations per self-play move. 1,024 and 2,048 trained no better. |
| `--c-exploration` | 2.0 | PUCT exploration constant. |
| `--c-ply-penalty` | 0.01 | Value penalty per ply, which makes faster wins worth more. |
| `--root-noise-alpha`, `--root-noise-frac` | 1.0, 0.25 | Dirichlet noise mixed into the root prior, in self-play only. |
| `--random-start-frac`, `--random-start-max-ply` | 0.5, 12 | Share of games that start from a random opening, and that opening's maximum length. |
| `--value-target-q-weight` | 1.0 | Value target = `(1 - w) * game result + w * root search value`. |
| `--replay-gens` | 8 | Train on the games of this many recent generations. |
| `--train-anneal-epochs` | 8 | Train exactly this many epochs per generation while a cosine schedule takes the rate to 1% of `--lr-schedule`'s; the lowest-validation-loss epoch is published. 0 trains at a constant rate with early stopping. |
| `--train-max-epochs`, `--train-patience` | 20, 3 | Epoch cap and early-stopping patience, used only with `--train-anneal-epochs 0`. |
| `--n-residual-blocks`, `--conv-filter-size` | 8, 64 | Residual trunk size. 8x128 was no better. |
| `--value-hidden` | 256 | Hidden width of the value head. Must match the saved model when resuming. |
| `--leaf-minimax-depth` | 0 | Exact alpha-beta to this depth on each MCTS leaf before the network sees it, in self-play and the generation's dev eval. Costs about 1.8x time per generation for a noise-level training gain, so it is off for training; its large gain is at evaluation (see below). |
| `--mirror-inference` | off | Average every network call with its left-right mirror, in self-play and evaluation. Doubles inference cost; worth 0.4 to 0.6 points strict at evaluation. |
| `--bf16` | off | BF16 autocast for self-play inference and Lightning training, including validation. Parameters stay FP32. Published-model dev evaluation stays FP32. |
| `--compile` | off | `torch.compile` of the self-play forward with dynamic batches. Falls back to eager on setup or execution failure, with a warning. Compilation stays outside the saved model. |
| `--train-steps N` | unset | Exactly N optimizer steps with full minibatches sampled with replacement from the augmented training split. Validates at the end and publishes the final model. Ignores epoch caps, patience and annealing. N must be positive. |
| `--eval-mcts-iterations` | 64 256 | MCTS budgets scored on the dev suite each generation. |
| `--device` | auto | `cuda`, then `mps`, then `cpu`. Lightning trains on the same device. |

The network is a residual trunk (a 3x3 convolution, then residual blocks) with AlphaZero-style
heads. The policy head has two 1x1-convolution channels, batch normalization, ReLU, flattening,
and a linear layer that produces seven log probabilities. The value head has one channel,
batch normalization, ReLU, flattening, and two linear layers with ReLU between them; tanh
produces `q_penalty` and `q_no_penalty`. `--value-hidden` sets the width between the value
layers. Legal-move masking stays in search and evaluation. An 8x64 network has 605k parameters.

Fully connected heads, the binned value head and the global-pooling, transformer and hybrid
trunks were removed after they measured no better (see [experiments](experiments.md)).
Metadata that still records their fields loads, because unknown fields are ignored. Model
pickles with fully connected heads no longer load; conv-head pickles from before the removal
load and predict unchanged.

Leaf minimax and mirror averaging are evaluation levers more than training ones. The depth is
recorded in each generation's metadata and `eval` rescoring reuses it; mirror averaging is a
`ModelConfig` field saved in the pickle. Neither has an `eval` or `play` override yet: to score
an existing model with them, call `evaluate_model(..., leaf_minimax_depth=D)` on a copy whose
`mirror_inference` attribute is set.

`label` samples train-split positions and labels them with the solver; build the book first
with `mise run solver:book`. `ceiling` trains the same network on solver labels. Its defaults match `train`: 8x64 and
`--anneal-epochs 40` (0 early-stops on dev loss with `--max-epochs` and `--patience`). Its
`report.json` records the model configuration under `net` and the parameter count under
`parameter_count`. Use genuine train-split labels and at least three seeds per arm. Judge
solver-best move selection by strict agreement at MCTS-256 on dev; keep the test split for
the final check.

## Benchmarking

Build with `mise run build`. `scripts/bench_generation.py` appends exactly one generation to
a copy of a completed run, inheriting its configuration. `--phase all` measures normal
self-play. `--phase training` reuses the latest games and trains from the latest model,
with older ancestors included once. Both modes evaluate and publish; use copies only.

The last stdout line is JSON, with synchronized CUDA wall times for replay loading, sample
conversion and device transfer, self-play, training, evaluation and publishing. It includes
epochs, optimizer steps, config and validation loss. `total_seconds` excludes Python imports
and initial device startup. Use an external process timer when startup matters.

Use separate built checkouts at `b5eca94` (timing only, master
training behavior) and this branch. Set the three input paths below. This runs five
alternating measurements per arm and phase, sequentially. Use a snapshot with all three
experimental controls off:

```sh
RUN=/path/to/completed-run-snapshot
BASELINE=/path/to/baseline-checkout
FAST=/path/to/perf-checkout
OUT=$(mktemp -d /tmp/c4a0-generation-bench.XXXXXX)
for i in 1 2 3 4 5; do
  for arm in baseline fast; do
    case "$arm" in
      baseline) checkout="$BASELINE" ;;
      fast) checkout="$FAST" ;;
    esac
    for phase in all training; do
      destination="$OUT/$arm-$phase-$i"
      cp -a "$RUN" "$destination"
      (cd "$checkout" && timeout 900 uv run --locked python scripts/bench_generation.py "$destination" --device cuda --phase "$phase" --eval-mcts-iterations 64 256) > "$OUT/$arm-$phase-$i.log" 2>&1
    done
  done
done
```

Compare medians and ranges, epochs and optimizer steps. Do not compare different amounts of
training as a throughput change. Compilation's first-call cost is included in self-play.
Test each results-changing option separately on fresh copies, then measure convergence:

```sh
cd "$FAST"
cp -a "$RUN" "$OUT/bf16"
timeout 900 uv run --locked python scripts/bench_generation.py "$OUT/bf16" --device cuda --bf16
cp -a "$RUN" "$OUT/compile"
timeout 900 uv run --locked python scripts/bench_generation.py "$OUT/compile" --device cuda --compile
cp -a "$RUN" "$OUT/steps"
timeout 900 uv run --locked python scripts/bench_generation.py "$OUT/steps" --device cuda --train-steps 1000
```

The options change results and have no demonstrated convergence benefit. The target is
strict solver agreement at MCTS-256 of at least 99% on dev, including the fastest win and
slowest loss.

On CPU, reproduce the sample-conversion comparison with:

```sh
timeout 300 uv run --locked python scripts/bench_samples.py --samples 700000 --runs 5 --torch-threads 1
```

For 699,448 augmented samples, the legacy build took a median 6.16 s and the bulk build
0.040 s over five alternating runs. All dataset bytes matched. This isolates conversion,
not a full generation. Tiny CPU generations took medians 0.207 s before and 0.191 s after,
excluding imports, with overlapping ranges. No end-to-end speedup is established by that
small workload. [The CPU measurements](data/generation-speed-cpu.json) record every run,
configuration and measurement boundary. Default-size GPU timings remain unmeasured here.

Training tensors occupy 372 bytes per augmented sample. Replay and label data stay in host
memory and Lightning copies each batch to the device, so the replay window does not grow GPU
memory; `C4A0_CUDA_MEMORY_FRACTION` caps each process's CUDA memory.
Run CUDA parity tests before interpreting GPU benchmarks:

```sh
timeout 300 uv run --locked pytest -q tests/c4a0_tests/nn_test.py tests/c4a0_tests/training_test.py -k 'inference_callback or datasets_live'
```

## Generation artifacts

Each generation is a directory under the base directory (default `training/`) named by its creation time:

- `model.pkl` is a CPU copy of the trained network, loadable on machines without CUDA.
  Root and child suite evaluations reload these exact bytes onto the training device.
  `load_model` also maps legacy CUDA pickles through CPU before moving to the requested device.
- `games.pkl` holds the self-play games that trained it.
- `metadata.json` holds its `GenConfig` fields, its parent, its validation loss, and `evals`, the dev-suite scores of this exact `model.pkl`.

`metadata.json` is written last. A directory without one is an incomplete generation and is ignored. Older metadata files load, because unknown fields are ignored and new fields have defaults.
