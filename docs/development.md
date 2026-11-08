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
| `rust/src/solver.rs` | Strict stdin/stdout protocol for Pascal Pons's solver. |
| `rust/src/native_solver/`, `rust/src/bin/c4native.rs` | A clean-room Rust solver with its own opening book, a benchmark, and a diff harness against Pons. Not wired into labeling. See [native-solver.md](native-solver.md). |
| `rust/src/pybridge.rs` | The PyO3 API. Its stubs are `src/c4a0_rust/__init__.pyi`. |
| `rust/src/tui.rs`, `rust/src/interactive_play.rs` | The terminal UI behind `mise run play`. |
| `src/c4a0/nn.py` | The residual CNN and its loss. |
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
| `train:smoke` | Two tiny CPU runs, one per head architecture, through self-play, training, saving and evaluation. |
| `suite:build` | Rebuilds the committed suites. Needs the solver in `solver/`. |
| `solver:native:build`, `solver:native:book` | Build the native solver and its ply-10 opening book in `native-solver/` (gitignored). |
| `solver:diff` | Compares the native solver against Pons and adjudicates every disagreement. Set `C4A0_SOLVER` and `C4A0_BOOK` to the Pons binary and book. |
| `install`, `build` | `uv sync --locked`, then `maturin develop --release`. |

## Tooling

- **uv** owns the Python environment and the interpreter. `.python-version` picks the interpreter and `requires-python` states the supported range. Every task runs with `--locked`, so a stale `uv.lock` fails. Run `mise run install:update` after changing dependencies.
- **ty** is the only type checker. It runs with `--error-on-warning`.
- **Ruff** lints and formats. The ANN rules require annotations on every function signature, and `Any` is not allowed in signatures. Suppress a finding only with a rule-scoped comment and a reason.
- **Rust** uses the stable toolchain from mise. Run `cargo fmt` in `rust/` on code you change.

## Training options

`train` accepts these options. Every option is a `GenConfig` field and is recorded in each generation's `metadata.json`. The defaults are the best configuration found. A new option is a `GenConfig` field plus a CLI option, and nothing else threads it.

| Option | Default | Effect |
|---|---|---|
| `--n-self-play-games` | 2048 | Games per generation. All play at once, so this also sets the network batch size. |
| `--n-mcts-iterations` | 256 | Search iterations per self-play move. |
| `--c-exploration` | 2.0 | PUCT exploration constant. |
| `--c-ply-penalty` | 0.01 | Value penalty per ply, which makes faster wins worth more. |
| `--root-noise-alpha`, `--root-noise-frac` | 1.0, 0.25 | Dirichlet noise mixed into the root prior, in self-play only. |
| `--random-start-frac`, `--random-start-max-ply` | 0.5, 12 | Share of games that start from a random opening, and that opening's maximum length. |
| `--value-target-q-weight` | 1.0 | Value target = `(1 - w) * game result + w * root search value`. |
| `--replay-gens` | 8 | Train on the games of this many recent generations. |
| `--train-max-epochs`, `--train-patience` | 20, 3 | Per-generation training length and early-stopping patience. |
| `--bf16` | off | BF16 autocast for self-play inference and Lightning training, including validation. Parameters stay FP32. Published-model dev evaluation stays FP32. |
| `--compile` | off | `torch.compile` of the self-play forward with dynamic batches. Falls back to eager on setup or execution failure, with a warning. Compilation stays outside the saved model. |
| `--train-steps N` | unset | Exactly N optimizer steps with full minibatches sampled with replacement from the augmented training split. Validates at the end and publishes the final model. Ignores epoch caps and patience. N must be positive. |
| `--n-residual-blocks`, `--conv-filter-size` | 4, 64 | Network size. |
| `--head` | `fc` | Policy and value heads: `fc` uses the existing fully connected heads; `conv` uses AlphaZero-style 1×1 convolutional heads. |
| `--value-hidden` | 256 | Hidden value layer width for `--head conv`. Must be positive. Ignored with `--head fc`. |
| `--n-policy-layers`, `--n-value-layers` | 4, 2 | Fully connected head depths. Both are ignored with `--head conv`. |
| `--eval-mcts-iterations` | 64 256 | MCTS budgets scored on the dev suite each generation. |
| `--device` | auto | `cuda`, then `mps`, then `cpu`. Lightning trains on the same device. |

`ceiling` accepts the same head options and records them in `report.json` under `net`.
`train` records `head` and `value_hidden` as flat generation metadata fields.
Old configurations and metadata default to `fc` and a value width of 256. Old `model.pkl`
files retain their fully connected heads.

The convolutional policy head has two channels, batch normalization, ReLU, flattening,
and a linear layer that produces seven log probabilities. The convolutional value head
has one channel, batch normalization, ReLU, flattening, and two linear layers with ReLU
between them. Tanh produces the same `q_penalty` and `q_no_penalty` outputs as the FC head.
`--value-hidden` sets the width between those value layers. Legal-move masking stays in
the existing search and evaluation code.

Head options must match the saved model when resuming a run. A different head architecture
or convolutional value width requires a fresh `--base-dir`. Component tests and CPU smoke
runs establish compatibility and training behavior. The effect on solver-optimal play
requires convergence experiments at MCTS-256 on dev.

## Benchmarking

Build with `mise run build`. `scripts/bench_generation.py` appends exactly one generation to
+a copy of a completed run, inheriting its configuration. `--phase all` measures normal
+self-play. `--phase training` reuses the latest games and trains from the latest model,
+with older ancestors included once. Both modes evaluate and publish; use copies only.
+
+The last stdout line is JSON, with synchronized CUDA wall times for replay loading, sample
+conversion and device transfer, self-play, training, evaluation and publishing. It includes
+epochs, optimizer steps, config and validation loss. `total_seconds` excludes Python imports
+and initial device startup. Use an external process timer when startup matters.
+
+Use separate built checkouts at `b5eca94` (timing only, master
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

Training tensors occupy 372 bytes per augmented sample on the training device, in addition
to the model, optimizer and activations. Large replay windows can exhaust device memory.
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
