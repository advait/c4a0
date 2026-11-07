# Development reference

This page describes the code layout, tasks, tooling and training options. For install and usage, see the [README](../README.md). For what the experiments showed, see [what we learned trying to make self-play converge](experiments.md).

## Layout

| Path | Owns |
|---|---|
| `rust/src/c4r.rs` | Bitboard position, move rules, terminal detection, network input encoding. |
| `rust/src/mcts.rs` | PUCT search, root noise, visit-count policies, training samples. |
| `rust/src/self_play.rs` | Multi-threaded self-play, batched network calls, the per-run evaluation cache, random starts. |
| `rust/src/analysis.rs` | Single-threaded, batched MCTS over caller-supplied positions, used by evaluation. |
| `rust/src/labels.rs` | Mirror-canonical position keys, the dev/test/train split, the position sampler. |
| `rust/src/solver.rs` | Strict stdin/stdout protocol for Pascal Pons's solver. |
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
| `train:smoke` | One tiny CPU generation through self-play, training, saving and evaluation. |
| `suite:build` | Rebuilds the committed suites. Needs the solver in `solver/`. |
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
| `--n-residual-blocks`, `--conv-filter-size` | 4, 64 | Network size. |
| `--eval-mcts-iterations` | 64 256 | MCTS budgets scored on the dev suite each generation. |
| `--device` | auto | `cuda`, then `mps`, then `cpu`. Lightning trains on the same device. |

## Generation artifacts

Each generation is a directory under the base directory (default `training/`) named by its creation time:

- `model.pkl` is the trained network.
- `games.pkl` holds the self-play games that trained it.
- `metadata.json` holds its `GenConfig` fields, its parent, its validation loss, and `evals`, the dev-suite scores of this exact `model.pkl`.

`metadata.json` is written last. A directory without one is an incomplete generation and is ignored. Older metadata files load, because unknown fields are ignored and new fields have defaults.
