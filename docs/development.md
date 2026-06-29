# Development Guide

This project has two main parts:

- `rust/`: Connect Four rules, MCTS, self-play, solver interface, and TUI exposed to Python through PyO3.
- `src/c4a0/`: PyTorch/PyTorch Lightning model, training loop, sweeps, and CLI commands.

The CLI entrypoint is `src/c4a0/main.py`.

## Tooling model

Use [`mise`](https://mise.jdx.dev/) as the project entrypoint. `mise.toml` pins and bootstraps the required tools:

- `uv` for Python dependency management
- Rust stable, including `cargo`
- `clang`/`libclang` for Rust crates that use bindgen, including RocksDB bindings

Install mise once, then from the repo root run:

```sh
mise trust
mise install
```

`mise.toml` sets `LIBCLANG_PATH`/`LD_LIBRARY_PATH` from the mise-managed clang install, so developers and CI do not need manual clang environment exports.

## Common tasks

List tasks:

```sh
mise tasks
```

Install dependencies and build the editable package:

```sh
mise run install
mise run build
```

Run local validation:

```sh
mise run lint
mise run typecheck
mise run test:rust
mise run test:python
mise run check
```

Run the full CI suite locally, including smoke training:

```sh
mise run ci
```

CI uses the same task:

```sh
mise run ci
```

## Packaging/import check

This is a mixed maturin Python/Rust package. After `mise run build`, both imports should work without `PYTHONPATH` hacks:

```sh
mise exec -- uv run python - <<'PY'
import c4a0
import c4a0_rust

print(c4a0.__file__)
print(c4a0_rust.N_ROWS, c4a0_rust.N_COLS)
PY
```

`c4a0_rust` is a Python package that re-exports the native `c4a0_rust._native` PyO3 extension.

## CLI commands

Show available commands:

```sh
mise exec -- uv run python src/c4a0/main.py --help
```

Current commands:

- `train`: train via self-play
- `play`: open the terminal UI and play against a model/random/uniform player
- `score`: score generated policies with an external Connect Four solver
- `nn-sweep`: Optuna sweep over NN hyperparameters using existing training data
- `mcts-sweep`: Optuna sweep over self-play/MCTS hyperparameters

## Smoke train a model

Run the checked-in smoke task:

```sh
mise run train:smoke
```

The task runs a tiny CPU job equivalent to:

```sh
rm -rf training/ci-smoke
uv run python src/c4a0/main.py train \
  --base-dir training/ci-smoke \
  --device cpu \
  --n-self-play-games 4 \
  --n-mcts-iterations 4 \
  --self-play-batch-size 16 \
  --training-batch-size 16 \
  --n-residual-blocks 1 \
  --conv-filter-size 8 \
  --n-policy-layers 1 \
  --n-value-layers 1 \
  --lr-schedule 0 \
  --lr-schedule 0.001 \
  --l2-reg 0 \
  --max-gens 1
```

Verified smoke runs in this environment:

- generated 4 games
- generated a generation 0 root model and a generation 1 trained model
- produced self-play samples and unique-position counts in the training metadata/artifacts
- saved `metadata.json`, `games.pkl`, and `model.pkl` for each generation
- left `solver_score` as `null` because no external solver was configured

Training artifacts are stored as timestamped generation directories:

```text
training/<run-name>/<timestamp>/metadata.json
training/<run-name>/<timestamp>/games.pkl
training/<run-name>/<timestamp>/model.pkl
```

PyTorch Lightning logs are written to `lightning_logs/`.

## Inspect self-play stats

Use this after a training run:

```sh
mise exec -- uv run python - <<'PY'
from c4a0.training import TrainingGen

base = "training/ci-smoke"
for gen in reversed(TrainingGen.load_all(base)):
    games = gen.get_games(base)
    if games is None:
        print(f"gen={gen.gen_n}: root/no games, val_loss={gen.val_loss}, solver_score={gen.solver_score}")
        continue
    lengths = [len(g.samples) for g in games.results]
    scores = [g.player0_score() for g in games.results]
    print(
        f"gen={gen.gen_n}: games={len(games.results)}, "
        f"unique_positions={games.unique_positions()}, "
        f"samples={sum(lengths)}, min_len={min(lengths)}, max_len={max(lengths)}, "
        f"avg_len={sum(lengths)/len(lengths):.2f}, "
        f"p0_score_avg={sum(scores)/len(scores):.3f}, "
        f"val_loss={gen.val_loss}, solver_score={gen.solver_score}"
    )
PY
```

## Full/default training

The README default is much larger and intended for a GPU-class machine:

```sh
mise exec -- uv run python src/c4a0/main.py train --max-gens 10
```

Useful knobs:

- `--device cpu|cuda|mps`
- `--n-self-play-games`
- `--n-mcts-iterations`
- `--self-play-batch-size`
- `--training-batch-size`
- `--base-dir`
- `--max-gens`

## Play against a model

After training at least one generation:

```sh
mise exec -- uv run python src/c4a0/main.py play --base-dir training/ci-smoke --model best
```

Other model options:

```sh
mise exec -- uv run python src/c4a0/main.py play --model random
mise exec -- uv run python src/c4a0/main.py play --model uniform
```

This opens a terminal UI, so run it in an interactive terminal.

## TensorBoard / dev server

There is no web app dev server in this repo. The useful local servers are for experiment inspection:

```sh
mise exec -- uv run tensorboard --logdir lightning_logs --port 6006
```

For Optuna sweeps:

```sh
mise exec -- uv run optuna-dashboard sqlite:///optuna.db
```

## Optional solver scoring

The solver is optional and is not used for training. It scores generated policies against objective Connect Four solutions.

```sh
git clone https://github.com/PascalPons/connect4.git solver
cd solver
make
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
cd ..
```

Score an existing training directory:

```sh
mise exec -- uv run python src/c4a0/main.py score solver/c4solver solver/7x6.book --base-dir training/ci-smoke
```

Or score during training:

```sh
mise exec -- uv run python src/c4a0/main.py train \
  --solver-path solver/c4solver \
  --book-path solver/7x6.book
```

Scores are cached in `solutions.db` by default.

## Verified checks

Verified in this environment:

- `mise run build`: builds and installs the mixed Python/Rust package
- `mise run lint`: Ruff passed
- `mise run typecheck`: Pyright passed with 0 errors
- `mise run test:rust`: 29 passed
- `mise run test:python`: 4 passed
- `mise run train:smoke`: runs end-to-end self-play + model training

Current benign warnings observed:

- `tool.uv.dev-dependencies` is deprecated; migrate to `dependency-groups.dev` later.
- pytest-asyncio warns that `asyncio_default_fixture_loop_scope` is unset.
- `Cannot read termcap database; using dumb terminal settings` can appear in non-interactive terminals.
