# Development Guide

This project has two main parts:

- `rust/`: Connect Four rules, MCTS, self-play, position sampling, solver interface, and TUI exposed to Python through PyO3.
- `src/c4a0/`: PyTorch/PyTorch Lightning model, training loop, sweeps, and CLI commands.

The CLI entrypoint is `src/c4a0/main.py`.

## Tooling model

Use [`mise`](https://mise.jdx.dev/) as the project entrypoint. `mise.toml` pins the required tools:

- `uv` for Python dependency management
- Rust stable, including `cargo`

Install mise once, then from the repo root run:

```sh
mise trust
mise install
```

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
- `eval`: score every published generation's model on a solver suite (backfill, MCTS budgets)
- `build-suite`: rebuild the committed v1 dev/test suites (needs the solver)
- `label`: sample and solver-label positions in a split (train by default)
- `ceiling`: supervised ceiling, training the network directly on solver labels
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
- recorded `evals["v1/dev"]` (raw network on the committed dev suite) for both generations

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
        print(f"gen={gen.gen_n}: root/no games, val_loss={gen.val_loss}, dev_top1={gen.evals['v1/dev'].raw.top1}")
        continue
    lengths = [len(g.samples) for g in games.results]
    scores = [g.player0_score() for g in games.results]
    print(
        f"gen={gen.gen_n}: games={len(games.results)}, "
        f"unique_positions={games.unique_positions()}, "
        f"samples={sum(lengths)}, min_len={min(lengths)}, max_len={max(lengths)}, "
        f"avg_len={sum(lengths)/len(lengths):.2f}, "
        f"p0_score_avg={sum(scores)/len(scores):.3f}, "
        f"val_loss={gen.val_loss}, dev_top1={gen.evals['v1/dev'].raw.top1}"
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

## Solver-optimal evaluation

Every generation is scored when it is published: `TrainingGen.publish` saves `games.pkl` and
`model.pkl`, reloads the model from `model.pkl`, scores it on the committed v1 dev suite, and only
then writes `metadata.json` (atomically). A generation directory without `metadata.json` is
unpublished and loaders skip it. Older runs load with `evals == {}`; their `solver_score` is the
legacy parent self-play score and is never written again.

`gen.evals["v1/dev"]` is a `SuiteEval` (`src/c4a0/evaluation.py`):

- `raw.by_bucket`: per ply bucket, `hits` (credit summed over non-trivial positions) and `n`.
  Credit is `|A & best| / |A|` where `A` is the set of legal moves tied for the policy maximum.
  `raw.top1` is the overall rate.
- `mcts`: the same for root visit policies at each `--eval-mcts-iterations` budget.
- `value`: confusion matrix of solver W/D/L against `q_no_penalty` with cutoffs at -0.5 and 0.5.
- `illegal_argmax`, `model_sha256`, `suite_sha256`, `c_exploration`, `c_ply_penalty`.

Score MCTS budgets on an existing run (idempotent; replaces the stored v1/dev entry):

```sh
mise exec -- uv run python src/c4a0/main.py eval --base-dir training/ci-smoke --mcts-iterations 64
```

The suites are committed, so none of this needs the solver. To rebuild them or label training
positions, build the solver in `solver/`:

```sh
git clone https://github.com/PascalPons/connect4.git solver
cd solver
make
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
cd ..
mise run suite:build   # rewrites src/c4a0/suite/v1/{dev,test}.tsv byte-identically
```

Split membership (train, dev, test) is a salted hash of the mirror-canonical board in
`rust/src/labels.rs`, so labels from `label` are disjoint from both suites by construction.

## Supervised ceiling

Can the network represent solver-perfect play at all? Train it directly on solver labels:

```sh
mise exec -- uv run python src/c4a0/main.py label training/labels/train-s7.tsv --per-ply 24000 --seed 7
mise exec -- uv run python src/c4a0/main.py ceiling training/labels/train-s7.tsv /tmp/c4a0-ceiling/b1f32 --seed 0
```

`ceiling` refuses non-train files, early-stops on dev loss, saves `model.pkl`, and writes
`report.json` (a `CeilingReport`) scored on dev. Pass `--report-test` only after choosing a
configuration on dev. Pass the architecture flags (`--n-residual-blocks`, `--conv-filter-size`,
...) to compare network sizes.

## Verified checks

Verified in this environment:

- `mise run build`: builds and installs the mixed Python/Rust package
- `mise run lint`: Ruff passed
- `mise run typecheck`: Pyright passed with 0 errors
- `mise run test:rust`: 46 passed
- `mise run test:python`: 60 passed (pybridge self-play test can hang on the known shutdown race; exit 124)
- `mise run train:smoke`: runs end-to-end self-play + model training

Current benign warnings observed:

- `tool.uv.dev-dependencies` is deprecated; migrate to `dependency-groups.dev` later.
- pytest-asyncio warns that `asyncio_default_fixture_loop_scope` is unset.
- `Cannot read termcap database; using dumb terminal settings` can appear in non-interactive terminals.
