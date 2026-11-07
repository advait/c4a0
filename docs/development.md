# Development guide

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
mise exec -- uv run --locked python - <<'PY'
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
mise exec -- uv run --locked python src/c4a0/main.py --help
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
uv run --locked python src/c4a0/main.py train \
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

## Self-play experiment controls

Search uses PUCT with true-mean Q. The default `--c-exploration` is 2.0 in `train`,
`play`, and `ceiling`. Generation evaluation reads that constant from metadata.
Tournament callers supply it explicitly; `mcts-sweep` samples it with Optuna.

The following `train` flags are recorded on each generation in `metadata.json`.
Old metadata loads with the defaults below.

| Flag | Default | Metadata field | Behavior when enabled |
|---|---|---|---|
| `--greedy-after-ply K` | unset | `greedy_after_ply` | At ply >= K, choose a most-visited legal move with uniform random ties. Ply counts moves already played; 0 is greedy throughout. |
| `--root-noise-alpha A` | unset | `root_noise_alpha` | Positive finite Dirichlet concentration over legal root moves. Requires a positive noise fraction to take effect. |
| `--root-noise-frac E` | 0 | `root_noise_frac` | Mix `(1-E)*P + E*eta` once per self-play root. E is in [0, 1]; positive E requires alpha. |
| `--mask-terminal-policy` | off | `mask_terminal_policy` | Terminal rows contribute both value losses and zero policy loss. Policy loss averages over the full batch. |
| `--replay-gens G` | 1 | `replay_gens` | Train on new games plus up to G-1 ancestors' games loaded from `games.pkl`, split by whole game. G must be at least 1. |

Without a greedy cutoff, the move temperature remains 4.0 below ply 4, 2.0 below ply 8,
and 1.0 afterwards. Root noise uses a separate RNG stream seeded by game ID and ply.
It never affects evaluation or interactive play, and never changes cached NN estimates.
Terminal masking changes training copies and leaves saved targets intact.
Each generation saves only its new games, so replay does not duplicate past windows.
If a resumed run changes self-play settings, its replay window can include games from
the earlier settings until those generations leave the window.

To check all controls together, run the tiny command above with `--max-gens 2` and add
`--greedy-after-ply 8 --root-noise-alpha 0.3 --root-noise-frac 0.25
--mask-terminal-policy --replay-gens 2`.

To measure convergence, hold the MCTS budget fixed and include
`--eval-mcts-iterations 64`. Judge move selection by MCTS top-1 against the solver.
Raw-network top-1 describes the estimator. Historical MCTS scores used the previous
search, and `eval` backfills now use PUCT with each generation's saved exploration
constant. Label those results before comparing them.

## Inspect self-play stats

Use this after a training run:

```sh
mise exec -- uv run --locked python - <<'PY'
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
mise exec -- uv run --locked python src/c4a0/main.py train --max-gens 10
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
mise exec -- uv run --locked python src/c4a0/main.py play --base-dir training/ci-smoke --model best
```

Other model options:

```sh
mise exec -- uv run --locked python src/c4a0/main.py play --model random
mise exec -- uv run --locked python src/c4a0/main.py play --model uniform
```

This opens a terminal UI, so run it in an interactive terminal.

## TensorBoard / dev server

There is no web app dev server in this repo. The useful local servers are for experiment inspection:

```sh
mise exec -- uv run --locked tensorboard --logdir lightning_logs --port 6006
```

For Optuna sweeps:

```sh
mise exec -- uv run --locked optuna-dashboard sqlite:///optuna.db
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
mise exec -- uv run --locked python src/c4a0/main.py eval --base-dir training/ci-smoke --mcts-iterations 64
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
mise exec -- uv run --locked python src/c4a0/main.py label training/labels/train-s7.tsv --per-ply 24000 --seed 7
mise exec -- uv run --locked python src/c4a0/main.py ceiling training/labels/train-s7.tsv /tmp/c4a0-ceiling/b1f32 --seed 0
```

`ceiling` refuses non-train files, early-stops on dev loss, saves `model.pkl`, and writes
`report.json` (a `CeilingReport`) scored on dev. Pass `--report-test` only after choosing a
configuration on dev. Pass the architecture flags (`--n-residual-blocks`, `--conv-filter-size`,
...) to compare network sizes.

## Python environment

uv owns the interpreter and `.venv`. `project.requires-python` pins Python 3.11.15
for this training application. mise pins uv 0.13.7 and Rust. There is no separate
Python pin in mise or `.python-version`.

Tasks use `uv sync --locked` and `uv run --locked`; a stale lock fails.
Use `mise run install:update` after changing dependencies, then commit `uv.lock`.
Maturin remains the build backend and a development tool for `mise run build`.
Notebook packages are optional and unused by CI. Start Jupyter with
`mise exec -- uv run --locked --group notebook jupyter lab`.

## Typing and lint policy

ty 0.0.99 is the only type checker. It checks `src/` and `tests/` and reads the
Python target from `project.requires-python`. Default diagnostics remain enabled.
Import, reference, attribute, argument, assignment, return, call and override errors
fail the task, as do missing generic arguments and warnings.

Ruff 0.17.4 enables ANN, UP, I, B, F, E and W. ANN401 bans explicit `Any` in
function signatures. Use concrete types, protocols or `object` with narrowing.
Tests follow the same annotation policy. `ruff format --check` runs in `lint`;
E501 is disabled because the formatter owns wrapping. RUF100 and PGH003/PGH004
reject unused or blanket suppressions. Typer option factories are allowed calls
in CLI defaults. Exploratory notebooks are outside the lint and typecheck tasks.

Fix the type or native stub before suppressing an error. A necessary suppression
names one rule and explains why on the same line, for example
`# ty: ignore[invalid-argument-type]  # Upstream stub omits this supported overload.`
Unused, unknown and blanket ty suppressions fail. Legacy type-ignore comments do
not suppress ty diagnostics. Ruff suppressions also name a rule and include a reason.
There are no type suppressions in the package or tests after this migration.

The public native API lives in `src/c4a0_rust/__init__.pyi`. `_native.pyi` re-exports
that API for the compiled extension. `GenerationTrainer` declares Lightning's
extra generation field, and `SampleDataset` supplies the nominal Dataset interface.

The former checker's unused-expression diagnostic has no ty equivalent in the
migration probes; Ruff B018 covers useless expressions. Both checkers rejected
missing imports and references, incorrect native arguments, invalid returns,
invariant generic assignments and variadic tuple mismatches. The former basic
configuration reported no errors on the original project. This comparison does
not establish parity for every Python typing feature. Third-party `Any` can still
propagate through untyped library APIs.

Upstream references: [uv project sync](https://docs.astral.sh/uv/concepts/projects/sync/),
[ty configuration](https://docs.astral.sh/ty/reference/configuration/),
[Ruff configuration](https://docs.astral.sh/ruff/configuration/), and
[ANN401](https://docs.astral.sh/ruff/rules/any-type/).
`managed = true` was removed because it is uv's default, not because it is deprecated.

## Verified checks

Verified in this environment:

- `mise run build`: builds and installs the mixed Python/Rust package
- `mise run lint`: Ruff lint and formatting passed
- `mise run typecheck`: ty passed with 0 errors
- `mise run test:rust`: 48 passed
- `mise run test:python`: 60 passed, 4 skipped (pybridge self-play test can hang on the known shutdown race; exit 124)
- `mise run train:smoke`: runs end-to-end self-play + model training

The tooling migration passed `mise run check` and `mise run ci` under
`taskset -c 0 timeout 900` to avoid the known concurrent self-play shutdown race.
Two unrestricted check attempts stalled at self-play and were stopped with SIGTERM.
Single-CPU validation exercises every test but does not validate concurrent shutdown.

Current benign warnings observed:

- Lightning uses a deprecated PyTorch `LeafSpec` check.
- Tiny training runs have fewer batches than Lightning's logging interval.
- `Cannot read termcap database; using dumb terminal settings` can appear in non-interactive terminals.
