# Iteration 054 — Dev run with train_games=2048

## Hypothesis
Iteration 048 showed more self-play games helped substantially. With stronger 512-iteration MCTS targets, increasing from 1024 to 2048 games may further improve coverage and strict top-move agreement.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-games 2048 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

Uses current defaults: train_gens=2, train_mcts=512, champion selection on, policy/value weights `2.0 / 0.25 / 0.25`.

## Baseline
Active dev incumbent: `0.6458653211593628` from iteration 052/053.

## Result
Incomplete under the all-trajectory scoring scope.

## Observations
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T091314Z-71a3b997`
- Training completed for `train_gens=2`, `train_games=2048`, `train_mcts=512`.
- The command timed out during solver-based selection/evaluation before writing `metrics.json`.
- The partial run has saved generation artifacts, but no final comparable metric.
- Follow-up probing showed the solver can hang on some harder midgame trajectory positions from gen2, even on a 128-game selection probe.

## Decision
Do not keep or codify `train_games=2048` yet. Treat this as evidence that scoring all trajectory states is not robust at larger scales. Iteration 055 switches the benchmark to root-only fixed-opening scoring so the solver judges a bounded held-out position suite rather than arbitrary hard midgame positions.
