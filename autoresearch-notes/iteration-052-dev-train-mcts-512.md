# Iteration 052 — Dev run with train_mcts=512

## Hypothesis
Iteration 050 improved by doubling self-play search from 128 to 256. Another increase to 512 may further improve policy targets, though with diminishing returns and higher runtime.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-mcts 512 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

Uses current defaults: train_gens=2, train_games=1024, champion selection on, policy/value weights `2.0 / 0.25 / 0.25`.

## Baseline
Active dev incumbent: `0.6283280253410339` from iteration 050/051.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T085858Z-3f799821`
- Final strict top-move metric: `0.6458653211593628`
- 95% Wilson CI: `[0.64048944170905, 0.6512045903371123]`
- Non-terminal scored samples: `30607`
- Unique eval positions: `29923`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.6342085599899292`
- Gen2: `0.6398925185203552` (selected)

## Decision
Keep. This beats the active dev incumbent `0.6283280253410339` by about `+0.0175` absolute, with non-overlapping CIs. Stronger self-play search remains beneficial at dev scale.
