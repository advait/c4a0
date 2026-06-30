# Iteration 048 — Dev run with train_games=1024

## Hypothesis
With the best dev strategy now converging in 2 generations, increasing self-play games from 512 to 1024 may improve policy target quality and board-position coverage without the late-generation collapse seen in longer runs.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-games 1024 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

Uses current defaults: train_gens=2, train_mcts=128, champion selection on, policy/value weights `2.0 / 0.25 / 0.25`.

## Baseline
Active dev incumbent: `0.578309953212738` from iteration 046/047.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T083555Z-e5172fc0`
- Final strict top-move metric: `0.6151528358459473`
- 95% Wilson CI: `[0.6093563872256373, 0.6209167820817258]`
- Non-terminal scored samples: `27216`
- Unique eval positions: `27060`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.579236626625061`
- Gen2: `0.6168879866600037` (selected)

## Decision
Keep. This beats the active dev incumbent `0.578309953212738` by about `+0.0368` absolute with non-overlapping CIs. More self-play data at the stable 2-generation setting is a clear improvement at dev scale.
