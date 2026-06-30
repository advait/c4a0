# Iteration 050 — Dev run with train_mcts=256

## Hypothesis
Increasing self-play MCTS iterations from 128 to 256 may improve MCTS policy targets and final strict solver top-move agreement now that the run uses enough self-play games and only 2 generations.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-mcts 256 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

Uses current defaults: train_gens=2, train_games=1024, champion selection on, policy/value weights `2.0 / 0.25 / 0.25`.

## Baseline
Active dev incumbent: `0.6151528358459473` from iteration 048/049.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T084657Z-c4b628d0`
- Final strict top-move metric: `0.6283280253410339`
- 95% Wilson CI: `[0.6226562341343489, 0.6339646809780597]`
- Non-terminal scored samples: `28057`
- Unique eval positions: `27721`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.6016360521316528`
- Gen2: `0.6376503705978394` (selected)

## Decision
Keep. This beats the active dev incumbent `0.6151528358459473` by about `+0.0132` absolute, with non-overlapping final CIs. More self-play search improves policy targets at this scale.
