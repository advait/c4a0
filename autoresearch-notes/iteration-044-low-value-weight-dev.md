# Iteration 044 — Dev run with lower value loss weights

## Hypothesis
Policy loss weight `2.0` improved strict top-move agreement. Further reducing value-head loss weights may let the network track MCTS policy targets better without allowing value loss to dominate, while still training values enough for self-play.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-gens 10 \
  --policy-loss-weight 2.0 \
  --q-penalty-loss-weight 0.25 \
  --q-no-penalty-loss-weight 0.25 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active dev incumbent: `0.5683848857879639` from iteration 042 (10 gens, champion selection, policy weight 2.0, value weights 1.0).

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T081103Z-b41c3be0`
- Final strict top-move metric: `0.5744661688804626`
- 95% Wilson CI: `[0.5679007626255045, 0.5810054186141634]`
- Non-terminal scored samples: `21869`
- Unique eval positions: `21755`
- Trained final generation: `10`
- Selected generation: `2`

## Selection summary
- Gen1: `0.5842601656913757`
- Gen2: `0.5845881104469299` (selected)
- Gen3: `0.572510302066803`
- Gen4: `0.5788101553916931`
- Gen5: `0.5743939280509949`
- Gen6: `0.5520548224449158`
- Gen7: `0.5455945730209351`
- Gen8: `0.5140811204910278`
- Gen9: `0.5211377143859863`
- Gen10: `0.5182651877403259`

## Decision
Keep. This beats iteration 042's policy-weight-only incumbent `0.5683848857879639` by about `+0.0061` absolute. The CIs overlap, so this should be treated as a modest dev-scale improvement rather than conclusive final evidence, but it is directionally consistent with emphasizing policy targets and de-emphasizing value-head losses.
