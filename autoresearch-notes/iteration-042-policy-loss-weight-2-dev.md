# Iteration 042 — Dev run with policy loss weight 2.0

## Hypothesis
Strict top-move agreement should benefit from emphasizing the policy KL target relative to value heads.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-gens 10 \
  --policy-loss-weight 2.0 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active broad fixed-opening dev incumbent: `0.5448694229125977` from iteration 039/040 (10 gens with solver-eval champion selection).

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T075413Z-7ea259a4`
- Final strict top-move metric: `0.5683848857879639`
- 95% Wilson CI: `[0.5619330227810347, 0.5748136248517778]`
- Non-terminal scored samples: `22717`
- Unique eval positions: `22812`
- Trained final generation: `10`
- Selected generation: `2`

## Selection summary
- Gen1: `0.5480378866195679`
- Gen2: `0.5633413791656494` (selected)
- Gen3: `0.5583032965660095`
- Gen4: `0.5332682728767395`
- Gen5: `0.5473092794418335`
- Gen6: `0.5015002489089966`
- Gen7: `0.4938792288303375`
- Gen8: `0.47065916657447815`
- Gen9: `0.4838844835758209`
- Gen10: `0.488437682390213`

## Decision
Keep. This beats the active selected-generation dev incumbent `0.5448694229125977` by about `+0.0235` absolute. The final broad fixed-opening dev CI does not overlap the prior run's CI (`[0.5381962133377253, 0.5515265566327069]`), so this is a credible improvement at dev scale.
