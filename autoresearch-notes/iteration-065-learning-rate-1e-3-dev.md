# Iteration 065 — Dev run with learning_rate=1e-3

## Hypothesis
The wider 32-filter model may train too slowly at `5e-4`. A higher learning rate (`1e-3`) might improve policy fit within the 2-generation budget.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --learning-rate 1e-3 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active 10k root-position incumbent: `0.6736999750137329` from iteration 062/063.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T123145Z-099aa3d6`
- Final strict root top-move metric: `0.6872000098228455`
- 95% Wilson CI: `[0.6780425333689134, 0.6962137172792523]`
- Scored samples: `10000`
- Scored unique root positions: `3424`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.49609375` on 512 root positions
- Gen2: `0.58984375` on 512 root positions (selected)

## Decision
Keep. This beats the active `5e-4` incumbent `0.6736999750137329` by about `+0.0135` absolute. CIs narrowly overlap, but the effect is directionally positive and larger than the conv-width gain.
