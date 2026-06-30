# Iteration 067 — Dev run with learning_rate=2e-3

## Hypothesis
`1e-3` improved over `5e-4`. A further increase to `2e-3` may improve within-generation policy fitting, but may also destabilize training.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --learning-rate 2e-3 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active 10k root-position incumbent: `0.6872000098228455` from iteration 065/066.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T130633Z-feb1bd47`
- Final strict root top-move metric: `0.7109000086784363`
- 95% Wilson CI: `[0.701934972579941, 0.7197030742575151]`
- Scored samples: `10000`
- Scored unique root positions: `3424`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.5546875` on 512 root positions
- Gen2: `0.662109375` on 512 root positions (selected)

## Decision
Keep. This beats the active `1e-3` incumbent `0.6872000098228455` by about `+0.0237` absolute with non-overlapping CIs. Higher LR substantially improves the current 2-generation, 32-filter setup.
