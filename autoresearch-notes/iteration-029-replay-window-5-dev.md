# Iteration 029 — Replay window 5 dev-tier experiment

## Hypothesis
Using the last five generations of self-play samples should reduce forgetting/noise and improve strict solver top-move agreement compared with latest-generation-only training.

## Incumbent
- Dev-tier strict top-move metric: `0.652619481086731`
- 95% Wilson CI: `[0.6467263761020621, 0.6584661954139188]`

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --replay-window 5 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T060730Z-8dac8e75`
- Strict top-move metric: `0.5571916103363037`
- 95% Wilson CI: `[0.5518809414949281, 0.5624892363298709]`
- Non-terminal scored samples: `33685`
- Unique eval positions: `24688`
- Final replay window: `5`
- Final replay samples: `44380`

## Decision
Discard. This is substantially worse than the dev-tier incumbent `0.652619481086731`, so replay-window defaults remain `1`. The replay infrastructure remains useful for future variants, but naive rolling replay hurts this short dev-tier configuration.
