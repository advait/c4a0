# Iteration 031 — Greedy eval temperature dev-tier experiment

## Hypothesis
Strict top-move solver alignment should be measured on deterministic eval trajectories instead of stochastic self-play-style trajectories. Setting eval move temperature to `0.0` may reduce metric noise and better reflect the candidate policy's greedy play.

## Incumbent
- Dev-tier strict top-move metric: `0.652619481086731`
- 95% Wilson CI: `[0.6467263761020621, 0.6584661954139188]`

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --eval-temperature 0.0 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T061729Z-90f12cf9`
- Strict top-move metric: `0.5833333134651184`
- 95% Wilson CI: `[0.5789685345404573, 0.5876850676308673]`
- Non-terminal scored samples: `49152`
- Unique eval positions: `25`

## Decision
Discard. Metric is worse than the dev-tier incumbent `0.652619481086731`, and greedy eval collapses coverage to only 25 unique positions across 2,048 eval games. Deterministic greedy trajectories are therefore not a suitable default broad-alignment benchmark, though the infrastructure remains useful for separate greedy-play diagnostics.
