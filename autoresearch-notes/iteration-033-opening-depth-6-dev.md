# Iteration 033 — Opening-depth 6 dev-tier baseline

## Hypothesis
A deterministic opening-prefix suite of depth 6 should massively improve eval coverage and produce a broader, more stable benchmark distribution than sampling all eval trajectories only from the candidate model's stochastic self-play.

## Prior dev incumbent caveat
The prior dev incumbent `0.652619481086731` used no fixed openings (`eval_opening_depth=0`). Opening-depth 6 changes the distribution, so the result is a new broad-coverage baseline rather than a directly comparable improvement/regression.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --eval-opening-depth 6 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T062257Z-9ad4bc02`
- Strict top-move metric: `0.45066604018211365`
- 95% Wilson CI: `[0.4446053160110517, 0.45674143955361024]`
- Non-terminal scored samples: `25824`
- Unique eval positions: `22582`
- Eval opening depth: `6`

## Decision
Keep as a new broad-coverage benchmark baseline. The score is lower than the no-opening dev baseline because the distribution is harder/different, but coverage is much better (`22582` unique positions vs `9766`). Future convergence decisions should prefer this broader fixed-opening distribution for dev+ tiers.
