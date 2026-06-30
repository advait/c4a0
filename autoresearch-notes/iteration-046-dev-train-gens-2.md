# Iteration 046 — Dev run with train_gens=2

## Hypothesis
Recent 10-generation runs selected generation 2, while later generations regressed and had lower unique-position diversity. If generation 2 consistently captures the best dev-scale policy, `--train-gens 2` may provide nearly the same signal at much lower iteration cost.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-gens 2 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

Uses current defaults: champion selection on, policy/value loss weights `2.0 / 0.25 / 0.25`.

## Baseline
Active dev incumbent: `0.5744661688804626` from iteration 044 (10 gens, selected gen2).

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T082833Z-cf0c54a0`
- Final strict top-move metric: `0.578309953212738`
- 95% Wilson CI: `[0.571765970954719, 0.5848265477584885]`
- Non-terminal scored samples: `21964`
- Unique eval positions: `22006`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.5642974972724915`
- Gen2: `0.5869642496109009` (selected)

## Decision
Keep for dev iteration strategy. This beats iteration 044's 10-generation incumbent `0.5744661688804626` by about `+0.0038` absolute while costing far less. The CIs overlap, so the main win is iteration speed and avoiding late-generation collapse, not a definitive metric jump.
