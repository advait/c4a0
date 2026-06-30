# Iteration 034 — Default fixed openings for dev+ tiers

## Hypothesis
Because broad state-space coverage is more important than an easier no-opening eval distribution, dev/candidate/gate/long tiers should default to deterministic opening prefixes. Smoke can stay shallow/cheap.

## Change
- Add `eval_opening_depth` to benchmark tier presets.
- `smoke`: depth 2 for quick coverage sanity.
- `dev`, `candidate`, `gate`, `long`: depth 6 for broad opening coverage.
- Scalar `--eval-opening-depth` remains an override.

## Evidence
Iteration 033 showed depth 6 dev eval covered `22582` unique positions vs `9766` for no fixed openings.

## Verification
- Tier-default test passed.
- Pyright passed.
- Smoke tier run passed with default opening depth 2.

## Smoke result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T063036Z-a0160ca4`
- Strict top-move metric: `0.2522321343421936`
- 95% Wilson CI: `[0.23266894377725278, 0.2728553183139475]`
- Non-terminal scored samples: `1792`
- Unique eval positions: `1800`
- Eval opening depth: `2`

## Result
Kept. Broad fixed openings are now the default for dev+ tiers, with smoke using a cheaper depth-2 coverage sanity check.
