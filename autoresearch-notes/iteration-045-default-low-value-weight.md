# Iteration 045 — Default solver-alignment value loss weights to 0.25

## Hypothesis
Iteration 044 showed value-head loss weights `0.25` with policy weight `2.0` modestly improve dev strict top-move agreement. The solver-alignment harness should default to this loss mix, while leaving core `ModelConfig` defaults backward-compatible.

## Change
Planned:
- Change `scripts/solver_alignment_eval.py --q-penalty-loss-weight` default from `1.0` to `0.25`.
- Change `scripts/solver_alignment_eval.py --q-no-penalty-loss-weight` default from `1.0` to `0.25`.
- Keep `ModelConfig` defaults unchanged.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed with default loss weights `2.0 / 0.25 / 0.25`.
- Full `mise run check` passed.

## Smoke result
- Strict top-move metric: `0.36505288`
- Smoke is sanity-only.

## Result
Kept. The solver-alignment harness now defaults to policy/value loss weights `2.0 / 0.25 / 0.25`, matching iteration 044's best dev-scale result. Core `ModelConfig` defaults remain `1.0 / 1.0 / 1.0`.
