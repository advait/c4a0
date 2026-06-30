# Iteration 043 — Default solver-alignment policy loss weight to 2.0

## Hypothesis
Iteration 042 showed policy loss weight `2.0` materially improves strict top-move agreement at dev scale. The solver-alignment harness should use it by default while leaving the library `ModelConfig` default at `1.0` for backward compatibility.

## Change
Planned:
- Change `scripts/solver_alignment_eval.py --policy-loss-weight` default from `1.0` to `2.0`.
- Keep `ModelConfig.policy_loss_weight` default at `1.0`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed with default policy loss weight `2.0`.
- Full `mise run check` passed.

## Smoke result
- Strict top-move metric: `0.35166666`
- This is a smoke sanity check only, not an acceptance metric.

## Result
Kept. The solver-alignment harness now defaults to `--policy-loss-weight 2.0`, matching iteration 042's improved dev-scale result. Core `ModelConfig` defaults remain backward-compatible at `1.0`.
