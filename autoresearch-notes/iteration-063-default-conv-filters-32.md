# Iteration 063 — Default solver-alignment conv_filter_size to 32

## Hypothesis
Iteration 062 showed a modest root-only dev improvement from widening the network to 32 convolution filters. The solver-alignment harness should default to this capacity while leaving callers able to override.

## Change
Planned:
- Change `scripts/solver_alignment_eval.py --conv-filter-size` default from `16` to `32`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke eval passed.
- Full `mise run check` passed.

## Result
Kept. The solver-alignment harness now defaults to `conv_filter_size=32`, matching iteration 062's modest dev-scale improvement.
