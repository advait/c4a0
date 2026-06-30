# Iteration 068 — Default solver-alignment learning_rate to 2e-3

## Hypothesis
Iteration 067 showed `learning_rate=2e-3` clearly improves over `1e-3` with non-overlapping 10k-root CIs. The solver-alignment harness should default to this setting.

## Change
Planned:
- Change `scripts/solver_alignment_eval.py --learning-rate` default from `1e-3` to `2e-3`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke eval passed.
- Found intermittent Python test hang in a tiny-MCTS duplicate-root pybridge test; adjusted that split-test fixture to use distinct opening prefixes while preserving the split/non-mutating assertion.
- Full `mise run check` passed.

## Result
Kept. The solver-alignment harness now defaults to `learning_rate=2e-3`, matching iteration 067's non-overlapping dev-scale improvement.
