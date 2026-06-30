# Iteration 041 — Configurable policy/value loss weights

## Hypothesis
Strict solver top-move alignment is a policy objective, but the training loss currently weights policy KL, penalized value MSE, and unpenalized value MSE equally. Configurable loss weights will allow experiments that emphasize policy alignment or de-emphasize noisy value heads.

## Change
Planned:
- Add policy/value loss weights to `ModelConfig` with backward-compatible defaults of 1.0.
- Use weighted sum for total loss.
- Expose weights in solver-alignment eval config/CLI.
- Add tests for weighted-loss composition.

## Verification
- Focused lint/format passed.
- Focused tests passed: `tests/c4a0_tests/nn_test.py`, `tests/c4a0_tests/solver_alignment_eval_test.py`.
- Pyright passed.
- Full `mise run check` passed.

## Result
Kept as infrastructure. This does not change default training behavior because all new weights default to `1.0`, but it enables direct dev-tier experiments such as policy-emphasis and value-head ablations while keeping solver eval-only.
