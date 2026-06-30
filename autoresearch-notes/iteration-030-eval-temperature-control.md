# Iteration 030 — Eval move-temperature control

## Hypothesis
Self-play should remain exploratory, but solver-alignment evaluation should be able to run deterministically. The current Rust `play_games` path hardcodes the same stochastic temperature schedule for all callers, so eval position trajectories are noisy and not explicitly separable from training.

## Change
Planned:
- Add optional `move_temperature` to Rust/PyO3 `play_games`.
- `None` preserves the current hardcoded training/self-play temperature schedule.
- A numeric value overrides every move selection; `0.0` gives argmax deterministic move choice.
- Expose `--eval-temperature` in `scripts/solver_alignment_eval.py` for eval-game generation only.

## Solver safety invariant
No solver data enters training. This only changes how eval games are sampled after training.

## Verification
- Focused Python tests passed.
- Rust tests passed.
- `mise run check` passed.

## Result
Kept as infrastructure. Default behavior remains unchanged (`move_temperature=None` preserves the existing exploratory schedule), and solver-alignment evals can now pass `--eval-temperature 0.0` to sample greedy/deterministic eval trajectories.
