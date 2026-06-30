# Iteration 028 — Replay buffer v1 infrastructure

## Hypothesis
Training only on the latest generation's games is noisy and forgetful. A rolling replay window should improve convergence once enabled by letting each generation train on fresh self-play plus recent historical self-play.

## Change
Planned:
- Add opt-in `replay_window` to `training_loop` / `train_single_gen`.
- Preserve default `replay_window=1` for existing behavior.
- Combine current self-play games with recent historical `games.pkl` files only; solver outputs remain excluded.
- Record replay metadata on `TrainingGen`.
- Expose `--replay-window` in the solver-alignment eval script so future iterations can test larger windows.

## Solver safety invariant
Replay uses only stored self-play `games.pkl` samples. Solver scores/cache/results remain evaluation-only and are not replay inputs.

## Verification
- Focused replay/solver-alignment tests passed.
- `mise run check` passed.

## Result
Kept as infrastructure. Default behavior remains unchanged (`replay_window=1`), but solver-alignment evals can now test larger replay windows with `--replay-window` and training metadata records replay usage.
