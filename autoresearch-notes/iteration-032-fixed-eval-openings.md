# Iteration 032 — Fixed eval opening suite infrastructure

## Hypothesis
Eval games sampled only from the candidate model's trajectories can under-cover the state space, especially with greedy/deterministic eval. A fixed opening-prefix suite should improve eval coverage and make metrics more comparable across candidates, without using solver labels in training.

## Change
Planned:
- Extend `GameMetadata` with `initial_moves` defaulting to empty.
- Initialize Rust games from `Pos::from_moves(initial_moves)`.
- Add eval-script options to generate deterministic opening prefixes for eval games only.
- Keep training self-play starts unchanged unless explicitly configured elsewhere.

## Solver safety invariant
Opening prefixes are deterministic/generated, not solver-derived. Solver remains eval-only.

## Verification
- Focused Python tests passed.
- Rust tests passed.
- `mise run check` passed.

## Result
Kept as infrastructure. Default behavior remains unchanged (`--eval-opening-depth 0`), and eval runs can now use deterministic opening prefixes up to depth 6 for much broader fixed-suite coverage.
