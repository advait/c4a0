# Iteration 053 — Default dev tier to train_mcts=512

## Hypothesis
Iteration 052 showed dev self-play MCTS `512` clearly improves strict top-move agreement over `256`. Dev tier should use this stronger budget as the default.

## Change
Planned:
- Change `BENCHMARK_TIERS["dev"].train_mcts` from `256` to `512`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed.
- Full `mise run check` passed.

## Result
Kept. Dev tier now defaults to `train_mcts=512`, matching iteration 052's clear improvement.
