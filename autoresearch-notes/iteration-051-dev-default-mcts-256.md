# Iteration 051 — Default dev tier to train_mcts=256

## Hypothesis
Iteration 050 showed dev self-play MCTS `256` clearly improves strict top-move agreement over `128`. Dev tier should default to this stronger search budget.

## Change
Planned:
- Change `BENCHMARK_TIERS["dev"].train_mcts` from `128` to `256`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed.
- Full `mise run check` passed.

## Result
Kept. Dev tier now defaults to `train_mcts=256`, matching iteration 050's clear improvement.
