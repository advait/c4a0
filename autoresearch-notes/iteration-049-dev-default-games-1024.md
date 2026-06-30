# Iteration 049 — Default dev tier to train_games=1024

## Hypothesis
Iteration 048 showed increasing dev self-play games from 512 to 1024 produced a clear strict top-move improvement. Dev tier should default to 1024 games despite the higher cost because it remains practical and produces a much more credible signal.

## Change
Planned:
- Change `BENCHMARK_TIERS["dev"].train_games` from `512` to `1024`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed.
- Full `mise run check` passed.

## Result
Kept. Dev tier now defaults to `train_games=1024`, matching the clear improvement from iteration 048.
