# Iteration 061 — Default dev tier to train_games=2048 and eval_games=10000

## Hypothesis
Iterations 059–060 showed the 2048-game model beats the 1024-game model on the same 10k root-position eval suite with non-overlapping CIs. Root-only scoring makes 10k eval practical, so dev should default to the more precise 10k root-position eval and the stronger 2048-game training budget.

## Change
Planned:
- Change `BENCHMARK_TIERS["dev"].train_games` from `1024` to `2048`.
- Change `BENCHMARK_TIERS["dev"].eval_games` from `2048` to `10000`.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke eval passed.
- Full `mise run check` passed.

## Result
Kept. Dev now defaults to `train_games=2048` and `eval_games=10000`, matching the non-overlapping 10k root-position comparison from iterations 059–060.
