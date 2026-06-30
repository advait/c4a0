# Iteration 047 — Default dev tier to train_gens=2

## Hypothesis
Iteration 046 showed dev `train_gens=2` slightly improved the metric and greatly reduces iteration cost compared with 10-generation ablations. Dev tier should default to 2 generations so the autoresearch loop can test more ideas before spending candidate/gate budget.

## Change
Planned:
- Change `BENCHMARK_TIERS["dev"].train_gens` from `5` to `2`.
- Leave candidate/gate/long as serious longer tiers.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed.
- Full `mise run check` passed.

## Result
Kept. Dev tier now defaults to `train_gens=2`, using the faster strategy validated in iteration 046. Candidate/gate/long remain longer, serious tiers.
