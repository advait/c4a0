# Iteration 027 — Dev-tier scaled baseline

## Hypothesis
The previous smoke-scale metrics are not credible convergence signals. The first post-infrastructure baseline should use a tier with thousands of evaluation games and materially larger self-play/search than smoke.

## Command
```sh
mise run solver:eval:dev
```

## Tier scale
- Train generations: 5
- Self-play games/gen: 512
- Self-play MCTS iterations: 128
- Eval games: 2,048
- Eval MCTS iterations: 128

## Solver safety invariant
Solver remains evaluation-only. Training receives `solver_config=None`; solver output is used only after self-play training for strict top-move scoring.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T060046Z-211700e0`
- Strict top-move metric: `0.652619481086731`
- 95% Wilson CI: `[0.6467263761020621, 0.6584661954139188]`
- Non-terminal scored samples: `25272`
- Unique eval positions: `9766`
- Final generation: `5`
- Final validation loss: `1.743735671043396`
- Runtime: about 157 seconds.

## Interpretation
This is the first credible incumbent for the resumed loop. It is still much smaller than `candidate/gate/long`, but it already uses thousands of eval games and >25k non-terminal scored samples, so it should be far less noisy than the old tiny loop.
