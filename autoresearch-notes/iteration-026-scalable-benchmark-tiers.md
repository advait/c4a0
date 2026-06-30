# Iteration 026 — Scalable solver-alignment benchmark tiers

## Hypothesis
The previous autoresearch loop optimized a tiny smoke-scale metric. That was too noisy and too small for Connect Four convergence. A tiered benchmark harness should separate quick sanity checks from credible candidate/gate/long convergence measurements.

## Change
- Add `--tier/--benchmark-tier` presets to `scripts/solver_alignment_eval.py`:
  - `smoke`: fast sanity only.
  - `dev`: moderate iteration signal.
  - `candidate`: primary acceptance tier with thousands of self-play games and 10k eval games.
  - `gate`: expensive stability gate.
  - `long`: 50-generation / 100k-eval-game convergence tier.
- Keep scalar args as explicit overrides for controlled ablations.
- Make the default solver cache stable under `autoresearch/solver-cache/` instead of run-local.
- Write richer run metadata: tier, git commit, argv, sample counts, strict metric confidence interval, training dir.
- Add Rust/PyO3 sample count helpers so strict top-move CI uses the number of non-terminal scored samples.
- Add mise tasks for each tier.

## Solver safety invariant
Solver remains evaluation-only. Training still calls `training_loop(..., solver_config=None)` and no solver move/value enters samples, games, replay, or loss targets.

## Expected impact
This does not directly improve model quality; it makes future keep/discard decisions meaningful at the scales the user expects.

## Verification
- Ruff format/check passed for touched Python files.
- Pyright passed.
- Rust tests passed.
- Focused Python tests passed.
- Smoke tier run passed.

## Smoke result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T055916Z-0474a9a0`
- Strict top-move metric: `0.267977237701416`
- 95% Wilson CI: `[0.2487072695399037, 0.28816757637934265]`
- Eval games: `128`
- Non-terminal scored samples: `1933`
- Unique eval positions: `1682`
- Cache: `autoresearch/solver-cache/top-move-solutions.db`
