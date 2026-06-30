# Iteration 040 — Default champion selection for dev+ tiers

## Hypothesis
Since iteration 039 showed solver-eval-only champion selection materially improves longer runs and prevents latest-generation regression, dev/candidate/gate/long benchmark tiers should enable it by default. Smoke remains simple/cheap.

## Change
Planned:
- Add champion-selection defaults to `BenchmarkTier`.
- `smoke`: selection off.
- `dev`: selection on with 512 games / 128 MCTS.
- `candidate`: selection on with 2,048 games / 256 MCTS.
- `gate`: selection on with 5,000 games / 512 MCTS.
- `long`: selection on with 10,000 games / 800 MCTS.
- Add `--no-select-best-generation-by-solver` escape hatch.

## Verification
- Config tests passed.
- Pyright passed.
- Smoke tier run passed with selection disabled by tier default.
- Lint/typecheck/Rust phases passed.
- Full `mise run check` timed out during Python tests; direct `uv run pytest -q` rerun passed (13 passed).

## Smoke result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T072732Z-c698707b`
- Strict top-move metric: `0.35758253931999207`
- Selection enabled: `false`
- Selection rows: `0`

## Result
Kept. Dev/candidate/gate/long now default to solver-eval-only champion selection, with `--no-select-best-generation-by-solver` available for ablations.
