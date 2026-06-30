# Iteration 055 — Root-only fixed-opening eval scope

## Hypothesis
Scoring every non-terminal trajectory state can hit very hard midgame positions and wedge Pascal Pons solver evaluation. For held-out fixed-opening evaluation, scoring only the first/root MCTS policy from each eval game should preserve a clean fixed-position metric, avoid hard trajectory positions, and make serious evals more robust.

## Change
Planned:
- Add Rust/PyO3 methods for first non-terminal sample scoring/counts.
- Add solver-alignment config/CLI `--score-initial-positions-only` / `--no-score-initial-positions-only`.
- Default the harness to root-only scoring.
- Report scored sample/unique-position counts according to the chosen scope.

## Verification
- Rust/PyO3 build passed.
- Focused PyBridge and solver-alignment tests passed.
- Pyright passed.
- Rust tests passed.
- Smoke solver eval passed under root-only scoring.
- Full `mise run check` passed.

## Smoke result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T113225Z-934e80a1`
- Strict root top-move metric: `0.2109375`
- Eval scoring scope: `initial_nonterminal_position`
- Scored samples: `128`
- Scored unique positions: `49`
- Full trajectory non-terminal samples still recorded: `1794`

## Decision
Keep as a benchmark-safety and metric-clarity improvement. Metrics after this point are root-only fixed-opening strict top-move agreement and must be rebaselined; prior all-trajectory dev metrics are not directly comparable.
