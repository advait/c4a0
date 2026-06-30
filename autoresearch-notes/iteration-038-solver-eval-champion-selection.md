# Iteration 038 — Solver-eval-only champion selection

## Hypothesis
Iteration 037 showed later generations can improve validation loss while degrading strict solver alignment. A solver-eval-only champion selection step after training can select the best generation for final evaluation without allowing solver outputs into training samples or losses.

## Change
Planned:
- Add optional `--select-best-generation-by-solver` to `scripts/solver_alignment_eval.py`.
- Score each trained generation on a smaller fixed-opening solver-eval suite.
- Use the selected generation for the final benchmark eval.
- Persist selection scores in `metrics.json`.

## Solver safety invariant
The solver is used only after self-play training for model selection/evaluation. Solver outputs do not enter `Sample`, `games.pkl`, replay buffers, or loss targets.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke selection run passed.
- Lint/typecheck/Rust phases passed.
- Full `mise run check` timed out during Python tests; direct `uv run pytest -q` rerun passed (13 passed).

## Smoke selection result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T065026Z-9f3af0b0`
- Final strict top-move metric: `0.3664208650588989`
- Selection rows: gen1 metric `0.3106796145439148` on 16 selection games.
- Trained final generation: `1`
- Selected generation: `1`

## Result
Kept as infrastructure. This enables future longer runs to select the best solver-judged generation for final eval without using solver outputs as training data.
