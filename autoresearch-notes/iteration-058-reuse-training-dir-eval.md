# Iteration 058 — Reuse training dir for larger eval-only rescoring

## Hypothesis
After root-only scoring, we should be able to re-score saved self-play runs at larger eval budgets without retraining. A `--training-dir` eval-only mode will make candidate/gate metric checks cheaper and reduce noise while preserving solver-as-judge-only semantics.

## Change
Planned:
- Add `scripts/solver_alignment_eval.py --training-dir`.
- When provided, skip training and symlink the existing training directory into the new eval run directory.
- Load the latest saved generation as the trained final generation, then optionally run solver-only champion selection and final eval as usual.
- Include source training directory in metrics.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke rescoring with `--training-dir` passed.
- Full `mise run check` passed.

## Smoke reuse result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T114849Z-83f8a3dd`
- Source training dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T113225Z-934e80a1/training`
- Strict root top-move metric: `0.2109375`
- Scored samples: `128`

## Decision
Keep as eval infrastructure. This enables larger eval-only rescoring of saved self-play models without allowing solver data into training.
