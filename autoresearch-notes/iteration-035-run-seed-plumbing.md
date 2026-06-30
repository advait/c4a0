# Iteration 035 — Python-side benchmark seed plumbing

## Hypothesis
For large-scale autoresearch, benchmark runs must record and control Python/Torch seeds at minimum. Full Rust sampling determinism will require deeper RNG plumbing, but Python-side seeding and metadata are a low-risk first step.

## Change
Planned:
- Add `--seed` to `scripts/solver_alignment_eval.py`.
- Seed PyTorch/Lightning before training/eval orchestration.
- Store seed in config/metrics.

## Limitation
Rust MCTS move sampling still uses its existing RNG path, so this is not full end-to-end determinism yet.

## Verification
- Focused solver-alignment tests passed.
- Pyright passed.
- Smoke tier run passed.

## Smoke result
- Run dir: `autoresearch/eval-runs/solver-alignment-smoke-20260630T063310Z-0042a312`
- Strict top-move metric: `0.3664208650588989`
- 95% Wilson CI: `[0.3442426869134011, 0.389179897523406]`
- Non-terminal scored samples: `1763`
- Unique eval positions: `1777`
- Seed: `1337`

## Result
Kept as infrastructure. This records/controls Python and Torch randomness; full Rust RNG determinism remains a future task.
