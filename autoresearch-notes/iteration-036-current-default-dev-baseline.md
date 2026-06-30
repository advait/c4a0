# Iteration 036 — Current default dev-tier baseline

## Purpose
After adopting fixed openings by default and adding benchmark seed metadata, re-run the default dev tier to establish the active incumbent for subsequent iterations.

## Command
```sh
mise run solver:eval:dev
```

## Expected config highlights
- `tier=dev`
- `eval_opening_depth=6`
- `seed=1337`
- `replay_window=1`

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T063455Z-30f6b902`
- Strict top-move metric: `0.478190153837204`
- 95% Wilson CI: `[0.47217189052791914, 0.4842147553942748]`
- Non-terminal scored samples: `26433`
- Unique eval positions: `24363`
- Final generation: `5`
- Final validation loss: `1.842969298362732`
- Runtime: about 340 seconds.

## Decision
Use this as the active dev-tier broad fixed-opening incumbent for subsequent experiments unless a candidate/gate tier supersedes it.
