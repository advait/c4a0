# Iteration 060 — Rescore 1024-game dev baseline on 10k root positions

## Hypothesis
Iteration 059's 10k root-position metric needs an apples-to-apples comparison against the current 1024-game dev default model. Re-scoring the iteration 056 training dir at `eval_games=10000` will show whether `train_games=2048` is truly better under the same evaluation suite size.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --training-dir autoresearch/eval-runs/solver-alignment-dev-20260630T113333Z-99bb4a8e/training \
  --eval-games 10000 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Iteration 059 2048-game model on 10k roots: `0.66839998960495` with CI `[0.659109592074837, 0.6775610564918023]`.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T115308Z-431b2c09`
- Source training dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T113333Z-99bb4a8e/training`
- Strict root top-move metric: `0.6455000042915344`
- 95% Wilson CI: `[0.6360700527141468, 0.6548182123397591]`
- Scored samples: `10000`
- Scored unique root positions: `3424`
- Selected generation: `2`

## Selection summary
- Gen1: `0.45703125` on 512 root positions
- Gen2: `0.51171875` on 512 root positions (selected)

## Decision
Keep the comparison. On the same 10k root-position eval size, the 2048-game model from iteration 059 scores `0.66839998960495`, beating this 1024-game baseline by about `+0.0229` absolute with non-overlapping CIs. This supports codifying `train_games=2048` for dev/candidate-style runs.
