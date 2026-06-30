# Iteration 059 — Rescore 2048-game dev model on 10k root positions

## Hypothesis
The 2048-game root-only result from iteration 057 improved modestly but had a wide 2048-sample CI. Re-scoring the saved model on 10,000 root positions should give a more precise acceptance signal without retraining.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --training-dir autoresearch/eval-runs/solver-alignment-dev-20260630T113835Z-136b2070/training \
  --train-games 2048 \
  --eval-games 10000 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Root-only dev baseline from iteration 056: `0.6025390625` on 2048 scored root positions.
Iteration 057 2048-game result: `0.61328125` on 2048 scored root positions.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T114952Z-d3beebdb`
- Source training dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T113835Z-136b2070/training`
- Strict root top-move metric: `0.66839998960495`
- 95% Wilson CI: `[0.659109592074837, 0.6775610564918023]`
- Scored samples: `10000`
- Scored unique root positions: `3424`
- Selected generation: `2`

## Selection summary
- Gen1: `0.54296875` on 512 root positions
- Gen2: `0.56640625` on 512 root positions (selected)

## Decision
Keep as a precise eval-only result for the 2048-game model. The absolute value is not directly comparable to the 2048-sample dev default because `eval_games=10000` covers a broader root-position suite; iteration 060 provides the apples-to-apples 1024-game comparison.
