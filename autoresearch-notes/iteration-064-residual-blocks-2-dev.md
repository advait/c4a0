# Iteration 064 — Dev run with n_residual_blocks=2

## Hypothesis
After widening to 32 filters, adding a second residual block may improve policy learning and MCTS guidance further, at a moderate runtime cost.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --n-residual-blocks 2 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active 10k root-position incumbent: `0.6736999750137329` from iteration 062/063 using 32 filters and 1 residual block.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T121552Z-0f8fa399`
- Final strict root top-move metric: `0.597100019454956`
- 95% Wilson CI: `[0.5874512563470274, 0.606674210064358]`
- Scored samples: `10000`
- Scored unique root positions: `3424`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.439453125` on 512 root positions
- Gen2: `0.529296875` on 512 root positions (selected)

## Decision
Discard. This is far below the active 1-block/32-filter incumbent `0.6736999750137329`. The deeper model likely needs different training hyperparameters or more data; do not codify `n_residual_blocks=2`.
