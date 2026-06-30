# Iteration 062 — Dev run with conv_filter_size=32

## Hypothesis
The current 16-filter network may be under-capacity for matching stronger MCTS root policies. Doubling convolution filters to 32 could improve root top-move agreement at the current 2048-game / 512-MCTS / 10k-root dev scale.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --conv-filter-size 32 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active 10k root-position incumbent: `0.66839998960495` from iteration 059/061 with default 16 filters.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T115800Z-9154e7fc`
- Final strict root top-move metric: `0.6736999750137329`
- 95% Wilson CI: `[0.6644453386576604, 0.6828212103550216]`
- Scored samples: `10000`
- Scored unique root positions: `3424`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.546875` on 512 root positions
- Gen2: `0.630859375` on 512 root positions (selected)

## Decision
Keep as a modest dev-scale improvement. This beats the active 16-filter 10k-root incumbent `0.66839998960495` by about `+0.0053`, but CIs overlap. Directionally, more model capacity helps and selection strongly favored gen2.
