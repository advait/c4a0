# Iteration 056 — Root-only dev rebaseline

## Hypothesis
After switching the metric scope to first/root fixed-opening positions only, all prior all-trajectory dev metrics are no longer directly comparable. Establish a new dev baseline using current defaults.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --solver solver/c4solver \
  --book solver/7x6.book
```

Current dev defaults:
- train_gens=2
- train_games=1024
- train_mcts=512
- eval_games=2048
- eval_mcts=128
- fixed eval opening depth=6
- root-only scoring
- champion selection on
- loss weights `2.0 / 0.25 / 0.25`

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T113333Z-99bb4a8e`
- Final strict root top-move metric: `0.6025390625`
- 95% Wilson CI: `[0.5811715793078561, 0.6235225983024341]`
- Eval scoring scope: `initial_nonterminal_position`
- Scored samples: `2048`
- Scored unique root positions: `1029`
- Full trajectory non-terminal samples generated but not scored: `30357`
- Full trajectory unique positions: `29793`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.45703125` on 512 root positions
- Gen2: `0.51171875` on 512 root positions (selected)

## Decision
Keep as the new root-only dev baseline. Prior all-trajectory dev metrics are not directly comparable. Active root-only dev incumbent is now `0.6025390625`.
