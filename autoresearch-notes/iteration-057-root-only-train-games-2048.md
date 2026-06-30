# Iteration 057 — Root-only dev run with train_games=2048

## Hypothesis
The earlier 2048-game run timed out only because all-trajectory solver scoring hit hard midgame positions. Under root-only fixed-opening scoring, the same larger self-play budget should be evaluable and may improve the new root-only dev metric.

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-games 2048 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Baseline
Active root-only dev incumbent: `0.6025390625` from iteration 056.

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T113835Z-136b2070`
- Final strict root top-move metric: `0.61328125`
- 95% Wilson CI: `[0.5919961998192047, 0.6341421297312111]`
- Eval scoring scope: `initial_nonterminal_position`
- Scored samples: `2048`
- Scored unique root positions: `1029`
- Full trajectory non-terminal samples generated but not scored: `34329`
- Trained final generation: `2`
- Selected generation: `2`

## Selection summary
- Gen1: `0.54296875` on 512 root positions
- Gen2: `0.56640625` on 512 root positions (selected)

## Decision
Keep as a modest root-only dev improvement. This beats the new root-only dev baseline `0.6025390625` by about `+0.0107`, but CIs overlap. More self-play remains directionally positive, and root-only scoring avoids the earlier all-trajectory solver hang.
