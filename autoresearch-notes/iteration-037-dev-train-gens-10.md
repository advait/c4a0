# Iteration 037 — Dev tier with 10 training generations

## Hypothesis
The most direct convergence lever is more self-play/training generations. Increasing dev tier from 5 to 10 generations should improve strict top-move agreement on the broad fixed-opening eval suite if the current pipeline is still learning.

## Incumbent
- Current broad fixed-opening dev metric: `0.478190153837204`
- 95% Wilson CI: `[0.47217189052791914, 0.4842147553942748]`

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-gens 10 \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T064117Z-8e5664ad`
- Strict top-move metric: `0.4402356445789337`
- 95% Wilson CI: `[0.43426554873862083, 0.4462230786124048]`
- Non-terminal scored samples: `26479`
- Unique eval positions: `22360`
- Final generation: `10`
- Final validation loss: `1.1111961603164673`

## Decision
Discard. More generations improved validation loss but hurt strict solver top-move alignment vs the current broad fixed-opening dev incumbent `0.478190153837204`. This is strong evidence that validation loss alone is not a safe promotion criterion and that promotion gating is high priority.
