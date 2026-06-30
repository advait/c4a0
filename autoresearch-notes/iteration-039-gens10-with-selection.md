# Iteration 039 — 10 generations with solver-eval champion selection

## Hypothesis
Iteration 037 showed that blindly returning gen10 was worse than gen5. With solver-eval-only champion selection, the same longer training run can return the best generation instead of the latest generation.

## Incumbent
- Current broad fixed-opening dev metric: `0.478190153837204`
- 95% Wilson CI: `[0.47217189052791914, 0.4842147553942748]`

## Command
```sh
uv run python scripts/solver_alignment_eval.py \
  --tier dev \
  --train-gens 10 \
  --select-best-generation-by-solver \
  --solver solver/c4solver \
  --book solver/7x6.book
```

## Result
- Run dir: `autoresearch/eval-runs/solver-alignment-dev-20260630T071201Z-9b7d85a4`
- Final strict top-move metric: `0.5448694229125977`
- 95% Wilson CI: `[0.5381962133377253, 0.5515265566327069]`
- Non-terminal scored samples: `21440`
- Unique eval positions: `21093`
- Trained final generation: `10`
- Selected generation: `3`

## Selection summary
- Gen1: `0.5412115454673767`
- Gen2: `0.5568404197692871`
- Gen3: `0.5603321194648743` (selected)
- Gen4: `0.5453265309333801`
- Gen5: `0.5426385402679443`
- Gen6: `0.5382323861122131`
- Gen7: `0.5116326212882996`
- Gen8: `0.5158407688140869`
- Gen9: `0.5090272426605225`
- Gen10: `0.5086362957954407`

## Decision
Keep. Solver-eval-only champion selection improves the 10-generation experiment from iteration 037's `0.4402356445789337` to `0.5448694229125977`, beating the active 5-generation broad fixed-opening dev incumbent `0.478190153837204`. This confirms that longer training can produce better intermediate models, but latest-generation promotion is unsafe.
