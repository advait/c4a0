# What we learned trying to make self-play converge

This page records every hypothesis tested while driving c4a0's self-play toward solver-perfect play, what each run showed, and what is still open. Read it before starting new convergence work, so you don't repeat a dead end.

## Result

The target, MCTS-256 moves keeping the game result on at least 98% of dev positions, was reached by the final defaults before MCTS-Solver was added: run 2 scored 98.11% at generation 41 (98.0% averaged over its last 10 generations). That generation's model scored **98.68% on the 1,513 held-out test positions**, and generations 38 to 41 all scored between 98.41 and 98.68% on test. The test split played no part in any decision before this check. MCTS-Solver, added afterwards, was ahead of those defaults at every matched generation it ran.

## How the experiments were judged

The target is MCTS move selection, not the raw network. The network is only the estimator that guides search.

Every generation is scored on the committed v1 dev suite (`src/c4a0/suite/v1/dev.tsv`). The suite holds about 2,400 positions per split, sampled by random walks across plies 2 to 41 and labeled by Pascal Pons's solver. A separate test split is held back for a final check. Two metrics are recorded per generation in each `metadata.json`:

- **Keeps the result** (`mcts_keeps`). The share of positions where the MCTS-chosen move keeps the game result: a won position stays won and a drawn position stays drawn. Positions where every legal move keeps the result (lost positions) are not counted. This is the convergence target, measured at 256 MCTS iterations.
- **Strict** (`mcts`). The share of positions where MCTS picks a move the solver ranks best, which means the fastest win or the slowest loss. Ties are split evenly. This is stricter than perfect play needs.

Each arm ran twice, because training has no seed control and repeat runs are the noise floor. Run-to-run spread was 0.1 to 0.9 points. An arm counts as a win only when both of its runs beat both control runs. Arms that ran at different speeds are compared on the same generation window, never on "the latest generation". `scripts/converge_report.py` and the per-generation metadata reproduce every number below.

Experiments ran on `alapin` (one RTX 3090) through `scripts/alapin`. A generation of the final configuration takes roughly 5 to 10 minutes when several runs share the GPU.

## Results by hypothesis

| # | Hypothesis | Change tested | Result | Decision |
|---|---|---|---|---|
| 1 | The exploration formula reinforces the network's own guesses, so search cannot correct it. | PUCT (`prior * sqrt(N_parent) / (1 + n)`) instead of `prior * sqrt(ln(N_parent) / (1 + n))`, and Q as a true mean instead of `sum / (visits + 1)`. | The original search **collapsed** during training: MCTS-64 strict fell from 76 to 78% at generation 0 to 55 to 59% at generation 20. PUCT held 80 to 81%. With a fixed estimator the two searches score within 1 point, so the bug only bites once self-play makes the network confidently wrong. | Kept. PUCT is the only search. |
| 2 | Self-play explores too little and overfits each generation's games. | Root Dirichlet noise (alpha 1.0, 25% mix). A replay window of 4 generations. | Each one added about 3 points over PUCT alone (generations 11 to 20) and stopped PUCT's slow decline. | Kept both. |
| 3 | Playing late moves greedily gives cleaner value targets. | Greedy move choice after ply 10. | Cost 5 points. Less varied games train worse. | Removed. |
| 4 | Training the policy toward a uniform target on finished games adds noise. | Mask terminal positions out of the policy loss. | Cost 1.5 points. | Removed. |
| 5 | More data per generation or a longer window helps. | 2,048 games per generation. A replay window of 8. 512 instead of 256 MCTS iterations. | 2,048 games added about 1.5 points, and a window of 8 about 1 point. Doubling the MCTS budget added nothing. | Kept 2,048 games and window 8. |
| 6 | Training longer is enough. | 60 generations of the best configuration so far. | Climbed slowly, then **plateaued at 96.7%** (keeps the result) from generation 40 on. | Not enough alone. |
| 7 | Opening value targets are poisoned by noisy early play. | Blend the search's root Q into value targets (weights 0.5 and 1.0). Temperature 1 from move 0. | Search-value targets at weight 1.0 beat both control runs by about 0.6 points (1.5 points strict). Temperature 1 gained 0.35, inside the spread. | Kept search-value targets. Temperature schedule left as is. |
| 8 | The network lacks capacity for the openings it does see. | A 4x64 network instead of 1x32. | About 1.2 points ahead at matched generations, in both runs. Also ahead per hour, despite running at half the speed. | Kept. |
| 9 | Self-play never visits the middle-game positions the suite asks about. | Start half of self-play games from random openings of up to 12 or 20 moves. | Both lengths were 0.3 to 0.5 points ahead at generations 9 to 18. By generations 21 to 30, up to 12 moves stayed 0.2 points ahead (96.2 to 96.3% against 96.0%), while up to 20 moves fell 0.2 to 0.3 points behind. | Kept, up to 12 moves. |
| 10 | Training to convergence every generation wastes time and overfits. | 20 epochs with patience 3, and 5 with patience 2, instead of 100 with patience 10. | 20/3 scored the same as 100/10 and finishes each generation faster. 5/2 was noisy and no better. | Kept 20/3 for speed. |
| 11 | The search treats exact terminal values as noisy averages, so it wastes budget on decided lines and blurs their targets. | MCTS-Solver proven-outcome backup: proofs from the rules, no descent below a proven node, a proven root stops and puts all mass on its proven-best moves with an exact Q. | With a fixed uniform estimator, keeps the result rose from 82.6 to 88.5% at MCTS-64 and from 86.2 to 90.1% at MCTS-256. Strict rose from 76.9 to 84.1% and from 81.6 to 86.8%. Every ply bucket below 30 gained at both budgets. The 1x8 network from a 4-game smoke run rose from 88.5 to 90.9% at MCTS-256. The estimator and search are deterministic, so each number is one exact run. | Kept. In self-play training on the final defaults it was ahead of the defaults-only control in both runs at generations 0 to 9 (92.9 and 93.2% against 91.5 and 92.8%) and 10 to 19 (96.2 and 96.1% against 95.6 and 95.9%). It was stopped at generation 20 when the target was reached, so later generations are untested. |
| 12 | Removing Python sample construction makes more training experiments practical. | Rust bulk conversion, device datasets, inference buffers and one-pass replay concatenation. CPU-only performance check on 2026-11-08. | For 699,448 augmented samples, conversion and loader construction fell from median 6.16 s to 0.040 s, five alternating runs per arm, with byte-equal tensors. Tiny CPU generation timings overlap; no end-to-end speedup is established. [All measurements](data/generation-speed-cpu.json). | Default-on changes match two master golden generations, including losses and tensors. GPU timing remains with the coordinator. BF16, compile and fixed-step options remain off; convergence effects are untested. |

### Diagnostics that shaped the order

- **Search depth does not fix the openings.** With a fixed, solver-trained network, 16,384 MCTS iterations still threw away the result in about 13% of positions at plies 0 to 5. The opening is decided too far ahead for any affordable search.
- **The network sees the openings.** Every dev position at plies 0 to 5 occurs in self-play, so opening errors are a learning problem, not missing data. Only 35% of positions at plies 6 to 11, and none from ply 12 on, occur in self-play.
- **The supervised ceiling is about 92% strict.** Trained directly on 846k solver-labeled positions, the network scores 91.8 to 92.6% strict on held-out positions whether it is 1x32, 4x64 or 8x128. That number measures generalization to unseen positions. It is not a cap on self-play, which visits its own positions repeatedly.
- **The labels are trustworthy.** A clean-room Rust solver (`rust/src/native_solver/`) agrees with Pons on every one of 228,041 compared positions across all plies. Brute-force minimax confirms both on 119,181 late positions, and Pons is self-consistent on 1,058,402 child positions.

## Bugs found along the way

These were real defects, each fixed with a test that fails before the fix:

- The self-play shutdown could deadlock when two workers finished the last two games at once.
- Every generation reused game ids 0 to n-1, so move sampling and root noise replayed the same random numbers each generation.
- The network kept a torchmetrics result attached to its autograd graph, so the best-model deep copy sometimes crashed.
- `train --device cpu` still trained on the GPU, because Lightning used `accelerator="auto"`.
- The original metric scored the parent's self-play games, gave half credit to slower wins, and broke ties by column order.

## What is still open

The final configuration is the default for `mise run train`. These are the most promising next steps, in rough order:

1. **Draws.** Drawn positions miss about 13% of the time, against 2.6% for won positions. A value head trained mostly on decisive games may not separate "drawn" from "slightly winning".
2. **Plies 6 to 17.** Most remaining misses are here. Random starts helped a little. A curriculum that weights starts toward these plies, or longer starts, is untested.
3. **MCTS-Solver past generation 20.** It led the control through generation 20 and is now the default, but nobody has run it to 60 generations.
4. **Larger networks over longer runs.** The 4x64 network was still climbing when measured. An 8x128 run started too late to report here.
5. **Faster generations.** Self-play is limited by Python dispatch and CPU-side MCTS (GPU about 20% busy). `torch.compile` gave 8 to 38% on the forward pass in a micro-benchmark and is not wired in.
6. **Drop Pons.** The Rust solver in `rust/src/native_solver/` is verified but runs at about 0.66x Pons's throughput and is not wired into labeling.
