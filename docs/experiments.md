# What we learned trying to make self-play converge

This page records every hypothesis tested while driving c4a0's self-play toward solver-perfect play, what each run showed, and what is still open. Read it before starting new convergence work, so you don't repeat a dead end.

## Result

The first target, MCTS-256 moves keeping the game result on at least 98% of dev positions, was reached by the final defaults before MCTS-Solver was added: run 2 scored 98.11% at generation 41 (98.0% averaged over its last 10 generations). That generation's model scored **98.68% on the 1,513 held-out test positions**, and generations 38 to 41 all scored between 98.41 and 98.68% on test. The test split played no part in any decision before this check. MCTS-Solver, added afterwards, was ahead of those defaults at every matched generation it ran.

The second target, optimal play, is close but not reached. The best self-play-only agent (conv 8x64, cosine-annealed self-play, 40 generations, scored with mirror averaging and leaf minimax depth 10) scores:

| Split | Strict, MCTS-256 | Strict, MCTS-2048 | Keeps the result, MCTS-256 | Keeps the result, MCTS-2048 |
|---|---|---|---|---|
| dev | 96.99% | 97.44% | 98.97% | 99.41% |
| test | 96.84% | 97.15% | 99.27% | 99.17% |

Keeping the result reaches 99% on test. Strict optimal-move accuracy sits near 97%, limited by opening positions: 48 of the roughly 57 remaining dev misses are at plies 0 to 11. No solver labels are used in training; the solver only labels the evaluation suites.

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

## Toward optimal play

After keeping the result reached 98%, the goal became optimal play: MCTS-256 must pick a solver-best move (the fastest win, or the slowest loss) on at least 99% of dev positions. This is the **strict** metric (`mcts` in the metadata). The 98% model scored 95.3% strict under the current search. Experiments ran on conv heads after hypothesis 15, with each arm compared to a same-code control at matched generation windows.

| # | Hypothesis | Change tested | Result | Decision |
|---|---|---|---|---|
| 12 | Strict misses pick the wrong win or loss distance because the distance signal (0.02 value per move) is inside the value head's noise. | Ply penalty 0.02 and 0.023, and a multiplicative discount `+-gamma^ply` with gamma 0.97 (`--ply-discount`). | All three trailed the control on strict at generations 10 to 19 (control 94.21 and 94.00%; penalty 0.02 93.22 and 93.08%; discount 93.27 and 93.40%; penalty 0.023 92.95 and 92.78%). | Refuted. Penalty stays 0.01. |
| 13 | Search noise between near-equal moves causes the misses. | Root visit counts on the 96 misses of the 98% model. | The solver-best move got 0.08 to 0.35 times the visits of the chosen move, and the network prior favoured the wrong move 2 to 5 times. Only 7 to 18% of misses were near-ties. | Refuted. The network misranks moves and 256 iterations cannot overturn it. |
| 14 | The network spends its capacity in the wrong place. | Parameter count of the 4x64 network. | 99% of 29.3M parameters sat in fully connected heads (2688x2688 layers); the convolutional trunk had 297k. | Led to hypothesis 15. |
| 15 | AlphaZero-style heads (1x1 convolution, then a small linear layer) lose nothing and train much faster. | `--head conv`: 0.31M parameters at 4x64. | Conv 4x64 plateaued at 95.4% strict, the same as FC heads (95.1 to 95.4%), and ran about twice as many generations per hour. | Kept for speed; the only head now. FC heads and `--head` were removed. |
| 16 | Self-play rarely reaches the middle-game positions where misses cluster. | Random starts for 80% of games instead of 50%. | Matched the control in every window (generations 30 to 39 strict: 94.97 and 95.23% against 95.03 and 95.28%; keeps the result 97.6 to 97.8% against 97.8%). | Rejected. Coverage is not the limit. |
| 17 | A deeper conv trunk ranks opening moves better. | Conv 8x64. | Both runs ahead of both controls by about 0.2 points at generations 10 to 19 (94.16 and 94.20% against 93.89 and 94.11%) and 20 to 29 (95.22 and 95.17% against 94.85 and 95.01%). | Small gain. Kept; 8x64 is the default. |
| 18 | Whole-board context may generalize better than local convolutions. | Global-pooling residual blocks (`resnet_gpool`), a 42-token transformer, and a conv-stem hybrid, on the supervised ceiling. | Global pooling 8x64 scored 95.08% strict, the same as the plain ResNet (95.08%). The transformer never trained to a result: 0.71 validation loss against 0.31 for conv at matched epochs. | Refuted. All three architectures were removed. |
| 19 | Misses sit on forced lines the network misjudges; PUCT averages them instead of resolving them, and MCTS-Solver only proves near terminals. | `--leaf-minimax-depth D` (Baier and Winands' MCTS-minimax hybrid): an exact depth-D alpha-beta search over the rules on each leaf before the network evaluates it; a proof marks the node proven for MCTS-Solver. Applies to self-play and to the generation's dev eval. | Implemented with exactness tests against brute force and the dev suite's Pons scores. Single-threaded cost on 1,152 dev rows at plies 12 to 29, mean per call (unproven calls): depth 4 0.1 us (0.4), 6 0.6 (1.9), 8 2.5 (8.4), 10 10 (36), 12 41 us (166). The 1x8 network from one 8-game CPU generation (trained at depth 6) on dev at MCTS-64, strict and keeps the result: depth 0 86.3 and 90.6%, 4 88.3 and 91.2%, 8 89.1 and 91.3%, 12 89.3 and 91.6% (eval 0.3 s at depth 0, 10 s at 12). Deterministic, so one run each; a weak estimator, not a convergence result. | See hypothesis 26 for the measured effect. Default 0 (off). |

Through hypothesis 19 the plateau sat near 95.2% strict and 97.8% keeps the result. Hypothesis 20 broke it.

| # | Hypothesis | Change tested | Result | Decision |
|---|---|---|---|---|
| 20 | The plateau is optimization noise: at a constant rate dev loss swings about 0.05 between epochs, so the published model is a noisy sample. | Cosine-anneal the rate to 1% over a fixed number of epochs (`--train-anneal-epochs`, ceiling `--anneal-epochs`), no early stopping. | Supervised 8x64 on 846k labels: MCTS-256 strict 95.10% to 96.20%. Self-play, generations 30 to 37: 95.91 and 96.10% against 95.19 and 95.18% for the constant-rate control. | Kept. Default for `train` (8 epochs) and `ceiling` (40). |
| 21 | More search at evaluation fixes the remaining misses. | Evaluate one model at MCTS-256 up to 8,192. | Flat at about 96.5 to 97% strict across the whole range. | Refuted. Search budget is not the limit. |
| 22 | The network is out of capacity. | 8x128 instead of 8x64, at a constant rate and annealed. | No gain in either setting. | Refuted. 8x64 stays. |
| 23 | More data generalizes better. | Four times the training data, annealed. | +0.1 points, inside the noise. | Refuted. |
| 24 | Connect Four is left-right symmetric, so averaging the network with its mirror cancels some of its error. | `--mirror-inference`: average every call with the mirrored board. | +0.4 to 0.6 points strict at evaluation. Doubles inference cost. | Kept as an option, off by default; use it at evaluation. |
| 25 | The value head cannot resolve a 0.01 distance step under MSE, so it misranks wins by length. | Two-hot binned value head (201 and 51 bins) and the q_penalty loss weighted 10x. | 201 bins 95.81%, 51 bins 95.70%, MSE x10 95.54%, against a same-CLI control of 95.69% (noise about 0.5). | Refuted. The code was removed. |
| 26 | Exact search on each leaf fixes the forced lines the network misjudges (hypothesis 19). | Leaf minimax at evaluation on the best supervised model; and in self-play training. | Evaluation: 96.33% to 96.85% strict at depth 12; plies 18 and up reach about 100%, the openings are unchanged. Training with it: +0.7 points early, +0.1 late, for about 1.8x time per generation. | Kept as an evaluation lever. Off for training by default. |
| 27 | Self-play targets need a deeper search. | 1,024 and 2,048 self-play MCTS iterations instead of 256. | No gain. | Refuted. 256 stays. |
| 28 | Two independently trained networks average out each other's errors. | Ensemble of two nets at evaluation. | Within noise. | Refuted. |
| 29 | Deep-search targets on openings teach the network what self-play cannot. | Distil deep MCTS policies and root values on unlabeled train-split openings into the network (`mcts_search`, `sample_split_positions`). | 97.04 and 97.34% strict against a 97.06% baseline. | Refuted. The code was removed. |

**Where the misses are.** 48 of the roughly 57 remaining dev misses of the best agent are at plies 0 to 11, and many of them change the game result. Late positions are essentially solved by leaf minimax. The opening is decided too far ahead for search, and neither capacity, data, deeper self-play search nor distillation moved it.

### Running many experiments on one GPU

A generation of the default configuration took about 2 minutes alone on the RTX 3090, and running several at once gave no more total throughput. Results-identical speedups (Rust bulk sample conversion, 5.7 s to 0.1 s per generation; generation 0 scored on the GPU) and conv heads made generations several times faster. Keeping the replay window in GPU memory saved only about 2.5 s and made every run's memory grow until concurrent runs crashed, so replay data stays in host memory. A full-replay conv 4x64 generation then peaks near 0.8 GB per process. `scripts/alapin run` caps each process (default 0.06 of the card) and refuses runs that would not fit beside a 6 GB desktop reserve. Of the optional speedups, `--bf16` was 1.46x faster on the FC network but is not validated for accuracy, `--compile` gave nothing, and `--train-steps 700` gave a worse validation loss.

## Bugs found along the way

These were real defects, each fixed with a test that fails before the fix:

- The self-play shutdown could deadlock when two workers finished the last two games at once.
- Every generation reused game ids 0 to n-1, so move sampling and root noise replayed the same random numbers each generation.
- The network kept a torchmetrics result attached to its autograd graph, so the best-model deep copy sometimes crashed.
- `train --device cpu` still trained on the GPU, because Lightning used `accelerator="auto"`.
- The original metric scored the parent's self-play games, gave half credit to slower wins, and broke ties by column order.

## What is still open

The final configuration is the default for `mise run train`. These are the most promising next steps:

1. **Openings.** Most remaining misses are at plies 0 to 11 and change the result. Search depth, capacity, data and opening distillation did not move them. An opening curriculum or a value target that sees further ahead is untested.
2. **Draws.** Drawn positions missed far more often than won positions in the earlier runs. A value head trained mostly on decisive games may not separate "drawn" from "slightly winning".
3. **Evaluation overrides.** Leaf minimax and mirror averaging help most at evaluation, but `eval` and `play` only reuse what the run recorded. A CLI override, recorded in `SuiteEval`, would make them first-class.
4. **Faster generations.** Self-play is limited by Python dispatch and CPU-side MCTS (GPU about 20% busy). `--compile` gave nothing end to end.
5. **Drop Pons.** The Rust solver in `rust/src/native_solver/` is verified but runs at about 0.66x Pons's throughput and is not wired into labeling.
