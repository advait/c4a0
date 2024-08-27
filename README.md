# c4a0: Connect Four Alpha-Zero

![CI](https://github.com/advait/c4a0/actions/workflows/ci.yaml/badge.svg?ts=2)

An Alpha-Zero-style Connect Four engine trained entirely via self play.

The game logic, Monte Carlo Tree Search, and multi-threaded self play engine is written in rust
[here](https://github.com/advait/c4a0/tree/master/rust).

The NN is written in Python/PyTorch [here](https://github.com/advait/c4a0/tree/master/src/c4a0?ts=2)
and interfaces with rust via [PyO3](https://pyo3.rs/v0.22.2/)

![Terminal UI](https://private-user-images.githubusercontent.com/504011/360721720-0267002e-2778-4fd5-a9f4-62aa4644fe84.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjQ3ODAwMDQsIm5iZiI6MTcyNDc3OTcwNCwicGF0aCI6Ii81MDQwMTEvMzYwNzIxNzIwLTAyNjcwMDJlLTI3NzgtNGZkNS1hOWY0LTYyYWE0NjQ0ZmU4NC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwODI3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDgyN1QxNzI4MjRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yZGU2ZGQwMzc0ZDEzODdiY2ZmNGQyZDMwMWYzY2QzZTFkMGYxMDU5NjhiMzhlNzRhMjdhOGY2Y2I3Mjc0YjE2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.lmjviibD8LtnRb2t-KjxXhxcRNCAxFziADMfl_ZEh2k)

## Usage

1. Install [rye](https://rye.astral.sh/) for python dep/env management
```
curl -sSf https://rye.astral.sh/get | bash
```

2. Install deps and create virtual env:
```
rye sync --no-lock
```

3. Compile rust code
```
maturin develop --release
```

4. Train a network
```
rye run python src/c4a0/main.py train --max-gens=10
```

5. Play against the network
```
rye run python src/c4a0/main.py ui --model=best
```

6. (Optional) Download a [connect four solver](https://github.com/PascalPons/connect4?ts=2) to
   objectively measure training progress:
```
git clone https://github.com/PascalPons/connect4.git solver
cd solver
make
# Download opening book to speed up solutions
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
```

Now pass the solver paths to `train`, `score` and other commands:
```
rye run python src/c4a0/main.py score --solver_path=solver/c4solver --book-path=solver/7x6.book
```

## Results
After 9 generations of training (approx ~15 min on an RTX 3090) we achieve the following results:

![Training Results](https://private-user-images.githubusercontent.com/504011/361914883-727773f6-0db3-4fcb-b7a4-00b2c4b9c155.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjQ3ODAzMzMsIm5iZiI6MTcyNDc4MDAzMywicGF0aCI6Ii81MDQwMTEvMzYxOTE0ODgzLTcyNzc3M2Y2LTBkYjMtNGZjYi1iN2E0LTAwYjJjNGI5YzE1NS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwODI3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDgyN1QxNzMzNTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lZGI5Y2ZkMGJlZDNkZDZlYzRiN2Y5MDUxMWI3Mjg1N2JiY2ZmZmM5NGJiZjQ4YzA1ZDFmYmYwODgwYjhhZThmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.m5bsLXHWC4WYQQaUgz-QWz_RERsoHxzhKruqqjy_uGg)

## Architecture

### PyTorch NN [`src/c4a0/nn.py`](https://github.com/advait/c4a0/blob/master/src/c4a0/nn.py?ts=2)

A resnet-style CNN that takes in as input a baord position and outputs a Policy (probability
distribution over moves weighted by promise) and Q Value (predicted win/loss value [-1, 1]).

Various NN hyperparameters can are sweepable via the `nn-sweep` command.

### Connect Four Game Logic [`rust/src/c4r.rs`](https://github.com/advait/c4a0/blob/master/rust/src/c4r.rs?ts=2)

Implements compact bitboard representation of board state (`Pos`) and all connect four rules
and game logic.

### Monte Carlo Tree Search (MCTS) [`rust/src/mcts.rs`](https://github.com/advait/c4a0/blob/master/rust/src/mcts.rs?ts=2)

Implements Monte Carlo Tree Search - the core algorithm behind Alpha-Zero. Probabalistically
explores potential game pathways and optimally hones in on the optimal move to play from any
position.

MCTS relies on outputs from the NN. The output of MCTS helps train the next generation's NN.

### Self Play [`rust/src/self_play.rs`](https://github.com/advait/c4a0/blob/master/rust/src/self_play.rs?ts=2)

Uses rust multi-threading to parallelize self play (training data generation).
