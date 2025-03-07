# c4a0: Connect Four Alpha-Zero

![CI](https://github.com/advait/c4a0/actions/workflows/ci.yaml/badge.svg?ts=2)

An Alpha-Zero-style Connect Four engine trained entirely via self play.

The game logic, Monte Carlo Tree Search, and multi-threaded self play engine is written in rust
[here](https://github.com/advait/c4a0/tree/master/rust).

The NN is written in Python/PyTorch [here](https://github.com/advait/c4a0/tree/master/src/c4a0?ts=2)
and interfaces with rust via [PyO3](https://pyo3.rs/v0.22.2/)

![Terminal UI](https://raw.githubusercontent.com/advait/c4a0/refs/heads/master/images/tui.png)

## Usage

1. Install clang
```sh
# Instructions for Ubuntu/Debian (other OSs may vary)
sudo apt install clang
```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for python dep/env management
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install deps and create virtual env:
```sh
uv sync
```

4. Compile rust code
```sh
uv run maturin develop --release
```

4. Train a network
```sh
uv run src/c4a0/main.py train --max-gens=10
```

5. Play against the network
```sh
uv run src/c4a0/main.py play --model=best
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
uv run python src/c4a0/main.py score solver/c4solver solver/7x6.book
```

## Results
After 9 generations of training (approx ~15 min on an RTX 3090) we achieve the following results:

![Training Results](https://raw.githubusercontent.com/advait/c4a0/refs/heads/master/images/learning.png)

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

### Solver [`rust/src/solver.rs`](https://github.com/advait/c4a0/blob/master/rust/src/solver.rs?ts=2)

Connect Four is a perfectly solved game. See Pascal Pons's [great
writeup](http://blog.gamesolver.org/) on how to build a perfect solver. We can use these solutions
to objectively measure our NN's performance. Importantly we **never train on these solutions**,
instead only using our self-play data to improve the NN's performance.

`solver.rs` contains the stdin/out interface to learn the objective solutions to our training
positions. Because solutions are expensive to compute, we cache them in a local
[rocksdb](https://docs.rs/rocksdb/latest/rocksdb/) database (solutions.db). We then measure our
training positions to see how often they recommend optimal moves as determined by the solver.
