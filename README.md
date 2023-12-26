# c4a0: Connect Four Alpha-Zero

A simple alpha-zero-style Connect Four neural network trained via self play.

## Components

### Game structure
- 6-row, 7-col board (6x7 matrix)
- Array values are -1 (opponent), 0 (empty), 1 (player)
- Move policy (recommendation) is given by 7 dim stochastic vector
- A given game state has a value range (-1 to 1) that indicates whether the position is winning 
  (1) or losing (-1)

### Neural network
- Input: Game State `s` (from the perspective of the playing player) (7x6 matrix)
- Two outputs:
  - Policy output: 7 dim stochastic vector suggesting which moves are favorable
      - Illegal moves are masked out (and the vector is re-normalized)
  - Value output: (-1 to 1) scalar (tanh activation)
- Architecture:
  - Consider some CNN structure
  - Determine how many layers are needed

### Monte Carlo Tree Search (MCTS)
- MCTS is critical to both learning and playing (both self-play and competitive-play)
- MCTS takes in as input a given game state and explores various "lines" (paths from the root
  state to a leaf state)
- A Node in the MCTS tree consists of:
  - `pos`: The 6x7 matrix representing the position
  - `visit_count`: Number of times we've visited this node
  - `exploitation_value_sum`: Cumulative "value" of this node, via child/descendant visits
  - `initial_policy_value`: The output of the policy vector that was generated by the parent
    when initially creating this node.
  - `exploitation_value()`: `exploitation_value_sum` / `visit_count`
    - The average win rate (the lucrativeness of this node)
  - `exploration_value()`: sqrt(ln(`parent.visit_count`) / `visit_count`) * `initial_policy_value`
    - Lower visit counts result in higher exploration values
    - We incorporate the `initial_policy_value` to bias MCTS to explore more promising moves
  - `uct_value()`: `exploitation_value()` + `exploration_constant` * `exploration_value()`
    - How appealing is this node to visit in the context of MCTS
    - The constant hyperparameter `exploration_constant` weights the balance between exploitation
      vs. exploration.
  - `children`: An array of child nodes, initially initialized to null indicating that this is
    a leaf node (as-yet unexplored)
- The process of generating lines is as follows:
  1. Start at the root node
  2. Randomly sample/select a child node to visit based on their `uct_value()`s
  3. Repeat #2 until we arrive at a leaf node
- Upon arriving a leaf node, we perform the following steps (expansion):
  1. Run the neural network on the leaf node to determine a `policy` vector and a `value` scalar
    - If this leaf node is actually a terminal node (game end), use 1, 0, or -1 as the `value`
      based on whether this was a win, draw, or loss
  2. For each legal move from the leaf node, add one child node representing the resulting states
    - Each child node's `initial_policy_value` is based on the policy from #1
  3. The value from #1 is then backpropagated up the tree
- The backpropagation process is as follows
  1. Increase this node's `visit_count` by 1
  2. Increase this node's `exploration_value_sum` by the `value`
  3. Repeat #1 for this node's parent until we get to the root
- By repeating the MCTS process, we gain the following:
  1. A more accurate `exploitation_value()` for the root
    - This scalar is comparable to the `value` output of the neural network
  2. More accurate `exploitation_value()`s for the root's children.
    - When normalized, this vector is comparable to the `policy` output of the neural network
    - This more-accurate policy is then used for self- and competitive play
