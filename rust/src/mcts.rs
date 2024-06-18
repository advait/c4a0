use std::collections::{HashMap, HashSet};

use crate::c4r::{Move, Pos, TerminalState};

/// Probabilities for how lucrative each column is.
pub type Policy = [f64; Pos::N_COLS];

/// The lucrativeness value of a given position.
pub type PosValue = f64;

/// Evaluate a batch of positions with an NN forward pass.
pub type EvalPosFn = fn(&Vec<Pos>) -> Vec<EvalPosResult>;

/// A batch of MCTS games that are generated together.
/// We a pytorch NN forward pass to expand a given node (to determine the initial policy values
/// based on the NN's output policy). Because we want to batch these NN calls for performance, we
/// partially compute many MCTS traversals simultaneously, pausing each until we reach the node
/// expansion phase. Then we are able to batch several NN calls simultaneously.
/// After the batched forward pass completes, we resume the next iteration of MCTS, continuing this
/// process until each perform n_iterations.
struct MctsBatch {
    games: Vec<MctsGame>,
    n_iterations: usize,
    exploration_constant: f64,
    eval_pos: EvalPosFn,
}

impl MctsBatch {
    fn new(
        n_games: usize,
        n_iterations: usize,
        exploration_constant: f64,
        eval_pos: EvalPosFn,
    ) -> MctsBatch {
        MctsBatch {
            games: vec![MctsGame::new(); n_games],
            n_iterations,
            exploration_constant,
            eval_pos,
        }
    }

    /// Runs n_iterations MCTS iterations.
    fn run(&mut self) {
        for _ in 0..self.n_iterations {
            self.run_single_iteration();
        }
    }

    /// Runs a single iteraton of MCTS for all batch games, calling eval_pos once.
    fn run_single_iteration(&mut self) {
        let position_set: HashSet<_> = self
            .games
            .iter()
            .map(|g| g.get_leaf_pos().clone())
            .collect();
        let positions: Vec<_> = position_set.into_iter().collect();

        let all_evals = (self.eval_pos)(&positions);
        let eval_map: HashMap<_, _> = positions.into_iter().zip(all_evals).collect();

        for game in self.games.iter_mut() {
            let pos = game.get_leaf_pos();
            let nn_result = &eval_map[pos];
            game.on_received_policy(nn_result.policy, nn_result.value, self.exploration_constant);
        }
    }
}

/// The returned output from the forward pass of the NN.
#[derive(Debug, Clone)]
pub struct EvalPosResult {
    pub policy: Policy,
    pub value: PosValue,
}

/// A single MCTS game.
/// We store the Monte Carlo Tree in Vec form where child pointers are indicated by NodeId (the
/// index within the Vec where the given node is stored).
/// The root_id indicates the root and the leaf_id indicates the leaf node that has yet to be
/// expanded.
#[derive(Debug, Clone)]
pub struct MctsGame {
    nodes: Vec<Node>,
    root_id: NodeId,
    leaf_id: NodeId,
    moves: Vec<Move>,
}

impl MctsGame {
    pub fn new() -> MctsGame {
        Self::new_from_pos(Pos::new())
    }

    pub fn new_from_pos(pos: Pos) -> MctsGame {
        let root_node = Node::new(pos, None, 0.0);
        MctsGame {
            nodes: vec![root_node],
            root_id: 0,
            leaf_id: 0,
            moves: Vec::new(),
        }
    }

    fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    /// Adds the node to the collection, return its id.
    fn _add_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Gets the leaf node position that needs to be evaluated by the NN
    pub fn get_leaf_pos(&self) -> &Pos {
        &self.get(self.leaf_id).pos
    }

    /// Called when we receive a new policy/value from the NN forward pass for this leaf node.
    /// Expands the current leaf with the given policy, backpropagates up the tree with the given
    /// value, and selects a new leaf for the next MCTS iteration.
    pub fn on_received_policy(
        &mut self,
        policy: Policy,
        nn_value: PosValue,
        exploration_constant: f64,
    ) {
        self._expand_leaf(self.leaf_id, policy);

        let leaf = self.get(self.leaf_id);
        let value = match leaf.pos.is_terminal_state() {
            Some(TerminalState::PlayerWin) => 1.0,
            Some(TerminalState::OpponentWin) => -1.0,
            Some(TerminalState::Draw) => 0.0,
            None => nn_value,
        };
        self._backpropagate(self.leaf_id, value);

        self._select_new_leaf(exploration_constant);
    }

    /// Expands the the leaf by adding child nodes to it which then be eligible for exploration via
    /// subsequent MCTS iterations. Each child node's initial_policy_value is determined by the
    /// provided policy.
    /// Noop for terminal nodes.
    fn _expand_leaf(&mut self, leaf_id: NodeId, policy: Policy) {
        let leaf = self.get_mut(leaf_id);
        if leaf.is_terminal() {
            return;
        }

        let children: [Option<NodeId>; Pos::N_COLS] = std::array::from_fn(|m| {
            if policy[m] > 0.0 {
                let child_pos = {
                    let leaf = self.get(leaf_id);
                    leaf.pos.make_move(m).unwrap()
                };
                let child = Node::new(child_pos, Some(leaf_id), policy[m]);
                Some(self._add_node(child))
            } else {
                None
            }
        });
        let leaf = self.get_mut(leaf_id);
        leaf.children = Some(children);
    }

    /// Backpropagate value up the tree, alternating value signs for each step.
    /// If the leaf node is a non-terminal node, the value is taken from the NN forward pass.
    /// If the leaf node is a terminal node, the value is the objective value of the win/loss/draw.
    fn _backpropagate(&mut self, leaf_id: NodeId, mut value: PosValue) {
        let mut pos = self.get_mut(leaf_id);
        loop {
            pos.visit_count += 1;
            pos.exploitation_value_sum += value;
            value = -value;

            if let Some(parent_id) = pos.parent {
                pos = self.get_mut(parent_id);
            } else {
                break;
            }
        }
    }

    /// Select the next leaf node by traversing from the root node, repeatedly selecting the child
    /// with the highest uct_value until we reach a node with no expanded children (leaf node).
    fn _select_new_leaf(&mut self, exploration_constant: f64) {
        let mut cur_id = self.root_id;

        while let Some(children) = self.get(cur_id).children {
            cur_id = children
                .iter()
                .flatten()
                .map(|&id| {
                    let child = self.get(id);
                    let score = child.uct_value(exploration_constant, self);
                    (id, score)
                })
                .max_by(|(_a_id, a_score), (_b_id, b_score)| {
                    a_score
                        .partial_cmp(b_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _score)| id)
                .expect("attempted to expand a terminal node")
        }

        self.leaf_id = cur_id;
    }

    /// After performing many MCTS iterations, the resulting policy is determined by the visit count
    /// of each child (more visits implies more lucrative).
    pub fn final_policy(&self) -> Policy {
        let root = self.get(self.root_id);

        let child_counts = if let Some(children) = root.children {
            children.map(|o| match o {
                Some(child_id) => self.get(child_id).visit_count,
                None => 0,
            })
        } else {
            [0; Pos::N_COLS]
        };

        // Prevent div by zero
        let total = usize::max(child_counts.iter().sum(), 1) as f64;
        child_counts.map(|c| c as f64 / total)
    }

    /// The number of visits to the root node.
    pub fn root_visit_count(&self) -> usize {
        self.get(self.root_id).visit_count
    }
}

/// We use integer Node IDs to represent pointers to nodes instead of weak referenes for
/// convenience.
type NodeId = usize;

#[derive(Debug, Clone)]
struct Node {
    pos: Pos,
    parent: Option<NodeId>,
    visit_count: usize,
    exploitation_value_sum: f64,
    initial_policy_value: PosValue,
    children: Option<[Option<NodeId>; Pos::N_COLS]>,
}

impl Node {
    const EPS: f64 = 1e-8;

    fn new(pos: Pos, parent: Option<NodeId>, initial_policy_value: PosValue) -> Node {
        Node {
            pos,
            parent,
            visit_count: 0,
            exploitation_value_sum: 0.0,
            initial_policy_value,
            children: None,
        }
    }

    /// The exploitation component of the UCT value, i.e. the average win rate.
    /// Because we are viewing the value from the perspective of the parent node, we negate it.
    fn exploitation_value(&self) -> PosValue {
        -1.0 * self.exploitation_value_sum / ((self.visit_count as f64) + 1.0)
    }

    /// The exploration component of the UCT value. Higher visit counts result in lower values.
    /// We also weight the exploration value by the initial policy value to allow the network
    /// to guide the search.
    fn exploration_value(&self, game: &MctsGame) -> PosValue {
        let parent_visit_count = match self.parent {
            Some(parent_id) => game.get(parent_id).visit_count,
            None => self.visit_count,
        } as f64;
        let exploration_value = (parent_visit_count.ln() / (self.visit_count as f64 + 1.)).sqrt();
        exploration_value * (self.initial_policy_value + Self::EPS)
    }

    /// The UCT value of this node. Represents the lucrativeness of this node according to MCTS.
    fn uct_value(&self, exploration_constant: f64, game: &MctsGame) -> PosValue {
        self.exploitation_value() + exploration_constant * self.exploration_value(game)
    }

    /// Whether the game is over (won, los, draw) from this position.
    fn is_terminal(&self) -> bool {
        self.pos.is_terminal_state() != None
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, assert_relative_ne};
    use more_asserts::assert_gt;

    use super::*;

    const CONST_COL_WEIGHT: f64 = 1.0 / (Pos::N_COLS as f64);
    const CONST_POLICY: [f64; Pos::N_COLS] = [CONST_COL_WEIGHT; Pos::N_COLS];

    /// Runs a batch with a single game and a constant evaluation function.
    fn run_batch_with_pos(pos: Pos, n_iterations: usize) -> Policy {
        let mut batch = MctsBatch {
            games: vec![MctsGame::new_from_pos(pos); 1],
            n_iterations,
            exploration_constant: 1.4,
            eval_pos: constant_eval_pos,
        };
        batch.run();
        batch.games[0].final_policy()
    }

    /// A constant evaluation function that returns a uniform policy and 0.0 value.
    fn constant_eval_pos(pos: &Vec<Pos>) -> Vec<EvalPosResult> {
        pos.into_iter()
            .map(|_p| EvalPosResult {
                policy: CONST_POLICY,
                value: 0.0,
            })
            .collect()
    }

    #[test]
    fn mcts_prefers_center_column() {
        let policy = run_batch_with_pos(Pos::new(), 1000);
        assert_gt!(policy[3], CONST_COL_WEIGHT);
    }

    #[test]
    fn mcts_depth_one() {
        let policy = run_batch_with_pos(Pos::new(), 1 + Pos::N_COLS + Pos::N_COLS);
        policy.iter().for_each(|p| {
            assert_eq!(*p, CONST_COL_WEIGHT);
        });
    }

    #[test]
    fn mcts_depth_two() {
        let policy = run_batch_with_pos(
            Pos::new(),
            1 + Pos::N_COLS + (Pos::N_COLS * Pos::N_COLS) + (Pos::N_COLS * Pos::N_COLS),
        );
        policy.iter().for_each(|p| {
            assert_eq!(*p, CONST_COL_WEIGHT);
        });
    }

    #[test]
    fn mcts_depth_uneven() {
        let policy = run_batch_with_pos(Pos::new(), 47);
        policy.iter().for_each(|p| {
            assert_relative_ne!(*p, CONST_COL_WEIGHT, epsilon = 0.001);
        });
    }

    /// From a winning position, mcts should end up with a policy that prefers the winning move.
    #[test]
    fn winning_position() {
        let pos = Pos::from(
            [
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫🔵🔵🔵⚫⚫⚫",
                "⚫🔴🔴🔴⚫⚫⚫",
            ]
            .join("\n")
            .as_str(),
        );
        let policy = run_batch_with_pos(pos, 1_000);
        let winning_moves = policy[0] + policy[4];
        assert_relative_eq!(winning_moves, 1.0, epsilon = 0.01)
    }

    /// From a definitively losing position, mcts should end up with a uniform policy because it's
    /// desperately trying to find a non-losing move.
    #[test]
    fn losing_position() {
        let pos = Pos::from(
            [
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫🔴🔴⚫⚫⚫⚫",
                "⚫🔵🔵🔵⚫⚫⚫",
            ]
            .join("\n")
            .as_str(),
        );
        let policy = run_batch_with_pos(pos, 100_000);
        policy.iter().for_each(|p| {
            assert_relative_eq!(*p, CONST_COL_WEIGHT, epsilon = 0.02);
        });
    }
}
