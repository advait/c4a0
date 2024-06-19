use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

use crate::c4r::{Move, Pos, TerminalState};

/// Probabilities for how lucrative each column is.
pub type Policy = [f64; Pos::N_COLS];

/// The lucrativeness value of a given position.
pub type PosValue = f64;

/// A training sample generated via self-play.
#[derive(Debug)]
pub struct Sample {
    pub game_id: u64,
    pub pos: Pos,
    pub policy: Policy,
    pub value: PosValue,
}

/// A single MCTS game.
/// We store the Monte Carlo Tree in Vec form where child pointers are indicated by NodeId (the
/// index within the Vec where the given node is stored).
/// The root_id indicates the root and the leaf_id indicates the leaf node that has yet to be
/// expanded.
/// fn play_move allows us to play a move (updating the root node to the played child) so we can
/// preserve any prior MCTS iterations through that node.
#[derive(Debug, Clone)]
pub struct MctsGame {
    pub game_id: u64,
    nodes: Vec<Node>,
    root_id: NodeId,
    leaf_id: NodeId,
    moves: Vec<Move>,
}

impl MctsGame {
    pub const UNIFORM_POLICY: Policy = [1.0 / Pos::N_COLS as f64; Pos::N_COLS];

    pub fn new_with_id(game_id: u64) -> MctsGame {
        Self::new_from_pos(Pos::new(), game_id)
    }

    pub fn new_from_pos(pos: Pos, game_id: u64) -> MctsGame {
        let root_node = Node::new(pos, None, 0.0);
        MctsGame {
            nodes: vec![root_node],
            root_id: 0,
            leaf_id: 0,
            moves: Vec::new(),
            game_id,
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

    /// Gets the root position - the last moved that was played.
    pub fn root_pos(&self) -> &Pos {
        &self.get(self.root_id).pos
    }

    /// Gets the leaf node position that needs to be evaluated by the NN.
    pub fn leaf_pos(&self) -> &Pos {
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
        let legal_moves = leaf.pos.legal_moves();
        let children: [Option<NodeId>; Pos::N_COLS] = std::array::from_fn(|m| {
            if legal_moves[m] {
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

    /// Makes a move, updating the root node to be the child node corresponding to the move.
    /// Note that this method does not perform garbage collection for un-played sub-trees.
    pub fn make_move(&mut self, m: Move) {
        let root = self.get(self.root_id);
        let children = root.children.expect("root node has no children");
        let child_id = children[m as usize].expect("attempted to make an invalid move");
        self.root_id = child_id;
        self.moves.push(m);
    }

    /// Makes a move probabalistically based on the root node's policy.
    /// Uses the game_id and ply as rng seeds for deterministic sampling.
    pub fn make_random_move(&mut self) {
        let seed = self.game_id * ((Pos::N_ROWS * Pos::N_COLS) + self.moves.len()) as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let policy = self.root_policy();
        let dist = WeightedIndex::new(policy).unwrap();
        let mov = dist.sample(&mut rng);
        self.make_move(mov);
    }

    /// The number of visits to the root node.
    pub fn root_visit_count(&self) -> usize {
        self.get(self.root_id).visit_count
    }

    /// After performing many MCTS iterations, the resulting policy is determined by the visit count
    /// of each child (more visits implies more lucrative).
    pub fn root_policy(&self) -> Policy {
        let root = self.get(self.root_id);
        root.policy(self)
    }

    /// Converts a finished game into a Vec of training samples for future NN training.
    pub fn to_training_samples(&self) -> Vec<Sample> {
        let final_value = self
            .root_pos()
            .is_terminal_state()
            .map(|ts| match ts {
                TerminalState::PlayerWin => 1.0,
                TerminalState::OpponentWin => -1.0,
                TerminalState::Draw => 0.0,
            })
            .expect("attempted to convert a non-terminal game to a training sample");

        let mut cur = self.get(self.leaf_id);
        let mut cur_value = final_value;
        let mut ret = vec![Sample {
            game_id: self.game_id,
            pos: cur.pos.clone(),
            policy: cur.policy(&self),
            value: cur_value,
        }];
        while let Some(parent_id) = cur.parent {
            // Alternate values as the each consecutive position alternates player vs. opponent
            cur_value = -cur_value;
            cur = self.get(parent_id);
            ret.push(Sample {
                game_id: self.game_id,
                pos: cur.pos.clone(),
                policy: cur.policy(&self),
                value: cur_value,
            });
        }

        ret.reverse();
        ret
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
        self.pos.is_terminal_state().is_some()
    }

    /// Uses the child counts to determine the policy from this position.
    fn policy(&self, game: &MctsGame) -> Policy {
        if let Some(children) = self.children {
            let child_counts = children.map(|maybe_child| {
                maybe_child
                    .map(|child_id| game.get(child_id).visit_count as f64)
                    .unwrap_or(0 as f64)
            });
            let child_counts_sum: f64 = child_counts.iter().sum();
            if child_counts_sum == 0.0 {
                MctsGame::UNIFORM_POLICY
            } else {
                child_counts.map(|c| c / child_counts_sum)
            }
        } else {
            MctsGame::UNIFORM_POLICY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_relative_ne};
    use more_asserts::assert_gt;

    const CONST_COL_WEIGHT: f64 = 1.0 / (Pos::N_COLS as f64);
    const TEST_EXPLORATION_CONSTANT: f64 = 1.0;

    /// Runs a batch with a single game and a constant evaluation function.
    fn run_mcts(pos: Pos, n_iterations: usize) -> Policy {
        let mut game = MctsGame::new_from_pos(pos, 0);
        for _ in 0..n_iterations {
            game.on_received_policy(MctsGame::UNIFORM_POLICY, 0.0, TEST_EXPLORATION_CONSTANT)
        }
        game.root_policy()
    }

    #[test]
    fn mcts_prefers_center_column() {
        let policy = run_mcts(Pos::new(), 1000);
        assert_relative_eq!(policy.iter().sum::<f64>(), 1.0);
        assert_gt!(policy[3], CONST_COL_WEIGHT);
    }

    #[test]
    fn mcts_depth_one() {
        let policy = run_mcts(Pos::new(), 1 + Pos::N_COLS + Pos::N_COLS);
        assert_relative_eq!(policy.iter().sum::<f64>(), 1.0);
        policy.iter().for_each(|p| {
            assert_relative_eq!(*p, CONST_COL_WEIGHT);
        });
    }

    #[test]
    fn mcts_depth_two() {
        let policy = run_mcts(
            Pos::new(),
            1 + Pos::N_COLS + (Pos::N_COLS * Pos::N_COLS) + (Pos::N_COLS * Pos::N_COLS),
        );
        assert_relative_eq!(policy.iter().sum::<f64>(), 1.0);
        policy.iter().for_each(|p| {
            assert_relative_eq!(*p, CONST_COL_WEIGHT);
        });
    }

    #[test]
    fn mcts_depth_uneven() {
        let policy = run_mcts(Pos::new(), 47);
        assert_relative_eq!(policy.iter().sum::<f64>(), 1.0);
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
        let policy = run_mcts(pos, 1_000);
        let winning_moves = policy[0] + policy[4];
        assert_relative_eq!(policy.iter().sum::<f64>(), 1.0);
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
        let policy = run_mcts(pos, 100_000);
        assert_relative_eq!(policy.iter().sum::<f64>(), 1.0);
        policy.iter().for_each(|p| {
            assert_relative_eq!(*p, CONST_COL_WEIGHT, epsilon = 0.02);
        });
    }
}
