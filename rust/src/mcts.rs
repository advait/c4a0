use std::array;

use more_asserts::debug_assert_gt;
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

use crate::{
    c4r::{Move, Pos, TerminalState},
    types::{GameMetadata, GameResult, ModelID, Policy, QValue, Sample},
};

/// A single Monte Carlo Tree Search connect four game.
/// We store the MCTS tree in Vec form where child pointers are indicated by NodeId (the index
/// within the Vec where the given node is stored).
/// The [Self::root_id] indicates the root and the [Self::leaf_id] indicates the leaf node that has
/// yet to be expanded.
/// [Self::make_move] allows us to play a move (updating the root node to the played child) so we
/// can preserve any prior MCTS iterations that happened through that node.
#[derive(Debug, Clone)]
pub struct MctsGame {
    metadata: GameMetadata,
    nodes: Vec<Node>,
    root_id: NodeId,
    leaf_id: NodeId,
    moves: Vec<Move>,
}

impl Default for MctsGame {
    fn default() -> Self {
        MctsGame::new_from_pos(Pos::default(), GameMetadata::default())
    }
}

impl MctsGame {
    pub const UNIFORM_POLICY: Policy = [1.0 / Pos::N_COLS as f32; Pos::N_COLS];

    /// New game with the given id and start position.
    /// `c_exploration` is an MCTS parameter that guides how aggressively to explore vs.
    /// exploit.
    pub fn new_from_pos(pos: Pos, metadata: GameMetadata) -> MctsGame {
        let root_node = Node::new(pos, None, 1.0);
        MctsGame {
            metadata,
            nodes: vec![root_node],
            root_id: 0,
            leaf_id: 0,
            moves: Vec::new(),
        }
    }

    /// Gets a [Node] with the given [NodeID].
    fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// Gets a &mut [Node] with the given [NodeID].
    fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    /// Adds the node to the collection, return its id.
    fn add_node(&mut self, node: Node) -> NodeId {
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

    /// Gets the [ModelID] that is to play in the leaf position. The [ModelID] corresponds to which
    /// NN we need to call to evaluate the position.
    pub fn leaf_model_id_to_play(&self) -> ModelID {
        if self.leaf_pos().ply() % 2 == 0 {
            self.metadata.player0_id
        } else {
            self.metadata.player1_id
        }
    }

    /// Called when we receive a new policy/value from the NN forward pass for this leaf node.
    /// This is the heart of the MCTS algorithm:
    /// 1. Expands the current leaf with the given policy
    /// 2. Backpropagates up the tree with the given value
    /// 3. selects a new leaf for the next MCTS iteration.
    pub fn on_received_policy(
        &mut self,
        mut policy_logprobs: Policy,
        q_penalty: QValue,
        q_no_penalty: QValue,
        c_exploration: f32,
        c_ply_penalty: f32,
    ) {
        if let Some(terminal) = self.leaf_pos().is_terminal_state() {
            // If this is a terminal state, the received policy is irrelevant. We backpropagate
            // the objective terminal value and select a new leaf.
            let ply_penalty_magnitude = c_ply_penalty * self.leaf_pos().ply() as f32;
            let (q_no_penalty, q_penalty) = match terminal {
                // If the player wins, we apply a penalty to encourage shorter wins
                TerminalState::PlayerWin => (1.0, 1.0 - ply_penalty_magnitude),
                // If the player loses, we apply a penalty to encourage more drawn out games
                TerminalState::OpponentWin => (-1.0, -1.0 + ply_penalty_magnitude),
                TerminalState::Draw => (0.0, 0.0 - ply_penalty_magnitude),
            };
            log::debug!("on_received_policy: terminal state, value={}", q_no_penalty);

            self._backpropagate(q_penalty, q_no_penalty);
            self._select_new_leaf(c_exploration);
            return;
        }

        let leaf = self.get(self.leaf_id);
        let legal_moves = leaf.pos.legal_moves();
        debug_assert_gt!(
            { legal_moves.iter().filter(|&&m| m).count() },
            0,
            "no legal moves in leaf node"
        );

        // Mask policy for illegal moves and softmax
        for mov in 0..Pos::N_COLS {
            if !legal_moves[mov] {
                policy_logprobs[mov] = f32::NEG_INFINITY;
            }
        }
        let policy_probs = softmax(policy_logprobs);
        log::debug!(
            "policy after softmax={:?}\nnn_value={:.2}",
            policy_probs,
            q_penalty
        );

        self._expand_leaf(policy_probs);
        self._backpropagate(q_penalty, q_no_penalty);
        self._select_new_leaf(c_exploration);
    }

    /// Expands the the leaf by adding child nodes to it which then be eligible for exploration via
    /// subsequent MCTS iterations. Each child node's [Node::initial_policy_value] is determined by
    /// the provided policy.
    /// Noop for terminal nodes.
    fn _expand_leaf(&mut self, policy_probs: Policy) {
        let leaf = self.get_mut(self.leaf_id);
        if leaf.is_terminal() {
            return;
        }
        let legal_moves = leaf.pos.legal_moves();
        let children: [Option<NodeId>; Pos::N_COLS] = std::array::from_fn(|m| {
            if legal_moves[m] {
                let child_pos = {
                    let leaf = self.get(self.leaf_id);
                    leaf.pos.make_move(m).unwrap()
                };
                let child = Node::new(child_pos, Some(self.leaf_id), policy_probs[m]);
                Some(self.add_node(child))
            } else {
                None
            }
        });
        let leaf = self.get_mut(self.leaf_id);
        leaf.children = Some(children);
    }

    /// Backpropagate value up the tree, alternating value signs for each step.
    /// If the leaf node is a non-terminal node, the value is taken from the NN forward pass.
    /// If the leaf node is a terminal node, the value is the objective value of the win/loss/draw.
    ///
    /// Note we continue backpropagating counts/values up past the root's parent ancestors,
    /// effectively invalidating the policy for these nodes. This should not be a problem as we only
    /// expose a [MctsGame::root_policy] method (which will be valid), preventing callers from
    /// accessing policies for the root node's parent ancestors (which will be invalid).
    fn _backpropagate(&mut self, mut q_penalty: QValue, mut q_no_penalty: QValue) {
        let mut pos = self.get_mut(self.leaf_id);
        loop {
            pos.visit_count += 1;
            pos.q_sum_penalty += q_penalty;
            pos.q_sum_no_penalty += q_no_penalty;
            log::debug!(
                "backpropagate: pos=\n{}\nq_penalty / no_penalty={}/{}\nexploit_value={:.2}/{}={:.2}",
                pos.pos,
                q_penalty,
                q_no_penalty,
                pos.q_sum_penalty,
                pos.visit_count,
                pos.q_with_penalty(),
            );

            q_penalty = -q_penalty;
            q_no_penalty = -q_no_penalty;

            if let Some(parent_id) = pos.parent {
                pos = self.get_mut(parent_id);
            } else {
                break;
            }
        }
    }

    /// Select the next leaf node by traversing from the root node, repeatedly selecting the child
    /// with the highest [Node::uct_value] until we reach a node with no expanded children (leaf
    /// node).
    fn _select_new_leaf(&mut self, c_exploration: f32) {
        let mut cur_id = self.root_id;

        while let Some(children) = self.get(cur_id).children {
            cur_id = children
                .iter()
                .flatten()
                .map(|&id| {
                    let child = self.get(id);
                    let score = child.uct_value(self, c_exploration);
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
        log::debug!("select_new_leaf: leaf=\n{}", self.leaf_pos());
    }

    /// Makes a move, updating the root node to be the child node corresponding to the move.
    /// Note that this method does not perform garbage collection for un-played sub-trees.
    pub fn make_move(&mut self, m: Move, c_exploration: f32) {
        let root = self.get(self.root_id);
        let children = root.children.expect("root node has no children");
        let child_id = children[m as usize].expect("attempted to make an invalid move");
        self.root_id = child_id;
        self.moves.push(m);
        // We must select a new leaf as the old leaf might not be in the subtree of the new root
        self._select_new_leaf(c_exploration);
    }

    /// Makes a move probabalistically based on the root node's policy.
    /// Uses the game_id and ply as rng seeds for deterministic sampling.
    /// The temperature parameter scales the policy probabilities, with values > 1.0 making the
    /// sampled distribution more uniform and values < 1.0 making the sampled distribution favor
    /// the most lucrative moves.
    pub fn make_random_move(&mut self, c_exploration: f32, temperature: f32) {
        let seed = self.metadata.game_id * ((Pos::N_ROWS * Pos::N_COLS) + self.moves.len()) as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let policy = self.root_policy();
        let policy = apply_temperature(&policy, temperature);
        let dist = WeightedIndex::new(policy).unwrap();
        let mov = dist.sample(&mut rng);
        self.make_move(mov, c_exploration);
    }

    /// Sets the root node to the given position, clearing the entire tree and moves.
    /// This is necessary when updating the root position to a non-child state (e.g. via
    /// [Self::reset_game] or [Self::undo_move]) as ancestor states are no longer valid after
    /// [Self::make_move] is called. Therefore, instead of re-using stale ancestor states, we
    /// simply blow away the tree and start from scratch.
    fn reset_root_and_moves(&mut self, pos: Pos, moves: Vec<Move>) {
        self.nodes.clear();
        let root_node = Node::new(pos, None, 1.0);
        self.nodes.push(root_node);
        self.root_id = 0;
        self.leaf_id = 0;
        self.moves = moves;
    }

    /// Resets the game to the starting position.
    pub fn reset_game(&mut self) {
        self.reset_root_and_moves(Pos::default(), Vec::default())
    }

    /// Undoes the last move by manually replaying the moves from the start and calling
    /// [Self::reset_root_and_moves] accordingly. Returns whether the undo actually happened.
    pub fn undo_move(&mut self) -> bool {
        if self.moves.is_empty() {
            return false;
        }

        let mut moves = self.moves.clone();
        let mut pos = Pos::default();
        moves.pop();
        for &mov in &moves {
            pos = pos
                .make_move(mov)
                .expect("attempted to undo an invalid move");
        }
        self.reset_root_and_moves(pos, moves);
        true
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

    /// The average [QValue] of the root node as a consequence of performing MCTS iterations.
    pub fn root_q_with_penalty(&self) -> QValue {
        let root = self.get(self.root_id);
        root.q_with_penalty()
    }

    pub fn root_q_no_penalty(&self) -> QValue {
        let root = self.get(self.root_id);
        root.q_no_penalty()
    }

    /// Converts a finished game into a Vec of [Sample] for future NN training.
    pub fn to_result(&self, c_ply_penalty: f32) -> GameResult {
        let ply_penalty_magnitude = c_ply_penalty * self.root_pos().ply() as f32;
        let (mut q_no_penalty, mut q_penalty) = self
            .root_pos()
            .is_terminal_state()
            .map(|ts| match ts {
                TerminalState::PlayerWin => (1.0, 1.0 - ply_penalty_magnitude),
                TerminalState::OpponentWin => (-1.0, -1.0 + ply_penalty_magnitude),
                TerminalState::Draw => (0.0, 0.0 - ply_penalty_magnitude),
            })
            .expect("attempted to convert a non-terminal game to a training sample");

        let mut cur = self.get(self.leaf_id);
        let mut samples = vec![Sample {
            pos: cur.pos.clone(),
            policy: cur.policy(&self),
            q_penalty,
            q_no_penalty,
        }];
        while let Some(parent_id) = cur.parent {
            // Alternate values as the each consecutive position alternates player vs. opponent
            q_penalty = -q_penalty;
            q_no_penalty = -q_no_penalty;
            cur = self.get(parent_id);
            samples.push(Sample {
                pos: cur.pos.clone(),
                policy: cur.policy(&self),
                q_penalty,
                q_no_penalty,
            });
        }

        samples.reverse();
        GameResult {
            metadata: self.metadata.clone(),
            samples: samples,
        }
    }
}

/// We use integer Node IDs to represent pointers to nodes instead of weak referenes for
/// convenience.
type NodeId = usize;

/// A node within an MCTS tree.
#[derive(Debug, Clone)]
struct Node {
    pos: Pos,
    parent: Option<NodeId>,
    visit_count: usize,
    q_sum_penalty: f32,
    q_sum_no_penalty: f32,
    initial_policy_value: QValue,
    children: Option<[Option<NodeId>; Pos::N_COLS]>,
}

impl Node {
    const EPS: f32 = 1e-8;

    fn new(pos: Pos, parent: Option<NodeId>, initial_policy_value: QValue) -> Node {
        Node {
            pos,
            parent,
            visit_count: 0,
            q_sum_penalty: 0.0,
            q_sum_no_penalty: 0.0,
            initial_policy_value,
            children: None,
        }
    }

    /// The exploitation component of the UCT value (i.e. the average win rate) with a penalty
    /// applied for additional plys to discourage longer sequences.
    fn q_with_penalty(&self) -> QValue {
        self.q_sum_penalty / ((self.visit_count as f32) + 1.0)
    }

    /// The exploitation component of the UCT value (i.e. the average win rate) without any
    /// ply penalty.
    fn q_no_penalty(&self) -> QValue {
        self.q_sum_no_penalty / ((self.visit_count as f32) + 1.0)
    }

    /// The exploration component of the UCT value. Higher visit counts result in lower values.
    /// We also weight the exploration value by the initial policy value to allow the network
    /// to guide the search.
    fn exploration_value(&self, game: &MctsGame) -> QValue {
        let parent_visit_count = match self.parent {
            Some(parent_id) => game.get(parent_id).visit_count,
            None => self.visit_count,
        } as f32;
        let exploration_value = (parent_visit_count.ln() / (self.visit_count as f32 + 1.)).sqrt();
        exploration_value * (self.initial_policy_value + Self::EPS)
    }

    /// The UCT value of this node. Represents the lucrativeness of this node according to MCTS.
    /// Because [Self::utc_value] is called from the perspective of the *parent* node, we negate
    /// the exploration value.
    fn uct_value(&self, game: &MctsGame, c_exploration: f32) -> QValue {
        -self.q_with_penalty() + c_exploration * self.exploration_value(game)
    }

    /// Whether the game is over (won, los, draw) from this position.
    fn is_terminal(&self) -> bool {
        self.pos.is_terminal_state().is_some()
    }

    /// Uses the child counts as weights to determine the implied policy from this position.
    fn policy(&self, game: &MctsGame) -> Policy {
        if let Some(children) = self.children {
            let child_counts = children.map(|maybe_child| {
                maybe_child
                    .map(|child_id| game.get(child_id).visit_count as f32)
                    .unwrap_or(0.)
            });
            let child_counts_sum = child_counts.iter().sum::<f32>();
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

/// Softmax function for a policy.
fn softmax(policy_logprobs: Policy) -> Policy {
    let max = policy_logprobs
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        // If the policy is all negative infinity, we fall back to uniform policy.
        // This can happen if the NN dramatically underflows.
        // We panic as this is an issue that should be fixed in the NN.
        panic!("softmax: policy is all negative infinity, debug NN on why this is happening.");
    }
    let exps = policy_logprobs
        .iter()
        // Subtract max value to avoid overflow
        .map(|p| (p - max).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    array::from_fn(|i| exps[i] / sum)
}

/// Applies temperature scaling to a policy.
/// Expects the policy to be in [0-1] (non-log) space.
pub fn apply_temperature(policy: &Policy, temperature: f32) -> Policy {
    if temperature == 1.0 || policy.iter().all(|&p| p == policy[0]) {
        // Temp 1.0 or uniform policy is noop
        return policy.clone();
    } else if temperature == 0.0 {
        // Temp 0.0 is argmax
        let max = policy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ret = policy.map(|p| if p == max { 1.0 } else { 0.0 });
        let sum = ret.iter().sum::<f32>();
        return ret.map(|p| p / sum); // Potentially multiple argmaxes
    }

    let policy_log = policy.map(|p| p.ln() / temperature);
    let policy_log_sum_exp = policy_log.map(|p| p.exp()).iter().sum::<f32>().ln();
    policy_log.map(|p| (p - policy_log_sum_exp).exp().clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use more_asserts::{assert_gt, assert_lt};
    use proptest::prelude::*;

    const CONST_COL_WEIGHT: f32 = 1.0 / (Pos::N_COLS as f32);
    const CONST_POLICY: Policy = [CONST_COL_WEIGHT; Pos::N_COLS];
    const TEST_C_EXPLORATION: f32 = 4.0;
    const TEST_C_PLY_PENALTY: f32 = 0.01;

    /// Runs a batch with a single game and a constant evaluation function.
    fn run_mcts(pos: Pos, n_iterations: usize) -> (Policy, QValue, QValue) {
        let mut game = MctsGame::new_from_pos(pos, GameMetadata::default());
        for _ in 0..n_iterations {
            game.on_received_policy(
                MctsGame::UNIFORM_POLICY,
                0.0,
                0.0,
                TEST_C_EXPLORATION,
                TEST_C_PLY_PENALTY,
            )
        }
        (
            game.root_policy(),
            game.root_q_with_penalty(),
            game.root_q_no_penalty(),
        )
    }

    #[test]
    fn mcts_prefers_center_column() {
        let (policy, _q_penalty, _q_no_penalty) = run_mcts(Pos::default(), 1000);
        assert_policy_sum_1(&policy);
        assert_gt!(policy[3], CONST_COL_WEIGHT);
    }

    #[test]
    fn mcts_depth_one() {
        let (policy, _q_penalty, _q_no_penalty) =
            run_mcts(Pos::default(), 1 + Pos::N_COLS + Pos::N_COLS);
        assert_policy_eq(&policy, &CONST_POLICY, Node::EPS);
    }

    #[test]
    fn mcts_depth_two() {
        let (policy, _q_penalty, _q_no_penalty) = run_mcts(
            Pos::default(),
            1 + Pos::N_COLS + (Pos::N_COLS * Pos::N_COLS) + (Pos::N_COLS * Pos::N_COLS),
        );
        assert_policy_eq(&policy, &CONST_POLICY, Node::EPS);
    }

    #[test]
    fn mcts_depth_uneven() {
        let (policy, _q_penalty, _q_no_penalty) = run_mcts(Pos::default(), 47);
        assert_policy_ne(&policy, &CONST_POLICY, Node::EPS);
    }

    /// From an obviously winning position, mcts should end up with a policy that prefers the
    /// winning move.
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
        let (policy, q_penalty, q_no_penalty) = run_mcts(pos, 10_000);
        let winning_moves = policy[0] + policy[4];
        assert_relative_eq!(policy.iter().sum::<f32>(), 1.0);
        assert_gt!(winning_moves, 0.99);
        assert_gt!(q_penalty, 0.92);
        assert_gt!(q_no_penalty, 0.99);
    }

    /// From a winning position, mcts should end up with a policy that prefers the winning move.
    #[test]
    fn winning_position2() {
        let pos = Pos::from(
            [
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫🔵🔵⚫⚫⚫",
                "⚫⚫🔴🔴⚫⚫⚫",
            ]
            .join("\n")
            .as_str(),
        );
        let (policy, q_penalty, q_no_penalty) = run_mcts(pos, 10_000);
        let winning_moves = policy[1] + policy[4];
        assert_gt!(winning_moves, 0.98);
        assert_gt!(q_penalty, 0.90);
        assert_gt!(q_no_penalty, 0.98);
        assert_gt!(q_no_penalty, q_penalty);
    }

    /// From a winning position, mcts should end up with a policy that prefers the winning move.
    #[test]
    fn winning_position3() {
        let pos = Pos::from(
            [
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫⚫⚫⚫⚫⚫⚫",
                "⚫🔴🔵🔵⚫⚫⚫",
                "⚫🔵🔴🔴🔴⚫⚫",
                "⚫🔵🔵🔴🔵🔴⚫",
            ]
            .join("\n")
            .as_str(),
        );
        let (policy, q_penalty, q_no_penalty) = run_mcts(pos, 10_000);
        assert_gt!(policy[5], 0.99);
        assert_gt!(q_penalty, 0.86);
        assert_gt!(q_no_penalty, 0.99);
        assert_gt!(q_no_penalty, q_penalty);
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
        let (policy, q_penalty, q_no_penalty) = run_mcts(pos, 300_000);
        assert_policy_sum_1(&policy);
        policy.iter().for_each(|&p| {
            assert_relative_eq!(p, CONST_COL_WEIGHT, epsilon = 0.01);
        });
        assert_lt!(q_penalty, -0.93);
        assert_lt!(q_no_penalty, -0.99);
        assert_lt!(q_no_penalty, q_penalty);
    }

    /// From a position with two wins, prefer the shorter win. Here, playing 0 leads to a forced
    /// win, but playing 4 leads to an immediate win.
    #[test]
    fn prefer_shorter_wins() {
        let pos = Pos::from(
            [
                "⚫⚫⚫🔵⚫⚫⚫",
                "⚫🔵🔵🔵⚫⚫⚫",
                "⚫🔴🔵🔵⚫⚫⚫",
                "⚫🔴🔴🔴⚫⚫⚫",
                "⚫🔴🔴🔴⚫⚫⚫",
                "⚫🔵🔴🔵⚫⚫⚫",
            ]
            .join("\n")
            .as_str(),
        );
        let (policy, q_penalty, q_no_penalty) = run_mcts(pos, 10_000);
        assert_gt!(policy[4], 0.99);
        assert_gt!(q_penalty, 0.82);
        assert_gt!(q_no_penalty, 0.99);
        assert_gt!(q_no_penalty, q_penalty);
    }

    /// Strategy for generating a policy with at least one non-zero value.
    fn policy_strategy() -> impl Strategy<Value = Policy> {
        let min = 0.0f32;
        let max = 10.0f32;
        let positive_strategy = min..max;
        let neg_inf_strategy = Just(f32::NEG_INFINITY);
        prop::array::uniform7(prop_oneof![positive_strategy, neg_inf_strategy])
            .prop_filter("all neg infinity not allowed", |policy_logits| {
                !policy_logits.iter().all(|&p| p == f32::NEG_INFINITY)
            })
            .prop_map(|policy_log| softmax(policy_log))
    }

    proptest! {
        /// Softmax policies should sum up to one.
        #[test]
        fn softmax_sum_1(policy in policy_strategy()) {
            assert_policy_sum_1(&policy);
        }

        /// Temperature of 1.0 should not affect the policy.
        #[test]
        fn temperature_1(policy in policy_strategy()) {
            let policy_with_temp = apply_temperature(&policy, 1.0);
            assert_policy_eq(&policy, &policy_with_temp, 1e-5);
        }

        /// Temperature of 2.0 should change the policy.
        #[test]
        fn temperature_2(policy in policy_strategy()) {
            let policy_with_temp = apply_temperature(&policy, 2.0);
            assert_policy_sum_1(&policy_with_temp);
            // If policy is nonuniform and there are at least two non-zero probabilities, the
            // policy with temperature should be different from the original policy
            if policy.iter().filter(|&&p| p != CONST_COL_WEIGHT && p > 0.0).count() >= 2 {
                assert_policy_ne(&policy, &policy_with_temp, Node::EPS);
            }
        }

        /// Temperature of 0.0 should be argmax.
        #[test]
        fn temperature_0(policy in policy_strategy()) {
            let policy_with_temp = apply_temperature(&policy, 0.0);
            let max = policy_with_temp.iter().fold(f32::NEG_INFINITY, |a, &b| f32::max(a, b));
            let max_count = policy_with_temp.iter().filter(|&&p| p == max).count() as f32;
            assert_policy_sum_1(&policy_with_temp);
            for p in policy_with_temp {
                if p == max {
                    assert_eq!(1.0 / max_count, p);
                }
            }
        }
    }

    fn assert_policy_sum_1(policy: &Policy) {
        let sum = policy.iter().sum::<f32>();
        if (sum - 1.0).abs() > 1e-5 {
            panic!("policy sum {:?} is not 1.0: {:?}", sum, policy);
        }
    }

    fn assert_policy_eq(p1: &Policy, p2: &Policy, epsilon: f32) {
        let eq = p1
            .iter()
            .zip(p2.iter())
            .all(|(a, b)| (a - b).abs() < epsilon);
        if !eq {
            panic!("policies are not equal: {:?} {:?}", p1, p2);
        }
    }

    fn assert_policy_ne(p1: &Policy, p2: &Policy, epsilon: f32) {
        let ne = p1
            .iter()
            .zip(p2.iter())
            .any(|(a, b)| (a - b).abs() > epsilon);
        if !ne {
            panic!("policies are equal: {:?} {:?}", p1, p2);
        }
    }
}
