use std::{
    array,
    cell::RefCell,
    rc::{Rc, Weak},
};

use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

use crate::{
    c4r::{Move, Pos},
    types::{policy_from_iter, GameMetadata, GameResult, ModelID, Policy, QValue, Sample},
    utils::OrdF32,
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
    root: Rc<RefCell<Node>>,
    leaf: Rc<RefCell<Node>>,
    moves: Vec<RecordedMove>,
}

impl Default for MctsGame {
    fn default() -> Self {
        MctsGame::new_from_pos(Pos::default(), GameMetadata::default())
    }
}

/// SAFETY: MctsGame is Send because it doesn't have any public methods that expose the Rc/RefCell
/// allowing for illegal cross-thread mutation.
unsafe impl Send for MctsGame {}

impl MctsGame {
    pub const UNIFORM_POLICY: Policy = [1.0 / Pos::N_COLS as f32; Pos::N_COLS];

    /// New game with the given id and start position.
    pub fn new_from_pos(pos: Pos, metadata: GameMetadata) -> MctsGame {
        let root_node = Rc::new(RefCell::new(Node::new(pos, Weak::new(), 1.0)));
        MctsGame {
            metadata,
            root: Rc::clone(&root_node),
            leaf: root_node,
            moves: Vec::new(),
        }
    }

    /// Gets the root position - the last moved that was played.
    pub fn root_pos(&self) -> Pos {
        self.root.borrow().pos.clone()
    }

    /// Gets the leaf node position that needs to be evaluated by the NN.
    pub fn leaf_pos(&self) -> Pos {
        self.leaf.borrow().pos.clone()
    }

    /// Gets the [ModelID] that is to play in the leaf position. The [ModelID] corresponds to which
    /// NN we need to call to evaluate the position.
    pub fn leaf_model_id_to_play(&self) -> ModelID {
        if self.leaf.borrow().pos.ply() % 2 == 0 {
            self.metadata.player0_id
        } else {
            self.metadata.player1_id
        }
    }

    /// Called when we receive a new policy/value from the NN forward pass for this leaf node.
    /// This is the heart of the MCTS algorithm:
    /// 1. Expands the current leaf with the given policy (if it is non-terminal)
    /// 2. Backpropagates up the tree with the given value (or the objective terminal value)
    /// 3. selects a new leaf for the next MCTS iteration.
    pub fn on_received_policy(
        &mut self,
        mut policy_logprobs: Policy,
        q_penalty: QValue,
        q_no_penalty: QValue,
        c_exploration: f32,
        c_ply_penalty: f32,
    ) {
        let leaf_pos = self.leaf_pos();
        if let Some((q_penalty, q_no_penalty)) =
            leaf_pos.terminal_value_with_ply_penalty(c_ply_penalty)
        {
            // If this is a terminal state, the received policy is irrelevant. We backpropagate
            // the objective terminal value and select a new leaf.
            self.backpropagate_value(q_penalty, q_no_penalty);
            self.select_new_leaf(c_exploration);
        } else {
            // If this is a non-terminal state, we use the received policy to expand the leaf,
            // backpropagate the received value, and select a new leaf.
            leaf_pos.mask_policy(&mut policy_logprobs);
            let policy_probs = softmax(policy_logprobs);
            self.expand_leaf(policy_probs);
            self.backpropagate_value(q_penalty, q_no_penalty);
            self.select_new_leaf(c_exploration);
        }
    }

    /// Expands the the leaf by adding child nodes to it which then be eligible for exploration via
    /// subsequent MCTS iterations. Each child node's [Node::initial_policy_value] is determined by
    /// the provided policy.
    /// Noop for terminal nodes.
    fn expand_leaf(&self, policy_probs: Policy) {
        let leaf_pos = self.leaf_pos();
        if leaf_pos.is_terminal_state().is_some() {
            return;
        }
        let legal_moves = leaf_pos.legal_moves();

        let children: [Option<Rc<RefCell<Node>>>; Pos::N_COLS] = std::array::from_fn(|m| {
            if legal_moves[m] {
                let child_pos = leaf_pos.make_move(m).unwrap();
                let child = Node::new(child_pos, Rc::downgrade(&self.leaf), policy_probs[m]);
                Some(Rc::new(RefCell::new(child)))
            } else {
                None
            }
        });
        let mut leaf = self.leaf.borrow_mut();
        leaf.children = Some(children);
    }

    /// Backpropagate value up the tree, alternating value signs for each step.
    /// If the leaf node is a non-terminal node, the value is taken from the NN forward pass.
    /// If the leaf node is a terminal node, the value is the objective value of the win/loss/draw.
    fn backpropagate_value(&self, mut q_penalty: QValue, mut q_no_penalty: QValue) {
        let mut node_ref = Rc::clone(&self.leaf);
        loop {
            let mut node = node_ref.borrow_mut();
            node.visit_count += 1;
            node.q_sum_penalty += q_penalty;
            node.q_sum_no_penalty += q_no_penalty;

            q_penalty = -q_penalty;
            q_no_penalty = -q_no_penalty;

            if let Some(parent) = node.parent.upgrade() {
                drop(node); // Drop node_ref borrow so we can reassign node_ref
                node_ref = parent;
            } else {
                break;
            }
        }
    }

    /// Select the next leaf node by traversing from the root node, repeatedly selecting the child
    /// with the highest [Node::uct_value] until we reach a node with no expanded children (leaf
    /// node).
    fn select_new_leaf(&mut self, c_exploration: f32) {
        let mut node_ref = Rc::clone(&self.root);

        loop {
            let next = node_ref.borrow().children.as_ref().and_then(|children| {
                children
                    .iter()
                    .flatten()
                    .max_by_key(|&child| {
                        let score = child.borrow().uct_value(c_exploration);
                        OrdF32(score)
                    })
                    .cloned()
            });

            if let Some(next) = next {
                node_ref = Rc::clone(&next)
            } else {
                break;
            }
        }

        self.leaf = node_ref;
    }

    /// Makes a move, updating the root node to be the child node corresponding to the move.
    /// Stores the previous position and policy in the [Self::moves] vector.
    pub fn make_move(&mut self, m: Move, c_exploration: f32) {
        self.moves.push(RecordedMove {
            pos: self.root_pos(),
            policy: self.root_policy(),
            mov: m,
        });

        let child = {
            let root = self.root.borrow();
            let children = root.children.as_ref().expect("root node has no children");
            let child = children[m as usize]
                .as_ref()
                .expect("attempted to make an invalid move");
            Rc::clone(&child)
        };
        self.root = child;

        // We must select a new leaf as the old leaf might not be in the subtree of the new root
        self.select_new_leaf(c_exploration);
    }

    /// Makes a move probabalistically based on the root node's policy.
    /// Uses the game_id and ply as rng seeds for deterministic sampling.
    ///
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

    /// Resets the game to the starting position.
    pub fn reset_game(&mut self) {
        while self.undo_move() {}
    }

    /// Undo the last move.
    pub fn undo_move(&mut self) -> bool {
        if self.moves.is_empty() {
            return false;
        }

        let mut moves = self.moves.clone();
        let last_move = moves.pop().unwrap();

        // last_move.pos is the previous position
        let root = Node::new(last_move.pos, Weak::new(), 1.0);
        let root = Rc::new(RefCell::new(root));
        self.root = Rc::clone(&root);
        self.leaf = root;
        self.moves = moves;
        true
    }

    /// The number of visits to the root node.
    pub fn root_visit_count(&self) -> usize {
        self.root.borrow().visit_count
    }

    /// After performing many MCTS iterations, the resulting policy is determined by the visit count
    /// of each child (more visits implies more lucrative).
    pub fn root_policy(&self) -> Policy {
        self.root.borrow().policy()
    }

    /// The average [QValue] of the root node as a consequence of performing MCTS iterations
    /// (with ply penalties applied).
    pub fn root_q_with_penalty(&self) -> QValue {
        self.root.borrow().q_with_penalty()
    }

    /// The average [QValue] of the root node as a consequence of performing MCTS iterations
    /// (without ply penalties applied).
    pub fn root_q_no_penalty(&self) -> QValue {
        self.root.borrow().q_no_penalty()
    }

    /// Converts a finished game into a Vec of [Sample] for future NN training.
    pub fn to_result(self, c_ply_penalty: f32) -> GameResult {
        let (q_penalty, q_no_penalty) = self
            .root
            .borrow()
            .pos
            .terminal_value_with_ply_penalty(c_ply_penalty)
            .expect("attempted to convert a non-terminal game to a training sample");

        // Q values alternate for each ply as perspective alternates between players.
        let mut alternating_q = vec![(q_penalty, q_no_penalty), (-q_penalty, -q_no_penalty)]
            .into_iter()
            .cycle();
        if self.moves.len() % 2 == 1 {
            // If we have an odd number of moves (even number of total positions), the first Q value
            // should be inverted so that the final Q value is based on the terminal state above.
            alternating_q.next();
        }

        let mut samples: Vec<_> = self
            .moves
            .iter()
            .zip(alternating_q)
            .map(|(mov, (q_penalty, q_no_penalty))| Sample {
                pos: mov.pos.clone(),
                policy: mov.policy,
                q_penalty,
                q_no_penalty,
            })
            .collect();

        // Add the final (terminal) position with an arbitray uniform policy
        samples.push(Sample {
            pos: self.root.borrow().pos.clone(),
            policy: MctsGame::UNIFORM_POLICY,
            q_penalty,
            q_no_penalty,
        });

        GameResult {
            metadata: self.metadata.clone(),
            samples: samples,
        }
    }
}

/// Recorded move during the MCTS process.
#[derive(Debug, Clone)]
struct RecordedMove {
    pos: Pos,
    policy: Policy,
    mov: Move,
}

/// A node within an MCTS tree.
/// [Self::parent] is a weak reference to the parent node to avoid reference cycles.
/// [Self::children] is an array of optional child nodes. If a child is None, it means that the
/// move is illegal. Otherwise the child is a [Rc<RefCell<Node>>] reference to the child node.
/// We maintain two separate Q values: one with ply penalties applied ([Self::q_sum_penalty]) and
/// one without ([Self::q_sum_no_penalty]). These are normalized with [Self::visit_count] to get the
/// average [QValue]s in [Self::q_with_penalty()] and [Self::q_no_penalty()].
#[derive(Debug, Clone)]
struct Node {
    pos: Pos,
    parent: Weak<RefCell<Node>>,
    visit_count: usize,
    q_sum_penalty: f32,
    q_sum_no_penalty: f32,
    initial_policy_value: QValue,
    children: Option<[Option<Rc<RefCell<Node>>>; Pos::N_COLS]>,
}

impl Node {
    const EPS: f32 = 1e-8;

    fn new(pos: Pos, parent: Weak<RefCell<Node>>, initial_policy_value: QValue) -> Node {
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
    fn exploration_value(&self) -> QValue {
        let parent_visit_count = self
            .parent
            .upgrade()
            .map_or(self.visit_count as f32, |parent| {
                parent.borrow().visit_count as f32
            }) as f32;
        let exploration_value = (parent_visit_count.ln() / (self.visit_count as f32 + 1.)).sqrt();
        exploration_value * (self.initial_policy_value + Self::EPS)
    }

    /// The UCT value of this node. Represents the lucrativeness of this node according to MCTS.
    /// Because [Self::uct_value] is called from the perspective of the *parent* node, we negate
    /// the exploration value.
    fn uct_value(&self, c_exploration: f32) -> QValue {
        -self.q_with_penalty() + c_exploration * self.exploration_value()
    }

    /// Whether the game is over (won, los, draw) from this position.
    fn is_terminal(&self) -> bool {
        self.pos.is_terminal_state().is_some()
    }

    /// Uses the child counts as weights to determine the implied policy from this position.
    fn policy(&self) -> Policy {
        if let Some(children) = &self.children {
            let child_counts = policy_from_iter(children.iter().map(|maybe_child| {
                maybe_child
                    .as_ref()
                    .map_or(0., |child_ref| child_ref.borrow().visit_count as f32)
            }));
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
/// Temperature=0.0 is argmax, temperature=1.0 is a noop.
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
