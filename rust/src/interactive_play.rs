use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};

use crate::{
    c4r::{Move, Pos},
    mcts::MctsGame,
    types::{EvalPosT, GameMetadata, Policy, QValue},
};

/// Enables interactive play with a game using MCTS.
#[derive(Clone, Debug)]
pub struct InteractivePlay<E: EvalPosT> {
    state: Arc<Mutex<State<E>>>,
}

impl<E: EvalPosT + Send + Sync + 'static> InteractivePlay<E> {
    pub fn new(
        eval_pos: E,
        max_mcts_iterations: usize,
        c_exploration: f32,
        c_ply_penalty: f32,
    ) -> Self {
        Self::new_from_pos(
            Pos::default(),
            eval_pos,
            max_mcts_iterations,
            c_exploration,
            c_ply_penalty,
        )
    }

    pub fn new_from_pos(
        pos: Pos,
        eval_pos: E,
        max_mcts_iterations: usize,
        c_exploration: f32,
        c_ply_penalty: f32,
    ) -> Self {
        let state = State {
            eval_pos,
            game: MctsGame::new_from_pos(pos, GameMetadata::default()),
            max_mcts_iterations,
            c_exploration,
            c_ply_penalty,
            bg_thread_running: false,
        };

        let ret = Self {
            state: Arc::new(Mutex::new(state)),
        };
        ret.lock_and_ensure_bg_thread();
        ret
    }

    /// Returns a snapshot of the current state of the interactive play.
    pub fn snapshot(&self) -> Snapshot {
        let state_guard = self.state.lock();
        state_guard.snapshot()
    }

    /// Increases the number of MCTS iterations by the given amount.
    pub fn increase_mcts_iters(&self, n: usize) {
        let mut state_guard = self.state.lock();
        state_guard.max_mcts_iterations += n;
        self.ensure_bg_thread(state_guard);
    }

    /// Makes the given move.
    pub fn make_move(&self, mov: Move) {
        let mut state_guard = self.state.lock();
        let move_successful = state_guard.make_move(mov);
        if move_successful {
            self.ensure_bg_thread(state_guard);
        }
    }

    /// Makes a random move using the given temperature.
    pub fn make_random_move(&self, temperature: f32) {
        let mut state_guard = self.state.lock();
        let move_successful = state_guard.make_random_move(temperature);
        if move_successful {
            self.ensure_bg_thread(state_guard);
        }
    }

    /// Resets the game to the starting position.
    pub fn reset_game(&self) {
        let mut state_guard = self.state.lock();
        state_guard.game.reset_game();
        self.ensure_bg_thread(state_guard);
    }

    /// Undoes the last move if possible.
    pub fn undo_move(&self) {
        let mut state_guard = self.state.lock();
        let move_successful = state_guard.game.undo_move();
        if move_successful {
            self.ensure_bg_thread(state_guard);
        }
    }

    /// Locks the state and then ensures that the background thread is running.
    fn lock_and_ensure_bg_thread(&self) {
        let state_guard = self.state.lock();
        self.ensure_bg_thread(state_guard);
    }

    /// Ensures that the background thread is running with the given lock.
    fn ensure_bg_thread(&self, mut state_guard: MutexGuard<State<E>>) {
        if state_guard.bg_thread_should_stop() || state_guard.bg_thread_running {
            return;
        }

        state_guard.bg_thread_running = true;
        drop(state_guard);
        let state = Arc::clone(&self.state);
        std::thread::Builder::new()
            .name("mcts_bg_thread".into())
            .spawn(move || loop {
                let mut state_guard = state.lock();
                if state_guard.bg_thread_should_stop() {
                    state_guard.bg_thread_running = false;
                    return;
                }

                state_guard.bg_thread_tick();
            })
            .expect("failed to start mcts_bg_thread");
    }
}

/// The state of the interactive play.
#[derive(Debug)]
struct State<E: EvalPosT> {
    eval_pos: E,
    game: MctsGame,
    max_mcts_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    bg_thread_running: bool,
}

impl<E: EvalPosT> State<E> {
    fn snapshot(&self) -> Snapshot {
        let mut pos = self.game.root_pos();
        let mut q_penalty = self.game.root_q_with_penalty();
        let mut q_no_penalty = self.game.root_q_no_penalty();
        if pos.ply() % 2 == 1 {
            pos = pos.invert();
            q_penalty = -q_penalty;
            q_no_penalty = -q_no_penalty;
        }

        Snapshot {
            pos,
            policy: self.game.root_policy(),
            q_penalty,
            q_no_penalty,
            n_mcts_iterations: self.game.root_visit_count(),
            max_mcts_iterations: self.max_mcts_iterations,
            c_exploration: self.c_exploration,
            c_ply_penalty: self.c_ply_penalty,
            bg_thread_running: self.bg_thread_running,
        }
    }

    /// Makes the given move returning whether it was successfully played.
    pub fn make_move(&mut self, mov: Move) -> bool {
        let pos = &self.game.root_pos();
        if pos.is_terminal_state().is_some() || !pos.legal_moves()[mov] {
            return false;
        }
        self.game.make_move(mov, self.c_exploration);
        true
    }

    /// Makes a random move using the given temperature.
    pub fn make_random_move(&mut self, temperature: f32) -> bool {
        if self.game.root_pos().is_terminal_state().is_some() {
            return false;
        }
        self.game.make_random_move(self.c_exploration, temperature);
        true
    }

    /// Returns true if the background thread should stop.
    fn bg_thread_should_stop(&self) -> bool {
        self.game.root_visit_count() >= self.max_mcts_iterations
            || self.game.root_pos().is_terminal_state().is_some()
    }

    /// A single tick of the background thread.
    /// Performs a single MCTS iteration and updates the game state accordingly.
    fn bg_thread_tick(&mut self) {
        // TODO: Preemptively forward pass additional pos leafs and store their results in cache
        // to maximize GPU parallelism instead of evaluating a single pos at a time.
        let leaf_pos = self.game.leaf_pos();
        let eval = self
            .eval_pos
            .eval_pos(0, vec![leaf_pos])
            .into_iter()
            .next()
            .unwrap();

        self.game.on_received_policy(
            eval.policy,
            eval.q_penalty,
            eval.q_no_penalty,
            self.c_exploration,
            self.c_ply_penalty,
        );

        let snapshot = self.snapshot();
        log::debug!(
            "bg_thread_tick finished; root_policy: {:?}\nroot_value: {:.2}",
            snapshot.policy,
            snapshot.q_penalty
        );
    }
}

/// A snapshot of the current state of the interactive play. The snapshot is always from the
/// perspective of Player 0 (i.e. odd plys have inverted [Pos] and [QValue] to reflect
/// Player 0's perspective).
#[derive(Debug)]
pub struct Snapshot {
    pub pos: Pos,
    pub policy: Policy,
    pub q_penalty: QValue,
    pub q_no_penalty: QValue,
    pub n_mcts_iterations: usize,
    pub max_mcts_iterations: usize,
    pub c_exploration: f32,
    pub c_ply_penalty: f32,
    pub bg_thread_running: bool,
}

#[cfg(test)]
mod tests {
    use more_asserts::assert_ge;

    use crate::{c4r::Pos, self_play::tests::UniformEvalPos};

    use super::{InteractivePlay, Snapshot};

    const TEST_C_EXPLORATION: f32 = 4.0;
    const TEST_C_PLY_PENALTY: f32 = 0.01;

    impl InteractivePlay<UniformEvalPos> {
        fn new_test(pos: Pos, max_mcts_iters: usize) -> InteractivePlay<UniformEvalPos> {
            InteractivePlay::new_from_pos(
                pos,
                UniformEvalPos {},
                max_mcts_iters,
                TEST_C_EXPLORATION,
                TEST_C_PLY_PENALTY,
            )
        }

        fn block_then_snapshot(&self) -> Snapshot {
            loop {
                let state_guard = self.state.lock();
                if !state_guard.bg_thread_running {
                    return state_guard.snapshot();
                }
                std::thread::yield_now();
            }
        }
    }

    #[test]
    fn forcing_position() {
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
        let play = InteractivePlay::new_test(pos, 10_000);
        let snapshot = play.block_then_snapshot();
        let winning_moves = snapshot.policy[1] + snapshot.policy[4];
        assert_ge!(winning_moves, 0.98);
        assert_ge!(snapshot.q_penalty, 0.91);
        assert_ge!(snapshot.q_no_penalty, 0.98);

        play.make_move(1);
        let snapshot = play.block_then_snapshot();
        assert_ge!(snapshot.q_penalty, 0.91);
        assert_ge!(snapshot.q_no_penalty, 0.98);

        play.make_move(0);
        let snapshot = play.block_then_snapshot();
        assert_ge!(snapshot.policy[4], 0.99);
        assert_ge!(snapshot.q_penalty, 0.91);
        assert_ge!(snapshot.q_no_penalty, 0.98);
    }
}
