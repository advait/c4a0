use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};

use crate::{
    c4r::{Move, Pos},
    mcts::MctsGame,
    types::{EvalPosT, Policy, PosValue},
};

/// Enables interactive play with a game using MCTS.
#[derive(Clone, Debug)]
pub struct InteractivePlay<E: EvalPosT> {
    state: Arc<Mutex<State<E>>>,
}

impl<E: EvalPosT + Send + Sync + 'static> InteractivePlay<E> {
    pub fn new(eval_pos: E, max_mcts_iterations: usize, exploration_constant: f32) -> Self {
        let state = State {
            eval_pos,
            game: MctsGame::default(),
            max_mcts_iterations,
            exploration_constant,
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

    /// Makes the given move returning whether it was successfully played.
    pub fn make_move(&self, mov: Move) -> bool {
        let mut state_guard = self.state.lock();
        let move_successful = state_guard.make_move(mov);

        if move_successful {
            self.ensure_bg_thread(state_guard);
        }
        move_successful
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
    exploration_constant: f32,
    bg_thread_running: bool,
}

impl<E: EvalPosT> State<E> {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            root_pos: self.game.root_pos().clone(),
            policy: self.game.root_policy(),
            value: self.game.root_value(),
            n_mcts_iterations: self.game.root_visit_count(),
            max_mcts_iterations: self.max_mcts_iterations,
            exploration_constant: self.exploration_constant,
            bg_thread_running: self.bg_thread_running,
        }
    }

    /// Makes the given move returning whether it was successfully played.
    pub fn make_move(&mut self, mov: Move) -> bool {
        if self.game.root_pos().is_terminal_state().is_some() {
            return false;
        }
        self.game.make_move(mov, self.exploration_constant);
        true
    }

    /// Returns true if the background thread should stop.
    fn bg_thread_should_stop(&self) -> bool {
        self.game.root_visit_count() > self.max_mcts_iterations
            || self.game.root_pos().is_terminal_state().is_some()
    }

    /// A single tick of the background thread.
    /// Performs a single MCTS iteration and updates the game state accordingly.
    fn bg_thread_tick(&mut self) {
        // TODO: Preemptively forward pass additional pos leafs and store their results in cache
        // to maximize GPU parallelism instead of evaluating a single pos at a time.
        let leaf_pos = self.game.leaf_pos().clone();
        let eval = self
            .eval_pos
            .eval_pos(0, vec![leaf_pos])
            .into_iter()
            .next()
            .unwrap();

        self.game
            .on_received_policy(eval.policy, eval.value, self.exploration_constant);
    }
}

/// A snapshot of the current state of the interactive play.
#[derive(Debug)]
pub struct Snapshot {
    pub root_pos: Pos,
    pub policy: Policy,
    pub value: PosValue,
    pub n_mcts_iterations: usize,
    pub max_mcts_iterations: usize,
    pub exploration_constant: f32,
    pub bg_thread_running: bool,
}
