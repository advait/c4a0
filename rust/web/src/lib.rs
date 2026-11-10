//! The browser player's engine: the same MCTS that `mcts_policies` runs, driven one network
//! evaluation at a time so JavaScript can await the GPU between iterations.

use c4a0_rust::c4r::{Move, Pos, TerminalState};
use c4a0_rust::mcts::MctsGame;
use c4a0_rust::types::GameMetadata;
use wasm_bindgen::prelude::*;

/// The evaluation settings behind the README results table.
const C_EXPLORATION: f32 = 2.0;
const C_PLY_PENALTY: f32 = 0.01;

/// Parses a move string ("3304", 0-indexed columns) into the position it reaches, rejecting
/// illegal moves and moves after the game ended.
fn parse(moves: &str) -> Result<Pos, String> {
    let mut pos = Pos::default();
    for (i, ch) in moves.chars().enumerate() {
        if pos.is_terminal_state().is_some() {
            return Err(format!("move {i} is after the game ended"));
        }
        let col = ch
            .to_digit(10)
            .map(|d| d as Move)
            .filter(|&c| c < Pos::N_COLS)
            .ok_or_else(|| format!("move {i}: {ch:?} is not a column 0-6"))?;
        pos = pos
            .make_move(col)
            .ok_or_else(|| format!("move {i}: column {col} is full"))?;
    }
    Ok(pos)
}

/// The columns that accept a stone: empty if the game is over.
#[wasm_bindgen]
pub fn legal_moves(moves: &str) -> Result<Vec<u8>, JsError> {
    let pos = parse(moves).map_err(|e| JsError::new(&e))?;
    if pos.is_terminal_state().is_some() {
        return Ok(Vec::new());
    }
    Ok((0..Pos::N_COLS)
        .filter(|&c| pos.legal_moves()[c])
        .map(|c| c as u8)
        .collect())
}

/// "playing", "draw", or "win": the player who made the last move connected four.
#[wasm_bindgen]
pub fn outcome(moves: &str) -> Result<String, JsError> {
    let pos = parse(moves).map_err(|e| JsError::new(&e))?;
    Ok(match pos.is_terminal_state() {
        None => "playing",
        Some(TerminalState::Draw) => "draw",
        // Terminal states are seen by the player to move, who just lost.
        Some(TerminalState::OpponentWin) | Some(TerminalState::PlayerWin) => "win",
    }
    .into())
}

/// One move's search from a fresh root, as the evaluation suites run it.
#[wasm_bindgen]
pub struct Search {
    game: MctsGame,
    iterations: usize,
}

#[wasm_bindgen]
impl Search {
    #[wasm_bindgen(constructor)]
    pub fn new(
        moves: &str,
        iterations: usize,
        leaf_minimax_depth: usize,
    ) -> Result<Search, JsError> {
        let pos = parse(moves).map_err(|e| JsError::new(&e))?;
        if pos.is_terminal_state().is_some() {
            return Err(JsError::new("cannot search a finished game"));
        }
        let game = MctsGame::new_from_pos(pos, GameMetadata::default())
            .with_leaf_minimax_depth(leaf_minimax_depth);
        Ok(Search { game, iterations })
    }

    /// The next position the network must evaluate, as the (2, 6, 7) mover/opponent planes, or
    /// `undefined` once the budget is spent or the root is proven. Proven leaves back up their
    /// exact value here without a network call.
    pub fn next_leaf(&mut self) -> Option<Vec<f32>> {
        while self.game.needs_search(self.iterations) {
            if self.game.leaf_needs_eval() {
                let mut buf = vec![0.0; Pos::BUF_LEN];
                self.game.leaf_pos().write_numpy_buffer(&mut buf);
                return Some(buf);
            }
            self.receive(&MctsGame::UNIFORM_POLICY, 0.0, 0.0);
        }
        None
    }

    /// The network's answer for the last `next_leaf`: 7 policy log-probabilities and both values.
    pub fn receive(&mut self, policy_logprobs: &[f32], q_penalty: f32, q_no_penalty: f32) {
        let policy = std::array::from_fn(|c| policy_logprobs[c]);
        self.game.on_received_policy(
            policy,
            q_penalty,
            q_no_penalty,
            C_EXPLORATION,
            C_PLY_PENALTY,
        );
    }

    /// Share of root visits per column (all mass on the best moves once the root is proven).
    pub fn policy(&self) -> Vec<f32> {
        self.game.root_policy().to_vec()
    }

    /// Completed iterations.
    pub fn visits(&self) -> usize {
        self.game.root_visit_count()
    }

    /// Expected result for the player to move, in [-1, 1] (no ply penalty).
    pub fn value(&self) -> f32 {
        self.game.root_q(C_PLY_PENALTY).1
    }

    /// The most-visited legal column, lowest index on ties: the move the evaluation scores.
    pub fn best_move(&self) -> u8 {
        let pos = self.game.root_pos();
        let legal = pos.legal_moves();
        let policy = self.game.root_policy();
        (0..Pos::N_COLS)
            .filter(|&c| legal[c])
            .fold(None, |best: Option<Move>, c| match best {
                Some(b) if policy[b] >= policy[c] => Some(b),
                _ => Some(c),
            })
            .expect("a searched position has a legal move") as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a search the way the page does, with a uniform stand-in for the network.
    fn search(moves: &str, iterations: usize, depth: usize) -> Search {
        let mut s = Search::new(moves, iterations, depth).unwrap();
        while s.next_leaf().is_some() {
            s.receive(&[(1.0f32 / 7.0).ln(); 7], 0.0, 0.0);
        }
        s
    }

    #[test]
    fn takes_an_immediate_win() {
        // First player has three in column 0; the move is column 0 even with a blind network.
        assert_eq!(search("010101", 64, 0).best_move(), 0);
    }

    #[test]
    fn leaf_search_proves_the_win_without_the_network() {
        let mut s = Search::new("010101", 64, 2).unwrap();
        assert!(s.next_leaf().is_none(), "proven root needs no evaluation");
        assert_eq!(s.best_move(), 0);
    }

    #[test]
    fn blocks_the_opponent() {
        // Second player must block column 0.
        assert_eq!(search("01010", 256, 4).best_move(), 0);
    }

    #[test]
    fn spends_the_budget() {
        assert_eq!(search("", 50, 0).visits(), 50);
    }

    #[test]
    fn rejects_bad_move_strings() {
        assert!(parse("7").is_err());
        assert!(parse("0000000").is_err(), "column 0 holds six");
        assert!(parse("01010101").is_err(), "move after the win");
        assert!(parse("0101010").unwrap().is_terminal_state().is_some());
    }
}
