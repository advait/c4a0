use core::panic;
use std::array;

use numpy::{
    ndarray::{Array0, Array3},
    PyArray0, PyArray1, PyArray3,
};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::c4r::Pos;

/// Probabilities for how lucrative each column is.
pub type Policy = [f32; Pos::N_COLS];

/// The lucrativeness value of a given position. This is the objective we are trying to maximize.
pub type QValue = f32;

/// ID of the Model's NN.
pub type ModelID = u64;

/// Evaluate a batch of positions with an NN forward pass.
/// The ordering of the results corresponds to the ordering of the input positions.
pub trait EvalPosT {
    fn eval_pos(&self, model_id: ModelID, pos: Vec<Pos>) -> Vec<EvalPosResult>;
}

/// The returned output from the forward pass of the NN.
#[derive(Debug, Clone)]
pub struct EvalPosResult {
    pub policy: Policy,       // Probability distribution over moves from the position.
    pub q_penalty: QValue,    // Lucrativeness [-1, 1] of the position with ply penalty.
    pub q_no_penalty: QValue, // Lucrativeness [-1, 1] of the position without ply penalty.
}

/// Metadata about a game.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[pyclass]
pub struct GameMetadata {
    #[pyo3(get)]
    pub game_id: u64,

    #[pyo3(get)]
    pub player0_id: ModelID,

    #[pyo3(get)]
    pub player1_id: ModelID,
}

#[pymethods]
impl GameMetadata {
    #[new]
    fn new(game_id: u64, player0_id: ModelID, player1_id: ModelID) -> Self {
        GameMetadata {
            game_id,
            player0_id,
            player1_id,
        }
    }
}

/// The finished result of a game.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct GameResult {
    #[pyo3(get)]
    pub metadata: GameMetadata,

    #[pyo3(get)]
    pub samples: Vec<Sample>,
}

#[pymethods]
impl GameResult {
    /// Returns the score of the game from the perspective of Player 0.
    /// If Player 0 wins, 1.0. If Player 0 loses, 0.0. If it's a draw, 0.5.
    fn player0_score(&self) -> f32 {
        for sample in self.samples.iter() {
            if let Some(terminal) = sample.pos.is_terminal_state() {
                let score = match terminal {
                    crate::c4r::TerminalState::PlayerWin => 1.0,
                    crate::c4r::TerminalState::OpponentWin => 0.0,
                    crate::c4r::TerminalState::Draw => 0.5,
                };

                // When we play positions, we flip the pieces so that the "player to play" is
                // activte. This means the terminal state is from the perspective of the player
                // who is about to player. For odd ply positions, the player to play is player 1
                // so we must flip the score.
                return if sample.pos.ply() % 2 == 1 {
                    1.0 - score
                } else {
                    score
                };
            }
        }

        panic!("player0_score called on an unfinished game");
    }
}

/// A training sample generated via self-play.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Sample {
    pub pos: Pos,
    pub policy: Policy,
    pub q_penalty: QValue,
    pub q_no_penalty: QValue,
}

#[pymethods]
impl Sample {
    /// Returns a new sample that is flipped horizontally.
    pub fn flip_h(&self) -> Sample {
        Sample {
            pos: self.pos.flip_h(),
            policy: array::from_fn(|col| self.policy[Pos::N_COLS - 1 - col]),
            q_penalty: self.q_penalty,
            q_no_penalty: self.q_no_penalty,
        }
    }

    /// [numpy] representation of the sample.
    pub fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray0<f32>>,
        Bound<'py, PyArray0<f32>>,
    ) {
        let mut pos_buffer = vec![0.0; Pos::BUF_LEN];
        self.pos.write_numpy_buffer(&mut pos_buffer);
        let pos =
            Array3::from_shape_vec([Pos::BUF_N_CHANNELS, Pos::N_ROWS, Pos::N_COLS], pos_buffer)
                .unwrap();
        let pos = PyArray3::from_array_bound(py, &pos);
        let policy = PyArray1::from_slice_bound(py, &self.policy);
        let q_penalty = Array0::from_elem([] /* shape */, self.q_penalty);
        let q_penalty = PyArray0::from_array_bound(py, &q_penalty);
        let q_no_penalty = Array0::from_elem([] /* shape */, self.q_no_penalty);
        let q_no_penalty = PyArray0::from_array_bound(py, &q_no_penalty);

        (pos, policy, q_penalty, q_no_penalty)
    }

    /// String representation of the position.
    pub fn pos_str(&self) -> String {
        self.pos.to_string()
    }
}

pub fn policy_from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Policy {
    let mut policy = [0.0; Pos::N_COLS];
    for (i, p) in iter.into_iter().enumerate() {
        policy[i] = p;
    }
    policy
}
