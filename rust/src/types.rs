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

/// The lucrativeness value of a given position.
pub type PosValue = f32;

/// ID of the Player's NN.
pub type PlayerID = u64;

/// Evaluate a batch of positions with an NN forward pass.
/// The ordering of the results corresponds to the ordering of the input positions.
pub trait EvalPosT {
    fn eval_pos(&self, player_id: PlayerID, pos: Vec<Pos>) -> Vec<EvalPosResult>;
}

/// The returned output from the forward pass of the NN.
#[derive(Debug, Clone)]
pub struct EvalPosResult {
    pub policy: Policy,  // Probability distribution over moves from the position.
    pub value: PosValue, // Lucrativeness [-1, 1] of the position.
}

/// Metadata about a game.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[pyclass]
pub struct GameMetadata {
    #[pyo3(get)]
    pub game_id: u64,

    #[pyo3(get)]
    pub player0_id: PlayerID,

    #[pyo3(get)]
    pub player1_id: PlayerID,
}

#[pymethods]
impl GameMetadata {
    #[new]
    fn new(game_id: u64, player0_id: PlayerID, player1_id: PlayerID) -> Self {
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

/// A training sample generated via self-play.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Sample {
    pub pos: Pos,
    pub policy: Policy,
    pub value: PosValue,
}

#[pymethods]
impl Sample {
    /// Returns a new sample that is flipped horizontally.
    pub fn flip_h(&self) -> Sample {
        Sample {
            pos: self.pos.flip_h(),
            policy: array::from_fn(|col| self.policy[Pos::N_COLS - 1 - col]),
            value: self.value,
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
    ) {
        let mut pos_buffer = vec![0.0; Pos::BUF_LEN];
        self.pos.write_numpy_buffer(&mut pos_buffer);
        let pos =
            Array3::from_shape_vec([Pos::BUF_N_CHANNELS, Pos::N_ROWS, Pos::N_COLS], pos_buffer)
                .unwrap();
        let pos = PyArray3::from_array_bound(py, &pos);
        let policy = PyArray1::from_slice_bound(py, &self.policy);
        let value = Array0::from_elem([] /* shape */, self.value);
        let value = PyArray0::from_array_bound(py, &value);

        (pos, policy, value)
    }
}