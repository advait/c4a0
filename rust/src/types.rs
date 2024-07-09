use pyo3::pyclass;

use crate::c4r::Pos;

/// Probabilities for how lucrative each column is.
pub type Policy = [f32; Pos::N_COLS];

/// The lucrativeness value of a given position.
pub type PosValue = f32;

/// ID of the Player's NN.
pub type PlayerID = u64;

/// Metadata about a game.
#[derive(Debug, Clone, Default)]
pub struct GameMetadata {
    pub game_id: u64,
    pub player0_id: PlayerID,
    pub player1_id: PlayerID,
}

/// The finished result of a game.
#[derive(Debug, Clone)]
#[pyclass]
pub struct GameResult {
    pub metadata: GameMetadata,
    pub samples: Vec<Sample>,
}

/// A training sample generated via self-play.
#[derive(Debug, Clone)]
#[pyclass]
pub struct Sample {
    pub pos: Pos,
    pub policy: Policy,
    pub value: PosValue,
}

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
