#![allow(dead_code)]
mod c4r;
mod mcts;
mod pybridge;
mod self_play;
mod types;

use c4r::Pos;
use pybridge::SampleBatch;
use pyo3::prelude::*;
use types::{GameResult, Sample};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn c4a0_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("N_COLS", Pos::N_COLS)?;
    m.add("N_ROWS", Pos::N_ROWS)?;

    m.add_class::<GameResult>()?;
    m.add_class::<Sample>()?;
    m.add_class::<SampleBatch>()?;

    m.add_function(wrap_pyfunction!(pybridge::gen_samples, m)?)?;

    Ok(())
}
