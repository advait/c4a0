#![allow(dead_code)]
pub mod analysis;
pub mod c4r;
#[cfg(feature = "python")]
mod interactive_play;
pub mod labels;
pub mod mcts;
mod minimax;
pub mod native_solver;
#[cfg(feature = "python")]
mod pybridge;
#[cfg(feature = "python")]
mod self_play;
#[cfg(feature = "python")]
mod tui;
pub mod types;
mod utils;

#[cfg(feature = "python")]
use c4r::Pos;
#[cfg(feature = "python")]
use env_logger::Env;
#[cfg(feature = "python")]
use pybridge::PlayGamesResult;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use types::{GameMetadata, GameResult, Sample};

/// A Python module implemented in Rust.
///
/// Maturin installs this extension as `c4a0_rust._native`; the Python
/// `c4a0_rust` package re-exports the native API for backwards-compatible
/// top-level imports.
#[cfg(feature = "python")]
#[pymodule(gil_used = true)]
#[pyo3(name = "_native")]
fn c4a0_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    m.add("N_COLS", Pos::N_COLS)?;
    m.add("N_ROWS", Pos::N_ROWS)?;
    m.add("BUF_N_CHANNELS", Pos::BUF_N_CHANNELS)?;

    m.add_class::<GameMetadata>()?;
    m.add_class::<GameResult>()?;
    m.add_class::<Sample>()?;
    m.add_class::<PlayGamesResult>()?;

    m.add_function(wrap_pyfunction!(pybridge::play_games, m)?)?;
    m.add_function(wrap_pyfunction!(pybridge::run_tui, m)?)?;
    m.add_function(wrap_pyfunction!(pybridge::encode_positions, m)?)?;
    m.add_function(wrap_pyfunction!(pybridge::mcts_policies, m)?)?;
    m.add_function(wrap_pyfunction!(pybridge::label_positions, m)?)?;

    Ok(())
}
