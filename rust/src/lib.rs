#![allow(dead_code)]
mod c4r;
mod mcts;
mod pybridge;
mod self_play;

use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn c4a0_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pybridge::gen_samples, m)?)?;
    Ok(())
}
