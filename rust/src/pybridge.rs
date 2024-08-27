use numpy::{ndarray::Array4, IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyList},
};
use serde::{Deserialize, Serialize};

use crate::{
    c4r::Pos,
    self_play::self_play,
    solver::CachingSolver,
    tui,
    types::{EvalPosResult, EvalPosT, GameMetadata, GameResult, ModelID, Policy, Sample},
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

/// Play games via MCTS. This is a python wrapper around [self_play].
/// `reqs` is a list of [GameMetadata] that describes the games to play.
/// The `py_eval_pos_cb` callback is expected to be a pytorch model that runs on the GPU.
#[pyfunction]
pub fn play_games<'py>(
    py: Python<'py>,
    reqs: &Bound<'py, PyList>,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
) -> PyResult<PlayGamesResult> {
    let reqs: Vec<GameMetadata> = reqs.extract().expect("error extracting reqs");

    let eval_pos = PyEvalPos {
        py_eval_pos_cb: py_eval_pos_cb.to_object(py),
    };

    let results = {
        // Start background processing threads while releasing the GIL with allow_threads.
        // This allows other python threads (e.g. pytorch) to continue while we generate training
        // samples. When we need to call the py_eval_pos callback, we will re-acquire the GIL.
        py.allow_threads(move || {
            self_play(
                eval_pos,
                reqs,
                max_nn_batch_size,
                n_mcts_iterations,
                c_exploration,
                c_ply_penalty,
            )
        })
    };

    Ok(PlayGamesResult { results })
}

/// The result of [play_games].
/// Note we explicitly spcify pyclass(module="c4a0_rust") as the module name is required in
/// order for pickling to work.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "c4a0_rust")]
pub struct PlayGamesResult {
    #[pyo3(get)]
    pub results: Vec<GameResult>,
}

#[pymethods]
impl PlayGamesResult {
    /// Empty constructor is required for unpickling.
    #[new]
    fn new() -> Self {
        PlayGamesResult { results: vec![] }
    }

    fn to_cbor(&self, py: Python) -> PyResult<PyObject> {
        let cbor = serde_cbor::to_vec(self).map_err(pyify_err)?;
        Ok(PyBytes::new_bound(py, &cbor).into())
    }

    #[staticmethod]
    fn from_cbor(_py: Python, cbor: &[u8]) -> PyResult<Self> {
        serde_cbor::from_slice(cbor).map_err(pyify_err)
    }

    /// Used for pickling serialization.
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        self.to_cbor(py)
    }

    /// Used for pickling deserialization.
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let cbor: &[u8] = state.extract(py)?;
        *self = Self::from_cbor(py, cbor)?;
        Ok(())
    }

    /// Combine two PlayGamesResult objects.
    fn __add__<'py>(&mut self, py: Python<'py>, other: PyObject) -> PyResult<Self> {
        let other = other.extract::<PlayGamesResult>(py)?;
        Ok(PlayGamesResult {
            results: self
                .results
                .iter()
                .chain(other.results.iter())
                .cloned()
                .collect(),
        })
    }

    /// Splits the results into training and test datasets.
    /// Ensures that whole games end up in either the training set or test set.
    /// Expects `train_frac` to be in [0, 1].
    fn split_train_test(
        &mut self,
        train_frac: f32,
        seed: u64,
    ) -> PyResult<(Vec<Sample>, Vec<Sample>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.results.shuffle(&mut rng);
        let n_train = (self.results.len() as f32 * train_frac).round() as usize;
        let (train, test) = self.results.split_at(n_train);
        Ok((
            train.into_iter().flat_map(|r| r.samples.clone()).collect(),
            test.into_iter().flat_map(|r| r.samples.clone()).collect(),
        ))
    }

    /// Scores the policies in the results using the given solver.
    /// `solver_path` is the path to the solver binary, see:
    ///     https://github.com/PascalPons/connect4
    /// `solver_book_path` is the path to the solver book:
    ///     https://github.com/PascalPons/connect4/releases/tag/book?ts=2
    /// `solution_cache_path` is the path to the solution cache file which will be created if it
    /// is missing.
    fn score_policies(
        &self,
        solver_path: String,
        solver_book_path: String,
        solution_cache_path: String,
    ) -> PyResult<f32> {
        let solver = CachingSolver::new(solver_path, solver_book_path, solution_cache_path);
        let pos_and_policies = self
            .results
            .iter()
            .flat_map(|r| r.samples.iter())
            .filter(|s| s.pos.is_terminal_state().is_none())
            .map(|p| (p.pos.clone(), p.policy.clone()))
            .collect::<Vec<_>>();
        let scores = solver.score_policies(pos_and_policies).map_err(pyify_err)?;
        let n_scores = scores.len();
        let avg_score = scores.into_iter().sum::<f32>() / n_scores as f32;
        Ok(avg_score)
    }

    /// Returns the number of unique positions in the results.
    fn unique_positions(&self) -> usize {
        self.results
            .iter()
            .flat_map(|r| r.samples.iter())
            .map(|s| s.pos.clone())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
}

/// [EvalPosT] implementation that calls the `py_eval_pos_cb` python callback.
struct PyEvalPos {
    py_eval_pos_cb: PyObject,
}

impl EvalPosT for PyEvalPos {
    /// Evaluates a batch of positions by calling the [Self::py_eval_pos_cb] callback.
    /// This is intended to be a pytorch model that runs on the GPU. Because this is a python
    /// call we need to first re-acquire the GIL to call this function from a background thread
    /// before performing the callback.
    fn eval_pos(&self, model_id: ModelID, pos: Vec<Pos>) -> Vec<EvalPosResult> {
        Python::with_gil(|py| {
            let batch_size = pos.len();
            let pos_batch = create_pos_batch(py, &pos);

            let (policy, q_penalty, q_no_penalty): (
                PyReadonlyArray2<f32>,
                PyReadonlyArray1<f32>,
                PyReadonlyArray1<f32>,
            ) = (&self
                .py_eval_pos_cb
                .call_bound(py, (model_id, pos_batch), None)
                .expect("Failed to call py_eval_pos_cb"))
                .extract(py)
                .expect("Failed to extract result");

            let policy = policy.as_slice().expect("Failed to get policy slice");
            let q_penalty = q_penalty.as_slice().expect("Failed to get value slice");
            let q_no_penalty = q_no_penalty.as_slice().expect("Failed to get value slice");

            (0..batch_size)
                .map(|i| EvalPosResult {
                    policy: policy_from_slice(&policy[i * Pos::N_COLS..(i + 1) * Pos::N_COLS]),
                    q_penalty: q_penalty[i],
                    q_no_penalty: q_no_penalty[i],
                })
                .collect()
        })
    }
}

/// Creates a batch of positions in tensor format.
fn create_pos_batch<'py>(py: Python<'py>, positions: &Vec<Pos>) -> Bound<'py, PyArray4<f32>> {
    let mut buffer = vec![0.0; positions.len() * Pos::BUF_LEN];
    for i in 0..positions.len() {
        let pos = &positions[i];
        let pos_buffer = &mut buffer[i * Pos::BUF_LEN..(i + 1) * Pos::BUF_LEN];
        pos.write_numpy_buffer(pos_buffer);
    }

    Array4::from_shape_vec(
        (
            positions.len(),
            Pos::BUF_N_CHANNELS,
            Pos::N_ROWS,
            Pos::N_COLS,
        ),
        buffer,
    )
    .expect("Failed to create Array4 from buffer")
    .into_pyarray_bound(py)
}

/// Convert a slice of probabilities into a [Policy].
fn policy_from_slice(policy: &[f32]) -> Policy {
    debug_assert_eq!(policy.len(), Pos::N_COLS);
    let mut ret = Policy::default();
    ret.copy_from_slice(policy);
    ret
}

#[pyfunction]
pub fn run_tui<'py>(
    py: Python<'py>,
    py_eval_pos_cb: &Bound<'py, PyAny>,
    max_mcts_iters: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
) -> PyResult<()> {
    let eval_pos = PyEvalPos {
        py_eval_pos_cb: py_eval_pos_cb.to_object(py),
    };

    // Start the TUI while releasing the GIL with allow_threads.
    py.allow_threads(move || {
        let mut terminal = tui::init()?;
        let mut app = tui::App::new(eval_pos, max_mcts_iters, c_exploration, c_ply_penalty);
        app.run(&mut terminal)?;
        tui::restore()?;
        Ok(())
    })
}

/// Convert a Rust error into a Python exception.
fn pyify_err<T>(e: T) -> PyErr
where
    T: std::fmt::Debug,
{
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e))
}
