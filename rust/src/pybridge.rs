use numpy::{
    ndarray::{Array1, Array2, Array4},
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyList},
};
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{search_positions, search_roots},
    c4r::{Move, Pos},
    labels::{moves_string, parse_moves, sample_positions, Split},
    self_play::{self_play, SelfPlayOptions},
    solver::solve_parallel,
    tui,
    types::{EvalPosResult, EvalPosT, GameMetadata, GameResult, ModelID, Sample},
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

/// Play games via MCTS. This is a python wrapper around [self_play].
/// `reqs` is a list of [GameMetadata] that describes the games to play.
/// The `py_eval_pos_cb` callback is expected to be a pytorch model that runs on the GPU.
#[pyfunction(signature = (reqs, max_nn_batch_size, n_mcts_iterations, c_exploration, c_ply_penalty, py_eval_pos_cb, root_noise_alpha=None, root_noise_frac=0.0, value_target_q_weight=0.0, random_start_frac=0.0, random_start_max_ply=12, leaf_minimax_depth=0))]
pub fn play_games<'py>(
    py: Python<'py>,
    reqs: &Bound<'py, PyList>,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
    root_noise_alpha: Option<f32>,
    root_noise_frac: f32,
    value_target_q_weight: f32,
    random_start_frac: f32,
    random_start_max_ply: usize,
    leaf_minimax_depth: usize,
) -> PyResult<PlayGamesResult> {
    let options = SelfPlayOptions {
        root_noise_alpha,
        root_noise_frac,
        value_target_q_weight,
        random_start_frac,
        random_start_max_ply,
        leaf_minimax_depth,
    };
    options.validate().map_err(PyValueError::new_err)?;
    let reqs: Vec<GameMetadata> = reqs.extract().expect("error extracting reqs");

    let eval_pos = PyEvalPos {
        py_eval_pos_cb: py_eval_pos_cb.clone().unbind(),
    };

    let results = {
        // Start background processing threads while detaching from Python.
        // This allows other python threads (e.g. pytorch) to continue while we generate training
        // samples. When we need to call the py_eval_pos callback, we will re-attach.
        py.detach(move || {
            self_play(
                eval_pos,
                reqs,
                max_nn_batch_size,
                n_mcts_iterations,
                c_exploration,
                c_ply_penalty,
                options,
            )
        })
    };

    Ok(PlayGamesResult { results })
}

/// The result of [play_games].
/// Note we explicitly spcify pyclass(module="c4a0_rust") as the module name is required in
/// order for pickling to work.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "c4a0_rust", from_py_object)]
pub struct PlayGamesResult {
    #[pyo3(get)]
    pub results: Vec<GameResult>,
}

type SampleArrays<'py> = (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);

/// Original samples followed by their mirrors, exactly as the Python augmentation did.
fn sample_arrays<'py>(py: Python<'py>, games: &[&GameResult]) -> SampleArrays<'py> {
    let n = games.iter().map(|g| g.samples.len()).sum::<usize>() * 2;
    let mut positions = vec![0.0; n * Pos::BUF_LEN];
    let mut policies = Vec::with_capacity(n * Pos::N_COLS);
    let mut q_penalty = Vec::with_capacity(n);
    let mut q_no_penalty = Vec::with_capacity(n);
    let mut row = 0;
    for mirror in [false, true] {
        for sample in games.iter().flat_map(|g| &g.samples) {
            let flipped;
            let sample = if mirror {
                flipped = sample.flip_h();
                &flipped
            } else {
                sample
            };
            sample
                .pos
                .write_numpy_buffer(&mut positions[row * Pos::BUF_LEN..(row + 1) * Pos::BUF_LEN]);
            policies.extend_from_slice(&sample.policy);
            q_penalty.push(sample.q_penalty);
            q_no_penalty.push(sample.q_no_penalty);
            row += 1;
        }
    }
    (
        Array4::from_shape_vec(
            (n, Pos::BUF_N_CHANNELS, Pos::N_ROWS, Pos::N_COLS),
            positions,
        )
        .expect("sample position shape")
        .into_pyarray(py),
        Array2::from_shape_vec((n, Pos::N_COLS), policies)
            .expect("sample policy shape")
            .into_pyarray(py),
        Array1::from_vec(q_penalty).into_pyarray(py),
        Array1::from_vec(q_no_penalty).into_pyarray(py),
    )
}

#[pymethods]
impl PlayGamesResult {
    /// Empty constructor is required for unpickling.
    #[new]
    fn new() -> Self {
        PlayGamesResult { results: vec![] }
    }

    fn to_cbor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let cbor = serde_cbor::to_vec(self).map_err(pyify_err)?;
        Ok(PyBytes::new(py, &cbor))
    }

    #[staticmethod]
    fn from_cbor(_py: Python, cbor: &[u8]) -> PyResult<Self> {
        serde_cbor::from_slice(cbor).map_err(pyify_err)
    }

    /// Used for pickling serialization.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_cbor(py)
    }

    /// Used for pickling deserialization.
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        *self = serde_cbor::from_slice(state).map_err(pyify_err)?;
        Ok(())
    }

    /// Combine two PlayGamesResult objects.
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = other.extract::<PlayGamesResult>()?;
        Ok(PlayGamesResult {
            results: self
                .results
                .iter()
                .chain(other.results.iter())
                .cloned()
                .collect(),
        })
    }

    fn __len__(&self) -> usize {
        self.results.len()
    }

    /// Clone the replay window once rather than cloning its growing prefix G-1 times.
    #[staticmethod]
    fn concat(games: Vec<PyRef<'_, PlayGamesResult>>) -> Self {
        Self {
            results: games
                .iter()
                .flat_map(|g| g.results.iter().cloned())
                .collect(),
        }
    }

    /// Splits the results into training and test datasets.
    /// Ensures that whole games end up in either the training set or test set.
    /// Expects `train_frac` to be in [0, 1].
    fn split_train_test(&self, train_frac: f32, seed: u64) -> PyResult<(Vec<Sample>, Vec<Sample>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut results = self.results.clone();
        results.shuffle(&mut rng);
        let n_train = (results.len() as f32 * train_frac).round() as usize;
        let (train, test) = results.split_at(n_train);
        Ok((
            train.iter().flat_map(|r| r.samples.clone()).collect(),
            test.iter().flat_map(|r| r.samples.clone()).collect(),
        ))
    }

    /// Same split and sample order, without cloning games or crossing Python per sample.
    fn split_train_test_numpy<'py>(
        &self,
        py: Python<'py>,
        train_frac: f32,
        seed: u64,
    ) -> PyResult<(SampleArrays<'py>, SampleArrays<'py>)> {
        if !train_frac.is_finite() || !(0.0..=1.0).contains(&train_frac) {
            return Err(PyValueError::new_err("train_frac must be in [0, 1]"));
        }
        let mut games: Vec<_> = self.results.iter().collect();
        games.shuffle(&mut StdRng::seed_from_u64(seed));
        let n_train = (games.len() as f32 * train_frac).round() as usize;
        let (train, test) = games.split_at(n_train);
        Ok((sample_arrays(py, train), sample_arrays(py, test)))
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
    py_eval_pos_cb: Py<PyAny>,
}

impl EvalPosT for PyEvalPos {
    /// Evaluates a batch of positions by calling the [Self::py_eval_pos_cb] callback.
    /// This is intended to be a pytorch model that runs on the GPU. Because this is a python
    /// call we need to first attach to Python from a background thread before performing the
    /// callback.
    fn eval_pos(&self, model_id: ModelID, pos: Vec<Pos>) -> Vec<EvalPosResult> {
        Python::attach(|py| {
            call_eval_cb(py, self.py_eval_pos_cb.bind(py), model_id, &pos)
                .expect("Failed to call py_eval_pos_cb")
        })
    }
}

/// Calls `cb(model_id, pos[B,2,6,7]) -> (policy_logprobs[B,7], q_penalty[B], q_no_penalty[B])`
/// and checks the output shapes.
fn call_eval_cb<'py>(
    py: Python<'py>,
    cb: &Bound<'py, PyAny>,
    model_id: ModelID,
    pos: &[Pos],
) -> PyResult<Vec<EvalPosResult>> {
    let batch_size = pos.len();
    let ret = cb.call1((model_id, create_pos_batch(py, pos)))?;
    let (policy, q_penalty, q_no_penalty): (
        PyReadonlyArray2<f32>,
        PyReadonlyArray1<f32>,
        PyReadonlyArray1<f32>,
    ) = ret.extract()?;
    if policy.shape() != [batch_size, Pos::N_COLS]
        || q_penalty.len() != batch_size
        || q_no_penalty.len() != batch_size
    {
        return Err(PyValueError::new_err(format!(
            "eval callback returned shapes {:?}, {}, {} for a batch of {}",
            policy.shape(),
            q_penalty.len(),
            q_no_penalty.len(),
            batch_size
        )));
    }
    let policy = policy.as_array();
    let q_penalty = q_penalty.as_array();
    let q_no_penalty = q_no_penalty.as_array();
    // A NaN reaching MCTS panics in its float comparisons; reject it here with a Python error.
    if policy.iter().any(|p| p.is_nan())
        || q_penalty
            .iter()
            .chain(q_no_penalty.iter())
            .any(|q| !q.is_finite())
    {
        return Err(PyValueError::new_err(
            "eval callback returned NaN policy or non-finite values",
        ));
    }
    Ok((0..batch_size)
        .map(|i| EvalPosResult {
            policy: std::array::from_fn(|c| policy[[i, c]]),
            q_penalty: q_penalty[i],
            q_no_penalty: q_no_penalty[i],
        })
        .collect())
}

/// Parses move strings (0-indexed column digits), raising ValueError on the first invalid one.
fn parse_all(moves: &[String]) -> PyResult<Vec<Pos>> {
    moves
        .iter()
        .map(|m| parse_moves(m).map_err(PyValueError::new_err))
        .collect()
}

/// Encodes positions given as move strings into the network input, shape (N, 2, 6, 7).
#[pyfunction]
pub fn encode_positions<'py>(
    py: Python<'py>,
    moves: Vec<String>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    Ok(create_pos_batch(py, &parse_all(&moves)?))
}

/// Runs evaluation MCTS ([search_positions]) from every non-terminal root and returns the root
/// visit policies, shape (N, 7). Single-threaded: `py_eval_pos_cb` is called on this thread with
/// model_id 0 and its exceptions propagate. `leaf_minimax_depth` > 0 proves leaves by exact
/// search to that depth before evaluating them.
#[pyfunction(signature = (moves, n_mcts_iterations, c_exploration, c_ply_penalty, py_eval_pos_cb, leaf_minimax_depth=0))]
pub fn mcts_policies<'py>(
    py: Python<'py>,
    moves: Vec<String>,
    n_mcts_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
    leaf_minimax_depth: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let roots = parse_all(&moves)?;
    if let Some(i) = roots.iter().position(|p| p.is_terminal_state().is_some()) {
        return Err(PyValueError::new_err(format!(
            "{:?}: cannot search a finished game",
            moves[i]
        )));
    }
    if n_mcts_iterations < 2 {
        return Err(PyValueError::new_err(
            "n_mcts_iterations must be at least 2",
        ));
    }
    let policies = search_positions(
        |batch| call_eval_cb(py, py_eval_pos_cb, 0, &batch),
        &roots,
        n_mcts_iterations,
        c_exploration,
        c_ply_penalty,
        leaf_minimax_depth,
    )?;
    let flat: Vec<f32> = policies.iter().flatten().copied().collect();
    Ok(Array2::from_shape_vec((policies.len(), Pos::N_COLS), flat)
        .expect("policy buffer shape")
        .into_pyarray(py))
}

/// [mcts_policies], also returning each root's (q_penalty, q_no_penalty), shapes (N,) each: a
/// deeper search's value targets for distilling it into the network.
#[pyfunction(signature = (moves, n_mcts_iterations, c_exploration, c_ply_penalty, py_eval_pos_cb, leaf_minimax_depth=0))]
#[allow(clippy::type_complexity)]
pub fn mcts_search<'py>(
    py: Python<'py>,
    moves: Vec<String>,
    n_mcts_iterations: usize,
    c_exploration: f32,
    c_ply_penalty: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
    leaf_minimax_depth: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let roots = parse_all(&moves)?;
    if let Some(i) = roots.iter().position(|p| p.is_terminal_state().is_some()) {
        return Err(PyValueError::new_err(format!(
            "{:?}: cannot search a finished game",
            moves[i]
        )));
    }
    if n_mcts_iterations < 2 {
        return Err(PyValueError::new_err(
            "n_mcts_iterations must be at least 2",
        ));
    }
    let searched = search_roots(
        |batch| call_eval_cb(py, py_eval_pos_cb, 0, &batch),
        &roots,
        n_mcts_iterations,
        c_exploration,
        c_ply_penalty,
        leaf_minimax_depth,
    )?;
    let flat: Vec<f32> = searched.iter().flat_map(|r| r.policy).collect();
    let policies = Array2::from_shape_vec((searched.len(), Pos::N_COLS), flat)
        .expect("policy buffer shape")
        .into_pyarray(py);
    let q_penalty = Array1::from_iter(searched.iter().map(|r| r.q_penalty)).into_pyarray(py);
    let q_no_penalty = Array1::from_iter(searched.iter().map(|r| r.q_no_penalty)).into_pyarray(py);
    Ok((policies, q_penalty, q_no_penalty))
}

/// Samples positions in `split` ([sample_positions]) without labeling them: 0-indexed move
/// strings, up to `per_ply` per ply in `first_ply..=last_ply`.
#[pyfunction]
pub fn sample_split_positions(
    split: &str,
    first_ply: usize,
    last_ply: usize,
    per_ply: usize,
    seed: u64,
) -> PyResult<Vec<String>> {
    let split = Split::parse(split).map_err(PyValueError::new_err)?;
    let sampled = sample_positions(split, first_ply..=last_ply, per_ply, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(sampled.iter().map(|s| moves_string(&s.moves)).collect())
}

/// Samples positions in `split` ([sample_positions]) and labels them with Pons's solver
/// ([solve_parallel]). Returns (moves, scores) rows: 0-indexed move strings and seven scores per
/// row, None for full columns.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn label_positions(
    py: Python<'_>,
    split: &str,
    first_ply: usize,
    last_ply: usize,
    per_ply: usize,
    seed: u64,
    solver_path: String,
    book_path: String,
    n_procs: usize,
) -> PyResult<Vec<(String, Vec<Option<i8>>)>> {
    let split = Split::parse(split).map_err(PyValueError::new_err)?;
    py.detach(move || {
        let sampled = sample_positions(split, first_ply..=last_ply, per_ply, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let moves: Vec<Vec<Move>> = sampled.into_iter().map(|s| s.moves).collect();
        let scores = solve_parallel(&solver_path, &book_path, &moves, n_procs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(moves
            .iter()
            .zip(scores)
            .map(|(m, s)| (moves_string(m), s.to_vec()))
            .collect())
    })
}

/// Creates a batch of positions in tensor format.
fn create_pos_batch<'py>(py: Python<'py>, positions: &[Pos]) -> Bound<'py, PyArray4<f32>> {
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
    .into_pyarray(py)
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
        py_eval_pos_cb: py_eval_pos_cb.clone().unbind(),
    };

    // Start the TUI while detaching from Python.
    py.detach(move || {
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
