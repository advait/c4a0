use numpy::{
    ndarray::{Array1, Array2, Array4},
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyList},
};
use rand::{seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    c4r::Pos,
    self_play::self_play,
    types::{EvalPosResult, EvalPosT, GameMetadata, GameResult, PlayerID, Policy, Sample},
};

/// Play games via MCTS. This is a python wrapper around [self_play].
/// `reqs` is a list of [GameMetadata] that describes the games to play.
/// The `py_eval_pos_cb` callback is expected to be a pytorch model that runs on the GPU.
#[pyfunction]
pub fn play_games<'py>(
    py: Python<'py>,
    reqs: &Bound<'py, PyList>,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    exploration_constant: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
) -> PyResult<GenSamplesResult> {
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
                exploration_constant,
            )
        })
    };

    Ok(GenSamplesResult { results })
}

/// The result of [play_games].
#[derive(Debug, Serialize, Deserialize)]
#[pyclass]
pub struct GenSamplesResult {
    #[pyo3(get)]
    pub results: Vec<GameResult>,
}

#[pymethods]
impl GenSamplesResult {
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

    /// Returns the number of samples in the results.
    fn n_samples(&self) -> usize {
        self.results.iter().map(|r| r.samples.len()).sum()
    }

    /// Splits the training samples into a training and test set.
    fn split_train_test(
        &self,
        py: Python,
        train_frac: f32,
        seed: u64,
    ) -> PyResult<(SampleBatch, SampleBatch)> {
        let mut samples = self
            .results
            .iter()
            .flat_map(|r| r.samples.iter().map(|s| s.clone()))
            .collect::<Vec<_>>();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        samples.as_mut_slice().shuffle(&mut rng);

        let n_test = (samples.len() as f32 * train_frac).round() as usize;
        let (train, test) = samples.split_at(n_test);

        Ok((
            SampleBatch::from_samples(py, train.to_vec()),
            SampleBatch::from_samples(py, test.to_vec()),
        ))
    }
}

/// Training samples in [numpy] batch form.
#[pyclass]
pub struct SampleBatch {
    #[pyo3(get, set)]
    pos: Py<PyArray4<f32>>, // b, c, h, w

    #[pyo3(get, set)]
    policy: Py<PyArray2<f32>>, // b, w

    #[pyo3(get, set)]
    value: Py<PyArray1<f32>>, // b
}

#[pymethods]
impl SampleBatch {
    #[staticmethod]
    fn from_samples<'py>(py: Python<'py>, samples: Vec<Sample>) -> Self {
        Self {
            pos: create_pos_batch(py, &samples.iter().map(|s| s.pos.clone()).collect()).into(),
            policy: create_policy_batch(py, &samples.iter().map(|s| s.policy.clone()).collect())
                .into(),
            value: create_value_batch(py, &samples.iter().map(|s| s.value).collect()).into(),
        }
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
    fn eval_pos(&self, player_id: PlayerID, pos: Vec<Pos>) -> Vec<EvalPosResult> {
        Python::with_gil(|py| {
            let batch_size = pos.len();
            let pos_batch = create_pos_batch(py, &pos);

            let (policy, value): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>) = (&self
                .py_eval_pos_cb
                .call_bound(py, (player_id, pos_batch), None)
                .expect("Failed to call py_eval_pos_cb"))
                .extract(py)
                .expect("Failed to extract result");

            let policy = policy.as_slice().expect("Failed to get policy slice");
            let value = value.as_slice().expect("Failed to get value slice");

            (0..batch_size)
                .map(|i| EvalPosResult {
                    policy: policy_from_slice(&policy[i * Pos::N_COLS..(i + 1) * Pos::N_COLS]),
                    value: value[i],
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

/// Creates a batch of policies in tensor format.
fn create_policy_batch<'py>(py: Python<'py>, policies: &Vec<Policy>) -> Bound<'py, PyArray2<f32>> {
    let mut buffer = vec![0.0; policies.len() * Pos::N_COLS];
    for i in 0..policies.len() {
        let policy = &policies[i];
        let policy_buffer = &mut buffer[i * Pos::N_COLS..(i + 1) * Pos::N_COLS];
        policy_buffer.copy_from_slice(policy);
    }

    Array2::from_shape_vec((policies.len(), Pos::N_COLS), buffer)
        .expect("Failed to create Array2 from buffer")
        .into_pyarray_bound(py)
}

/// Creates a batch of position values in tensor format.
fn create_value_batch<'py>(py: Python<'py>, values: &Vec<f32>) -> Bound<'py, PyArray1<f32>> {
    Array1::from_shape_vec((values.len(),), values.clone())
        .expect("Failed to create Array1 from buffer")
        .into_pyarray_bound(py)
}

/// Convert a slice of probabilities into a [Policy].
fn policy_from_slice(policy: &[f32]) -> Policy {
    debug_assert_eq!(policy.len(), Pos::N_COLS);
    let mut ret = Policy::default();
    ret.copy_from_slice(policy);
    ret
}

/// Convert a Rust error into a Python exception.
fn pyify_err<T>(e: T) -> PyErr
where
    T: std::fmt::Debug,
{
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e))
}
