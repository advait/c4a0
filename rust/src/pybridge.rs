use numpy::{
    ndarray::{Array1, Array2, Array4},
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::{prelude::*, types::PyList};

use crate::{
    c4r::Pos,
    mcts::Policy,
    self_play::{self_play, EvalPosResult, EvalPosT},
};

/// A batch of training samples.
#[pyclass]
pub struct PySamples {
    player0_ids: Py<PyArray1<u64>>, // b
    player1_ids: Py<PyArray1<u64>>, // b
    pos: Py<PyArray4<f32>>,         // b, c, h, w
    policy: Py<PyArray2<f32>>,      // b, w
    value: Py<PyArray1<f32>>,       // b
}

#[pymethods]
impl PySamples {}

/// Generate training samples with self-play. This is a python wrapper around [self_play].
#[pyfunction]
pub fn gen_samples<'py>(
    py: Python<'py>,
    reqs: &Bound<'py, PyList>,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    exploration_constant: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
) -> PyResult<PySamples> {
    if !py_eval_pos_cb.is_callable() {
        panic!("py_eval_pos_cb must be callable");
    }

    let reqs: Vec<(usize, usize, usize)> = reqs.extract().expect("error extracting reqs");
    let batcher = PyEvalPos {
        py_eval_pos_cb: py_eval_pos_cb.to_object(py),
    };

    let samples = {
        let batcher = batcher.clone();
        // Start background processing threads while releasing the GIL with allow_threads.
        // This allows other python threads (e.g. pytorch) to continue while we generate training
        // samples. When we need to call the py_eval_pos callback, we will re-acquire the GIL.
        py.allow_threads(move || {
            self_play(
                batcher,
                reqs.len(),
                max_nn_batch_size,
                n_mcts_iterations,
                exploration_constant,
            )
        })
    };

    let ret = PySamples {
        player0_ids: Array1::from_vec(vec![0; samples.len()])
            .into_pyarray_bound(py)
            .into(),
        player1_ids: Array1::from_vec(vec![1; samples.len()])
            .into_pyarray_bound(py)
            .into(),
        pos: batcher
            .create_pos_batch(py, &samples.iter().map(|s| s.pos.clone()).collect())
            .into(),
        policy: batcher
            .create_policy_batch(py, &samples.iter().map(|s| s.policy.clone()).collect())
            .into(),
        value: batcher
            .create_value_batch(py, &samples.iter().map(|s| s.value).collect())
            .into(),
    };

    Ok(ret)
}

#[derive(Clone)]
struct PyEvalPos {
    py_eval_pos_cb: PyObject,
}

impl PyEvalPos {
    /// Creates a batch of positions in tensor format.
    fn create_pos_batch<'py>(
        &self,
        py: Python<'py>,
        positions: &Vec<Pos>,
    ) -> Bound<'py, PyArray4<f32>> {
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
    fn create_policy_batch<'py>(
        &self,
        py: Python<'py>,
        policies: &Vec<Policy>,
    ) -> Bound<'py, PyArray2<f32>> {
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
    fn create_value_batch<'py>(
        &self,
        py: Python<'py>,
        values: &Vec<f32>,
    ) -> Bound<'py, PyArray1<f32>> {
        Array1::from_shape_vec((values.len(),), values.clone())
            .expect("Failed to create Array1 from buffer")
            .into_pyarray_bound(py)
    }

    fn policy_from_slice(&self, policy: &[f32]) -> Policy {
        debug_assert_eq!(policy.len(), Pos::N_COLS);
        let mut ret = Policy::default();
        ret.copy_from_slice(policy);
        ret
    }
}

impl EvalPosT for PyEvalPos {
    /// Evaluates a batch of positions by calling the [Self::py_eval_pos_cb] callback.
    /// This is intended to be a pytorch model that runs on the GPU. Because this is a python
    /// call we need to first re-acquire the GIL to call this function from a background thread
    /// before performing the callback.
    fn eval_pos(&self, pos: &Vec<Pos>) -> Vec<EvalPosResult> {
        Python::with_gil(|py| {
            let pos_batch = self.create_pos_batch(py, pos);

            let (policy, value): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>) = (&self
                .py_eval_pos_cb
                .call1(py, (pos_batch,))
                .expect("Failed to call py_eval_pos_cb"))
                .extract(py)
                .expect("Failed to extract result");

            let policy = policy.as_slice().expect("Failed to get policy slice");
            let value = value.as_slice().expect("Failed to get value slice");

            (0..pos.len())
                .map(|i| EvalPosResult {
                    policy: self.policy_from_slice(&policy[i * Pos::N_COLS..(i + 1) * Pos::N_COLS]),
                    value: value[i],
                })
                .collect()
        })
    }
}
