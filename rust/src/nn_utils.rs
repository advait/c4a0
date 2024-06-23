use burn::tensor::{backend::Backend, Tensor};

/// KL Divergence loss function.
/// `y_pred_log` should be the predicted values in log space.
/// `y_true` should be the target values as ordinary probabilities.
pub fn kl_divergence_loss<B: Backend>(
    y_pred_log: Tensor<B, 2>,
    y_true: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let eps = 1e-10; // Small constant to prevent log(0)
    let kl_div = y_true.clone() * ((y_true + eps).log() - y_pred_log);
    kl_div.sum_dim(1).mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};

    type TestBackend = NdArray<f32>;

    #[test]
    fn kl_perfect1() {
        let device = NdArrayDevice::default();
        let y_true = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
        let y_pred_log = Tensor::<TestBackend, 2>::from_floats([[0.0, -1e6], [-1e6, 0.0]], &device);
        let loss = kl_divergence_loss(y_pred_log, y_true);
        assert_relative_eq!(loss.into_scalar(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn kl_perfect2() {
        let device = NdArrayDevice::default();
        let y_true = Tensor::<TestBackend, 2>::from_floats([[0.1, 0.9], [0.2, 0.8]], &device);
        let y_pred_log = y_true.clone().log();
        let loss = kl_divergence_loss(y_pred_log, y_true);
        assert_relative_eq!(loss.into_scalar(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn kl_imperfect() {
        let device = NdArrayDevice::default();
        let y_true = Tensor::<TestBackend, 2>::from_floats([[0.7, 0.3], [0.2, 0.8]], &device);
        let y_pred_log =
            Tensor::<TestBackend, 2>::from_floats([[0.6, 0.4], [0.7, 0.3]], &device).log();
        let loss = kl_divergence_loss(y_pred_log, y_true);
        assert_relative_eq!(loss.into_scalar(), 0.27785, epsilon = 1e-4);
    }
}
