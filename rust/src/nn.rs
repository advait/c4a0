use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig,
    },
    prelude::*,
    tensor::{
        activation::{log_softmax, relu},
        backend::Backend,
        Tensor,
    },
};

use crate::{c4r::Pos, nn_utils::kl_divergence_loss};

/// A CNN that inputs a position from [Pos::to_batched_tensor] and produces a [crate::mcts::Policy]
/// and a [crate::mcts::PosValue] to help guide MCTS.
/// Internally consists of a set of shared CNN blocks that split of into a separate Policy Head and
/// Value Head.
#[derive(Debug, Module)]
pub struct ConnectFourNet<B: Backend> {
    // Shared conv blocks
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    conv3: ConvBlock<B>,

    // Policy Head
    fc_policy1: nn::Linear<B>,
    fc_policy2: nn::Linear<B>,
    fc_policy3: nn::Linear<B>,

    // Value Head
    fc_value1: nn::Linear<B>,
    fc_value2: nn::Linear<B>,
    fc_value3: nn::Linear<B>,
}

impl<B: Backend> ConnectFourNet<B> {
    const CNN_CHANNELS: usize = 64;

    pub fn new(device: &B::Device) -> Self {
        let fc_size = Self::CNN_CHANNELS * Pos::N_ROWS * Pos::N_COLS;

        Self {
            conv1: ConvBlock::new([2, Self::CNN_CHANNELS], [3, 3], device), // b c=2 h=6 w=7
            conv2: ConvBlock::new([Self::CNN_CHANNELS, Self::CNN_CHANNELS], [3, 3], device),
            conv3: ConvBlock::new([Self::CNN_CHANNELS, Self::CNN_CHANNELS], [3, 3], device),

            fc_policy1: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_policy2: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_policy3: nn::LinearConfig::new(fc_size, Pos::N_COLS).init(device),

            fc_value1: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_value2: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_value3: nn::LinearConfig::new(fc_size, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);
        let x: Tensor<B, 2> = x.flatten(1, 3); // b c h w -> b (c h w)

        let x_p = x.clone();
        let x_p = self.fc_policy1.forward(x_p);
        let x_p = self.fc_policy2.forward(x_p);
        let x_p = self.fc_policy3.forward(x_p);
        let policy_logprobs = log_softmax(x_p, 1);

        let x_v = x;
        let x_v = self.fc_value1.forward(x_v);
        let x_v = self.fc_value2.forward(x_v);
        let x_v = self.fc_value3.forward(x_v);
        let value = x_v.flatten(0, 1).tanh();

        (policy_logprobs, value)
    }

    fn forward_classification(&self, item: PosBatch<B>) -> ClassificationOutput<B> {
        let (policy_logprobs, value) = self.forward(item.pos);
        let kl_loss = kl_divergence_loss(policy_logprobs, item.policy_targets);
        let mse_loss = (item.value_targets - value).powi_scalar(2).mean().sqrt();
        let loss = kl_loss + mse_loss;
        ClassificationOutput { loss }
    }
}

struct PosBatch<B: Backend> {
    pos: Tensor<B, 4>,
    policy_targets: Tensor<B, 2>,
    value_targets: Tensor<B, 1>,
}

struct ClassificationOutput<B: Backend> {
    loss: Tensor<B, 1>,
}

// impl<B: AutodiffBackend> TrainStep<PosBatch<B>, ClassificationOutput<B>> for ConnectFourNet<B> {
//     fn step(&self, item: PosBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_classification(item);
//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

/// CNN Block with batch normalization.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(channels, kernel_size)
            .with_padding(nn::PaddingConfig2d::Same)
            .init(device);
        let norm = BatchNormConfig::new(channels[1]).init(device);

        Self { conv, norm }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        relu(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use more_asserts::{assert_ge, assert_le};

    type TestBackend = NdArray<f32>;

    #[test]
    fn forward_pass() {
        let device = NdArrayDevice::default();
        let model = ConnectFourNet::<TestBackend>::new(&device);
        let batch = Pos::to_batched_tensor::<TestBackend>(&vec![Pos::new()], &device);
        let (policy, value) = model.forward(batch);
        assert_eq!(policy.shape(), [1, Pos::N_COLS].into());
        assert_relative_eq!(policy.exp().sum().into_scalar(), 1.0);
        assert_eq!(value.shape(), vec![1 as usize].into());
        let value = value.into_scalar();
        assert_ge!(value, -1.);
        assert_le!(value, 1.);
    }
}
