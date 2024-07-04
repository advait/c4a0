use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig,
    },
    prelude::*,
    tensor::{
        activation::{log_softmax, relu},
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{
        metric::{Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

use crate::{batching::TrainingBatch, c4r::Pos, nn_utils::kl_divergence_loss};

#[derive(Config, Debug)]
pub struct ConnectFourNetConfig {
    #[config(default = "64")]
    cnn_channels: usize,
}

impl ConnectFourNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConnectFourNet<B> {
        let fc_size = self.cnn_channels * Pos::N_ROWS * Pos::N_COLS;

        ConnectFourNet {
            conv1: ConvBlock::new([2, self.cnn_channels], [3, 3], device), // b c=2 h=6 w=7
            conv2: ConvBlock::new([self.cnn_channels, self.cnn_channels], [3, 3], device),
            conv3: ConvBlock::new([self.cnn_channels, self.cnn_channels], [3, 3], device),

            fc_policy1: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_policy2: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_policy3: nn::LinearConfig::new(fc_size, Pos::N_COLS).init(device),

            fc_value1: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_value2: nn::LinearConfig::new(fc_size, fc_size).init(device),
            fc_value3: nn::LinearConfig::new(fc_size, 1).init(device),
        }
    }
}

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

    fn forward_with_loss(&self, batch: TrainingBatch<B>) -> ConnectFourLoss<B> {
        let (policy_logprobs, value) = self.forward(batch.pos);
        let kl_loss = kl_divergence_loss(policy_logprobs.clone(), batch.policy_target);
        let mse_loss = (batch.value_target - value.clone())
            .powi_scalar(2)
            .mean()
            .sqrt();
        let loss = kl_loss.clone() + mse_loss.clone();

        ConnectFourLoss {
            kl_loss,
            mse_loss,
            loss,
            policy_logprobs,
            value,
        }
    }
}

pub struct ConnectFourLoss<B: Backend> {
    kl_loss: Tensor<B, 1>,
    mse_loss: Tensor<B, 1>,
    loss: Tensor<B, 1>,
    policy_logprobs: Tensor<B, 2>,
    value: Tensor<B, 1>,
}

impl<B: Backend> Adaptor<LossInput<B>> for ConnectFourLoss<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, ConnectFourLoss<B>> for ConnectFourNet<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<ConnectFourLoss<B>> {
        let output = self.forward_with_loss(batch);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<TrainingBatch<B>, ConnectFourLoss<B>> for ConnectFourNet<B> {
    fn step(&self, batch: TrainingBatch<B>) -> ConnectFourLoss<B> {
        self.forward_with_loss(batch)
    }
}

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
    use std::array;

    use crate::{batching::TrainingBatcher, mcts::Sample};

    use super::*;
    use approx::assert_relative_eq;
    use burn::{
        backend::{candle::CandleDevice, ndarray::NdArrayDevice, Candle, NdArray},
        data::dataloader::batcher::Batcher,
    };
    use more_asserts::{assert_ge, assert_le};

    type TestBackend = NdArray<f32>;

    #[test]
    fn forward_ndarray() {
        let device = NdArrayDevice::default();
        let model = ConnectFourNetConfig::new().init::<TestBackend>(&device);
        let batcher = TrainingBatcher::new(device);
        let batch = batcher.batch_pos(&vec![Pos::new()]);
        let (policy, value) = model.forward(batch);
        assert_eq!(policy.shape(), [1, Pos::N_COLS].into());
        assert_relative_eq!(policy.exp().sum().into_scalar(), 1.0);
        assert_eq!(value.shape(), vec![1 as usize].into());
        let value = value.into_scalar();
        assert_ge!(value, -1.);
        assert_le!(value, 1.);
    }

    #[test]
    fn forward_candle() {
        let device = CandleDevice::default();
        let model = ConnectFourNetConfig::new().init::<Candle>(&device);
        let batcher = TrainingBatcher::new(device);
        let batch = batcher.batch_pos(&vec![Pos::new()]);
        let (policy, value) = model.forward(batch);
        assert_eq!(policy.shape(), [1, Pos::N_COLS].into());
        assert_relative_eq!(policy.exp().sum().into_scalar(), 1.0);
        assert_eq!(value.shape(), vec![1 as usize].into());
        let value = value.into_scalar();
        assert_ge!(value, -1.);
        assert_le!(value, 1.);
    }

    /// Using the NN's output as training data should yield zero loss.
    #[test]
    fn zero_loss() {
        let device = NdArrayDevice::default();
        let model = ConnectFourNetConfig::new().init::<TestBackend>(&device);
        let batcher = TrainingBatcher::new(device);
        let pos = vec![Pos::new()];
        let batch = batcher.batch_pos(&pos);
        let (policy, value) = model.forward(batch);
        let policy = policy.clone().exp().into_data().value;

        let sample = Sample {
            pos: pos[0].clone(),
            policy: array::from_fn(|i| policy[i]),
            value: value.clone().into_data().value[0],
            game_id: 0,
        };

        let batch = batcher.batch(vec![sample]);

        let loss = model.forward_with_loss(batch);
        assert_relative_eq!(loss.mse_loss.into_scalar(), 0.0);
        assert_relative_eq!(loss.kl_loss.into_scalar(), 0.0);
        assert_relative_eq!(loss.loss.into_scalar(), 0.0);
    }
}
