use rand::prelude::*;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    tensor::{backend::Backend, Tensor},
};
use rand::rngs::StdRng;

use crate::mcts::Sample;

pub fn split_train_test(
    samples: &mut Vec<Sample>,
    rng: &mut StdRng,
) -> (InMemDataset<Sample>, InMemDataset<Sample>) {
    let n = samples.len();
    let n_train = (n as f64 * 0.8) as usize;

    samples.shuffle(rng);

    let (train, test) = samples.split_at(n_train);
    (
        InMemDataset::new(train.to_vec()),
        InMemDataset::new(test.to_vec()),
    )
}

#[derive(Clone)]
pub struct TrainingBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TrainingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub pos: Tensor<B, 4>,           // b, c, h, w
    pub policy_target: Tensor<B, 2>, // b, w
    pub value_target: Tensor<B, 1>,  // b
}

impl<B: Backend> Batcher<Sample, TrainingBatch<B>> for TrainingBatcher<B> {
    fn batch(&self, items: Vec<Sample>) -> TrainingBatch<B> {
        let pos = Tensor::cat(
            items
                .iter()
                .map(|sample| sample.pos.to_tensor(&self.device).unsqueeze_dim(0))
                .collect(),
            0,
        );

        let policy_target = Tensor::cat(
            items
                .iter()
                .map(|sample| Tensor::from_floats(sample.policy, &self.device).unsqueeze_dim(0))
                .collect(),
            0,
        );

        let value_target = Tensor::cat(
            items
                .iter()
                .map(|sample| Tensor::from_floats([sample.value], &self.device).unsqueeze_dim(0))
                .collect(),
            0,
        );

        TrainingBatch {
            pos,
            policy_target,
            value_target,
        }
    }
}
