use rand::prelude::*;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    tensor::{backend::Backend, Data, Tensor},
};
use rand::rngs::StdRng;

use crate::{c4r::Pos, mcts::Sample};

/// Splits samples into a training and test set.
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

    pub fn batch_pos(&self, pos: &Vec<Pos>) -> Tensor<B, 4> {
        let positions = pos.iter().flat_map(|pos| pos.to_data().value).collect();
        let pos_data = Data::new(positions, [pos.len(), 2, Pos::N_ROWS, Pos::N_COLS].into());
        Tensor::from_data(pos_data.convert(), &self.device)
    }
}

#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub pos: Tensor<B, 4>,           // b, c, h, w
    pub policy_target: Tensor<B, 2>, // b, w
    pub value_target: Tensor<B, 1>,  // b
}

impl<B: Backend> Batcher<Sample, TrainingBatch<B>> for TrainingBatcher<B> {
    /// Converts a batch of [Sample]s into a [TrainingBatch] in [Tensor] form.
    fn batch(&self, samples: Vec<Sample>) -> TrainingBatch<B> {
        let batch_size = samples.len();

        let policies = samples.iter().flat_map(|sample| sample.policy).collect();
        let policy_data = Data::new(policies, [batch_size, Pos::N_COLS].into());

        let values = samples.iter().map(|sample| sample.value).collect();
        let value_data = Data::new(values, [batch_size].into());

        TrainingBatch {
            pos: self.batch_pos(&samples.into_iter().map(|sample| sample.pos).collect()),
            policy_target: Tensor::from_data(policy_data.convert(), &self.device),
            value_target: Tensor::from_data(value_data.convert(), &self.device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c4r::Pos;
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        data::dataset::Dataset,
    };

    fn new_sample() -> Sample {
        Sample {
            pos: Pos::new(),
            policy: [0.0; 7],
            value: 0.0,
            game_id: 0,
        }
    }

    #[test]
    fn test_split_train_test() {
        let mut samples: Vec<_> = (0..10).map(|_| new_sample()).collect();

        let mut rng = StdRng::seed_from_u64(0);
        let (train, test) = split_train_test(&mut samples, &mut rng);

        assert_eq!(train.len(), 8);
        assert_eq!(test.len(), 2);
    }

    #[test]
    fn test_batcher() {
        let device = NdArrayDevice::default();
        let batch_size = 10;
        let samples: Vec<_> = (0..batch_size).map(|_| new_sample()).collect();
        let batcher = TrainingBatcher::<NdArray<f32>>::new(device);
        let batch = batcher.batch(samples);

        assert_eq!(
            batch.pos.shape(),
            [batch_size, 2, Pos::N_ROWS, Pos::N_COLS].into()
        );
        assert_eq!(
            batch.policy_target.shape(),
            [batch_size, Pos::N_COLS].into()
        );
        assert_eq!(batch.value_target.shape(), [batch_size].into());
    }
}
