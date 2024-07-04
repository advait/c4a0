use std::array;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{metric::LossMetric, LearnerBuilder},
};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    batching::{split_train_test, TrainingBatcher},
    c4r::Pos,
    mcts::Sample,
    nn::{ConnectFourNet, ConnectFourNetConfig},
    self_play::{self, EvalPosResult, EvalPosT},
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ConnectFourNetConfig,
    pub optimizer: AdamConfig,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 1.0e-3)]
    pub learning_rate: f64,

    #[config(default = 100)]
    pub n_games: usize,

    #[config(default = 1000)]
    pub batch_size: usize,

    #[config(default = 10)]
    pub mcts_iterations: usize,

    #[config(default = 1.4)]
    pub exploration_constant: f32,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Main training loop:
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    let seed = 1337;
    // B::seed(seed);

    let model = ConnectFourNetConfig::new().init::<B>(&device);
    println!("Generating games");
    let mut samples = gen_games(
        &device,
        &model,
        config.n_games,
        config.mcts_iterations,
        config.exploration_constant,
        config.batch_size,
    );
    println!("Done generating {} games", samples.len());

    let (train, test) = split_train_test(&mut samples, &mut StdRng::seed_from_u64(seed));

    let batcher_train = TrainingBatcher::<B>::new(device.clone());
    let batcher_test = TrainingBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(seed)
        .num_workers(config.num_workers)
        .build(test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn gen_games<B: Backend>(
    device: &B::Device,
    model: &ConnectFourNet<B>,
    n_games: usize,
    mcts_iterations: usize,
    exploration_constant: f32,
    batch_size: usize,
) -> Vec<Sample> {
    let batch_eval_pos = BatchEvalPos::new(device, model);

    self_play::self_play(
        batch_eval_pos,
        n_games,
        batch_size,
        mcts_iterations,
        exploration_constant,
    )
}

struct BatchEvalPos<'a, B: Backend> {
    model: &'a ConnectFourNet<B>,
    batcher: TrainingBatcher<B>,
}

impl<'a, B: Backend> BatchEvalPos<'a, B> {
    fn new(device: &'a B::Device, model: &'a ConnectFourNet<B>) -> Self {
        Self {
            model,
            batcher: TrainingBatcher::<B>::new(device.clone()),
        }
    }
}

impl<'a, B: Backend> EvalPosT for BatchEvalPos<'a, B> {
    /// Evaluate a batch of positions with an NN forward pass.
    /// The ordering of the results corresponds to the ordering of the input positions.
    fn eval_pos(&self, pos: &Vec<Pos>) -> Vec<EvalPosResult> {
        let batch = self.batcher.batch_pos(pos);
        let (policy_logits, value) = self.model.forward(batch);
        let policy: Vec<_> = policy_logits
            .detach()
            .exp()
            .into_data()
            .convert::<f32>()
            .value;
        let value: Vec<_> = value.detach().into_data().convert::<f32>().value;

        (0..pos.len())
            .map(|i| EvalPosResult {
                policy: array::from_fn(|j| policy[i * Pos::N_COLS + j]),
                value: value[i],
            })
            .collect()
    }
}

unsafe impl<'a, B: Backend> Sync for BatchEvalPos<'a, B> {}
unsafe impl<'a, B: Backend> Send for BatchEvalPos<'a, B> {}

#[cfg(test)]
mod tests {
    use burn::backend::{candle::CandleDevice, Candle};
    use more_asserts::assert_ge;

    use super::*;

    #[test]
    fn gen_games_test() {
        let device = CandleDevice::default();
        let model = ConnectFourNetConfig::new().init::<Candle>(&device);
        let n_games = 20;
        let mcts_iterations = 2;
        let exploration_constant = 1.0;
        let batch_size = n_games;

        let samples = gen_games(
            &device,
            &model,
            n_games,
            mcts_iterations,
            exploration_constant,
            batch_size,
        );

        for g in 0..n_games {
            let game_samples = samples
                .iter()
                .filter(|Sample { game_id, .. }| *game_id == (g as u64))
                .collect::<Vec<_>>();

            assert_ge!(game_samples.len(), 7);
            assert_eq!(
                game_samples
                    .iter()
                    .filter(|Sample { pos, .. }| *pos == Pos::new())
                    .count(),
                1,
                "game {} should have a single starting position",
                g
            );

            let terminal_positions = game_samples
                .iter()
                .filter(|Sample { pos, .. }| pos.is_terminal_state().is_some())
                .collect::<Vec<_>>();
            assert_eq!(
                terminal_positions.len(),
                1,
                "game {} should have a single terminal position",
                g
            );
            let terminal_value = terminal_positions[0].value;
            if terminal_value != -1.0 && terminal_value != 0.0 && terminal_value != 1.0 {
                assert!(
                    false,
                    "expected terminal value {} to be -1, 0, or 1",
                    terminal_value
                );
            }
        }
    }
}
