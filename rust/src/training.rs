use std::array;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    tensor::backend::Backend,
};
use num_traits::ToPrimitive;

use crate::{
    c4r::Pos,
    mcts::Sample,
    nn::ConnectFourNet,
    self_play::{self, EvalPosResult, EvalPosT},
};

type WgpuBackend = Autodiff<Wgpu>;

fn train_gen(n_games: usize, mcts_iterations: usize, exploration_constant: f32, batch_size: usize) {
    let device = WgpuDevice::default();
    let model = ConnectFourNet::<WgpuBackend>::new(&device);

    let _samples = gen_games(
        &device,
        &model,
        n_games,
        mcts_iterations,
        exploration_constant,
        batch_size,
    );
}

fn gen_games<B: Backend>(
    device: &B::Device,
    model: &ConnectFourNet<B>,
    n_games: usize,
    mcts_iterations: usize,
    exploration_constant: f32,
    batch_size: usize,
) -> Vec<Sample> {
    let batch_eval_pos = BatchEvalPos {
        device: device,
        model: model,
    };

    self_play::self_play(
        batch_eval_pos,
        n_games,
        batch_size,
        mcts_iterations,
        exploration_constant,
    )
}

struct BatchEvalPos<'a, B: Backend> {
    device: &'a B::Device,
    model: &'a ConnectFourNet<B>,
}

impl<'a, B: Backend> EvalPosT for BatchEvalPos<'a, B> {
    /// Evaluate a batch of positions with an NN forward pass.
    /// The ordering of the results corresponds to the ordering of the input positions.
    fn eval_pos(&self, pos: &Vec<Pos>) -> Vec<EvalPosResult> {
        let batch = Pos::to_batched_tensor(pos, self.device);
        let (policy_logits, value) = self.model.forward(batch);
        let policy: Vec<_> = policy_logits
            .detach()
            .exp()
            .into_data()
            .value
            .into_iter()
            .map(|x| x.to_f32().unwrap())
            .collect();
        let value: Vec<_> = value
            .detach()
            .into_data()
            .value
            .into_iter()
            .map(|x| x.to_f32().unwrap())
            .collect();

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
    use more_asserts::assert_ge;

    use super::*;

    #[test]
    fn gen_games_test() {
        let device = WgpuDevice::default();
        let model = ConnectFourNet::<Wgpu>::new(&device);
        let n_games = 2;
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
