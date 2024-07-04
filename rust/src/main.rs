use burn::{
    backend::{candle::CandleDevice, wgpu::WgpuDevice, Autodiff, Candle, Wgpu},
    optim::AdamConfig,
};
use nn::ConnectFourNetConfig;
use training::TrainingConfig;

mod batching;
mod c4r;
mod mcts;
mod nn;
mod nn_utils;
mod self_play;
mod training;

fn main() {
    println!("Hello, world!");

    type Backend = Autodiff<Candle>;
    let device = CandleDevice::default();

    let training_config = TrainingConfig::new(ConnectFourNetConfig::new(), AdamConfig::new());
    crate::training::train::<Backend>("artifacts", training_config, device);
    println!("Done")
}
