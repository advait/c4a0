use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    optim::AdamConfig,
};
use nn::ConnectFourNetConfig;
use training::TrainingConfig;

mod c4r;
mod data;
mod mcts;
mod nn;
mod nn_utils;
mod self_play;
mod training;

fn main() {
    println!("Hello, world!");

    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let training_config = TrainingConfig::new(ConnectFourNetConfig::new(), AdamConfig::new());
    crate::training::train::<Backend>("artifacts", training_config, device);
    println!("Done")
}
