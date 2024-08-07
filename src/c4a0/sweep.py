import functools
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from c4a0.training import SampleDataModule, TrainingGen
from c4a0.nn import ConnectFourNet, ModelConfig

from c4a0_rust import Sample  # type: ignore


def load_samples(base_dir: str, n_gens: int = 5) -> List[Sample]:
    gens = TrainingGen.load_all(base_dir)[:n_gens]
    game_results = [gen.get_games(base_dir) for gen in gens]
    samples = [
        sample
        for game_result in game_results
        if game_result
        for results in game_result.results
        for sample in results.samples
    ]
    return samples


def train_once(samples: List[Sample]):
    wandb.init()
    config = wandb.config
    wandb_logger = WandbLogger(project="c4a0")

    model_config = ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        learning_rate=config.learning_rate,
        l2_reg=config.l2_reg,
    )
    model = ConnectFourNet(model_config)

    split_idx = int(0.8 * len(samples))
    train, test = samples[:split_idx], samples[split_idx:]

    data_module = SampleDataModule(train, test, config.batch_size)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ],
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)


sweep_config = {
    "method": "bayes",  # can be 'random', 'grid', 'bayes'
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [2e-3]},
        "l2_reg": {"values": [4e-4]},
        "batch_size": {"values": [512]},
        "n_residual_blocks": {"values": [0, 1]},
        "conv_filter_size": {"values": [32]},
        "n_policy_layers": {"values": [0, 4]},
        "n_value_layers": {"values": [0, 2]},
    },
}


def perform_sweep(base_dir: str):
    samples = load_samples(base_dir)
    sweep_id = wandb.sweep(sweep_config, project="c4a0")
    wandb.agent(sweep_id, functools.partial(train_once, samples), count=100)
