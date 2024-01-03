"""
Generation-based network training, alternating between self-play and training.
"""
import logging
import os
import pickle
from typing import List, NewType, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from nn import ConnectFourNet
from self_play import Sample, gen_samples


async def train(
    mcts_iterations=100, exploration_constant=1.4, n_games=2000, batch_size=100
):
    logger = logging.getLogger(__name__)

    gen, model = load_latest_model()
    logger.info(f"Loaded model from gen {gen}")
    model.to("cuda")
    gen += 1  # Training next gen

    tb_logger = TensorBoardLogger("lightning_logs", name=f"gen_{gen}")
    logger.info(
        f"Generating self_play games to train next gen\n"
        f"- gen: {gen}\n"
        f"- n_games: {n_games}\n"
        f"- mcts_iterations: {mcts_iterations}\n"
        f"- exploration_constant: {exploration_constant}\n"
        f"- batch_size: {batch_size}\n"
    )

    samples = load_cached_samples(gen)
    if not samples:
        logger.info("No cached samples found. Generating samples from self-play.")
        samples = await gen_samples(
            eval1=model,
            n_games=n_games,
            mcts_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            tb_logger=tb_logger,
        )
        store_samples(samples, gen)
        logger.info(f"Done generating {len(samples)} samples. Caching for re-use.")
    else:
        logger.info("Loaded cached samples")

    checkpoint_path = os.path.join("checkpoints", str(gen))
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(
                dirpath=checkpoint_path, save_top_k=1, monitor="val_loss", mode="min"
            ),
        ],
    )
    data_module = PosDataModule(samples, batch_size)
    trainer.fit(
        model,
        data_module,
    )


ModelGen = NewType("ModelGen", int)


def load_latest_model() -> (ModelGen, ConnectFourNet):
    """Loads the latest generation of model from the checkpoints directory."""
    logger = logging.getLogger(__name__)
    gens = sorted(os.listdir("checkpoints"))
    if len(gens) == 0:
        logger.info("No checkpoints found. Starting from scratch.")
        return 0, ConnectFourNet()

    latest_gen = int(gens[-1])
    gen_dir = os.path.join("checkpoints", str(latest_gen))
    latest_checkpoint = max(
        os.listdir(gen_dir),
        key=lambda cpkt: os.path.getctime(os.path.join(gen_dir, cpkt)),
    )
    model = ConnectFourNet.load_from_checkpoint(
        os.path.join(gen_dir, latest_checkpoint)
    )
    return latest_gen, model


def store_samples(samples: List[Sample], gen: ModelGen):
    """Cache samples for re-use."""
    path = os.path.join("samples", str(gen), "samples.pkl")
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_cached_samples(gen: ModelGen) -> Optional[List[Sample]]:
    """Attempts to load previously generated samples to skip self play."""
    path = os.path.join("samples", str(gen), "samples.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


class PosDataModule(pl.LightningDataModule):
    def __init__(self, samples: List[Sample], batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        samples = [
            (
                game_id,
                torch.from_numpy(pos).float(),
                torch.from_numpy(policy).float(),
                torch.tensor(value, dtype=torch.float),
            )
            for game_id, pos, policy, value in samples
        ]
        self.training_data, self.validation_data = self.split_samples(samples)

    def split_samples(self, samples, split_ratio=0.8):
        game_ids = set(sample[0] for sample in samples)
        split_game_id = int(len(game_ids) * split_ratio) + min(game_ids)
        training_data = []
        validation_data = []
        for sample in samples:
            if sample[0] < split_game_id:
                training_data.append(sample)
            else:
                validation_data.append(sample)
        return training_data, validation_data

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)