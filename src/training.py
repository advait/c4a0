"""
Generation-based network training, alternating between self-play and training.
"""
import logging
import os
from typing import List, NewType, Tuple
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from c4 import N_COLS, N_ROWS, Pos
from model_storage import (
    ModelGen,
    load_cached_samples,
    load_latest_model,
    store_samples,
)

from nn import Policy
from self_play import Sample, generate_samples


async def train(
    n_games: int,
    n_processes: int,
    mcts_iterations: int,
    exploration_constant,
    batch_size: int,
    device: torch.device,
):
    logger = logging.getLogger(__name__)

    gen, model = load_latest_model()
    logger.info(f"Loaded model from gen {gen}")
    model.to(device)
    gen = ModelGen(gen + 1)  # Training next gen

    logger.info(f"Generating self_play games to train next gen {gen}")

    samples = load_cached_samples(gen)
    if not samples:
        logger.info("No cached samples found. Generating samples from self-play.")
        samples = await generate_samples(
            model=model,
            n_games=n_games,
            mcts_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            nn_max_batch_size=20000,
            device=device,
            n_processes=n_processes,
        )
        print(f"{len(samples)} Samples generated")
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
    logger.info(f"Beginning training gen {gen}")
    trainer.fit(
        model,
        data_module,
    )
    logger.info(f"Finished training gen {gen}")


SampleTensor = NewType(
    "SampleTensor", Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
)


class PosDataModule(pl.LightningDataModule):
    def __init__(self, samples: List[Sample], batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        samples = samples + [self.flip_sample(s) for s in samples]
        tensors = list(self.sample_to_tensor(s) for s in samples)
        self.training_data, self.validation_data = self.split_samples(tensors)

    def flip_sample(self, sample: Sample) -> Sample:
        """Flips the sample and policy horizontally to account for connect four symmetry."""
        game_id, pos, policy, value = sample
        # copy is needed as flip returns a view with negative stride that is incompatible w torch
        pos = Pos(np.fliplr(pos).copy())
        policy = Policy(np.flip(policy).copy())
        return Sample((game_id, pos, policy, value))

    def sample_to_tensor(self, sample: Sample) -> SampleTensor:
        game_id, pos, policy, value = sample
        pos_t = torch.from_numpy(pos).float()
        policy_t = torch.from_numpy(policy).float()
        value_t = torch.tensor(value, dtype=torch.float)
        assert pos_t.shape == (N_ROWS, N_COLS)
        assert policy_t.shape == (N_COLS,)
        assert value_t.shape == ()
        return SampleTensor((game_id, pos_t, policy_t, value_t))

    def split_samples(
        self, samples: List[SampleTensor], split_ratio=0.8
    ) -> Tuple[List[SampleTensor], List[SampleTensor]]:
        """Splits samples into training and validation sets."""
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
        return DataLoader(
            self.training_data,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,  # type: ignore
            batch_size=self.batch_size,
        )
