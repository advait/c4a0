"""
Generation-based network training, alternating between self-play and training.
"""

import copy
from datetime import datetime
import os
import pickle
from typing import List, NewType, Optional, Tuple

from loguru import logger
from pydantic import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from c4a0.nn import ConnectFourNet

import c4a0_rust  # type: ignore
from c4a0_rust import PlayGamesResult, BUF_N_CHANNELS, N_COLS, N_ROWS, Sample  # type: ignore


class TrainingGen(BaseModel):
    """
    Represents a single generation of training.
    """

    created_at: datetime
    n_mcts_iterations: int
    exploration_constant: float
    self_play_batch_size: int
    training_batch_size: int
    parent: Optional[datetime] = None

    @staticmethod
    def _gen_folder(created_at: datetime, base_dir: str) -> str:
        return os.path.join(base_dir, created_at.isoformat())

    def gen_folder(self, base_dir: str) -> str:
        return TrainingGen._gen_folder(self.created_at, base_dir)

    def save(
        self,
        base_dir: str,
        games: Optional[PlayGamesResult],
        model: ConnectFourNet,
    ):
        gen_dir = self.gen_folder(base_dir)
        os.makedirs(gen_dir, exist_ok=True)

        metadata_path = os.path.join(gen_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            f.write(self.model_dump_json())

        play_result_path = os.path.join(gen_dir, "games.pkl")
        with open(play_result_path, "wb") as f:
            pickle.dump(games, f)

        model_path = os.path.join(gen_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(base_dir: str, created_at: datetime) -> "TrainingGen":
        gen_folder = TrainingGen._gen_folder(created_at, base_dir)
        with open(os.path.join(gen_folder, "metadata.json"), "r") as f:
            return TrainingGen.model_validate_json(f.read())

    @staticmethod
    def load_all(base_dir: str) -> List["TrainingGen"]:
        timestamps = sorted(
            [
                datetime.fromisoformat(f)
                for f in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, f))
            ],
            reverse=True,
        )
        return [TrainingGen.load(base_dir, t) for t in timestamps]

    @staticmethod
    def load_latest(base_dir: str) -> "TrainingGen":
        timestamps = sorted(
            [
                datetime.fromisoformat(f)
                for f in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, f))
            ],
            reverse=True,
        )
        if not timestamps:
            raise FileNotFoundError("No existing generations")
        return TrainingGen.load(base_dir, timestamps[0])

    @staticmethod
    def load_latest_with_default(
        base_dir: str,
        n_mcts_iterations: int,
        exploration_constant: float,
        self_play_batch_size: int,
        training_batch_size: int,
    ):
        try:
            return TrainingGen.load_latest(base_dir)
        except FileNotFoundError:
            logger.info("No existing generations found, initializing root")
            gen = TrainingGen(
                created_at=datetime.now(),
                n_mcts_iterations=n_mcts_iterations,
                exploration_constant=exploration_constant,
                self_play_batch_size=self_play_batch_size,
                training_batch_size=training_batch_size,
            )
            gen.save(base_dir, None, ConnectFourNet())
            return gen

    def get_games(self, base_dir: str) -> Optional[PlayGamesResult]:
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "games.pkl"), "rb") as f:
            return pickle.load(f)

    def get_model(self, base_dir: str) -> ConnectFourNet:
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "model.pkl"), "rb") as f:
            return pickle.load(f)


def train_single_gen(
    base_dir: str,
    device: torch.device,
    parent: TrainingGen,
    n_self_play_games: int,
    n_mcts_iterations: int,
    exploration_constant: float,
    self_play_batch_size: int,
    training_batch_size: int,
) -> TrainingGen:
    """
    Trains a new generation from the given parent.
    First generate games using c4a0_rust.play_games.
    Then train a new model based on the parent model using the generated samples.
    Finally, save the resulting games and model in the training directory.
    """
    logger.info("Beginning new generation from", parent=parent.created_at)

    wandb_logger = WandbLogger(project="c4a0")
    # TODO: add experiment metadata

    # Self play
    model = parent.get_model(base_dir)
    model.to(device)
    reqs = [c4a0_rust.GameMetadata(id, 0, 0) for id in range(n_self_play_games)]  # type: ignore
    games = c4a0_rust.play_games(  # type: ignore
        reqs,
        self_play_batch_size,
        n_mcts_iterations,
        exploration_constant,
        lambda player_id, pos: model.forward_numpy(pos),
    )

    # Training
    logger.info("Beginning training")
    model = copy.deepcopy(model)
    train, test = games.split_train_test(0.8, 1337)  # type: ignore
    data_module = SampleDataModule(train, test, training_batch_size)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        logger=wandb_logger,
    )
    model.train()  # Switch batch normalization to train mode for training bn params
    trainer.fit(model, data_module)
    logger.info("Finished training")

    gen = TrainingGen(
        created_at=datetime.now(),
        n_mcts_iterations=n_mcts_iterations,
        exploration_constant=exploration_constant,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        parent=parent.created_at,
    )
    gen.save(base_dir, games, model)
    return gen


def training_loop(
    base_dir: str,
    device: torch.device,
    n_self_play_games: int,
    n_mcts_iterations: int,
    exploration_constant: float,
    self_play_batch_size: int,
    training_batch_size: int,
):
    """Main training loop. Sequentially trains generation after generation."""
    gen = TrainingGen.load_latest_with_default(
        base_dir=base_dir,
        n_mcts_iterations=n_mcts_iterations,
        exploration_constant=exploration_constant,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
    )

    while True:
        gen = train_single_gen(
            base_dir=base_dir,
            device=device,
            parent=gen,
            n_self_play_games=n_self_play_games,
            n_mcts_iterations=n_mcts_iterations,
            exploration_constant=exploration_constant,
            self_play_batch_size=self_play_batch_size,
            training_batch_size=training_batch_size,
        )


SampleTensor = NewType(
    "SampleTensor",
    Tuple[
        torch.Tensor,  # Pos
        torch.Tensor,  # Policy
        torch.Tensor,  # Value
    ],
)


class SampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_data: List[Sample],
        validation_data: List[Sample],
        batch_size: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        training_data += [s.flip_h() for s in training_data]
        validation_data += [s.flip_h() for s in validation_data]
        self.training_data = [self.sample_to_tensor(s) for s in training_data]
        self.validation_data = [self.sample_to_tensor(s) for s in validation_data]

    @staticmethod
    def sample_to_tensor(sample: Sample) -> "SampleTensor":
        pos, policy, value = sample.to_numpy()
        pos_t = torch.from_numpy(pos)
        policy_t = torch.from_numpy(policy)
        value_t = torch.from_numpy(value)
        assert pos_t.shape == (BUF_N_CHANNELS, N_ROWS, N_COLS)
        assert policy_t.shape == (N_COLS,)
        assert value_t.shape == ()
        return SampleTensor((pos_t, policy_t, value_t))

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
