"""
Generation-based network training, alternating between self-play and training.
"""
from dataclasses import dataclass
from datetime import datetime
from glob import glob
import logging
import os
import pickle
from typing import Dict, List, NewType, Optional, Tuple
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader
from c4 import N_COLS, N_ROWS, Pos

from nn import ConnectFourNet, Policy
from self_play import GameID, Sample, generate_samples
from tournament import (
    GenID,
    ModelPlayer,
    Player,
    RandomPlayer,
    TournamentResult,
    UniformPlayer,
    play_tournament,
)


@dataclass
class TrainingState:
    """The full historical state of the training process."""

    models: Dict[GenID, ConnectFourNet]
    training_gens: List["TrainingGen"]

    def get_next_gen_id(self) -> GenID:
        return GenID(max(self.models.keys()) + 1)  # type: ignore

    @staticmethod
    def load_training_state() -> "TrainingState":
        logger = logging.getLogger(__name__)
        if not os.path.exists("training"):
            os.mkdir("training")
        files = sorted(glob("training/*.pkl"), reverse=True)
        if len(files) > 0:
            with open(files[0], "rb") as f:
                return pickle.load(f)
        logger.info("No historical training state found. Initializing a new one.")
        new_state = TrainingState(models={}, training_gens=[])
        return new_state

    def save_training_state(self):
        for model in self.models.values():
            model.cpu()  # Relocate models to cpu before pickling
        now_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        with open(
            os.path.join("training", f"training_state_{now_date}.pkl"), "wb"
        ) as f:
            pickle.dump(self, f)

    def get_model(self, gen: GenID) -> ConnectFourNet:
        return self.models[gen]

    def get_top_models(self) -> List[GenID]:
        """Returns the top models from the most recent tournamnet."""
        for training_gen in reversed(self.training_gens):
            if training_gen.tournament is None:
                continue
            return training_gen.tournament.get_top_models()
        return []

    def get_top_model(self) -> Tuple[GenID, ConnectFourNet]:
        """Gets the winner of the most recent tournament."""
        logger = logging.getLogger(__name__)
        try:
            gen = self.get_top_models()[0]
            return gen, self.get_model(gen)
        except IndexError:
            logger.info("No models found, creating a new one.")
            gen = GenID(0)
            model = ConnectFourNet()
            self.models[gen] = model
            return gen, model

    def create_new_gen(self) -> Tuple["TrainingGen", ConnectFourNet]:
        """Creates a new generation of training by cloning the best model."""
        if len(self.training_gens) > 0 and self.training_gens[-1].winning_gen is None:
            # If the latest generation hasn't completed training, return it
            latest_gen = self.training_gens[-1]
            return latest_gen, self.get_model(latest_gen.trained_gen)

        best_gen_id, best_model = self.get_top_model()
        next_gen = self.get_next_gen_id()
        next_model = ConnectFourNet()
        next_model.load_state_dict(best_model.state_dict())  # Clone model
        self.models[next_gen] = next_model
        training_gen = TrainingGen(start_gen=best_gen_id, trained_gen=next_gen)
        self.training_gens.append(training_gen)
        return training_gen, next_model


@dataclass
class TrainingGen:
    """
    State of a single training generation.

    We begin with the start_gen model and then use self play to generate training_samples.
    Then we train a new model called trained_gen. Then we play a tournament with the last five
    models. The winning model (winning_gen) is used as the start_gen for the next generation.
    """

    start_gen: GenID
    trained_gen: GenID
    date: datetime = datetime.now()

    training_data: Optional[List[Sample]] = None
    validation_data: Optional[List[Sample]] = None

    tournament: Optional[TournamentResult] = None
    winning_gen: Optional[GenID] = None


async def train_gen(
    n_games: int,
    n_processes: int,
    max_coros_per_process: int,
    mcts_iterations: int,
    exploration_constant,
    batch_size: int,
    device: torch.device,
) -> TrainingState:
    """Trains a new model generation. See TrainingGen docstring."""
    logger = logging.getLogger(__name__)

    state = TrainingState.load_training_state()
    gen, model = state.create_new_gen()
    model.to(device)
    logger.info(f"Training gen {gen.trained_gen} from gen {gen.start_gen}")

    # Self play
    if gen.training_data is None or gen.validation_data is None:
        logger.info("No cached samples found. Generating samples from self-play.")
        samples = await generate_samples(
            model=model,
            n_games=n_games,
            mcts_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            max_coros_per_process=max_coros_per_process,
            nn_max_batch_size=20000,
            device=device,
            n_processes=n_processes,
        )
        gen.training_data, gen.validation_data = PosDataModule.split_samples(samples)
        logger.info(f"Done generating {len(samples)} samples. Caching for re-use.")
        state.save_training_state()
    else:
        logger.info("Loaded cached samples")

    # Training
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
    )
    data_module = PosDataModule(gen.training_data, gen.validation_data, batch_size)
    logger.info(f"Beginning training gen {gen.trained_gen}")
    trainer.fit(model, data_module)
    logger.info(f"Finished training gen {gen.trained_gen}")

    # Play tournament with best players
    top_gens = [gen.trained_gen] + state.get_top_models()
    players: List[Player] = [
        ModelPlayer(gen_id=gen, model=state.get_model(gen), device=device)
        for gen in top_gens
    ]
    players.append(RandomPlayer())
    players.append(UniformPlayer())
    players = players[:5]  # Only play top 5 models
    logger.info(
        f"Playing tournament with {len(players)} players: {', '.join(p.name for p in players)}"
    )
    gen.tournament = await play_tournament(
        players=players,
        games_per_match=20,
        mcts_iterations=mcts_iterations,
        exploration_constant=exploration_constant,
    )
    logger.info(f"Tournament results:\n{gen.tournament.scores_table()}")
    gen.winning_gen = gen.trained_gen
    logger.info(f"Winning gen: {gen.winning_gen} (used for training next gen)")

    state.save_training_state()
    return state


SampleTensor = NewType(
    "SampleTensor",
    Tuple[
        GameID,  # Game ID
        torch.Tensor,  # Pos
        torch.Tensor,  # Policy
        torch.Tensor,  # Value
    ],
)


class PosDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_data: List[Sample],
        validation_data: List[Sample],
        batch_size: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        training_data += [self.flip_sample(s) for s in training_data]
        validation_data += [self.flip_sample(s) for s in validation_data]
        self.training_data = [self.sample_to_tensor(s) for s in training_data]
        self.validation_data = [self.sample_to_tensor(s) for s in validation_data]

    @staticmethod
    def flip_sample(sample: Sample) -> Sample:
        """Flips the sample and policy horizontally to account for connect four symmetry."""
        game_id, pos, policy, value = sample
        # copy is needed as flip returns a view with negative stride that is incompatible w torch
        pos = Pos(np.fliplr(pos).copy())
        policy = Policy(np.flip(policy).copy())
        return Sample((game_id, pos, policy, value))

    @staticmethod
    def sample_to_tensor(sample: Sample) -> SampleTensor:
        game_id, pos, policy, value = sample
        pos_t = torch.from_numpy(pos).float()
        policy_t = torch.from_numpy(policy).float()
        value_t = torch.tensor(value, dtype=torch.float)
        assert pos_t.shape == (N_ROWS, N_COLS)
        assert policy_t.shape == (N_COLS,)
        assert value_t.shape == ()
        return SampleTensor((game_id, pos_t, policy_t, value_t))

    @staticmethod
    def split_samples(
        samples: List[Sample], split_ratio=0.8
    ) -> Tuple[List[Sample], List[Sample]]:
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
