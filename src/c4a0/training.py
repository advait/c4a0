"""
Generation-based network training, alternating between self-play and training.
"""

import copy
from dataclasses import dataclass
from datetime import datetime
from glob import glob
import logging
import os
import pickle
from typing import Dict, List, NewType, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader

from c4a0.nn import ConnectFourNet
from c4a0.tournament import (
    ModelID,
    ModelPlayer,
    Player,
    TournamentResult,
    play_tournament,
)

import c4a0_rust  # type: ignore
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS, Sample  # type: ignore


@dataclass
class TrainingState:
    """The full historical state of the training process."""

    models: Dict[ModelID, ConnectFourNet]
    training_gens: List["TrainingGen"]

    def get_next_gen_id(self) -> ModelID:
        return ModelID(max(self.models.keys()) + 1)  # type: ignore

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
        initial_model_id = 2
        new_state = TrainingState(
            models={ModelID(initial_model_id): ConnectFourNet()},
            training_gens=[TrainingGen(source_id=ModelID(initial_model_id))],
        )
        return new_state

    def save_training_state(self):
        for model in self.models.values():
            model.cpu()  # Relocate models to cpu before pickling
        now_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        with open(
            os.path.join("training", f"training_state_{now_date}.pkl"), "wb"
        ) as f:
            pickle.dump(self, f)

    def get_model(self, model_id: ModelID) -> ConnectFourNet:
        return self.models[model_id]

    def register_new_model(self, model: ConnectFourNet) -> ModelID:
        next_id = ModelID(max(self.models.keys()) + 1)
        self.models[next_id] = model
        return next_id

    def continue_training(
        self,
        n_games: int,
        mcts_iterations: int,
        exploration_constant: float,
        batch_size: int,
        device: torch.device,
    ):
        logger = logging.getLogger(__name__)

        latest_gen = self.training_gens[-1]
        if latest_gen.child_id is not None:
            logger.info(
                f"Finished training gen {latest_gen.child_id}. Starting new generation."
            )
            next_gen = TrainingGen(source_id=latest_gen.child_id)
            self.training_gens.append(next_gen)
            latest_gen = next_gen

        latest_gen.train(
            self, n_games, mcts_iterations, exploration_constant, batch_size, device
        )


@dataclass
class TrainingGen:
    """
    State of a single training generation.

    We begin with the source_gen model and then use self play to generate training_samples.
    Then we train a new model called child_gen. Then we play a tournament with the last five
    models. The winning model is used as the source_gen for the next generation.
    """

    source_id: ModelID
    child_id: Optional[ModelID] = None
    date: datetime = datetime.now()
    games: Optional[c4a0_rust.PlayGamesResult] = None
    tournament: Optional[TournamentResult] = None

    def train(
        self,
        state: TrainingState,
        n_games: int,
        mcts_iterations: int,
        exploration_constant,
        batch_size: int,
        device: torch.device,
    ) -> TrainingState:
        """Trains a new model generation. See TrainingGen docstring."""
        logger = logging.getLogger(__name__)
        assert self.child_id is None, "Cant re-train an already trained generation"

        logger.info(f"Training new gen from gen {self.source_id}")

        # Self play
        if self.games is None:
            logger.info("No cached samples found. Generating samples from self-play.")
            model = state.get_model(self.source_id)
            model.to(device)
            reqs = [c4a0_rust.GameMetadata(id, 0, 0) for id in range(n_games)]
            self.games = c4a0_rust.play_games(
                reqs,
                batch_size,
                mcts_iterations,
                exploration_constant,
                lambda player_id, pos: model.forward_numpy(pos),
            )
            state.save_training_state()
            logger.info(f"Done generating {len(self.games.results)} games")  # type: ignore
        else:
            logger.info(f"Loaded {len(self.games.results)} cached games")

        # TODO: Include smaples from prior generations

        # Training
        model = copy.deepcopy(state.get_model(self.source_id))
        train, test = self.games.split_train_test(0.8, 1337)  # type: ignore
        data_module = SampleDataModule(train, test, batch_size)
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            devices="auto",
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ],
        )
        logger.info("Beginning training")
        model.train()  # Switch batch normalization to train mode for training bn params
        trainer.fit(model, data_module)
        self.child_id = state.register_new_model(model)
        logger.info(f"Finished training gen {self.child_id} from gen {self.source_id}")

        # Play tournament with parent and child
        players: List[Player] = [
            ModelPlayer(model_id=id, model=state.get_model(id), device=device)
            for id in [self.source_id, self.child_id]
        ]
        logger.info(
            f"Playing tournament with {len(players)} players: {', '.join(p.name for p in players)}"
        )
        model.eval()  # Switch batch normalization to eval mode for tournament
        self.tournament = play_tournament(
            players=players,
            games_per_match=100,
            batch_size=batch_size,
            mcts_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
        )
        table = self.tournament.scores_table(
            lambda id: [p.name for p in players if p.model_id == id][0]
        )
        logger.info(f"Tournament results:\n{table}")
        winning_gen = self.tournament.get_top_models()[0]
        logger.info(f"Winning gen: {winning_gen} (used for training next gen)")

        state.save_training_state()
        return state


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
