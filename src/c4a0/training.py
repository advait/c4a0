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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader

from c4a0.nn import ConnectFourNet
from c4a0.tournament import (
    ModelID,
    ModelPlayer,
    Player,
    RandomPlayer,
    TournamentResult,
    UniformPlayer,
    # play_tournament,
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

    def get_model(self, gen: ModelID) -> ConnectFourNet:
        return self.models[gen]

    def get_top_models(self) -> List[ModelID]:
        """Returns the top models from the most recent tournamnet."""
        for training_gen in reversed(self.training_gens):
            if training_gen.tournament is None:
                continue
            return training_gen.tournament.get_top_models()
        return []

    def get_top_model(self) -> Tuple[ModelID, ConnectFourNet]:
        """Gets the winner of the most recent tournament."""
        logger = logging.getLogger(__name__)
        try:
            gen = self.get_top_models()[0]
            return gen, self.get_model(gen)
        except IndexError:
            logger.info("No models found, creating a new one.")
            gen = ModelID(0)
            model = ConnectFourNet()
            self.models[gen] = model
            return gen, model

    def create_new_gen(self) -> Tuple["TrainingGen", ConnectFourNet]:
        """Creates a new generation of training by cloning the best model."""
        if len(self.training_gens) > 0 and self.training_gens[-1].tournament is None:
            # If the latest generation hasn't completed training, return it
            latest_gen = self.training_gens[-1]
            return latest_gen, self.get_model(latest_gen.child_gen)

        best_gen_id, best_model = self.get_top_model()
        next_gen = self.get_next_gen_id()
        next_model = ConnectFourNet()
        next_model.load_state_dict(best_model.state_dict())  # Clone model
        self.models[next_gen] = next_model
        training_gen = TrainingGen(source_gen=best_gen_id, child_gen=next_gen)
        self.training_gens.append(training_gen)
        return training_gen, next_model


@dataclass
class TrainingGen:
    """
    State of a single training generation.

    We begin with the source_gen model and then use self play to generate training_samples.
    Then we train a new model called child_gen. Then we play a tournament with the last five
    models. The winning model is used as the source_gen for the next generation.
    """

    source_gen: ModelID
    child_gen: ModelID
    date: datetime = datetime.now()
    games: Optional[c4a0_rust.PlayGamesResult] = None
    tournament: Optional[TournamentResult] = None


async def train_gen(
    n_games: int,
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
    logger.info(f"Training gen {gen.child_gen} from gen {gen.source_gen}")

    # Self play
    if gen.games is None:
        logger.info("No cached samples found. Generating samples from self-play.")
        reqs = [c4a0_rust.GameMetadata(id, 0, 0) for id in range(n_games)]
        gen.games = c4a0_rust.play_games(
            reqs,
            batch_size,
            mcts_iterations,
            exploration_constant,
            lambda player_id, pos: model.forward_numpy(pos),
        )
        breakpoint()
        state.save_training_state()
        logger.info(f"Done generating {len(gen.games.results)} games")  # type: ignore
    else:
        logger.info(f"Loaded {len(gen.games.results)} cached games")

    # TODO: Include smaples from prior generations

    # Training
    train, test = gen.games.split_train_test(0.8, 1337)  # type: ignore
    data_module = SampleDataModule(train, test, batch_size)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
    )
    logger.info(f"Beginning training gen {gen.child_gen}")
    model.train()  # Switch batch normalization to train mode for training bn params
    trainer.fit(model, data_module)
    logger.info(f"Finished training gen {gen.child_gen}")

    exit(0)

    # Play tournament with best players
    top_gens = [gen.child_gen] + state.get_top_models()
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
    model.eval()  # Switch batch normalization to eval mode for tournament
    gen.tournament = await play_tournament(
        players=players,
        games_per_match=20,
        mcts_iterations=mcts_iterations,
        exploration_constant=exploration_constant,
    )
    logger.info(f"Tournament results:\n{gen.tournament.scores_table()}")
    winning_gen = gen.tournament.get_top_models()[0]
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
