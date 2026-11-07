"""
Generation-based network training, alternating between self-play and training.
"""

import copy
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NewType

import pytorch_lightning as pl
import torch
from loguru import logger
from pydantic import BaseModel
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset

import c4a0_rust
from c4a0.evaluation import SuiteEval, evaluate_model
from c4a0.labels import Split, Suite, load_suite
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.utils import BestModelCheckpoint, GenerationTrainer, lightning_accelerator
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS, PlayGamesResult, Sample


@dataclass(frozen=True)
class EvalConfig:
    """What each generation's saved model is scored on before its metadata is written."""

    suite: Suite
    mcts_iterations: tuple[int, ...] = ()
    """MCTS budgets to score in addition to the raw network. Empty means raw only."""


def default_eval_config(mcts_iterations: Sequence[int] = ()) -> EvalConfig:
    return EvalConfig(load_suite(Split.DEV), tuple(mcts_iterations))


class TrainingGen(BaseModel):
    """
    Represents a single generation of training.
    """

    created_at: datetime
    gen_n: int
    n_mcts_iterations: int
    c_exploration: float
    c_ply_penalty: float
    self_play_batch_size: int
    training_batch_size: int
    greedy_after_ply: int | None = None
    root_noise_alpha: float | None = None
    root_noise_frac: float = 0.0
    mask_terminal_policy: bool = False
    parent: datetime | None = None
    val_loss: float | None = None
    solver_score: float | None = None
    """Legacy: the PARENT's self-play policy score (half credit for slower wins, ties broken by
    column order). Never written again; use `evals`."""
    evals: dict[str, SuiteEval] = {}
    """Suite name -> score of THIS generation's saved model.pkl."""

    @staticmethod
    def _gen_folder(created_at: datetime, base_dir: str) -> str:
        return os.path.join(base_dir, created_at.isoformat())

    def gen_folder(self, base_dir: str) -> str:
        return TrainingGen._gen_folder(self.created_at, base_dir)

    def publish(
        self,
        base_dir: str,
        games: PlayGamesResult | None,
        model: ConnectFourNet,
        eval_config: EvalConfig,
    ) -> None:
        """Saves games and model, scores the model as reloaded from model.pkl, then writes
        metadata.json last. If scoring fails, no metadata is written and loaders skip the
        generation."""
        gen_dir = self.gen_folder(base_dir)
        os.makedirs(gen_dir, exist_ok=True)

        play_result_path = os.path.join(gen_dir, "games.pkl")
        with open(play_result_path, "wb") as f:
            pickle.dump(games, f)

        model_path = Path(gen_dir) / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        suite_eval = evaluate_model(
            model_path,
            eval_config.suite,
            eval_config.mcts_iterations,
            self.c_exploration,
            self.c_ply_penalty,
        )
        self.evals[suite_eval.suite] = suite_eval
        logger.info(
            "Gen {} on {}: raw top1 {}, mcts top1 {}, value accuracy {}",
            self.gen_n,
            suite_eval.suite,
            suite_eval.raw.top1,
            {n: e.top1 for n, e in suite_eval.mcts.items()},
            suite_eval.value.accuracy,
        )
        self.save_metadata(base_dir)

    def save_metadata(self, base_dir: str) -> None:
        """Writes metadata.json atomically (temp file, then rename)."""
        gen_dir = self.gen_folder(base_dir)
        os.makedirs(gen_dir, exist_ok=True)

        metadata_path = os.path.join(gen_dir, "metadata.json")
        tmp_path = metadata_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        os.replace(tmp_path, metadata_path)

    @staticmethod
    def load(base_dir: str, created_at: datetime) -> "TrainingGen":
        gen_folder = TrainingGen._gen_folder(created_at, base_dir)
        with open(os.path.join(gen_folder, "metadata.json")) as f:
            return TrainingGen.model_validate_json(f.read())

    @staticmethod
    def _published_timestamps(base_dir: str) -> list[datetime]:
        """Newest first. Directories without metadata.json are unpublished and skipped."""
        return sorted(
            [
                datetime.fromisoformat(f)
                for f in os.listdir(base_dir)
                if os.path.isfile(os.path.join(base_dir, f, "metadata.json"))
            ],
            reverse=True,
        )

    @staticmethod
    def load_all(base_dir: str) -> list["TrainingGen"]:
        return [
            TrainingGen.load(base_dir, t)
            for t in TrainingGen._published_timestamps(base_dir)
        ]

    @staticmethod
    def load_latest(base_dir: str) -> "TrainingGen":
        timestamps = TrainingGen._published_timestamps(base_dir)
        if not timestamps:
            raise FileNotFoundError("No existing generations")
        return TrainingGen.load(base_dir, timestamps[0])

    @staticmethod
    def load_latest_with_default(
        base_dir: str,
        n_mcts_iterations: int,
        c_exploration: float,
        c_ply_penalty: float,
        self_play_batch_size: int,
        training_batch_size: int,
        model_config: ModelConfig,
        eval_config: EvalConfig,
        greedy_after_ply: int | None = None,
        root_noise_alpha: float | None = None,
        root_noise_frac: float = 0.0,
        mask_terminal_policy: bool = False,
    ) -> "TrainingGen":
        try:
            return TrainingGen.load_latest(base_dir)
        except FileNotFoundError:
            logger.info("No existing generations found, initializing root")
            gen = TrainingGen(
                created_at=datetime.now(),
                gen_n=0,
                n_mcts_iterations=n_mcts_iterations,
                c_exploration=c_exploration,
                c_ply_penalty=c_ply_penalty,
                self_play_batch_size=self_play_batch_size,
                training_batch_size=training_batch_size,
                greedy_after_ply=greedy_after_ply,
                root_noise_alpha=root_noise_alpha,
                root_noise_frac=root_noise_frac,
                mask_terminal_policy=mask_terminal_policy,
            )
            model = ConnectFourNet(model_config)
            gen.publish(base_dir, None, model, eval_config)
            return gen

    def get_games(self, base_dir: str) -> PlayGamesResult | None:
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "games.pkl"), "rb") as f:
            return pickle.load(f)

    def get_model(self, base_dir: str) -> ConnectFourNet:
        """Gets the model for this generation."""
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "model.pkl"), "rb") as f:
            model = pickle.load(f)
            return model


def train_single_gen(
    base_dir: str,
    device: torch.device,
    parent: TrainingGen,
    n_self_play_games: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    self_play_batch_size: int,
    training_batch_size: int,
    eval_config: EvalConfig,
    greedy_after_ply: int | None = None,
    root_noise_alpha: float | None = None,
    root_noise_frac: float = 0.0,
    mask_terminal_policy: bool = False,
) -> TrainingGen:
    """
    Trains a new generation from the given parent.
    First generate games using c4a0_rust.play_games.
    Then train a new model based on the parent model using the generated samples.
    Finally, publish the games and model, scoring the saved model with `eval_config`.
    """
    gen_n = parent.gen_n + 1
    logger.info(f"Beginning new generation {gen_n} from {parent.gen_n}")

    # TODO: log experiment metadata in MLFlow

    # Self play
    model = parent.get_model(base_dir)
    model.to(device)
    reqs = [c4a0_rust.GameMetadata(id, 0, 0) for id in range(n_self_play_games)]
    games = c4a0_rust.play_games(
        reqs,
        self_play_batch_size,
        n_mcts_iterations,
        c_exploration,
        c_ply_penalty,
        lambda player_id, pos: model.forward_numpy(pos),
        greedy_after_ply=greedy_after_ply,
        root_noise_alpha=root_noise_alpha,
        root_noise_frac=root_noise_frac,
    )

    # Training
    logger.info("Beginning training")
    model = copy.deepcopy(model)
    train, test = games.split_train_test(
        0.8, 1337, mask_terminal_policy=mask_terminal_policy
    )
    data_module = SampleDataModule(train, test, training_batch_size)
    best_model_cb = BestModelCheckpoint[ConnectFourNet](monitor="val_loss", mode="min")
    trainer = GenerationTrainer(
        max_epochs=100,
        accelerator=lightning_accelerator(device),
        devices=1,
        callbacks=[
            best_model_cb,
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        num_sanity_val_steps=0,
    )
    trainer.gen_n = gen_n
    model.train()  # Switch batch normalization to train mode for training bn params
    trainer.fit(model, data_module)
    logger.info("Finished training")

    best_model = best_model_cb.get_best_model()
    gen = TrainingGen(
        created_at=datetime.now(),
        gen_n=parent.gen_n + 1,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        greedy_after_ply=greedy_after_ply,
        root_noise_alpha=root_noise_alpha,
        root_noise_frac=root_noise_frac,
        mask_terminal_policy=mask_terminal_policy,
        parent=parent.created_at,
        val_loss=best_model_cb.best_score,
    )
    gen.publish(base_dir, games, best_model, eval_config)
    return gen


def training_loop(
    base_dir: str,
    device: torch.device,
    n_self_play_games: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    self_play_batch_size: int,
    training_batch_size: int,
    model_config: ModelConfig,
    max_gens: int | None = None,
    eval_config: EvalConfig | None = None,
    greedy_after_ply: int | None = None,
    root_noise_alpha: float | None = None,
    root_noise_frac: float = 0.0,
    mask_terminal_policy: bool = False,
) -> TrainingGen:
    """Main training loop. Sequentially trains generation after generation. Each generation's
    saved model is scored on `eval_config` (default: raw network on the v1 dev suite)."""
    eval_config = eval_config or default_eval_config()
    logger.info("Beginning training loop")
    logger.info("device: {}", device)
    logger.info("n_self_play_games: {}", n_self_play_games)
    logger.info("n_mcts_iterations: {}", n_mcts_iterations)
    logger.info("c_exploration: {}", c_exploration)
    logger.info("c_ply_penalty: {}", c_ply_penalty)
    logger.info("self_play_batch_size: {}", self_play_batch_size)
    logger.info("training_batch_size: {}", training_batch_size)
    logger.info("model_config: \n{}", model_config.model_dump_json(indent=2))
    logger.info("max_gens: {}", max_gens)
    logger.info(
        "eval: {} with MCTS budgets {}",
        eval_config.suite.name,
        eval_config.mcts_iterations,
    )

    gen = TrainingGen.load_latest_with_default(
        base_dir=base_dir,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        model_config=model_config,
        eval_config=eval_config,
        greedy_after_ply=greedy_after_ply,
        root_noise_alpha=root_noise_alpha,
        root_noise_frac=root_noise_frac,
        mask_terminal_policy=mask_terminal_policy,
    )

    while True:
        gen = train_single_gen(
            base_dir=base_dir,
            device=device,
            parent=gen,
            n_self_play_games=n_self_play_games,
            n_mcts_iterations=n_mcts_iterations,
            c_exploration=c_exploration,
            c_ply_penalty=c_ply_penalty,
            self_play_batch_size=self_play_batch_size,
            training_batch_size=training_batch_size,
            eval_config=eval_config,
            greedy_after_ply=greedy_after_ply,
            root_noise_alpha=root_noise_alpha,
            root_noise_frac=root_noise_frac,
            mask_terminal_policy=mask_terminal_policy,
        )
        if max_gens is not None and gen.gen_n >= max_gens:
            return gen


SampleTensor = NewType(
    "SampleTensor",
    tuple[
        torch.Tensor,  # Pos
        torch.Tensor,  # Policy
        torch.Tensor,  # Q Value with penalty
        torch.Tensor,  # Q Value without penalty
    ],
)


class SampleDataset(Dataset[SampleTensor]):
    """Map-style adapter for the precomputed samples consumed by DataLoader."""

    def __init__(self, samples: list[SampleTensor]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleTensor:
        return self.samples[index]


class SampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_data: list[Sample],
        validation_data: list[Sample],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        training_data += [s.flip_h() for s in training_data]
        validation_data += [s.flip_h() for s in validation_data]
        self.training_data = [self.sample_to_tensor(s) for s in training_data]
        self.validation_data = [self.sample_to_tensor(s) for s in validation_data]

    @staticmethod
    def sample_to_tensor(sample: Sample) -> "SampleTensor":
        pos, policy, q_penalty, q_no_penalty = sample.to_numpy()
        pos_t = torch.from_numpy(pos)
        policy_t = torch.from_numpy(policy)
        q_penalty_t = torch.from_numpy(q_penalty)
        q_no_penalty_t = torch.from_numpy(q_no_penalty)
        assert pos_t.shape == (BUF_N_CHANNELS, N_ROWS, N_COLS)
        assert policy_t.shape == (N_COLS,)
        assert q_penalty_t.shape == ()
        assert q_no_penalty.shape == ()
        return SampleTensor((pos_t, policy_t, q_penalty_t, q_no_penalty_t))

    def train_dataloader(self) -> DataLoader[SampleTensor]:
        return DataLoader(
            SampleDataset(self.training_data),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[SampleTensor]:
        return DataLoader(
            SampleDataset(self.validation_data),
            batch_size=self.batch_size,
        )


def parse_lr_schedule(floats: list[float]) -> dict[int, float]:
    """Parses an lr_schedule sequence like "0 2e-3 10 8e-4" into a dict of {0: 2e-3, 10: 8e-4}."""
    assert len(floats) % 2 == 0, "lr_schedule must have an even number of elements"
    schedule = {}
    for i in range(0, len(floats), 2):
        threshold = int(floats[i])
        assert threshold == floats[i], (
            "lr_schedule must alternate between gen_id (int) and lr (float)"
        )
        lr = floats[i + 1]
        schedule[threshold] = lr
    return schedule
