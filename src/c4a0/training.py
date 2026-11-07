"""
Generation-based network training, alternating between self-play and training.
"""

import copy
import math
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
from pydantic import BaseModel, ConfigDict, Field
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

import c4a0_rust
from c4a0.evaluation import SuiteEval, evaluate_model
from c4a0.labels import Split, Suite, load_suite
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.utils import (
    BestModelCheckpoint,
    GenerationTrainer,
    lightning_accelerator,
    tensor_batches,
)
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS, PlayGamesResult, Sample

TemperatureSchedule = list[tuple[int, float]]
DEFAULT_TEMPERATURE_SCHEDULE: tuple[tuple[int, float], ...] = ((4, 4.0), (8, 2.0))


@dataclass(frozen=True)
class EvalConfig:
    """What each generation's saved model is scored on before its metadata is written."""

    suite: Suite
    mcts_iterations: tuple[int, ...] = ()
    """MCTS budgets to score in addition to the raw network. Empty means raw only."""


def default_eval_config(mcts_iterations: Sequence[int] = ()) -> EvalConfig:
    return EvalConfig(load_suite(Split.DEV), tuple(mcts_iterations))


class GenConfig(BaseModel):
    """Per-generation controls, shared by the CLI, loop and published metadata.

    TrainingGen inherits these fields to preserve the flat metadata.json schema.
    Defaults also apply when loading older metadata that omits newer controls.
    """

    model_config = ConfigDict(frozen=True)

    # Older metadata did not record this count; use the historical CLI default.
    n_self_play_games: int = 1700
    n_mcts_iterations: int
    c_exploration: float
    c_ply_penalty: float
    self_play_batch_size: int
    training_batch_size: int
    greedy_after_ply: int | None = None
    root_noise_alpha: float | None = None
    root_noise_frac: float = 0.0
    random_start_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    random_start_max_ply: int = Field(default=12, ge=1)
    value_target_q_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_schedule: tuple[tuple[int, float], ...] = DEFAULT_TEMPERATURE_SCHEDULE
    mask_terminal_policy: bool = False
    replay_gens: int = Field(default=1, ge=1)
    train_max_epochs: int = Field(default=100, ge=1)
    train_patience: int = Field(default=10, ge=1)


class TrainingGen(GenConfig):
    """A published generation with its config fields kept flat for legacy readers."""

    created_at: datetime
    gen_n: int
    parent: datetime | None = None
    val_loss: float | None = None
    solver_score: float | None = None
    """Legacy: the PARENT's self-play policy score (half credit for slower wins, ties broken by
    column order). Never written again; use `evals`."""
    evals: dict[str, SuiteEval] = {}
    """Suite name -> score of THIS generation's saved model.pkl."""

    @property
    def config(self) -> GenConfig:
        """Derive only config fields, including future controls, from flat metadata."""
        return GenConfig.model_validate(
            self.model_dump(include=set(GenConfig.model_fields))
        )

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
        config: GenConfig,
        model_config: ModelConfig,
        eval_config: EvalConfig,
    ) -> "TrainingGen":
        try:
            return TrainingGen.load_latest(base_dir)
        except FileNotFoundError:
            logger.info("No existing generations found, initializing root")
            gen = TrainingGen(
                created_at=datetime.now(),
                gen_n=0,
                **config.model_dump(),
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


def replay_training_games(
    base_dir: str, parent: TrainingGen, current: PlayGamesResult, replay_gens: int
) -> PlayGamesResult:
    """Current games plus up to G-1 ancestors' games, newest first. Follow the parent
    chain so an unrelated generation in the same directory cannot enter the window."""
    if replay_gens < 1:
        raise ValueError("replay_gens must be at least 1")
    combined = current
    ancestor = parent
    for history_index in range(1, replay_gens):
        games = ancestor.get_games(base_dir)
        if games is not None:  # Generation zero has no self-play games.
            combined = combined + games
        if ancestor.parent is None or history_index == replay_gens - 1:
            break
        ancestor = TrainingGen.load(base_dir, ancestor.parent)
    return combined


def train_single_gen(
    base_dir: str,
    device: torch.device,
    parent: TrainingGen,
    config: GenConfig,
    eval_config: EvalConfig,
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
    # game_id seeds random starts, move sampling and root noise in Rust. Ids unique
    # across generations give each generation fresh random streams.
    first_id = gen_n * config.n_self_play_games
    reqs = [
        c4a0_rust.GameMetadata(first_id + i, 0, 0)
        for i in range(config.n_self_play_games)
    ]
    games = c4a0_rust.play_games(
        reqs,
        config.self_play_batch_size,
        config.n_mcts_iterations,
        config.c_exploration,
        config.c_ply_penalty,
        lambda player_id, pos: model.forward_numpy(pos),
        greedy_after_ply=config.greedy_after_ply,
        root_noise_alpha=config.root_noise_alpha,
        root_noise_frac=config.root_noise_frac,
        value_target_q_weight=config.value_target_q_weight,
        temperature_schedule=config.temperature_schedule,
        random_start_frac=config.random_start_frac,
        random_start_max_ply=config.random_start_max_ply,
    )

    # Training
    logger.info("Beginning training")
    model = copy.deepcopy(model)
    training_games = replay_training_games(base_dir, parent, games, config.replay_gens)
    logger.info(
        "Training on {} games from up to {} generations",
        len(training_games.results),
        config.replay_gens,
    )
    train, test = training_games.split_train_test(
        0.8, 1337, mask_terminal_policy=config.mask_terminal_policy
    )
    data_module = SampleDataModule(train, test, config.training_batch_size)
    best_model_cb = BestModelCheckpoint[ConnectFourNet](monitor="val_loss", mode="min")
    trainer = GenerationTrainer(
        max_epochs=config.train_max_epochs,
        accelerator=lightning_accelerator(device),
        devices=1,
        callbacks=[
            best_model_cb,
            EarlyStopping(
                monitor="val_loss", patience=config.train_patience, mode="min"
            ),
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
        **config.model_dump(),
        parent=parent.created_at,
        val_loss=best_model_cb.best_score,
    )
    gen.publish(base_dir, games, best_model, eval_config)
    return gen


def training_loop(
    base_dir: str,
    device: torch.device,
    config: GenConfig,
    model_config: ModelConfig,
    max_gens: int | None = None,
    eval_config: EvalConfig | None = None,
) -> TrainingGen:
    """Main training loop. Sequentially trains generation after generation. Each generation's
    saved model is scored on `eval_config` (default: raw network on the v1 dev suite)."""
    eval_config = eval_config or default_eval_config()
    logger.info("Beginning training loop")
    logger.info("device: {}", device)
    logger.info("config: \n{}", config.model_dump_json(indent=2))
    logger.info("model_config: \n{}", model_config.model_dump_json(indent=2))
    logger.info("max_gens: {}", max_gens)
    logger.info(
        "eval: {} with MCTS budgets {}",
        eval_config.suite.name,
        eval_config.mcts_iterations,
    )

    gen = TrainingGen.load_latest_with_default(
        base_dir=base_dir,
        config=config,
        model_config=model_config,
        eval_config=eval_config,
    )

    while True:
        gen = train_single_gen(
            base_dir=base_dir,
            device=device,
            parent=gen,
            config=config,
            eval_config=eval_config,
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


def _stack(samples: Sequence["SampleTensor"]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.stack(column) for column in zip(*samples, strict=True))


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

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        return tensor_batches(_stack(self.training_data), self.batch_size, True, "cpu")

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        return tensor_batches(
            _stack(self.validation_data), self.batch_size, False, "cpu"
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


def parse_temperature_schedule(spec: str) -> TemperatureSchedule:
    """Parse exclusive ply boundaries and their move temperatures."""
    schedule: TemperatureSchedule = []
    for pair in spec.split(","):
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError("expected a comma list of until_ply:temperature pairs")
        try:
            until_ply = int(parts[0])
        except ValueError as exc:
            raise ValueError("until_ply must be an integer") from exc
        if until_ply < 0:
            raise ValueError("until_ply must be non-negative")
        if schedule and until_ply <= schedule[-1][0]:
            raise ValueError("until_ply values must be strictly increasing")
        try:
            temperature = float(parts[1])
        except ValueError as exc:
            raise ValueError("temperature must be a number") from exc
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        schedule.append((until_ply, temperature))
    return schedule
