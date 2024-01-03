"""
Generation-based network training, alternating between self-play and training.
"""
import itertools
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from nn import ConnectFourNet
from self_play import gen_samples


async def train_gen(mcts_iterations=100, exploration_constant=1.4, n_games=100):
    logger = logging.getLogger(__name__)

    model = ConnectFourNet().to("cuda")

    logger.info(
        f"Generating {n_games} self-play games:\n"
        f"- MCTS iterations: {mcts_iterations}\n"
        f"- Exploration constant: {exploration_constant}"
    )
    samples = await gen_samples(
        eval1=model,
        n_games=n_games,
        mcts_iterations=mcts_iterations,
        exploration_constant=exploration_constant,
    )
    training_data, validation_data = split_samples(samples)
    logger.info("Done generating games")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(
                dirpath="checkpoints/", save_top_k=1, monitor="val_loss", mode="min"
            ),
        ],
    )
    trainer.fit(model, train_dataloader=training_data, val_dataloader=validation_data)


def split_samples(samples, split_ratio=0.8):
    def sample_game_id(sample, f=lambda x: x):
        return f(sample[0])

    min_game_id = min(samples, key=sample_game_id)
    max_game_id = max(samples, key=sample_game_id)
    n_games = max_game_id - min_game_id + 1
    split_id = int(n_games * split_ratio) + min_game_id
    return list(
        itertools.groupby(samples, key=sample_game_id(f=lambda id: id > split_id))
    )
