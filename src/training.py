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
    game_ids = set(sample[0] for sample in samples)
    split_id = int(len(game_ids) * split_ratio) + min(game_ids)
    return list(itertools.groupby(samples, key=lambda sample: sample[0] < split_id))
