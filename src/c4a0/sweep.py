import functools
from typing import List

from loguru import logger
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from c4a0.training import SampleDataModule, TrainingGen
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0_rust import Sample  # type: ignore


def load_samples(base_dir: str, n_gens: int = 5) -> List[Sample]:
    gens = TrainingGen.load_all(base_dir)[:n_gens]
    game_results = [gen.get_games(base_dir) for gen in gens]
    samples = [
        sample
        for game_result in game_results
        if game_result
        for results in game_result.results
        for sample in results.samples
    ]
    return samples


def objective(trial: optuna.Trial, samples: List[Sample]):
    model_config = ModelConfig(
        n_residual_blocks=trial.suggest_int("n_residual_blocks", 0, 1),
        conv_filter_size=trial.suggest_int("conv_filter_size", 16, 64),
        n_policy_layers=trial.suggest_int("n_policy_layers", 0, 4),
        n_value_layers=trial.suggest_int("n_value_layers", 0, 2),
        lr_schedule={0: trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)},
        l2_reg=trial.suggest_loguniform("l2_reg", 1e-5, 1e-3),
    )
    model = ConnectFourNet(model_config)

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    split_idx = int(0.8 * len(samples))
    train, test = samples[:split_idx], samples[split_idx:]
    data_module = SampleDataModule(train, test, batch_size)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ],
        enable_progress_bar=False,  # Disable progress bar for cleaner logs
    )

    trainer.fit(model, data_module)

    val_loss = trainer.callback_metrics["val_loss"].item()
    trial.report(val_loss, step=trainer.current_epoch)
    return val_loss


def perform_hparam_sweep(base_dir: str, study_name: str = "sweep_hparam"):
    samples = load_samples(base_dir)

    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        functools.partial(objective, samples=samples), n_trials=100, catch=(Exception,)
    )

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    logger.info("")
    logger.info("Study statistics:")
    logger.info(f"  Finished trials: {len(study.trials)}")
    logger.info(
        f"  Pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}"
    )
    logger.info(
        f"  Completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}"
    )
