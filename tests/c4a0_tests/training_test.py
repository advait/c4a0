import pickle

import pytest
import torch

from c4a0.nn import ModelConfig
from c4a0.training import (
    SampleDataModule,
    TrainingGen,
    parse_lr_schedule,
    training_loop,
)


def _manual_loss(model, samples):
    pos = torch.stack([sample[0] for sample in samples])
    policy_target = torch.stack([sample[1] for sample in samples])
    q_penalty_target = torch.stack([sample[2] for sample in samples])
    q_no_penalty_target = torch.stack([sample[3] for sample in samples])
    model.eval()
    with torch.no_grad():
        policy_logprob, q_penalty_pred, q_no_penalty_pred = model(pos)
        policy_loss = (
            (policy_target * (torch.log(policy_target + model.EPS) - policy_logprob))
            .sum(dim=1)
            .mean()
        )
        q_penalty_loss = ((q_penalty_pred - q_penalty_target) ** 2).mean()
        q_no_penalty_loss = ((q_no_penalty_pred - q_no_penalty_target) ** 2).mean()
    return (policy_loss + (q_penalty_loss + q_no_penalty_loss) / 2.0).item()


def _sum_abs_model_diff(left, right):
    return sum(
        float(
            (value.detach().cpu() - right.state_dict()[key].detach().cpu()).abs().sum()
        )
        for key, value in left.state_dict().items()
    )


def test_training_loop_saves_trained_best_model_and_matching_metadata(tmp_path):
    model_config = ModelConfig(
        n_residual_blocks=1,
        conv_filter_size=8,
        n_policy_layers=1,
        n_value_layers=1,
        lr_schedule=parse_lr_schedule([0, 1e-3]),
        l2_reg=0.0,
    )

    gen = training_loop(
        base_dir=str(tmp_path),
        device=torch.device("cpu"),
        n_self_play_games=4,
        n_mcts_iterations=4,
        c_exploration=6.6,
        c_ply_penalty=0.01,
        self_play_batch_size=16,
        training_batch_size=10_000,
        model_config=model_config,
        max_gens=1,
    )

    parent = TrainingGen.load(str(tmp_path), gen.parent)  # type: ignore[arg-type]
    parent_model = parent.get_model(str(tmp_path))
    saved_model = gen.get_model(str(tmp_path))
    assert _sum_abs_model_diff(parent_model, saved_model) > 0.0

    games = gen.get_games(str(tmp_path))
    assert games is not None
    train, validation = games.split_train_test(0.8, 1337)
    data_module = SampleDataModule(list(train), list(validation), 10_000)

    assert gen.val_loss == pytest.approx(
        _manual_loss(saved_model, data_module.validation_data),
        abs=1e-5,
    )

    with open(gen.gen_folder(str(tmp_path)) + "/model.pkl", "rb") as f:
        pickled_model = pickle.load(f)
    assert _sum_abs_model_diff(saved_model, pickled_model) == pytest.approx(0.0)
