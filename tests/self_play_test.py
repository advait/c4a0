from c4 import N_COLS, is_game_over

import torch
import pytest

from self_play import generate_samples


@pytest.mark.asyncio
async def test_gen_sample():
    poss = await generate_samples(
        model=MockModel(),
        n_games=1,
        mcts_iterations=5,
        exploration_constant=1.4,
        n_processes=1,
    )

    _, pos, _, value = poss[-1]  # The last position in the game

    assert len(poss) > 6
    assert value == -1 or value == 0 or value == 1
    assert is_game_over(pos) is not None


class MockModel:
    def __call__(self, x: torch.Tensor):
        """Simple mock model that always returns a value of zero and a uniform policy."""
        batch_size = x.shape[0]
        policy = torch.ones(N_COLS) / N_COLS
        policy = policy.repeat(batch_size, 1)
        value = torch.tensor(0.0)
        value = value.repeat(batch_size)
        return policy.float(), value.float()

    def forward(self, x):
        return self(x)
