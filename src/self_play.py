"""
Generating training data via self-play
"""

import asyncio
import logging
from typing import Awaitable, List, NewType, Optional, Tuple

import numpy as np
import torch

from c4 import N_COLS, STARTING_POS, Pos, get_ply, is_game_over, make_move, pos_str
from mcts import AsyncEvaluatePos, mcts_async
from nn import ConnectFourNet, Policy, Value
from utils import unzip

NetworkID = NewType("NetworkID", int)
"""Which network generation this sample is from.""" ""

GameID = NewType("GameID", int)
"""Which game this sample is from."""

Sample = NewType("Sample", Tuple[GameID, Pos, Policy, Value])
"""A sample of training data representing a position and the desired policy and value outputs."""


async def gen_samples(
    eval1: ConnectFourNet,
    n_games: int,
    mcts_iterations: int,
    exploration_constant: float,
    eval2: Optional[ConnectFourNet] = None,
) -> List[Sample]:
    """Generates a seqence of Samples for n_games number of games."""
    logger = logging.getLogger(__name__)
    if eval2 is not None:
        raise NotImplementedError("TODO: Implement two-player self-play")

    out_samples: List[Sample] = []
    pos_processing_queue: List[Tuple(Pos, asyncio.Future)] = []

    async def enqueue_pos(pos: Pos):
        future = asyncio.get_running_loop().create_future()
        pos_processing_queue.append((pos, future))
        return await future

    game_gen_coro = asyncio.gather(
        *[
            _gen_game(
                enqueue_pos,
                enqueue_pos,
                game_id,
                mcts_iterations,
                exploration_constant,
                out_samples,
            )
            for game_id in range(n_games)
        ]
    )
    logger.info("Started game generation")

    async def process_pos_queue():
        """
        Asynchronously process position queue by batching multiple positions together for nn
        inference.
        """
        logger.info("Starting processing queue")
        while True:
            if len(pos_processing_queue) > 0:
                positions, futures = unzip(pos_processing_queue)
                pos_processing_queue.clear()
                pos_tensor = torch.from_numpy(np.array(positions)).float().to("cuda")
                policies, values = await eval1.forward_bg_thread(pos_tensor)
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
                for i in range(len(futures)):
                    futures[i].set_result((policies[i], values[i]))
            elif game_gen_coro.done():
                logger.info(
                    "Finished game generation, exiting queue processing coroutine"
                )
                return
            else:
                await asyncio.sleep(0)

    process_queue_coro = process_pos_queue()  # Start queue processing coroutine
    await process_queue_coro  # Block on queue processing coroutine
    await game_gen_coro  # Block on game generation

    # Sort by game_id and then ply
    out_samples.sort(key=lambda t: (t[0], get_ply(t[1])))
    return out_samples


async def _gen_game(
    eval1: AsyncEvaluatePos,
    eval2: AsyncEvaluatePos,
    game_id: GameID,
    mcts_iterations: int,
    exploration_constant: float,
    out_samples: List[Sample],
) -> Awaitable[None]:
    """
    Generates a single game of self-play, appending the results to out_queue.
    """
    results: List[Tuple[GameID, Pos, Policy]] = []

    # Generate moves until terminal state
    pos = STARTING_POS
    while (res := is_game_over(pos)) is None:
        f = eval1 if get_ply(pos) % 2 == 0 else eval2
        policy = await mcts_async(
            pos,
            eval_pos=f,
            n_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
        )
        results.append((game_id, pos, policy))
        move = np.random.choice(range(len(policy)), p=policy)
        pos = make_move(pos, move)

    # Because there isn't a valid policy a terminal state, we simply use a uniform policy
    final_policy = np.ones(N_COLS) / N_COLS
    results.append((game_id, pos, final_policy))

    final_value = res.value
    if len(results) % 2 == 0:
        # If there are an even number of positions, the final value is flipped for the first move
        final_value *= -1

    for result in results:
        out_samples.append(result + (final_value,))
        # Alternate the sign of the final value as the board perspective changes between each move
        final_value *= -1
