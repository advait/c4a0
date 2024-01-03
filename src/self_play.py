"""
Generating training data via self-play
"""

import asyncio
from collections import defaultdict
import logging
from typing import Awaitable, List, NewType, Optional, Tuple


import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from c4 import (
    N_COLS,
    STARTING_POS,
    Pos,
    get_ply,
    is_game_over,
    make_move,
    pos_from_str,
    post_to_str,
)
from mcts import AsyncEvaluatePos, mcts_async
from nn import ConnectFourNet, Policy, Value
from utils import model_forward_bg_thread

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
    tb_logger: TensorBoardLogger,
    eval2: Optional[ConnectFourNet] = None,
) -> List[Sample]:
    """Generates a seqence of Samples for n_games number of games."""
    logger = logging.getLogger(__name__)

    async_eval_1, stop_batching_1 = batch_model(eval1)
    async_eval_2, stop_batching_2 = async_eval_1, stop_batching_1
    if eval2 is not None:
        async_eval_2, stop_batching_2 = batch_model(eval2)

    logger.info("Started game generation")
    out_samples: List[Sample] = []
    await asyncio.gather(
        *[
            _gen_game(
                async_eval_1,
                async_eval_2,
                game_id,
                mcts_iterations,
                exploration_constant,
                out_samples,
            )
            for game_id in range(n_games)
        ]
    )
    logger.info(f"Finished game generation with {len(out_samples)} samples")

    # Stop the batching loops
    stop_batching_1.set_result(None)
    if stop_batching_2 is not stop_batching_1:
        stop_batching_2.set_result(None)

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

    for game_id, pos, policy in results:
        out_samples.append((game_id, pos, policy, final_value))

        # If the position is horizontally asymmetric, add the flipped variation as well
        flipped_pos = np.flip(pos, axis=0)
        if not np.array_equal(pos, flipped_pos):
            out_samples.append((game_id, flipped_pos, policy, final_value))

        # Alternate the sign of the final value as the board perspective changes between each move
        final_value *= -1


def batch_model(
    model: ConnectFourNet,
    max_batch_size: int = 20000,
) -> Tuple[AsyncEvaluatePos, asyncio.Future]:
    """
    Given a model, returns a function that batches multiple model calls together for more efficient
    inference. Also returns a future that, when set, will cause the batching loop to stop.
    """
    logger = logging.getLogger(__name__)

    pos_queue: List[Tuple(Pos, asyncio.Future)] = []

    async def ret_fn(pos: Pos) -> Tuple[Policy, Value]:
        """
        Queues a position for batch evaluation, returning a future that will be set with the result
        after the batch inference happens.
        """
        future = asyncio.get_running_loop().create_future()
        pos_queue.append((pos, future))
        return await future

    # Future that, when set, will cause the batching loop below to stop
    stop_loop_future = asyncio.get_running_loop().create_future()

    async def process_pos_queue():
        logger.info("Starting batch inference pos processing")
        while True:
            if len(pos_queue) > 0:
                # Group identical positions together to avoid duplicate work
                pos_dict = defaultdict(list)
                for pos, future in pos_queue[:max_batch_size]:
                    # Note that we convert np.ndarray into a string for hashing
                    pos_dict[post_to_str(pos)].append(future)
                del pos_queue[:max_batch_size]
                positions = [pos_from_str(s) for s in pos_dict.keys()]
                futures = list(pos_dict.values())

                pos_tensor = torch.from_numpy(np.array(positions)).float().to("cuda")
                policies, values = await model_forward_bg_thread(model, pos_tensor)
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()

                for i in range(len(positions)):
                    for future in futures[i]:
                        future.set_result((policies[i], values[i]))
            elif stop_loop_future.done():
                logger.info(
                    "Finished game generation, exiting queue processing coroutine"
                )
                return
            else:
                await asyncio.sleep(0)  # Yield to other coroutines

    # Run process_pos_queue on current loop without blocking
    asyncio.create_task(process_pos_queue())

    return ret_fn, stop_loop_future
