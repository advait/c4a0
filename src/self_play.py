"""
Generating training data via self-play
"""

import asyncio
from collections import defaultdict
import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Awaitable, Callable, Dict, List, NewType, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from c4 import (
    N_COLS,
    STARTING_POS,
    Ply,
    Pos,
    get_ply,
    is_game_over,
    make_move,
    pos_from_bytes,
    pos_to_bytes,
)
from mcts import mcts
from nn import ConnectFourNet, Policy, Value
from utils import async_to_sync

NetworkID = NewType("NetworkID", int)
"""Which network generation this sample is from.""" ""

GameID = NewType("GameID", int)
"""Which game this sample is from."""

Sample = NewType("Sample", Tuple[GameID, Pos, Policy, Value])
"""A sample of training data representing a position and the desired policy and value outputs."""

ProcessID = NewType("ProcessID", int)


def gen_samples_mp(
    nn: ConnectFourNet,
    n_games: int,
    mcts_iterations: int,
    exploration_constant: float,
    n_processes: int = mp.cpu_count() * 2,
    n_coroutines_per_process: int = 100,
    max_nn_batch_size: int = 20000,
) -> List[Sample]:
    """Uses multiprocessing to generate Samples for n_games number of games."""
    logger = logging.getLogger(__name__)

    n_coroutines_per_process = max(n_games // n_processes, 1)
    logger.info(
        f"Starting game generation:\n"
        f"- n_games: {n_games}\n"
        f"- n_processes: {n_processes}\n"
        f"- n_coroutines_per_process: {n_coroutines_per_process}\n"
        f"- max_nn_batch_size: {max_nn_batch_size}\n"
    )

    game_id_queue = mp.Queue()  # Input game_id queue
    for game_id in range(n_games):
        game_id_queue.put(game_id)
    pos_in_queue = mp.Queue()  # Queue for batch nn inference
    pos_out_queues = [mp.Queue() for i in range(n_processes)]  # NN output queues
    sample_out_queue = mp.Queue()  # Queue for output samples
    tqdm_mcts_iters_queue = mp.Queue()  # TQDM stats queue for MCTS iterations

    fn = async_to_sync(mp_sample_generator)
    processes = [
        mp.Process(
            target=fn,
            args=(
                process_id,
                game_id_queue,
                pos_in_queue,
                pos_out_queues[process_id],
                sample_out_queue,
                tqdm_mcts_iters_queue,
                mcts_iterations,
                exploration_constant,
                n_coroutines_per_process,
            ),
        )
        for process_id in range(n_processes)
    ]
    for p in processes:
        p.start()

    inference_thread = threading.Thread(
        target=batch_nn_inference_loop,
        args=(nn, pos_in_queue, pos_out_queues, max_nn_batch_size),
    )
    inference_thread.start()

    stats_thread = threading.Thread(
        target=tqdm_stats_loop,
        args=(tqdm_mcts_iters_queue, n_games * 40 * mcts_iterations),
    )
    stats_thread.start()

    for process in processes:
        process.join()
        process.close()
    print("All processes finished")

    # Poison pill for the inference thread and stats threads
    pos_in_queue.put(None)
    inference_thread.join()
    print("Inference thread finished")
    tqdm_mcts_iters_queue.put(None)
    stats_thread.join()
    print("Stats thread finished")

    samples = []
    while True:
        try:
            sample = sample_out_queue.get(block=False)
            samples.append(sample)
        except queue.Empty:
            break

    # Sort by game_id and then ply
    samples.sort(key=lambda t: (t[0], get_ply(t[1])))
    return samples


def batch_nn_inference_loop(
    nn: ConnectFourNet,
    pos_in_queue: mp.Queue,
    pos_out_queues: List[mp.Queue],
    max_batch_size: int = 2000,
    flush_every_s: int = 0.1,
):
    """
    Loop intended to run on a background thread. Batches nn inference.
    Loop stops when it receives None in the pos_in_queue.
    """
    last_flushed_time_s = time.time()
    accumulated_batch = []

    def flush_batch():
        nonlocal last_flushed_time_s

        if len(accumulated_batch) == 0:
            last_flushed_time_s = time.time()
            return

        # Group identical positions together to avoid duplicate work
        pos_dict = defaultdict(list)
        for process_id, game_id, ply, pos in accumulated_batch:
            pos_dict[pos_to_bytes(pos)].append((process_id, game_id, ply))
        accumulated_batch.clear()

        positions_bytes = list(pos_dict.keys())
        positions = [pos_from_bytes(s) for s in positions_bytes]
        pos_tensor = torch.from_numpy(np.array(positions)).float().to("cuda")
        with torch.no_grad():
            policies, values = nn(pos_tensor)
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        assert len(positions) == len(policies) == len(values)

        for i in range(len(positions)):
            for process_id, game_id, ply in pos_dict[positions_bytes[i]]:
                pos_out_queues[process_id].put((game_id, ply, policies[i], values[i]))
            del pos_dict[positions_bytes[i]]

        last_flushed_time_s = time.time()

    while True:
        try:
            item = pos_in_queue.get(block=False)
            if item is None:  # None is a poison pill for this loop
                print("Batch inference loop terminating")
                return
            accumulated_batch.append(item)
        except queue.Empty:
            elapsed_s = time.time() - last_flushed_time_s
            if elapsed_s > flush_every_s:
                flush_batch()
                continue
            else:
                # If the queue is empy, sleep for a bit to allow items to accumulate
                time.sleep(flush_every_s - elapsed_s)
                continue

        if len(accumulated_batch) >= max_batch_size:
            flush_batch()


def tqdm_stats_loop(tqdm_mcts_iters_queue: mp.Queue, total: int | None) -> None:
    with tqdm(desc="MCTS Iterations", unit="it", total=total) as pbar:
        while True:
            stat = tqdm_mcts_iters_queue.get()
            if stat is None:
                return
            pbar.update(stat)


async def mp_sample_generator(
    process_id: int,
    game_id_queue: mp.Queue,
    pos_in_queue: mp.Queue,
    pos_out_queue: mp.Queue,
    sample_out_queue: mp.Queue,
    tqdm_mcts_iters_queue: mp.Queue,
    mcts_iterations: int,
    exploration_constant: float,
    n_coroutines_per_process: int,
) -> None:
    """Background process that takes game_ids and outputs samples."""

    pos_result_futures: Dict[(GameID, Ply), asyncio.Future] = {}

    async def pos_out_queue_manager():
        """
        Reads results from pos_out_queue and dispatches them to the appropriate futures.
        """

        while True:
            try:
                game_id, ply, policy, value = pos_out_queue.get(block=False)
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue

            future = pos_result_futures[(game_id, ply)]
            future.set_result((policy, value))
            del pos_result_futures[(game_id, ply)]

    async def enqueue_pos(game_id: GameID, pos: Pos) -> Tuple[Policy, Value]:
        """
        Queues a position for batch evaluation whose result will be provided by
        pos_out_queue_manager above.
        """
        ply = get_ply(pos)
        result_future = asyncio.get_running_loop().create_future()
        pos_result_futures[(game_id, ply)] = result_future
        pos_in_queue.put((process_id, game_id, ply, pos))
        return await result_future

    pos_manager_task = asyncio.create_task(pos_out_queue_manager())

    async def game_gen_coro(coro_id: int):
        """Loop that generates games and puts them into sample_out_queue."""
        while True:
            try:
                game_id = game_id_queue.get(block=False)
            except queue.Empty:
                print(f"Coroutine {coro_id} for process {process_id} terminating")
                return

            samples = await gen_game(
                eval=enqueue_pos,
                game_id=game_id,
                mcts_iterations=mcts_iterations,
                exploration_constant=exploration_constant,
                tqdm_mcts_iters_queue=tqdm_mcts_iters_queue,
            )
            for sample in samples:
                sample_out_queue.put(sample)

    await asyncio.gather(*[game_gen_coro(i) for i in range(n_coroutines_per_process)])
    print(f"P{process_id} Game gen coros finished")

    stop_pos_manager = True
    pos_manager_task.cancel()
    print(f"P{process_id} Pos manager task finished")


async def gen_game(
    eval: Callable[[GameID, Pos], Awaitable[Tuple[Policy, Value]]],
    game_id: GameID,
    mcts_iterations: int,
    exploration_constant: float,
    tqdm_mcts_iters_queue: Optional[mp.Queue] = None,
) -> List[Sample]:
    """
    Generates a single game of self-play, appending the results to out_samples.  Asyncio coroutine.
    """

    async def stateless_eval(pos: Pos) -> Tuple[Policy, Value]:
        return await eval(game_id, pos)

    results: List[Tuple[GameID, Pos, Policy]] = []

    # Generate moves until terminal state
    pos = STARTING_POS
    while (res := is_game_over(pos)) is None:
        policy = await mcts(
            pos,
            eval_pos=stateless_eval,
            n_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            tqdm_mcts_iters_queue=tqdm_mcts_iters_queue,
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

    ret = []
    for game_id, pos, policy in results:
        ret.append((game_id, pos, policy, final_value))

        # Alternate the sign of the final value as the board perspective changes between each move
        final_value *= -1
    return ret
