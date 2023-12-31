"""
Concurrent self-play to generate training data.

- Uses multiprocessing to take advantage of CPU parallelism
- generate_samples() is the main entrypoint. It spawns worker_process() processes.
- Uses multiprocessing.Pipe as IPC between the workers and the parent.
- worker_process() is the entrypoint for each worker process. It spawns worker_coro() coroutines.
- See the various Req and Res subclasses to understand IPC.
- We batch many NNReq requests together before submitting to the GPU.
"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import sys
from textwrap import dedent
import time
from typing import Awaitable, Callable, Dict, List, NewType, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from c4 import (
    N_COLS,
    STARTING_POS,
    Pos,
    get_ply,
    is_game_over,
    make_move,
    pos_from_bytes,
    pos_to_bytes,
)
from mcts import mcts
from nn import ConnectFourNet, Policy, Value

ReqID = NewType("ReqID", int)

GameID = NewType("GameID", int)
"""Which game this sample is from."""

Sample = NewType("Sample", Tuple[GameID, Pos, Policy, Value])
"""A sample of training data representing a position and the desired policy and value outputs."""

ProcessID = NewType("ProcessID", int)


@dataclass
class Req:
    """A request sent from the child process to the parent process."""

    req_id: ReqID


@dataclass
class Res:
    """A response from the parent process to the child process."""

    req_id: ReqID


@dataclass
class GetGameIDReq(Req):
    """Request a new game ID to be generated."""


@dataclass
class GetGameIDRes(Res):
    """
    Response to GetGameIDReq. If game_id is None, then no more game IDs are available, signaling to
    the child process that it should exit.
    """

    game_id: Optional[GameID]


@dataclass
class NNReq(Req):
    """Request to run NN inference on a position."""

    pos: Pos


@dataclass
class NNRes(Res):
    """Response to NNReq."""

    policy: Policy
    value: Value


@dataclass
class SubmitMCTSIter(Req):
    """Indicates to the parent that an MCTS iteration was complete (for tqdm stats)."""


@dataclass
class SubmitGameReq(Req):
    """Submits a completed game to the parent process. No response is needed."""

    samples: List[Sample]


@dataclass
class WorkerExitSignalReq(Req):
    """Signal from the worker indicating that it has exited. No responses is needed."""

    worker_id: ProcessID


async def generate_samples(
    model: ConnectFourNet,
    n_games: int,
    mcts_iterations: int,
    exploration_constant: float,
    n_processes: int = mp.cpu_count() - 1,
    nn_flush_freq_s: float = 0.01,
    nn_max_batch_size: int = 20000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[Sample]:
    """
    Uses multiprocessing to generate n_games worth of training data.
    """

    logger = logging.getLogger(__name__)

    n_processes = min(n_processes, n_games)
    n_coroutines_per_process = max(n_games // n_processes, 1)

    logger.info(
        dedent(
            f"""
            Beginning multi-process self-play generation:
            - n_games: {n_games}
            - n_processes: {n_processes}
            - n_coroutines_per_process: {n_coroutines_per_process}
            - mcts_iterations: {mcts_iterations}
            - exploration_constant: {exploration_constant}
            - nn_flush_freq_s: {nn_flush_freq_s}
            - nn_max_batch_size: {nn_max_batch_size}
            """
        ).strip()
    )

    pipes: List[Optional[Connection]] = []
    workers: List[mp.Process] = []
    n_alive_workers = n_processes
    generated_samples: List[Sample] = []
    cur_game_id: GameID = GameID(0)

    nn_pos_queue: List[Tuple[Pos, Connection, ReqID]] = []
    nn_bg_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nn_bg_thread")
    nn_last_flush_s = time.time()

    pbar = tqdm(
        total=n_games * 30 * mcts_iterations, desc="MCTS iterations", unit="iter"
    )

    for worker_id in range(n_processes):
        parent_conn, child_conn = mp.Pipe()
        pipes.append(parent_conn)
        worker = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                child_conn,
                n_coroutines_per_process,
                mcts_iterations,
                exploration_constant,
            ),
        )
        workers.append(worker)
        worker.start()

    continue_server_loops = True
    # Key flag that, when set to False, tells the poll_worker_pipes_loop and nn_flush_loop to exit

    async def poll_worker_pipes_loop():
        while continue_server_loops:
            if all(pipe is None for pipe in pipes):
                return

            for process_id, pipe in enumerate(pipes):
                msg_handled = False
                if pipe is None:
                    continue
                if pipe.poll():
                    try:
                        msg = pipe.recv()
                        handle_req(msg, pipe)
                        msg_handled = True
                    except EOFError:
                        # Pipe closed via terminated process
                        pipes[process_id] = None
                if msg_handled:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.01)

    def handle_req(req: Req, pipe: Connection) -> None:
        nonlocal n_alive_workers, continue_server_loops, n_games, cur_game_id
        if isinstance(req, WorkerExitSignalReq):
            n_alive_workers -= 1
            if n_alive_workers == 0:
                continue_server_loops = False

        elif isinstance(req, GetGameIDReq):
            if cur_game_id >= n_games:
                pipe.send(GetGameIDRes(req_id=req.req_id, game_id=None))
            else:
                pipe.send(GetGameIDRes(req_id=req.req_id, game_id=cur_game_id))
                cur_game_id = GameID(cur_game_id + 1)

        elif isinstance(req, NNReq):
            nn_pos_queue.append((req.pos, pipe, req.req_id))

        elif isinstance(req, SubmitGameReq):
            generated_samples.extend(req.samples)

        elif isinstance(req, SubmitMCTSIter):
            pbar.update(1)

    async def nn_flush_loop():
        nonlocal nn_last_flush_s
        while continue_server_loops:
            elapsed = time.time() - nn_last_flush_s
            if elapsed >= nn_flush_freq_s or len(nn_pos_queue) >= nn_max_batch_size:
                await nn_flush()
            await asyncio.sleep(0)

    async def nn_flush():
        nonlocal nn_last_flush_s, nn_pos_queue

        if len(nn_pos_queue) == 0:
            # If there are no positions to evaluate, exit early and reset the timer
            nn_last_flush_s = time.time()
            await asyncio.sleep(0.1)
            return

        # Group identical positions together to avoid duplicate work
        pos_dict = defaultdict(list)
        for pos, pipe, req_id in nn_pos_queue:
            pos_dict[pos_to_bytes(pos)].append((pipe, req_id))
        nn_pos_queue.clear()

        positions_bytes = list(pos_dict.keys())
        positions = [pos_from_bytes(s) for s in positions_bytes]
        pos_tensor = torch.from_numpy(np.array(positions)).float().to(device)
        policies, values = await asyncio.get_running_loop().run_in_executor(
            nn_bg_thread, run_model_no_grad, model, pos_tensor
        )
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        assert len(positions) == len(policies) == len(values)

        for i in range(len(positions)):
            for pipe, req_id in pos_dict[positions_bytes[i]]:
                pipe.send(NNRes(req_id=req_id, policy=policies[i], value=values[i]))
            del pos_dict[positions_bytes[i]]

        nn_last_flush_s = time.time()

    nn_flush_loop_task = asyncio.create_task(nn_flush_loop())
    await poll_worker_pipes_loop()
    await nn_flush_loop_task
    pbar.close()

    for worker in workers:
        worker.join()

    # Sort by game_id and then ply
    generated_samples.sort(key=lambda t: (t[0], get_ply(t[1])))
    return generated_samples


def worker_process(
    worker_id: ProcessID,
    pipe: Connection,
    n_coroutines_per_process: int,
    mcts_iterations: int,
    exploration_constant: float,
) -> None:
    logger = logging.getLogger(__name__)
    pending_reqs: Dict[ReqID, asyncio.Future] = {}
    continue_polling_pipes = True

    async def get_game_id() -> Optional[GameID]:
        nonlocal pending_reqs
        future = asyncio.get_running_loop().create_future()
        req_id = create_req_id()
        pending_reqs[req_id] = future
        pipe.send(GetGameIDReq(req_id=req_id))
        res: GetGameIDRes = await future
        return res.game_id

    async def eval_pos(pos: Pos) -> Tuple[Policy, Value]:
        nonlocal pending_reqs
        future = asyncio.get_running_loop().create_future()
        req_id = create_req_id()
        pending_reqs[req_id] = future
        pipe.send(NNReq(req_id=req_id, pos=pos))
        res: NNRes = await future
        return res.policy, res.value

    async def submit_game(samples: List[Sample]) -> None:
        pipe.send(SubmitGameReq(req_id=create_req_id(), samples=samples))

    async def submit_mcts_iter() -> None:
        pipe.send(SubmitMCTSIter(req_id=create_req_id()))

    async def poll_pipe():
        nonlocal continue_polling_pipes
        while continue_polling_pipes:
            if pipe.poll():
                msg = pipe.recv()
                future = pending_reqs[msg.req_id]
                del pending_reqs[msg.req_id]
                future.set_result((msg))
            elif len(pending_reqs) > 0:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.1)

    async def create_worker_coros():
        nonlocal continue_polling_pipes
        asyncio.create_task(poll_pipe())
        await asyncio.gather(
            *(
                worker_coro(
                    worker_id=worker_id,
                    coro_id=i,
                    get_game_id=get_game_id,
                    eval_pos=eval_pos,
                    submit_game=submit_game,
                    submit_mcts_iter=submit_mcts_iter,
                    mcts_iterations=mcts_iterations,
                    exploration_constant=exploration_constant,
                )
                for i in range(n_coroutines_per_process)
            )
        )
        continue_polling_pipes = False

    asyncio.run(create_worker_coros())
    pipe.send(WorkerExitSignalReq(req_id=create_req_id(), worker_id=worker_id))
    logger.info(f"{worker_id} Worker process exiting")


async def worker_coro(
    worker_id: int,
    coro_id: int,
    get_game_id: Callable[[], Awaitable[Optional[GameID]]],
    eval_pos: Callable[[Pos], Awaitable[Tuple[Policy, Value]]],
    submit_game: Callable[[List[Sample]], Awaitable[None]],
    submit_mcts_iter: Callable[[], Awaitable[None]],
    mcts_iterations: int,
    exploration_constant: float,
) -> None:
    logger = logging.getLogger(__name__)
    while True:
        game_id = await get_game_id()
        if game_id is None:
            logger.info(f"{worker_id}.{coro_id} Worker coro exiting")
            return
        game = await gen_game(
            game_id=game_id,
            eval_pos=eval_pos,
            mcts_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            submit_mcts_iter=submit_mcts_iter,
        )
        await submit_game(game)


async def gen_game(
    game_id: GameID,
    eval_pos: Callable[[Pos], Awaitable[Tuple[Policy, Value]]],
    mcts_iterations: int,
    exploration_constant: float,
    submit_mcts_iter: Optional[Callable[[], Awaitable[None]]],
) -> List[Sample]:
    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(seed=game_id)

    pos = STARTING_POS
    results: List[Tuple[GameID, Pos, Policy]] = []
    while (res := is_game_over(pos)) is None:
        policy = await mcts(
            pos,
            eval_pos=eval_pos,
            n_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            submit_mcts_iter=submit_mcts_iter,
        )
        results.append((game_id, pos, policy))
        move = rng.choice(range(len(policy)), p=policy)
        logger.info(f"{game_id}.{get_ply(pos)} move: {move}, policy: {policy}")
        pos = make_move(pos, move)

    # Because there isn't a valid policy a terminal state, we simply use a uniform policy
    final_policy = Policy(np.ones(N_COLS) / N_COLS)
    results.append((game_id, pos, final_policy))
    # logger.info(f"{game_id}.{get_ply(pos)} finished result: {res.value}")

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


def create_req_id() -> ReqID:
    # TODO: Use a better req_id generator than rand int
    return ReqID(np.random.randint(0, sys.maxsize))


def run_model_no_grad(model, x):
    with torch.no_grad():
        return model(x)
