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
from dataclasses import dataclass
import logging
import math
import multiprocessing as mp
from multiprocessing.connection import Connection
import sys
from textwrap import dedent
from typing import Awaitable, Callable, Dict, List, NewType, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from c4a0.c4 import (
    N_COLS,
    STARTING_POS,
    Pos,
    get_ply,
    is_game_over,
    make_move,
)
from c4a0.mcts import mcts
from c4a0.nn import ConnectFourNet, EvalPos, Policy, Value, create_batcher

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
    n_processes: int,
    mcts_iterations: int,
    exploration_constant: float,
    max_coros_per_process: int = 1000,
    nn_flush_freq_s: float = 0.01,
    nn_max_batch_size: int = 20000,
    device: torch.device = torch.device("cpu"),
) -> List[Sample]:
    """
    Uses multiprocessing to generate n_games worth of training data.
    """

    logger = logging.getLogger(__name__)

    n_processes = min(n_processes, n_games)
    n_coroutines_per_process = math.ceil(n_games / n_processes)
    n_coroutines_per_process = min(n_coroutines_per_process, max_coros_per_process)

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
            - device: {device}
            """
        ).strip()
    )

    pipes: List[Optional[Connection]] = []
    workers: List[mp.Process] = []
    n_alive_workers = n_processes
    generated_samples: List[Sample] = []
    cur_game_id: GameID = GameID(0)

    nn_end_signal, nn_enqueue_pos = create_batcher(
        model,
        device,
        nn_flush_freq_s=nn_flush_freq_s,
        nn_max_batch_size=nn_max_batch_size,
    )

    approx_mcts_iters = n_games * 21 * mcts_iterations  # approx ~21 ply per game
    mcts_pbar = tqdm(total=approx_mcts_iters, desc="mcts iterations", unit="it")
    games_pbar = tqdm(total=n_games, desc="games generated", unit="gm")

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

    # Key flag that, when set to False, tells the poll_worker_pipes_loop and nn_flush_loop to exit
    continue_server_loops = True

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

            async def run_inference():
                policy, value = await nn_enqueue_pos(req.pos)
                pipe.send(NNRes(req_id=req.req_id, policy=policy, value=value))

            asyncio.create_task(run_inference())

        elif isinstance(req, SubmitGameReq):
            generated_samples.extend(req.samples)
            games_pbar.update(1)

        elif isinstance(req, SubmitMCTSIter):
            mcts_pbar.update(1)

    await poll_worker_pipes_loop()
    nn_end_signal.set_result(True)
    mcts_pbar.close()
    games_pbar.close()

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

    def submit_mcts_iter() -> None:
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
    logger.debug(f"{worker_id} Worker process exiting")


async def worker_coro(
    worker_id: int,
    coro_id: int,
    get_game_id: Callable[[], Awaitable[Optional[GameID]]],
    eval_pos: Callable[[Pos], Awaitable[Tuple[Policy, Value]]],
    submit_game: Callable[[List[Sample]], Awaitable[None]],
    submit_mcts_iter: Callable,
    mcts_iterations: int,
    exploration_constant: float,
) -> None:
    logger = logging.getLogger(__name__)
    while True:
        game_id = await get_game_id()
        if game_id is None:
            logger.debug(f"{worker_id}.{coro_id} Worker coro exiting")
            return
        game = await gen_game(
            game_id=game_id,
            eval_pos0=eval_pos,
            eval_pos1=None,
            mcts_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            submit_mcts_iter=submit_mcts_iter,
        )
        await submit_game(game)


async def gen_game(
    game_id: GameID,
    eval_pos0: EvalPos,
    eval_pos1: Optional[EvalPos],
    mcts_iterations: int,
    exploration_constant: float,
    submit_mcts_iter: Optional[Callable],
) -> List[Sample]:
    """
    Generates a single game using MCTS.

    Supports different players by setting different values for eval_pos0 and eval_pos1.
    """
    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(seed=game_id)
    if eval_pos1 is None:
        eval_pos1 = eval_pos0

    pos = STARTING_POS
    results: List[Tuple[GameID, Pos, Policy]] = []
    while (res := is_game_over(pos)) is None:
        eval_pos = eval_pos0 if get_ply(pos) % 2 == 0 else eval_pos1
        policy = await mcts(
            pos,
            eval_pos=eval_pos,
            n_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            submit_mcts_iter=submit_mcts_iter,
        )
        results.append((game_id, pos, policy))
        move = rng.choice(range(len(policy)), p=policy)
        logger.debug(f"{game_id}.{get_ply(pos)} move: {move}, policy: {policy}")
        pos = make_move(pos, move)

    # Because there isn't a valid policy a terminal state, we simply use a uniform policy
    final_policy = Policy(np.ones(N_COLS) / N_COLS)
    results.append((game_id, pos, final_policy))
    logger.debug(f"{game_id}.{get_ply(pos)} finished result: {res.value}")

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
