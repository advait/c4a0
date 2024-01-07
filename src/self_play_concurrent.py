"""
Concurrent self-play to generate training data.

Design:
1. Use multiprocessing to take advantage of CPU parallelism
2. Each process runs its own asyncio event loop and runs co-routines to generate samples
3. NN calls are sent to the parent process, executed in batch in a background thread, and
   returned back to the child process via Pipe
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from typing import Awaitable, Callable, Dict, List, NewType, Optional, Tuple

import numpy as np
import torch

from c4 import N_COLS, STARTING_POS, Pos, get_ply, is_game_over, make_move
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
    req_id: ReqID


@dataclass
class Res:
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

    game_id: Optional[int]


@dataclass
class NNReq(Req):
    pos: Pos


@dataclass
class NNRes(Res):
    policy: Policy
    value: Value


@dataclass
class SubmitGameReq(Req):
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
    n_processes: int = mp.cpu_count() * 2,
    n_coroutines_per_process: int = 100,
    max_nn_batch_size: int = 20000,
) -> List[Sample]:
    pipes: List[mp.Pipe] = []
    workers: List[mp.Process] = []
    n_alive_workers = n_processes
    model_bg_thread = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="nn_bg_thread"
    )
    samples: List[Sample] = []
    cur_game_id: int = 0

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

    continue_polling_pipes = True

    async def poll_pipes():
        while continue_polling_pipes:
            for pipe in pipes:
                if pipe.poll():
                    msg = pipe.recv()
                    # Note we technically don't explicitly block on this task, but it's fine
                    # because we're manually waiting for worker termination via WorkerExitSignalReq
                    asyncio.create_task(handle_req(msg, pipe))
            await asyncio.sleep(0)

    async def handle_req(req: Req, pipe: mp.Pipe) -> None:
        nonlocal n_alive_workers, continue_polling_pipes, n_games, cur_game_id
        if isinstance(req, WorkerExitSignalReq):
            n_alive_workers -= 1
            if n_alive_workers == 0:
                continue_polling_pipes = False

        elif isinstance(req, GetGameIDReq):
            if cur_game_id >= n_games:
                pipe.send(GetGameIDRes(req_id=req.req_id, game_id=None))
            else:
                pipe.send(GetGameIDRes(req_id=req.req_id, game_id=cur_game_id))
                cur_game_id += 1

        elif isinstance(req, NNReq):
            # TODO: Handle batching positions
            pos_tensor = torch.from_numpy(req.pos).unsqueeze(0).float().cuda()
            policy, value = await asyncio.get_running_loop().run_in_executor(
                model_bg_thread, run_model_no_grad, model, pos_tensor
            )
            policy = policy.squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().item()
            pipe.send(NNRes(req_id=req.req_id, policy=policy, value=value))

        elif isinstance(req, SubmitGameReq):
            samples.extend(req.samples)

    await poll_pipes()
    for worker in workers:
        worker.join()

    # Sort by game_id and then ply
    samples.sort(key=lambda t: (t[0], get_ply(t[1])))
    return samples


def worker_process(
    worker_id: int,
    pipe: mp.Pipe,
    n_coroutines_per_process: int,
    mcts_iterations: int,
    exploration_constant: float,
) -> None:
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

    async def poll_pipe():
        nonlocal continue_polling_pipes
        while continue_polling_pipes:
            if pipe.poll():
                msg = pipe.recv()
                try:
                    future = pending_reqs[msg.req_id]
                except KeyError:
                    print("wtf")
                del pending_reqs[msg.req_id]
                future.set_result((msg))
            else:
                await asyncio.sleep(0)

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
                    mcts_iterations=mcts_iterations,
                    exploration_constant=exploration_constant,
                )
                for i in range(n_coroutines_per_process)
            )
        )
        continue_polling_pipes = False

    asyncio.run(create_worker_coros())
    pipe.send(WorkerExitSignalReq(req_id=create_req_id(), worker_id=worker_id))
    print(f"{worker_id} worker process exiting (check adjacent resources)")


async def worker_coro(
    worker_id: int,
    coro_id: int,
    get_game_id: Callable[[], Awaitable[Optional[GameID]]],
    eval_pos: Callable[[Pos], Awaitable[Tuple[Policy, Value]]],
    submit_game: Callable[[List[Sample]], Awaitable[None]],
    mcts_iterations: int,
    exploration_constant: float,
) -> None:
    while True:
        game_id = await get_game_id()
        if game_id is None:
            print(f"{worker_id}.{coro_id} worker coro exiting (no more games)")
            return
        game = await gen_game(game_id, eval_pos, mcts_iterations, exploration_constant)
        await submit_game(game)


async def gen_game(
    game_id: GameID,
    eval_pos: Callable[[Pos], Awaitable[Tuple[Policy, Value]]],
    mcts_iterations: int,
    exploration_constant: float,
) -> List[Tuple[GameID, Pos, Policy, Value]]:
    pos = STARTING_POS
    results: List[Tuple[GameID, Pos, Policy]] = []
    while (res := is_game_over(pos)) is None:
        policy = await mcts(
            pos,
            eval_pos=eval_pos,
            n_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            tqdm_mcts_iters_queue=None,
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


async def set_future(future: asyncio.Future, result: object) -> None:
    future.set_result(result)


def create_req_id() -> ReqID:
    # Use a better req_id generator than rand int
    return ReqID(np.random.randint(0, 1e12))


def run_model_no_grad(model, x):
    with torch.no_grad():
        return model(x)
