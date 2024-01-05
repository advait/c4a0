import asyncio
import concurrent.futures
from typing import List, Tuple, TypeVarTuple

import torch

Ts = TypeVarTuple("Ts")
Ls = TypeVarTuple("Ls")


def unzip(tuples: List[Tuple[*Ts]]) -> Tuple[*Ls]:
    """Unzips a list of tuples into a tuple of lists."""
    if len(tuples) == 0:
        return ((),)

    ret = tuple([] for _ in range(len(tuples[0])))
    for t in tuples:
        for i, item in enumerate(t):
            ret[i].append(item)

    return ret


bg_thread_executor = concurrent.futures.ThreadPoolExecutor()


async def model_forward_bg_thread(model, x):
    """
    Runs a forward pass on the model (in a no_grad context) on a background thread with an async
    interface.
    """

    def run_model_no_grad(x_):
        with torch.no_grad():
            return model(x_)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(bg_thread_executor, run_model_no_grad, x)


def async_to_sync(func):
    """Wraps an asyncio function to produce a synchronous/blocking version"""

    def wrapper(*args):
        return asyncio.run(func(*args))

    return wrapper
