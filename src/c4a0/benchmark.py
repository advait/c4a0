"""Synchronized, opt-in generation measurements; never written to run artifacts."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class GenerationMetrics:
    device: torch.device
    seconds: dict[str, float] = field(default_factory=dict)
    epochs: int = 0
    optimizer_steps: int = 0

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def phase(self, name: str) -> Generator[None]:
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] = (
                self.seconds.get(name, 0.0) + time.perf_counter() - start
            )
