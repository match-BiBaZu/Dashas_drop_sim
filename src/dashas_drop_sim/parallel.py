"""Independent MuJoCo worlds in worker processes for repeatable CPU parallelism."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import multiprocessing
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np

from .config import RunConfig
from .geometry import PreparedPart
from .physics import ChuteSimulator, TrialOutcome


_SIMULATOR: ChuteSimulator | None = None
_STOP_EVENT = None
Job = TypeVar("Job")


def _initialize(config: RunConfig, part: PreparedPart, stop_event) -> None:
    global _SIMULATOR, _STOP_EVENT
    _SIMULATOR = ChuteSimulator(config, part)
    _STOP_EVENT = stop_event


def drop_job(job: tuple[int, int]) -> TrialOutcome:
    if _SIMULATOR is None or _STOP_EVENT is None:
        raise RuntimeError("Simulation worker was not initialized")
    index, seed = job
    return _SIMULATOR.drop(index, seed, cancel=_STOP_EVENT.is_set)


def kick_job(job: tuple[int, str, int, tuple[float, ...], float, int, tuple[float, float, float]]) -> TrialOutcome:
    if _SIMULATOR is None or _STOP_EVENT is None:
        raise RuntimeError("Simulation worker was not initialized")
    _ordinal, _pose_key, source_trial, source_qpos, level, _direction_index, direction = job
    return _SIMULATOR.kick(source_qpos, level, np.asarray(direction),
                           index=source_trial, cancel=_STOP_EVENT.is_set)


class SimulationPool:
    """Bounded task queue and cooperative cancellation across spawned worlds."""

    def __init__(self, config: RunConfig, part: PreparedPart):
        self.workers = config.workers
        context = multiprocessing.get_context("spawn")
        self.stop_event = context.Event()
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialize,
            initargs=(config, part, self.stop_event),
        )

    def __enter__(self) -> "SimulationPool":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if _exc_type is not None:
            self.stop_event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def run(
        self,
        jobs: Iterable[Job],
        function: Callable[[Job], TrialOutcome],
        cancel: Callable[[], bool],
    ) -> Iterator[tuple[Job, TrialOutcome]]:
        source = iter(jobs)
        pending: dict[Future[TrialOutcome], Job] = {}

        def fill_queue() -> None:
            while len(pending) < 2 * self.workers and not self.stop_event.is_set():
                if cancel():
                    self.stop_event.set()
                    return
                try:
                    job = next(source)
                except StopIteration:
                    return
                pending[self.executor.submit(function, job)] = job

        fill_queue()
        while pending:
            if cancel():
                self.stop_event.set()
            done, _ = wait(tuple(pending), timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                if future.cancelled():
                    continue
                try:
                    outcome = future.result()
                except BaseException:
                    self.stop_event.set()
                    raise
                if outcome.status != "cancelled":
                    yield job, outcome
            if self.stop_event.is_set():
                for future in pending:
                    future.cancel()
            else:
                fill_queue()
