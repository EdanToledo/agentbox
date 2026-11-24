# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FIFO scheduling on a worker-local asyncio runtime thread."""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cloudpickle
from agentbox.tasks.registry import TaskHandler


logger = logging.getLogger(__name__)

type _TaskOutcome = tuple[str, Any]


@dataclass
class _TaskWork:
    task_id: str
    kind: str
    payload: tuple[Any, ...]
    enqueued_at: float


@dataclass(frozen=True)
class _TaskAccepted:
    task_id: str


@dataclass(frozen=True)
class _TaskRejected:
    task_id: str
    error: BaseException


@dataclass(frozen=True)
class _TaskStarted:
    task_id: str


@dataclass(frozen=True)
class _TaskNeverStarted:
    task_id: str


@dataclass(frozen=True)
class _TaskCompletion:
    task_id: str
    outcome: _TaskOutcome


type _SchedulerEvent = (
    _TaskAccepted | _TaskRejected | _TaskStarted | _TaskNeverStarted | _TaskCompletion
)


@dataclass(frozen=True)
class _CancelResult:
    operation_id: str
    found: bool


type _RuntimeEvent = _SchedulerEvent | _CancelResult


class _FifoTaskScheduler:
    """Schedule bounded coroutine work on the current asyncio loop.

    This class owns no thread and performs no transport. Its only job is FIFO
    admission, bounded concurrency, cancellation, and draining. All methods
    must be called from the event-loop thread on which it was constructed.
    """

    def __init__(
        self,
        capacity: int,
        handlers: dict[str, TaskHandler],
        on_event: Callable[[_SchedulerEvent], None],
        on_drained: Callable[[], None],
    ) -> None:
        if capacity < 1:
            raise ValueError("task pool size must be positive")
        if not handlers:
            raise ValueError("task pool requires at least one handler")
        self.capacity = capacity
        self._handlers = handlers.copy()
        self._on_event = on_event
        self._on_drained = on_drained
        self._pending: deque[_TaskWork] = deque()
        self._active: dict[str, asyncio.Task[Any]] = {}
        self._started_at: dict[str, float] = {}
        self._accepting = True
        self._shutdown_requested = False
        self._drained_notified = False

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    @property
    def accepting(self) -> bool:
        return self._accepting

    def submit(self, work: _TaskWork) -> None:
        if not self._accepting:
            self._on_event(
                _TaskRejected(
                    work.task_id,
                    RuntimeError("task pool is shutting down"),
                )
            )
            return
        if work.kind not in self._handlers:
            self._on_event(
                _TaskRejected(
                    work.task_id,
                    RuntimeError(f"unknown task kind: {work.kind}"),
                )
            )
            return
        self._on_event(_TaskAccepted(work.task_id))
        if len(self._active) < self.capacity:
            logger.debug(
                "Task accepted task_id=%s kind=%s active=%d queued=%d",
                work.task_id,
                work.kind,
                len(self._active),
                len(self._pending),
            )
            self._start(work)
            return
        self._pending.append(work)
        logger.info(
            "Task queued task_id=%s kind=%s queue_depth=%d",
            work.task_id,
            work.kind,
            len(self._pending),
        )

    def _start(self, work: _TaskWork) -> None:
        handler = self._handlers[work.kind]
        task = asyncio.create_task(handler(*work.payload))
        self._active[work.task_id] = task
        self._started_at[work.task_id] = time.monotonic()
        self._on_event(_TaskStarted(work.task_id))
        logger.debug(
            "Task started task_id=%s kind=%s wait_seconds=%.6f active=%d queued=%d",
            work.task_id,
            work.kind,
            self._started_at[work.task_id] - work.enqueued_at,
            len(self._active),
            len(self._pending),
        )

        def finish(completed: asyncio.Task[Any]) -> None:
            self._finish(work, completed)

        task.add_done_callback(finish)

    def _finish(self, work: _TaskWork, task: asyncio.Task[Any]) -> None:
        self._active.pop(work.task_id, None)
        started_at = self._started_at.pop(work.task_id, work.enqueued_at)
        if task.cancelled():
            outcome: _TaskOutcome = ("cancelled", None)
            logger.debug("Task cancelled task_id=%s kind=%s", work.task_id, work.kind)
        else:
            error = task.exception()
            if error is not None:
                outcome = ("error", error)
                logger.debug(
                    "Task failed task_id=%s kind=%s",
                    work.task_id,
                    work.kind,
                    exc_info=(type(error), error, error.__traceback__),
                )
            else:
                outcome = ("result", task.result())
                logger.debug(
                    "Task completed task_id=%s kind=%s run_seconds=%.6f",
                    work.task_id,
                    work.kind,
                    time.monotonic() - started_at,
                )
        self._complete(work, outcome)
        self._fill_capacity()
        self._maybe_stop()

    def _fill_capacity(self) -> None:
        while self._pending and len(self._active) < self.capacity:
            self._start(self._pending.popleft())

    def cancel(self, task_id: str) -> bool:
        active = self._active.get(task_id)
        if active is not None:
            return active.cancel()
        retained: deque[_TaskWork] = deque()
        cancelled: _TaskWork | None = None
        while self._pending:
            work = self._pending.popleft()
            if cancelled is None and work.task_id == task_id:
                cancelled = work
            else:
                retained.append(work)
        self._pending = retained
        if cancelled is not None:
            logger.debug(
                "Queued task cancelled task_id=%s kind=%s",
                cancelled.task_id,
                cancelled.kind,
            )
            self._on_event(_TaskNeverStarted(cancelled.task_id))
            self._complete(cancelled, ("cancelled", None))
        self._maybe_stop()
        return cancelled is not None

    def shutdown(self, drain: bool = True) -> None:
        if self._shutdown_requested:
            return
        self._accepting = False
        self._shutdown_requested = True
        logger.debug(
            "Task pool shutdown requested drain=%s active=%d queued=%d",
            drain,
            len(self._active),
            len(self._pending),
        )
        if not drain:
            while self._pending:
                work = self._pending.popleft()
                self._on_event(_TaskNeverStarted(work.task_id))
                self._complete(work, ("cancelled", None))
            for task in tuple(self._active.values()):
                task.cancel()
        self._maybe_stop()

    def _maybe_stop(self) -> None:
        if (
            self._shutdown_requested
            and not self._active
            and not self._pending
            and not self._drained_notified
        ):
            self._drained_notified = True
            self._on_drained()

    def _complete(self, work: _TaskWork, outcome: _TaskOutcome) -> None:
        self._on_event(_TaskCompletion(task_id=work.task_id, outcome=outcome))


class _EventSink(Protocol):
    """Receiver of runtime events; ``queue.SimpleQueue`` satisfies it."""

    def put(self, event: _RuntimeEvent, /) -> None: ...


class _AsyncRuntimeThread:
    """Own one asyncio event-loop thread for a :class:`_FifoTaskScheduler`."""

    def __init__(
        self,
        capacity: int,
        handlers: dict[str, TaskHandler],
        events: _EventSink | None = None,
    ) -> None:
        self.capacity = capacity
        self._handlers = handlers.copy()
        self.events: _EventSink = events or queue.SimpleQueue()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._scheduler: _FifoTaskScheduler | None = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="agentbox-task-pool",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("task pool event loop failed to start") from (
                self._startup_error
            )

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def _thread_main(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._scheduler = _FifoTaskScheduler(
                self.capacity,
                self._handlers,
                self._record_event,
                loop.stop,
            )
            self._ready.set()
            logger.debug("Task pool started with capacity=%d", self.capacity)
            loop.run_forever()
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            logger.exception("Task pool event loop failed")
        finally:
            self._stopped.set()
            logger.debug("Task pool stopped")

    def _call_soon(self, callback: Callable[..., None], *args: Any) -> None:
        loop = self._loop
        if loop is None or self.stopped:
            raise RuntimeError("task pool is stopped")
        loop.call_soon_threadsafe(callback, *args)

    def submit(
        self,
        task_id: str,
        kind: str,
        payload: tuple[Any, ...],
    ) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            raise RuntimeError("task pool is not ready")

        def submit_work() -> None:
            scheduler.submit(_TaskWork(task_id, kind, payload, time.monotonic()))

        self._call_soon(submit_work)

    def cancel(self, task_id: str, operation_id: str | None) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            raise RuntimeError("task pool is not ready")

        def cancel_work() -> None:
            cancelled = scheduler.cancel(task_id)
            if operation_id is not None:
                self.events.put(_CancelResult(operation_id, cancelled))

        self._call_soon(cancel_work)

    def _record_event(self, event: _SchedulerEvent) -> None:
        self.events.put(event)

    def shutdown(self, *, drain: bool = True) -> None:
        if self.stopped:
            return
        loop = self._loop
        scheduler = self._scheduler
        if loop is None or scheduler is None:
            raise RuntimeError("task pool is not ready")
        try:
            loop.call_soon_threadsafe(scheduler.shutdown, drain)
        except RuntimeError:
            if not self.stopped:
                raise

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()


cloudpickle.register_pickle_by_value(sys.modules[__name__])
