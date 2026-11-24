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

"""Monarch transport and client implementation for worker-local task pools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass, fields
from typing import Any, cast, TYPE_CHECKING

import cloudpickle
from agentbox.tasks.api import Task, TaskHandle
from agentbox.tasks.registry import TaskHandler
from agentbox.tasks.scheduler import (
    _AsyncRuntimeThread,
    _CancelResult,
    _RuntimeEvent,
    _TaskAccepted,
    _TaskCompletion,
    _TaskNeverStarted,
    _TaskOutcome,
    _TaskRejected,
    _TaskStarted,
)
from monarch.actor import Actor, Channel, endpoint, Port, ProcMesh


if TYPE_CHECKING:
    from monarch._src.actor.actor_mesh import ActorMesh


logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.01
_STOP_TIMEOUT = 10.0


@dataclass(frozen=True)
class _ScheduleTask:
    task_id: str
    kind: str
    payload: tuple[Any, ...]


@dataclass(frozen=True)
class _CancelTask:
    task_id: str
    operation_id: str | None


@dataclass(frozen=True)
class _Shutdown:
    """Stop accepting tasks; *drain* waits for running ones, else cancels them."""

    drain: bool


type _TaskCommand = _ScheduleTask | _CancelTask | _Shutdown


@dataclass
class _TaskPoolReady:
    requests: Port[_TaskCommand]


@dataclass
class _TaskPoolClosed:
    pass


type _TaskPoolEvent = _TaskPoolReady | _TaskPoolClosed | _RuntimeEvent


# Wire format. Monarch pickles port messages with cloudpickle, and this module
# and the scheduler are pickled by value, so sending their dataclasses would
# serialize the class definitions with every message (~5 KB and ~100 us to
# pickle each; a command costs four messages). Per-task messages therefore
# cross the wire as builtin tuples tagged with their type's index.
_EVENT_TYPES: tuple[type, ...] = (
    _TaskAccepted,
    _TaskRejected,
    _TaskStarted,
    _TaskNeverStarted,
    _TaskCompletion,
    _CancelResult,
)
_COMMAND_TYPES: tuple[type, ...] = (_ScheduleTask, _CancelTask)
_FIELDS = {
    cls: tuple(f.name for f in fields(cls)) for cls in _EVENT_TYPES + _COMMAND_TYPES
}


def _encode(message: Any, types: tuple[type, ...]) -> Any:
    cls = type(message)
    if cls not in types:
        return message  # rare messages (ready/closed, shutdown) travel as-is
    return (types.index(cls), *(getattr(message, name) for name in _FIELDS[cls]))


def _decode(message: Any, types: tuple[type, ...]) -> Any:
    """Rebuild an encoded message; anything malformed is returned unchanged."""
    if not (isinstance(message, tuple) and message and type(message[0]) is int):
        return message
    tag = message[0]
    if not 0 <= tag < len(types) or len(message) - 1 != len(_FIELDS[types[tag]]):
        return message
    return types[tag](*message[1:])


class _PortEventSink:
    """Queue-shaped adapter that forwards runtime events straight to a port.

    ``put`` runs inside the runtime's asyncio callbacks, where an exception
    would only be logged; the first send failure is kept instead so ``serve``
    can fail the service, as it did when it sent the events itself.
    """

    def __init__(self, responses: Port[_TaskPoolEvent]) -> None:
        self._responses = responses
        self.error: BaseException | None = None

    def put(self, event: _RuntimeEvent) -> None:
        try:
            self._responses.send(_encode(event, _EVENT_TYPES))
        except Exception as exc:
            if self.error is None:
                self.error = exc


class _TaskPoolActor(Actor):
    """Bridge Monarch task requests to one worker-local asyncio thread."""

    def __init__(self, capacity: int, handlers: dict[str, TaskHandler]) -> None:
        self._capacity = capacity
        self._handlers = handlers

    @endpoint
    def serve(self, responses: Port[_TaskPoolEvent]) -> None:
        request_port, requests = Channel[_TaskCommand].open()
        responses.send(_TaskPoolReady(request_port))
        # The runtime thread is the only event producer, so sending from it
        # keeps events ordered and delivers completions without waiting for
        # this loop's next poll.
        events = _PortEventSink(responses)
        runtime = _AsyncRuntimeThread(self._capacity, self._handlers, events)
        receive = requests.recv()
        shutting_down = False
        try:
            while True:
                if events.error is not None:
                    raise events.error
                if shutting_down and runtime.stopped:
                    responses.send(_TaskPoolClosed())
                    return
                try:
                    command = receive.get(timeout=_POLL_INTERVAL)
                except TimeoutError:
                    if shutting_down:
                        runtime.join(0)
                    continue
                receive = requests.recv()
                command = _decode(command, _COMMAND_TYPES)
                if isinstance(command, _Shutdown):
                    if not shutting_down:
                        shutting_down = True
                        runtime.shutdown(drain=command.drain)
                    continue
                if isinstance(command, _ScheduleTask):
                    runtime.submit(
                        command.task_id,
                        command.kind,
                        command.payload,
                    )
                elif isinstance(command, _CancelTask):
                    runtime.cancel(command.task_id, command.operation_id)
                else:
                    logger.debug("Ignoring unknown task-pool command: %r", command)
        finally:
            if not runtime.stopped:
                runtime.shutdown(drain=False)
                runtime.join(_STOP_TIMEOUT)


class _TaskPoolUnavailable(RuntimeError):
    """Raised when the remote task-pool service has stopped or failed."""


@dataclass
class _PendingTask:
    """Client-local state populated by the shared event dispatcher."""

    admission: concurrent.futures.Future[None]
    start: concurrent.futures.Future[bool]
    outcome: concurrent.futures.Future[_TaskOutcome]

    @classmethod
    def create(cls) -> _PendingTask:
        return cls(
            concurrent.futures.Future(),
            concurrent.futures.Future(),
            concurrent.futures.Future(),
        )


def _set_result[ValueT](
    future: concurrent.futures.Future[ValueT], value: ValueT
) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception(
    future: concurrent.futures.Future[Any], error: BaseException
) -> None:
    if not future.done():
        future.set_exception(error)


async def _await_local[ValueT](
    future: concurrent.futures.Future[ValueT], *, timeout: float | None = None
) -> ValueT:
    """Await local state without allowing observer cancellation to consume it."""
    observation = asyncio.shield(asyncio.wrap_future(future))
    if timeout is None:
        return await observation
    return await asyncio.wait_for(observation, timeout=timeout)


class _TaskHandle[ResultT]:
    """Concrete client-side handle for one remotely scheduled task."""

    def __init__(
        self,
        pool: _TaskPool,
        task_id: str,
        pending: _PendingTask,
    ) -> None:
        self._pool = pool
        self._task_id = task_id
        self._pending = pending

    @property
    def task_id(self) -> str:
        return self._task_id

    def wait_started(self, *, timeout: float | None = None) -> bool:
        return self._pending.start.result(timeout=timeout)

    async def await_started(self, *, timeout: float | None = None) -> bool:
        return await _await_local(self._pending.start, timeout=timeout)

    def cancel(self, *, timeout: float | None = None) -> bool:
        return self._pool._cancel_task(self._task_id, timeout=timeout)

    async def acancel(self, *, timeout: float | None = None) -> bool:
        return await self._pool._acancel_task(self._task_id, timeout=timeout)

    def _wait_outcome(self, *, timeout: float | None = None) -> _TaskOutcome:
        return self._pending.outcome.result(timeout=timeout)

    async def _await_outcome(self, *, timeout: float | None = None) -> _TaskOutcome:
        return await _await_local(self._pending.outcome, timeout=timeout)

    def result(self, *, timeout: float | None = None) -> ResultT:
        return cast(
            ResultT,
            _TaskPool._unwrap(self._wait_outcome(timeout=timeout)),
        )

    async def aresult(self, *, timeout: float | None = None) -> ResultT:
        outcome = await self._await_outcome(timeout=timeout)
        return cast(ResultT, _TaskPool._unwrap(outcome))


_dispatch_loop_lock = threading.Lock()
_dispatch_loop_instance: asyncio.AbstractEventLoop | None = None


def _dispatch_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide loop that routes every pool's response events.

    One thread serves all pools. A thread and event loop per pool (one per
    Machine) made a client with many Machines spend much of its GIL time
    waking loops; one loop drains whatever events are ready in each wakeup.
    """
    global _dispatch_loop_instance
    with _dispatch_loop_lock:
        if _dispatch_loop_instance is None or _dispatch_loop_instance.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="agentbox-task-pool-events", daemon=True
            ).start()
            _dispatch_loop_instance = loop
        return _dispatch_loop_instance


class _TaskPool:
    """Client for a worker-local task runtime using one shared event stream."""

    def __init__(
        self,
        proc_mesh: ProcMesh,
        *,
        capacity: int,
        handlers: dict[str, TaskHandler],
    ) -> None:
        from agentbox.actors.mesh_schema import make_name

        if capacity < 1:
            raise ValueError("task pool size must be positive")
        if not handlers:
            raise ValueError("task pool requires at least one handler")
        self._closed = False
        # Set by close(drain=False): tasks it cancels fail as unavailable.
        self._abandoning = False
        self._state_lock = threading.Lock()
        self._service_stopped = False
        self._service_failure: BaseException | None = None
        self._requests: Port[_TaskCommand] | None = None
        self._pending: dict[str, _PendingTask] = {}
        self._cancel_operations: dict[str, concurrent.futures.Future[bool]] = {}
        self._ready: concurrent.futures.Future[Port[_TaskCommand]] = (
            concurrent.futures.Future()
        )
        self._closed_response: concurrent.futures.Future[None] = (
            concurrent.futures.Future()
        )
        response_port, self._responses = Channel[_TaskPoolEvent].open()
        name = make_name("taskpool", uuid.uuid4().hex[:8])
        self._actor_mesh: ActorMesh[_TaskPoolActor] = proc_mesh.spawn(
            name,
            _TaskPoolActor,
            capacity,
            handlers,
        )
        try:
            self._service_future = self._actor_mesh.serve.choose(response_port)
            self._dispatcher: concurrent.futures.Future[None] | None = (
                asyncio.run_coroutine_threadsafe(
                    self._dispatcher_main(), _dispatch_loop()
                )
            )
        except BaseException:
            self._stop_actor_mesh(context="after setup failure")
            raise

    def _stop_actor_mesh(
        self,
        *,
        timeout: float = _STOP_TIMEOUT,
        context: str = "during close",
    ) -> None:
        """Best-effort synchronous teardown of the owned service mesh."""
        try:
            self._actor_mesh.stop().get(timeout=timeout)
        except Exception:
            logger.debug("Error stopping task pool %s", context, exc_info=True)

    async def _astop_actor_mesh(
        self,
        *,
        timeout: float = _STOP_TIMEOUT,
        context: str = "during close",
    ) -> None:
        """Best-effort asynchronous teardown of the owned service mesh."""
        try:
            await asyncio.wait_for(self._actor_mesh.stop(), timeout=timeout)
        except Exception:
            logger.debug("Error stopping task pool %s", context, exc_info=True)

    def _record_service_result(self, task: asyncio.Future[Any]) -> None:
        """Record why the long-lived task-pool service stopped.

        The endpoint's return value carries no application result: task results
        travel over the shared event stream. Calling ``task.result()`` here is
        solely how the dispatcher distinguishes normal service completion from
        an exception and retains that exception for later chaining.
        """
        try:
            task.result()
        except (Exception, asyncio.CancelledError) as exc:
            with self._state_lock:
                self._service_failure = exc
        else:
            with self._state_lock:
                self._service_stopped = True

    def _raise_service_unavailable(self) -> None:
        with self._state_lock:
            failure = self._service_failure
        if failure is not None:
            raise _TaskPoolUnavailable("task pool service failed") from failure
        raise _TaskPoolUnavailable("task pool service stopped unexpectedly")

    @staticmethod
    def _as_asyncio_observer(future: Any) -> asyncio.Future[Any]:
        as_asyncio = getattr(future, "as_asyncio", None)
        if as_asyncio is not None:
            return cast(asyncio.Future[Any], as_asyncio())
        return cast(asyncio.Future[Any], asyncio.ensure_future(future))

    def _service_error(self) -> _TaskPoolUnavailable:
        with self._state_lock:
            failure = self._service_failure
        error = _TaskPoolUnavailable(
            "task pool service failed"
            if failure is not None
            else "task pool service stopped unexpectedly"
        )
        if failure is not None:
            error.__cause__ = failure
        return error

    def _check_service_locked(self) -> None:
        if self._service_stopped or self._service_failure is not None:
            failure = self._service_failure
            if failure is not None:
                raise _TaskPoolUnavailable("task pool service failed") from failure
            raise _TaskPoolUnavailable("task pool service stopped unexpectedly")

    def _fail_waiters(self, error: BaseException) -> None:
        with self._state_lock:
            pending = tuple(self._pending.values())
            cancellations = tuple(self._cancel_operations.values())
            self._pending.clear()
            self._cancel_operations.clear()
        _set_exception(self._ready, error)
        _set_exception(self._closed_response, error)
        for state in pending:
            _set_exception(state.admission, error)
            _set_exception(state.start, error)
            _set_exception(state.outcome, error)
        for cancellation in cancellations:
            _set_exception(cancellation, error)

    def _dispatch_event(self, event: _TaskPoolEvent) -> bool:
        """Route one shared transport event to its client-local future."""
        event = _decode(event, _EVENT_TYPES)
        if isinstance(event, _TaskPoolReady):
            with self._state_lock:
                self._requests = event.requests
            _set_result(self._ready, event.requests)
            return False
        if isinstance(event, _TaskPoolClosed):
            _set_result(self._closed_response, None)
            return True
        if isinstance(event, _CancelResult):
            with self._state_lock:
                cancellation = self._cancel_operations.pop(event.operation_id, None)
            if cancellation is not None:
                _set_result(cancellation, event.found)
            return False

        with self._state_lock:
            state = self._pending.get(event.task_id)
            if isinstance(event, (_TaskRejected, _TaskCompletion)):
                self._pending.pop(event.task_id, None)
            abandoning = self._abandoning
        if state is None:
            logger.debug("Ignoring event for unknown task: %r", event)
            return False
        if isinstance(event, _TaskAccepted):
            _set_result(state.admission, None)
        elif isinstance(event, _TaskRejected):
            _set_exception(state.admission, event.error)
            _set_exception(state.start, event.error)
            _set_exception(state.outcome, event.error)
        elif isinstance(event, _TaskStarted):
            _set_result(state.start, True)
        elif isinstance(event, _TaskNeverStarted):
            _set_result(state.start, False)
        elif isinstance(event, _TaskCompletion):
            if abandoning and event.outcome[0] == "cancelled":
                _set_exception(
                    state.outcome,
                    _TaskPoolUnavailable("task pool closed before the task finished"),
                )
            else:
                _set_result(state.outcome, event.outcome)
        return False

    async def _dispatch_responses(self) -> None:
        """Own the response stream and exactly one service-future observer."""
        service = self._as_asyncio_observer(self._service_future)
        service_finished = False
        response: asyncio.Future[Any] | None = None
        try:
            while True:
                response = self._as_asyncio_observer(self._responses.recv())
                if service_finished:
                    event = cast(_TaskPoolEvent, await response)
                else:
                    done, _pending = await asyncio.wait(
                        (response, service),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if response in done:
                        event = cast(_TaskPoolEvent, response.result())
                    else:
                        self._record_service_result(service)
                        with self._state_lock:
                            expected_stop = (
                                self._closed and self._service_failure is None
                            )
                        if not expected_stop:
                            self._raise_service_unavailable()
                        service_finished = True
                        event = cast(_TaskPoolEvent, await response)
                if self._dispatch_event(event):
                    return
                response = None
        finally:
            observers = [
                item
                for item in (response, service)
                if item is not None and not item.done()
            ]
            for observer in observers:
                observer.cancel()
            if observers:
                await asyncio.gather(*observers, return_exceptions=True)

    async def _dispatcher_main(self) -> None:
        try:
            await self._dispatch_responses()
        except (Exception, asyncio.CancelledError) as exc:
            with self._state_lock:
                if self._service_failure is None:
                    self._service_failure = exc
            self._fail_waiters(self._service_error())

    def _join_dispatcher(self, timeout: float) -> None:
        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is None:
            return
        concurrent.futures.wait([dispatcher], timeout=timeout)
        if not dispatcher.done():
            logger.debug("Task-pool event dispatcher did not stop before timeout")

    async def _ajoin_dispatcher(self, timeout: float) -> None:
        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is None:
            return
        await asyncio.wait([asyncio.wrap_future(dispatcher)], timeout=timeout)
        if not dispatcher.done():
            logger.debug("Task-pool event dispatcher did not stop before timeout")

    def _ensure_ready(self, *, timeout: float | None = None) -> Port[_TaskCommand]:
        return self._ready.result(timeout=timeout)

    async def _aensure_ready(
        self, *, timeout: float | None = None
    ) -> Port[_TaskCommand]:
        return await _await_local(self._ready, timeout=timeout)

    @staticmethod
    def _unwrap(outcome: _TaskOutcome) -> Any:
        kind, payload = outcome
        if kind == "result":
            return payload
        if kind == "error" and isinstance(payload, BaseException):
            raise payload
        if kind == "cancelled":
            raise asyncio.CancelledError
        raise RuntimeError(f"unexpected task-pool outcome: {outcome!r}")

    def _prepare_schedule[ResultT](
        self, task: Task[ResultT]
    ) -> tuple[_ScheduleTask, _PendingTask, _TaskHandle[ResultT]]:
        task_id = uuid.uuid4().hex
        state = _PendingTask.create()
        command = _ScheduleTask(task_id, task.kind, task.to_payload())
        return command, state, _TaskHandle[ResultT](self, task_id, state)

    def _request_cancel(self, task_id: str) -> None:
        """Best-effort cancellation used when a convenience waiter is cancelled."""
        try:
            with self._state_lock:
                if self._requests is None:
                    return
                if self._service_stopped or self._service_failure is not None:
                    return
                self._requests.send(_encode(_CancelTask(task_id, None), _COMMAND_TYPES))
        except Exception:
            logger.debug(
                "Unable to request task cancellation task_id=%s",
                task_id,
                exc_info=True,
            )

    def _cancel_task(self, task_id: str, *, timeout: float | None = None) -> bool:
        requests = self._ensure_ready(timeout=timeout)
        operation_id = uuid.uuid4().hex
        result: concurrent.futures.Future[bool] = concurrent.futures.Future()
        with self._state_lock:
            self._check_service_locked()
            self._cancel_operations[operation_id] = result
            try:
                requests.send(
                    _encode(_CancelTask(task_id, operation_id), _COMMAND_TYPES)
                )
            except BaseException:
                self._cancel_operations.pop(operation_id, None)
                raise
        try:
            return result.result(timeout=timeout)
        finally:
            with self._state_lock:
                self._cancel_operations.pop(operation_id, None)

    async def _acancel_task(
        self, task_id: str, *, timeout: float | None = None
    ) -> bool:
        requests = await self._aensure_ready(timeout=timeout)
        operation_id = uuid.uuid4().hex
        result: concurrent.futures.Future[bool] = concurrent.futures.Future()
        with self._state_lock:
            self._check_service_locked()
            self._cancel_operations[operation_id] = result
            try:
                requests.send(
                    _encode(_CancelTask(task_id, operation_id), _COMMAND_TYPES)
                )
            except BaseException:
                self._cancel_operations.pop(operation_id, None)
                raise
        try:
            return await _await_local(result, timeout=timeout)
        finally:
            with self._state_lock:
                self._cancel_operations.pop(operation_id, None)

    def schedule[ResultT](self, task: Task[ResultT]) -> TaskHandle[ResultT]:
        requests = self._ensure_ready()
        command, state, handle = self._prepare_schedule(task)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("task pool is closed")
            self._check_service_locked()
            self._pending[command.task_id] = state
            try:
                requests.send(_encode(command, _COMMAND_TYPES))
            except BaseException:
                self._pending.pop(command.task_id, None)
                raise
        state.admission.result()
        return handle

    async def aschedule[ResultT](self, task: Task[ResultT]) -> TaskHandle[ResultT]:
        requests = await self._aensure_ready()
        command, state, handle = self._prepare_schedule(task)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("task pool is closed")
            self._check_service_locked()
            self._pending[command.task_id] = state
            try:
                requests.send(_encode(command, _COMMAND_TYPES))
            except BaseException:
                self._pending.pop(command.task_id, None)
                raise
        try:
            await _await_local(state.admission)
        except asyncio.CancelledError:
            self._request_cancel(command.task_id)
            raise
        return handle

    def submit[ResultT](self, task: Task[ResultT]) -> ResultT:
        return self.schedule(task).result()

    async def asubmit[ResultT](self, task: Task[ResultT]) -> ResultT:
        handle = cast(_TaskHandle[ResultT], await self.aschedule(task))
        try:
            outcome = await handle._await_outcome()
        except asyncio.CancelledError:
            self._request_cancel(handle.task_id)
            raise
        return cast(ResultT, self._unwrap(outcome))

    def close(self, *, timeout: float = _STOP_TIMEOUT, drain: bool = True) -> None:
        deadline = time.monotonic() + timeout
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._abandoning = not drain
        try:
            requests = self._ensure_ready(timeout=max(0.0, deadline - time.monotonic()))
            with self._state_lock:
                requests.send(_Shutdown(drain))
            self._closed_response.result(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            self._stop_actor_mesh()
            self._join_dispatcher(max(0.0, deadline - time.monotonic()))

    async def aclose(
        self, *, timeout: float = _STOP_TIMEOUT, drain: bool = True
    ) -> None:
        deadline = time.monotonic() + timeout
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._abandoning = not drain
        try:
            requests = await self._aensure_ready(
                timeout=max(0.0, deadline - time.monotonic())
            )
            with self._state_lock:
                requests.send(_Shutdown(drain))
            await _await_local(
                self._closed_response,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        finally:
            await self._astop_actor_mesh()
            await self._ajoin_dispatcher(max(0.0, deadline - time.monotonic()))


cloudpickle.register_pickle_by_value(sys.modules[__name__])
