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

"""Unit tests for the independently testable task-pool layers."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _endpoints import endpoint_method
from agentbox.tasks.pool import (
    _CancelTask,
    _COMMAND_TYPES,
    _decode,
    _encode,
    _EVENT_TYPES,
    _PendingTask,
    _ScheduleTask,
    _Shutdown,
    _TaskHandle,
    _TaskPool,
    _TaskPoolActor,
    _TaskPoolClosed,
    _TaskPoolReady,
    _TaskPoolUnavailable,
)
from agentbox.tasks.registry import default_task_handlers, TaskHandlerRegistry
from agentbox.tasks.scheduler import (
    _AsyncRuntimeThread,
    _CancelResult,
    _FifoTaskScheduler,
    _TaskAccepted,
    _TaskCompletion,
    _TaskNeverStarted,
    _TaskRejected,
    _TaskStarted,
    _TaskWork,
)


def _work(task_id: str, *payload: Any) -> _TaskWork:
    return _TaskWork(task_id, "test", payload, time.monotonic())


def _completions(events: list[Any]) -> list[_TaskCompletion]:
    return [event for event in events if isinstance(event, _TaskCompletion)]


class _TestTask:
    def __init__(self, kind: str, payload: tuple[Any, ...]) -> None:
        self.kind = kind
        self._payload = payload

    def to_payload(self) -> tuple[Any, ...]:
        return self._payload


def _make_task_pool(
    *,
    ready: bool = True,
    service_future: Any = None,
    responses: Any = None,
) -> tuple[_TaskPool, MagicMock, MagicMock]:
    """Build a real pool over a mock mesh, with the event dispatcher stubbed.

    Returns ``(pool, actor_mesh, requests)``. Tests feed service events through
    ``_dispatch_event``, as the dispatcher would. With *ready*, the service has
    already announced its *requests* port.
    """
    proc_mesh = MagicMock()
    actor_mesh = proc_mesh.spawn.return_value
    if service_future is not None:
        actor_mesh.serve.choose.return_value = service_future

    async def handler() -> None:
        pass

    with (
        patch("agentbox.tasks.pool.Channel") as channel,
        patch.object(_TaskPool, "_dispatcher_main", AsyncMock()),
    ):
        channel.__getitem__.return_value.open.return_value = (
            MagicMock(),
            responses if responses is not None else MagicMock(),
        )
        task_pool = _TaskPool(proc_mesh, capacity=1, handlers={"test": handler})
    requests = MagicMock()
    if ready:
        task_pool._dispatch_event(_TaskPoolReady(requests))
    return task_pool, actor_mesh, requests


class _FakeService:
    """Answers pool commands with events, as the remote service would.

    Replies are dispatched from another thread: the pool sends commands while
    holding its state lock, which event dispatch also takes.
    """

    def __init__(self, task_pool: _TaskPool, requests: MagicMock, reply: Any) -> None:
        self.commands: list[Any] = []
        self._threads: list[threading.Thread] = []

        def send(wire: Any) -> None:
            command = _decode(wire, _COMMAND_TYPES)
            self.commands.append(command)
            events = reply(command)
            thread = threading.Thread(
                target=lambda: [task_pool._dispatch_event(e) for e in events]
            )
            thread.start()
            self._threads.append(thread)

        requests.send.side_effect = send

    def join(self) -> None:
        for thread in self._threads:
            thread.join(timeout=10)


def _close_on_shutdown(command: Any) -> list[Any]:
    return [_TaskPoolClosed()] if isinstance(command, _Shutdown) else []


class _ReusableAsyncFuture:
    """Minimal model of Monarch's non-consuming asyncio observers."""

    def __init__(self) -> None:
        self.source: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.observer_count = 0

    def as_asyncio(self) -> asyncio.Future[Any]:
        self.observer_count += 1

        async def observe() -> Any:
            return await asyncio.shield(self.source)

        return asyncio.create_task(observe())


async def _wait_until(predicate: Any) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met")


class TestTaskHandlerRegistry:
    def test_default_registry_contains_subprocess_handler(self):
        assert "subprocess" in default_task_handlers.snapshot()

    def test_registers_handlers_and_returns_isolated_snapshot(self):
        registry = TaskHandlerRegistry()

        async def handler(value: int) -> int:
            return value

        registry.register("test", handler)
        snapshot = registry.snapshot()
        snapshot.clear()

        assert registry.snapshot() == {"test": handler}

    def test_rejects_empty_and_duplicate_kinds(self):
        registry = TaskHandlerRegistry()

        async def first() -> None:
            pass

        async def second() -> None:
            pass

        with pytest.raises(ValueError, match="must not be empty"):
            registry.register("", first)
        registry.register("test", first)
        registry.register("test", first)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", second)

    def test_can_extend_a_registry_without_mutating_it(self):
        base = TaskHandlerRegistry()

        async def base_handler() -> None:
            pass

        async def extra_handler() -> None:
            pass

        base.register("base", base_handler)
        extended = TaskHandlerRegistry(base)
        extended.register("extra", extra_handler)

        assert set(base.snapshot()) == {"base"}
        assert set(extended.snapshot()) == {"base", "extra"}


class TestFifoTaskScheduler:
    async def test_enforces_capacity_and_starts_queued_tasks_fifo(self):
        gates = {name: asyncio.Event() for name in ("a", "b", "c", "d")}
        started: list[str] = []
        events: list[Any] = []

        async def handler(name: str) -> str:
            started.append(name)
            await gates[name].wait()
            return name

        scheduler = _FifoTaskScheduler(
            2, {"test": handler}, events.append, lambda: None
        )
        for name in gates:
            scheduler.submit(_work(name, name))
        await asyncio.sleep(0)

        try:
            assert started == ["a", "b"]
            assert scheduler.active_count == 2
            assert scheduler.queue_depth == 2

            gates["b"].set()
            await _wait_until(lambda: started == ["a", "b", "c"])
            gates["a"].set()
            await _wait_until(lambda: started == ["a", "b", "c", "d"])
        finally:
            for gate in gates.values():
                gate.set()
            await _wait_until(lambda: len(_completions(events)) == 4)

        assert [
            event.task_id for event in events if isinstance(event, _TaskAccepted)
        ] == ["a", "b", "c", "d"]
        assert [
            event.task_id for event in events if isinstance(event, _TaskStarted)
        ] == ["a", "b", "c", "d"]
        assert {item.outcome for item in _completions(events)} == {
            ("result", "a"),
            ("result", "b"),
            ("result", "c"),
            ("result", "d"),
        }

    async def test_cancels_queued_and_active_tasks_independently(self):
        gate = asyncio.Event()
        events: list[Any] = []

        async def handler(_name: str) -> None:
            await gate.wait()

        scheduler = _FifoTaskScheduler(
            1, {"test": handler}, events.append, lambda: None
        )
        scheduler.submit(_work("active", "active"))
        scheduler.submit(_work("queued", "queued"))
        await asyncio.sleep(0)

        assert scheduler.cancel("queued")
        assert scheduler.queue_depth == 0
        assert any(
            isinstance(event, _TaskNeverStarted) and event.task_id == "queued"
            for event in events
        )
        assert _completions(events)[-1].task_id == "queued"
        assert _completions(events)[-1].outcome == ("cancelled", None)

        assert scheduler.cancel("active")
        assert not scheduler.cancel("missing")
        await _wait_until(lambda: len(_completions(events)) == 2)
        assert _completions(events)[-1].task_id == "active"
        assert _completions(events)[-1].outcome == ("cancelled", None)

    async def test_graceful_shutdown_drains_and_notifies_once(self):
        gate = asyncio.Event()
        events: list[Any] = []
        drained = 0

        async def handler(value: int) -> int:
            await gate.wait()
            return value

        def on_drained() -> None:
            nonlocal drained
            drained += 1

        scheduler = _FifoTaskScheduler(1, {"test": handler}, events.append, on_drained)
        scheduler.submit(_work("first", 1))
        scheduler.submit(_work("second", 2))
        scheduler.shutdown()

        assert not scheduler.accepting
        assert scheduler.queue_depth == 1
        scheduler.submit(_work("late", 3))
        assert isinstance(events[-1], _TaskRejected)
        assert isinstance(events[-1].error, RuntimeError)

        gate.set()
        await _wait_until(lambda: len(_completions(events)) == 2)
        assert drained == 1
        scheduler.shutdown()
        assert drained == 1

    async def test_immediate_shutdown_cancels_all_work(self):
        events: list[Any] = []
        drained = asyncio.Event()

        async def handler() -> None:
            await asyncio.Event().wait()

        scheduler = _FifoTaskScheduler(1, {"test": handler}, events.append, drained.set)
        scheduler.submit(_work("active"))
        scheduler.submit(_work("queued"))
        await asyncio.sleep(0)
        scheduler.shutdown(drain=False)

        await asyncio.wait_for(drained.wait(), timeout=1)
        completions = _completions(events)
        assert {item.task_id for item in completions} == {"active", "queued"}
        assert all(item.outcome == ("cancelled", None) for item in completions)
        assert any(
            isinstance(event, _TaskNeverStarted) and event.task_id == "queued"
            for event in events
        )

    def test_validates_configuration_without_starting_a_thread(self):
        async def handler() -> None:
            pass

        with pytest.raises(ValueError, match="positive"):
            _FifoTaskScheduler(0, {"test": handler}, lambda _item: None, lambda: None)
        with pytest.raises(ValueError, match="at least one handler"):
            _FifoTaskScheduler(1, {}, lambda _item: None, lambda: None)


class TestAsyncRuntimeThread:
    def test_multiplexes_async_tasks_on_exactly_one_thread(self):
        thread_ids: set[int] = set()
        running = 0
        max_running = 0
        all_running = asyncio.Event()

        async def handler(value: int) -> int:
            nonlocal running, max_running
            thread_ids.add(threading.get_ident())
            running += 1
            max_running = max(max_running, running)
            if running == 4:
                all_running.set()
            # Times out (failing the task) unless all four run at once.
            await asyncio.wait_for(all_running.wait(), timeout=10)
            running -= 1
            return value * 2

        runtime = _AsyncRuntimeThread(4, {"test": handler})
        try:
            for value in range(4):
                runtime.submit(str(value), "test", (value,))
            runtime.shutdown()
            assert runtime.join(timeout=30)
        finally:
            runtime.shutdown(drain=False)
            runtime.join(timeout=30)

        events: list[Any] = []
        while not runtime.events.empty():
            events.append(runtime.events.get_nowait())
        outcomes = {
            event.outcome for event in events if isinstance(event, _TaskCompletion)
        }
        assert thread_ids == {runtime.thread.ident}
        assert threading.get_ident() not in thread_ids
        assert max_running == 4
        assert outcomes == {
            ("result", 0),
            ("result", 2),
            ("result", 4),
            ("result", 6),
        }

    def test_rejects_invalid_configuration_from_runtime_thread(self):
        async def handler() -> None:
            pass

        with pytest.raises(RuntimeError, match="failed to start") as exc_info:
            _AsyncRuntimeThread(0, {"test": handler})
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_event_queue_can_be_injected(self):
        events: queue.SimpleQueue[Any] = queue.SimpleQueue()

        async def handler() -> str:
            return "done"

        runtime = _AsyncRuntimeThread(1, {"test": handler}, events)
        runtime.submit("task", "test", ())
        runtime.shutdown()
        assert runtime.join(timeout=2)
        emitted = [events.get_nowait() for _ in range(3)]
        assert [type(event) for event in emitted] == [
            _TaskAccepted,
            _TaskStarted,
            _TaskCompletion,
        ]
        assert emitted[-1].outcome == ("result", "done")

    def test_cancel_results_are_correlated_by_operation_id(self):
        events: queue.SimpleQueue[Any] = queue.SimpleQueue()

        async def handler() -> None:
            await asyncio.Event().wait()

        runtime = _AsyncRuntimeThread(1, {"test": handler}, events)
        try:
            runtime.submit("task", "test", ())
            runtime.cancel("task", "cancel-1")
            runtime.shutdown()
            assert runtime.join(timeout=2)
        finally:
            runtime.shutdown(drain=False)
            runtime.join(timeout=2)

        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        assert _CancelResult("cancel-1", True) in emitted


class TestTaskPoolActor:
    def test_non_draining_shutdown_reaches_runtime(self):
        requests = MagicMock()
        shutdown = MagicMock()
        shutdown.get.return_value = _Shutdown(drain=False)
        requests.recv.side_effect = [shutdown, MagicMock()]
        runtime = MagicMock()
        runtime.stopped = False

        def stop_runtime(*, drain: bool) -> None:
            runtime.stopped = True

        runtime.shutdown.side_effect = stop_runtime
        actor = _TaskPoolActor(4, {"subprocess": MagicMock()})

        with (
            patch("agentbox.tasks.pool.Channel") as channel,
            patch("agentbox.tasks.pool._AsyncRuntimeThread", return_value=runtime),
        ):
            channel.__getitem__.return_value.open.return_value = (
                MagicMock(),
                requests,
            )
            endpoint_method(_TaskPoolActor, "serve")(actor, MagicMock())

        runtime.shutdown.assert_called_once_with(drain=False)

    def test_translates_transport_commands_to_runtime_operations(self):
        request_port = MagicMock()
        requests = MagicMock()
        submit = MagicMock()
        submit.get.return_value = _ScheduleTask(
            "task-1",
            "subprocess",
            ("payload",),
        )
        cancel = MagicMock()
        cancel.get.return_value = _CancelTask("task-1", "cancel-reply")
        shutdown = MagicMock()
        shutdown.get.return_value = _Shutdown(drain=True)
        requests.recv.side_effect = [submit, cancel, shutdown, MagicMock()]

        runtime = MagicMock()
        runtime.stopped = False

        def stop_runtime(*, drain: bool) -> None:
            assert drain
            runtime.stopped = True

        runtime.shutdown.side_effect = stop_runtime
        responses = MagicMock()
        actor = _TaskPoolActor(4, {"subprocess": MagicMock()})

        with (
            patch("agentbox.tasks.pool.Channel") as channel,
            patch(
                "agentbox.tasks.pool._AsyncRuntimeThread",
                return_value=runtime,
            ) as runtime_type,
        ):
            channel.__getitem__.return_value.open.return_value = (
                request_port,
                requests,
            )
            endpoint_method(_TaskPoolActor, "serve")(actor, responses)

        runtime_type.assert_called_once()
        runtime.submit.assert_called_once_with("task-1", "subprocess", ("payload",))
        runtime.cancel.assert_called_once_with("task-1", "cancel-reply")
        runtime.shutdown.assert_called_once_with(drain=True)
        assert isinstance(responses.send.call_args_list[0].args[0], _TaskPoolReady)
        assert isinstance(responses.send.call_args_list[-1].args[0], _TaskPoolClosed)


class TestTaskHandle:
    def test_exposes_start_result_and_cancellation_without_transport_details(self):
        pool = MagicMock()
        pool._cancel_task.return_value = True
        pending = _PendingTask.create()
        pending.start.set_result(True)
        pending.outcome.set_result(("result", 42))
        handle = _TaskHandle[int](pool, "task-1", pending)

        assert handle.task_id == "task-1"
        assert handle.wait_started(timeout=1)
        assert handle.cancel(timeout=1)
        assert handle.result(timeout=1) == 42
        pool._cancel_task.assert_called_once_with("task-1", timeout=1)

    def test_reports_when_a_queued_task_never_started(self):
        pending = _PendingTask.create()
        pending.start.set_result(False)
        handle = _TaskHandle[Any](MagicMock(), "task-1", pending)

        assert not handle.wait_started()

    async def test_cancelled_observer_does_not_consume_local_result(self):
        pending = _PendingTask.create()
        handle = _TaskHandle[str](MagicMock(), "task-1", pending)

        first = asyncio.create_task(handle.aresult())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        pending.outcome.set_result(("result", "done"))
        assert await handle.aresult() == "done"
        assert await handle.aresult() == "done"


class TestTaskPoolClient:
    def test_rejects_empty_handler_snapshot_before_spawning_actor(self):
        proc_mesh = MagicMock()
        with pytest.raises(ValueError, match="at least one handler"):
            _TaskPool(proc_mesh, capacity=1, handlers={})
        proc_mesh.spawn.assert_not_called()

    def test_initialization_owns_only_remote_transport_setup(self):
        proc_mesh = MagicMock()
        actor_mesh = proc_mesh.spawn.return_value
        response_port = MagicMock()

        async def handler() -> None:
            pass

        handlers = {"test": handler}
        with (
            patch("agentbox.tasks.pool.Channel") as channel,
            patch.object(_TaskPool, "_dispatcher_main", AsyncMock()) as dispatcher,
        ):
            channel.__getitem__.return_value.open.return_value = (
                response_port,
                MagicMock(),
            )
            _TaskPool(proc_mesh, capacity=3, handlers=handlers)

        spawn_args = proc_mesh.spawn.call_args.args
        assert spawn_args[1:] == (_TaskPoolActor, 3, handlers)
        actor_mesh.serve.choose.assert_called_once_with(response_port)
        dispatcher.assert_called_once_with()

    def test_close_drains_service_and_stops_owned_actor_mesh(self):
        task_pool, actor_mesh, requests = _make_task_pool()
        service = _FakeService(task_pool, requests, _close_on_shutdown)

        task_pool.close()

        assert service.commands == [_Shutdown(drain=True)]
        actor_mesh.stop.assert_called_once_with()
        actor_mesh.stop.return_value.get.assert_called_once_with(timeout=10.0)

    def test_close_without_drain_sends_non_draining_shutdown(self):
        task_pool, actor_mesh, requests = _make_task_pool()
        service = _FakeService(task_pool, requests, _close_on_shutdown)

        task_pool.close(drain=False)

        assert service.commands == [_Shutdown(drain=False)]
        actor_mesh.stop.assert_called_once_with()

    @pytest.mark.parametrize("abandoning", [False, True])
    def test_cancelled_completion_fails_only_when_close_abandons(self, abandoning):
        # Machine.release closes without draining, cancelling in-flight
        # commands; their callers must get an error, not a bare CancelledError.
        task_pool, _actor_mesh, requests = _make_task_pool()

        def reply(command: Any) -> list[Any]:
            if isinstance(command, _ScheduleTask):
                return [_TaskAccepted(command.task_id)]
            if isinstance(command, _Shutdown):
                return [_TaskCompletion(handle.task_id, ("cancelled", None))] + [
                    _TaskPoolClosed()
                ]
            return []

        service = _FakeService(task_pool, requests, reply)
        handle = task_pool.schedule(_TestTask("test", ()))

        if abandoning:
            task_pool.close(drain=False)
            service.join()
            with pytest.raises(_TaskPoolUnavailable, match="closed before"):
                handle.result(timeout=0)
        else:
            task_pool._dispatch_event(
                _TaskCompletion(handle.task_id, ("cancelled", None))
            )
            with pytest.raises(asyncio.CancelledError):
                handle.result(timeout=0)

    def test_close_stops_owned_actor_mesh_when_graceful_drain_fails(self):
        # The service never confirms the shutdown, so the drain times out.
        task_pool, actor_mesh, _requests = _make_task_pool()

        with pytest.raises(TimeoutError):
            task_pool.close(timeout=0.05)

        actor_mesh.stop.assert_called_once_with()
        actor_mesh.stop.return_value.get.assert_called_once_with(timeout=10.0)

    async def test_aclose_stops_owned_actor_mesh_when_graceful_drain_fails(self):
        task_pool, actor_mesh, _requests = _make_task_pool()
        actor_mesh.stop = AsyncMock()

        with pytest.raises(TimeoutError):
            await task_pool.aclose(timeout=0.05)

        actor_mesh.stop.assert_awaited_once_with()

    def test_submit_encodes_request_and_unwraps_result(self):
        task_pool, _actor_mesh, requests = _make_task_pool()

        def reply(command: Any) -> list[Any]:
            return [
                _TaskAccepted(command.task_id),
                _TaskStarted(command.task_id),
                _TaskCompletion(command.task_id, ("result", "done")),
            ]

        service = _FakeService(task_pool, requests, reply)
        with patch("agentbox.tasks.pool.Channel") as channel:
            result = task_pool.submit(_TestTask("test", (1, 2)))
        service.join()

        assert result == "done"
        wire = requests.send.call_args.args[0]
        assert isinstance(wire, tuple)  # per-task messages travel as builtins
        (command,) = service.commands
        assert isinstance(command, _ScheduleTask)
        assert command.kind == "test"
        assert command.payload == (1, 2)
        channel.assert_not_called()

    async def test_cancelled_submit_sends_matching_cancel_command(self):
        task_pool, _actor_mesh, requests = _make_task_pool()

        def reply(command: Any) -> list[Any]:
            if isinstance(command, _ScheduleTask):
                return [_TaskAccepted(command.task_id)]
            return []

        service = _FakeService(task_pool, requests, reply)
        submit = asyncio.create_task(task_pool.asubmit(_TestTask("test", ())))
        await _wait_until(lambda: len(service.commands) == 1)
        service.join()
        await asyncio.sleep(0)
        submit.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submit

        schedule_command, cancel_command = service.commands
        assert isinstance(schedule_command, _ScheduleTask)
        assert schedule_command.kind == "test"
        assert cancel_command == _CancelTask(schedule_command.task_id, None)

    def test_shared_events_are_correlated_and_release_pool_state(self):
        task_pool, _actor_mesh, requests = _make_task_pool()
        _FakeService(
            task_pool,
            requests,
            lambda command: [_TaskAccepted(command.task_id)],
        )
        first = task_pool.schedule(_TestTask("test", ()))
        second = task_pool.schedule(_TestTask("test", ()))

        task_pool._dispatch_event(_TaskStarted(second.task_id))
        task_pool._dispatch_event(_TaskCompletion(second.task_id, ("result", 2)))

        assert second.wait_started(timeout=0)
        assert second.result(timeout=0) == 2
        with pytest.raises(TimeoutError):
            first.wait_started(timeout=0)
        # Completed tasks are forgotten; others are still tracked.
        assert set(task_pool._pending) == {first.task_id}

    def test_wire_codec_round_trips_and_stays_small(self):
        import cloudpickle

        command = _ScheduleTask("t" * 32, "subprocess", (["true"], None))
        event = _TaskCompletion("t" * 32, ("result", {"returncode": 0}))
        assert _decode(_encode(command, _COMMAND_TYPES), _COMMAND_TYPES) == command
        assert _decode(_encode(event, _EVENT_TYPES), _EVENT_TYPES) == event
        # By-value dataclasses pickle their class (~5 KB); the wire form must not.
        assert len(cloudpickle.dumps(_encode(event, _EVENT_TYPES))) < 512
        # Rare commands travel as-is.
        assert _encode(_Shutdown(drain=False), _COMMAND_TYPES) == _Shutdown(False)
        # Malformed frames fall through to the unknown-command path.
        for frame in [
            (),
            (-1, "t", None),
            (9, "t"),
            (0, "too", "few"),
            ("0", "t", "k", ()),
        ]:
            assert _decode(frame, _COMMAND_TYPES) is frame

    def test_event_send_failure_fails_the_service(self):
        # Events are sent from the runtime thread's callbacks, where an
        # exception would only be logged; serve() must still fail the service.
        async def handler() -> str:
            return "done"

        def send(message: Any) -> None:
            if isinstance(_decode(message, _EVENT_TYPES), _TaskCompletion):
                raise RuntimeError("unpicklable result")

        responses = MagicMock()
        responses.send.side_effect = send
        submit = MagicMock()
        submit.get.return_value = _encode(
            _ScheduleTask("task-1", "test", ()), _COMMAND_TYPES
        )
        idle = MagicMock()
        idle.get.side_effect = TimeoutError
        requests = MagicMock()
        requests.recv.side_effect = [submit] + [idle] * 10_000
        actor = _TaskPoolActor(1, {"test": handler})
        with patch("agentbox.tasks.pool.Channel") as channel:
            channel.__getitem__.return_value.open.return_value = (MagicMock(), requests)
            with pytest.raises(RuntimeError, match="unpicklable result"):
                endpoint_method(_TaskPoolActor, "serve")(actor, responses)

    async def test_dispatcher_reuses_one_service_observer_for_all_events(self):
        service_future = _ReusableAsyncFuture()
        responses = [_ReusableAsyncFuture() for _ in range(4)]
        response_stream = MagicMock()
        response_stream.recv.side_effect = responses
        task_pool, _actor_mesh, _requests = _make_task_pool(
            ready=False,
            service_future=service_future,
            responses=response_stream,
        )

        dispatch = asyncio.create_task(task_pool._dispatch_responses())
        requests = MagicMock()
        for source, event in zip(
            responses,
            (
                _TaskPoolReady(requests),
                _TaskAccepted("task"),
                _TaskCompletion("task", ("result", "done")),
                _TaskPoolClosed(),
            ),
            strict=True,
        ):
            source.source.set_result(event)
            await asyncio.sleep(0)
        await dispatch

        assert service_future.observer_count == 1
        assert task_pool._ensure_ready(timeout=0) is requests

    def test_unwrap_preserves_results_errors_and_cancellation(self):
        assert _TaskPool._unwrap(("result", 3)) == 3
        with pytest.raises(ValueError, match="bad"):
            _TaskPool._unwrap(("error", ValueError("bad")))
        with pytest.raises(asyncio.CancelledError):
            _TaskPool._unwrap(("cancelled", None))
