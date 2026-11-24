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

"""Tests for AgentBoxManager machine-acquisition wait policies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentbox.actors.manager import AgentBoxManager
from agentbox.actors.resource_managers import (
    PoolExhaustedError,
    PoolStatus,
    ResourceHandle,
)
from monarch.actor import ActorError


def _manager() -> AgentBoxManager:
    manager = AgentBoxManager.__new__(AgentBoxManager)
    manager._manager = MagicMock()
    del manager._manager.poll_pending
    return manager


def _handle() -> ResourceHandle:
    return ResourceHandle(
        handle_id="handle-1",
        hostnames=["worker-1"],
        addresses=["tcp://worker-1:1234"],
    )


def test_nonblocking_acquire_does_not_wait_for_pool() -> None:
    manager = _manager()
    manager._manager.acquire.choose.return_value.get.side_effect = PoolExhaustedError(
        1, 0
    )

    with (
        patch("agentbox.actors.manager.wait_for_pool") as wait_for_pool,
        pytest.raises(PoolExhaustedError),
    ):
        manager._acquire_resource_handle(blocking=False, timeout=None)

    wait_for_pool.assert_not_called()


def test_nonblocking_acquire_unwraps_remote_pool_error() -> None:
    manager = _manager()
    manager._manager.acquire.choose.return_value.get.side_effect = ActorError(
        PoolExhaustedError(1, 0)
    )

    with (
        patch("agentbox.actors.manager.wait_for_pool") as wait_for_pool,
        pytest.raises(PoolExhaustedError),
    ):
        manager._acquire_resource_handle(blocking=False, timeout=None)

    wait_for_pool.assert_not_called()


def test_unexpected_actor_error_is_not_unwrapped() -> None:
    manager = _manager()
    actor_error = ActorError(RuntimeError("remote failure"))
    manager._manager.acquire.choose.return_value.get.side_effect = actor_error

    with pytest.raises(ActorError) as exc_info:
        manager._acquire_resource_handle(blocking=False, timeout=None)

    assert exc_info.value is actor_error


def test_blocking_acquire_passes_none_for_infinite_wait() -> None:
    manager = _manager()
    handle = _handle()
    manager._manager.acquire.choose.return_value.get.side_effect = [
        PoolExhaustedError(1, 0),
        handle,
    ]

    with patch("agentbox.actors.manager.wait_for_pool") as wait_for_pool:
        result = manager._acquire_resource_handle(blocking=True, timeout=None)

    assert result is handle
    wait_for_pool.assert_called_once_with(
        manager._manager,
        min_available=1,
        timeout=None,
    )


def test_blocking_acquire_passes_remaining_timeout() -> None:
    manager = _manager()
    handle = _handle()
    clock = [100.0]
    attempts = iter([PoolExhaustedError(1, 0), handle])

    def slow_acquire() -> ResourceHandle:
        clock[0] += 1.5  # The failed attempt uses 1.5s of the 5s budget.
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    manager._manager.acquire.choose.return_value.get.side_effect = slow_acquire

    with (
        patch("agentbox.actors.manager._monotonic", side_effect=lambda: clock[0]),
        patch("agentbox.actors.manager.wait_for_pool") as wait_for_pool,
    ):
        result = manager._acquire_resource_handle(blocking=True, timeout=5)

    assert result is handle
    wait_for_pool.assert_called_once_with(
        manager._manager,
        min_available=1,
        timeout=3.5,
    )


def test_zero_timeout_attempts_once_then_times_out() -> None:
    manager = _manager()
    manager._manager.acquire.choose.return_value.get.side_effect = PoolExhaustedError(
        1, 0
    )

    with (
        patch("agentbox.actors.manager._monotonic", return_value=100.0),
        patch("agentbox.actors.manager.wait_for_pool") as wait_for_pool,
        pytest.raises(TimeoutError, match="available machine"),
    ):
        manager._acquire_resource_handle(blocking=True, timeout=0)

    wait_for_pool.assert_not_called()


def test_zero_timeout_polls_pending_workers_before_acquiring() -> None:
    manager = _manager()
    handle = _handle()
    calls: list[str] = []
    manager._manager.poll_pending = MagicMock()
    manager._manager.poll_pending.choose.return_value.get.side_effect = lambda: (
        calls.append("poll_pending")
    )
    manager._manager.acquire.choose.return_value.get.side_effect = lambda: (
        calls.append("acquire") or handle
    )

    result = manager._acquire_resource_handle(blocking=True, timeout=0)

    assert result is handle
    assert calls == ["poll_pending", "acquire"]
    manager._manager.poll_pending.choose.assert_called_once_with()


def test_nan_timeout_is_rejected_before_acquisition() -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="NaN is not supported"):
        manager._acquire_resource_handle(blocking=True, timeout=float("nan"))

    manager._manager.acquire.choose.assert_not_called()


def test_wait_for_workers_defaults_to_indefinite_wait() -> None:
    manager = _manager()
    status = PoolStatus(available=1, in_use=0, total=1)

    with patch(
        "agentbox.actors.manager.wait_for_pool", return_value=status
    ) as wait_for_pool:
        result = manager.wait_for_workers()

    assert result is status
    wait_for_pool.assert_called_once_with(
        manager._manager,
        min_available=1,
        timeout=None,
    )


def test_blocking_acquire_normalizes_wait_timeout() -> None:
    manager = _manager()
    manager._manager.acquire.choose.return_value.get.side_effect = PoolExhaustedError(
        1, 0
    )
    wait_error = TimeoutError("Only 0/1 workers available after 5s")

    with (
        patch("agentbox.actors.manager.wait_for_pool", side_effect=wait_error),
        pytest.raises(
            TimeoutError, match="^Timed out waiting for an available machine$"
        ) as exc_info,
    ):
        manager._acquire_resource_handle(blocking=True, timeout=5)

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert str(exc_info.value.__cause__) == str(wait_error)


async def test_async_blocking_acquire_passes_none_for_infinite_wait() -> None:
    manager = _manager()
    handle = _handle()
    manager._manager.acquire.choose = AsyncMock(
        side_effect=[PoolExhaustedError(1, 0), handle]
    )

    with patch(
        "agentbox.actors.manager.await_for_pool", new_callable=AsyncMock
    ) as await_for_pool:
        result = await manager._aacquire_resource_handle(
            blocking=True,
            timeout=None,
        )

    assert result is handle
    await_for_pool.assert_awaited_once_with(
        manager._manager,
        min_available=1,
        timeout=None,
    )


async def test_async_nonblocking_acquire_does_not_wait_for_pool() -> None:
    manager = _manager()
    manager._manager.acquire.choose = AsyncMock(side_effect=PoolExhaustedError(1, 0))

    with (
        patch(
            "agentbox.actors.manager.await_for_pool", new_callable=AsyncMock
        ) as await_for_pool,
        pytest.raises(PoolExhaustedError),
    ):
        await manager._aacquire_resource_handle(blocking=False, timeout=None)

    await_for_pool.assert_not_awaited()


async def test_async_nonblocking_acquire_unwraps_remote_pool_error() -> None:
    manager = _manager()
    manager._manager.acquire.choose = AsyncMock(
        side_effect=ActorError(PoolExhaustedError(1, 0))
    )

    with (
        patch(
            "agentbox.actors.manager.await_for_pool", new_callable=AsyncMock
        ) as await_for_pool,
        pytest.raises(PoolExhaustedError),
    ):
        await manager._aacquire_resource_handle(blocking=False, timeout=None)

    await_for_pool.assert_not_awaited()


async def test_async_zero_timeout_polls_pending_workers_before_acquiring() -> None:
    manager = _manager()
    handle = _handle()
    calls: list[str] = []

    async def poll_pending() -> None:
        calls.append("poll_pending")

    async def acquire(_count: int) -> ResourceHandle:
        calls.append("acquire")
        return handle

    manager._manager.poll_pending = MagicMock()
    manager._manager.poll_pending.choose = AsyncMock(side_effect=poll_pending)
    manager._manager.acquire.choose = AsyncMock(side_effect=acquire)

    result = await manager._aacquire_resource_handle(blocking=True, timeout=0)

    assert result is handle
    assert calls == ["poll_pending", "acquire"]
    manager._manager.poll_pending.choose.assert_awaited_once_with()


async def test_async_blocking_acquire_normalizes_wait_timeout() -> None:
    manager = _manager()
    manager._manager.acquire.choose = AsyncMock(side_effect=PoolExhaustedError(1, 0))
    wait_error = TimeoutError("Only 0/1 workers available after 5s")

    with (
        patch(
            "agentbox.actors.manager.await_for_pool",
            new_callable=AsyncMock,
            side_effect=wait_error,
        ),
        pytest.raises(
            TimeoutError, match="^Timed out waiting for an available machine$"
        ) as exc_info,
    ):
        await manager._aacquire_resource_handle(blocking=True, timeout=5)

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert str(exc_info.value.__cause__) == str(wait_error)


async def test_async_indefinite_pool_wait_is_cancellable() -> None:
    manager = _manager()
    manager._manager.acquire.choose = AsyncMock(side_effect=PoolExhaustedError(1, 0))
    pool_polled = asyncio.Event()

    async def empty_pool_status() -> PoolStatus:
        pool_polled.set()
        return PoolStatus(available=0, in_use=0, total=0)

    manager._manager.pool_status.choose = AsyncMock(side_effect=empty_pool_status)
    acquire_task = asyncio.create_task(
        manager._aacquire_resource_handle(blocking=True, timeout=None)
    )
    await asyncio.wait_for(pool_polled.wait(), timeout=1)
    await asyncio.sleep(0)

    acquire_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(acquire_task, timeout=1)


def test_connect_retries_transient_failure() -> None:
    manager = _manager()
    machine = MagicMock()
    with (
        patch(
            "agentbox.actors.manager.Machine.from_handle",
            side_effect=[RuntimeError("endpoint proc query timed out"), machine],
        ) as from_handle,
        patch("agentbox.actors.manager.time.sleep") as sleep,
    ):
        assert manager._connect_machine(_handle(), "w") is machine
    assert from_handle.call_count == 2
    sleep.assert_called_once()
    manager._manager.release.choose.assert_not_called()


def test_connect_releases_handle_after_final_failure() -> None:
    manager = _manager()
    with (
        patch(
            "agentbox.actors.manager.Machine.from_handle",
            side_effect=RuntimeError("down"),
        ) as from_handle,
        patch("agentbox.actors.manager.time.sleep"),
        pytest.raises(RuntimeError, match="down"),
    ):
        manager._connect_machine(_handle(), "w")
    assert from_handle.call_count == 3
    manager._manager.release.choose.assert_called_once_with("handle-1")


async def test_async_connect_retries_then_releases() -> None:
    manager = _manager()
    manager._manager.release.choose = AsyncMock()
    with (
        patch(
            "agentbox.actors.manager.Machine.afrom_handle",
            new=AsyncMock(side_effect=RuntimeError("down")),
        ) as afrom_handle,
        patch("agentbox.actors.manager.asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError, match="down"),
    ):
        await manager._aconnect_machine(_handle(), "w")
    assert afrom_handle.call_count == 3
    manager._manager.release.choose.assert_called_once_with("handle-1")


async def test_async_connect_cancelled_during_backoff_releases() -> None:
    manager = _manager()
    manager._manager.release.choose = AsyncMock()
    with (
        patch(
            "agentbox.actors.manager.Machine.afrom_handle",
            new=AsyncMock(side_effect=RuntimeError("endpoint proc query timed out")),
        ),
        patch(
            "agentbox.actors.manager.asyncio.sleep",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await manager._aconnect_machine(_handle(), "w")
    manager._manager.release.choose.assert_called_once_with("handle-1")


def test_connect_interrupted_during_backoff_releases() -> None:
    manager = _manager()
    with (
        patch(
            "agentbox.actors.manager.Machine.from_handle",
            side_effect=RuntimeError("endpoint proc query timed out"),
        ),
        patch("agentbox.actors.manager.time.sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        manager._connect_machine(_handle(), "w")
    manager._manager.release.choose.assert_called_once_with("handle-1")
