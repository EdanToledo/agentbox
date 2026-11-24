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

"""Tests for agentbox.actors.resource_managers.base (unified SlurmResourceManager)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _endpoints import run_endpoint
from agentbox.actors.mesh_schema import make_name, make_scoped_name
from agentbox.actors.resource_managers.base import (
    await_for_pool,
    PoolExhaustedError,
    PoolStatus,
    ResourceManager,
    wait_for_pool,
)
from agentbox.actors.resource_managers.slurm import SlurmResourceManager
from agentbox.actors.resource_managers.slurm_array import (
    SlurmAllocation,
    SlurmArrayChunk,
)


def _make_chunk(
    chunk_size: int = 4, max_parallel: int = 2, job_id: str = "99999"
) -> SlurmArrayChunk:
    """Create a SlurmArrayChunk with a pre-set job ID (bypass submit)."""
    chunk = SlurmArrayChunk.__new__(SlurmArrayChunk)
    chunk.chunk_size = chunk_size
    chunk.allocation = SlurmAllocation()
    chunk.max_parallel = max_parallel
    chunk.python_exe = "python"
    chunk.array_job_id = job_id
    chunk._known_running = {}
    chunk._known_dead = set()
    chunk._last_poll = []
    chunk._last_poll_ok = True
    return chunk


def _make_manager(
    hostnames: list[str],
    chunk: SlurmArrayChunk | None = None,
    job_id: str = "99999",
) -> tuple[SlurmResourceManager, SlurmArrayChunk]:
    """Construct a SlurmResourceManager with pre-built slot mappings (bypassing __init__)."""
    if chunk is None:
        chunk = _make_chunk(chunk_size=len(hostnames), job_id=job_id)

    mgr = SlurmResourceManager.__new__(SlurmResourceManager)
    mgr._allocation = SlurmAllocation()
    mgr._max_parallel = 128
    mgr._chunk_size = len(hostnames)
    mgr._python_exe = "python"
    mgr._chunks = [chunk]
    mgr._available = []
    mgr._handles = {}
    mgr._failed_slots = set()
    mgr._slot_to_host = {}
    mgr._slot_to_chunk = {}
    mgr._slot_to_task_id = {}
    mgr._slot_to_port = {}
    mgr._mesh_name_to_slot = {}
    mgr._slot_to_monitor_mesh = {}
    mgr._port_server = MagicMock()

    for i, h in enumerate(hostnames):
        slot_key = f"{chunk.array_job_id}_{i}"
        mgr._available.append(slot_key)
        mgr._slot_to_host[slot_key] = h
        mgr._slot_to_chunk[slot_key] = chunk
        mgr._slot_to_task_id[slot_key] = i
        mgr._slot_to_port[slot_key] = 22222

    return mgr, chunk


class TestAcquireRelease:
    def test_acquire_single_host(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        handle = run_endpoint(mgr, "acquire", 1)

        assert len(handle.hostnames) == 1
        assert handle.hostnames[0] in ("node-001", "node-002")
        assert handle.handle_id
        assert handle.hostnames[0] in handle.metadata["host_to_port"]

    def test_acquire_all_hosts(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        handle = run_endpoint(mgr, "acquire", 2)

        assert len(handle.hostnames) == 2
        assert set(handle.hostnames) == {"node-001", "node-002"}

    def test_acquire_too_many_raises(self):
        mgr, _ = _make_manager(["node-001"])
        with pytest.raises(PoolExhaustedError, match="only 1 available") as exc_info:
            run_endpoint(mgr, "acquire", 2)
        assert exc_info.value.requested == 2
        assert exc_info.value.available == 1

    def test_acquire_zero_raises(self):
        mgr, _ = _make_manager(["node-001"])
        with pytest.raises(ValueError, match="n must be >= 1"):
            run_endpoint(mgr, "acquire", 0)

    def test_release_returns_hosts_to_pool(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        handle = run_endpoint(mgr, "acquire", 2)
        assert len(mgr._available) == 0

        run_endpoint(mgr, "release", handle.handle_id)
        assert len(mgr._available) == 2

    def test_release_unknown_handle_raises(self):
        mgr, _ = _make_manager(["node-001"])
        with pytest.raises(KeyError, match="Unknown handle"):
            run_endpoint(mgr, "release", "nonexistent")

    def test_double_release_raises(self):
        mgr, _ = _make_manager(["node-001"])
        handle = run_endpoint(mgr, "acquire", 1)
        run_endpoint(mgr, "release", handle.handle_id)
        with pytest.raises(KeyError):
            run_endpoint(mgr, "release", handle.handle_id)


class TestPoolStatus:
    def test_initial_status(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        status = run_endpoint(mgr, "pool_status")

        assert status.available == 2
        assert status.in_use == 0
        assert status.total == 2
        assert "99999" in status.extras["active_array_jobs"]

    def test_status_after_acquire(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        run_endpoint(mgr, "acquire", 1)
        status = run_endpoint(mgr, "pool_status")

        assert status.available == 1
        assert status.in_use == 1
        assert status.total == 2

    def test_status_after_release(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        handle = run_endpoint(mgr, "acquire", 2)
        run_endpoint(mgr, "release", handle.handle_id)
        status = run_endpoint(mgr, "pool_status")

        assert status.available == 2
        assert status.in_use == 0


class TestReportFailure:
    def test_failed_worker_is_omitted_while_teardown_completes(self):
        mgr, chunk = _make_manager(["node-001"])
        chunk.cancel_task = MagicMock()
        slot_key = mgr._available[0]

        run_endpoint(mgr, "report_failure", slot_key, "connection lost")

        status = run_endpoint(mgr, "pool_status")
        assert status.total == 0
        assert status.available == 0
        assert slot_key in mgr._failed_slots
        chunk.cancel_task.assert_called_once_with(0)

        chunk.get_dead_tasks = MagicMock(return_value=[(0, "node-001")])
        run_endpoint(mgr, "health_check")

        status = run_endpoint(mgr, "pool_status")
        assert status.total == 0
        assert status.available == 0
        assert slot_key not in mgr._failed_slots


class TestScaleDown:
    def test_scale_down_removes_idle_hosts(self):
        mgr, chunk = _make_manager(["node-001", "node-002", "node-003"])
        chunk.cancel_task = MagicMock()
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "scale_by", -1)
        assert len(mgr._available) == 2

    def test_scale_down_cancels_empty_chunk(self):
        mgr, chunk = _make_manager(["node-001"])
        chunk.cancel_task = MagicMock()
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "scale_by", -1)

        chunk.cancel_all.assert_called_once()
        assert chunk not in mgr._chunks

    def test_scale_down_keeps_chunk_with_remaining_hosts(self):
        mgr, chunk = _make_manager(["node-001", "node-002"])
        chunk.cancel_task = MagicMock()
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "scale_by", -1)

        chunk.cancel_all.assert_not_called()
        assert chunk in mgr._chunks

    def test_scale_down_stops_monitor_mesh_before_cancel(self):
        mgr, chunk = _make_manager(["node-001"])
        chunk.cancel_task = MagicMock()
        chunk.cancel_all = MagicMock()

        # Attach a fake supervision mesh whose stop() must be called *before*
        # the scancel, so Monarch never sees a broken link.
        order: list[str] = []
        proc_mesh = MagicMock()

        async def stop_monitor() -> None:
            order.append("stop")

        proc_mesh.stop.side_effect = stop_monitor
        del proc_mesh.host_mesh  # no attached host mesh in this fixture
        chunk.cancel_task.side_effect = lambda *a, **k: order.append("cancel")
        slot_key = f"{chunk.array_job_id}_0"
        mgr._slot_to_monitor_mesh[slot_key] = proc_mesh
        mgr._mesh_name_to_slot[make_name("heartbeat", slot_key)] = slot_key

        run_endpoint(mgr, "scale_by", -1)

        proc_mesh.stop.assert_called_once()
        assert order == ["stop", "cancel"], "monitor mesh must stop before scancel"
        # Supervision state for the slot is fully dropped.
        assert slot_key not in mgr._slot_to_monitor_mesh
        assert make_name("heartbeat", slot_key) not in mgr._mesh_name_to_slot

    def test_stop_monitor_meshes_tolerates_stop_failure(self):
        mgr, _chunk = _make_manager(["node-001"])
        proc_mesh = MagicMock()

        async def fail_stop() -> None:
            raise RuntimeError("link already gone")

        proc_mesh.stop.side_effect = fail_stop
        del proc_mesh.host_mesh
        mgr._slot_to_monitor_mesh["99999_0"] = proc_mesh
        mgr._mesh_name_to_slot[make_name("heartbeat", "99999_0")] = "99999_0"

        # Must not raise even when stop() fails; state is still dropped.
        asyncio.run(mgr._stop_monitor_meshes("99999_0"))

        assert "99999_0" not in mgr._slot_to_monitor_mesh
        assert make_name("heartbeat", "99999_0") not in mgr._mesh_name_to_slot

    def test_stop_monitor_meshes_awaits_proc_then_host(self):
        mgr, _chunk = _make_manager(["node-001"])
        order: list[str] = []
        proc_mesh = MagicMock()
        host_mesh = MagicMock()
        proc_mesh.host_mesh = host_mesh
        mgr._slot_to_monitor_mesh["99999_0"] = proc_mesh

        async def stop_proc() -> None:
            order.append("wait:proc")

        async def stop_host() -> None:
            order.append("wait:host")

        proc_mesh.stop.side_effect = lambda: order.append("stop:proc") or stop_proc()
        host_mesh.stop.side_effect = lambda: order.append("stop:host") or stop_host()

        asyncio.run(mgr._stop_monitor_meshes("99999_0"))

        assert order == ["stop:proc", "wait:proc", "stop:host", "wait:host"]
        assert "99999_0" not in mgr._slot_to_monitor_mesh

    def test_scale_down_too_many_raises(self):
        mgr, _ = _make_manager(["node-001"])
        with pytest.raises(RuntimeError, match="only 1 idle"):
            run_endpoint(mgr, "scale_by", -2)

    def test_scale_down_zero_raises(self):
        mgr, _ = _make_manager(["node-001"])
        with pytest.raises(ValueError, match="n must be non-zero"):
            run_endpoint(mgr, "scale_by", 0)


class TestScaleUp:
    def test_scale_up_submits_chunk(self):
        mgr, _ = _make_manager(["node-001"])
        mgr._submit_chunk = MagicMock()

        run_endpoint(mgr, "scale_by", 2)

        mgr._submit_chunk.assert_called_once_with(2)


class TestShutdown:
    def test_shutdown_cancels_all_chunks(self):
        mgr, chunk = _make_manager(["node-001", "node-002"])
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "shutdown")

        chunk.cancel_all.assert_called_once()
        assert len(mgr._chunks) == 0
        assert len(mgr._available) == 0
        assert len(mgr._handles) == 0

    def test_shutdown_tolerates_cancel_failure(self):
        mgr, chunk = _make_manager(["node-001"])
        chunk.cancel_all = MagicMock(side_effect=RuntimeError("scancel failed"))

        run_endpoint(mgr, "shutdown")  # should not raise

        assert len(mgr._chunks) == 0

    def test_shutdown_after_acquire(self):
        mgr, chunk = _make_manager(["node-001", "node-002"])
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "acquire", 1)
        run_endpoint(mgr, "shutdown")

        chunk.cancel_all.assert_called_once()
        assert len(mgr._handles) == 0


class TestHandleMetadata:
    def test_metadata_contains_slurm_array_info(self):
        mgr, _ = _make_manager(["node-001"])
        handle = run_endpoint(mgr, "acquire", 1)

        assert handle.metadata["host_to_port"]["node-001"] == 22222
        assert handle.metadata["host_to_array_job_id"]["node-001"] == "99999"
        assert "host_to_task_id" in handle.metadata


class TestHandleAddresses:
    def test_addresses_populated_on_acquire(self):
        mgr, _ = _make_manager(["node-001"])
        handle = run_endpoint(mgr, "acquire", 1)

        assert len(handle.addresses) == 1
        assert handle.addresses[0] == "tcp://node-001:22222"

    def test_addresses_multiple_hosts(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        handle = run_endpoint(mgr, "acquire", 2)

        assert len(handle.addresses) == 2
        assert set(handle.addresses) == {
            "tcp://node-001:22222",
            "tcp://node-002:22222",
        }

    def test_addresses_default_empty_on_dataclass(self):
        from agentbox.actors.resource_managers.base import ResourceHandle

        handle = ResourceHandle(handle_id="test", hostnames=["h1"])
        assert handle.addresses == []


class TestSlurmAllocation:
    def test_allocation_is_frozen(self):
        alloc = SlurmAllocation()
        with pytest.raises(AttributeError):
            alloc.cpus_per_task = 16  # type: ignore[misc]

    def test_allocation_defaults(self):
        alloc = SlurmAllocation()
        assert alloc.cpus_per_task == 4
        assert alloc.gpus_per_node == 0
        assert alloc.time_limit == "04:00:00"
        assert alloc.qos is None
        assert alloc.account is None
        assert alloc.exclusive is False
        assert alloc.whole_node is False
        assert alloc.extra_slurm_args == ()
        assert alloc.extra_srun_args == ()


class TestWaitForPool:
    def _make_mock_manager(self, available: int = 1, has_poll_pending: bool = False):
        """Create a mock manager that responds to pool_status.choose().get()."""
        mgr = MagicMock()
        status = PoolStatus(available=available, in_use=0, total=available)
        mgr.pool_status.choose.return_value.get.return_value = status

        if not has_poll_pending:
            del mgr.poll_pending

        return mgr

    def test_returns_immediately_when_available(self):
        mgr = self._make_mock_manager(available=2)
        status = wait_for_pool(mgr, min_available=1, timeout=5.0)

        assert status.available >= 1
        assert isinstance(status, PoolStatus)

    def test_timeout_raises(self):
        mgr = self._make_mock_manager(available=0)
        with pytest.raises(TimeoutError, match="0/2 workers available"):
            wait_for_pool(mgr, min_available=2, timeout=0.1, poll_interval=0.05)

    def test_default_timeout_waits_indefinitely(self):
        mgr = self._make_mock_manager(available=0)
        mgr.pool_status.choose.return_value.get.side_effect = [
            PoolStatus(available=0, in_use=0, total=1),
            PoolStatus(available=1, in_use=0, total=1),
        ]

        with patch("agentbox.actors.resource_managers.base.time.sleep") as sleep:
            status = wait_for_pool(
                mgr,
                min_available=1,
                poll_interval=0.25,
            )

        assert status.available == 1
        sleep.assert_called_once_with(0.25)

    def test_rejects_negative_timeout(self):
        mgr = self._make_mock_manager(available=0)

        with pytest.raises(ValueError, match="timeout must be non-negative"):
            wait_for_pool(mgr, timeout=-1)

    def test_rejects_nan_timeout(self):
        mgr = self._make_mock_manager(available=0)

        with pytest.raises(ValueError, match="NaN is not supported"):
            wait_for_pool(mgr, timeout=float("nan"))

        mgr.pool_status.choose.assert_not_called()

    def test_calls_poll_pending_when_present(self):
        mgr = self._make_mock_manager(available=1, has_poll_pending=True)
        wait_for_pool(mgr, min_available=1, timeout=5.0)

        mgr.poll_pending.choose.assert_called()


class TestAwaitForPool:
    def _make_mock_manager(self, available: int = 1, has_poll_pending: bool = False):
        """Create a mock manager with awaitable resource-manager endpoints."""
        mgr = MagicMock()
        status = PoolStatus(available=available, in_use=0, total=available)
        mgr.pool_status.choose = AsyncMock(return_value=status)

        if has_poll_pending:
            mgr.poll_pending.choose = AsyncMock()
        else:
            del mgr.poll_pending

        return mgr

    async def test_returns_immediately_when_available(self):
        mgr = self._make_mock_manager(available=2)

        status = await await_for_pool(mgr, min_available=1, timeout=5.0)

        assert status.available >= 1
        assert isinstance(status, PoolStatus)

    async def test_default_timeout_waits_asynchronously(self):
        mgr = self._make_mock_manager(available=0)
        mgr.pool_status.choose = AsyncMock(
            side_effect=[
                PoolStatus(available=0, in_use=0, total=1),
                PoolStatus(available=1, in_use=0, total=1),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            status = await await_for_pool(
                mgr,
                min_available=1,
                poll_interval=0.25,
            )

        assert status.available == 1
        sleep.assert_awaited_once_with(0.25)

    @pytest.mark.parametrize(
        ("timeout", "message"),
        [
            pytest.param(-1, "timeout must be non-negative", id="negative"),
            pytest.param(float("nan"), "NaN is not supported", id="nan"),
        ],
    )
    async def test_rejects_invalid_timeout_before_polling(
        self,
        timeout: float,
        message: str,
    ) -> None:
        mgr = self._make_mock_manager(available=0)

        with pytest.raises(ValueError, match=message):
            await await_for_pool(mgr, timeout=timeout)

        mgr.pool_status.choose.assert_not_awaited()

    async def test_indefinite_wait_is_cancellable(self):
        mgr = self._make_mock_manager(available=0)
        pool_polled = asyncio.Event()

        async def empty_pool_status() -> PoolStatus:
            pool_polled.set()
            return PoolStatus(available=0, in_use=0, total=0)

        mgr.pool_status.choose = AsyncMock(side_effect=empty_pool_status)
        wait_task = asyncio.create_task(await_for_pool(mgr, poll_interval=3600))
        await asyncio.wait_for(pool_polled.wait(), timeout=1)
        await asyncio.sleep(0)

        wait_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(wait_task, timeout=1)


class FakeMeshFailure:
    """Minimal stand-in for monarch MeshFailure used in supervision tests."""

    def __init__(self, name: str) -> None:
        self.mesh_name = name

    def report(self) -> str:
        return f"simulated failure of {self.mesh_name}"


class TestSupervision:
    def test_registration_scopes_heartbeat_to_owning_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr, _ = _make_manager(["node-001"])
        mgr._fault_scope = "mgr1111111111111111"
        host_mesh = MagicMock()
        proc_mesh = host_mesh.spawn_procs.return_value
        attach = MagicMock(return_value=host_mesh)
        monkeypatch.setattr(
            "monarch._src.actor.bootstrap.attach_to_workers",
            attach,
        )

        mgr._register_worker_for_supervision("99999_0", ["tcp://node-001:22222"])

        mesh_name = make_scoped_name(
            "heartbeat",
            mgr._fault_scope,
            "99999_0",
        )
        attach.assert_called_once_with(
            name=mesh_name,
            ca="trust_all_connections",
            workers=["tcp://node-001:22222"],
        )
        host_mesh.spawn_procs.assert_called_once_with(name=mesh_name)
        proc_mesh.spawn.assert_called_once()
        assert mgr._mesh_name_to_slot[mesh_name] == "99999_0"

    def test_supervise_evicts_available_slot(self):
        mgr, _ = _make_manager(["node-001"])
        slot_key = mgr._available[0]
        mesh_name = make_name("heartbeat", slot_key)
        mgr._mesh_name_to_slot[mesh_name] = slot_key

        result = mgr.__supervise__(FakeMeshFailure(mesh_name))

        assert result is True
        assert slot_key not in mgr._available

    def test_supervise_force_releases_in_use_slot(self):
        mgr, _ = _make_manager(["node-001"])
        slot_key = mgr._available[0]
        handle = run_endpoint(mgr, "acquire", 1)
        assert slot_key not in mgr._available

        mesh_name = make_name("heartbeat", slot_key)
        mgr._mesh_name_to_slot[mesh_name] = slot_key

        result = mgr.__supervise__(FakeMeshFailure(mesh_name))

        assert result is True
        assert slot_key not in mgr._available
        # Handle should be removed once its last slot is gone.
        assert handle.handle_id not in mgr._handles

    def test_supervise_matches_suffixed_mesh_name(self):
        # Real faults carry the actor name with a Monarch "-<rand>" suffix; the
        # handler must still match the registered (un-suffixed) mesh name.
        mgr, _ = _make_manager(["node-001"])
        slot_key = mgr._available[0]
        mgr._mesh_name_to_slot[make_name("heartbeat", slot_key)] = slot_key

        result = mgr.__supervise__(
            FakeMeshFailure(make_name("heartbeat", slot_key) + "-RaNd42")
        )

        assert result is True
        assert slot_key not in mgr._available

    def test_supervise_unknown_mesh_returns_false(self):
        mgr, _ = _make_manager(["node-001"])

        result = mgr.__supervise__(FakeMeshFailure("unknown_mesh_xyz"))

        assert result is False
        assert len(mgr._available) == 1  # unchanged

    def test_report_failure_evicts_slot(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        slot_key = mgr._available[-1]

        run_endpoint(mgr, "report_failure", slot_key, "test connection drop")

        assert slot_key not in mgr._available
        assert len(mgr._available) == 1

    def test_report_failure_evicts_handle(self):
        mgr, _ = _make_manager(["node-001", "node-002"])
        handle = run_endpoint(mgr, "acquire", 2)
        assert len(mgr._available) == 0
        assert handle.handle_id in mgr._handles

        run_endpoint(mgr, "report_failure", handle.handle_id, "host unreachable")

        assert handle.handle_id not in mgr._handles
        assert len(mgr._available) == 0

    def test_on_worker_failure_hook_called(self):
        class TrackingManager(SlurmResourceManager):
            def __init__(self) -> None:
                # deliberately skip SlurmResourceManager.__init__
                ResourceManager.__init__(self)
                self._slot_to_host = {}
                self._slot_to_chunk = {}
                self._slot_to_task_id = {}
                self._slot_to_port = {}
                self._chunks = []
                self.failure_calls: list[tuple[str, Any]] = []

            def _on_worker_failure(self, slot_key: str, failure: Any) -> None:
                self.failure_calls.append((slot_key, failure))

        mgr = TrackingManager()
        slot_key = "local_0"
        mgr._available.append(slot_key)
        mgr._slot_to_host[slot_key] = "localhost"
        mgr._slot_to_port[slot_key] = 12345
        mesh_name = make_name("heartbeat", slot_key)
        mgr._mesh_name_to_slot[mesh_name] = slot_key

        failure = FakeMeshFailure(mesh_name)
        mgr.__supervise__(failure)

        assert len(mgr.failure_calls) == 1
        assert mgr.failure_calls[0][0] == slot_key
        assert mgr.failure_calls[0][1] is failure


class TestPoolExhaustedErrorPickle:
    def test_pickle_preserves_attributes(self):
        """PoolExhaustedError must survive a pickle/unpickle cycle with all attributes intact.

        Monarch serialises exceptions via pickle; without __reduce__ the default
        args-tuple form passes only the message string to __init__, which requires
        two positional ints and would raise TypeError on deserialisation.
        """
        import pickle

        exc = PoolExhaustedError(requested=7, available=3)
        exc2 = pickle.loads(pickle.dumps(exc))

        assert exc2.requested == 7
        assert exc2.available == 3
        assert str(exc2) == "Requested 7 hosts but only 3 available"
        assert isinstance(exc2, RuntimeError)
