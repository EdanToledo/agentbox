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

"""Tests for Machine and its metadata actor."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentbox.actors.machine import Machine


class TestMachine:
    def test_hostname_delegates_to_actor(self):
        """Machine.hostname() calls through to the actor mesh."""
        mock_proc_mesh = MagicMock()
        mock_actor_mesh = MagicMock()
        mock_proc_mesh.spawn.return_value = mock_actor_mesh
        mock_actor_mesh.hostname.choose.return_value.get.return_value = "worker-01"

        machine = Machine(proc_mesh=mock_proc_mesh)
        assert machine.hostname() == "worker-01"


class TestMachineRelease:
    """release()/close() must tear down meshes the Machine created — and only those."""

    def test_release_does_not_stop_caller_supplied_mesh(self):
        # Public constructor: caller owns the proc mesh, so release() must not stop it.
        mock_proc_mesh = MagicMock()
        machine = Machine(proc_mesh=mock_proc_mesh)
        task_pool = MagicMock()
        machine._task_pool = task_pool
        machine.release()

        task_pool.close.assert_called_once_with(drain=False)
        mock_proc_mesh.stop.assert_not_called()

    @staticmethod
    def _started_container(machine: Machine):
        from agentbox.actors.container import Container

        container = Container(machine, image="test.sif", instance_name="inst")
        actor = MagicMock()
        actor.create_staging_dir.choose.return_value.get.return_value = "/tmp/stage"
        container._actor_mesh = actor
        container.start()
        return container, actor

    def test_release_stops_container_left_running(self):
        # An Apptainer instance outlives the mesh; leaving it running would
        # carry it into the worker's next lease.
        machine = Machine(proc_mesh=MagicMock())
        container, actor = self._started_container(machine)

        machine.release()

        actor.stop_instance.choose.assert_called_once_with("inst", None)
        assert not container._running

    def test_release_skips_container_already_stopped(self):
        machine = Machine(proc_mesh=MagicMock())
        container, actor = self._started_container(machine)
        container.stop()

        machine.release()

        actor.stop_instance.choose.assert_called_once()

    def test_release_tolerates_container_stop_failure(self):
        proc_mesh = MagicMock()
        machine = Machine(proc_mesh=proc_mesh)
        _container, actor = self._started_container(machine)
        actor.stop_instance.choose.return_value.get.side_effect = RuntimeError("boom")
        task_pool = MagicMock()
        machine._task_pool = task_pool

        machine.release()  # must not raise

        task_pool.close.assert_called_once_with(drain=False)

    async def test_arelease_stops_container_left_running(self):
        from agentbox.actors.container import Container

        machine = Machine(proc_mesh=MagicMock())
        container = Container(machine, image="test.sif", instance_name="inst")
        actor = MagicMock()
        actor.create_staging_dir.choose = AsyncMock(return_value="/tmp/stage")
        actor.start_instance.choose = AsyncMock(return_value=None)
        actor.stop_instance.choose = AsyncMock(return_value=None)
        actor.cleanup_staging_dir.choose = AsyncMock(return_value=None)
        container._actor_mesh = actor
        await container.astart()

        await machine.arelease()

        actor.stop_instance.choose.assert_awaited_once_with("inst", None)
        assert not container._running

    def test_release_tolerates_task_pool_close_failure(self):
        proc_mesh = MagicMock()
        machine = Machine(proc_mesh=proc_mesh)
        task_pool = MagicMock()
        task_pool.close.side_effect = TimeoutError("command still running")
        machine._task_pool = task_pool

        machine.release()

        task_pool.close.assert_called_once_with(drain=False)
        proc_mesh.stop.assert_not_called()

    async def test_arelease_tolerates_task_pool_close_failure(self):
        proc_mesh = MagicMock()
        machine = Machine(proc_mesh=proc_mesh)
        task_pool = MagicMock()
        task_pool.aclose = AsyncMock(side_effect=TimeoutError("command still running"))
        machine._task_pool = task_pool

        await machine.arelease()

        task_pool.aclose.assert_awaited_once_with(drain=False)
        proc_mesh.stop.assert_not_called()

    def test_release_stops_owned_remote_mesh_and_host(self):
        # from_handle() records the connector's owned host_mesh as
        # machine._owned_host_mesh, and release() stops it explicitly.
        # proc_mesh.host_mesh is a SLICE (reference) and would raise
        # "cannot stop HostMesh that is a reference instead of owned" if used,
        # so release() must NOT consult it.
        mock_proc_mesh = MagicMock()
        owned_host = MagicMock()
        # proc_mesh.host_mesh remains a reference; release should NOT consult it.
        slice_host = MagicMock()
        mock_proc_mesh.host_mesh = slice_host
        machine = Machine(proc_mesh=mock_proc_mesh)
        machine._owns_proc_mesh = True  # as set by from_handle()
        machine._owned_host_mesh = owned_host  # as set by from_handle()

        machine.release()

        mock_proc_mesh.stop.assert_called_once()
        mock_proc_mesh.stop.return_value.get.assert_called_once_with(timeout=10.0)
        owned_host.stop.assert_called_once()
        owned_host.stop.return_value.get.assert_called_once_with(timeout=10.0)
        slice_host.stop.assert_not_called()

    def test_release_without_owned_host_stops_proc_only(self):
        # When no owned host mesh was recorded (e.g. caller built the Machine via
        # the public Machine(proc_mesh=...) constructor), release stops only the
        # proc mesh and never touches the slice/reference at mesh.host_mesh.
        mock_proc_mesh = MagicMock()
        slice_host = MagicMock()
        mock_proc_mesh.host_mesh = slice_host
        machine = Machine(proc_mesh=mock_proc_mesh)
        machine._owns_proc_mesh = True
        # _owned_host_mesh defaults to None.

        machine.release()

        mock_proc_mesh.stop.assert_called_once()
        slice_host.stop.assert_not_called()

    def test_release_is_idempotent(self):
        mock_proc_mesh = MagicMock()
        mock_proc_mesh.host_mesh = MagicMock()
        machine = Machine(proc_mesh=mock_proc_mesh)
        machine._owns_proc_mesh = True

        machine.release()
        machine.release()

        mock_proc_mesh.stop.assert_called_once()

    def test_release_swallows_stop_errors(self):
        mock_proc_mesh = MagicMock()
        mock_proc_mesh.host_mesh = MagicMock()
        mock_proc_mesh.stop.side_effect = RuntimeError("worker gone")
        machine = Machine(proc_mesh=mock_proc_mesh)
        machine._owns_proc_mesh = True

        machine.release()  # must not raise

    def test_stop_mesh_uses_requested_timeout(self):
        proc_mesh = MagicMock()
        host_mesh = MagicMock()

        Machine._stop_mesh(proc_mesh, host_mesh, timeout=0.25)

        proc_mesh.stop.return_value.get.assert_called_once_with(timeout=0.25)
        host_mesh.stop.return_value.get.assert_called_once_with(timeout=0.25)

    @pytest.mark.asyncio
    async def test_astop_mesh_uses_requested_timeout(self):
        proc_mesh = MagicMock()
        host_mesh = MagicMock()

        with patch(
            "agentbox.actors.machine.asyncio.wait_for", side_effect=[None, None]
        ) as wait_for:
            await Machine._astop_mesh(proc_mesh, host_mesh, timeout=0.25)

        assert wait_for.call_args_list == [
            ((proc_mesh.stop.return_value,), {"timeout": 0.25}),
            ((host_mesh.stop.return_value,), {"timeout": 0.25}),
        ]

    def test_from_handle_owns_mesh_and_release_tears_down(self):
        from agentbox.actors.connector import ConnectResult

        proc = MagicMock()
        owned_host = MagicMock()
        # connect() returns the owned host_mesh alongside the proc mesh.
        result = ConnectResult(proc_mesh=proc, host_mesh=owned_host)
        handle = MagicMock()
        handle.hostnames = ["node-1"]

        with patch("agentbox.actors.connector.connect", return_value=result):
            machine = Machine.from_handle(handle)

        assert machine._owns_proc_mesh is True
        assert machine._owned_host_mesh is owned_host
        machine.release()
        proc.stop.assert_called_once()
        owned_host.stop.assert_called_once()

    def test_close_and_context_exit_release_owned_mesh(self):
        proc = MagicMock()
        proc.host_mesh = MagicMock()
        with Machine(proc_mesh=proc) as machine:
            machine._owns_proc_mesh = True
        proc.stop.assert_called_once()  # __exit__ -> release()


class TestMachineConvenienceMethods:
    def test_task_pool_is_not_public(self):
        machine = Machine(proc_mesh=MagicMock())
        assert not hasattr(machine, "task_pool")

    def test_task_pool_configuration_is_not_accepted(self):
        with pytest.raises(TypeError, match="task_pool_size"):
            Machine(proc_mesh=MagicMock(), task_pool_size=1)  # type: ignore[call-arg]

    def test_internal_task_pool_uses_fixed_defaults(self):
        machine = Machine(proc_mesh=MagicMock())
        with patch("agentbox.tasks.pool._TaskPool") as constructor:
            machine._ensure_task_pool()

        assert constructor.call_args.kwargs["capacity"] == 16
        assert set(constructor.call_args.kwargs["handlers"]) == {"subprocess"}

    def test_task_pool_initialized_once_under_concurrent_first_use(self):
        machine = Machine(proc_mesh=MagicMock())
        task_pool = MagicMock()
        constructor_entered = threading.Event()
        second_constructor_entered = threading.Event()
        allow_constructor = threading.Event()
        call_count = 0
        call_count_lock = threading.Lock()

        def create_task_pool(
            _proc_mesh: Any,
            *,
            capacity: int,
            handlers: dict[str, Any],
        ) -> Any:
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                if call_count == 2:
                    second_constructor_entered.set()
            constructor_entered.set()
            assert allow_constructor.wait(timeout=10.0)
            return task_pool

        results: list[Any] = []

        def ensure_task_pool() -> None:
            results.append(machine._ensure_task_pool())

        with patch(
            "agentbox.tasks.pool._TaskPool",
            side_effect=create_task_pool,
        ) as constructor:
            first = threading.Thread(target=ensure_task_pool)
            first.start()
            assert constructor_entered.wait(timeout=10.0)
            second = threading.Thread(target=ensure_task_pool)
            second.start()
            try:
                assert not second_constructor_entered.wait(timeout=0.1)
            finally:
                allow_constructor.set()
                first.join(timeout=10.0)
                second.join(timeout=10.0)

        assert not first.is_alive()
        assert not second.is_alive()
        constructor.assert_called_once()
        assert results == [task_pool, task_pool]

    def test_task_pool_cannot_be_created_after_release(self):
        machine = Machine(proc_mesh=MagicMock())
        machine.release()

        with patch("agentbox.tasks.pool._TaskPool") as constructor:
            with pytest.raises(RuntimeError, match="Machine is closed"):
                machine._ensure_task_pool()

        constructor.assert_not_called()

    def test_from_handle(self):
        from agentbox.actors.connector import ConnectResult
        from agentbox.actors.resource_managers import ResourceHandle

        handle = ResourceHandle(
            handle_id="test-1",
            hostnames=["node-1"],
            addresses=["tcp://node-1:22222"],
        )

        with patch("agentbox.actors.connector.connect") as mock_connect:
            mock_proc_mesh = MagicMock()
            mock_connect.return_value = ConnectResult(
                proc_mesh=mock_proc_mesh, host_mesh=MagicMock()
            )

            machine = Machine.from_handle(handle)

        mock_connect.assert_called_once_with(handle, mesh_name=None)
        assert machine.proc_mesh is mock_proc_mesh
        assert machine.resource_handle_id == "test-1"
        assert machine.hostname() == "node-1"
