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

"""Tests for agentbox.actors.connector (Connector protocol and DefaultConnector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from agentbox.actors.connector import (
    connect,
    Connector,
    ConnectResult,
    DefaultConnector,
    WorkerConnectionError,
)
from agentbox.actors.resource_managers import ResourceHandle


class TestConnectorProtocol:
    def test_default_connector_satisfies_protocol(self):
        assert isinstance(DefaultConnector(), Connector)

    def test_protocol_is_runtime_checkable(self):
        class MyConnector:
            def connect(self, handle: ResourceHandle) -> MagicMock:
                return MagicMock()

        assert isinstance(MyConnector(), Connector)


class TestDefaultConnector:
    def test_connect_uses_addresses(self):
        handle = ResourceHandle(
            handle_id="h1",
            hostnames=["node-1"],
            addresses=["tcp://node-1:22222"],
        )

        mock_host_mesh = MagicMock()
        mock_proc_mesh = MagicMock()
        mock_host_mesh.spawn_procs.return_value = mock_proc_mesh

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ) as mock_attach:
            result = DefaultConnector().connect(handle)

        mock_attach.assert_called_once_with(
            name="agentbox_worker",
            ca="trust_all_connections",
            workers=["tcp://node-1:22222"],
        )
        mock_host_mesh.spawn_procs.assert_called_once()
        assert isinstance(result, ConnectResult)
        assert result.proc_mesh is mock_proc_mesh
        # The OWNED host mesh (not proc_mesh.host_mesh) is carried for teardown.
        assert result.host_mesh is mock_host_mesh

    def test_connect_falls_back_to_hostnames(self):
        handle = ResourceHandle(
            handle_id="h2",
            hostnames=["ipc:///tmp/sock"],
            addresses=[],
        )

        mock_host_mesh = MagicMock()
        mock_host_mesh.spawn_procs.return_value = MagicMock()

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ) as mock_attach:
            DefaultConnector().connect(handle)

        mock_attach.assert_called_once_with(
            name="agentbox_worker",
            ca="trust_all_connections",
            workers=["ipc:///tmp/sock"],
        )

    def test_custom_mesh_name(self):
        handle = ResourceHandle(
            handle_id="h3",
            hostnames=["node-1"],
            addresses=["tcp://node-1:22222"],
        )

        mock_host_mesh = MagicMock()
        mock_host_mesh.spawn_procs.return_value = MagicMock()

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ) as mock_attach:
            DefaultConnector(mesh_name="custom").connect(handle)

        mock_attach.assert_called_once_with(
            name="custom",
            ca="trust_all_connections",
            workers=["tcp://node-1:22222"],
        )


class TestConnectFunction:
    def test_connect_convenience(self):
        handle = ResourceHandle(
            handle_id="h4",
            hostnames=["node-1"],
            addresses=["tcp://node-1:22222"],
        )

        mock_host_mesh = MagicMock()
        mock_proc_mesh = MagicMock()
        mock_host_mesh.spawn_procs.return_value = mock_proc_mesh

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ):
            result = connect(handle)

        assert isinstance(result, ConnectResult)
        assert result.proc_mesh is mock_proc_mesh
        assert result.host_mesh is mock_host_mesh

    def test_connect_with_custom_mesh_name(self):
        handle = ResourceHandle(
            handle_id="h5",
            hostnames=["node-1"],
            addresses=["tcp://node-1:22222"],
        )

        mock_host_mesh = MagicMock()
        mock_host_mesh.spawn_procs.return_value = MagicMock()

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ) as mock_attach:
            connect(handle, mesh_name="my_mesh")

        mock_attach.assert_called_once_with(
            name="my_mesh",
            ca="trust_all_connections",
            workers=["tcp://node-1:22222"],
        )


class TestWorkerConnectionError:
    def test_rejects_zero_connection_attempts(self):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            DefaultConnector(max_retries=0)

    def test_all_retries_exhausted_raises_worker_connection_error(self):
        handle = ResourceHandle(
            handle_id="h6",
            hostnames=["node-1"],
            addresses=["tcp://node-1:22222"],
        )

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(WorkerConnectionError) as exc_info:
                DefaultConnector(max_retries=2, retry_delay=0.0).connect(handle)

        err = exc_info.value
        assert err.addresses == ["tcp://node-1:22222"]
        assert err.attempts == 2
        assert isinstance(err.last_error, OSError)
        assert isinstance(err, RuntimeError)
        assert err.__cause__ is err.last_error


class TestConnectCleanup:
    """A failed spawn_procs must not leak the host mesh, and any error is retried."""

    def _handle(self) -> ResourceHandle:
        return ResourceHandle(
            handle_id="h", hostnames=["node-1"], addresses=["tcp://node-1:1"]
        )

    def test_spawn_procs_failure_tears_down_host_mesh_each_attempt(self):
        mock_host_mesh = MagicMock()
        # A non-OSError/ValueError failure also exercises the broadened except.
        mock_host_mesh.spawn_procs.side_effect = RuntimeError("spawn failed")

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ):
            with pytest.raises(WorkerConnectionError) as exc_info:
                DefaultConnector(max_retries=2, retry_delay=0.0).connect(self._handle())

        # The partially-connected host mesh was stopped (and reaped) on every attempt.
        assert mock_host_mesh.stop.call_count == 2
        assert mock_host_mesh.stop.return_value.get.call_count == 2
        # The RuntimeError was retried and wrapped rather than escaping.
        assert isinstance(exc_info.value.last_error, RuntimeError)

    def test_success_does_not_stop_host_mesh(self):
        mock_host_mesh = MagicMock()
        mock_host_mesh.spawn_procs.return_value = MagicMock()

        with patch(
            "agentbox.actors.connector.attach_to_workers",
            return_value=mock_host_mesh,
        ):
            DefaultConnector().connect(self._handle())

        mock_host_mesh.stop.assert_not_called()

    def test_attach_failure_has_no_mesh_to_stop(self):
        # attach_to_workers itself fails -> nothing to tear down, no crash.
        with patch(
            "agentbox.actors.connector.attach_to_workers",
            side_effect=RuntimeError("no route"),
        ):
            with pytest.raises(WorkerConnectionError):
                DefaultConnector(max_retries=1, retry_delay=0.0).connect(self._handle())
