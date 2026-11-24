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

"""Tests for worker_bootstrap module."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from ipaddress import IPv4Address
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest


class _RegistrationHandler(BaseHTTPRequestHandler):
    """Captures POST registrations for testing."""

    registrations: dict[str, dict] = {}

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        key = self.path.strip("/")
        _RegistrationHandler.registrations[key] = body
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture()
def mock_registration_server():
    """Start a mock HTTP server that records registrations."""
    _RegistrationHandler.registrations = {}
    server = HTTPServer(("", 0), _RegistrationHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"localhost:{port}"
    server.shutdown()


class TestPortProbe:
    def test_probe_free_port_returns_valid_port(self):
        """_probe_free_port() yields a usable ephemeral port."""
        from agentbox.worker_bootstrap import _probe_free_port

        port = _probe_free_port()
        assert 1024 < port < 65536


class TestAwaitWorkerReady:
    """Readiness must be driven by the worker-thread error, not a blind connect,
    so a foreign process squatting on the port is never mistaken for our worker."""

    def test_bind_error_returns_false_without_probing(self):
        from agentbox.worker_bootstrap import _await_worker_ready

        # Worker thread already failed to bind — must report not-ready (so the
        # caller retries) without ever connecting to whatever holds the port.
        def _connect_should_not_be_called(*a, **k):
            raise AssertionError("must not probe once the worker has errored")

        with patch(
            "socket.create_connection", side_effect=_connect_should_not_be_called
        ):
            ready = _await_worker_ready(
                "host",
                1234,
                get_error=lambda: OSError("Address already in use"),
                is_alive=lambda: False,
                timeout=1.0,
            )
        assert ready is False

    def test_ready_when_connectable_and_healthy(self):
        from agentbox.worker_bootstrap import _await_worker_ready

        with patch("socket.create_connection") as conn:
            conn.return_value.__enter__.return_value = object()
            ready = _await_worker_ready(
                "host", 1234, get_error=lambda: None, is_alive=lambda: True, timeout=1.0
            )
        assert ready is True

    def test_times_out_when_never_connectable(self):
        from agentbox.worker_bootstrap import _await_worker_ready

        with patch("socket.create_connection", side_effect=OSError("refused")):
            ready = _await_worker_ready(
                "host", 1234, get_error=lambda: None, is_alive=lambda: True, timeout=0.3
            )
        assert ready is False


class TestRegisterWithRetry:
    def test_retries_then_succeeds(self):
        from agentbox.worker_bootstrap import _register_with_retry

        calls = {"n": 0}

        def _urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 2:
                raise OSError("connection refused")
            return MagicMock()

        with (
            patch("urllib.request.OpenerDirector.open", side_effect=_urlopen),
            patch("time.sleep"),
        ):
            _register_with_retry("http://x/1/0", b"{}")
        assert calls["n"] == 2

    def test_raises_after_exhausting_retries(self):
        from agentbox.worker_bootstrap import _register_with_retry

        with (
            patch("urllib.request.OpenerDirector.open", side_effect=OSError("down")),
            patch("time.sleep"),
        ):
            with pytest.raises(OSError, match="down"):
                _register_with_retry("http://x/1/0", b"{}", retries=2)


class TestWorkerBootstrapMain:
    @patch.dict(
        "os.environ",
        {
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTBOX_PORT_SERVER": "coordinator:1234",
        },
        clear=True,
    )
    def test_slurm_worker_uses_routable_ipv4(self):
        with (
            patch(
                "agentbox.port_server.routable_ipv4",
                return_value=IPv4Address("10.20.30.40"),
            ) as route,
            patch(
                "agentbox.worker_bootstrap._run_worker_attempt", return_value=True
            ) as attempt,
        ):
            from agentbox.worker_bootstrap import _run_slurm_worker_from_env

            _run_slurm_worker_from_env()

        route.assert_called_once_with()
        attempt.assert_called_once_with(
            "12345", "0", "10.20.30.40", "coordinator:1234", 1
        )

    @patch.dict(
        "os.environ",
        {
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTBOX_PORT_SERVER": "coordinator:1234",
            "AGENTBOX_WORKER_HOSTNAME": "worker.example.com",
        },
        clear=True,
    )
    def test_slurm_worker_prefers_explicit_hostname_override(self):
        with (
            patch("agentbox.port_server.routable_ipv4") as route,
            patch(
                "agentbox.worker_bootstrap._run_worker_attempt", return_value=True
            ) as attempt,
        ):
            from agentbox.worker_bootstrap import _run_slurm_worker_from_env

            _run_slurm_worker_from_env()

        route.assert_not_called()
        attempt.assert_called_once_with(
            "12345", "0", "worker.example.com", "coordinator:1234", 1
        )

    @patch.dict(
        "os.environ",
        {
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTBOX_WORKER_HOSTNAME": "127.0.0.1",
        },
    )
    def test_main_registers_and_starts_worker(self, mock_registration_server):
        import os

        os.environ["AGENTBOX_PORT_SERVER"] = mock_registration_server

        # _await_worker_ready is stubbed True (the worker-loop mock doesn't really
        # listen, and the real probe needs a resolvable hostname — not guaranteed
        # on every host).  We're testing the bootstrap logic (env handling +
        # registration POST), not real sockets.
        with (
            patch("monarch.actor.run_worker_loop_forever") as mock_loop,
            patch("agentbox.worker_bootstrap._await_worker_ready", return_value=True),
        ):
            from agentbox.worker_bootstrap import main

            main()

            mock_loop.assert_called_once()
            call_kwargs = mock_loop.call_args
            addr = call_kwargs.kwargs.get("address") or call_kwargs[0][0]
            assert addr.startswith("tcp://")
            assert "trust_all_connections" in str(call_kwargs)

        assert "12345/0" in _RegistrationHandler.registrations
        reg = _RegistrationHandler.registrations["12345/0"]
        assert "hostname" in reg
        assert "port" in reg
        assert isinstance(reg["port"], int)

    def test_main_requires_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            from agentbox.worker_bootstrap import main

            with pytest.raises(KeyError, match="SLURM_ARRAY_JOB_ID"):
                main()

    @patch.dict(
        "os.environ",
        {
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTBOX_WORKER_HOSTNAME": "127.0.0.1",
        },
    )
    def test_main_retries_on_address_in_use(self, mock_registration_server):
        import os

        os.environ["AGENTBOX_PORT_SERVER"] = mock_registration_server

        call_count = 0

        def fake_worker_loop(address: str, ca: str = "") -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Address already in use")
            # 3rd attempt: "running" worker — return so the t.join() in main()
            # completes and the loop breaks (a real worker would block forever).

        # Readiness is False while the worker keeps hitting the bind conflict,
        # True once it binds — driving main()'s retry loop without real
        # networking. Waiting for the thread to exit guarantees worker_error is
        # recorded before readiness is decided.
        def fake_ready(host, port, get_error, is_alive, timeout=60.0):
            deadline = time.monotonic() + 30
            while is_alive() and time.monotonic() < deadline:
                time.sleep(0.001)
            return get_error() is None

        with (
            patch(
                "monarch.actor.run_worker_loop_forever", side_effect=fake_worker_loop
            ),
            patch(
                "agentbox.worker_bootstrap._await_worker_ready",
                side_effect=fake_ready,
            ),
        ):
            from agentbox.worker_bootstrap import main

            main()

        assert call_count == 3

    @patch.dict(
        "os.environ",
        {
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTBOX_WORKER_HOSTNAME": "127.0.0.1",
        },
    )
    def test_main_does_not_register_when_worker_never_ready(
        self, mock_registration_server
    ):
        import os

        os.environ["AGENTBOX_PORT_SERVER"] = mock_registration_server

        # Worker never becomes ready and never errors -> main must raise
        # TimeoutError and must NOT register a port it doesn't own.
        with (
            patch("monarch.actor.run_worker_loop_forever"),
            patch(
                "agentbox.worker_bootstrap._await_worker_ready",
                return_value=False,
            ),
        ):
            from agentbox.worker_bootstrap import main

            with pytest.raises(TimeoutError):
                main()

        assert "12345/0" not in _RegistrationHandler.registrations
