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

"""Tests for PortRegistrationServer."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import urllib.request
from ipaddress import IPv4Address
from unittest.mock import MagicMock, patch

import pytest
from agentbox.port_server import _is_login_node, PortRegistrationServer, routable_ipv4


@pytest.fixture()
def server():
    srv = PortRegistrationServer()
    yield srv
    srv.shutdown()


def _local_address(server: PortRegistrationServer) -> str:
    """Return ``localhost:<port>`` for the server (avoids hostname DNS issues in tests)."""
    _, port_str = server.address.rsplit(":", 1)
    return f"localhost:{port_str}"


def _post_registration(
    address: str, job_id: str, task_id: int, hostname: str, port: int
) -> None:
    url = f"http://{address}/{job_id}/{task_id}"
    data = json.dumps({"hostname": hostname, "port": port}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)  # noqa: S310


class TestPortRegistrationServer:
    def test_server_binds_to_port(self, server: PortRegistrationServer):
        assert server.address
        host, port_str = server.address.rsplit(":", 1)
        assert host
        assert int(port_str) > 0

    def test_try_get_port_returns_none_before_registration(
        self, server: PortRegistrationServer
    ):
        result = server.try_get_port("99999", 0)
        assert result is None

    def test_post_and_try_get_port(self, server: PortRegistrationServer):
        _post_registration(_local_address(server), "12345", 0, "node-001", 55555)

        result = server.try_get_port("12345", 0)
        assert result == ("node-001", 55555)

    def test_post_and_get_port_blocking(self, server: PortRegistrationServer):
        addr = _local_address(server)

        # Post from a background thread after a short delay.
        def delayed_post():
            import time

            time.sleep(0.1)
            _post_registration(addr, "12345", 1, "node-002", 44444)

        t = threading.Thread(target=delayed_post)
        t.start()

        result = server.get_port("12345", 1, timeout=5.0)
        assert result == ("node-002", 44444)
        t.join()

    def test_get_port_returns_none_on_timeout(self, server: PortRegistrationServer):
        result = server.get_port("99999", 99, timeout=0.1)
        assert result is None

    def test_multiple_registrations(self, server: PortRegistrationServer):
        addr = _local_address(server)
        _post_registration(addr, "111", 0, "node-a", 10000)
        _post_registration(addr, "111", 1, "node-b", 10001)
        _post_registration(addr, "222", 0, "node-c", 20000)

        assert server.try_get_port("111", 0) == ("node-a", 10000)
        assert server.try_get_port("111", 1) == ("node-b", 10001)
        assert server.try_get_port("222", 0) == ("node-c", 20000)
        assert server.try_get_port("222", 1) is None

    def test_get_port_returns_immediately_if_already_registered(
        self, server: PortRegistrationServer
    ):
        _post_registration(_local_address(server), "12345", 0, "node-001", 55555)
        # Should return immediately, not block.
        result = server.get_port("12345", 0, timeout=0.01)
        assert result == ("node-001", 55555)


def _post_raw(server: PortRegistrationServer, path: str, body: bytes) -> int:
    """POST a raw body and return the HTTP status code."""
    _, port_str = server.address.rsplit(":", 1)
    conn = http.client.HTTPConnection("127.0.0.1", int(port_str), timeout=5)
    try:
        conn.request("POST", path, body=body)
        resp = conn.getresponse()
        resp.read()
        return resp.status
    finally:
        conn.close()


class TestPostRobustness:
    """A malformed registration must return 400 (so the worker retries) instead of
    raising in the handler and leaving the coordinator blocked on get_port()."""

    def test_malformed_json_returns_400(self, server: PortRegistrationServer):
        assert _post_raw(server, "/job/0", b"not json at all") == 400
        assert server.try_get_port("job", 0) is None

    def test_empty_body_returns_400(self, server: PortRegistrationServer):
        assert _post_raw(server, "/job/0", b"") == 400
        assert server.try_get_port("job", 0) is None

    def test_missing_hostname_returns_400(self, server: PortRegistrationServer):
        assert _post_raw(server, "/job/0", json.dumps({"port": 1234}).encode()) == 400

    def test_missing_port_returns_400(self, server: PortRegistrationServer):
        assert (
            _post_raw(server, "/job/0", json.dumps({"hostname": "n"}).encode()) == 400
        )

    def test_non_int_port_returns_400(self, server: PortRegistrationServer):
        assert (
            _post_raw(
                server, "/job/0", json.dumps({"hostname": "n", "port": "abc"}).encode()
            )
            == 400
        )
        assert server.try_get_port("job", 0) is None

    def test_valid_post_still_returns_200(self, server: PortRegistrationServer):
        assert (
            _post_raw(
                server, "/job/0", json.dumps({"hostname": "n", "port": 7}).encode()
            )
            == 200
        )
        assert server.try_get_port("job", 0) == ("n", 7)


class TestIsLoginNode:
    @pytest.mark.parametrize(
        "host",
        [
            "login",
            "login0",
            "login-1",
            "host-login-0",
            "gpu.login",
            "cluster-login01",
            "LOGIN-2",
            "alice-login-0",
        ],
    )
    def test_login_nodes_detected(self, host: str):
        assert _is_login_node(host)

    @pytest.mark.parametrize(
        "host",
        [
            "mylogins-gpu-04",
            "clogin-worker",
            "loginx-server",
            "alogin",
            "node-0",
            "compute-12",
            "gpu001",
        ],
    )
    def test_non_login_nodes_not_detected(self, host: str):
        assert not _is_login_node(host)


def test_routable_ipv4_uses_kernel_route_source_address():
    sock = MagicMock()
    sock.getsockname.return_value = ("10.20.30.40", 43210)
    context = MagicMock()
    context.__enter__.return_value = sock

    with patch("agentbox.port_server.socket.socket", return_value=context) as factory:
        address = routable_ipv4()

    assert address == IPv4Address("10.20.30.40")
    assert isinstance(address, IPv4Address)

    factory.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect.assert_called_once_with(("192.0.2.1", 9))


def _failing_route_socket():
    sock = MagicMock()
    sock.connect.side_effect = OSError("network unreachable")
    context = MagicMock()
    context.__enter__.return_value = sock
    return sock, patch("agentbox.port_server.socket.socket", return_value=context)


def test_routable_ipv4_falls_back_to_non_loopback_host_address():
    sock, patch_socket = _failing_route_socket()
    infos = [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.1.1", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.1.2.3", 0)),
    ]

    with (
        patch_socket,
        patch("agentbox.port_server.socket.gethostname", return_value="node"),
        patch(
            "agentbox.port_server.socket.getaddrinfo", return_value=infos
        ) as getaddrinfo,
    ):
        address = routable_ipv4()

    assert address == IPv4Address("10.1.2.3")
    getaddrinfo.assert_called_once_with("node", None, socket.AF_INET)
    sock.getsockname.assert_not_called()


@pytest.mark.parametrize(
    "getaddrinfo",
    [
        {"return_value": [(socket.AF_INET, 0, 0, "", ("127.0.0.1", 0))]},
        {"side_effect": OSError("unknown host")},
    ],
)
def test_routable_ipv4_describes_connect_error(getaddrinfo):
    sock, patch_socket = _failing_route_socket()

    with (
        patch_socket,
        patch("agentbox.port_server.socket.getaddrinfo", **getaddrinfo),
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to determine this node's routable IPv4 address: "
                "network unreachable"
            ),
        ) as exc_info:
            routable_ipv4()

    assert isinstance(exc_info.value.__cause__, OSError)
    sock.getsockname.assert_not_called()
