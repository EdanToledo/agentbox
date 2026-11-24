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

"""HTTP server for dynamic worker port registration.

Workers bind to port 0 (OS-assigned) and POST their ``(hostname, port)``
to this server.  The coordinator reads the registered port via
:meth:`PortRegistrationServer.get_port` or :meth:`try_get_port` instead
of computing a deterministic hash.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from ipaddress import IPv4Address


logger = logging.getLogger(__name__)


__all__ = ["PortRegistrationServer", "get_resolvable_hostname", "routable_ipv4"]


# Match ``login`` only as a delimited hostname component (e.g. ``login0``,
# ``host-login-1``, ``gpu.login``) — NOT as a substring of an unrelated name
# like ``mylogins-gpu-04`` or ``clogin-worker``.
_LOGIN_NODE_RE = re.compile(r"(?:^|[-.])login\d*(?:[-.]|$)")

# TEST-NET-1 is reserved for documentation.
# https://datatracker.ietf.org/doc/html/rfc5737
_ROUTE_LOOKUP_TARGET = ("192.0.2.1", 9)


def routable_ipv4() -> IPv4Address:
    """Return this node's routable IPv4 address.

    Both ends of a Monarch channel must agree on the address family.  If they
    resolve a dual-stack hostname independently, one can select IPv4 while the
    other binds IPv6.  Asking the kernel which source address it would use for
    an off-box IPv4 route avoids that ambiguity.  The destination is an IANA
    documentation-only address; no packet is sent and no remote service is
    contacted.

    When the route lookup is not permitted, the first non-loopback IPv4
    address of this host's name is used instead.

    Raises:
        RuntimeError: If neither the route lookup nor name resolution yields
            a non-loopback address.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # UDP connect performs a local route lookup without sending a datagram.
            sock.connect(_ROUTE_LOOKUP_TARGET)
            return IPv4Address(sock.getsockname()[0])
    except OSError as route_error:
        # Some hosts forbid UDP connect outright; fall back to the IPv4
        # address this host's name resolves to.
        try:
            infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        except OSError:
            infos = []
        for *_, sockaddr in infos:
            address = IPv4Address(sockaddr[0])
            if not address.is_loopback:
                return address
        raise RuntimeError(
            f"Failed to determine this node's routable IPv4 address: {route_error}"
        ) from route_error


def _is_login_node(hostname: str) -> bool:
    """Whether *hostname* looks like a Slurm login node (delimited ``login`` token)."""
    return _LOGIN_NODE_RE.search(hostname.lower()) is not None


def get_resolvable_hostname() -> str:
    """Return a hostname or IP that is resolvable from compute nodes.

    Login node hostnames (e.g. ``alice-login-0``) are often internal
    and not DNS-resolvable from compute nodes.  When detected, we fall
    back to the machine's routable IP address.
    """
    hostname = socket.gethostname()

    if _is_login_node(hostname):
        logger.info(
            "Detected login node hostname '%s', falling back to IP address", hostname
        )
        try:
            ip_address = routable_ipv4()
            logger.info(
                "Using IP address %s instead of hostname %s", ip_address, hostname
            )
            return str(ip_address)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get IP address for login node {hostname}: {e}"
            ) from e

    try:
        # This is a best-effort local-hostname resolvability check, not
        # application DNS discovery.
        # ast-grep-ignore: python/python-dns-deps
        socket.gethostbyname(hostname)
        return hostname
    except socket.gaierror:
        logger.info("Hostname '%s' is not DNS-resolvable, using 'localhost'", hostname)
        return "localhost"


class PortRegistrationServer:
    """Collects (hostname, port) registrations from Slurm workers via HTTP.

    Starts an HTTP server on port 0 (OS-assigned) in a daemon thread.
    Workers POST their bound port; the coordinator reads it via
    :meth:`get_port` or :meth:`try_get_port`.
    """

    def __init__(self) -> None:
        self._ports: dict[
            str, tuple[str, int]
        ] = {}  # "job_id/task_id" → (hostname, port)
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

        server = self  # closure reference for Handler

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Parse defensively: a malformed registration must return a clear
                # 400 (and be retried by the worker) rather than raising in the
                # handler — which would drop the connection and leave the
                # coordinator blocked on get_port() for the full timeout.
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(content_length) if content_length > 0 else b""
                    body = json.loads(raw)
                    hostname = body["hostname"]
                    port = body["port"]
                    if not isinstance(port, int) or isinstance(port, bool):
                        raise ValueError(f"port must be an int, got {port!r}")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                    logger.warning(
                        "Rejecting malformed registration POST to %s: %s", self.path, e
                    )
                    self.send_response(400)
                    self.end_headers()
                    return

                key = self.path.strip("/")  # "job_id/task_id"
                with server._lock:
                    server._ports[key] = (hostname, port)
                    logger.debug("Port registered: %s → %s:%s", key, hostname, port)
                    evt = server._events.get(key)
                    if evt:
                        evt.set()
                self.send_response(200)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass  # suppress noisy per-request logging

        self._httpd = HTTPServer(("", 0), _Handler)
        self._host = get_resolvable_hostname()
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Port registration server started on %s:%d", self._host, self._port)

    @property
    def address(self) -> str:
        """Return the ``host:port`` address workers should POST to."""
        return f"{self._host}:{self._port}"

    def get_port(
        self, job_id: str, task_id: int, timeout: float = 60.0
    ) -> tuple[str, int] | None:
        """Block until the worker registers, or return ``None`` on timeout."""
        key = f"{job_id}/{task_id}"
        with self._lock:
            if key in self._ports:
                return self._ports[key]
            evt = self._events.setdefault(key, threading.Event())
        logger.debug("Waiting for port registration: %s (timeout=%.0fs)", key, timeout)
        if evt.wait(timeout):
            with self._lock:
                return self._ports.get(key)
        logger.warning(
            "Timed out waiting for port registration: %s after %.0fs", key, timeout
        )
        return None

    def try_get_port(self, job_id: str, task_id: int) -> tuple[str, int] | None:
        """Non-blocking check.  Returns ``(hostname, port)`` or ``None``."""
        key = f"{job_id}/{task_id}"
        with self._lock:
            return self._ports.get(key)

    def shutdown(self) -> None:
        """Stop the HTTP server and its background thread."""
        logger.info("Shutting down port registration server")
        self._httpd.shutdown()
        logger.info("Port registration server stopped")
