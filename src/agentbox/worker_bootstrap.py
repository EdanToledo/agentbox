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

"""Worker bootstrap with dynamic port allocation via HTTP registration.

Invoked as ``python -m agentbox.worker_bootstrap`` from the sbatch script.
Picks an OS-assigned free port, starts the Monarch worker loop on it in a
background thread, waits for it to accept TCP connections, and only then
registers the port with the coordinator.  This prevents the coordinator from
trying to connect before the worker is ready.

Picking a port and then handing it to Monarch to bind is an unavoidable
time-of-check/time-of-use gap (Monarch's worker loop takes an address string
and cannot report back an OS-assigned ephemeral port).  We make it safe rather
than racy by treating a bind failure as authoritative: if the chosen port was
taken between probe and bind, Monarch's loop raises (recorded in
``worker_error``); readiness is reported only when the worker thread is healthy
*and* listening, and a bind conflict triggers a retry with a fresh port.  This
ensures we never register a port a foreign process is squatting on.
"""

from __future__ import annotations

import errno
import ipaddress
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from typing import Callable


logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_READY_POLL_INTERVAL = 0.5
_READY_TIMEOUT = 60.0
_REGISTER_RETRIES = 3
_REGISTER_RETRY_DELAY = 1.0
_REGISTER_TIMEOUT = 30.0


def _configure_worker_logging() -> None:
    """Install the worker log format before any configuration/setup output."""

    logging.basicConfig(
        format="[worker] %(message)s",
        level=logging.DEBUG,
        stream=sys.stderr,
    )


def _check_expected_torchmonarch_version() -> None:
    """Fail fast when the worker's torchmonarch does not match the client."""

    expected = os.environ.get("AGENTBOX_EXPECTED_TORCHMONARCH_VERSION")
    if not expected:
        return
    try:
        actual = version("torchmonarch")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "AGENTBOX_EXPECTED_TORCHMONARCH_VERSION is set but torchmonarch is not installed on the worker"
        ) from exc
    if actual != expected:
        raise RuntimeError(
            f"torchmonarch version mismatch: client={expected!r} worker={actual!r}"
        )


def _probe_free_port() -> int:
    """Return an OS-assigned free TCP port (the socket is closed before return)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _is_address_in_use(err: BaseException | None) -> bool:
    """Whether *err* indicates the chosen port was already bound."""
    if err is None:
        return False
    if isinstance(err, OSError) and err.errno == errno.EADDRINUSE:
        return True
    return "address already in use" in str(err).lower()


def _await_worker_ready(
    host: str,
    port: int,
    get_error: Callable[[], BaseException | None],
    is_alive: Callable[[], bool],
    timeout: float = _READY_TIMEOUT,
) -> bool:
    """Return ``True`` once the worker is confirmed listening and healthy.

    Returns ``False`` if the worker thread errored (e.g. the port was taken
    between probe and bind), exited, or did not become ready within *timeout* —
    the caller inspects ``get_error()`` to decide whether to retry or raise.

    The worker-thread error is checked *before* (and again after) the TCP probe:
    a plain connect cannot tell our worker apart from a foreign process squatting
    on the port, so the bind failure recorded by the worker thread — not the
    connect — is the source of truth for ownership.
    """
    deadline = time.monotonic() + timeout
    first_failure_logged = False
    while time.monotonic() < deadline:
        if get_error() is not None or not is_alive():
            return False
        try:
            with socket.create_connection((host, port), timeout=2):
                pass
        except OSError as e:
            if not first_failure_logged:
                logger.debug("Port %s:%d not ready yet: %s", host, port, e)
                first_failure_logged = True
            time.sleep(_READY_POLL_INTERVAL)
            continue
        # Connectable.  Re-check the worker didn't just error/exit so we never
        # treat a foreign squatter's socket as our ready worker.
        if get_error() is not None or not is_alive():
            return False
        return True
    return False


def _register_with_retry(
    url: str, data: bytes, *, retries: int = _REGISTER_RETRIES
) -> None:
    """POST the registration to the coordinator, retrying on transient errors.

    The coordinator is always an internal node addressed by hostname/IP, so the
    POST must bypass any ``http_proxy``/``HTTP_PROXY`` set in the environment —
    an internal hostname is typically not covered by ``no_proxy`` (which usually
    lists IP CIDRs), so a default ``urlopen`` would route the request through the
    proxy and time out.  Use an opener with an empty ``ProxyHandler`` to force a
    direct connection.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: OSError | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            opener.open(req, timeout=_REGISTER_TIMEOUT)  # noqa: S310
            return
        except OSError as e:
            last_error = e
            if attempt < retries:
                logger.warning(
                    "Registration POST to %s failed (%s); retry %d/%d",
                    url,
                    e,
                    attempt,
                    retries,
                )
                time.sleep(_REGISTER_RETRY_DELAY)
    assert last_error is not None
    raise last_error


def _run_worker_attempt(
    jid: str, tid: str, hostname: str, reg_addr: str, attempt: int
) -> bool:
    """Probe a port, start the worker, and register one time.

    Returns ``True`` once the worker is listening and registered (the caller
    should stop), or ``False`` if the chosen port was taken before bind and a
    retry with a fresh port is warranted.  Raises on any other failure.

    This lives in its own function (rather than inline in :func:`main`'s retry
    loop) so the ``_run_worker`` closure and the readiness ``lambda`` bind
    ``addr``/``worker_error`` as ordinary locals instead of late-bound loop
    variables.
    """
    port = _probe_free_port()
    addr = f"tcp://{hostname}:{port}"
    logger.info("%s job=%s task=%s binding %s", hostname, jid, tid, addr)

    # Start the worker loop in a background thread so we can wait for it to
    # be listening (and detect a bind failure) before registering.
    worker_error: BaseException | None = None

    def _run_worker() -> None:
        nonlocal worker_error
        try:
            from monarch.actor import run_worker_loop_forever

            run_worker_loop_forever(address=addr, ca="trust_all_connections")
        except Exception as e:
            worker_error = e

    t = threading.Thread(target=_run_worker, daemon=True)
    t.start()

    ready = _await_worker_ready(hostname, port, lambda: worker_error, t.is_alive)
    if not ready:
        # The worker failed to come up.  If the port was taken between probe
        # and bind, retry with a fresh one (the thread has already exited
        # with the error, so there is nothing to stop).
        if _is_address_in_use(worker_error) and attempt < _MAX_RETRIES:
            logger.warning(
                "Port %d was taken before bind (%s); retrying with a new port (%d/%d)",
                port,
                worker_error,
                attempt,
                _MAX_RETRIES,
            )
            return False
        if worker_error is not None:
            raise worker_error
        raise TimeoutError(
            f"Worker on {hostname}:{port} did not become ready within {_READY_TIMEOUT:.0f}s"
        )

    logger.info("%s job=%s task=%s listening on %s", hostname, jid, tid, addr)

    # Now that the worker is confirmed listening and healthy, register with
    # the coordinator.
    url = f"http://{reg_addr}/{jid}/{tid}"
    data = json.dumps({"hostname": hostname, "port": port}).encode()
    _register_with_retry(url, data)

    logger.info("%s job=%s task=%s registered", hostname, jid, tid)

    # Block on the worker thread forever (the worker serves until killed).
    t.join()
    return True


def _run_slurm_worker_from_env() -> None:
    from agentbox.port_server import routable_ipv4

    jid = os.environ["SLURM_ARRAY_JOB_ID"]
    tid = os.environ["SLURM_ARRAY_TASK_ID"]
    # Bind an explicit IPv4 literal so the coordinator and worker cannot choose
    # different address families when resolving the same dual-stack hostname.
    hostname = os.environ.get("AGENTBOX_WORKER_HOSTNAME")
    if not hostname:
        hostname = str(routable_ipv4())
    reg_addr = os.environ["AGENTBOX_PORT_SERVER"]  # "coordinator_host:port"

    # Give this worker a unique Monarch log path so that multiple workers on
    # the same physical node do not clobber each other's monarch_log.log.
    # MONARCH_LOG_SUFFIX is read by hyperactor_telemetry at worker startup to
    # form: /tmp/${USER}/monarch_log_task_${SLURM_ARRAY_TASK_ID}.log
    # Only set if not already provided (e.g. by the sbatch script).
    os.environ.setdefault("MONARCH_LOG_SUFFIX", f"task_{tid}")

    for attempt in range(1, _MAX_RETRIES + 1):
        if _run_worker_attempt(jid, tid, hostname, reg_addr, attempt):
            break


def _format_uri_host(host: str) -> str:
    try:
        parsed = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return host
    if parsed.version == 6:
        return f"[{parsed}]"
    return str(parsed)


def run_worker(address: str) -> None:
    """Run a standalone Monarch worker at *address*.

    Local and legacy-Slurm launchers call this entry point so they receive the
    same preflight as workers started through :func:`main`.
    """

    _configure_worker_logging()
    _check_expected_torchmonarch_version()

    from monarch.actor import run_worker_loop_forever

    run_worker_loop_forever(address=address, ca="trust_all_connections")


def main() -> None:
    _configure_worker_logging()
    _check_expected_torchmonarch_version()
    _run_slurm_worker_from_env()


if __name__ == "__main__":
    main()
