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

"""Connector protocol and default implementation for ResourceHandle → ProcMesh bridging.

Provides a pluggable abstraction for connecting to acquired resources
without backend-specific knowledge.  The :func:`connect` convenience
function covers the common case; custom :class:`Connector` implementations
can add TLS, retry logic, or alternative CA configurations.  Connecting
yields a :class:`ConnectResult` carrying both the :class:`ProcMesh` and the
owned :class:`~monarch.actor.HostMesh` that backs it, so callers can tear the
connection down cleanly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, TYPE_CHECKING

from agentbox.actors.mesh_schema import make_name
from agentbox.actors.resource_managers import ResourceHandle
from monarch._src.actor.bootstrap import attach_to_workers
from monarch.actor import ProcMesh


if TYPE_CHECKING:
    from monarch.actor import HostMesh


logger = logging.getLogger(__name__)

__all__ = [
    "ConnectResult",
    "Connector",
    "DefaultConnector",
    "WorkerConnectionError",
    "connect",
]


class WorkerConnectionError(RuntimeError):
    """Raised when all connection attempts to worker addresses are exhausted."""

    def __init__(
        self, addresses: list[str], attempts: int, last_error: Exception
    ) -> None:
        self.addresses = addresses
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Failed to connect to workers at {addresses} after {attempts} attempt(s): {last_error}"
        )


@dataclass(frozen=True)
class ConnectResult:
    """A connected :class:`ProcMesh` plus the owned host mesh that backs it.

    ``host_mesh`` is the OWNED :class:`~monarch.actor.HostMesh` returned by
    ``attach_to_workers`` — the handle a caller must ``.stop()`` to tear the
    connection down.  Ownership is carried explicitly here because
    ``proc_mesh.host_mesh`` is only a *slice* (reference) of that mesh, which
    monarch refuses to stop (``"cannot stop `HostMesh` that is a reference
    instead of owned"``).  A connector that does not own a separate host mesh
    (e.g. one spawning procs on ``this_host()``) may leave it ``None``.
    """

    proc_mesh: ProcMesh
    host_mesh: "HostMesh | None" = None


@runtime_checkable
class Connector(Protocol):
    """Protocol for connecting a :class:`ResourceHandle` to a :class:`ConnectResult`."""

    def connect(self, handle: ResourceHandle) -> ConnectResult: ...


def _stop_host_mesh_quietly(host_mesh: object | None) -> None:
    """Best-effort teardown of a host mesh that failed mid-connect (errors logged)."""
    if host_mesh is None:
        return
    try:
        host_mesh.stop().get()  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Error stopping partially-connected host mesh", exc_info=True)


class DefaultConnector:
    """Default connector that bridges :class:`ResourceHandle` → :class:`ProcMesh`.

    Uses ``handle.addresses`` (falling back to ``handle.hostnames``) to
    call :func:`~monarch._src.actor.bootstrap.attach_to_workers`.

    Args:
        mesh_name: Name for the Monarch host mesh.
        max_retries: Maximum number of connection attempts (default: 3).
        retry_delay: Seconds between retries (default: 5).
    """

    def __init__(
        self,
        mesh_name: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._mesh_name = mesh_name or make_name("worker")
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def connect(self, handle: ResourceHandle) -> ConnectResult:
        """Connect to the workers described by *handle* and return a :class:`ConnectResult`.

        Retries on **any** connection failure (not just ``ValueError``/``OSError``)
        since Slurm workers may not be fully ready when their port is first
        registered, and Monarch surfaces connection problems as its own exception
        types.  If ``attach_to_workers`` succeeds but ``spawn_procs`` fails, the
        partially-connected host mesh is torn down before the next attempt so it
        is not leaked.

        The returned :class:`ConnectResult` carries the owned ``host_mesh`` so
        the caller can stop it on teardown; see that class for why ownership
        cannot be rederived from ``proc_mesh.host_mesh``.
        """
        addrs = handle.addresses or handle.hostnames
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            host_mesh = None
            try:
                logger.debug(
                    "Connection attempt %d/%d to %s",
                    attempt,
                    self._max_retries,
                    addrs,
                )
                host_mesh = attach_to_workers(
                    name=self._mesh_name,
                    ca="trust_all_connections",
                    workers=addrs,
                )
                proc_mesh = host_mesh.spawn_procs()
                logger.info("Connected to workers at %s", addrs)
                return ConnectResult(proc_mesh=proc_mesh, host_mesh=host_mesh)
            except Exception as e:
                last_error = e
                # Record each failed attempt as a span event so retries are visible.
                # Tear down a host mesh that attached but failed at spawn_procs,
                # so a partial connection isn't leaked across retries / on failure.
                _stop_host_mesh_quietly(host_mesh)
                if attempt < self._max_retries:
                    logger.warning(
                        "Connection attempt %d/%d failed (%s), retrying in %.0fs...",
                        attempt,
                        self._max_retries,
                        e,
                        self._retry_delay,
                    )
                    time.sleep(self._retry_delay)

        assert last_error is not None
        raise WorkerConnectionError(
            addrs, self._max_retries, last_error
        ) from last_error


def connect(handle: ResourceHandle, *, mesh_name: str | None = None) -> ConnectResult:
    """Convenience: connect to a :class:`ResourceHandle` using :class:`DefaultConnector`."""
    return DefaultConnector(mesh_name=mesh_name).connect(handle)
