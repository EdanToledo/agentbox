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

"""Base resource manager actor for distributed container execution.

Provides a scheduler-agnostic interface for acquiring and releasing
compute resources (machines/nodes) from a pool.  Concrete backends live in
sibling modules (:mod:`agentbox.actors.resource_managers.local` and
:mod:`~agentbox.actors.resource_managers.slurm`).

The base :class:`ResourceManager` uses the Template Method pattern:
common pool bookkeeping (validation, slot popping, handle tracking)
lives in async ``@endpoint`` methods on the base class, while subclasses
override mostly synchronous hook methods for backend-specific behaviour.
Keeping every endpoint async lets lifecycle code await Monarch futures without
mixing endpoint kinds, which Monarch does not permit within one actor class.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable

from agentbox.actors.mesh_schema import base_name, make_name, make_scoped_name
from monarch.actor import Actor, endpoint


class _WorkerHeartbeatActor(Actor):
    """Trivial actor used purely as a supervision anchor.

    Spawned by ResourceManager on each worker proc-mesh. When the worker
    host dies, Monarch delivers a MeshFailure to ResourceManager.__supervise__.
    """

    @endpoint
    def ping(self) -> str:
        return "ok"


logger = logging.getLogger(__name__)

__all__ = [
    "HealthCheckResult",
    "PoolExhaustedError",
    "PoolStatus",
    "ResourceHandle",
    "ResourceManager",
    "await_for_pool",
    "wait_for_pool",
]


@dataclass
class ResourceHandle:
    """A handle representing a set of acquired compute resources.

    Attributes:
        handle_id: Unique identifier for this allocation.
        hostnames: List of hostnames/addresses for the acquired machines.
        addresses: Monarch-connectable URIs (IPC or TCP), populated by the
            ResourceManager.  Use these directly with
            :func:`~monarch._src.actor.bootstrap.attach_to_workers` without
            backend-specific branching.
        metadata: Scheduler-specific metadata (e.g. ``slurm_job_id``, ``port``).
    """

    handle_id: str
    hostnames: list[str]
    addresses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolStatus:
    """Snapshot of pool utilisation returned by :meth:`ResourceManager.pool_status`.

    Backend-specific fields live in *extras*.
    """

    available: int
    in_use: int
    total: int
    extras: dict[str, Any] = field(default_factory=dict)


class PoolExhaustedError(RuntimeError):
    """Raised when more hosts are requested than the pool has available."""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(f"Requested {requested} hosts but only {available} available")

    def __reduce__(self) -> tuple:
        # Monarch serialises exceptions via pickle; override so the two-arg
        # constructor is used rather than the default args-tuple form (which
        # would pass the message string as the first positional arg and omit
        # the required ``available`` argument).
        return (self.__class__, (self.requested, self.available))


@dataclass
class HealthCheckResult:
    """Result of a health check returned by :meth:`ResourceManager.health_check`."""

    evicted_hosts: list[str]
    healthy_hosts: list[str]
    extras: dict[str, Any] = field(default_factory=dict)


class ResourceManager(Actor):
    """Abstract resource manager actor using the Template Method pattern.

    Common pool bookkeeping (validation, slot popping, UUID handle
    generation, ``_handles`` tracking) is implemented here.  Subclasses
    override **hook methods** (plain, non-endpoint) for backend-specific
    behaviour.

    Hook methods to override:

    * :meth:`_build_addresses` — map slot keys to Monarch-connectable URIs (default: identity)
    * :meth:`_build_acquire_metadata` — backend-specific handle metadata (default: empty)
    * :meth:`_on_release` — post-release cleanup (default: no-op)
    * :meth:`_extra_pool_status` — extra status keys (default: empty)
    * :meth:`_do_health_check` — detect dead resources (**must override**)
    * :meth:`_do_spawn` — spawn *n* new resources (**must override**)
    * :meth:`_do_kill` — kill specific resources (**must override**)
    * :meth:`_should_restore_slots_after_kill_failure` — classify ambiguous removals
    * :meth:`_do_shutdown` — kill all resources (**must override**)

    Endpoint implementations are async so they can await Monarch lifecycle
    futures. Hooks that participate in awaited operation ordering, such as
    :meth:`_do_kill` and :meth:`_on_release`, are async as well.
    """

    def __init__(self, *, fault_scope: str | None = None) -> None:
        """Initialize empty resource-pool bookkeeping.

        Args:
            fault_scope: Unique scope encoded in supervision mesh names so faults
                can be routed to the manager that owns this pool.
        """
        self._available: list[str] = []
        self._handles: dict[str, list[str]] = {}
        self._fault_scope = fault_scope
        # Failed slots are removed from worker snapshots immediately. Keep only
        # their IDs while backend teardown is pending.
        self._failed_slots: set[str] = set()
        self._mesh_name_to_slot: dict[str, str] = {}
        self._slot_to_monitor_mesh: dict[str, Any] = {}

    # Endpoint methods (common logic)

    @endpoint
    async def acquire(self, n: int = 1) -> ResourceHandle:
        """Check out *n* machines from the available pool."""

        if n < 1:
            raise ValueError("n must be >= 1")
        if n > len(self._available):
            raise PoolExhaustedError(n, len(self._available))

        acquired = [self._available.pop() for _ in range(n)]
        handle_id = uuid.uuid4().hex[:12]
        self._handles[handle_id] = acquired

        hostnames = self._build_hostnames(acquired)
        addresses = self._build_addresses(acquired)
        metadata = self._build_acquire_metadata(acquired)

        logger.info("Acquired handle %s → %s", handle_id, hostnames)
        return ResourceHandle(
            handle_id=handle_id,
            hostnames=hostnames,
            addresses=addresses,
            metadata=metadata,
        )

    @endpoint
    async def release(self, handle_id: str) -> None:
        """Return machines associated with *handle_id* to the pool."""

        slots = self._handles.pop(handle_id, None)
        if slots is None:
            raise KeyError(f"Unknown handle: {handle_id}")
        self._available.extend(slots)
        await self._on_release(slots)
        logger.info("Released handle %s back to pool", handle_id)

    @endpoint
    async def pool_status(self) -> PoolStatus:
        """Return available / in-use / total counts."""
        in_use = sum(len(h) for h in self._handles.values())
        extras = self._extra_pool_status()
        return PoolStatus(
            available=len(self._available),
            in_use=in_use,
            total=len(self._available) + in_use,
            extras=extras,
        )

    @endpoint
    async def scale_by(self, n: int) -> Any:
        """Scale the pool by *n* workers and return any backend transaction handle."""

        if n == 0:
            raise ValueError("n must be non-zero")
        if n > 0:
            return self._do_spawn(n)

        remove_count = -n
        if remove_count > len(self._available):
            raise RuntimeError(
                f"Cannot scale down by {remove_count}: "
                f"only {len(self._available)} idle hosts"
            )
        removed = [self._available.pop() for _ in range(remove_count)]
        try:
            return await self._do_kill(removed)
        except Exception as error:
            if self._should_restore_slots_after_kill_failure(error, removed):
                # A backend rejection means the workers are still part of
                # the pool. Restore the exact stack order so they remain
                # usable and the same scale-down can be retried.
                self._available.extend(reversed(removed))
            raise

    @endpoint
    async def health_check(self) -> HealthCheckResult:
        """Detect dead resources and evict them from the pool."""

        dead_keys, evicted_hosts, healthy_hosts, extras = self._do_health_check()
        await self._stop_monitor_meshes(*dead_keys)
        for key in dead_keys:
            if key in self._available:
                self._available.remove(key)
            self._failed_slots.discard(key)
        if evicted_hosts:
            logger.warning(
                "Health check evicted %d host(s): %s",
                len(evicted_hosts),
                evicted_hosts,
            )
        return HealthCheckResult(
            evicted_hosts=evicted_hosts, healthy_hosts=healthy_hosts, extras=extras
        )

    @endpoint
    async def shutdown(self) -> None:
        """Kill all underlying jobs and clean up."""
        await self._stop_monitor_meshes(*self._slot_to_monitor_mesh)
        self._do_shutdown()
        self._available.clear()
        self._handles.clear()
        self._failed_slots.clear()

    # Hook methods (override in subclasses)

    def _build_hostnames(self, slot_keys: list[str]) -> list[str]:
        """Map slot keys to human-readable hostnames.  Default: identity."""
        return list(slot_keys)

    def _build_addresses(self, slot_keys: list[str]) -> list[str]:
        """Map slot keys to Monarch-connectable URIs.  Default: identity."""
        return list(slot_keys)

    def _build_acquire_metadata(self, slot_keys: list[str]) -> dict[str, Any]:
        """Return backend-specific metadata for a new handle.  Default: empty."""
        return {}

    async def _on_release(self, slot_keys: list[str]) -> None:
        """Post-release cleanup hook.  Default: no-op."""

    def _extra_pool_status(self) -> dict[str, Any]:
        """Return extra backend-specific keys for pool status.  Default: empty."""
        return {}

    def _do_health_check(
        self,
    ) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
        """Identify dead resources and clean up backend state.

        Returns:
            A 4-tuple of ``(dead_slot_keys, evicted_hostnames,
            healthy_hostnames, extras)``.  The base class handles removing
            *dead_slot_keys* from ``_available``.
        """
        raise NotImplementedError

    def _do_spawn(self, n: int) -> Any:
        """Spawn *n* new resources and append their slot keys to ``self._available``."""
        raise NotImplementedError

    async def _do_kill(self, slot_keys: list[str]) -> Any:
        """Kill the resources identified by *slot_keys*."""
        raise NotImplementedError

    def _should_restore_slots_after_kill_failure(
        self, error: Exception, slot_keys: list[str]
    ) -> bool:
        """Return whether a failed kill should restore its slots to the pool."""
        return True

    def _do_shutdown(self) -> None:
        """Kill all backend resources (the base clears ``_available`` / ``_handles``)."""
        raise NotImplementedError

    def __supervise__(self, failure: Any) -> bool:
        """Handle Monarch supervision events when a worker proc-mesh dies.

        Returns ``True`` if the failure is handled (belongs to a registered
        worker); ``False`` to propagate it up the tree.
        """
        # A fault's mesh_name is the actor name with a random suffix appended
        # (``<name>-<rand>``); strip it to match the registered mesh name.
        mesh_name = base_name(failure.mesh_name)
        slot_key = self._mesh_name_to_slot.get(mesh_name)
        if slot_key is None:
            return False
        logger.error(
            "Worker supervision failure for slot %s: %s", slot_key, failure.report()
        )
        self._evict_failed_worker(slot_key)
        self._slot_to_monitor_mesh.pop(slot_key, None)
        self._mesh_name_to_slot.pop(mesh_name, None)
        self._on_worker_failure(slot_key, failure)
        return True

    def _evict_failed_worker(self, slot_key: str) -> bool:
        """Remove *slot_key* from snapshots while backend teardown completes."""
        if slot_key in self._failed_slots:
            return False
        was_known = slot_key in self._available
        if slot_key in self._available:
            self._available.remove(slot_key)
        for candidate_handle, slots in list(self._handles.items()):
            if slot_key not in slots:
                continue
            was_known = True
            logger.warning(
                "Slot %s was in-use (handle=%s) when it failed",
                slot_key,
                candidate_handle,
            )
            slots.remove(slot_key)
            if not slots:
                self._handles.pop(candidate_handle, None)
            break
        if not was_known:
            logger.debug("Ignoring failure report for unknown slot %s", slot_key)
            return False
        self._failed_slots.add(slot_key)
        return True

    def _on_worker_failure(self, slot_key: str, failure: Any) -> None:
        """Called after a worker is evicted due to a Monarch supervision failure.

        Override in subclasses to add restart logic (e.g., re-submit Slurm task).
        """

    @endpoint
    async def report_failure(self, slot_key: str, reason: str) -> None:
        """Report an externally-detected worker failure (e.g., connection drop).

        Accepts either a slot key (e.g. ``local_0``) or a handle ID.  When
        a handle ID is given, all slots associated with that handle are evicted.
        Calls :meth:`_on_worker_failure_external` for each evicted slot.
        """
        logger.error("External failure report for slot/handle %s: %s", slot_key, reason)
        # If it looks like a handle_id, evict all its slots.
        if slot_key in self._handles:
            slots = list(self._handles[slot_key])
            for sk in slots:
                self._evict_failed_worker(sk)
                self._slot_to_monitor_mesh.pop(sk, None)
                stale = [k for k, v in self._mesh_name_to_slot.items() if v == sk]
                for k in stale:
                    self._mesh_name_to_slot.pop(k, None)
                self._on_worker_failure_external(sk, reason)
            return
        # Otherwise treat as a slot_key directly.
        self._evict_failed_worker(slot_key)
        self._slot_to_monitor_mesh.pop(slot_key, None)
        stale = [k for k, v in self._mesh_name_to_slot.items() if v == slot_key]
        for k in stale:
            self._mesh_name_to_slot.pop(k, None)
        self._on_worker_failure_external(slot_key, reason)

    def _on_worker_failure_external(self, slot_key: str, reason: str) -> None:
        """Called after an externally-reported failure evicts a slot. Override to respawn."""

    def _register_worker_for_supervision(
        self, slot_key: str, addresses: list[str]
    ) -> None:
        """Connect to a ready worker and spawn a heartbeat actor to anchor supervision.

        Must be called from within an endpoint method (Monarch event loop context).
        """
        mesh_name = self._heartbeat_mesh_name(slot_key)
        try:
            from monarch._src.actor.bootstrap import attach_to_workers

            host_mesh = attach_to_workers(
                name=mesh_name,
                ca="trust_all_connections",
                workers=addresses,
            )
            proc_mesh = host_mesh.spawn_procs(name=mesh_name)
            proc_mesh.spawn(mesh_name, _WorkerHeartbeatActor)
            self._slot_to_monitor_mesh[slot_key] = proc_mesh
            self._mesh_name_to_slot[mesh_name] = slot_key
            logger.debug("Supervision registered for slot %s → %s", slot_key, addresses)
        except Exception:
            logger.warning(
                "Failed to register supervision for slot %s", slot_key, exc_info=True
            )

    def _heartbeat_mesh_name(self, slot_key: str) -> str:
        """Return the exact supervision mesh name used for *slot_key*."""
        fault_scope = getattr(self, "_fault_scope", None)
        return (
            make_scoped_name("heartbeat", fault_scope, slot_key)
            if fault_scope is not None
            else make_name("heartbeat", slot_key)
        )

    async def _stop_monitor_meshes(
        self, *slot_keys: str, timeout: float = 10.0
    ) -> None:
        """Stop supervision meshes in parallel before their workers disappear.

        All stop requests in a phase are started before any future is awaited.
        All waits share one absolute deadline, so bulk shutdown is bounded by
        ``timeout`` instead of paying that timeout once per worker. Resource-manager
        endpoints are async so Monarch futures are awaited directly rather than
        blocking the actor runtime with ``Future.get()``.
        """
        resources: list[tuple[str, Any, Any]] = []
        for slot_key in slot_keys:
            proc_mesh = self._slot_to_monitor_mesh.pop(slot_key, None)
            stale = [
                key
                for key, value in self._mesh_name_to_slot.items()
                if value == slot_key
            ]
            for key in stale:
                self._mesh_name_to_slot.pop(key, None)
            if proc_mesh is None:
                continue
            try:
                host = proc_mesh.host_mesh
            except Exception:
                host = None
            resources.append((slot_key, proc_mesh, host))

        if not resources:
            return
        deadline = time.monotonic() + timeout
        proc_stops: list[tuple[str, Awaitable[Any]]] = []
        for slot_key, proc_mesh, _host in resources:
            try:
                proc_stops.append((slot_key, proc_mesh.stop()))
            except Exception:
                logger.debug(
                    "Error starting monitor proc mesh stop for slot %s",
                    slot_key,
                    exc_info=True,
                )
        await self._stop_monitor_mesh_phase(proc_stops, deadline, "proc")

        host_stops: list[tuple[str, Awaitable[Any]]] = []
        for slot_key, _proc_mesh, host in resources:
            if host is None:
                continue
            try:
                host_stops.append((slot_key, host.stop()))
            except Exception:
                logger.debug(
                    "Error starting monitor host mesh stop for slot %s",
                    slot_key,
                    exc_info=True,
                )
        await self._stop_monitor_mesh_phase(host_stops, deadline, "host")

    @staticmethod
    async def _stop_monitor_mesh_phase(
        stops: list[tuple[str, Awaitable[Any]]],
        deadline: float,
        resource_kind: str,
    ) -> None:
        """Await one phase of monitor cleanup concurrently and best-effort."""
        if not stops:
            return

        async def stop_one(slot_key: str, stop_future: Awaitable[Any]) -> None:
            try:
                await stop_future
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Error stopping monitor %s mesh for slot %s",
                    resource_kind,
                    slot_key,
                    exc_info=True,
                )

        tasks = [
            asyncio.create_task(stop_one(slot_key, stop_future))
            for slot_key, stop_future in stops
        ]
        remaining = max(0.0, deadline - time.monotonic())
        _done, pending = await asyncio.wait(tasks, timeout=remaining)
        if not pending:
            return

        logger.debug(
            "Timed out stopping %d monitor %s mesh(es)",
            len(pending),
            resource_kind,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def await_for_pool(
    manager: ResourceManager,
    min_available: int = 1,
    timeout: float | None = None,
    poll_interval: float = 5.0,
) -> PoolStatus:
    """Wait asynchronously until enough workers are available in the pool.

    This is the cancellation-safe counterpart of :func:`wait_for_pool`. Pool
    endpoints are awaited directly and the delay between polls uses
    :func:`asyncio.sleep`, so cancelling the caller does not leave an executor
    thread polling in the background.

    If *manager* exposes a ``poll_pending`` endpoint (e.g.
    :class:`SlurmResourceManager`), it is called each iteration to discover
    newly-running workers.

    Args:
        manager: The resource manager to poll.
        min_available: Minimum number of available workers to wait for.
        timeout: Maximum seconds to wait before raising :class:`TimeoutError`.
            ``None`` waits indefinitely.
        poll_interval: Seconds between polls.

    Returns:
        The final :class:`PoolStatus` once enough workers are available.

    Raises:
        TimeoutError: If the pool does not reach *min_available* within *timeout*.
        ValueError: If *timeout* is negative or NaN.

    """
    import asyncio

    if timeout is not None and (timeout < 0 or math.isnan(timeout)):
        raise ValueError("timeout must be non-negative or None; NaN is not supported")

    start = time.monotonic()
    has_poll_pending = hasattr(manager, "poll_pending")

    polls = 0
    while True:
        polls += 1
        if has_poll_pending:
            await manager.poll_pending.choose()

        status: PoolStatus = await manager.pool_status.choose()
        if status.available >= min_available:
            return status

        elapsed = time.monotonic() - start
        if timeout is not None and elapsed >= timeout:
            raise TimeoutError(
                f"Only {status.available}/{min_available} workers available "
                f"after {elapsed:.0f}s (pending={status.extras.get('pending_tasks', '?')})"
            )

        logger.info(
            "Waiting for workers: %d/%d available (%.0fs elapsed)",
            status.available,
            min_available,
            elapsed,
        )
        logger.debug(
            "Pool details: in_use=%d, total=%d, pending_tasks=%s",
            status.in_use,
            status.total,
            status.extras.get("pending_tasks", "?"),
        )
        sleep_for = poll_interval
        if timeout is not None:
            sleep_for = min(poll_interval, max(0.0, timeout - elapsed))
        await asyncio.sleep(sleep_for)


def wait_for_pool(
    manager: ResourceManager,
    min_available: int = 1,
    timeout: float | None = None,
    poll_interval: float = 5.0,
) -> PoolStatus:
    """Block until at least *min_available* workers are in the pool.

    If *manager* exposes a ``poll_pending`` endpoint (e.g.
    :class:`SlurmResourceManager`), it is called each iteration to discover
    newly-running workers.

    Args:
        manager: The resource manager to poll.
        min_available: Minimum number of available workers to wait for.
        timeout: Maximum seconds to wait before raising :class:`TimeoutError`.
            ``None`` waits indefinitely.
        poll_interval: Seconds between polls.

    Returns:
        The final :class:`PoolStatus` once enough workers are available.

    Raises:
        TimeoutError: If the pool does not reach *min_available* within *timeout*.
        ValueError: If *timeout* is negative or NaN.
    """

    if timeout is not None and (timeout < 0 or math.isnan(timeout)):
        raise ValueError("timeout must be non-negative or None; NaN is not supported")

    start = time.monotonic()
    has_poll_pending = hasattr(manager, "poll_pending")

    polls = 0
    while True:
        polls += 1
        if has_poll_pending:
            manager.poll_pending.choose().get()

        status: PoolStatus = manager.pool_status.choose().get()
        if status.available >= min_available:
            return status

        elapsed = time.monotonic() - start
        if timeout is not None and elapsed >= timeout:
            raise TimeoutError(
                f"Only {status.available}/{min_available} workers available "
                f"after {elapsed:.0f}s (pending={status.extras.get('pending_tasks', '?')})"
            )

        logger.info(
            "Waiting for workers: %d/%d available (%.0fs elapsed)",
            status.available,
            min_available,
            elapsed,
        )
        logger.debug(
            "Pool details: in_use=%d, total=%d, pending_tasks=%s",
            status.in_use,
            status.total,
            status.extras.get("pending_tasks", "?"),
        )
        sleep_for = poll_interval
        if timeout is not None:
            sleep_for = min(poll_interval, max(0.0, timeout - elapsed))
        time.sleep(sleep_for)
