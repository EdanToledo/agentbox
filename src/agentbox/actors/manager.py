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

"""High-level AgentBoxManager facade for machine allocation."""

from __future__ import annotations

import asyncio
import logging
import math
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager, ExitStack
from time import monotonic as _monotonic
from typing import Any, AsyncGenerator, Callable, cast, Generator, TYPE_CHECKING

from agentbox.actors.machine import Machine
from agentbox.actors.mesh_schema import (
    FAULT_POLICY,
    fault_scope,
    FaultAction,
    is_internal_droppable,
    make_scoped_name,
    MeshRegistry,
    MeshType,
    parse_name,
    strip_fault_scope,
)
from agentbox.actors.resource_managers import (
    await_for_pool,
    LocalResourceManager,
    PoolExhaustedError,
    PoolStatus,
    ResourceHandle,
    SlurmResourceManager,
    wait_for_pool,
)
from monarch.actor import ActorError, enable_transport, ProcMesh, this_host


if TYPE_CHECKING:
    from agentbox.actors.resource_managers import SlurmAllocation


logger = logging.getLogger(__name__)

# Connecting to an acquired worker can time out transiently when one client
# attaches hundreds of workers at once. Retry with backoff; on final failure
# return the handle to the pool instead of leaking the worker.
_CONNECT_ATTEMPTS = 3
_CONNECT_BACKOFF_SECONDS = 2.0


async def _awaited(future: Any) -> Any:
    """Await a Monarch future (``asyncio.wait_for`` needs a coroutine)."""
    return await future


def _new_fault_scope() -> str:
    """Return a process-unique namespace for manager-owned fault sources."""
    return f"mgr{secrets.token_hex(8)}"


def _unwrap_pool_exhausted_error(
    error: PoolExhaustedError | ActorError,
) -> PoolExhaustedError | None:
    """Return the pool error hidden by Monarch's actor RPC boundary, if any."""
    if isinstance(error, PoolExhaustedError):
        return error
    # ResourceManager.acquire() uses pool exhaustion as expected control flow:
    # the manager either retries, times out, or reports a nonblocking miss. Since
    # Monarch wraps endpoint exceptions at the actor RPC boundary, recover that
    # specific error here before applying the caller's wait policy.
    if isinstance(error.exception, PoolExhaustedError):
        return error.exception
    return None


class UnhandledProcMeshFault(RuntimeError):
    """Raised by the client fault hook for a genuinely unexpected proc-mesh fault.

    Monarch treats a raising hook as unhandled and tears the client down (see
    ``supervision.py``). Raising a named exception that carries the failure
    ``report`` makes that teardown debuggable rather than opaque. The hook also
    logs the fault in full before raising.
    """


class _FaultHookDispatcher:
    """Route Monarch's process-wide fault callback to the owning manager.

    Monarch exposes one process-global hook, while AgentBox may keep multiple
    managers alive for independent pools. The dispatcher tracks those managers,
    routes scoped or registered meshes to their owner, and prevents ambiguous or
    late faults from mutating an unrelated pool.
    """

    def __init__(self, previous_hook: object) -> None:
        """Create a dispatcher that can delegate fatal foreign faults.

        Args:
            previous_hook: Monarch fault hook installed before this dispatcher.
                Callable hooks receive faults that no live AgentBox manager owns.
        """
        self.previous_hook = previous_hook
        self._managers: list[AgentBoxManager] = []
        self._lock = threading.Lock()

    def add(self, manager: AgentBoxManager) -> None:
        """Register *manager* once, preserving registration order."""
        with self._lock:
            if not any(candidate is manager for candidate in self._managers):
                self._managers.append(manager)

    def remove(self, manager: AgentBoxManager) -> bool:
        """Unregister *manager* and report whether the dispatcher is now empty."""
        with self._lock:
            self._managers = [
                candidate for candidate in self._managers if candidate is not manager
            ]
            return not self._managers

    def __call__(self, failure: object) -> None:
        """Classify and deliver a Monarch fault without crossing pool ownership."""
        with self._lock:
            managers = list(self._managers)

        if not managers:
            if callable(self.previous_hook):
                self.previous_hook(failure)
            return

        mesh_name = getattr(failure, "mesh_name", None)
        if mesh_name:
            # Registered worker meshes and manager-scoped coordinator/heartbeat
            # meshes have exactly one owner.
            for manager in managers:
                if manager._owns_fault(mesh_name):
                    manager._client_fault_hook(failure)
                    return

        # A scoped fault whose owner is gone is a late teardown notification.
        # Never send it to another pool merely because that pool is still live.
        if fault_scope(mesh_name) is not None:
            logger.warning("Dropping fault for an inactive manager: %r", mesh_name)
            return

        # Undeliverable-message and Monarch-internal faults name no AgentBox
        # mesh, and unscoped auxiliary actors (shells, PTYs, file transfer,
        # containers, ...) name no manager. Every manager classifies these
        # identically (as droppable); Monarch's default hook would exit.
        parsed = parse_name(mesh_name)
        if (
            not mesh_name
            or mesh_name == "<none>"
            or is_internal_droppable(mesh_name)
            or (parsed is not None and parsed.mesh_type is MeshType.AUXILIARY)
        ):
            managers[-1]._client_fault_hook(failure)
            return

        if callable(self.previous_hook):
            self.previous_hook(failure)
            return
        raise UnhandledProcMeshFault(
            f"unhandled proc-mesh fault with no identifiable manager: {mesh_name!r}"
        )


_fault_hook_lock = threading.Lock()
_fault_hook_dispatchers: list[tuple[Any, _FaultHookDispatcher]] = []


@contextmanager
def _registered_fault_hook(
    actor_module: Any, manager: AgentBoxManager
) -> Generator[None, None, None]:
    """Register *manager* with one process-wide Monarch fault dispatcher."""
    dispatcher: _FaultHookDispatcher | None = None
    registered = False
    with _fault_hook_lock:
        for candidate_module, candidate_dispatcher in _fault_hook_dispatchers:
            if candidate_module is actor_module:
                dispatcher = candidate_dispatcher
                break

        if dispatcher is None:
            dispatcher = _FaultHookDispatcher(
                getattr(actor_module, "unhandled_fault_hook", None)
            )
            dispatcher.add(manager)
            try:
                actor_module.unhandled_fault_hook = dispatcher
            except (AttributeError, TypeError):
                logger.debug(
                    "monarch.actor.unhandled_fault_hook not settable; "
                    "skipping client fault hook"
                )
            else:
                _fault_hook_dispatchers.append((actor_module, dispatcher))
                registered = True
        else:
            dispatcher.add(manager)
            registered = True

    try:
        yield
    finally:
        if registered:
            assert dispatcher is not None
            active_dispatcher = dispatcher
            with _fault_hook_lock:
                if active_dispatcher.remove(manager):
                    try:
                        if (
                            getattr(actor_module, "unhandled_fault_hook", None)
                            is active_dispatcher
                        ):
                            actor_module.unhandled_fault_hook = (
                                active_dispatcher.previous_hook
                            )
                    except (AttributeError, TypeError):
                        pass
                    _fault_hook_dispatchers[:] = [
                        entry
                        for entry in _fault_hook_dispatchers
                        if entry[1] is not active_dispatcher
                    ]


class AgentBoxManager:
    """Thin facade that hides ProcMesh / ResourceManager wiring.

    Usage::

        manager = AgentBoxManager.create_local(num_workers=1)
        with manager.acquire_machine() as machine:
            ...
        manager.shutdown()

    Or as a context manager::

        with AgentBoxManager.create_local() as manager:
            with manager.acquire_machine() as machine:
                ...
    """

    def __init__(
        self,
        manager_actor: Any,
        proc_mesh: ProcMesh | None,
        on_machine_failure: Callable[[str, str], None] | None = None,
        fault_scope: str | None = None,
    ) -> None:
        """Initialize a manager around an AgentBox resource provider.

        Args:
            manager_actor: Resource-manager actor that serves pool operations.
            proc_mesh: Coordinator process mesh that owns ``manager_actor``, or
                ``None`` when the manager does not own a coordinator mesh.
            on_machine_failure: Optional callback receiving the failed resource
                handle and its fault report.
            fault_scope: Unique scope encoded in coordinator and heartbeat mesh
                names so process-wide fault dispatch reaches this manager.
        """
        self._manager: Any = manager_actor
        self._proc_mesh = proc_mesh
        self._on_machine_failure = on_machine_failure
        self._fault_scope = fault_scope or _new_fault_scope()
        # Controller-side registry of every mesh this client creates, tagged with its
        # MeshType + metadata (see mesh_schema). The fault hook classifies faults via
        # this registry plus the name schema, so an agentbox mesh is never unhandled.
        # Each mesh's record is marked not-live on release, so its asynchronous
        # teardown faults are dropped instead of reported.
        self._mesh_registry = MeshRegistry()

        # Monarch's local runtime leaks helper processes (this_host/proc-mesh
        # bootstrap_main and pool run_worker_loop_forever) that its own teardown
        # (ProcMesh/HostMesh.stop + the atexit shutdown_context) does not reap;
        # arm a process-exit reaper to clean up this process's own descendants
        # so they don't accumulate across runs.
        from agentbox.process import ensure_monarch_reaper, track_monarch_descendants

        ensure_monarch_reaper()
        track_monarch_descendants()

        import monarch.actor as _monarch_actor

        # Keep this manager registered with the process-wide client fault
        # dispatcher for its full lifetime. The first live manager installs the
        # dispatcher and the last one to shut down restores Monarch's prior hook.
        self._fault_hook_stack = ExitStack()
        self._fault_hook_stack.enter_context(
            _registered_fault_hook(_monarch_actor, self)
        )

        self._shutdown_done = False

    # Correctness sketch — release vs. asynchronous fault delivery
    # Monarch delivers faults asynchronously and at the ACTOR level: a fault's
    # mesh_name is an actor name with a random suffix (``<actor>-<rand>``). Every
    # agentbox actor is schema-named (mesh_schema.make_name → ``agentbox_<kind>...``),
    # so _client_fault_hook classifies every fault it owns via parse_name +
    # FAULT_POLICY, backed by self._mesh_registry for released-state + metadata:
    #
    #     released (registry)              -> expected teardown: drop
    #     undeliverable (no mesh_name)          -> transport co-symptom: drop
    #     WORKER   (...worker_<scope>__<handle>) -> report handle to the pool, drop
    #     HEARTBEAT(...heartbeat_<scope>__<slot>) -> report slot to the pool, drop
    #     AUXILIARY(shells / file transfer / …) -> drop (co-symptom of a proc death)
    #     COORDINATOR (...coordinator_<scope>)  -> FATAL: raise UnhandledProcMeshFault
    #     logger-* / host_agent (Monarch-internal) -> drop (allowlisted co-symptom)
    #     anything else (truly foreign)         -> raise UnhandledProcMeshFault
    #
    # Why release is race-free. On release we mark the worker released BEFORE
    # tearing it down:
    #
    #     registry.mark_released(worker_name)  ->  machine.release()  # kills the worker
    #
    # Tearing down the worker produces broken-link faults, but those can only
    # arrive after machine.release() — by then the record is not-live, so the hook
    # drops them. A fault that arrives BEFORE mark_released is a live worker dying
    # for real, reported via the WORKER branch. The not-live record is retained for
    # a 60s TTL (longer than the observed teardown-fault latency) then evicted;
    # worker names are never reused, so a fault after eviction is a harmless no-op.
    def _owns_fault(self, mesh_name: str) -> bool:
        """Return whether *mesh_name* belongs to this manager."""
        if self._mesh_registry.get(mesh_name) is not None:
            return True
        scope = getattr(self, "_fault_scope", None)
        return scope is not None and fault_scope(mesh_name) == scope

    def _client_fault_hook(self, failure: object) -> None:
        """Classify a Monarch fault via the mesh schema/registry and act on it.

        No agentbox-owned mesh is ever unhandled: every one is schema-named and maps
        to a :class:`~agentbox.actors.mesh_schema.FaultAction`. Only a genuinely
        foreign mesh (not agentbox, not an allowlisted Monarch-internal name) raises
        :class:`UnhandledProcMeshFault` — a debuggable teardown, never a bare exit.
        """
        mesh_name = getattr(failure, "mesh_name", None)
        report = getattr(failure, "report", lambda: str(failure))()

        # Undeliverable-message fault with no owning mesh: a transport co-symptom of
        # a host/worker being drained (e.g. a host_agent DrainHost broken-link during
        # release). Monarch reports these with no owning mesh — mesh_name is absent
        # (Python None) or the literal sentinel "<none>". They name no agentbox mesh,
        # and a genuine worker/host death is reported authoritatively via the
        # WORKER/HEARTBEAT branches (which carry a real mesh_name) — so drop it rather
        # than treat it as a fatal foreign fault.
        if not mesh_name or mesh_name == "<none>":
            logger.warning(
                "Dropping undeliverable-message proc-mesh fault (mesh=%r): %s",
                mesh_name,
                report,
            )
            return

        record = self._mesh_registry.get(mesh_name)

        # Expected teardown fault from a mesh we intentionally released (not live).
        if record is not None and not record.live:
            logger.warning(
                "Dropping post-release proc-mesh fault for %s: %s", mesh_name, report
            )
            return

        parsed = parse_name(mesh_name)
        if parsed is None:
            # Not agentbox-owned. A known-benign Monarch-internal mesh (e.g. the
            # per-proc logger-*) is always a co-symptom of a proc death that an
            # agentbox actor on the same proc reports authoritatively — drop it.
            if is_internal_droppable(mesh_name):
                logger.warning(
                    "Dropping Monarch-internal proc-mesh fault for %s: %s",
                    mesh_name,
                    report,
                )
                return
            logger.error(
                "Unhandled foreign proc-mesh fault: mesh=%r report=%s",
                mesh_name,
                report,
            )
            raise UnhandledProcMeshFault(
                f"unhandled proc-mesh fault for foreign mesh {mesh_name!r}: {report}"
            )

        action = FAULT_POLICY[parsed.mesh_type]
        if action is FaultAction.FATAL:
            logger.error(
                "Fatal %s proc-mesh fault: mesh=%r report=%s",
                parsed.mesh_type.name,
                mesh_name,
                report,
            )
            raise UnhandledProcMeshFault(
                f"unhandled proc-mesh fault for {parsed.mesh_type.name} mesh {mesh_name!r}: {report}"
            )
        if action is FaultAction.REPORT_HANDLE_AND_DROP:
            fallback_handle_id = strip_fault_scope(
                parsed.ident,
                getattr(self, "_fault_scope", None),
            )
            handle_id = (
                record.metadata.get("handle_id", fallback_handle_id)
                if record is not None
                else fallback_handle_id
            )
            logger.error(
                "Worker proc-mesh failure for handle %s: %s", handle_id, report
            )
            # Mark failed so the release path skips returning this dead worker to the
            # pool (report_failure below already evicts its slot).
            self._mesh_registry.mark_failed(mesh_name)
            self._report_to_pool(handle_id, report)
            if self._on_machine_failure is not None:
                self._on_machine_failure(handle_id, report)
            return
        if action is FaultAction.REPORT_SLOT_AND_DROP:
            slot_key = strip_fault_scope(
                parsed.ident,
                getattr(self, "_fault_scope", None),
            )
            logger.error("Worker-supervision fault for slot %s: %s", slot_key, report)
            self._report_to_pool(slot_key, report)
            return
        # AUXILIARY (shells, file transfer, admin, …): a co-symptom of a proc death
        # the WORKER/HEARTBEAT branch reports; nothing to report here.
        logger.warning(
            "Dropping auxiliary proc-mesh fault for %s: %s", mesh_name, report
        )

    def _report_to_pool(self, slot_or_handle: str, report: str) -> None:
        """Best-effort eviction report to the resource manager (never raises)."""
        try:
            self._manager.report_failure.choose(slot_or_handle, report).get(timeout=5.0)
        except Exception:
            logger.warning(
                "Failed to report failure to resource manager", exc_info=True
            )

    @classmethod
    def create_local(
        cls,
        num_workers: int = 1,
        env: dict[str, str] | None = None,
        on_machine_failure: Callable[[str, str], None] | None = None,
    ) -> AgentBoxManager:
        """Create a manager backed by local subprocess workers."""
        fault_scope_id = _new_fault_scope()
        coordinator_name = make_scoped_name("coordinator", fault_scope_id)
        host = this_host()
        proc_mesh: ProcMesh = host.spawn_procs(name=coordinator_name)
        manager = proc_mesh.spawn(
            coordinator_name,
            LocalResourceManager,
            initial_nodes=num_workers,
            env=env,
            fault_scope=fault_scope_id,
        )
        return cls(
            manager,
            proc_mesh,
            on_machine_failure=on_machine_failure,
            fault_scope=fault_scope_id,
        )

    @classmethod
    def create_slurm(
        cls,
        allocation: SlurmAllocation | None = None,
        num_workers: int = 1,
        python_exe: str | None = None,
        on_machine_failure: Callable[[str, str], None] | None = None,
        chunk_size: int | None = None,
        max_parallel: int = 128,
    ) -> AgentBoxManager:
        """Create a manager backed by Slurm job-array workers."""
        from agentbox.port_server import routable_ipv4

        # Use the same explicit address family as the workers.  Resolving a
        # dual-stack hostname independently at either end can otherwise make
        # the coordinator dial IPv4 while a worker listens on IPv6 (or vice versa).
        ip = routable_ipv4()
        logger.info("Enabling TCP transport on %s for Slurm coordinator", ip)
        enable_transport(f"tcp://{ip}:0")

        fault_scope_id = _new_fault_scope()
        coordinator_name = make_scoped_name("coordinator", fault_scope_id)
        host = this_host()
        proc_mesh: ProcMesh = host.spawn_procs(name=coordinator_name)
        manager = proc_mesh.spawn(
            coordinator_name,
            SlurmResourceManager,
            allocation=allocation,
            initial_nodes=num_workers,
            chunk_size=chunk_size,
            max_parallel=max_parallel,
            python_exe=python_exe or sys.executable,
            fault_scope=fault_scope_id,
        )
        return cls(
            manager,
            proc_mesh,
            on_machine_failure=on_machine_failure,
            fault_scope=fault_scope_id,
        )

    def wait_for_workers(
        self, min_available: int = 1, timeout: float | None = None
    ) -> PoolStatus:
        """Block until the pool contains enough available workers.

        This is primarily useful for scheduler-backed managers whose workers
        start asynchronously. Pending scheduler tasks are polled when supported.
        ``timeout=None`` waits
        indefinitely.

        Args:
            min_available: Minimum number of idle workers required before
                returning.
            timeout: Maximum seconds to wait. ``None`` waits indefinitely and
                zero performs one availability check before timing out.

        Returns:
            The pool status that first satisfies *min_available*.

        Raises:
            ValueError: If *min_available* is less than one, or if *timeout* is
                negative or NaN.
            TimeoutError: If the requested number of workers does not become
                available within *timeout*.

        """
        if min_available < 1:
            raise ValueError("min_available must be >= 1")
        timeout = self._normalize_machine_acquire_timeout(timeout)
        if timeout is None:
            logger.info("Waiting for %d worker(s) indefinitely...", min_available)
        else:
            logger.info(
                "Waiting for %d worker(s) (timeout=%.0fs)...",
                min_available,
                timeout,
            )
        status = wait_for_pool(
            self._manager, min_available=min_available, timeout=timeout
        )
        logger.info("Workers ready: %d available", status.available)
        return status

    @staticmethod
    def _normalize_machine_acquire_timeout(timeout: float | None) -> float | None:
        if timeout is not None:
            timeout = float(timeout)
            if timeout < 0 or math.isnan(timeout):
                raise ValueError(
                    "timeout must be non-negative or None; NaN is not supported"
                )
        return timeout

    @classmethod
    def _machine_acquire_deadline(
        cls, *, blocking: bool, timeout: float | None
    ) -> float | None:
        timeout = cls._normalize_machine_acquire_timeout(timeout)
        if not blocking or timeout is None:
            return None
        return _monotonic() + timeout

    def _acquire_resource_handle(
        self, *, blocking: bool, timeout: float | None
    ) -> ResourceHandle:
        """Acquire one pool handle using the requested wait policy."""
        deadline = self._machine_acquire_deadline(
            blocking=blocking,
            timeout=timeout,
        )
        if (not blocking or timeout == 0) and hasattr(self._manager, "poll_pending"):
            self._manager.poll_pending.choose().get()

        while True:
            try:
                return cast(ResourceHandle, self._manager.acquire.choose(1).get())
            except (PoolExhaustedError, ActorError) as error:
                pool_error = _unwrap_pool_exhausted_error(error)
                if pool_error is None:
                    raise
                if not blocking:
                    if isinstance(error, PoolExhaustedError):
                        raise
                    raise pool_error from None
                remaining = None if deadline is None else deadline - _monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for an available machine"
                    ) from pool_error
                logger.warning("No workers available, waiting for pool...")
                try:
                    wait_for_pool(
                        self._manager,
                        min_available=1,
                        timeout=remaining,
                    )
                except TimeoutError as wait_error:
                    raise TimeoutError(
                        "Timed out waiting for an available machine"
                    ) from wait_error

    async def _aacquire_resource_handle(
        self, *, blocking: bool, timeout: float | None
    ) -> ResourceHandle:
        """Async-native counterpart of :meth:`_acquire_resource_handle`."""
        deadline = self._machine_acquire_deadline(
            blocking=blocking,
            timeout=timeout,
        )
        if (not blocking or timeout == 0) and hasattr(self._manager, "poll_pending"):
            await self._manager.poll_pending.choose()

        while True:
            try:
                return cast(ResourceHandle, await self._manager.acquire.choose(1))
            except (PoolExhaustedError, ActorError) as error:
                pool_error = _unwrap_pool_exhausted_error(error)
                if pool_error is None:
                    raise
                if not blocking:
                    if isinstance(error, PoolExhaustedError):
                        raise
                    raise pool_error from None
                remaining = None if deadline is None else deadline - _monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for an available machine"
                    ) from pool_error
                logger.warning("No workers available, waiting for pool...")
                try:
                    await await_for_pool(
                        self._manager,
                        min_available=1,
                        timeout=remaining,
                    )
                except TimeoutError as wait_error:
                    raise TimeoutError(
                        "Timed out waiting for an available machine"
                    ) from wait_error

    def _connect_machine(self, handle: ResourceHandle, worker_name: str) -> Machine:
        """Connect to an acquired worker, retrying transient failures.

        If connecting fails for good or is interrupted, the handle goes back to
        the pool so the worker is not leaked.
        """
        try:
            for attempt in range(1, _CONNECT_ATTEMPTS + 1):
                try:
                    return Machine.from_handle(
                        handle, mesh_name=worker_name, metadata_actor_name=worker_name
                    )
                except Exception:
                    if attempt == _CONNECT_ATTEMPTS:
                        raise
                    self._log_connect_retry(handle, attempt)
                    time.sleep(_CONNECT_BACKOFF_SECONDS * attempt)
            raise AssertionError("unreachable")
        except BaseException:
            try:
                self._manager.release.choose(handle.handle_id).get(timeout=30.0)
            except Exception:
                self._log_release_failure(handle)
            raise

    async def _aconnect_machine(
        self, handle: ResourceHandle, worker_name: str
    ) -> Machine:
        """Async-native counterpart of :meth:`_connect_machine`."""
        try:
            for attempt in range(1, _CONNECT_ATTEMPTS + 1):
                try:
                    return await Machine.afrom_handle(
                        handle, mesh_name=worker_name, metadata_actor_name=worker_name
                    )
                except Exception:
                    if attempt == _CONNECT_ATTEMPTS:
                        raise
                    self._log_connect_retry(handle, attempt)
                    await asyncio.sleep(_CONNECT_BACKOFF_SECONDS * attempt)
            raise AssertionError("unreachable")
        except BaseException:
            # Send the release before awaiting it, so it goes out even if this
            # task is cancelled again while waiting for the reply.
            release = self._manager.release.choose(handle.handle_id)
            try:
                await asyncio.wait_for(_awaited(release), timeout=30.0)
            except (Exception, asyncio.CancelledError):
                self._log_release_failure(handle)
            raise

    @staticmethod
    def _log_connect_retry(handle: ResourceHandle, attempt: int) -> None:
        logger.warning(
            "Connecting to %s failed (attempt %d/%d); retrying",
            handle.handle_id,
            attempt,
            _CONNECT_ATTEMPTS,
            exc_info=True,
        )

    @staticmethod
    def _log_release_failure(handle: ResourceHandle) -> None:
        logger.warning(
            "Failed to release unconnected handle %s",
            handle.handle_id,
            exc_info=True,
        )

    @contextmanager
    def acquire_machine(
        self,
        *,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> Generator[Machine, None, None]:
        """Acquire a worker, wrap it as a :class:`Machine`, and release on exit.

        When *blocking* is true, wait up to *timeout* seconds for pool capacity;
        ``timeout=None`` waits indefinitely. When *blocking* is false, attempt
        acquisition once and raise :class:`PoolExhaustedError` if no worker is
        available; *timeout* only applies when *blocking* is true. The timeout
        covers pool capacity, not worker connection setup.

        .. todo:: Warm proc_mesh cache.
            Every call here ends with ``Machine.release()`` tearing down the
            remote proc_mesh + host_mesh, and the next call rebuilds them via
            ``Machine.from_handle`` (attach_to_workers + spawn_procs + first
            RPC), so a loop pattern like::

                for task in tasks:
                    with manager.acquire_machine() as m:
                        ...

            pays that setup cost on every iteration even though the underlying
            worker is still running, and is much slower than a hand-rolled
            "acquire N once, dispatch many" loop.

            Fix sketch: cache proc_meshes here keyed on slot key (surface
            ``ResourceHandle.slot_keys`` from the resource manager).  On
            acquire, hand back the cached Machine if its slot is still live;
            on release, ``detach`` rather than ``release`` to keep the
            proc_mesh warm.  Tear the cache down in :meth:`shutdown`.  Slot
            keys are never recycled (see
            :class:`~agentbox.actors.resource_managers.ResourceManager`),
            so eviction can be lazy via a ``live_slot_keys`` endpoint.
        """
        timeout = self._normalize_machine_acquire_timeout(timeout)
        handle = self._acquire_resource_handle(
            blocking=blocking,
            timeout=timeout,
        )
        logger.info("Acquired machine: handle=%s", handle.handle_id)
        worker_name = make_scoped_name("worker", self._fault_scope, handle.handle_id)
        machine = self._connect_machine(handle, worker_name)
        # Record the client-side proc-mesh helper processes just spawned by
        # the connect, while they are still descendants of this process, so
        # the exit reaper can clean them up even after Monarch orphans them.
        from agentbox.process import track_monarch_descendants

        track_monarch_descendants()
        # Register the worker mesh so the fault hook classifies its faults (and any
        # teardown fault after release) against this handle.
        self._mesh_registry.register(
            worker_name,
            MeshType.WORKER,
            ident=handle.handle_id,
            handle_id=handle.handle_id,
        )

        try:
            yield machine
        finally:
            # A worker that faulted mid-use was already evicted by the fault hook;
            # don't return it to the pool (its release RPC would KeyError).
            record = self._mesh_registry.get(worker_name)
            worker_failed = record is not None and record.failed
            self._mesh_registry.mark_released(worker_name)
            machine.release()
            if not worker_failed:
                self._manager.release.choose(handle.handle_id).get()
            else:
                logger.info(
                    "Skipping pool-release of failed worker handle %s (already evicted)",
                    handle.handle_id,
                )
            logger.info("Released machine: handle=%s", handle.handle_id)

    @asynccontextmanager
    async def aacquire_machine(
        self,
        *,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> AsyncGenerator[Machine, None]:
        """Async-native counterpart of :meth:`acquire_machine`.

        Awaits the pool-status / acquire / connect / release RPCs, so it can run
        inside a running asyncio event loop.  Use as
        ``async with manager.aacquire_machine() as machine:``.

        ``timeout=None`` waits indefinitely for pool capacity. Nonblocking
        acquisition raises :class:`PoolExhaustedError` immediately when the pool
        is empty; *timeout* only applies when *blocking* is true. The timeout does
        not cover worker connection setup.

        """
        timeout = self._normalize_machine_acquire_timeout(timeout)
        handle = await self._aacquire_resource_handle(
            blocking=blocking,
            timeout=timeout,
        )
        logger.info("Acquired machine: handle=%s", handle.handle_id)
        worker_name = make_scoped_name("worker", self._fault_scope, handle.handle_id)
        machine = await self._aconnect_machine(handle, worker_name)
        from agentbox.process import track_monarch_descendants

        track_monarch_descendants()
        self._mesh_registry.register(
            worker_name,
            MeshType.WORKER,
            ident=handle.handle_id,
            handle_id=handle.handle_id,
        )

        try:
            yield machine
        finally:
            # A worker that faulted mid-use was already evicted by the fault hook;
            # don't return it to the pool (its release RPC would KeyError).
            record = self._mesh_registry.get(worker_name)
            worker_failed = record is not None and record.failed
            self._mesh_registry.mark_released(worker_name)
            await machine.arelease()
            if not worker_failed:
                await self._manager.release.choose(handle.handle_id)
            else:
                logger.info(
                    "Skipping pool-release of failed worker handle %s (already evicted)",
                    handle.handle_id,
                )
            logger.info("Released machine: handle=%s", handle.handle_id)

    def scale_by(self, n: int) -> None:
        """Grow (``n > 0``) or shrink (``n < 0``, idle workers only) the pool."""
        if n == 0:
            raise ValueError("n must be non-zero")
        self._manager.scale_by.choose(n).get()

    async def ascale_by(self, n: int) -> None:
        """Async-native counterpart of :meth:`scale_by`."""
        if n == 0:
            raise ValueError("n must be non-zero")
        await self._manager.scale_by.choose(n)

    def pool_status(self) -> PoolStatus:
        """Return the current resource pool status."""
        return self._manager.pool_status.choose().get()  # type: ignore[no-any-return]

    async def apool_status(self) -> PoolStatus:
        """Async-native counterpart of :meth:`pool_status`."""
        return await self._manager.pool_status.choose()  # type: ignore[no-any-return]

    def shutdown(self) -> None:
        """Shut down the resource manager and all workers."""
        if self._shutdown_done:
            return

        logger.info("Shutting down AgentBoxManager...")
        self._manager.shutdown.choose().get()

        self._fault_hook_stack.close()
        self._shutdown_done = True
        # shutdown_context() is process-global and terminal, so calling it
        # per-manager would prevent other managers from using the shared client.
        # Monarch handles that context's cleanup at normal interpreter exit.
        logger.info("AgentBoxManager shutdown complete")

    async def ashutdown(self) -> None:
        """Async-native counterpart of :meth:`shutdown` (awaits the shutdown RPCs)."""
        if self._shutdown_done:
            return

        logger.info("Shutting down AgentBoxManager...")
        await self._manager.shutdown.choose()

        self._fault_hook_stack.close()
        self._shutdown_done = True
        # As in shutdown(), leave the shared client to Monarch's exit handler.
        logger.info("AgentBoxManager shutdown complete")

    def __enter__(self) -> AgentBoxManager:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            self.shutdown()
        except BaseException as shutdown_error:
            if not isinstance(exc_value, BaseException):
                raise
            exc_value.add_note(
                "AgentBoxManager shutdown also failed: "
                f"{type(shutdown_error).__name__}: {shutdown_error}"
            )
            logger.error(
                "AgentBoxManager shutdown failed while preserving the active exception",
                exc_info=True,
            )

    async def __aenter__(self) -> AgentBoxManager:
        return self

    async def __aexit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None:
        try:
            await self.ashutdown()
        except BaseException as shutdown_error:
            if not isinstance(exc_value, BaseException):
                raise
            exc_value.add_note(
                "AgentBoxManager shutdown also failed: "
                f"{type(shutdown_error).__name__}: {shutdown_error}"
            )
            logger.error(
                "AgentBoxManager shutdown failed while preserving the active exception",
                exc_info=True,
            )
