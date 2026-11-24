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

"""Machine: a client-side handle for one remote Monarch worker.

A :class:`Machine` wraps a Monarch :class:`ProcMesh` and runs commands and
file transfers on that worker through the :class:`ExecutionTarget` API
(``run``, ``spawn``, ``upload``, ``download``)::

    from monarch.actor import this_host

    machine = Machine(proc_mesh=this_host().spawn_procs())
    print(machine.run("uname -a").stdout)

Actor classes here are registered with
:func:`cloudpickle.register_pickle_by_value`, so they always match the client's
code. Workers still import ``agentbox`` for process-wide state such as
the RDMA buffer pool; every backend's workers run from an environment that
has it.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import sys
import threading
import uuid
from typing import Any, TYPE_CHECKING

import cloudpickle
from agentbox.actors.mesh_schema import make_name
from agentbox.actors.target import ExecutionTarget
from agentbox.remote.file_transfer import (
    _acopy_file,
    _copy_file,
    FileCopyError,
    local_transfer_mesh,
    TransferEndpoints,
)
from monarch.actor import Actor, endpoint, ProcMesh


if TYPE_CHECKING:
    from agentbox.actors.container import Container
    from agentbox.actors.resource_managers import ResourceHandle
    from agentbox.tasks.api import TaskPool
    from monarch._src.actor.actor_mesh import ActorMesh


# Register so cloudpickle serializes actor classes by value to remote workers.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

logger = logging.getLogger(__name__)

_MESH_STOP_TIMEOUT = 10.0
# Index of each Machine, used to spread Machines over client-side transfer procs.
_transfer_shards = itertools.count()
_DEFAULT_TASK_POOL_SIZE = 16


class _MachineActor(Actor):
    """Lightweight actor running on the remote machine for metadata queries."""

    @endpoint
    def init(self) -> None:
        """Readiness probe; also loads agentbox in this (possibly fresh) proc.

        Actors are unpickled by value and import the agentbox modules they
        reference on first use. A Machine's first transfer spawns its source
        and sink actors together, and two threads importing the same modules
        at once can trip Python's import deadlock detection, aborting the
        proc and, through supervision, the client. The probe always runs
        first and alone, so importing here makes later spawns race-free.
        """
        import agentbox.remote.file_transfer  # noqa: F401
        import agentbox.remote.pty  # noqa: F401
        import agentbox.remote.subprocess  # noqa: F401
        import agentbox.tasks.pool  # noqa: F401

    @endpoint
    def hostname(self) -> str:
        """Return the hostname of the remote machine."""
        import socket

        return socket.gethostname()


# Every agentbox mesh holds one actor, so endpoints are invoked with choose(),
# which is equivalent to call_one() but avoids call_one()'s per-call native
# memory retention in torchmonarch.


class Machine(ExecutionTarget):
    """A remote worker host backed by a Monarch :class:`ProcMesh`.

    Commands run directly on the worker host; wrap the Machine in a
    :class:`~agentbox.actors.container.Container` for isolation.
    Actors for metadata, commands, and file transfer are spawned lazily.

    Args:
        proc_mesh: The Monarch ProcMesh connected to the remote machine.
        _record_init: When ``True`` (default), the constructor issues the
            worker-side readiness-probe RPC synchronously. When ``False``
            the caller is responsible for it; the async factories (:meth:`acreate`
            / :meth:`afrom_handle`) await :meth:`_arecord_init` themselves so it
            can run on an event loop.
        _metadata_actor_name: Schema name for the lazily-spawned metadata actor.
            When set (by the manager via :meth:`from_handle`), the actor — and thus
            any supervision fault it emits — carries this ``agentbox_`` name so the
            client fault hook can classify it.
    """

    def __init__(
        self,
        proc_mesh: ProcMesh,
        _record_init: bool = True,
        _metadata_actor_name: str | None = None,
    ) -> None:
        self._proc_mesh = proc_mesh
        self._metadata_actor_name = _metadata_actor_name
        self._actor_mesh: ActorMesh[_MachineActor] | None = None
        self._lifecycle_lock = threading.RLock()
        self._task_pool: TaskPool | None = None
        self._task_pool_lock = threading.Lock()
        self._closed = False
        self._id: str = f"m-{uuid.uuid4().hex[:8]}"
        self._resource_handle_id: str | None = None
        self._hostname_cache: str | None = None
        # File-transfer actors are spawned lazily on first use: one endpoint
        # pair on the shared client-side mesh and one on ``proc_mesh``.
        self._transfer: tuple[TransferEndpoints, TransferEndpoints] | None = None
        self._transfer_shard = next(_transfer_shards)
        # Whether this Machine *created* ``proc_mesh`` (via :meth:`from_handle`)
        # and is therefore responsible for tearing it down in :meth:`release`.
        # A caller-supplied proc mesh (the public constructor) is left alone.
        self._owns_proc_mesh = False
        # The OWNED host mesh backing ``proc_mesh``, set by :meth:`from_handle`
        # from the connector's :class:`~agentbox.actors.connector.ConnectResult`.
        # ``None`` for caller-supplied proc meshes (the public constructor),
        # whose host the caller owns.  Stopped alongside ``proc_mesh`` in
        # :meth:`release` — see :meth:`_stop_mesh` for why it cannot be
        # rederived from ``proc_mesh.host_mesh``.
        self._owned_host_mesh: object | None = None
        # Running containers on this host; release() stops any left running.
        self._containers: set[Container] = set()

        # Probe the worker eagerly so an unreachable mesh fails at construction.
        if _record_init:
            self._ensure_actor().init.choose().get()

    async def _arecord_init(self) -> None:
        """Await the worker-side readiness probe (async init path)."""
        await self._ensure_actor().init.choose()

    @classmethod
    def from_handle(
        cls,
        handle: "ResourceHandle",
        *,
        mesh_name: str | None = None,
        metadata_actor_name: str | None = None,
    ) -> Machine:
        """Create a :class:`Machine` from a :class:`ResourceHandle`.

        Combines :func:`~agentbox.actors.connector.connect` and
        ``Machine(proc_mesh=...)`` into a single call.  *metadata_actor_name* names
        the metadata actor (see :meth:`__init__`) so its faults are classifiable.
        """
        from agentbox.actors.connector import connect as _connect

        result = _connect(handle, mesh_name=mesh_name)
        try:
            machine = cls(
                proc_mesh=result.proc_mesh,
                _metadata_actor_name=metadata_actor_name,
            )
        except BaseException:
            # The readiness probe failed: don't leak the connection we just made.
            cls._stop_mesh(result.proc_mesh, host=result.host_mesh)
            raise
        # connect() created a fresh host mesh + proc mesh for this handle, so this
        # Machine owns them and must tear them down on release()/close().
        machine._owns_proc_mesh = True
        machine._owned_host_mesh = result.host_mesh
        machine._resource_handle_id = handle.handle_id
        if getattr(handle, "hostnames", None):
            machine._hostname_cache = handle.hostnames[0]
        return machine

    @classmethod
    async def acreate(
        cls,
        proc_mesh: ProcMesh,
    ) -> Machine:
        """Async-native counterpart of the ``Machine(proc_mesh=...)`` constructor.

        Awaits the machine-init RPC on the running event loop, so a Machine can
        be built from inside ``async`` code:
        ``machine = await Machine.acreate(proc_mesh=procs)``.
        """
        machine = cls(
            proc_mesh=proc_mesh,
            _record_init=False,
        )
        await machine._arecord_init()
        return machine

    @classmethod
    async def afrom_handle(
        cls,
        handle: "ResourceHandle",
        *,
        mesh_name: str | None = None,
        metadata_actor_name: str | None = None,
    ) -> Machine:
        """Async-native counterpart of :meth:`from_handle`.

        Awaits the worker-side init RPC on the running event loop, so it can be
        called from ``async`` code.  Used by
        :meth:`AgentBoxManager.aacquire_machine`.
        """
        from agentbox.actors.connector import connect as _connect

        result = _connect(handle, mesh_name=mesh_name)
        machine = cls(
            proc_mesh=result.proc_mesh,
            _record_init=False,
            _metadata_actor_name=metadata_actor_name,
        )
        machine._owns_proc_mesh = True
        machine._owned_host_mesh = result.host_mesh
        machine._resource_handle_id = handle.handle_id
        if getattr(handle, "hostnames", None):
            machine._hostname_cache = handle.hostnames[0]
        try:
            await machine._arecord_init()
        except BaseException:
            await machine.arelease()  # owned meshes: don't leak the new connection
            raise
        return machine

    @property
    def proc_mesh(self) -> ProcMesh:
        """The underlying Monarch :class:`ProcMesh`."""
        return self._proc_mesh

    @property
    def resource_handle_id(self) -> str | None:
        """Return the resource-manager handle currently assigned to this machine."""
        return self._resource_handle_id

    def _ensure_actor(self) -> ActorMesh[_MachineActor]:
        """Lazily spawn the metadata actor on first use."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Machine is closed")
            if self._actor_mesh is None:
                name = self._metadata_actor_name or make_name(
                    "machineinit", uuid.uuid4().hex[:8]
                )
                self._actor_mesh = self._proc_mesh.spawn(name, _MachineActor)
            return self._actor_mesh

    @property
    def _proc_mesh_for_exec(self) -> ProcMesh:
        return self._proc_mesh

    def _exec_task_pool(self) -> TaskPool:
        return self._ensure_task_pool()

    def _host_argv(
        self, argv: list[str], env: dict[str, str] | None, cwd: str | None
    ) -> tuple[list[str], str | None]:
        # ``env`` adds to (rather than replaces) the worker's environment.
        if env:
            argv = ["env", *(f"{k}={v}" for k, v in env.items()), *argv]
        return argv, cwd

    def _ensure_task_pool(self) -> TaskPool:
        """Lazily create the worker's shared single-thread async task pool."""
        with self._task_pool_lock:
            if self._closed:
                raise RuntimeError("Machine is closed")
            if self._task_pool is None:
                from agentbox.tasks.pool import _TaskPool
                from agentbox.tasks.registry import default_task_handlers

                self._task_pool = _TaskPool(
                    self._proc_mesh,
                    capacity=_DEFAULT_TASK_POOL_SIZE,
                    handlers=default_task_handlers.snapshot(),
                )
            return self._task_pool

    def _ensure_transfer(self) -> tuple[TransferEndpoints, TransferEndpoints]:
        """Return this Machine's (local, remote) transfer endpoints."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Machine is closed")
            if self._transfer is None:
                self._transfer = (
                    TransferEndpoints.spawn(local_transfer_mesh(self._transfer_shard)),
                    TransferEndpoints.spawn(self._proc_mesh),
                )
            return self._transfer

    async def _aensure_transfer(self) -> tuple[TransferEndpoints, TransferEndpoints]:
        """Async-native counterpart of :meth:`_ensure_transfer`."""
        if self._transfer is not None and not self._closed:
            return self._transfer
        # The first call may create the client-side mesh, which blocks.
        return await asyncio.to_thread(self._ensure_transfer)

    def warm_up(self) -> None:
        """Spawn and ping the file-transfer actors ahead of the first copy."""
        for endpoints in self._ensure_transfer():
            endpoints.ping()

    async def awarm_up(self) -> None:
        """Async-native counterpart of :meth:`warm_up`."""
        for endpoints in await self._aensure_transfer():
            await endpoints.aping()

    @staticmethod
    def _stop_mesh(
        mesh: Any, host: Any = None, *, timeout: float = _MESH_STOP_TIMEOUT
    ) -> bool:
        """Best-effort teardown of a proc mesh and, optionally, its owned host mesh.

        Each stop is bounded by *timeout*. Errors are swallowed (logged at
        debug): teardown runs in release/cleanup paths that must not raise, and
        the mesh may already be gone.

        Pass the OWNED host mesh — the ``host_mesh`` from the connector's
        :class:`~agentbox.actors.connector.ConnectResult`, one per
        ``attach_to_workers`` handle — as *host* to also stop it.  Pass
        ``host=None`` (the default) for the local file-transfer mesh, whose host
        is the process-wide ``this_host()`` shared by every Machine.

        Never derive the host from ``mesh.host_mesh``: that is a SLICE (via
        :meth:`ProcMesh._host` + ``.slice()``) which monarch's Rust side flags as
        a *reference*, not owned, and ``HostMesh.stop()`` on a reference raises
        ``"cannot stop `HostMesh` that is a reference instead of owned"`` — which
        flips the SlurmResourceManager actor to ``stopped: stop host`` and makes
        manager.shutdown() re-raise a :class:`SupervisionError` out of the
        cleanup path, failing the job at teardown.
        """
        if mesh is None:
            return True
        mesh_stopped = True
        try:
            mesh.stop().get(timeout=timeout)
        except Exception:
            mesh_stopped = False
            logger.debug("Error stopping proc mesh", exc_info=True)
        if host is not None:
            try:
                host.stop().get(timeout=timeout)
            except Exception:
                logger.debug("Error stopping host mesh", exc_info=True)
        return mesh_stopped

    @staticmethod
    async def _astop_mesh(
        mesh: Any, host: Any = None, *, timeout: float = _MESH_STOP_TIMEOUT
    ) -> bool:
        """Async-native, timeout-guarded counterpart of :meth:`_stop_mesh`."""
        if mesh is None:
            return True
        mesh_stopped = True
        try:
            await asyncio.wait_for(mesh.stop(), timeout=timeout)
        except Exception:
            mesh_stopped = False
            logger.debug("Error stopping proc mesh", exc_info=True)
        if host is not None:
            try:
                await asyncio.wait_for(host.stop(), timeout=timeout)
            except Exception:
                logger.debug("Error stopping host mesh", exc_info=True)
        return mesh_stopped

    def _take_containers(self) -> list[Container]:
        with self._lifecycle_lock:
            containers, self._containers = list(self._containers), set()
        return containers

    def release(self) -> None:
        """Stop this Machine's actors and, when owned, its remote meshes.

        Containers still running on it are stopped first: an Apptainer
        instance is a daemon outside the Monarch procs, so tearing down the
        mesh would leave it running into the worker's next lease.

        Idempotent: safe to call multiple times.  After ``release()`` on an
        owned Machine the underlying ProcMesh is gone and the Machine can no
        longer be used.
        """
        for container in self._take_containers():
            try:
                container.stop()
            except Exception:
                logger.warning(
                    "Error stopping container %s on release",
                    container.instance_name,
                    exc_info=True,
                )
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            transfer, self._transfer = self._transfer, None
            actor_mesh = self._actor_mesh
            self._actor_mesh = None

        with self._task_pool_lock:
            task_pool = self._task_pool
            self._task_pool = None

        if task_pool is not None:
            try:
                # Don't wait for in-flight commands: cancelling them kills
                # their processes, which must not outlive the Machine.
                task_pool.close(drain=False)
            except Exception:
                logger.debug("Error closing task pool", exc_info=True)
        for endpoints in transfer or ():
            endpoints.stop()
        self._stop_mesh(actor_mesh)
        # Remote mesh is only ours to stop when we created it via from_handle();
        # its owned host mesh travels with us as ``_owned_host_mesh``.
        if self._owns_proc_mesh:
            self._stop_mesh(self._proc_mesh, host=self._owned_host_mesh)

    async def arelease(self) -> None:
        """Async-native counterpart of :meth:`release` (awaits the mesh-stop RPCs)."""
        for container in self._take_containers():
            try:
                await container.astop()
            except Exception:
                logger.warning(
                    "Error stopping container %s on release",
                    container.instance_name,
                    exc_info=True,
                )
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            transfer, self._transfer = self._transfer, None
            actor_mesh = self._actor_mesh
            self._actor_mesh = None

        with self._task_pool_lock:
            task_pool = self._task_pool
            self._task_pool = None

        if task_pool is not None:
            try:
                await task_pool.aclose(drain=False)
            except Exception:
                logger.debug("Error closing task pool", exc_info=True)
        for endpoints in transfer or ():
            await endpoints.astop()
        await self._astop_mesh(actor_mesh)
        if self._owns_proc_mesh:
            await self._astop_mesh(self._proc_mesh, host=self._owned_host_mesh)

    def __enter__(self) -> "Machine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    async def __aenter__(self) -> "Machine":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.arelease()

    def hostname(self) -> str:
        """Return the hostname of the remote machine."""
        if self._hostname_cache is None:
            self._hostname_cache = self._ensure_actor().hostname.choose().get()
        return self._hostname_cache

    async def ahostname(self) -> str:
        """Async-native counterpart of :meth:`hostname` (awaits the RPC, caches)."""
        if self._hostname_cache is None:
            self._hostname_cache = await self._ensure_actor().hostname.choose()
        return self._hostname_cache

    @staticmethod
    def _check_local_source(local_path: str) -> str:
        local_path = os.path.abspath(local_path)
        if not (os.path.isfile(local_path) or os.path.isdir(local_path)):
            raise FileCopyError(f"Local file or directory not found: {local_path}")
        return local_path

    def upload(self, local_path: str, remote_path: str) -> None:
        local_path = self._check_local_source(local_path)
        local, remote = self._ensure_transfer()
        _copy_file(
            local.proc_mesh,
            remote.proc_mesh,
            local_path,
            remote_path,
            source_actor=local.source,
            sink_actor=remote.sink,
            source_archiver=local.archiver,
            sink_archiver=remote.archiver,
        )

    def download(self, remote_path: str, local_path: str) -> None:
        local, remote = self._ensure_transfer()
        _copy_file(
            remote.proc_mesh,
            local.proc_mesh,
            remote_path,
            os.path.abspath(local_path),
            source_actor=remote.source,
            sink_actor=local.sink,
            source_archiver=remote.archiver,
            sink_archiver=local.archiver,
        )

    async def aupload(self, local_path: str, remote_path: str) -> None:
        """Async-native counterpart of :meth:`upload`."""
        local_path = self._check_local_source(local_path)
        local, remote = await self._aensure_transfer()
        await _acopy_file(
            local.proc_mesh,
            remote.proc_mesh,
            local_path,
            remote_path,
            source_actor=local.source,
            sink_actor=remote.sink,
            source_archiver=local.archiver,
            sink_archiver=remote.archiver,
        )

    async def adownload(self, remote_path: str, local_path: str) -> None:
        """Async-native counterpart of :meth:`download`."""
        local, remote = await self._aensure_transfer()
        await _acopy_file(
            remote.proc_mesh,
            local.proc_mesh,
            remote_path,
            os.path.abspath(local_path),
            source_actor=remote.source,
            sink_actor=local.sink,
            source_archiver=remote.archiver,
            sink_archiver=local.archiver,
        )

    def copy_to(self, destination: Machine, remote_path: str, dest_path: str) -> None:
        """Copy *remote_path* on this machine to *dest_path* on *destination*.

        Directories are copied as in :meth:`upload`.

        The bytes go directly between the two workers, as in :meth:`upload`,
        never through the client.
        """
        _, source = self._ensure_transfer()
        _, sink = destination._ensure_transfer()
        _copy_file(
            source.proc_mesh,
            sink.proc_mesh,
            remote_path,
            dest_path,
            source_actor=source.source,
            sink_actor=sink.sink,
            source_archiver=source.archiver,
            sink_archiver=sink.archiver,
        )

    async def acopy_to(
        self, destination: Machine, remote_path: str, dest_path: str
    ) -> None:
        """Async-native counterpart of :meth:`copy_to`."""
        _, source = await self._aensure_transfer()
        _, sink = await destination._aensure_transfer()
        await _acopy_file(
            source.proc_mesh,
            sink.proc_mesh,
            remote_path,
            dest_path,
            source_actor=source.source,
            sink_actor=sink.sink,
            source_archiver=source.archiver,
            sink_archiver=sink.archiver,
        )
