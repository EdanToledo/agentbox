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

"""Local resource manager for development and testing.

Spawns Monarch worker processes on localhost via :mod:`subprocess`,
following the same pattern as Monarch's :class:`ProcessJob`.  This
enables local development and testing without a scheduler.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from agentbox.actors.resource_managers.base import ResourceManager
from agentbox.process import (
    LOCAL_WORKER_ADDRESS_ENV,
    LOCAL_WORKER_PROCESS_MARKER,
    terminate_process,
)


logger = logging.getLogger(__name__)

__all__ = ["LocalResourceManager"]


@dataclass
class LocalWorkerState:
    """Tracks a single local worker subprocess."""

    slot_key: str
    address: str
    pid: int
    process: subprocess.Popen[bytes]


class LocalResourceManager(ResourceManager):
    """Resource manager that spawns Monarch workers as local subprocesses.

    Each worker communicates over a Unix IPC socket.  Suitable for local
    development and testing of multi-host scenarios without SSH or a
    scheduler.

    Args:
        initial_nodes: Number of worker processes to spawn on construction.
        env: Extra environment variables for worker subprocesses.
    """

    def __init__(
        self,
        initial_nodes: int = 1,
        env: dict[str, str] | None = None,
        fault_scope: str | None = None,
    ) -> None:
        """Initialize a pool of local worker subprocesses.

        Args:
            initial_nodes: Number of worker subprocesses to start immediately.
            env: Additional environment variables for each worker subprocess.
            fault_scope: Unique scope encoded in supervision mesh names so faults
                can be routed to the manager that owns this pool.
        """
        super().__init__(fault_scope=fault_scope)
        self._workers: dict[str, LocalWorkerState] = {}
        self._next_index: int = 0
        self._env = env
        self._tmpdir = tempfile.mkdtemp(prefix="monarch_local_rm_")

        self._spawn_workers(initial_nodes)

    def _spawn_workers(self, n: int) -> None:
        """Spawn *n* new worker subprocesses."""
        # Keep this import lazy so ``python -m agentbox.local_worker_entry`` does
        # not preload its target through ``agentbox.__init__``, while still making
        # the dependency explicit rather than relying on a string-only import.
        from agentbox import local_worker_entry

        worker_module = local_worker_entry.__name__
        spawned: list[str] = []
        try:
            for _ in range(n):
                slot_key = f"local_{self._next_index}"
                self._next_index += 1
                addr = f"ipc://{self._tmpdir}/{slot_key}"

                env = {**os.environ, "HYPERACTOR_PROCESS_NAME": slot_key}
                if self._env is not None:
                    env.update(self._env)

                env[LOCAL_WORKER_ADDRESS_ENV] = addr
                cmd = [
                    sys.executable,
                    "-m",
                    worker_module,
                    LOCAL_WORKER_PROCESS_MARKER,
                ]
                # Give each worker its own stdio. Without this the worker
                # inherits the parent's stdout/stderr; since workers are
                # detached (start_new_session=True) and outlive a single
                # allocation, a worker holding an inherited *pipe* keeps it open
                # after the parent exits. That hangs any parent that drains that
                # pipe to EOF before exiting — most notably ``conda run``, whose
                # output-proxy never sees EOF, so ``conda run agentbox-script``
                # appears to hang forever even though the script finished.
                # Routing worker output to a per-worker log file (stdin from
                # /dev/null) releases the inherited descriptors and keeps the
                # logs available for debugging.
                log_path = os.path.join(self._tmpdir, f"{slot_key}.log")
                log_fh = open(log_path, "ab")  # noqa: SIM115 (closed below; child keeps its dup)
                try:
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh,
                        stderr=log_fh,
                    )
                finally:
                    log_fh.close()

                worker = LocalWorkerState(
                    slot_key=slot_key,
                    address=addr,
                    pid=proc.pid,
                    process=proc,
                )
                self._workers[slot_key] = worker
                self._available.append(slot_key)
                spawned.append(slot_key)
                logger.info(
                    "Spawned local worker %s (pid=%d, addr=%s)",
                    slot_key,
                    proc.pid,
                    addr,
                )
                self._register_worker_for_supervision(slot_key, [addr])
        except Exception:
            # Terminate and reap all workers spawned in this batch on failure.
            for key in spawned:
                w = self._workers.pop(key, None)
                if w is not None:
                    terminate_process(w.process)
                    if key in self._available:
                        self._available.remove(key)
            raise

    def _build_addresses(self, slot_keys: list[str]) -> list[str]:
        return [self._workers[sk].address for sk in slot_keys]

    def _build_acquire_metadata(self, slot_keys: list[str]) -> dict[str, Any]:
        return {
            "pids": {
                self._workers[sk].address: self._workers[sk].pid for sk in slot_keys
            },
        }

    def _do_spawn(self, n: int) -> None:
        self._spawn_workers(n)

    async def _do_kill(self, slot_keys: list[str]) -> None:
        await self._stop_monitor_meshes(*slot_keys)
        for slot_key in slot_keys:
            worker = self._workers.pop(slot_key, None)
            if worker is not None:
                terminate_process(worker.process)
                logger.info("Terminated local worker %s (pid=%d)", slot_key, worker.pid)

    def _do_health_check(
        self,
    ) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
        evicted_hosts: list[str] = []
        dead_pids: list[int] = []
        dead_keys: list[str] = []

        for slot_key, worker in list(self._workers.items()):
            # poll() reaps an exited child (preventing a zombie) and reports death.
            if worker.process.poll() is not None:
                dead_keys.append(slot_key)
                dead_pids.append(worker.pid)
                evicted_hosts.append(worker.address)
                logger.warning(
                    "Health check: worker %s (pid=%d) is dead", slot_key, worker.pid
                )

        for slot_key in dead_keys:
            self._workers.pop(slot_key, None)

        healthy_hosts = [w.address for w in self._workers.values()]
        return dead_keys, evicted_hosts, healthy_hosts, {"dead_pids": dead_pids}

    def _do_shutdown(self) -> None:
        logger.info("Shutting down %d local worker(s)", len(self._workers))
        for worker in self._workers.values():
            terminate_process(worker.process)
        self._workers.clear()
        self._slot_to_monitor_mesh.clear()
        self._mesh_name_to_slot.clear()
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        logger.info("Shutdown complete")
