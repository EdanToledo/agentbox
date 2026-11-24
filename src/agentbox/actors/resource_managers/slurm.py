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

"""Slurm job-array resource manager.

Submits Slurm job arrays where each array task is an independent single-node
job, providing per-node preemption resilience, non-blocking scaling, and
throttled parallelism.  See
:mod:`agentbox.actors.resource_managers.slurm_array` for the underlying
job-array primitives.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from agentbox.actors.resource_managers.base import ResourceManager
from agentbox.actors.resource_managers.slurm_array import (
    SlurmAllocation,
    SlurmArrayChunk,
)
from monarch.actor import endpoint


logger = logging.getLogger(__name__)

__all__ = ["SlurmResourceManager"]


class SlurmResourceManager(ResourceManager):
    """Resource manager backed by Slurm job arrays for per-node elasticity.

    Instead of submitting monolithic ``--nodes=N`` jobs, this manager
    submits job arrays where each array task is an independent single-node
    job.  This provides:

    - **Per-node preemption resilience**: losing one node does not kill the
      entire allocation.
    - **Non-blocking scaling**: new array chunks can be submitted without
      waiting for the previous ones to start.
    - **Throttled parallelism**: ``max_parallel`` limits how many array
      tasks run concurrently, preventing cluster spam.

    Args:
        allocation: Slurm resource specification (CPUs, GPUs, partition, etc.).
        initial_nodes: Total number of nodes to request initially.
        chunk_size: Array tasks per chunk (defaults to *initial_nodes*).
        max_parallel: Maximum concurrently running array tasks per chunk.
        python_exe: Python executable path for Slurm worker scripts.
    """

    def __init__(
        self,
        allocation: SlurmAllocation | None = None,
        initial_nodes: int = 1,
        chunk_size: int | None = None,
        max_parallel: int = 128,
        python_exe: str = "python",
        fault_scope: str | None = None,
    ) -> None:
        """Initialize a resource pool backed by Slurm job arrays.

        Args:
            allocation: Slurm resource requirements for each worker.
            initial_nodes: Number of workers to request immediately.
            chunk_size: Number of array tasks per submitted chunk. Defaults to
                ``initial_nodes``.
            max_parallel: Maximum number of concurrently running tasks per chunk.
            python_exe: Python executable used to bootstrap workers.
            fault_scope: Unique scope encoded in supervision mesh names so faults
                can be routed to the manager that owns this pool.
        """
        super().__init__(fault_scope=fault_scope)

        from agentbox.port_server import PortRegistrationServer

        self._allocation = allocation or SlurmAllocation()
        self._max_parallel = max_parallel
        self._chunk_size = chunk_size or initial_nodes
        self._python_exe = python_exe

        self._port_server = PortRegistrationServer()

        self._chunks: list[SlurmArrayChunk] = []
        # Internal slot key: "{array_job_id}_{task_id}" — globally unique even
        # when two tasks land on the same physical hostname.
        self._slot_to_host: dict[str, str] = {}  # slot key → hostname
        self._slot_to_chunk: dict[str, SlurmArrayChunk] = {}  # slot key → chunk
        self._slot_to_task_id: dict[str, int] = {}  # slot key → task_id within chunk
        self._slot_to_port: dict[str, int] = {}  # slot key → Monarch worker port

        # Submit initial chunks.
        n_chunks = math.ceil(initial_nodes / self._chunk_size)
        for i in range(n_chunks):
            remaining = initial_nodes - i * self._chunk_size
            size = min(self._chunk_size, remaining)
            self._submit_chunk(size)

    def _submit_chunk(self, size: int) -> None:
        """Create and submit a new :class:`SlurmArrayChunk`."""
        chunk = SlurmArrayChunk(
            chunk_size=size,
            allocation=self._allocation,
            max_parallel=self._max_parallel,
            python_exe=self._python_exe,
            port_server_addr=self._port_server.address,
        )
        chunk.submit()
        self._chunks.append(chunk)
        logger.info(
            "Submitted new array chunk (size=%d, job_id=%s)", size, chunk.array_job_id
        )

    # Subclass-only endpoints

    @endpoint
    async def poll_pending(self) -> dict[str, Any]:
        """Discover newly running array tasks and add their hosts to the pool.

        Returns a dict with ``new_hosts`` (list of ``{"hostname": …, "port": …}``
        dicts), ``total_pending``, and ``total_running`` counts.
        """
        new_hosts: list[dict[str, Any]] = []
        total_pending = 0
        total_running = 0

        for chunk in self._chunks:
            newly_running = chunk.get_newly_running(
                port_lookup=lambda jid, tid: self._port_server.try_get_port(jid, tid)
            )
            for task_id, hostname, port in newly_running:
                slot_key = f"{chunk.array_job_id}_{task_id}"
                self._available.append(slot_key)
                self._slot_to_host[slot_key] = hostname
                self._slot_to_chunk[slot_key] = chunk
                self._slot_to_task_id[slot_key] = task_id
                self._slot_to_port[slot_key] = port
                new_hosts.append({"hostname": hostname, "port": port})
                tcp_addr = f"tcp://{hostname}:{port}"
                self._register_worker_for_supervision(slot_key, [tcp_addr])

            total_pending += chunk.pending_count
            total_running += chunk.running_count

        if new_hosts:
            logger.info(
                "Discovered %d new host(s): %s",
                len(new_hosts),
                [h["hostname"] for h in new_hosts],
            )

        return {
            "new_hosts": new_hosts,
            "total_pending": total_pending,
            "total_running": total_running,
        }

    def _build_hostnames(self, slot_keys: list[str]) -> list[str]:
        return [self._slot_to_host[sk] for sk in slot_keys]

    def _build_addresses(self, slot_keys: list[str]) -> list[str]:
        return [
            f"tcp://{self._slot_to_host[sk]}:{self._slot_to_port[sk]}"
            for sk in slot_keys
        ]

    def _build_acquire_metadata(self, slot_keys: list[str]) -> dict[str, Any]:
        return {
            "host_to_port": {
                self._slot_to_host[sk]: self._slot_to_port[sk] for sk in slot_keys
            },
            "host_to_array_job_id": {
                self._slot_to_host[sk]: self._slot_to_chunk[sk].array_job_id
                for sk in slot_keys
            },
            "host_to_task_id": {
                self._slot_to_host[sk]: self._slot_to_task_id[sk] for sk in slot_keys
            },
        }

    def _extra_pool_status(self) -> dict[str, Any]:
        total_pending = sum(c.pending_count for c in self._chunks)
        return {
            "pending_tasks": total_pending,
            "active_array_jobs": [
                c.array_job_id for c in self._chunks if c.array_job_id
            ],
        }

    def _do_spawn(self, n: int) -> None:
        self._submit_chunk(n)

    def _cancel_failed_task(self, slot_key: str) -> None:
        """Request teardown while retaining metadata until Slurm confirms it."""
        chunk = self._slot_to_chunk.get(slot_key)
        task_id = self._slot_to_task_id.get(slot_key)
        if chunk is None or task_id is None:
            return
        try:
            chunk.cancel_task(task_id)
        except Exception:
            logger.warning(
                "Failed to cancel failed task %s_%s",
                chunk.array_job_id,
                task_id,
                exc_info=True,
            )

    def _on_worker_failure(self, slot_key: str, failure: Any) -> None:
        self._cancel_failed_task(slot_key)

    def _on_worker_failure_external(self, slot_key: str, reason: str) -> None:
        self._cancel_failed_task(slot_key)

    async def _do_kill(self, slot_keys: list[str]) -> None:
        await self._stop_monitor_meshes(*slot_keys)
        removed_hosts = [self._slot_to_host.get(sk, sk) for sk in slot_keys]

        for slot_key in slot_keys:
            chunk = self._slot_to_chunk.pop(slot_key, None)
            task_id = self._slot_to_task_id.pop(slot_key, None)
            self._slot_to_host.pop(slot_key, None)
            self._slot_to_port.pop(slot_key, None)

            if chunk is not None and task_id is not None:
                try:
                    chunk.cancel_task(task_id)
                except Exception:
                    logger.warning(
                        "Failed to cancel task %s_%s",
                        chunk.array_job_id,
                        task_id,
                        exc_info=True,
                    )

            # If no remaining slots reference this chunk, remove it.
            if chunk is not None and chunk not in self._slot_to_chunk.values():
                try:
                    chunk.cancel_all()
                except Exception:
                    logger.warning(
                        "Failed to cancel chunk %s", chunk.array_job_id, exc_info=True
                    )
                if chunk in self._chunks:
                    self._chunks.remove(chunk)

        logger.info("Scaled down by %d hosts: %s", len(slot_keys), removed_hosts)

    def _do_health_check(
        self,
    ) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
        evicted_hosts: list[str] = []
        dead_tasks: list[dict[str, Any]] = []
        dead_keys: list[str] = []

        for chunk in self._chunks:
            for task_id, hostname in chunk.get_dead_tasks():
                dead_tasks.append(
                    {
                        "array_job_id": chunk.array_job_id,
                        "task_id": task_id,
                        "hostname": hostname,
                    }
                )
                slot_key = f"{chunk.array_job_id}_{task_id}"
                self._slot_to_chunk.pop(slot_key, None)
                self._slot_to_task_id.pop(slot_key, None)
                self._slot_to_host.pop(slot_key, None)
                self._slot_to_port.pop(slot_key, None)
                # Only report as evicted if slot was idle (not checked out).
                if slot_key in self._available and hostname:
                    evicted_hosts.append(hostname)
                dead_keys.append(slot_key)

        healthy_hosts = [
            hostname
            for slot_key, hostname in self._slot_to_host.items()
            if slot_key not in self._failed_slots
        ]

        # Remove chunks that are fully exhausted (no pending or running tasks).
        exhausted = [
            c
            for c in self._chunks
            if c.is_done and not any(v is c for v in self._slot_to_chunk.values())
        ]
        for c in exhausted:
            logger.info("Removing exhausted chunk %s from tracking", c.array_job_id)
            self._chunks.remove(c)

        return dead_keys, evicted_hosts, healthy_hosts, {"dead_tasks": dead_tasks}

    def _do_shutdown(self) -> None:
        logger.info("Shutting down %d array chunk(s)", len(self._chunks))
        for chunk in self._chunks:
            try:
                chunk.cancel_all()
            except Exception:
                logger.warning(
                    "Failed to cancel chunk %s", chunk.array_job_id, exc_info=True
                )
        self._chunks.clear()
        self._slot_to_host.clear()
        self._slot_to_chunk.clear()
        self._slot_to_task_id.clear()
        self._slot_to_port.clear()
        self._slot_to_monitor_mesh.clear()
        self._mesh_name_to_slot.clear()
        self._port_server.shutdown()
        logger.info("Shutdown complete")
