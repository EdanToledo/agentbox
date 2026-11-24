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

"""Process-wide registered memory for RDMA file transfer.

Registering memory with the NIC pins it and costs milliseconds per chunk, so
every process that moves file bytes over RDMA keeps a small, bounded pool of
registered chunk buffers and reuses them for its lifetime. Source and sink
roles use separate pools: a process can be the source of one transfer and the
sink of another, and a sink must never wait for a slot that a source in the
same process holds while it waits on that sink.

This module is imported by reference (not pickled by value) so that all
transfer actors in a process share one pool.
"""

from __future__ import annotations

import logging
import mmap
import os
import threading
from dataclasses import dataclass
from typing import Any, Literal


logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 64 * 1024 * 1024
"""Bytes per registered slot, i.e. per RDMA read."""

BATCH_CHUNKS: int = 4
"""Chunks read with one RDMA submission. Monarch runs a submission's reads in
parallel (striped across NICs), and a NIC only reaches line rate with several
large reads in flight."""

BATCHES_IN_FLIGHT: int = 2
"""Batches a source keeps outstanding, so one is read while the next fills."""

MAX_POOL_BYTES: int = 4 * 1024**3
"""Most registered memory a process keeps per role for reuse."""

POOL_MEMORY_FRACTION: int = 64
"""A role's pool may keep at most 1/64 of the process's memory limit."""

MEMORY_FRACTION_LIMIT: int = 64
"""A role may hold at most 1/64 of the memory limit registered at once per
process (reusable plus in-flight one-off slots), so several transfer procs
sharing one cgroup together hold only a small fraction of it. Beyond that
:meth:`SlotPool.acquire` raises :class:`SlotBudgetExhausted` and the transfer
uses Ethernet instead, so concurrent large transfers cannot exhaust memory."""

Role = Literal["source", "sink"]


class SlotBudgetExhausted(RuntimeError):
    """The role's registered-memory budget is in use; retry over Ethernet."""


def read_all(reads: list[tuple[Any, memoryview]], timeout: int) -> None:
    """Read every ``(remote_handle, local_view)`` pair in one parallel submission."""
    from monarch.rdma import RDMAAction

    action = RDMAAction()
    for handle, view in reads:
        action.read_remote(view, handle)
    action.submit(timeout=timeout).get()


def backend() -> str:
    """Return Monarch's RDMA backend here: ``ibverbs``, ``tcp``, or ``none``."""
    try:
        from monarch.rdma import get_rdma_backend

        return str(get_rdma_backend())
    except Exception:
        return "none"


@dataclass(eq=False)
class Slot:
    """One registered chunk buffer.

    ``handle`` is an ``RDMABuffer`` over the whole of ``view``. It keeps the
    registration alive, and for source slots it is what the sink reads from.
    Monarch caches registrations by memoryview identity, so RDMA must go
    through ``view`` itself; slices of it are fine for plain file I/O.
    """

    buffer: mmap.mmap
    view: memoryview
    handle: Any
    pooled: bool = True


class SlotPool:
    """Registered :class:`Slot` objects, reused up to a fixed capacity.

    :meth:`acquire` never waits. Transfer actors share a process, and an
    actor blocked on a slot that another actor in the same process holds can
    stall the very calls that would free it. Beyond capacity, a slot is
    registered for one chunk and released with it, so pinned memory stays at
    ``capacity`` slots in steady state and overflow is bounded by the chunks
    in flight, as for messages.
    """

    def __init__(self, capacity: int, size: int, limit: int | None = None) -> None:
        self._capacity = capacity
        self._size = size
        # Most slots (pooled + overflow) registered at once; None = unbounded.
        self._limit = None if limit is None else max(capacity, limit)
        self._free: list[Slot] = []
        self._created = 0
        self._overflow = 0
        self._lock = threading.Lock()

    def acquire(self) -> Slot:
        """Return a free pooled slot, or a newly registered one.

        Raises:
            SlotBudgetExhausted: If registering another slot would exceed the
                pool's limit.
        """
        with self._lock:
            if self._free:
                return self._free.pop()
            pooled = self._created < self._capacity
            if pooled:
                self._created += 1
            elif (
                self._limit is not None
                and self._created + self._overflow >= self._limit
            ):
                raise SlotBudgetExhausted(
                    f"{self._created + self._overflow} slots registered (limit {self._limit})"
                )
            else:
                self._overflow += 1
        try:
            return self._register(pooled)
        except BaseException:
            with self._lock:
                if pooled:
                    self._created -= 1
                else:
                    self._overflow -= 1
            raise

    def release(self, slot: Slot) -> None:
        """Return a pooled *slot* for reuse; unregister an overflow slot."""
        if slot.pooled:
            with self._lock:
                self._free.append(slot)
            return
        # Dropping the last Python reference does not deregister the memory:
        # the native registration keeps the chunk pinned and resident, so
        # every overflow chunk would leak CHUNK_SIZE bytes for the life of the
        # process. Release the registration explicitly, then the mapping. The
        # slot counts against the limit until then (forever if the drop fails).
        try:
            slot.handle.drop().get()
        except Exception:
            logger.warning("Failed to release RDMA registration", exc_info=True)
            return
        with self._lock:
            self._overflow -= 1
        try:
            slot.view.release()
            slot.buffer.close()
        except BufferError:
            pass  # a write still references the chunk; the GC unmaps it later

    def _register(self, pooled: bool) -> Slot:
        from monarch.rdma import RDMABuffer

        # Anonymous, writable (read-only memory cannot be registered), and
        # page-aligned, so that chunks can be written with O_DIRECT.
        buffer = mmap.mmap(-1, self._size)
        view = memoryview(buffer)
        return Slot(buffer, view, RDMABuffer(view), pooled)


def _memory_limit() -> int:
    """Return physical memory, or the tightest cgroup (v2) memory limit if lower.

    The limit is usually set on an ancestor (e.g. a Slurm job's cgroup) while
    the process's own cgroup reads ``max``, so every level up to the root counts.
    """
    limit = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    try:
        with open("/proc/self/cgroup") as cgroup:
            path = next(line for line in cgroup if line.startswith("0::"))[3:].strip()
    except (OSError, StopIteration):
        return limit
    while True:
        try:
            with open(f"/sys/fs/cgroup{path}/memory.max") as memory_max:
                raw = memory_max.read().strip()
            if raw != "max":
                limit = min(limit, int(raw))
        except (OSError, ValueError):
            pass
        if path in ("", "/"):
            return limit
        path = os.path.dirname(path)


def pool_capacity() -> int:
    """Slots per role this process keeps registered for reuse.

    Concurrent transfers need slots for every chunk they have in flight;
    beyond the pool each chunk pays a fresh registration, which caps the
    aggregate rate. The pool therefore grows (lazily) to a share of memory:
    at least enough for one transfer at full speed, at most
    :data:`MAX_POOL_BYTES`.
    """
    one_transfer = 2 * BATCH_CHUNKS * BATCHES_IN_FLIGHT  # source or sink side
    budget = min(MAX_POOL_BYTES, _memory_limit() // POOL_MEMORY_FRACTION)
    return max(one_transfer, budget // CHUNK_SIZE)


def pool_limit() -> int:
    """Most slots per role registered at once (see :data:`MEMORY_FRACTION_LIMIT`)."""
    return max(pool_capacity(), _memory_limit() // MEMORY_FRACTION_LIMIT // CHUNK_SIZE)


_pools: dict[Role, SlotPool] = {}
_pools_lock = threading.Lock()


def pool(role: Role) -> SlotPool:
    """Return this process's slot pool for *role*."""
    with _pools_lock:
        if role not in _pools:
            _pools[role] = SlotPool(pool_capacity(), CHUNK_SIZE, pool_limit())
        return _pools[role]
