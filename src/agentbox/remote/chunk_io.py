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

"""Parallel file I/O and hashing for streamed file transfer.

A single SHA-256 stream runs at about 2 GB/s, well below an RDMA NIC.
Transfers therefore split every chunk into fixed :data:`PIECE_SIZE` pieces
aligned to file offsets and hash (and read) the pieces in parallel on
process-wide thread pools (``hashlib`` and file I/O release the GIL). File I/O
and hashing use separate pools so that writes, usually the bottleneck, never
queue behind hashing.

The transfer digest is SHA-256 over a format tag, the file size, and the
ordered SHA-256 digests of every piece. It is as collision-resistant as
SHA-256, and because pieces sit at fixed offsets, both ends compute the same
value whatever chunk sizes the bytes travelled in.

Large chunks written from page-aligned memory can bypass the page cache
(``O_DIRECT``) on local block filesystems, where buffered writes to one file
serialize on its inode lock far below the device's bandwidth; see
:func:`direct_io_filesystem`.

This module is imported by reference (not pickled by value) so that all
transfer actors in a process share its pools.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal


PIECE_SIZE: int = 4 * 1024 * 1024
"""Bytes per independently hashed piece; chunk sizes must be multiples."""

DIRECT_IO_ALIGN: int = 4096
"""Offset, length, and address alignment for ``O_DIRECT`` writes."""

_DIRECT_IO_FILESYSTEMS = frozenset({"ext4", "xfs"})
_MAX_HASH_WORKERS = 64
_MAX_IO_WORKERS = 16
_DIGEST_TAG = b"agentbox-transfer-v1\0"

type PoolKind = Literal["hash", "io"]
_executors: dict[PoolKind, ThreadPoolExecutor] = {}
_executors_lock = threading.Lock()


def executor(kind: PoolKind) -> ThreadPoolExecutor:
    """Return this process's *kind* pool, sized to its usable CPUs.

    Jobs never wait on other jobs, so the pools cannot deadlock.
    """
    with _executors_lock:
        if kind not in _executors:
            cpus = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count()
            ) or 2
            limit = _MAX_HASH_WORKERS if kind == "hash" else _MAX_IO_WORKERS
            _executors[kind] = ThreadPoolExecutor(
                max_workers=max(2, min(limit, cpus)),
                thread_name_prefix=f"agentbox-transfer-{kind}",
            )
        return _executors[kind]


def direct_io_filesystem(path: str) -> bool:
    """Whether *path* is on a local block filesystem that suits ``O_DIRECT``.

    Elsewhere (``tmpfs``, network and FUSE filesystems) direct I/O is slower
    or has different semantics, so writes stay buffered.
    """
    target = os.path.realpath(os.path.dirname(path) or ".")
    best, fstype = "", ""
    try:
        with open("/proc/self/mountinfo") as mounts:
            for line in mounts:
                fields = line.split()
                if "-" not in fields:
                    continue
                mount_point = fields[4].replace("\\040", " ")
                within = target == mount_point or target.startswith(
                    mount_point.rstrip("/") + "/"
                )
                if within and len(mount_point) >= len(best):
                    best, fstype = mount_point, fields[fields.index("-") + 1]
    except OSError:
        return False
    return fstype in _DIRECT_IO_FILESYSTEMS


class PieceDigests:
    """Per-piece SHA-256 digests of one file, filled in any order."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._digests: list[bytes | None] = [None] * -(-size // PIECE_SIZE)

    def hash_piece(self, view: memoryview, offset: int) -> None:
        """Record the digest of the piece that starts at *offset*."""
        self._digests[offset // PIECE_SIZE] = hashlib.sha256(view).digest()

    def hexdigest(self) -> str:
        """Return the transfer digest; every piece must have been hashed."""
        if any(d is None for d in self._digests):
            raise RuntimeError(
                "transfer digest requested before every piece was hashed"
            )
        root = hashlib.sha256(_DIGEST_TAG)
        root.update(self._size.to_bytes(8, "little"))
        for digest in self._digests:
            root.update(digest)  # type: ignore[arg-type]
        return root.hexdigest()


def submit_pieces(
    view: memoryview,
    offset: int,
    work: Callable[[memoryview, int], None],
    kind: PoolKind = "hash",
) -> list[Future[None]]:
    """Run ``work(piece_view, piece_offset)`` on the *kind* pool per piece of *view*.

    *view* holds the file bytes starting at *offset*, which must be
    piece-aligned. Every job owns its slice and releases it when done.
    """
    if offset % PIECE_SIZE:
        raise ValueError(f"chunk offset {offset} is not a multiple of {PIECE_SIZE}")
    pool = executor(kind)

    def run(start: int) -> None:
        with view[start : start + PIECE_SIZE] as piece:
            work(piece, offset + start)

    return [pool.submit(run, start) for start in range(0, len(view), PIECE_SIZE)]


def wait_all(futures: list[Future[None]]) -> None:
    """Wait for every future, then raise the first failure, if any."""
    error: BaseException | None = None
    for future in futures:
        exc = future.exception()
        if exc is not None and error is None:
            error = exc
    if error is not None:
        raise error


def pread_exact(fd: int, view: memoryview, offset: int) -> None:
    """Fill *view* from *fd* at *offset*, raising if the file ends early."""
    filled = 0
    while filled < len(view):
        with view[filled:] as rest:
            count = os.preadv(fd, [rest], offset + filled)
        if count <= 0:
            raise EOFError(f"file ended at byte {offset + filled}")
        filled += count


def pwrite_all(fd: int, view: memoryview, offset: int) -> None:
    """Write all of *view* to *fd* at *offset*."""
    written = 0
    while written < len(view):
        with view[written:] as rest:
            count = os.pwrite(fd, rest, offset + written)
        if count <= 0:
            raise OSError(f"short write at byte {offset + written}")
        written += count
