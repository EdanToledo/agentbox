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

"""Actor-to-actor file copy between local and remote Monarch ProcMeshes.

Provides actors and orchestration for streaming file bytes directly from a
source actor to a sink actor, with SHA-256 verification and bounded retries.
Chunks travel one of two ways:

* **Ethernet**: chunk bytes ride in Monarch messages.
* **RDMA**: the source reads each chunk into registered memory and sends only
  a handle; the sink pulls the bytes with a one-sided RDMA read
  (:mod:`agentbox.remote.rdma` owns the registered buffers).

``AGENTBOX_FILE_TRANSPORT`` selects the transport (see :func:`_file_transport`).
Both transports share one sink state machine and the same end-to-end digest
check: the sink hashes the bytes it writes, in order, and a file is only
accepted when that digest matches the source's.

A directory travels as one tar archive: the source packs it into a temporary
file (:func:`agentbox.remote.tree_archive.pack`), that file is copied like any
other, and the sink unpacks it in place
(:func:`agentbox.remote.tree_archive.unpack`).

The main entry point is :func:`_copy_file`, used by
``Machine.upload`` and ``Machine.download``.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import stat
import sys
import tempfile
import threading
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, wait as _wait_futures
from dataclasses import dataclass, field
from time import monotonic as _monotonic
from typing import Any, cast, Literal

import cloudpickle
from agentbox.remote import chunk_io as _io, rdma as _rdma, tree_archive
from monarch.actor import Actor, endpoint, ProcMesh


# Register so cloudpickle serializes actor classes by value to remote workers.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

logger = logging.getLogger(__name__)

PICKLE_CHUNK_SIZE: int = 8 * 1024 * 1024  # 8 MiB per Ethernet message
DEFAULT_PIPELINE_DEPTH: int = 4
_COPY_MAX_RETRIES: int = 3  # max end-to-end transfer attempts (initial + retries)
_PIPELINE_DEPTH_ENV = "AGENTBOX_TRANSFER_PIPELINE_DEPTH"
_MAX_PIPELINE_DEPTH = 64
_TRANSFER_CLEANUP_TIMEOUT = 10.0  # total best-effort error cleanup budget

type Transport = Literal["auto", "ethernet", "rdma"]
_TRANSPORT_ENV = "AGENTBOX_FILE_TRANSPORT"
RDMA_MIN_FILE_SIZE: int = 64 * 1024 * 1024  # smaller files use Ethernet under auto
_RDMA_TIMEOUT = 30  # seconds for one RDMA chunk read


def _file_transport() -> Transport:
    """Return the transport selected by ``AGENTBOX_FILE_TRANSPORT``.

    * ``auto`` (default): RDMA for files of at least :data:`RDMA_MIN_FILE_SIZE`
      when both ends have native ibverbs RDMA, else Ethernet. If an RDMA
      transfer fails, that file is resent over Ethernet and the source actor
      stops trying RDMA.
    * ``ethernet``: never RDMA.
    * ``rdma``: always RDMA, over whatever backend Monarch provides (including
      its TCP emulation); failures raise instead of falling back.
    """
    raw = os.environ.get(_TRANSPORT_ENV, "auto").strip().lower()
    if raw in ("auto", "ethernet", "rdma"):
        return cast(Transport, raw)
    logger.warning("Ignoring invalid %s=%r; using auto", _TRANSPORT_ENV, raw)
    return "auto"


def _pipeline_depth() -> int:
    """Return the bounded number of chunks a transfer keeps in flight."""
    raw = os.environ.get(_PIPELINE_DEPTH_ENV)
    if raw is None:
        return DEFAULT_PIPELINE_DEPTH
    try:
        depth = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %d",
            _PIPELINE_DEPTH_ENV,
            raw,
            DEFAULT_PIPELINE_DEPTH,
        )
        return DEFAULT_PIPELINE_DEPTH
    if depth < 1:
        logger.warning(
            "Ignoring non-positive %s=%r; using %d",
            _PIPELINE_DEPTH_ENV,
            raw,
            DEFAULT_PIPELINE_DEPTH,
        )
        return DEFAULT_PIPELINE_DEPTH
    return min(depth, _MAX_PIPELINE_DEPTH)


@dataclass
class FileMetadata:
    """Metadata about a file to be transferred."""

    size: int
    mode: int
    filename: str
    is_dir: bool = False


@dataclass(frozen=True)
class FileTransferDigests:
    """Source and sink digests computed by a direct streamed transfer."""

    source_sha256: str
    sink_sha256: str
    transport: Literal["ethernet", "rdma"] = "ethernet"


@dataclass
class _SinkByteTransferState:
    fd: int
    path: str
    expected_size: int
    max_pending: int  # accepted chunks whose writes may still be running
    digests: _io.PieceDigests
    direct_fd: int | None = None  # O_DIRECT descriptor for registered chunks
    next_offset: int = 0
    rdma_error: str | None = None  # first RDMA failure; later pulls repeat it
    # Background write-and-hash jobs per accepted chunk, oldest first, with the
    # registered slot (if any) that they read from.
    pending: deque[tuple[list[Future[None]], _rdma.Slot | None]] = field(
        default_factory=deque
    )


class FileCopyError(RuntimeError):
    """Raised when a file copy operation fails."""


class _RdmaError(FileCopyError):
    """An RDMA-specific failure that an Ethernet resend can recover from."""


class _RdmaBusy(_RdmaError):
    """Registered memory is in use by other transfers; resend this file only."""


_RDMA_BUSY = "RDMA memory budget exhausted"


def _pread_bytes(fd: int, length: int, offset: int) -> bytes:
    """Read exactly *length* bytes (Monarch messages carry ``bytes``)."""
    blocks = []
    remaining = length
    while remaining > 0:
        block = os.pread(fd, remaining, offset + length - remaining)
        if not block:
            raise EOFError(f"file ended at byte {offset + length - remaining}")
        blocks.append(block)
        remaining -= len(block)
    return blocks[0] if len(blocks) == 1 else b"".join(blocks)


def _staged_archive_path(path: str) -> str:
    """Return a unique hidden sibling of *path* to receive its archive."""
    parent, name = os.path.split(os.path.normpath(path))
    return os.path.join(parent, f".{name}.{uuid.uuid4().hex}.tar")


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class _FileSourceActor(Actor):
    """Actor that reads chunks from a local file for transfer.

    Spawned on the ProcMesh that has access to the source file. A source actor
    streams to one peer sink, so its RDMA circuit breaker is per pair.
    """

    def __init__(self) -> None:
        # None until the peer's RDMA capability is known; False once RDMA is
        # unavailable or has failed under auto.
        self._rdma_ok: bool | None = None

    @endpoint
    def get_metadata(self, path: str) -> FileMetadata:
        """Return metadata for the file at *path*."""
        st = os.stat(path)
        return FileMetadata(
            size=st.st_size,
            mode=stat.S_IMODE(st.st_mode),
            filename=os.path.basename(path),
            is_dir=stat.S_ISDIR(st.st_mode),
        )

    @endpoint
    def transfer_file_bytes_to(
        self,
        path: str,
        sink_actor: Any,
        sink_path: str,
        file_size: int,
        chunk_size: int,
        pipeline_depth: int,
        sink_mode: int | None,
        transport: Transport = "ethernet",
        transfer_id: str | None = None,
    ) -> FileTransferDigests:
        """Stream bytes directly to *sink_actor* with bounded backpressure.

        *chunk_size* applies to Ethernet; RDMA chunks are one registered slot
        (:data:`agentbox.remote.rdma.CHUNK_SIZE`). *transfer_id* names the
        sink-side transfer (default: a fresh one), so the caller can abort
        exactly this transfer if the source fails.
        """
        if file_size < 0:
            raise ValueError("file_size must be non-negative")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if pipeline_depth < 1:
            raise ValueError("pipeline_depth must be positive")

        transfer_id = transfer_id or uuid.uuid4().hex
        args = (
            path,
            sink_actor,
            sink_path,
            file_size,
            pipeline_depth,
            sink_mode,
            transfer_id,
        )
        if self._use_rdma(transport, sink_actor, file_size):
            try:
                return self._stream(*args, _rdma.CHUNK_SIZE, rdma=True)
            except _RdmaBusy as exc:
                if transport == "rdma":
                    raise
                # Memory pressure, not a broken path: keep RDMA for later files.
                logger.info("Sending %s over Ethernet: %s", path, exc)
            except _RdmaError as exc:
                if transport == "rdma":
                    raise
                self._rdma_ok = False
                logger.warning(
                    "RDMA transfer of %s failed; resending over Ethernet and "
                    "disabling RDMA for this peer: %s",
                    path,
                    exc,
                )
        return self._stream(*args, chunk_size, rdma=False)

    def _use_rdma(self, transport: Transport, sink_actor: Any, file_size: int) -> bool:
        if transport == "ethernet" or file_size == 0:
            return False
        if transport == "rdma":
            return True
        if file_size < RDMA_MIN_FILE_SIZE or self._rdma_ok is False:
            return False
        if self._rdma_ok is None:
            try:
                self._rdma_ok = (
                    _rdma.backend() == "ibverbs"
                    and sink_actor.rdma_backend.choose().get() == "ibverbs"
                )
            except Exception:
                logger.debug("RDMA capability probe failed", exc_info=True)
                self._rdma_ok = False
        return self._rdma_ok

    def _stream(
        self,
        path: str,
        sink_actor: Any,
        sink_path: str,
        file_size: int,
        pipeline_depth: int,
        sink_mode: int | None,
        transfer_id: str,
        chunk_size: int,
        *,
        rdma: bool,
    ) -> FileTransferDigests:
        if chunk_size % _io.PIECE_SIZE:
            raise ValueError(f"chunk_size must be a multiple of {_io.PIECE_SIZE}")
        slots = _rdma.pool("source") if rdma else None
        digests = _io.PieceDigests(file_size)
        # Per in-flight message: the sink's acknowledgement, the background
        # hashing of its chunks, and for RDMA the registered slots both read.
        pending: deque[tuple[Any, list[Future[None]], list[_rdma.Slot]]] = deque()
        max_pending = (
            min(pipeline_depth, _rdma.BATCHES_IN_FLIGHT) if rdma else pipeline_depth
        )

        def release(hashing: list[Future[None]], held: list[_rdma.Slot]) -> None:
            _wait_futures(hashing)
            if slots is not None:
                for slot in held:
                    slots.release(slot)

        def settle(
            ack: Any, hashing: list[Future[None]], held: list[_rdma.Slot]
        ) -> None:
            try:
                error = ack.get()
            finally:
                release(hashing, held)
            _io.wait_all(hashing)
            if error is not None:
                raise (_RdmaBusy if error.startswith(_RDMA_BUSY) else _RdmaError)(error)

        try:
            sink_actor.begin_byte_transfer.choose(
                transfer_id,
                sink_path,
                file_size,
                pipeline_depth,
            ).get()
            fd = os.open(path, os.O_RDONLY)
            try:
                # RDMA chunks acquired for the next batch but not yet sent.
                batch: list[tuple[int, int, _rdma.Slot]] = []
                offset = 0
                while offset < file_size:
                    length = min(chunk_size, file_size - offset)
                    offset += length
                    if slots is None:
                        at = offset - length
                        data = _pread_bytes(fd, length, at)
                        ack = sink_actor.write_byte_transfer_chunk.choose(
                            transfer_id, at, data
                        )
                        # The sink only reads the chunk, so hash it while it moves.
                        hashing = _io.submit_pieces(
                            memoryview(data), at, digests.hash_piece
                        )
                        pending.append((ack, hashing, []))
                    else:
                        batch.append(
                            (offset - length, length, self._acquire_slot(slots))
                        )
                        if len(batch) < _rdma.BATCH_CHUNKS and offset < file_size:
                            continue
                        pending.append(
                            self._send_batch(
                                fd, sink_actor, transfer_id, batch, digests
                            )
                        )
                        batch = []
                    if len(pending) >= max_pending:
                        settle(*pending.popleft())
            except EOFError as exc:
                raise FileCopyError(
                    f"Source file shrank during transfer ({exc}): {path}"
                ) from None
            finally:
                os.close(fd)
                for *_, slot in batch:  # acquired but never sent
                    release([], [slot])

            while pending:
                settle(*pending.popleft())
            source_digest = digests.hexdigest()
            sink_digest = cast(
                str,
                sink_actor.finish_byte_transfer.choose(transfer_id).get(),
            )
            if source_digest == sink_digest and sink_mode is not None:
                sink_actor.set_file_mode.choose(sink_path, sink_mode).get()
            return FileTransferDigests(
                source_digest, sink_digest, "rdma" if rdma else "ethernet"
            )
        except BaseException:
            cleanup_deadline = _monotonic() + _TRANSFER_CLEANUP_TIMEOUT
            while pending:
                ack, hashing, held = pending.popleft()
                try:
                    ack.get(timeout=max(0.0, cleanup_deadline - _monotonic()))
                except Exception:
                    pass
                finally:
                    release(hashing, held)
            try:
                sink_actor.abort_byte_transfer.choose(transfer_id).get(
                    timeout=max(0.0, cleanup_deadline - _monotonic())
                )
            except Exception:
                pass
            raise

    @staticmethod
    def _send_batch(
        fd: int,
        sink_actor: Any,
        transfer_id: str,
        batch: list[tuple[int, int, _rdma.Slot]],
        digests: _io.PieceDigests,
    ) -> tuple[Any, list[Future[None]], list[_rdma.Slot]]:
        """Fill *batch*'s slots from *fd* and ask the sink to pull them.

        Returns the pending entry for the batch; on failure the caller still
        owns the slots.
        """
        # Fill the whole batch at once: reads, unlike writes, scale with threads.
        reads: list[Future[None]] = []
        for at, length, slot in batch:
            reads += _io.submit_pieces(
                slot.view[:length],
                at,
                lambda piece, pos: _io.pread_exact(fd, piece, pos),
                "io",
            )
        _io.wait_all(reads)
        ack = sink_actor.pull_byte_transfer_chunks.choose(
            transfer_id, [(at, length, slot.handle) for at, length, slot in batch]
        )
        # The sink only reads the slots, so hash while it pulls.
        hashing: list[Future[None]] = []
        for at, length, slot in batch:
            hashing += _io.submit_pieces(slot.view[:length], at, digests.hash_piece)
        return ack, hashing, [slot for *_, slot in batch]

    @staticmethod
    def _acquire_slot(slots: _rdma.SlotPool) -> _rdma.Slot:
        try:
            return slots.acquire()
        except _rdma.SlotBudgetExhausted as exc:
            raise _RdmaBusy(f"{_RDMA_BUSY} (source): {exc}") from exc
        except Exception as exc:
            raise _RdmaError(f"Cannot register RDMA source memory: {exc!r}") from exc

    @endpoint
    def ping(self) -> None:
        """No-op endpoint used to verify the actor is ready to handle requests."""


class _FileSinkActor(Actor):
    """Actor that writes incoming chunks to a local file.

    Spawned on the ProcMesh that will store the destination file.
    """

    def __init__(self) -> None:
        self._sink_byte_transfers: dict[str, _SinkByteTransferState] = {}

    @staticmethod
    def _prepare_path(path: str, size: int, mode: int | None) -> None:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Size the file up front so that chunk writes never extend it (which
        # lets them proceed in parallel) and no page is left dirty.
        with open(path, "wb") as output:
            output.truncate(size)

        if mode is not None:
            os.chmod(path, mode)

    @endpoint
    def begin_byte_transfer(
        self,
        transfer_id: str,
        path: str,
        expected_size: int,
        pipeline_depth: int,
    ) -> None:
        """Open and initialize one ordered direct transfer."""
        if transfer_id in self._sink_byte_transfers:
            raise FileCopyError(f"Duplicate transfer ID: {transfer_id}")
        if expected_size < 0:
            raise ValueError("expected_size must be non-negative")
        if pipeline_depth < 1:
            raise ValueError("pipeline_depth must be positive")
        # Keep the destination writable until its bytes have been verified.
        # Applying a read-only source mode before this open makes empty-file
        # transfers fail for non-root users.
        self._prepare_path(path, expected_size, None)
        fd = os.open(path, os.O_WRONLY)
        direct_fd = None
        if expected_size >= _rdma.CHUNK_SIZE and _io.direct_io_filesystem(path):
            try:
                direct_fd = os.open(path, os.O_WRONLY | getattr(os, "O_DIRECT", 0))
            except OSError:
                pass
        self._sink_byte_transfers[transfer_id] = _SinkByteTransferState(
            fd=fd,
            path=path,
            expected_size=expected_size,
            max_pending=pipeline_depth,
            digests=_io.PieceDigests(expected_size),
            direct_fd=direct_fd,
        )

    def _next_chunk(self, transfer_id: str, offset: int) -> _SinkByteTransferState:
        state = self._sink_byte_transfers.get(transfer_id)
        if state is None:
            raise FileCopyError(f"Unknown transfer ID: {transfer_id}")
        if offset != state.next_offset:
            raise FileCopyError(
                f"Out-of-order chunk: expected offset {state.next_offset}, got {offset}"
            )
        return state

    def _accept_chunk(
        self,
        state: _SinkByteTransferState,
        offset: int,
        view: memoryview,
        slot: _rdma.Slot | None,
    ) -> None:
        """Write and hash *view* in the background; takes ownership of *slot*.

        Returns once the chunk is accepted, so the next chunk's transfer
        overlaps this one's writes. At most ``max_pending`` chunks stay
        unwritten; their errors surface by the time the transfer finishes.
        """
        try:
            if offset + len(view) > state.expected_size:
                raise FileCopyError(
                    f"Chunk at {offset} overruns the expected "
                    f"{state.expected_size} bytes"
                )

            jobs = self._submit_writes(state, offset, view, direct=slot is not None)
            jobs += _io.submit_pieces(view, offset, state.digests.hash_piece)
        except BaseException:
            if slot is not None:
                _rdma.pool("sink").release(slot)
            raise
        state.pending.append((jobs, slot))
        state.next_offset += len(view)
        while len(state.pending) > state.max_pending:
            self._settle_writes(*state.pending.popleft())

    @staticmethod
    def _submit_writes(
        state: _SinkByteTransferState, offset: int, view: memoryview, *, direct: bool
    ) -> list[Future[None]]:
        """Write a chunk with one job: more writers to one file only contend.

        Registered slots are page-aligned, so their block-aligned part may
        bypass the page cache; any tail, and anything a filesystem refuses to
        write directly, goes through the buffered descriptor.
        """
        io = _io.executor("io")
        direct_fd = state.direct_fd if direct else None
        aligned = len(view) - len(view) % _io.DIRECT_IO_ALIGN
        if direct_fd is None or not aligned:
            return [io.submit(_io.pwrite_all, state.fd, view, offset)]

        def write_direct(part: memoryview) -> None:
            try:
                _io.pwrite_all(direct_fd, part, offset)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                _io.pwrite_all(state.fd, part, offset)

        jobs = [io.submit(write_direct, view[:aligned])]
        if aligned < len(view):
            jobs.append(
                io.submit(_io.pwrite_all, state.fd, view[aligned:], offset + aligned)
            )
        return jobs

    @staticmethod
    def _settle_writes(jobs: list[Future[None]], slot: _rdma.Slot | None) -> None:
        try:
            _io.wait_all(jobs)
        finally:
            if slot is not None:
                _rdma.pool("sink").release(slot)

    def _drain(self, state: _SinkByteTransferState) -> None:
        """Wait for every background write, then raise the first failure."""
        error: BaseException | None = None
        while state.pending:
            try:
                self._settle_writes(*state.pending.popleft())
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    @endpoint
    def write_byte_transfer_chunk(
        self,
        transfer_id: str,
        offset: int,
        data: bytes,
    ) -> None:
        """Accept one ordered Ethernet chunk for writing."""
        state = self._next_chunk(transfer_id, offset)
        self._accept_chunk(state, offset, memoryview(data), None)

    @endpoint
    def pull_byte_transfer_chunks(
        self,
        transfer_id: str,
        chunks: list[tuple[int, int, Any]],
    ) -> str | None:
        """RDMA-read consecutive ``(offset, length, handle)`` chunks, then write them.

        The reads go out in one parallel submission; the writes continue in
        the background (see :meth:`_accept_chunk`). Returns ``None``, or a
        description of an RDMA failure. RDMA failures are returned rather than
        raised so that the source can resend over Ethernet; file errors raise
        as for :meth:`write_byte_transfer_chunk`.
        """
        state = self._sink_byte_transfers.get(transfer_id)
        if state is not None and state.rdma_error is not None:
            # Batches pipelined behind a failed one must not each wait out
            # their own RDMA timeout.
            return state.rdma_error
        state = self._next_chunk(transfer_id, chunks[0][0] if chunks else -1)
        expected = state.next_offset
        for offset, length, _ in chunks:
            if offset != expected or length < 1:
                raise FileCopyError(f"Non-consecutive chunk at offset {offset}")
            expected += length
        state.rdma_error = self._pull_chunks(state, chunks)
        return state.rdma_error

    def _pull_chunks(
        self, state: _SinkByteTransferState, chunks: list[tuple[int, int, Any]]
    ) -> str | None:
        slots = _rdma.pool("sink")
        held: list[_rdma.Slot] = []
        try:
            try:
                for _ in chunks:
                    held.append(slots.acquire())
            except _rdma.SlotBudgetExhausted as exc:
                return f"{_RDMA_BUSY} (sink): {exc}"
            except Exception as exc:
                return f"Cannot register RDMA sink memory: {exc!r}"
            for (_, length, handle), slot in zip(chunks, held, strict=True):
                # A read fills the whole destination, so slots must match in size.
                if not length <= handle.size() <= len(slot.view):
                    return (
                        f"RDMA chunk of {length} bytes in a {handle.size()}-byte "
                        f"handle does not fit a {len(slot.view)}-byte slot"
                    )
            try:
                _rdma.read_all(
                    [
                        (handle, slot.view)
                        for (_, _, handle), slot in zip(chunks, held, strict=True)
                    ],
                    timeout=_RDMA_TIMEOUT,
                )
            except Exception as exc:
                return f"RDMA read failed: {exc}"
            accepted, held = held, []
        finally:
            for slot in held:
                slots.release(slot)
        for index, ((offset, length, _), slot) in enumerate(
            zip(chunks, accepted, strict=True)
        ):
            try:
                self._accept_chunk(state, offset, slot.view[:length], slot)
            except BaseException:
                for unclaimed in accepted[index + 1 :]:
                    slots.release(unclaimed)
                raise
        return None

    @endpoint
    def rdma_backend(self) -> str:
        """Return this process's Monarch RDMA backend."""
        return _rdma.backend()

    @endpoint
    def finish_byte_transfer(self, transfer_id: str) -> str:
        """Validate and close a direct transfer, returning its sink digest.

        The digest covers the bytes this transfer wrote (see
        :class:`~agentbox.remote.chunk_io.PieceDigests`). The size and
        identity checks reject a destination that was truncated, extended, or
        replaced in the meantime.
        """
        state = self._sink_byte_transfers.pop(transfer_id, None)
        if state is None:
            raise FileCopyError(f"Unknown transfer ID: {transfer_id}")
        try:
            self._drain(state)
            if state.next_offset != state.expected_size:
                raise FileCopyError(
                    f"Incomplete transfer: received {state.next_offset} "
                    f"of {state.expected_size} bytes"
                )
            written = os.fstat(state.fd)
            if written.st_size != state.expected_size:
                raise FileCopyError(
                    f"Size mismatch after transfer: expected {state.expected_size}, "
                    f"got {written.st_size}"
                )
            try:
                current = os.stat(state.path)
            except FileNotFoundError:
                current = None
            if current is None or (current.st_dev, current.st_ino) != (
                written.st_dev,
                written.st_ino,
            ):
                raise FileCopyError(
                    f"Destination path was replaced during transfer: {state.path}"
                )
        finally:
            self._close(state)
        return state.digests.hexdigest()

    @staticmethod
    def _close(state: _SinkByteTransferState) -> None:
        os.close(state.fd)
        if state.direct_fd is not None:
            os.close(state.direct_fd)

    @endpoint
    def set_file_mode(self, path: str, mode: int) -> None:
        """Apply the source mode after destination verification succeeds."""
        os.chmod(path, mode)

    def _abort_byte_transfer(self, transfer_id: str) -> None:
        state = self._sink_byte_transfers.pop(transfer_id, None)
        if state is not None:
            # Background writes use the descriptor; never close it under them.
            try:
                self._drain(state)
            except BaseException:
                pass
            finally:
                self._close(state)

    @endpoint
    def abort_byte_transfer(self, transfer_id: str) -> None:
        """Close and discard sink state after a failed direct transfer."""
        self._abort_byte_transfer(transfer_id)

    @endpoint
    def ping(self) -> None:
        """No-op endpoint used to verify the actor is ready to handle requests."""


_ENDPOINT_STOP_TIMEOUT = 5.0


def _pack_to_temp(path: str) -> FileMetadata:
    fd, archive = tempfile.mkstemp(prefix="agentbox-", suffix=".tar")
    os.close(fd)
    try:
        tree_archive.pack(path, archive)
    except BaseException:
        os.unlink(archive)
        raise
    return FileMetadata(os.path.getsize(archive), 0o600, archive)


def _unpack_and_remove(archive: str, path: str) -> None:
    try:
        tree_archive.unpack(archive, path)
    finally:
        _remove_quietly(archive)


class _ArchiveActor(Actor):
    """Packs and unpacks directory archives on its ProcMesh.

    Kept apart from the transfer actors, whose sync endpoints run one at a
    time, and async with the work in threads, so a long pack or unpack blocks
    neither file transfers nor other directory copies.
    """

    @endpoint
    async def pack_directory(self, path: str) -> FileMetadata:
        """Archive directory *path* into a new temporary tar file.

        Returns the archive's metadata, with its path as ``filename``; the
        caller removes it with :meth:`discard`.
        """
        return await asyncio.to_thread(_pack_to_temp, path)

    @endpoint
    async def unpack_archive(self, archive: str, path: str) -> None:
        """Extract the received tar file *archive* into *path*, then remove it."""
        await asyncio.to_thread(_unpack_and_remove, archive, path)

    @endpoint
    async def discard(self, path: str) -> None:
        """Remove *path* if it exists."""
        _remove_quietly(path)


def _spawn_archiver(proc_mesh: ProcMesh) -> Any:
    from agentbox.actors.mesh_schema import make_name

    return proc_mesh.spawn(
        make_name("filearchive", uuid.uuid4().hex[:8]), _ArchiveActor
    )


@dataclass
class TransferEndpoints:
    """A source actor and a sink actor on one ProcMesh, and an archive actor
    spawned on the first directory copy.

    A source endpoint blocks while it streams to a sink, and sinks never call
    out, so keeping the roles in separate actors means concurrent uploads and
    downloads between two meshes can never wait on each other.
    """

    proc_mesh: ProcMesh
    source: Any
    sink: Any
    _archive: Any = field(default=None, init=False, repr=False)
    _archive_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @classmethod
    def spawn(cls, proc_mesh: ProcMesh) -> TransferEndpoints:
        from agentbox.actors.mesh_schema import make_name

        suffix = uuid.uuid4().hex[:8]
        return cls(
            proc_mesh,
            proc_mesh.spawn(make_name("filesource", suffix), _FileSourceActor),
            proc_mesh.spawn(make_name("filesink", suffix), _FileSinkActor),
        )

    def archiver(self) -> Any:
        """Return this mesh's :class:`_ArchiveActor`, spawning it on first use."""
        with self._archive_lock:
            if self._archive is None:
                self._archive = _spawn_archiver(self.proc_mesh)
            return self._archive

    def _actors(self) -> list[Any]:
        actors = [self.source, self.sink]
        with self._archive_lock:
            if self._archive is not None:
                actors.append(self._archive)
        return actors

    def ping(self) -> None:
        for future in [self.source.ping.choose(), self.sink.ping.choose()]:
            future.get()

    async def aping(self) -> None:
        await asyncio.gather(self.source.ping.choose(), self.sink.ping.choose())

    def stop(self) -> None:
        """Best-effort, bounded stop of every actor."""
        for actor in self._actors():
            try:
                actor.stop().get(timeout=_ENDPOINT_STOP_TIMEOUT)
            except Exception:
                logger.debug("Error stopping file-transfer actor", exc_info=True)

    async def astop(self) -> None:
        """Async-native counterpart of :meth:`stop`."""
        for actor in self._actors():
            try:
                await asyncio.wait_for(actor.stop(), timeout=_ENDPOINT_STOP_TIMEOUT)
            except Exception:
                logger.debug("Error stopping file-transfer actor", exc_info=True)


# Client-side transfer actors fill one proc with up to MACHINES_PER_TRANSFER_PROC
# Machines before the next proc is used (wrapping after LOCAL_TRANSFER_PROCS).
# One proc hosting every Machine's source and sink actors saturates (one GIL,
# one actor runtime) at several hundred concurrent large transfers, until it
# misses Monarch's supervision status checks and takes the client down.
MACHINES_PER_TRANSFER_PROC: int = 128
LOCAL_TRANSFER_PROCS: int = 16
_local_meshes: dict[int, ProcMesh] = {}
_local_mesh_lock = threading.Lock()


class _ImportWarmup(Actor):
    """Imports agentbox in a fresh proc before concurrent actors arrive."""

    @endpoint
    def ready(self) -> None:
        """Return once this proc has imported the transfer modules."""
        import agentbox.remote.chunk_io  # noqa: F401
        import agentbox.remote.file_transfer  # noqa: F401
        import agentbox.remote.rdma  # noqa: F401


def local_transfer_mesh(shard: int = 0) -> ProcMesh:
    """Return the client-side ProcMesh for the *shard*-th Machine.

    Machines ``0..127`` share proc 0, ``128..255`` proc 1, and so on, wrapping
    after :data:`LOCAL_TRANSFER_PROCS` procs; each proc is created on first use.
    """
    shard = (shard // MACHINES_PER_TRANSFER_PROC) % LOCAL_TRANSFER_PROCS
    with _local_mesh_lock:
        if shard not in _local_meshes:
            from agentbox.actors.mesh_schema import make_name
            from monarch.actor import this_host

            mesh = this_host().spawn_procs(name=make_name("filetransfer", str(shard)))
            # The proc starts without agentbox imported. Many Machines spawning
            # their transfer actors at once would unpickle them on several
            # threads, each importing agentbox, which can trip Python's import
            # deadlock detection and abort the client. Import it once first.
            try:
                warmup = mesh.spawn(make_name("filewarmup"), _ImportWarmup)
                warmup.ready.choose().get()
            except BaseException:
                try:
                    mesh.stop().get(timeout=_ENDPOINT_STOP_TIMEOUT)
                except Exception:
                    logger.debug("Error stopping transfer proc", exc_info=True)
                raise
            try:
                warmup.stop().get(timeout=_ENDPOINT_STOP_TIMEOUT)
            except Exception:
                logger.debug("Error stopping transfer warm-up actor", exc_info=True)
            _local_meshes[shard] = mesh
        return _local_meshes[shard]


def _spawn_transfer_actors(
    source_proc_mesh: ProcMesh,
    sink_proc_mesh: ProcMesh,
    source_actor: Any,
    sink_actor: Any,
) -> tuple[Any, Any]:
    """Return the caller's actors, spawning one-shot actors where absent."""
    from agentbox.actors.mesh_schema import make_name

    if source_actor is None:
        source_actor = source_proc_mesh.spawn(
            make_name("filesource", uuid.uuid4().hex[:8]), _FileSourceActor
        )
    if sink_actor is None:
        sink_actor = sink_proc_mesh.spawn(
            make_name("filesink", uuid.uuid4().hex[:8]), _FileSinkActor
        )
    return source_actor, sink_actor


def _check_digests(
    digests: FileTransferDigests, attempt: int, source_path: str, sink_path: str
) -> FileCopyError | None:
    """Return a retryable error for a digest mismatch, or ``None`` on success."""
    if digests.source_sha256 == digests.sink_sha256:
        logger.info(
            "Transfer complete and verified over %s (SHA-256 OK): %s -> %s",
            digests.transport,
            source_path,
            sink_path,
        )
        return None
    error = FileCopyError(
        f"SHA-256 mismatch after transfer (attempt {attempt}/{_COPY_MAX_RETRIES}): "
        f"src={digests.source_sha256[:16]}… dst={digests.sink_sha256[:16]}…"
    )
    logger.warning("%s", error)
    return error


def _log_retry(source_path: str, sink_path: str, attempt: int) -> None:
    if attempt > 1:
        logger.warning(
            "Retrying transfer %s -> %s (attempt %d/%d)",
            source_path,
            sink_path,
            attempt,
            _COPY_MAX_RETRIES,
        )


def _copy_file(
    source_proc_mesh: ProcMesh,
    sink_proc_mesh: ProcMesh,
    source_path: str,
    sink_path: str,
    *,
    source_actor: Any = None,
    sink_actor: Any = None,
    source_archiver: Callable[[], Any] | None = None,
    sink_archiver: Callable[[], Any] | None = None,
) -> None:
    """Copy a file or directory from *source_proc_mesh* to *sink_proc_mesh*.

    This is the orchestration behind
    ``Machine.upload`` and ``Machine.download``.

    Args:
        source_proc_mesh: ProcMesh with access to the source path.
        sink_proc_mesh: ProcMesh with access to the destination location.
        source_path: Path to the source file or directory (on source_proc_mesh).
        sink_path: Path to the destination (on sink_proc_mesh). A directory's
            contents are unpacked into it (see :func:`.tree_archive.unpack`).
        source_actor: Optional pre-created :class:`_FileSourceActor` mesh to
            reuse.  When provided, a new actor is not spawned and the caller
            is responsible for the actor's lifetime.
        sink_actor: Optional pre-created :class:`_FileSinkActor` mesh to
            reuse.  Same semantics as *source_actor*.
        source_archiver: Optional callable returning the source mesh's
            :class:`_ArchiveActor` (e.g. :meth:`TransferEndpoints.archiver`),
            called only for a directory. Without it a one-shot actor is
            spawned. *sink_archiver* is the same for the sink mesh.

    .. todo:: Stream directory archives.
        A directory is packed into a temporary file in the source's TMPDIR,
        sent to a staged copy beside *sink_path*, then unpacked, so every byte
        is written three times and the source needs the whole tree's size in
        TMPDIR (a small tmpfs fails with ENOSPC even when the destination has
        room).

        Fix sketch: have the source actor generate the tar while it streams
        instead of reading a file. :meth:`_FileSourceActor._stream` and
        ``_send_batch`` read the source by offset (``pread``, and RDMA reads
        from the descriptor) and the sink needs the size up front, so this
        needs a sequential reader in their place and an exact size from a
        first walk (failing, then retrying, if the tree changes before the
        stream ends); a digest mismatch would re-pack. Keep the sink writing
        through :func:`.tree_archive.unpack` rather than per-file paths, which
        would follow symlinks in the destination. Test the RDMA path on
        hardware that has it.
    """
    source_actor, sink_actor = _spawn_transfer_actors(
        source_proc_mesh, sink_proc_mesh, source_actor, sink_actor
    )
    metadata: FileMetadata = source_actor.get_metadata.choose(source_path).get()
    if not metadata.is_dir:
        _copy_verified(source_actor, sink_actor, source_path, sink_path, metadata)
        return
    packer, unpacker = _archivers(
        source_proc_mesh, sink_proc_mesh, source_archiver, sink_archiver
    )
    archive: FileMetadata = packer.pack_directory.choose(source_path).get()
    staged = _staged_archive_path(sink_path)
    try:
        _copy_verified(source_actor, sink_actor, archive.filename, staged, archive)
        unpacker.unpack_archive.choose(staged, sink_path).get()
    except BaseException:
        _discard_quietly(unpacker, staged)
        raise
    finally:
        _discard_quietly(packer, archive.filename)


async def _acopy_file(
    source_proc_mesh: ProcMesh,
    sink_proc_mesh: ProcMesh,
    source_path: str,
    sink_path: str,
    *,
    source_actor: Any = None,
    sink_actor: Any = None,
    source_archiver: Callable[[], Any] | None = None,
    sink_archiver: Callable[[], Any] | None = None,
) -> None:
    """Async-native counterpart of :func:`_copy_file`.

    Behaves identically but awaits every Monarch future, so it can run inside a
    running asyncio event loop.  Used by :meth:`Machine._acopy_to_impl` /
    :meth:`Machine._acopy_from_impl`.
    """
    source_actor, sink_actor = _spawn_transfer_actors(
        source_proc_mesh, sink_proc_mesh, source_actor, sink_actor
    )
    metadata: FileMetadata = await source_actor.get_metadata.choose(source_path)
    if not metadata.is_dir:
        await _acopy_verified(
            source_actor, sink_actor, source_path, sink_path, metadata
        )
        return
    packer, unpacker = _archivers(
        source_proc_mesh, sink_proc_mesh, source_archiver, sink_archiver
    )
    archive: FileMetadata = await packer.pack_directory.choose(source_path)
    staged = _staged_archive_path(sink_path)
    try:
        await _acopy_verified(
            source_actor, sink_actor, archive.filename, staged, archive
        )
        await unpacker.unpack_archive.choose(staged, sink_path)
    except BaseException:
        await _adiscard_quietly(unpacker, staged)
        raise
    finally:
        await _adiscard_quietly(packer, archive.filename)


def _archivers(
    source_proc_mesh: ProcMesh,
    sink_proc_mesh: ProcMesh,
    source_archiver: Callable[[], Any] | None,
    sink_archiver: Callable[[], Any] | None,
) -> tuple[Any, Any]:
    """Return the (source, sink) archive actors for one directory copy."""
    return (
        source_archiver() if source_archiver else _spawn_archiver(source_proc_mesh),
        sink_archiver() if sink_archiver else _spawn_archiver(sink_proc_mesh),
    )


def _discard_quietly(actor: Any, path: str) -> None:
    """Best-effort, bounded removal of a temporary archive."""
    try:
        actor.discard.choose(path).get(timeout=_TRANSFER_CLEANUP_TIMEOUT)
    except Exception:
        logger.debug("Error removing transfer archive %s", path, exc_info=True)


async def _adiscard_quietly(actor: Any, path: str) -> None:
    """Async-native counterpart of :func:`_discard_quietly`."""
    try:
        await asyncio.wait_for(
            asyncio.shield(actor.discard.choose(path)),
            timeout=_TRANSFER_CLEANUP_TIMEOUT,
        )
    except Exception:
        logger.debug("Error removing transfer archive %s", path, exc_info=True)


def _copy_verified(
    source_actor: Any,
    sink_actor: Any,
    source_path: str,
    sink_path: str,
    metadata: FileMetadata,
) -> None:
    """Stream one file and check its digests, retrying on a mismatch."""
    transfer_ids: list[str] = []
    succeeded = False
    try:
        logger.info(
            "Copying %s (%d bytes) -> %s", source_path, metadata.size, sink_path
        )
        last_error: Exception | None = None
        for attempt in range(1, _COPY_MAX_RETRIES + 1):
            _log_retry(source_path, sink_path, attempt)
            transfer_ids.append(uuid.uuid4().hex)
            digests = _transfer(
                source_actor,
                sink_actor,
                source_path,
                sink_path,
                metadata.size,
                sink_mode=metadata.mode,
                transfer_id=transfer_ids[-1],
            )
            last_error = _check_digests(digests, attempt, source_path, sink_path)
            if last_error is None:
                break
        if last_error is not None:
            raise last_error
        succeeded = True
    finally:
        # Safety net: close this copy's sink transfers in case the source
        # actor failed before sending an abort. Only these: the sink is shared
        # by every concurrent copy to its mesh. A copy that verified its digest
        # had every attempt finished by the sink, so it skips the round trip.
        for transfer_id in [] if succeeded else transfer_ids:
            try:
                sink_actor.abort_byte_transfer.choose(transfer_id).get(
                    timeout=_TRANSFER_CLEANUP_TIMEOUT
                )
            except Exception:
                pass


async def _acopy_verified(
    source_actor: Any,
    sink_actor: Any,
    source_path: str,
    sink_path: str,
    metadata: FileMetadata,
) -> None:
    """Async-native counterpart of :func:`_copy_verified`."""
    transfer_ids: list[str] = []
    succeeded = False
    try:
        logger.info(
            "Copying %s (%d bytes) -> %s", source_path, metadata.size, sink_path
        )
        last_error: Exception | None = None
        for attempt in range(1, _COPY_MAX_RETRIES + 1):
            _log_retry(source_path, sink_path, attempt)
            transfer_ids.append(uuid.uuid4().hex)
            digests = await _atransfer(
                source_actor,
                sink_actor,
                source_path,
                sink_path,
                metadata.size,
                sink_mode=metadata.mode,
                transfer_id=transfer_ids[-1],
            )
            last_error = _check_digests(digests, attempt, source_path, sink_path)
            if last_error is None:
                break
        if last_error is not None:
            raise last_error
        succeeded = True
    finally:
        for transfer_id in [] if succeeded else transfer_ids:  # As in _copy_verified.
            try:
                await asyncio.wait_for(
                    asyncio.shield(sink_actor.abort_byte_transfer.choose(transfer_id)),
                    timeout=_TRANSFER_CLEANUP_TIMEOUT,
                )
            except Exception:
                pass


def _transfer(
    source_actor: Any,
    sink_actor: Any,
    source_path: str,
    sink_path: str,
    file_size: int,
    *,
    sink_mode: int | None = None,
    transfer_id: str | None = None,
) -> FileTransferDigests:
    """Stream bytes directly from the source actor to the sink actor."""
    return cast(
        FileTransferDigests,
        source_actor.transfer_file_bytes_to.choose(
            source_path,
            sink_actor,
            sink_path,
            file_size,
            PICKLE_CHUNK_SIZE,
            _pipeline_depth(),
            sink_mode,
            _file_transport(),
            transfer_id,
        ).get(),
    )


async def _atransfer(
    source_actor: Any,
    sink_actor: Any,
    source_path: str,
    sink_path: str,
    file_size: int,
    *,
    sink_mode: int | None = None,
    transfer_id: str | None = None,
) -> FileTransferDigests:
    """Async-native counterpart of :func:`_transfer`."""
    return cast(
        FileTransferDigests,
        await source_actor.transfer_file_bytes_to.choose(
            source_path,
            sink_actor,
            sink_path,
            file_size,
            PICKLE_CHUNK_SIZE,
            _pipeline_depth(),
            sink_mode,
            _file_transport(),
            transfer_id,
        ),
    )
