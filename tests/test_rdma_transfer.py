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

"""Tests for the RDMA file-transfer transport and its registered-slot pool."""

from __future__ import annotations

import io
import mmap
import os
import tempfile
from unittest.mock import MagicMock

import pytest
from _endpoints import call_endpoint
from agentbox.remote import chunk_io, file_transfer as ft, rdma
from agentbox.remote.file_transfer import (
    _file_transport,
    _FileSinkActor,
    _FileSourceActor,
    _RdmaError,
    FileCopyError,
    FileTransferDigests,
    RDMA_MIN_FILE_SIZE,
)


class _FakeHandle:
    """Stands in for an ``RDMABuffer`` over a source slot."""

    def __init__(self, view: memoryview) -> None:
        self.view = view
        self.fail: str | None = None
        self.dropped = False

    def size(self) -> int:
        return len(self.view)

    def drop(self) -> MagicMock:
        self.dropped = True
        return MagicMock()

    def read_into(self, dst: memoryview, *, timeout: int) -> MagicMock:
        future = MagicMock()
        if self.fail is not None:
            future.get.side_effect = RuntimeError(self.fail)
        else:
            dst[: len(self.view)] = self.view
        return future


class _FakePool(rdma.SlotPool):
    """A slot pool that registers nothing."""

    def __init__(self, capacity: int = 4, size: int = 8) -> None:
        super().__init__(capacity, size)
        self.registered = 0
        self.fail: str | None = None

    def _register(self, pooled: bool) -> rdma.Slot:
        self.registered += 1
        buffer = mmap.mmap(-1, self._size)
        view = memoryview(buffer)
        handle = _FakeHandle(view)
        handle.fail = self.fail
        return rdma.Slot(buffer, view, handle, pooled)

    @property
    def free(self) -> int:
        return len(self._free)


def _fake_read_all(reads, timeout):
    for handle, view in reads:
        handle.read_into(view, timeout=timeout).get()


@pytest.fixture
def pools(monkeypatch):
    """Give each role a fake 8-byte-slot pool and make RDMA chunks 8 bytes."""
    created = {"source": _FakePool(capacity=16), "sink": _FakePool(capacity=16)}
    monkeypatch.setattr(rdma, "pool", created.__getitem__)
    monkeypatch.setattr(rdma, "CHUNK_SIZE", 8)
    monkeypatch.setattr(chunk_io, "PIECE_SIZE", 2)
    monkeypatch.setattr(rdma, "read_all", _fake_read_all)
    return created


def _sink_into(tmp_path, sink: _FileSinkActor | None = None) -> MagicMock:
    """Return a mock sink-actor mesh backed by a real :class:`_FileSinkActor`."""
    sink = _FileSinkActor() if sink is None else sink
    mesh = MagicMock()
    for name in (
        "begin_byte_transfer",
        "write_byte_transfer_chunk",
        "pull_byte_transfer_chunks",
        "finish_byte_transfer",
        "set_file_mode",
        "abort_byte_transfer",
        "rdma_backend",
    ):

        def choose(*args, _name=name):
            future = MagicMock()
            try:
                future.get.return_value = call_endpoint(sink, _name, *args)
            except Exception as exc:
                future.get.side_effect = exc
            return future

        getattr(mesh, name).choose.side_effect = choose
    return mesh


def _source_file(tmp_path, data: bytes) -> str:
    path = tmp_path / "src.bin"
    path.write_bytes(data)
    return str(path)


class TestFileTransport:
    @pytest.mark.parametrize("raw", ["auto", "ethernet", "rdma", " RDMA "])
    def test_valid_values(self, monkeypatch, raw):
        monkeypatch.setenv("AGENTBOX_FILE_TRANSPORT", raw)
        assert _file_transport() == raw.strip().lower()

    def test_default_and_invalid_are_auto(self, monkeypatch):
        monkeypatch.delenv("AGENTBOX_FILE_TRANSPORT", raising=False)
        assert _file_transport() == "auto"
        monkeypatch.setenv("AGENTBOX_FILE_TRANSPORT", "infiniband")
        assert _file_transport() == "auto"


class TestSlotPool:
    def test_registers_lazily_and_reuses_released_slots(self):
        pool = _FakePool(capacity=2)
        first = pool.acquire()
        pool.release(first)
        assert pool.acquire() is first
        assert pool.registered == 1

    def test_never_waits_and_keeps_only_capacity_registered(self):
        pool = _FakePool(capacity=2)
        held = [pool.acquire() for _ in range(3)]
        assert [slot.pooled for slot in held] == [True, True, False]
        for slot in held:
            pool.release(slot)
        assert pool.free == 2
        assert {pool.acquire().pooled, pool.acquire().pooled} == {True}
        assert pool.registered == 3

    def test_overflow_slots_are_unregistered_and_unmapped(self):
        # Discarding an overflow slot without dropping its registration kept
        # every overflow chunk pinned for the life of the process.
        pool = _FakePool(capacity=1)
        pooled, overflow = pool.acquire(), pool.acquire()
        pool.release(overflow)
        pool.release(pooled)
        assert overflow.handle.dropped and overflow.buffer.closed
        assert not pooled.handle.dropped and not pooled.buffer.closed

    def test_limit_bounds_registered_slots(self):
        pool = _FakePool(capacity=1)
        pool._limit = 2
        held = [pool.acquire(), pool.acquire()]
        with pytest.raises(rdma.SlotBudgetExhausted):
            pool.acquire()
        pool.release(held[1])  # an overflow slot frees budget
        assert not pool.acquire().pooled

    def test_overflow_slot_counts_until_deregistered(self):
        pool = _FakePool(capacity=1)
        pool._limit = 2
        pool.acquire()
        overflow = pool.acquire()
        overflow.handle.drop = MagicMock(side_effect=RuntimeError("busy"))
        pool.release(overflow)  # still pinned: keeps its budget
        with pytest.raises(rdma.SlotBudgetExhausted):
            pool.acquire()

    def test_failed_registration_frees_capacity(self):
        pool = _FakePool(capacity=1)
        pool._register = MagicMock(side_effect=RuntimeError("no MR"))  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            pool.acquire()
        del pool._register
        assert pool.acquire().pooled


class TestUseRdma:
    def _sink(self, backend: str = "ibverbs") -> MagicMock:
        sink = MagicMock()
        sink.rdma_backend.choose.return_value.get.return_value = backend
        return sink

    def test_ethernet_and_empty_files_never_use_rdma(self):
        source = _FileSourceActor()
        assert not source._use_rdma("ethernet", self._sink(), RDMA_MIN_FILE_SIZE)
        assert not source._use_rdma("rdma", self._sink(), 0)

    def test_forced_rdma_skips_capability_checks(self):
        sink = self._sink("tcp")
        assert _FileSourceActor()._use_rdma("rdma", sink, 1)
        sink.rdma_backend.choose.assert_not_called()

    def test_auto_needs_ibverbs_on_both_ends_and_caches_it(self, monkeypatch):
        monkeypatch.setattr(rdma, "backend", lambda: "ibverbs")
        source, sink = _FileSourceActor(), self._sink()
        assert not source._use_rdma("auto", sink, RDMA_MIN_FILE_SIZE - 1)
        assert source._use_rdma("auto", sink, RDMA_MIN_FILE_SIZE)
        assert source._use_rdma("auto", sink, RDMA_MIN_FILE_SIZE)
        sink.rdma_backend.choose.assert_called_once_with()

    @pytest.mark.parametrize(
        ("local", "peer"), [("tcp", "ibverbs"), ("ibverbs", "tcp")]
    )
    def test_auto_rejects_emulated_rdma(self, monkeypatch, local, peer):
        monkeypatch.setattr(rdma, "backend", lambda: local)
        assert not _FileSourceActor()._use_rdma(
            "auto", self._sink(peer), RDMA_MIN_FILE_SIZE
        )

    def test_auto_treats_probe_failure_as_unavailable(self, monkeypatch):
        monkeypatch.setattr(rdma, "backend", lambda: "ibverbs")
        sink = MagicMock()
        sink.rdma_backend.choose.return_value.get.side_effect = RuntimeError
        source = _FileSourceActor()
        assert not source._use_rdma("auto", sink, RDMA_MIN_FILE_SIZE)
        assert source._rdma_ok is False


class TestRdmaStream:
    @pytest.mark.parametrize("size", [1, 8, 21, 64])
    @pytest.mark.parametrize("depth", [1, 3])
    def test_pulls_every_chunk_and_verifies(self, tmp_path, pools, size, depth):
        data = os.urandom(size)
        dst = str(tmp_path / "out" / "dst.bin")
        sink = _sink_into(tmp_path)

        result = call_endpoint(
            _FileSourceActor(),
            "transfer_file_bytes_to",
            _source_file(tmp_path, data),
            sink,
            dst,
            size,
            4,
            depth,
            0o600,
            "rdma",
        )

        digests = chunk_io.PieceDigests(size)
        for at in range(0, size, 2):
            digests.hash_piece(memoryview(data)[at : at + 2], at)
        digest = digests.hexdigest()
        assert result == FileTransferDigests(digest, digest, "rdma")
        with open(dst, "rb") as f:
            assert f.read() == data
        assert os.stat(dst).st_mode & 0o777 == 0o600
        sink.write_byte_transfer_chunk.choose.assert_not_called()
        batches = [
            [offset for offset, _, _ in c.args[1]]
            for c in sink.pull_byte_transfer_chunks.choose.call_args_list
        ]
        offsets = list(range(0, size, 8))
        assert batches == [
            offsets[i : i + rdma.BATCH_CHUNKS]
            for i in range(0, len(offsets), rdma.BATCH_CHUNKS)
        ]
        # Every slot is back in its pool, and in-flight batches never exceed
        # the pipeline depth.
        in_flight = min(depth, rdma.BATCHES_IN_FLIGHT) * rdma.BATCH_CHUNKS
        assert pools["source"].free == pools["source"].registered <= in_flight
        # The sink overlaps reads with up to `depth` unfinished chunk writes.
        assert (
            pools["sink"].free == pools["sink"].registered <= depth + rdma.BATCH_CHUNKS
        )

    def test_auto_falls_back_to_ethernet_once_and_opens_breaker(
        self, tmp_path, pools, monkeypatch
    ):
        monkeypatch.setattr(ft, "RDMA_MIN_FILE_SIZE", 1)
        monkeypatch.setattr(rdma, "backend", lambda: "ibverbs")
        pools["source"].fail = "injected failure"
        data = os.urandom(20)
        source_path = _source_file(tmp_path, data)
        sink_actor = _FileSinkActor()
        sink = _sink_into(tmp_path, sink_actor)
        source = _FileSourceActor()
        args = (source_path, sink, str(tmp_path / "dst"), 20, 4, 2, None, "auto")

        result = call_endpoint(source, "transfer_file_bytes_to", *args)

        assert result.transport == "ethernet"
        assert result.source_sha256 == result.sink_sha256
        assert (tmp_path / "dst").read_bytes() == data
        assert source._rdma_ok is False
        assert sink_actor._sink_byte_transfers == {}
        assert sink.abort_byte_transfer.choose.call_count == 1
        assert pools["source"].free == pools["source"].registered

        sink.pull_byte_transfer_chunks.choose.reset_mock()
        assert call_endpoint(source, "transfer_file_bytes_to", *args).transport == (
            "ethernet"
        )
        sink.pull_byte_transfer_chunks.choose.assert_not_called()

    def test_memory_budget_falls_back_for_one_file_only(
        self, tmp_path, pools, monkeypatch
    ):
        monkeypatch.setattr(ft, "RDMA_MIN_FILE_SIZE", 1)
        monkeypatch.setattr(rdma, "backend", lambda: "ibverbs")
        pools["source"]._limit = 0  # every slot beyond the (empty) pool is over budget
        pools["source"]._capacity = 0
        data = os.urandom(20)
        sink_actor = _FileSinkActor()
        sink = _sink_into(tmp_path, sink_actor)
        source = _FileSourceActor()
        source._rdma_ok = True
        args = (
            _source_file(tmp_path, data),
            sink,
            str(tmp_path / "dst"),
            20,
            4,
            2,
            None,
            "auto",
        )

        result = call_endpoint(source, "transfer_file_bytes_to", *args)

        assert result.transport == "ethernet"
        assert (tmp_path / "dst").read_bytes() == data
        assert source._rdma_ok is True  # memory pressure does not open the breaker

    def test_forced_rdma_raises_instead_of_falling_back(self, tmp_path, pools):
        pools["source"].fail = "injected failure"
        sink_actor = _FileSinkActor()
        sink = _sink_into(tmp_path, sink_actor)

        with pytest.raises(_RdmaError, match="RDMA read failed: injected failure"):
            call_endpoint(
                _FileSourceActor(),
                "transfer_file_bytes_to",
                _source_file(tmp_path, b"x" * 20),
                sink,
                str(tmp_path / "dst"),
                20,
                4,
                2,
                None,
                "rdma",
            )

        sink.write_byte_transfer_chunk.choose.assert_not_called()
        assert sink_actor._sink_byte_transfers == {}
        assert pools["source"].free == pools["source"].registered
        assert issubclass(_RdmaError, FileCopyError)

    def test_registration_failure_is_an_rdma_error(self, tmp_path, pools):
        pools["source"]._register = MagicMock(side_effect=RuntimeError("no MR"))  # type: ignore[method-assign]
        with pytest.raises(_RdmaError, match="register RDMA source memory"):
            call_endpoint(
                _FileSourceActor(),
                "transfer_file_bytes_to",
                _source_file(tmp_path, b"abc"),
                _sink_into(tmp_path),
                str(tmp_path / "dst"),
                3,
                4,
                2,
                None,
                "rdma",
            )


def _broken_handle(size: int) -> _FakeHandle:
    handle = _FakeHandle(memoryview(bytearray(size)))
    handle.fail = "injected failure"
    return handle


class TestSinkPull:
    def _begin(self, sink: _FileSinkActor, path: str, size: int) -> None:
        call_endpoint(sink, "begin_byte_transfer", "t", path, size, 2)

    def test_read_failure_is_returned_and_does_not_advance(self, pools):
        sink = _FileSinkActor()
        with tempfile.TemporaryDirectory() as d:
            self._begin(sink, os.path.join(d, "f"), 4)
            handle = _broken_handle(8)
            error = call_endpoint(
                sink, "pull_byte_transfer_chunks", "t", [(0, 4, handle)]
            )
            assert error == "RDMA read failed: injected failure"
            assert sink._sink_byte_transfers["t"].next_offset == 0
            call_endpoint(sink, "abort_byte_transfer", "t")
        assert pools["sink"].free == 1

    def test_pulls_after_a_failure_repeat_it_without_reading(self, pools):
        sink = _FileSinkActor()
        with tempfile.TemporaryDirectory() as d:
            self._begin(sink, os.path.join(d, "f"), 16)
            call_endpoint(
                sink, "pull_byte_transfer_chunks", "t", [(0, 8, _broken_handle(8))]
            )
            later = MagicMock()
            error = call_endpoint(
                sink, "pull_byte_transfer_chunks", "t", [(8, 8, later)]
            )
            assert error == "RDMA read failed: injected failure"
            later.read_into.assert_not_called()
            call_endpoint(sink, "abort_byte_transfer", "t")

    def test_oversized_handle_is_rejected(self, pools):
        sink = _FileSinkActor()
        with tempfile.TemporaryDirectory() as d:
            self._begin(sink, os.path.join(d, "f"), 4)
            handle = _FakeHandle(memoryview(bytearray(16)))
            error = call_endpoint(
                sink, "pull_byte_transfer_chunks", "t", [(0, 4, handle)]
            )
            assert "does not fit" in error
            call_endpoint(sink, "abort_byte_transfer", "t")

    def test_out_of_order_pull_raises(self, pools):
        sink = _FileSinkActor()
        with tempfile.TemporaryDirectory() as d:
            self._begin(sink, os.path.join(d, "f"), 16)
            handle = _FakeHandle(memoryview(bytearray(8)))
            with pytest.raises(FileCopyError, match="Out-of-order"):
                call_endpoint(sink, "pull_byte_transfer_chunks", "t", [(8, 8, handle)])
            call_endpoint(sink, "abort_byte_transfer", "t")


class TestDirectIo:
    def _state(self):
        return MagicMock(fd=10, direct_fd=20)

    def _writes(self, monkeypatch):
        calls: list[tuple[int, bytes, int]] = []

        def record(fd, view, offset):
            calls.append((fd, bytes(view), offset))

        monkeypatch.setattr(chunk_io, "pwrite_all", record)
        return calls

    def test_registered_chunk_writes_aligned_part_directly(self, monkeypatch):
        monkeypatch.setattr(chunk_io, "DIRECT_IO_ALIGN", 4)
        calls = self._writes(monkeypatch)
        jobs = _FileSinkActor._submit_writes(
            self._state(), 8, memoryview(b"abcdefghij"), direct=True
        )
        chunk_io.wait_all(jobs)
        assert sorted(calls) == [(10, b"ij", 16), (20, b"abcdefgh", 8)]

    def test_messages_and_unsupported_filesystems_stay_buffered(self, monkeypatch):
        monkeypatch.setattr(chunk_io, "DIRECT_IO_ALIGN", 4)
        calls = self._writes(monkeypatch)
        state = self._state()
        chunk_io.wait_all(
            _FileSinkActor._submit_writes(state, 0, memoryview(b"abcd"), direct=False)
        )
        state.direct_fd = None
        chunk_io.wait_all(
            _FileSinkActor._submit_writes(state, 4, memoryview(b"efgh"), direct=True)
        )
        assert calls == [(10, b"abcd", 0), (10, b"efgh", 4)]

    def test_direct_write_rejected_by_filesystem_falls_back(self, monkeypatch):
        import errno

        monkeypatch.setattr(chunk_io, "DIRECT_IO_ALIGN", 4)
        calls: list[int] = []

        def write(fd, view, offset):
            if fd == 20:
                raise OSError(errno.EINVAL, "unaligned")
            calls.append(fd)

        monkeypatch.setattr(chunk_io, "pwrite_all", write)
        chunk_io.wait_all(
            _FileSinkActor._submit_writes(
                self._state(), 0, memoryview(b"abcd"), direct=True
            )
        )
        assert calls == [10]

    def test_mountinfo_picks_longest_mount_point(self, monkeypatch, tmp_path):
        mountinfo = tmp_path / "mountinfo"
        mountinfo.write_text(
            "1 0 0:1 / / rw - xfs /dev/root rw\n"
            "2 1 0:2 / /dev/shm rw - tmpfs tmpfs rw\n"
            "3 1 0:3 / /mnt/my\\040disk rw shared:1 - ext4 /dev/sdb rw\n"
        )
        real_open = open
        monkeypatch.setattr(
            "builtins.open",
            lambda path, *a, **k: real_open(
                mountinfo if path == "/proc/self/mountinfo" else path, *a, **k
            ),
        )
        monkeypatch.setattr(os.path, "realpath", lambda p: p)
        assert chunk_io.direct_io_filesystem("/tmp/f")
        assert not chunk_io.direct_io_filesystem("/dev/shm/f")
        assert chunk_io.direct_io_filesystem("/mnt/my disk/f")


class TestPoolCapacity:
    @pytest.mark.parametrize(
        ("memory", "slots"),
        [
            (1 << 30, 2 * rdma.BATCH_CHUNKS * rdma.BATCHES_IN_FLIGHT),  # floor
            (128 << 30, (128 << 30) // 64 // rdma.CHUNK_SIZE),  # 1/64 of memory
            (4 << 40, rdma.MAX_POOL_BYTES // rdma.CHUNK_SIZE),  # cap
        ],
    )
    def test_scales_with_memory_between_floor_and_cap(self, monkeypatch, memory, slots):
        monkeypatch.setattr(rdma, "_memory_limit", lambda: memory)
        assert rdma.pool_capacity() == slots

    def test_limit_is_a_fraction_of_memory(self, monkeypatch):
        monkeypatch.setattr(rdma, "_memory_limit", lambda: 400 << 30)
        assert (
            rdma.pool_limit()
            == (400 << 30) // rdma.MEMORY_FRACTION_LIMIT // rdma.CHUNK_SIZE
        )
        monkeypatch.setattr(rdma, "_memory_limit", lambda: 8 << 30)
        assert rdma.pool_limit() == rdma.pool_capacity()

    def test_memory_limit_honors_ancestor_cgroup(self, monkeypatch):
        # Under Slurm the job's limit sits above the task's own cgroup.
        files = {
            "/proc/self/cgroup": "0::/job_1/step_0/task_0\n",
            "/sys/fs/cgroup/job_1/step_0/task_0/memory.max": "max\n",
            "/sys/fs/cgroup/job_1/step_0/memory.max": "max\n",
            "/sys/fs/cgroup/job_1/memory.max": f"{4 << 30}\n",
        }

        def fake_open(path, *args, **kwargs):
            if path not in files:
                raise FileNotFoundError(path)
            return io.StringIO(files[path])

        monkeypatch.setattr(rdma, "open", fake_open, raising=False)
        assert rdma._memory_limit() == 4 << 30

    def test_memory_limit_is_positive(self):
        assert rdma._memory_limit() > 0
