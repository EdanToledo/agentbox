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

"""Tests for the streamed actor-to-actor file copy module."""

from __future__ import annotations

import asyncio
import os
import stat
import tarfile
import tempfile
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from _endpoints import call_endpoint, endpoint_method
from agentbox.remote import chunk_io
from agentbox.remote.file_transfer import (
    _acopy_file,
    _ArchiveActor,
    _atransfer,
    _copy_file,
    _FileSinkActor,
    _FileSourceActor,
    _pipeline_depth,
    _transfer,
    _TRANSFER_CLEANUP_TIMEOUT,
    DEFAULT_PIPELINE_DEPTH,
    FileCopyError,
    FileMetadata,
    FileTransferDigests,
    PICKLE_CHUNK_SIZE,
    TransferEndpoints,
)


@pytest.fixture(autouse=True)
def byte_pieces(monkeypatch):
    """Hash single-byte pieces so tiny test chunks stay piece-aligned."""
    monkeypatch.setattr(chunk_io, "PIECE_SIZE", 1)


def _digest(data: bytes) -> str:
    """Return the transfer digest of *data*."""
    digests = chunk_io.PieceDigests(len(data))
    for offset in range(len(data)):
        digests.hash_piece(memoryview(data)[offset : offset + 1], offset)
    return digests.hexdigest()


class _AwaitableValue:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def resolve():
            return self.value

        return resolve().__await__()


def _escaping_archive(tmp_path):
    path = str(tmp_path / "evil.tar")
    with tarfile.open(path, "w") as tar:
        tar.addfile(tarfile.TarInfo("../escaped"))
    return path


class TestPipelineDepth:
    def test_pipeline_depth_from_environment(self, monkeypatch):
        monkeypatch.setenv("AGENTBOX_TRANSFER_PIPELINE_DEPTH", "7")
        assert _pipeline_depth() == 7

    def test_pipeline_depth_is_bounded(self, monkeypatch):
        monkeypatch.setenv("AGENTBOX_TRANSFER_PIPELINE_DEPTH", "1000")
        assert _pipeline_depth() == 64

    def test_invalid_pipeline_depth_uses_default(self, monkeypatch):
        monkeypatch.setenv("AGENTBOX_TRANSFER_PIPELINE_DEPTH", "invalid")
        assert _pipeline_depth() == DEFAULT_PIPELINE_DEPTH


class TestFileSourceActor:
    def test_get_metadata(self):
        actor = _FileSourceActor()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = f.name

        try:
            os.chmod(path, 0o644)
            md = call_endpoint(actor, "get_metadata", path)
            assert isinstance(md, FileMetadata)
            assert md.size == 11
            assert md.mode == 0o644
            assert md.filename == os.path.basename(path)
        finally:
            os.unlink(path)

    def test_get_metadata_reports_directories(self, tmp_path):
        actor = _FileSourceActor()
        assert call_endpoint(actor, "get_metadata", str(tmp_path)).is_dir
        (tmp_path / "f").write_text("x")
        assert not call_endpoint(actor, "get_metadata", str(tmp_path / "f")).is_dir

    def test_get_metadata_file_not_found(self):
        actor = _FileSourceActor()
        with pytest.raises((FileNotFoundError, OSError)):
            call_endpoint(actor, "get_metadata", "/nonexistent/path/to/file.txt")

    def test_transfer_file_bytes_goes_directly_to_sink(self):
        actor = _FileSourceActor()
        sink = MagicMock()
        sink.begin_byte_transfer.choose.return_value.get.return_value = None
        sink.write_byte_transfer_chunk.choose.return_value.get.return_value = None
        expected_digest = _digest(b"abcdefghij")
        sink.finish_byte_transfer.choose.return_value.get.return_value = expected_digest

        with tempfile.NamedTemporaryFile() as source:
            source.write(b"abcdefghij")
            source.flush()
            result = call_endpoint(
                actor,
                "transfer_file_bytes_to",
                source.name,
                sink,
                "/dst/f",
                10,
                4,
                2,
                0o644,
            )

        transfer_id = sink.begin_byte_transfer.choose.call_args.args[0]
        sink.begin_byte_transfer.choose.assert_called_once_with(
            transfer_id,
            "/dst/f",
            10,
            2,
        )
        assert [
            call.args for call in sink.write_byte_transfer_chunk.choose.call_args_list
        ] == [
            (transfer_id, 0, b"abcd"),
            (transfer_id, 4, b"efgh"),
            (transfer_id, 8, b"ij"),
        ]
        sink.set_file_mode.choose.assert_called_once_with("/dst/f", 0o644)
        assert result == FileTransferDigests(expected_digest, expected_digest)

    def test_transfer_file_bytes_accumulates_short_reads(self):
        actor = _FileSourceActor()
        sink = MagicMock()
        sink.begin_byte_transfer.choose.return_value.get.return_value = None
        sink.write_byte_transfer_chunk.choose.return_value.get.return_value = None
        expected_digest = _digest(b"abcdef")
        sink.finish_byte_transfer.choose.return_value.get.return_value = expected_digest
        real_pread = os.pread

        def short_pread(fd, length, offset):
            return real_pread(fd, min(length, 3), offset)

        with (
            tempfile.NamedTemporaryFile() as source,
            patch("os.pread", side_effect=short_pread) as pread,
        ):
            source.write(b"abcdef")
            source.flush()
            result = call_endpoint(
                actor,
                "transfer_file_bytes_to",
                source.name,
                sink,
                "/dst/f",
                6,
                4,
                2,
                0o644,
            )

        transfer_id = sink.begin_byte_transfer.choose.call_args.args[0]
        assert [c.args[1:] for c in pread.call_args_list] == [(4, 0), (1, 3), (2, 4)]
        assert [
            write.args for write in sink.write_byte_transfer_chunk.choose.call_args_list
        ] == [
            (transfer_id, 0, b"abcd"),
            (transfer_id, 4, b"ef"),
        ]
        sink.set_file_mode.choose.assert_called_once_with("/dst/f", 0o644)
        assert result == FileTransferDigests(expected_digest, expected_digest)

    def test_transfer_file_bytes_defers_mode_until_digests_match(self):
        actor = _FileSourceActor()
        sink = MagicMock()
        sink.begin_byte_transfer.choose.return_value.get.return_value = None
        sink.write_byte_transfer_chunk.choose.return_value.get.return_value = None
        source_digest = _digest(b"content")
        sink_digest = _digest(b"changed")
        sink.finish_byte_transfer.choose.return_value.get.return_value = sink_digest

        with tempfile.NamedTemporaryFile() as source:
            source.write(b"content")
            source.flush()
            result = call_endpoint(
                actor,
                "transfer_file_bytes_to",
                source.name,
                sink,
                "/dst/f",
                7,
                7,
                1,
                0o444,
            )

        sink.set_file_mode.choose.assert_not_called()
        assert result == FileTransferDigests(source_digest, sink_digest)

    def test_transfer_file_bytes_aborts_when_setup_acknowledgement_fails(self):
        actor = _FileSourceActor()
        sink = MagicMock()
        sink.begin_byte_transfer.choose.return_value.get.side_effect = RuntimeError(
            "ack lost"
        )
        sink.abort_byte_transfer.choose.return_value.get.return_value = None

        with pytest.raises(RuntimeError, match="ack lost"):
            call_endpoint(
                actor,
                "transfer_file_bytes_to",
                "/src/f",
                sink,
                "/dst/f",
                6,
                4,
                2,
                0o644,
            )

        transfer_id = sink.begin_byte_transfer.choose.call_args.args[0]
        sink.abort_byte_transfer.choose.assert_called_once_with(transfer_id)

    def test_transfer_file_bytes_bounds_error_cleanup_waits(self):
        actor = _FileSourceActor()
        sink = MagicMock()
        sink.begin_byte_transfer.choose.return_value.get.return_value = None
        first_write = MagicMock()
        first_write.get.side_effect = RuntimeError("write failed")
        clock = [10.0]

        def slow_timeout(timeout: float) -> None:
            clock[0] += 1.0
            raise TimeoutError

        def slow_ack(timeout: float) -> None:
            clock[0] += 1.0

        second_write = MagicMock()
        second_write.get.side_effect = slow_timeout
        third_write = MagicMock()
        third_write.get.side_effect = slow_ack
        sink.write_byte_transfer_chunk.choose.side_effect = [
            first_write,
            second_write,
            third_write,
        ]
        abort = sink.abort_byte_transfer.choose.return_value

        # Keep the clock mock local: replacing ``time.monotonic`` on the shared
        # stdlib module can terminate Monarch's background asyncio loops.
        with (
            tempfile.NamedTemporaryFile() as source,
            patch(
                "agentbox.remote.file_transfer._monotonic",
                side_effect=lambda: clock[0],
            ),
        ):
            source.write(b"abcdef")
            source.flush()
            with pytest.raises(RuntimeError, match="write failed"):
                call_endpoint(
                    actor,
                    "transfer_file_bytes_to",
                    source.name,
                    sink,
                    "/dst/f",
                    6,
                    2,
                    4,
                    0o644,
                )

        # Each cleanup wait gets only what the earlier waits left of the budget.
        second_write.get.assert_called_once_with(timeout=_TRANSFER_CLEANUP_TIMEOUT)
        third_write.get.assert_called_once_with(timeout=_TRANSFER_CLEANUP_TIMEOUT - 1.0)
        abort.get.assert_called_once_with(timeout=_TRANSFER_CLEANUP_TIMEOUT - 2.0)


class TestArchiveActor:
    def test_endpoints_are_async_so_others_run_meanwhile(self):
        import inspect

        for name in ("pack_directory", "unpack_archive", "discard"):
            method = endpoint_method(_ArchiveActor, name)
            assert inspect.iscoroutinefunction(method), name

    async def test_pack_directory_returns_archive_and_discard_removes_it(
        self, tmp_path
    ):
        actor = _ArchiveActor()
        (tmp_path / "f").write_text("x")

        archive = await call_endpoint(actor, "pack_directory", str(tmp_path))

        assert os.path.getsize(archive.filename) == archive.size
        with tarfile.open(archive.filename) as tar:
            assert sorted(tar.getnames()) == [".", "f"]
        await call_endpoint(actor, "discard", archive.filename)
        assert not os.path.exists(archive.filename)
        await call_endpoint(actor, "discard", archive.filename)  # Idempotent.

    async def test_unpack_removes_archive_even_on_failure(self, tmp_path):
        archive = _escaping_archive(tmp_path)

        with pytest.raises(ValueError, match="leaves the destination"):
            await call_endpoint(
                _ArchiveActor(), "unpack_archive", archive, str(tmp_path / "dest")
            )

        assert not os.path.exists(archive)
        assert not (tmp_path / "escaped").exists()


class TestTransferEndpoints:
    def test_spawns_separate_source_and_sink_actors(self):
        proc_mesh = MagicMock()
        proc_mesh.spawn.side_effect = lambda name, cls: MagicMock(name=name)
        endpoints = TransferEndpoints.spawn(proc_mesh)
        classes = [call.args[1] for call in proc_mesh.spawn.call_args_list]
        assert classes == [_FileSourceActor, _FileSinkActor]
        assert endpoints.source is not endpoints.sink

    def test_archiver_is_spawned_once_on_demand_and_stopped(self):
        proc_mesh = MagicMock()
        proc_mesh.spawn.side_effect = lambda name, cls: MagicMock(name=name)
        endpoints = TransferEndpoints.spawn(proc_mesh)
        assert proc_mesh.spawn.call_count == 2

        archiver = endpoints.archiver()

        assert endpoints.archiver() is archiver
        assert proc_mesh.spawn.call_args.args[1] is _ArchiveActor
        assert proc_mesh.spawn.call_count == 3
        endpoints.stop()
        archiver.stop.return_value.get.assert_called_once()

    def test_stop_is_best_effort(self):
        endpoints = TransferEndpoints(MagicMock(), MagicMock(), MagicMock())
        endpoints.source.stop.side_effect = RuntimeError("gone")
        endpoints.stop()
        endpoints.sink.stop.return_value.get.assert_called_once()


class TestFileSinkActor:
    @pytest.mark.parametrize(
        ("relpath", "size", "mode"),
        [
            ("output.bin", 1024, 0o644),
            (os.path.join("sub", "dir", "output.bin"), 100, 0o600),
            ("empty.bin", 0, 0o644),
        ],
    )
    def test_prepare_path(self, relpath, size, mode):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, relpath)
            actor._prepare_path(path, size, mode)

            assert os.path.getsize(path) == size
            assert stat.S_IMODE(os.stat(path).st_mode) == mode

    def test_direct_byte_transfer_prepares_writes_and_verifies(self):
        actor = _FileSinkActor()
        payload = b"0123456789abcdef"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "output.bin")
            call_endpoint(
                actor,
                "begin_byte_transfer",
                "transfer",
                path,
                len(payload),
                2,
            )
            call_endpoint(
                actor,
                "write_byte_transfer_chunk",
                "transfer",
                0,
                payload[:8],
            )
            call_endpoint(
                actor,
                "write_byte_transfer_chunk",
                "transfer",
                8,
                payload[8:],
            )
            digest = call_endpoint(actor, "finish_byte_transfer", "transfer")
            call_endpoint(actor, "set_file_mode", path, 0o640)

            with open(path, "rb") as output:
                assert output.read() == payload
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
            assert digest == _digest(payload)

    def test_direct_byte_transfer_without_mode_creates_path(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "output.bin")
            call_endpoint(
                actor,
                "begin_byte_transfer",
                "transfer",
                path,
                4,
                1,
            )

            assert os.path.getsize(path) == 4
            call_endpoint(actor, "abort_byte_transfer", "transfer")

    def test_direct_byte_transfer_without_mode_resizes_existing_path(self):
        actor = _FileSinkActor()
        with tempfile.NamedTemporaryFile() as destination:
            destination.write(b"stale data")
            destination.flush()
            os.chmod(destination.name, 0o640)

            call_endpoint(
                actor,
                "begin_byte_transfer",
                "transfer",
                destination.name,
                4,
                1,
            )

            assert os.path.getsize(destination.name) == 4
            assert stat.S_IMODE(os.stat(destination.name).st_mode) == 0o640
            call_endpoint(actor, "abort_byte_transfer", "transfer")

    def test_background_write_failure_surfaces_at_finish_after_drain(self):
        actor = _FileSinkActor()
        closed_with_pending: list[bool] = []
        real_close = os.close

        def close(fd):
            state_pending = [s.pending for s in actor._sink_byte_transfers.values()]
            closed_with_pending.append(any(state_pending))
            real_close(fd)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "t", path, 4, 8)
            with patch(
                "agentbox.remote.chunk_io.pwrite_all",
                side_effect=OSError("disk full"),
            ):
                call_endpoint(actor, "write_byte_transfer_chunk", "t", 0, b"ab")
                call_endpoint(actor, "write_byte_transfer_chunk", "t", 2, b"cd")
                with (
                    patch("os.close", side_effect=close),
                    pytest.raises(OSError, match="disk full"),
                ):
                    call_endpoint(actor, "finish_byte_transfer", "t")
        assert closed_with_pending == [False]
        assert actor._sink_byte_transfers == {}

    def test_chunk_past_expected_size_is_rejected(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "t", path, 2, 1)
            with pytest.raises(FileCopyError, match="overruns"):
                call_endpoint(actor, "write_byte_transfer_chunk", "t", 0, b"abc")
            call_endpoint(actor, "abort_byte_transfer", "t")

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7])
    def test_digest_is_independent_of_chunk_size(self, chunk):
        actor = _FileSinkActor()
        payload = b"0123456"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "t", path, len(payload), 2)
            for at in range(0, len(payload), chunk):
                call_endpoint(
                    actor,
                    "write_byte_transfer_chunk",
                    "t",
                    at,
                    payload[at : at + chunk],
                )
            assert call_endpoint(actor, "finish_byte_transfer", "t") == _digest(payload)

    def test_direct_byte_transfer_rejects_resized_destination(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "transfer", path, 3, 1)
            call_endpoint(actor, "write_byte_transfer_chunk", "transfer", 0, b"abc")
            with open(path, "ab") as f:
                f.write(b"extra")

            with pytest.raises(FileCopyError, match="Size mismatch"):
                call_endpoint(actor, "finish_byte_transfer", "transfer")

    def test_direct_byte_transfer_rejects_deleted_destination(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "transfer", path, 3, 1)
            call_endpoint(actor, "write_byte_transfer_chunk", "transfer", 0, b"abc")
            os.remove(path)

            with pytest.raises(FileCopyError, match="replaced during transfer"):
                call_endpoint(actor, "finish_byte_transfer", "transfer")

    def test_direct_byte_transfer_rejects_replaced_destination(self):
        actor = _FileSinkActor()
        payload = b"expected"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            replacement = os.path.join(tmpdir, "replacement.bin")
            call_endpoint(
                actor,
                "begin_byte_transfer",
                "transfer",
                path,
                len(payload),
                1,
            )
            call_endpoint(
                actor,
                "write_byte_transfer_chunk",
                "transfer",
                0,
                payload,
            )
            with open(replacement, "wb") as destination:
                destination.write(payload)
            os.replace(replacement, path)

            with pytest.raises(FileCopyError, match="replaced during transfer"):
                call_endpoint(actor, "finish_byte_transfer", "transfer")

    def test_direct_empty_read_only_transfer_opens_before_chmod(self):
        actor = _FileSinkActor()
        real_open = os.open

        def enforce_write_permissions(path, flags, *args, **kwargs):
            if flags & os.O_WRONLY:
                mode = stat.S_IMODE(os.stat(path).st_mode)
                if not mode & stat.S_IWUSR:
                    raise PermissionError(path)
            return real_open(path, flags, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.bin")
            with patch(
                "agentbox.remote.file_transfer.os.open",
                side_effect=enforce_write_permissions,
            ):
                call_endpoint(
                    actor,
                    "begin_byte_transfer",
                    "transfer",
                    path,
                    0,
                    1,
                )
                digest = call_endpoint(actor, "finish_byte_transfer", "transfer")
                call_endpoint(actor, "set_file_mode", path, 0o444)

            assert digest == _digest(b"")
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o444

    def test_direct_byte_transfer_rejects_out_of_order_chunk(self):
        actor = _FileSinkActor()
        with tempfile.NamedTemporaryFile() as destination:
            call_endpoint(
                actor,
                "begin_byte_transfer",
                "transfer",
                destination.name,
                4,
                1,
            )
            with pytest.raises(FileCopyError, match="expected offset 0, got 2"):
                call_endpoint(
                    actor,
                    "write_byte_transfer_chunk",
                    "transfer",
                    2,
                    b"xx",
                )
            call_endpoint(actor, "abort_byte_transfer", "transfer")

    def test_abort_closes_only_its_own_transfer(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            for transfer_id in ("first", "second"):
                call_endpoint(
                    actor,
                    "begin_byte_transfer",
                    transfer_id,
                    os.path.join(tmpdir, transfer_id),
                    4,
                    1,
                )
            first = actor._sink_byte_transfers["first"]

            call_endpoint(actor, "abort_byte_transfer", "first")

            assert list(actor._sink_byte_transfers) == ["second"]
            with pytest.raises(OSError):
                os.fstat(first.fd)
            call_endpoint(actor, "write_byte_transfer_chunk", "second", 0, b"abcd")
            call_endpoint(actor, "finish_byte_transfer", "second")

    def test_finish_byte_transfer_rejects_incomplete_transfer(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "transfer", path, 4, 1)
            call_endpoint(actor, "write_byte_transfer_chunk", "transfer", 0, b"ab")

            with pytest.raises(FileCopyError, match="Incomplete transfer"):
                call_endpoint(actor, "finish_byte_transfer", "transfer")

            assert actor._sink_byte_transfers == {}

    def test_begin_byte_transfer_rejects_duplicate_id(self):
        actor = _FileSinkActor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.bin")
            call_endpoint(actor, "begin_byte_transfer", "transfer", path, 4, 1)
            with pytest.raises(FileCopyError, match="Duplicate"):
                call_endpoint(actor, "begin_byte_transfer", "transfer", path, 4, 1)
            call_endpoint(actor, "abort_byte_transfer", "transfer")


class TestTransfer:
    def test_streams_directly_to_sink(self, monkeypatch):
        source = MagicMock()
        sink = MagicMock()
        expected = FileTransferDigests("a" * 64, "b" * 64)
        source.transfer_file_bytes_to.choose.return_value.get.return_value = expected
        monkeypatch.setenv("AGENTBOX_TRANSFER_PIPELINE_DEPTH", "3")
        monkeypatch.setenv("AGENTBOX_FILE_TRANSPORT", "rdma")

        result = _transfer(
            source,
            sink,
            "/src/direct",
            "/dst/direct",
            PICKLE_CHUNK_SIZE * 2,
            sink_mode=0o644,
        )

        source.transfer_file_bytes_to.choose.assert_called_once_with(
            "/src/direct",
            sink,
            "/dst/direct",
            PICKLE_CHUNK_SIZE * 2,
            PICKLE_CHUNK_SIZE,
            3,
            0o644,
            "rdma",
            None,
        )
        assert result == expected

    def test_async_uses_same_interface(self):
        source = MagicMock()
        sink = MagicMock()
        expected = FileTransferDigests("a" * 64, "b" * 64)
        source.transfer_file_bytes_to.choose.return_value = _AwaitableValue(expected)

        result = asyncio.run(
            _atransfer(
                source,
                sink,
                "/src/direct",
                "/dst/direct",
                PICKLE_CHUNK_SIZE,
                sink_mode=0o600,
            )
        )

        source.transfer_file_bytes_to.choose.assert_called_once_with(
            "/src/direct",
            sink,
            "/dst/direct",
            PICKLE_CHUNK_SIZE,
            PICKLE_CHUNK_SIZE,
            DEFAULT_PIPELINE_DEPTH,
            0o600,
            "auto",
            None,
        )
        assert result == expected


class TestCopyFile:
    # Shared SHA-256 value returned by both source and sink mocks so the
    # checksum comparison passes without touching real files.
    _FAKE_SHA256 = "aabbccdd" * 8

    def _setup_actors(self, source_pm, sink_pm):
        """Return (source_actor, sink_actor) with common defaults mocked."""
        source_actor = MagicMock()
        sink_actor = MagicMock()
        source_pm.spawn.return_value = source_actor
        sink_pm.spawn.return_value = sink_actor
        return source_actor, sink_actor

    def test_copy_file_streams_with_fused_digests(self):
        source_pm = MagicMock()
        sink_pm = MagicMock()
        source_actor, sink_actor = self._setup_actors(source_pm, sink_pm)
        source_actor.get_metadata.choose.return_value.get.return_value = FileMetadata(
            size=100,
            mode=0o644,
            filename="test.txt",
        )
        source_actor.transfer_file_bytes_to.choose.return_value.get.return_value = (
            FileTransferDigests(self._FAKE_SHA256, self._FAKE_SHA256)
        )

        _copy_file(source_pm, sink_pm, "/src/file", "/dst/file")

        source_actor.transfer_file_bytes_to.choose.assert_called_once_with(
            "/src/file",
            sink_actor,
            "/dst/file",
            100,
            PICKLE_CHUNK_SIZE,
            DEFAULT_PIPELINE_DEPTH,
            0o644,
            "auto",
            ANY,
        )
        assert source_pm.spawn.call_count == 1
        assert sink_pm.spawn.call_count == 1
        source_actor.get_metadata.choose.assert_called_once_with("/src/file")
        # A verified copy had its transfer finished by the sink: no cleanup RPC.
        sink_actor.abort_byte_transfer.choose.assert_not_called()

    def test_copy_file_cleanup_spares_concurrent_transfers(self):
        """A finished copy must not abort other copies sharing the sink.

        Every copy to a mesh uses that mesh's one sink actor, so cleaning up
        all of its open transfers would cancel concurrent copies mid-stream.
        """
        source_pm = MagicMock()
        sink_pm = MagicMock()
        source_actor, sink_actor = self._setup_actors(source_pm, sink_pm)
        source_actor.get_metadata.choose.return_value.get.return_value = FileMetadata(
            size=4, mode=0o644, filename="a"
        )
        source_actor.transfer_file_bytes_to.choose.return_value.get.return_value = (
            FileTransferDigests(self._FAKE_SHA256, self._FAKE_SHA256)
        )

        _copy_file(source_pm, sink_pm, "/src/a", "/dst/a")
        _copy_file(source_pm, sink_pm, "/src/b", "/dst/b")

        sent = [
            c.args[8] for c in source_actor.transfer_file_bytes_to.choose.call_args_list
        ]
        aborted = [
            c.args[0] for c in sink_actor.abort_byte_transfer.choose.call_args_list
        ]
        assert len(set(sent)) == 2
        assert aborted == []  # verified copies never touch the shared sink's state
        assert not [c for c in sink_actor.method_calls if "cleanup" in c[0]]

    @pytest.mark.asyncio
    async def test_async_copy_file_cleanup_waits_are_bounded(self):
        source_actor = MagicMock()
        sink_actor = MagicMock()
        loop = asyncio.get_running_loop()
        metadata = loop.create_future()
        metadata.set_result(FileMetadata(size=10, mode=0o644, filename="f"))
        source_actor.get_metadata.choose.return_value = metadata
        failed = loop.create_future()
        failed.set_exception(RuntimeError("source actor failed"))
        source_actor.transfer_file_bytes_to.choose.return_value = failed
        sink_abort = loop.create_future()
        sink_abort.set_result(None)
        sink_actor.abort_byte_transfer.choose.return_value = sink_abort
        wait_for = asyncio.wait_for

        with patch(
            "agentbox.remote.file_transfer.asyncio.wait_for",
            wraps=wait_for,
        ) as mock_wait_for:
            with pytest.raises(RuntimeError, match="source actor failed"):
                await _acopy_file(
                    MagicMock(),
                    MagicMock(),
                    "/src/f",
                    "/dst/f",
                    source_actor=source_actor,
                    sink_actor=sink_actor,
                )

        transfer_id = source_actor.transfer_file_bytes_to.choose.call_args.args[8]
        sink_actor.abort_byte_transfer.choose.assert_called_once_with(transfer_id)
        assert mock_wait_for.await_count == 1
        for cleanup_call in mock_wait_for.await_args_list:
            timeout = cleanup_call.kwargs["timeout"]
            assert 0.0 <= timeout <= _TRANSFER_CLEANUP_TIMEOUT

    def test_copy_file_skips_cleanup_before_any_transfer(self):
        source_pm = MagicMock()
        sink_pm = MagicMock()
        source_actor, sink_actor = self._setup_actors(source_pm, sink_pm)

        source_actor.get_metadata.choose.return_value.get.side_effect = (
            FileNotFoundError("not found")
        )

        with pytest.raises(FileNotFoundError):
            _copy_file(source_pm, sink_pm, "/src/missing", "/dst/missing")

        sink_actor.abort_byte_transfer.choose.assert_not_called()

    def test_copy_file_cleans_abandoned_sink_after_source_failure(self):
        source_pm = MagicMock()
        sink_pm = MagicMock()
        source_actor, sink_actor = self._setup_actors(source_pm, sink_pm)
        source_actor.get_metadata.choose.return_value.get.return_value = FileMetadata(
            size=10,
            mode=0o644,
            filename="f.bin",
        )
        source_actor.transfer_file_bytes_to.choose.return_value.get.side_effect = (
            RuntimeError("source actor failed")
        )

        sink_abort = sink_actor.abort_byte_transfer.choose.return_value

        with pytest.raises(RuntimeError, match="source actor failed"):
            _copy_file(source_pm, sink_pm, "/src/f", "/dst/f")

        transfer_id = source_actor.transfer_file_bytes_to.choose.call_args.args[8]
        sink_actor.abort_byte_transfer.choose.assert_called_once_with(transfer_id)
        timeout = sink_abort.get.call_args.kwargs["timeout"]
        assert 0.0 <= timeout <= _TRANSFER_CLEANUP_TIMEOUT

    def test_copy_file_retries_on_checksum_mismatch(self):
        """Verify that a SHA-256 mismatch triggers up to _COPY_MAX_RETRIES attempts."""
        from agentbox.remote.file_transfer import _COPY_MAX_RETRIES

        source_pm = MagicMock()
        sink_pm = MagicMock()
        source_actor, sink_actor = self._setup_actors(source_pm, sink_pm)

        metadata = FileMetadata(size=10, mode=0o644, filename="f.bin")
        source_actor.get_metadata.choose.return_value.get.return_value = metadata
        source_actor.transfer_file_bytes_to.choose.return_value.get.return_value = (
            FileTransferDigests("aaa", "bbb")
        )

        with pytest.raises(FileCopyError, match="SHA-256 mismatch"):
            _copy_file(source_pm, sink_pm, "/src/f", "/dst/f")

        assert (
            source_actor.transfer_file_bytes_to.choose.call_count == _COPY_MAX_RETRIES
        )

    def test_copy_file_succeeds_on_second_attempt(self):
        """Verify success on the second attempt after one checksum mismatch."""
        source_pm = MagicMock()
        sink_pm = MagicMock()
        source_actor, sink_actor = self._setup_actors(source_pm, sink_pm)

        metadata = FileMetadata(size=10, mode=0o644, filename="f.bin")
        source_actor.get_metadata.choose.return_value.get.return_value = metadata
        source_actor.transfer_file_bytes_to.choose.return_value.get.side_effect = [
            FileTransferDigests("aaa", "bbb"),
            FileTransferDigests("aaa", "aaa"),
        ]

        _copy_file(source_pm, sink_pm, "/src/f", "/dst/f")

        assert source_actor.transfer_file_bytes_to.choose.call_count == 2


class TestCopyDirectory:
    _SHA = "ab" * 32

    def _actors(self):
        source, sink = MagicMock(), MagicMock()
        packer, unpacker = MagicMock(), MagicMock()
        source.get_metadata.choose.return_value.get.return_value = FileMetadata(
            4096, 0o755, "tree", is_dir=True
        )
        packer.pack_directory.choose.return_value.get.return_value = FileMetadata(
            512, 0o600, "/tmp/agentbox-x.tar"
        )
        source.transfer_file_bytes_to.choose.return_value.get.return_value = (
            FileTransferDigests(self._SHA, self._SHA)
        )
        return source, sink, packer, unpacker

    def _copy(self, source, sink, packer, unpacker):
        _copy_file(
            MagicMock(),
            MagicMock(),
            "/src/tree",
            "/dst/tree",
            source_actor=source,
            sink_actor=sink,
            source_archiver=lambda: packer,
            sink_archiver=lambda: unpacker,
        )

    def test_directory_travels_as_one_archive(self):
        source, sink, packer, unpacker = self._actors()

        self._copy(source, sink, packer, unpacker)

        packer.pack_directory.choose.assert_called_once_with("/src/tree")
        args = source.transfer_file_bytes_to.choose.call_args.args
        assert args[:4] == ("/tmp/agentbox-x.tar", sink, args[2], 512)
        staged = args[2]
        assert staged.startswith("/dst/.tree.") and staged.endswith(".tar")
        unpacker.unpack_archive.choose.assert_called_once_with(staged, "/dst/tree")
        packer.discard.choose.assert_called_once_with("/tmp/agentbox-x.tar")
        # The unpacker removed its archive while unpacking.
        unpacker.discard.choose.assert_not_called()
        # Packing and unpacking never occupy the transfer actors.
        source.pack_directory.choose.assert_not_called()
        sink.unpack_archive.choose.assert_not_called()

    def test_files_never_need_an_archiver(self):
        source, sink, _, _ = self._actors()
        source.get_metadata.choose.return_value.get.return_value = FileMetadata(
            1, 0o644, "f"
        )

        def unexpected():
            raise AssertionError("archiver requested for a file")

        _copy_file(
            MagicMock(),
            MagicMock(),
            "/src/f",
            "/dst/f",
            source_actor=source,
            sink_actor=sink,
            source_archiver=unexpected,
            sink_archiver=unexpected,
        )

    def test_failed_transfer_removes_both_archives(self):
        source, sink, packer, unpacker = self._actors()
        source.transfer_file_bytes_to.choose.return_value.get.side_effect = (
            RuntimeError("link down")
        )

        with pytest.raises(RuntimeError, match="link down"):
            self._copy(source, sink, packer, unpacker)

        staged = source.transfer_file_bytes_to.choose.call_args.args[2]
        unpacker.unpack_archive.choose.assert_not_called()
        unpacker.discard.choose.assert_called_once_with(staged)
        packer.discard.choose.assert_called_once_with("/tmp/agentbox-x.tar")

    def test_one_shot_archivers_without_endpoints(self):
        source, sink, _, _ = self._actors()
        source_pm, sink_pm = MagicMock(), MagicMock()
        packer, unpacker = source_pm.spawn.return_value, sink_pm.spawn.return_value
        packer.pack_directory.choose.return_value.get.return_value = FileMetadata(
            512, 0o600, "/tmp/agentbox-x.tar"
        )

        _copy_file(
            source_pm,
            sink_pm,
            "/src/tree",
            "/dst/tree",
            source_actor=source,
            sink_actor=sink,
        )

        assert source_pm.spawn.call_args.args[1] is _ArchiveActor
        assert sink_pm.spawn.call_args.args[1] is _ArchiveActor
        unpacker.unpack_archive.choose.assert_called_once()

    async def test_async_directory_follows_same_sequence(self):
        source, sink = MagicMock(), MagicMock()
        packer, unpacker = MagicMock(), MagicMock()
        source.get_metadata.choose.return_value = _AwaitableValue(
            FileMetadata(4096, 0o755, "tree", is_dir=True)
        )
        packer.pack_directory.choose.return_value = _AwaitableValue(
            FileMetadata(512, 0o600, "/tmp/agentbox-x.tar")
        )
        source.transfer_file_bytes_to.choose.return_value = _AwaitableValue(
            FileTransferDigests(self._SHA, self._SHA)
        )
        unpacker.unpack_archive.choose.return_value = _AwaitableValue(None)
        packer.discard.choose.return_value = _AwaitableValue(None)

        await _acopy_file(
            MagicMock(),
            MagicMock(),
            "/src/tree",
            "/dst/tree",
            source_actor=source,
            sink_actor=sink,
            source_archiver=lambda: packer,
            sink_archiver=lambda: unpacker,
        )

        staged = source.transfer_file_bytes_to.choose.call_args.args[2]
        unpacker.unpack_archive.choose.assert_called_once_with(staged, "/dst/tree")
        packer.discard.choose.assert_called_once_with("/tmp/agentbox-x.tar")


def _patched_local_mesh():
    local_mesh = MagicMock()
    local_mesh.spawn.side_effect = lambda name, cls: MagicMock(name=name)
    return local_mesh, patch(
        "agentbox.actors.machine.local_transfer_mesh", return_value=local_mesh
    )


class TestMachineUpload:
    def test_upload_delegates_to_copy_file(self):
        from agentbox.actors.machine import Machine

        mock_proc_mesh = MagicMock()
        machine = Machine(proc_mesh=mock_proc_mesh)
        mock_proc_mesh.spawn.side_effect = lambda name, cls: MagicMock(name=name)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            local_path = f.name

        try:
            local_mesh, patch_local = _patched_local_mesh()
            with (
                patch_local,
                patch("agentbox.actors.machine._copy_file") as mock_copy,
            ):
                machine.upload(local_path, "/remote/dest.txt")
                machine.upload(local_path, "/remote/dest2.txt")

            assert mock_copy.call_count == 2
            args, kwargs = mock_copy.call_args
            assert args == (
                local_mesh,
                mock_proc_mesh,
                os.path.abspath(local_path),
                "/remote/dest2.txt",
            )
            local, remote = machine._transfer
            assert kwargs == {
                "source_actor": local.source,
                "sink_actor": remote.sink,
                "source_archiver": local.archiver,
                "sink_archiver": remote.archiver,
            }
            # Endpoints are spawned once per Machine and stopped on release.
            assert local_mesh.spawn.call_count == 2
            machine.release()
            local.source.stop.assert_called_once()
            local.sink.stop.assert_called_once()
        finally:
            os.unlink(local_path)

    def test_upload_raises_on_missing_local(self):
        from agentbox.actors.machine import Machine

        mock_proc_mesh = MagicMock()
        machine = Machine(proc_mesh=mock_proc_mesh)

        with pytest.raises(FileCopyError, match="Local file or directory not found"):
            machine.upload("/nonexistent/file.txt", "/remote/dest.txt")


class TestMachineCopyTo:
    @staticmethod
    def _machines():
        from agentbox.actors.machine import Machine

        source_mesh, sink_mesh = MagicMock(), MagicMock()
        for mesh in (source_mesh, sink_mesh):
            mesh.spawn.side_effect = lambda name, cls: MagicMock(name=name)
        return Machine(proc_mesh=source_mesh), Machine(proc_mesh=sink_mesh)

    def test_copy_to_goes_worker_to_worker(self):
        source, sink = self._machines()
        _, patch_local = _patched_local_mesh()
        with patch_local, patch("agentbox.actors.machine._copy_file") as mock_copy:
            source.copy_to(sink, "/src/f", "/dst/f")

        args, kwargs = mock_copy.call_args
        assert args == (source._proc_mesh, sink._proc_mesh, "/src/f", "/dst/f")
        assert kwargs == {
            "source_actor": source._transfer[1].source,
            "sink_actor": sink._transfer[1].sink,
            "source_archiver": source._transfer[1].archiver,
            "sink_archiver": sink._transfer[1].archiver,
        }
        source.release()
        sink.release()

    async def test_acopy_to_goes_worker_to_worker(self):
        source, sink = self._machines()
        _, patch_local = _patched_local_mesh()
        with (
            patch_local,
            patch("agentbox.actors.machine._acopy_file", new=AsyncMock()) as mock_copy,
        ):
            await source.acopy_to(sink, "/src/f", "/dst/f")

        args, kwargs = mock_copy.await_args
        assert args == (source._proc_mesh, sink._proc_mesh, "/src/f", "/dst/f")
        assert kwargs == {
            "source_actor": source._transfer[1].source,
            "sink_actor": sink._transfer[1].sink,
            "source_archiver": source._transfer[1].archiver,
            "sink_archiver": sink._transfer[1].archiver,
        }
        source.release()
        sink.release()


class TestMachineDownload:
    def test_download_delegates_to_copy_file(self):
        from agentbox.actors.machine import Machine

        mock_proc_mesh = MagicMock()
        machine = Machine(proc_mesh=mock_proc_mesh)

        local_mesh, patch_local = _patched_local_mesh()
        with (
            patch_local,
            patch("agentbox.actors.machine._copy_file") as mock_copy,
        ):
            machine.download("/remote/src.txt", "/tmp/local_dest.txt")

        args, kwargs = mock_copy.call_args
        assert args == (
            mock_proc_mesh,
            local_mesh,
            "/remote/src.txt",
            os.path.abspath("/tmp/local_dest.txt"),
        )
        local, remote = machine._transfer
        assert kwargs == {
            "source_actor": remote.source,
            "sink_actor": local.sink,
            "source_archiver": remote.archiver,
            "sink_archiver": local.archiver,
        }
        machine.release()


def test_local_transfer_mesh_imports_agentbox_before_returning(monkeypatch):
    """The fresh client-side proc must load agentbox on one thread first."""
    from agentbox.remote import file_transfer

    mesh = MagicMock()
    host = MagicMock()
    host.spawn_procs.return_value = mesh
    monkeypatch.setattr(file_transfer, "_local_meshes", {})
    monkeypatch.setattr("monarch.actor.this_host", lambda: host)

    assert file_transfer.local_transfer_mesh() is mesh
    assert mesh.spawn.call_args.args[1] is file_transfer._ImportWarmup
    mesh.spawn.return_value.ready.choose.return_value.get.assert_called_once_with()
    assert file_transfer.local_transfer_mesh() is mesh  # cached: no second warm-up
    assert mesh.spawn.call_count == 1


def test_local_transfer_mesh_stopped_if_warmup_fails(monkeypatch):
    from agentbox.remote import file_transfer

    mesh = MagicMock()
    mesh.spawn.return_value.ready.choose.return_value.get.side_effect = RuntimeError(
        "proc died"
    )
    host = MagicMock()
    host.spawn_procs.return_value = mesh
    monkeypatch.setattr(file_transfer, "_local_meshes", {})
    monkeypatch.setattr("monarch.actor.this_host", lambda: host)

    with pytest.raises(RuntimeError, match="proc died"):
        file_transfer.local_transfer_mesh()
    mesh.stop.assert_called_once_with()
    assert file_transfer._local_meshes == {}


def test_local_transfer_meshes_fill_then_spill(monkeypatch):
    from agentbox.remote import file_transfer

    host = MagicMock()
    host.spawn_procs.side_effect = lambda name: MagicMock(name=name)
    monkeypatch.setattr(file_transfer, "_local_meshes", {})
    monkeypatch.setattr("monarch.actor.this_host", lambda: host)
    per = file_transfer.MACHINES_PER_TRANSFER_PROC
    procs = file_transfer.LOCAL_TRANSFER_PROCS

    first = file_transfer.local_transfer_mesh(0)
    assert (
        file_transfer.local_transfer_mesh(per - 1) is first
    )  # small clients: one proc
    second = file_transfer.local_transfer_mesh(per)
    assert second is not first
    assert file_transfer.local_transfer_mesh(per * procs) is first  # wraps around
    assert host.spawn_procs.call_count == 2
