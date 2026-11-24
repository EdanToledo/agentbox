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

"""Tests for packing and unpacking directory trees (agentbox.remote.tree_archive)."""

from __future__ import annotations

import errno
import io
import os
import shutil
import stat
import subprocess
import tarfile
from unittest.mock import patch

import pytest
from agentbox.remote import tree_archive


def _is_gnu(tool: str) -> bool:
    if shutil.which(tool) is None:
        return False
    version = subprocess.run([tool, "--version"], capture_output=True, text=True)
    return "GNU" in version.stdout


def _round_trip(src, dest, tmp_path):
    archive = str(tmp_path / "tree.tar")
    tree_archive.pack(str(src), archive)
    tree_archive.unpack(archive, str(dest))


def _archive(tmp_path, *members):
    """Write a tar of (TarInfo, bytes | None) *members*; return its path."""
    path = str(tmp_path / "crafted.tar")
    with tarfile.open(path, "w") as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
            tar.addfile(info, None if data is None else io.BytesIO(data))
    return path


def _member(name, kind=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, linkname
    return info


class TestRoundTrip:
    def test_keeps_tree_modes_mtimes_and_symlinks(self, tmp_path):
        src = tmp_path / "src"
        (src / "sub" / "empty").mkdir(parents=True)
        (src / "top.txt").write_text("top")
        (src / "sub" / "run.sh").write_text("#!/bin/sh\n")
        os.chmod(src / "sub" / "run.sh", 0o755)
        os.chmod(src / "top.txt", 0o600)
        os.utime(src / "top.txt", (1_000_000, 1_000_000))
        os.symlink("top.txt", src / "rel-link")
        os.symlink("/usr/bin/env", src / "abs-link")  # e.g. a venv's python
        os.symlink("sub", src / "dir-link")
        os.mkfifo(src / "fifo")

        dest = tmp_path / "a" / "b"
        _round_trip(src, dest, tmp_path)

        assert (dest / "top.txt").read_text() == "top"
        assert (dest / "sub" / "run.sh").read_text() == "#!/bin/sh\n"
        assert stat.S_IMODE(os.stat(dest / "sub" / "run.sh").st_mode) == 0o755
        assert stat.S_IMODE(os.stat(dest / "top.txt").st_mode) == 0o600
        assert os.stat(dest / "top.txt").st_mtime == 1_000_000
        assert (dest / "sub" / "empty").is_dir()
        assert os.readlink(dest / "rel-link") == "top.txt"
        assert os.readlink(dest / "abs-link") == "/usr/bin/env"
        # A symlinked directory is copied as a link, not descended into.
        assert os.readlink(dest / "dir-link") == "sub"
        assert not (dest / "fifo").exists()

    def test_large_file_and_read_only_directory(self, tmp_path):
        src = tmp_path / "src"
        (src / "ro").mkdir(parents=True)
        data = os.urandom(3 * (1 << 20) + 17)
        (src / "ro" / "big.bin").write_bytes(data)
        os.chmod(src / "ro", 0o555)
        dest = tmp_path / "dest"
        try:
            _round_trip(src, dest, tmp_path)
            # The owner keeps write access, so the tree can be copied again.
            assert stat.S_IMODE(os.stat(dest / "ro").st_mode) == 0o755
            _round_trip(src, dest, tmp_path)
            assert (dest / "ro" / "big.bin").read_bytes() == data
        finally:
            os.chmod(src / "ro", 0o755)
            if (dest / "ro").exists():
                os.chmod(dest / "ro", 0o755)

    def test_masks_unsafe_mode_bits(self, tmp_path):
        archive = _archive(tmp_path, (_member("f"), b"x"), (_member("ro"), b"y"))
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
        members[0].mode, members[1].mode = 0o4777, 0o444
        _archive(tmp_path, (members[0], b"x"), (members[1], b"y"))

        tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert stat.S_IMODE(os.stat(tmp_path / "dest" / "f").st_mode) == 0o755
        assert stat.S_IMODE(os.stat(tmp_path / "dest" / "ro").st_mode) == 0o644

    @pytest.mark.parametrize("kernel_copy", ["unsupported", "absent"])
    def test_copies_without_kernel_copy(self, tmp_path, kernel_copy):
        src = tmp_path / "src"
        src.mkdir()
        data = os.urandom(tree_archive._COPY_CHUNK * 2 + 5)
        (src / "f").write_bytes(data)

        def unsupported(*args):
            raise OSError(errno.EXDEV, "cross-device")

        replacement = unsupported if kernel_copy == "unsupported" else None
        with patch.object(tree_archive, "_KERNEL_COPY", replacement):
            _round_trip(src, tmp_path / "dest", tmp_path)

        assert (tmp_path / "dest" / "f").read_bytes() == data

    def test_truncated_archive_fails(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f").write_bytes(b"x" * 100_000)
        archive = str(tmp_path / "tree.tar")
        tree_archive.pack(str(src), archive)
        os.truncate(archive, 50_000)

        with pytest.raises((EOFError, tarfile.ReadError)):
            tree_archive.unpack(archive, str(tmp_path / "dest"))

    def test_unpack_merges_into_existing_directory(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "new").write_text("new")
        (src / "both").write_text("from src")
        (src / "was-link").write_text("file now")
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "kept").write_text("kept")
        (dest / "both").write_text("old")
        os.symlink("/etc/passwd", dest / "was-link")

        _round_trip(src, dest, tmp_path)

        assert not (dest / "was-link").is_symlink()
        assert {p.name: p.read_text() for p in dest.iterdir()} == {
            "new": "new",
            "both": "from src",
            "kept": "kept",
            "was-link": "file now",
        }

    @pytest.mark.skipif(shutil.which("tar") is None, reason="needs tar")
    def test_unpacks_tar_made_archives(self, tmp_path):
        src = tmp_path / "src"
        (src / "d").mkdir(parents=True)
        (src / "d" / "f").write_text("x")
        os.link(src / "d" / "f", src / "hard")
        archive = str(tmp_path / "gnu.tar")
        subprocess.run(["tar", "-cf", archive, "-C", str(src), "."], check=True)

        tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert (tmp_path / "dest" / "d" / "f").read_text() == "x"
        assert os.path.samefile(
            tmp_path / "dest" / "d" / "f", tmp_path / "dest" / "hard"
        )


class TestPackFormat:
    @staticmethod
    def _tree(src):
        long_dir = src / ("d" * 120)
        long_dir.mkdir(parents=True)
        (long_dir / ("f" * 120)).write_text("long name")
        (src / "naïve-ü.txt").write_text("utf-8 name")
        (src / "big.bin").write_bytes(os.urandom(tree_archive._SMALL_FILE + 1))
        os.symlink("t" * 150, src / "long-link")
        (src / "old").write_text("1969")
        os.utime(src / "old", (-86400, -86400))
        (src / "empty").write_bytes(b"")

    @pytest.mark.skipif(
        not (_is_gnu("tar") and _is_gnu("diff")), reason="needs GNU tar and diff"
    )
    def test_gnu_tar_reads_packed_archives(self, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        self._tree(src)
        archive = str(tmp_path / "tree.tar")
        tree_archive.pack(str(src), archive)
        dest.mkdir()

        subprocess.run(["tar", "-xf", archive, "-C", str(dest)], check=True)

        subprocess.run(["diff", "-r", "--no-dereference", src, dest], check=True)
        assert os.stat(dest / "old").st_mtime == -86400
        assert os.path.getsize(archive) % tarfile.RECORDSIZE == 0

    def test_headers_match_tarfile(self, tmp_path):
        src = tmp_path / "src"
        self._tree(src)
        archive = str(tmp_path / "tree.tar")
        tree_archive.pack(str(src), archive)

        with tarfile.open(archive) as tar:
            members = {m.name: m for m in tar}
        for path in src.rglob("*"):
            name = str(path.relative_to(src))
            st = os.lstat(path)
            member = members.pop(name)
            assert member.mode == stat.S_IMODE(st.st_mode)
            assert member.mtime == int(st.st_mtime)
            assert member.uid == member.gid == 0
            if path.is_symlink():
                assert member.issym() and member.linkname == os.readlink(path)
            elif path.is_file():
                assert member.isreg() and member.size == st.st_size
        assert list(members) == ["."]


class TestPackRoot:
    def test_root_may_be_a_symlink_to_a_directory(self, tmp_path):
        (tmp_path / "real" / "sub").mkdir(parents=True)
        (tmp_path / "real" / "sub" / "f").write_text("x")
        os.symlink("real", tmp_path / "link")

        _round_trip(tmp_path / "link", tmp_path / "dest", tmp_path)

        assert (tmp_path / "dest" / "sub" / "f").read_text() == "x"


class TestPackErrors:
    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
    def test_unreadable_subdirectory_fails_the_pack(self, tmp_path):
        src = tmp_path / "src"
        (src / "locked").mkdir(parents=True)
        (src / "locked" / "f").write_text("x")
        os.chmod(src / "locked", 0o300)
        try:
            with pytest.raises(PermissionError):
                tree_archive.pack(str(src), str(tmp_path / "tree.tar"))
        finally:
            os.chmod(src / "locked", 0o755)


class TestPackSafety:
    def test_does_not_read_through_symlinked_directories(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret").write_text("secret")
        src = tmp_path / "src"
        src.mkdir()
        os.symlink(outside, src / "escape")
        archive = str(tmp_path / "tree.tar")

        tree_archive.pack(str(src), archive)

        with tarfile.open(archive) as tar:
            assert sorted(tar.getnames()) == [".", "escape"]


class TestUnpackSafety:
    @pytest.mark.parametrize("name", ["../escaped", "a/../../escaped"])
    def test_refuses_paths_outside_destination(self, tmp_path, name):
        archive = _archive(tmp_path, (_member(name), b"x"))

        with pytest.raises(ValueError, match="leaves the destination"):
            tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert not (tmp_path / "escaped").exists()

    def test_keeps_absolute_names_inside_destination(self, tmp_path):
        absolute = str(tmp_path / "escaped")
        archive = _archive(tmp_path, (_member(absolute), b"x"))
        dest = tmp_path / "dest"

        tree_archive.unpack(archive, str(dest))

        assert not os.path.exists(absolute)
        assert (dest / absolute.lstrip("/")).read_text() == "x"

    def test_refuses_writing_through_extracted_symlink(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        archive = _archive(
            tmp_path,
            (_member("link", tarfile.SYMTYPE, str(outside)), None),
            (_member("link/planted"), b"x"),
        )

        with pytest.raises(OSError):
            tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert not (outside / "planted").exists()

    def test_refuses_writing_through_existing_symlink(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()
        os.symlink(outside, dest / "sub")
        archive = _archive(tmp_path, (_member("sub/planted"), b"x"))

        with pytest.raises(OSError):
            tree_archive.unpack(archive, str(dest))

        assert not (outside / "planted").exists()

    def test_does_not_overwrite_symlink_target(self, tmp_path):
        victim = tmp_path / "victim"
        victim.write_text("original")
        archive = _archive(
            tmp_path,
            (_member("f", tarfile.SYMTYPE, str(victim)), None),
            (_member("f"), b"overwritten"),
        )

        tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert victim.read_text() == "original"
        assert (tmp_path / "dest" / "f").read_text() == "overwritten"

    def test_hard_links_stay_inside_destination(self, tmp_path):
        (tmp_path / "secret").write_text("secret")
        for linkname in ("../secret", str(tmp_path / "secret")):
            archive = _archive(
                tmp_path, (_member("h", tarfile.LNKTYPE, linkname), None)
            )
            with pytest.raises((ValueError, FileNotFoundError)):
                tree_archive.unpack(archive, str(tmp_path / "dest"))
            assert not (tmp_path / "dest" / "h").exists()

    def test_skips_devices_and_fifos(self, tmp_path):
        archive = _archive(
            tmp_path,
            (_member("dev", tarfile.CHRTYPE), None),
            (_member("fifo", tarfile.FIFOTYPE), None),
            (_member("ok"), b"x"),
        )

        tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert sorted(os.listdir(tmp_path / "dest")) == ["ok"]

    def test_unpack_into_existing_file_fails(self, tmp_path):
        (tmp_path / "file").write_text("x")
        archive = _archive(tmp_path, (_member("f"), b"x"))

        with pytest.raises(FileExistsError):
            tree_archive.unpack(archive, str(tmp_path / "file"))

    def test_opens_only_a_bounded_number_of_directories(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        for i in range(300):
            (src / f"d{i}" / "inner").mkdir(parents=True)
            (src / f"d{i}" / "inner" / "f").write_text("x")
        archive = str(tmp_path / "tree.tar")
        tree_archive.pack(str(src), archive)
        # Count only tree_archive's own descriptors, not other threads'.
        held, peak = set(), [0]

        class CountingOs:
            def __getattr__(self, name):
                return getattr(os, name)

            def open(self, *args, **kwargs):
                fd = os.open(*args, **kwargs)
                held.add(fd)
                peak[0] = max(peak[0], len(held))
                return fd

            def close(self, fd):
                held.discard(fd)
                os.close(fd)

        monkeypatch.setattr(tree_archive, "os", CountingOs())
        tree_archive.unpack(archive, str(tmp_path / "dest"))

        assert not held  # Nothing leaked.
        assert peak[0] <= 5  # Root, the current chain, and one file.
        assert len(os.listdir(tmp_path / "dest")) == 300
