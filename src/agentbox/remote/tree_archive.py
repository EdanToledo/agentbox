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

"""Directory trees as tar archives, so a directory can be copied as one file.

:func:`pack` and :func:`unpack` work relative to directory descriptors and never
follow a symlink inside the tree, so neither can be steered outside it, even by
a tree that changes while it runs (for example, one a sandbox can write).
:func:`unpack` accepts archives from anywhere, including ``tar`` run inside a
container.

Both keep directories, regular files (with mode and mtime), and symlinks (as
links, whatever they point to); :func:`unpack` also accepts hard links within
the tree. Other file types are skipped and owners are never recorded or set.

:func:`unpack` applies modes without setuid, setgid, sticky, or group and other
write bits (an archive from a sandbox must not leave world-writable or setuid
files), and keeps every entry writable by its owner, so a tree can always be
copied over again.
"""

from __future__ import annotations

import errno
import os
import stat
import struct
import sys
import tarfile
from collections.abc import Callable
from typing import Any, TypeVar


_T = TypeVar("_T")

_COPY_CHUNK = 1 << 20
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK  # FIFOs never block
_MODE_MASK = 0o755  # Drops setuid, setgid, sticky, and group/other write.
_FILE_OWNER_BITS = 0o600
_DIR_OWNER_BITS = 0o700

# Kernel copy failures that mean "use pread/write instead".
_NO_KERNEL_COPY = {
    errno.EXDEV,
    errno.ENOSYS,
    errno.EINVAL,
    errno.EOPNOTSUPP,
    errno.ENOTSOCK,
}


_KERNEL_COPY: Callable[[int, int, int, int], int] | None = None

# Kernel file-to-file copy is Linux-only (elsewhere os.copy_file_range is
# missing and sendfile needs a socket); other platforms use pread/write.
if sys.platform == "linux":

    def _copy_file_range(src: int, dst: int, offset: int, count: int) -> int:
        return os.copy_file_range(src, dst, count, offset)

    def _sendfile(src: int, dst: int, offset: int, count: int) -> int:
        return os.sendfile(dst, src, offset, count)

    # Some Python builds lack copy_file_range; sendfile also copies
    # file-to-file in the kernel.
    _KERNEL_COPY = _copy_file_range if hasattr(os, "copy_file_range") else _sendfile


def _copy_range(src: int, offset: int, dst: int, count: int) -> None:
    """Copy *count* bytes at *offset* in *src* to the position of *dst*."""
    kernel_copy = _KERNEL_COPY
    while count:
        n = None
        if kernel_copy is not None:
            try:
                n = kernel_copy(src, dst, offset, min(count, 1 << 30))
            except OSError as exc:
                if exc.errno not in _NO_KERNEL_COPY:
                    raise
                kernel_copy = None
        if n is None:
            n = os.write(dst, os.pread(src, min(count, _COPY_CHUNK), offset))
        if n == 0:
            raise EOFError(f"input ended {count} bytes early")
        offset, count = offset + n, count - n


# A POSIX ustar header; pack() writes this directly in the common case and lets
# tarfile add pax records for anything it cannot hold.
_USTAR = struct.Struct("100s8s8s8s12s12s8sc100s6s2s32s32s8s8s155s12x")
_USTAR_MAX = 8**11  # Largest size or mtime in a 12-byte octal field.
_ZERO_ID = b"0000000\0"
_SMALL_FILE = 64 * 1024  # Files up to this size are buffered with headers.


def _header(
    name: str, kind: bytes, st: os.stat_result, size: int = 0, linkname: str = ""
) -> bytes:
    """Return the tar header block(s) for one entry."""
    raw_name = name.encode("utf-8", "surrogateescape")
    raw_link = linkname.encode("utf-8", "surrogateescape")
    mode, mtime = stat.S_IMODE(st.st_mode), int(st.st_mtime)
    if (
        len(raw_name) > 100
        or len(raw_link) > 100
        or not (size < _USTAR_MAX and 0 <= mtime < _USTAR_MAX)
    ):
        info = tarfile.TarInfo(name)
        info.type, info.mode, info.mtime = kind, mode, mtime
        info.size, info.linkname = size, linkname
        return info.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape")
    header = bytearray(
        _USTAR.pack(
            raw_name,
            b"%07o\0" % mode,
            _ZERO_ID,
            _ZERO_ID,
            b"%011o\0" % size,
            b"%011o\0" % mtime,
            b" " * 8,  # The checksum is computed with this field as spaces.
            kind,
            raw_link,
            b"ustar\0",
            b"00",
            b"",
            b"",
            _ZERO_ID,
            _ZERO_ID,
            b"",
        )
    )
    header[148:155] = b"%06o\0" % sum(header)
    return bytes(header)


class _ArchiveWriter:
    """Appends tar blocks to a descriptor, batching small writes."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = bytearray()
        self._offset = 0

    def write(self, data: bytes) -> None:
        self._buffer += data
        self._offset += len(data)
        if len(self._buffer) >= _COPY_CHUNK:
            self.flush()

    def flush(self) -> None:
        written = 0
        with memoryview(self._buffer) as view:
            while written < len(view):
                with view[written:] as rest:
                    written += os.write(self._fd, rest)
        self._buffer.clear()

    def write_file(self, fd: int, size: int) -> None:
        """Append *size* bytes of file *fd* and the padding after them."""
        if size <= _SMALL_FILE:
            data = os.pread(fd, size, 0)
            while len(data) < size:
                more = os.pread(fd, size - len(data), len(data))
                if not more:
                    raise EOFError(f"{size - len(data)} bytes missing")
                data += more
            self.write(data)
        else:
            self.flush()
            _copy_range(fd, 0, self._fd, size)
            self._offset += size
        self.write(bytes(-size % tarfile.BLOCKSIZE))

    def close(self) -> None:
        """End the archive with two zero blocks, padded to a full record."""
        self.write(bytes(2 * tarfile.BLOCKSIZE))
        self.write(bytes(-self._offset % tarfile.RECORDSIZE))
        self.flush()


def pack(root: str, archive: str) -> None:
    """Write the tree at *root* to the tar file *archive*.

    *root* itself may be a symlink to a directory; links inside it are not
    followed. A directory that cannot be read raises, so the archive is never
    silently incomplete.
    """
    # fwalk never follows a symlinked top, so open the root here and walk
    # from its descriptor.
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with open(archive, "wb", buffering=0) as out:
            _pack_from(root_fd, _ArchiveWriter(out.fileno()))
    finally:
        os.close(root_fd)


def _raise(error: OSError) -> None:
    raise error


def _pack_from(root_fd: int, writer: _ArchiveWriter) -> None:
    for top, dirnames, filenames, top_fd in os.fwalk(
        ".", dir_fd=root_fd, onerror=_raise
    ):
        prefix = os.path.relpath(top, ".")
        writer.write(_header(prefix, tarfile.DIRTYPE, os.fstat(top_fd)))
        # fwalk lists symlinks to directories among dirnames but does not
        # descend into them, so links are picked out of both lists here.
        for entry in dirnames + filenames:
            name = os.path.normpath(os.path.join(prefix, entry))
            st = os.stat(entry, dir_fd=top_fd, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                link = os.readlink(entry, dir_fd=top_fd)
                writer.write(_header(name, tarfile.SYMTYPE, st, linkname=link))
            elif stat.S_ISREG(st.st_mode):
                fd = os.open(entry, _READ_FLAGS, dir_fd=top_fd)
                try:
                    st = os.fstat(fd)
                    if stat.S_ISREG(st.st_mode):
                        size = st.st_size
                        writer.write(_header(name, tarfile.REGTYPE, st, size))
                        writer.write_file(fd, size)
                finally:
                    os.close(fd)
    writer.close()


def _parts(name: str) -> tuple[str, ...]:
    """Split archive path *name* into components below the destination."""
    parts = tuple(p for p in name.split("/") if p not in ("", "."))
    if ".." in parts:
        raise ValueError(f"archive entry leaves the destination: {name!r}")
    return parts


class _Dirs:
    """Directories under a root, opened component by component without
    following symlinks and created as needed.

    Archives list entries directory by directory, so only the chain of
    directories above the last lookup is kept open.
    """

    def __init__(self, root_fd: int) -> None:
        self._chain: list[tuple[tuple[str, ...], int]] = [((), root_fd)]

    def get(self, parts: tuple[str, ...]) -> int:
        keep = 1
        while (
            keep < len(self._chain)
            and keep <= len(parts)
            and self._chain[keep][0] == parts[:keep]
        ):
            keep += 1
        self._close_from(keep)
        while len(self._chain) <= len(parts):
            depth = len(self._chain)
            parent = self._chain[-1][1]
            name = parts[depth - 1]
            try:
                fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                try:
                    os.mkdir(name, 0o755, dir_fd=parent)
                except FileExistsError:
                    pass
                fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
            self._chain.append((parts[:depth], fd))
        return self._chain[-1][1]

    def _close_from(self, keep: int) -> None:
        for _, fd in self._chain[keep:]:
            os.close(fd)
        del self._chain[keep:]

    def close(self) -> None:
        """Close every directory, including the root."""
        self._close_from(0)


def _replacing(
    name: str, parent: int, create: Callable[..., _T], *args: Any, **kwargs: Any
) -> _T:
    """Call *create*; if *name* already exists, remove it and call it again."""
    try:
        return create(*args, **kwargs)
    except FileExistsError:
        os.unlink(name, dir_fd=parent)  # A directory raises instead.
        return create(*args, **kwargs)


def _copy_data(
    tar: tarfile.TarFile, member: tarfile.TarInfo, src: int, dst: int
) -> None:
    """Copy *member*'s data from archive descriptor *src* to file *dst*."""
    if not member.issparse():
        _copy_range(src, member.offset_data, dst, member.size)
        return
    data = tar.extractfile(member)
    assert data is not None
    with data:
        while block := data.read(_COPY_CHUNK):
            os.write(dst, block)


def unpack(archive: str, dest: str) -> None:
    """Extract tar file *archive* into directory *dest*, creating it if needed.

    Entries already under *dest* are replaced or kept, as with ``tar -x``.
    Entries never land outside *dest*: ``..`` components are refused, leading
    ``/`` is dropped, and an existing symlink in place of a directory fails
    the extraction instead of being followed.
    """
    os.makedirs(dest, exist_ok=True)
    dirs = _Dirs(os.open(dest, os.O_RDONLY | os.O_DIRECTORY))
    # Directory modes and mtimes go on last, deepest first, so creating
    # entries does not disturb them.
    dir_meta: list[tuple[tuple[str, ...], int, int]] = []
    try:
        with open(archive, "rb") as raw, tarfile.open(fileobj=raw, mode="r:") as tar:
            for member in tar:
                parts = _parts(member.name)
                if member.isdir():
                    dirs.get(parts)
                    dir_meta.append((parts, member.mode, int(member.mtime)))
                    continue
                if not parts:
                    raise ValueError(f"archive entry is not a file: {member.name!r}")
                parent, name = dirs.get(parts[:-1]), parts[-1]
                if member.isreg():
                    fd = _replacing(
                        name,
                        parent,
                        os.open,
                        name,
                        _NEW_FILE_FLAGS,
                        0o600,
                        dir_fd=parent,
                    )
                    try:
                        _copy_data(tar, member, raw.fileno(), fd)
                        os.fchmod(fd, member.mode & _MODE_MASK | _FILE_OWNER_BITS)
                        os.utime(fd, (member.mtime, member.mtime))
                    finally:
                        os.close(fd)
                elif member.issym():
                    _replacing(
                        name, parent, os.symlink, member.linkname, name, dir_fd=parent
                    )
                elif member.islnk():
                    _link(dirs, _parts(member.linkname), parent, name)
        dir_meta.sort(key=lambda meta: len(meta[0]), reverse=True)
        for parts, mode, mtime in dir_meta:
            fd = dirs.get(parts)
            os.fchmod(fd, mode & _MODE_MASK | _DIR_OWNER_BITS)
            os.utime(fd, (mtime, mtime))
    finally:
        dirs.close()


def _link(dirs: _Dirs, target: tuple[str, ...], parent: int, name: str) -> None:
    """Hard-link *name* in *parent* to the earlier entry *target*."""
    if not target:
        raise ValueError("hard link to the destination itself")
    parent = os.dup(parent)  # Looking up the target may close the original.
    try:
        source_dir = dirs.get(target[:-1])
        _replacing(
            name,
            parent,
            os.link,
            target[-1],
            name,
            src_dir_fd=source_dir,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
    finally:
        os.close(parent)
