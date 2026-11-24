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

"""Process-environment helpers.

Temporarily removes Slurm variables that would leak the submitting job's
allocation into child ``sbatch`` submissions, keeps macOS temp paths short
enough for Monarch's Unix-domain sockets, and keeps glibc from hoarding the
large buffers that file transfers allocate.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from collections.abc import Generator


logger = logging.getLogger(__name__)

__all__ = [
    "scrub_slurm_env",
    "restore_slurm_env",
    "scrubbed_slurm_env",
    "ensure_short_tmpdir",
    "bound_native_heap",
]

# glibc raises its mmap threshold (up to 32 MiB) whenever a large mmapped block
# is freed, after which multi-MiB message buffers are carved from per-thread
# heap arenas that are rarely returned to the OS.  A client streaming file
# chunks across many threads then grows without bound on long runs with many
# workers.  Pinning the threshold keeps such buffers mmapped, so
# they are unmapped when freed.
_MMAP_THRESHOLD = 1 << 20
_M_MMAP_THRESHOLD = -3  # mallopt parameter number from <malloc.h>

# macOS caps Unix-domain-socket paths at 104 bytes (``SUN_LEN``).  Monarch builds
# its host<->worker socket paths under ``tempfile.gettempdir()``; leaving enough
# headroom for the subdir + socket name it appends means the tmp dir itself must
# stay well under that limit.
_SOCKET_SAFE_TMPDIR_LEN = 40

# Slurm vars to keep when scrubbing — needed by the cluster task epilog
# (``/etc/slurm/other-scripts/task_epilog.sh``) which uses ``set -u``.
SLURM_KEEP: set[str] = {"SLURM_CLUSTER_NAME"}


def scrub_slurm_env() -> dict[str, str]:
    """Remove ``SLURM_*`` env vars except those in :data:`SLURM_KEEP`.

    Returns a dict of the removed variables so they can be restored
    with :func:`restore_slurm_env`.
    """
    saved: dict[str, str] = {}
    for key in list(os.environ):
        if key.startswith("SLURM_") and key not in SLURM_KEEP:
            saved[key] = os.environ.pop(key)
    if saved:
        logger.info("Scrubbed %d Slurm env vars: %s", len(saved), list(saved.keys()))
    return saved


def restore_slurm_env(saved: dict[str, str]) -> None:
    """Restore previously scrubbed ``SLURM_*`` env vars."""
    if saved:
        logger.debug("Restoring %d Slurm env vars", len(saved))
    os.environ.update(saved)


def ensure_short_tmpdir() -> str | None:
    """On macOS, point ``tempfile``/``TMPDIR`` at a short, ``$HOME``-rooted dir.

    Solves two macOS-only problems at once:

    1. **``SUN_LEN``** — Monarch builds its host<->worker Unix-domain-socket
       paths under ``tempfile.gettempdir()``.  The default macOS ``$TMPDIR``
       (``/var/folders/.../T/...``, ~75 chars) overflows the 104-byte socket
       path limit once Monarch appends its own subdir + socket name, raising
       ``RuntimeError: path must be shorter than SUN_LEN``.
    2. **Lima visibility** — Apptainer runs inside a Lima VM that mounts
       ``$HOME`` but not ``/tmp`` or ``/var/folders``, so temp dirs used as
       bind-mount sources must live under ``$HOME`` to be visible in the guest.

    A single short, ``$HOME``-rooted dir (``~/.agentbox/tmp`` by default)
    satisfies both.  No-op on non-darwin.  Honors an explicit
    ``AGENTBOX_TMPDIR`` override and leaves an already-short, non-default
    ``TMPDIR`` untouched.  Idempotent.

    Returns the directory that was set, or ``None`` if nothing changed.
    """
    if sys.platform != "darwin":
        return None

    home = os.path.expanduser("~")
    override = os.environ.get("AGENTBOX_TMPDIR")
    current = os.environ.get("TMPDIR") or tempfile.gettempdir()
    if override:
        target = override
    elif (
        current and len(current) <= _SOCKET_SAFE_TMPDIR_LEN and current.startswith(home)
    ):
        # Already short *and* under $HOME (socket-safe and Lima-visible) — keep it.
        return None
    else:
        target = os.path.join(home, ".agentbox", "tmp")

    # The chosen target (an AGENTBOX_TMPDIR override, or ~/.agentbox/tmp under a
    # long $HOME) can itself exceed the safe length.  We still set it — it is the
    # best available option — but warn so the eventual SUN_LEN socket-path error
    # is diagnosable rather than opaque.
    if len(target) > _SOCKET_SAFE_TMPDIR_LEN:
        logger.warning(
            "TMPDIR %r is %d bytes (> %d); Monarch host<->worker socket paths may overflow "
            "the macOS SUN_LEN limit (104B). Set AGENTBOX_TMPDIR to a shorter path.",
            target,
            len(target),
            _SOCKET_SAFE_TMPDIR_LEN,
        )

    os.makedirs(target, exist_ok=True)
    os.environ["TMPDIR"] = target
    tempfile.tempdir = target  # override gettempdir()'s cached value
    logger.info(
        "macOS: set TMPDIR=%s (short, $HOME-rooted) for Monarch sockets + Lima binds",
        target,
    )
    return target


def bound_native_heap() -> bool:
    """Pin glibc's mmap threshold so freed transfer buffers return to the OS.

    No-op (returns ``False``) off glibc or when the user already set
    ``MALLOC_MMAP_THRESHOLD_``.
    """
    if not sys.platform.startswith("linux") or "MALLOC_MMAP_THRESHOLD_" in os.environ:
        return False
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        return bool(libc.mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD))
    except (OSError, AttributeError):
        return False


@contextlib.contextmanager
def scrubbed_slurm_env() -> Generator[dict[str, str], None, None]:
    """Context manager that scrubs ``SLURM_*`` env vars and restores them on exit."""
    saved = scrub_slurm_env()
    try:
        yield saved
    finally:
        restore_slurm_env(saved)
