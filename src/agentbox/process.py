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

"""Process lifecycle utilities."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess


logger = logging.getLogger(__name__)

__all__ = [
    "terminate_process",
    "reap_monarch_descendants",
    "ensure_monarch_reaper",
    "track_monarch_descendants",
]

LOCAL_WORKER_ADDRESS_ENV = "_AGENTBOX_LOCAL_WORKER_ADDRESS"
LOCAL_WORKER_PROCESS_MARKER = "--agentbox-local-worker"

# Command-line markers identifying Monarch worker / host (allocator) subprocesses.
# ``LOCAL_WORKER_PROCESS_MARKER`` identifies local pool workers.
# ``bootstrap_main`` backs ``this_host()`` / ``spawn_procs()``
# and the client-side proc meshes created when a machine is acquired. All appear
# in the process's command line.
_MONARCH_PROC_MARKERS = (
    LOCAL_WORKER_PROCESS_MARKER,
    "monarch._src.actor.bootstrap_main",
    "run_worker_loop_forever",
)

# Monarch job sidecars deliberately outlive the client process so cached jobs
# can refresh existing remote mounts on the next invocation.  Their actor
# subprocesses match the markers above, but belong to the sidecar lifecycle
# rather than this client's lifecycle.
_MONARCH_PERSISTENT_ROOT_MARKERS = ("monarch._src.job._job_sidecar_worker",)

_reaper_registered = False

# PIDs of Monarch helper processes observed as descendants of this process.
# Recorded while still attributable so they can be reaped at exit even after an
# intermediate Monarch process exits and orphans them (reparenting them to init,
# out of our descendant tree).
_tracked_monarch_pids: set[int] = set()


def terminate_process(
    proc: subprocess.Popen,
    soft_timeout: float = 10,
    hard_timeout: float = 5,
) -> None:
    """Gracefully terminate a subprocess and reap it, escalating to kill if needed.

    Sends ``SIGTERM`` and waits up to *soft_timeout* seconds.  If the
    process does not exit in time, sends ``SIGKILL`` and waits up to
    *hard_timeout* seconds.  The final ``wait()`` reaps the child so it does
    not linger as a zombie.  Silently ignores :class:`OSError` from
    ``terminate``/``kill`` (process already dead).

    Never raises: if the process is still unkillable after ``SIGKILL`` and
    *hard_timeout* (e.g. stuck in uninterruptible sleep), the timeout is logged
    rather than propagated, so callers using this in cleanup/shutdown loops are
    not aborted partway through.
    """
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=soft_timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Process pid=%s did not exit %.0fs after SIGKILL; leaving it unreaped",
            getattr(proc, "pid", "?"),
            hard_timeout,
        )


def _descendant_pids(
    root_pid: int,
    preserved_root_markers: tuple[str, ...] = (),
) -> tuple[list[int], dict[int, str]]:
    """Return (descendant_pids, pid→command) for the subtree rooted at *root_pid*.

    Subtrees rooted at commands matching *preserved_root_markers* are omitted.
    Parses ``ps`` once.  Best-effort: returns empty on any failure so callers in
    shutdown/atexit paths never raise.
    """
    try:
        out = subprocess.run(
            ["ps", "-ax", "-o", "pid,ppid,command"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return [], {}

    children: dict[int, list[int]] = {}
    cmd: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue  # skip the header row and any malformed lines
        pid, ppid = int(parts[0]), int(parts[1])
        children.setdefault(ppid, []).append(pid)
        cmd[pid] = parts[2]

    descendants: list[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if any(marker in cmd.get(pid, "") for marker in preserved_root_markers):
            continue
        descendants.append(pid)
        stack.extend(children.get(pid, []))
    return descendants, cmd


def _proc_command(pid: int) -> str | None:
    """Return the command line of *pid*, or ``None`` if it is gone/unreadable."""
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def track_monarch_descendants(markers: tuple[str, ...] = _MONARCH_PROC_MARKERS) -> None:
    """Record Monarch helper processes currently descended from this process.

    Call this while the processes are still attributable (e.g. right after
    ``create_local`` / ``acquire_machine``).  Recording the PIDs lets
    :func:`reap_monarch_descendants` clean them up at exit even after an
    intermediate Monarch process exits first and reparents them to init —
    which would otherwise drop them out of our descendant tree.  Best-effort.
    """
    descendants, cmd = _descendant_pids(os.getpid(), _MONARCH_PERSISTENT_ROOT_MARKERS)
    for pid in descendants:
        if any(marker in cmd.get(pid, "") for marker in markers):
            _tracked_monarch_pids.add(pid)


def reap_monarch_descendants(markers: tuple[str, ...] = _MONARCH_PROC_MARKERS) -> int:
    """SIGKILL leftover Monarch worker/host subprocesses spawned by this process.

    Monarch's local runtime spawns helper processes (``bootstrap_main`` backing
    ``this_host()``/proc meshes, and ``run_worker_loop_forever`` pool workers).
    Its own teardown — ``ProcMesh.stop()`` / ``HostMesh.stop()`` and the
    ``atexit: shutdown_context().get(timeout=1.0)`` handler — does not reliably
    reap the *operating-system processes*, so they linger after the owning
    Python process exits and accumulate across runs.

    Targets two disjoint sets, so it works regardless of teardown timing:

    * Monarch processes that are *still* descendants of this process.
    * PIDs recorded by :func:`track_monarch_descendants` that are still alive and
      *still look like Monarch processes* — catching helpers that were orphaned
      (reparented to init) when an intermediate Monarch process exited first.
      The command-line recheck guards against PID reuse.

    Processes owned by a persistent Monarch job sidecar are excluded.  The
    sidecar intentionally keeps remote mounts alive for cached-job refreshes and
    shuts down its own process tree when the job is killed.

    Either way it only kills processes this run actually spawned (current
    descendants) or recorded as ours; it never touches another live process's
    workers.  Best-effort: never raises.  Returns the number of processes signalled.
    """
    targets: set[int] = set()

    descendants, cmd = _descendant_pids(os.getpid(), _MONARCH_PERSISTENT_ROOT_MARKERS)
    for pid in descendants:
        if any(marker in cmd.get(pid, "") for marker in markers):
            targets.add(pid)

    for pid in list(_tracked_monarch_pids):
        command = _proc_command(pid)
        if command and any(marker in command for marker in markers):
            targets.add(pid)

    killed = 0
    for pid in targets:
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError:
            pass  # already gone
    if killed:
        logger.debug("Reaped %d leftover Monarch subprocess(es)", killed)
    return killed


def ensure_monarch_reaper() -> None:
    """Register :func:`reap_monarch_descendants` to run once at interpreter exit.

    Idempotent — safe to call from every ``AgentBoxManager`` construction.
    """
    global _reaper_registered
    if _reaper_registered:
        return
    _reaper_registered = True
    atexit.register(reap_monarch_descendants)
