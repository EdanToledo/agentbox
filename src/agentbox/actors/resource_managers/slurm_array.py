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

"""Slurm job array management for chunked, elastic node allocation.

Provides :class:`SlurmArrayChunk`, a standalone class that manages a single
Slurm job array submission.  Each array task is an independent single-node
job, enabling per-node preemption resilience and elastic scaling.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field

from agentbox.env import scrubbed_slurm_env


logger = logging.getLogger(__name__)

__all__ = ["ArrayTaskInfo", "SlurmAllocation", "SlurmArrayChunk"]

# Wall-clock cap for Slurm CLI calls (sbatch/squeue/scancel).  A wedged Slurm
# controller must not block the ResourceManager actor endpoint indefinitely.
_SLURM_CMD_TIMEOUT = 60.0

# Terminal Slurm job states (locally defined to avoid coupling to Monarch internals).
_SLURM_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)


@dataclass(frozen=True)
class SlurmAllocation:
    """Typed specification for Slurm resource requirements.

    Replaces untyped ``**slurm_kwargs``.  Immutable after construction
    so the spec can be safely shared and introspected.
    """

    cpus_per_task: int = 4
    gpus_per_node: int = 0
    partition: str | None = None
    time_limit: str = "04:00:00"
    qos: str | None = None
    account: str | None = None
    exclusive: bool = False
    mem: str | None = None
    extra_slurm_args: tuple[str, ...] = ()
    # Trusted shell fragments inserted after ``srun`` for the worker step.
    extra_srun_args: tuple[str, ...] = ()
    job_name: str = "agentbox-worker"
    worker_logs_base_path: str | None = None
    # Reserve the entire node and let the worker step use all of its CPUs and
    # GPUs.  Kept last so existing positional construction remains compatible.
    whole_node: bool = False


@dataclass
class ArrayTaskInfo:
    """Status snapshot for a single Slurm array task.

    Attributes:
        array_job_id: The parent array job ID (e.g. ``"12345"``).
        array_task_id: The task index within the array (e.g. ``3``).
        status: Slurm job state string (``"PENDING"``, ``"RUNNING"``, or a terminal state).
        hostname: Hostname of the allocated node, set once the task is ``RUNNING``.
    """

    array_job_id: str
    array_task_id: int
    status: str
    hostname: str | None = None


@dataclass
class SlurmArrayChunk:
    """Manages a single Slurm job array submission.

    Each array task requests ``--nodes=1``, and the array is throttled
    via ``--array=0-(chunk_size-1)%max_parallel``.

    Args:
        chunk_size: Number of array tasks in this chunk.
        allocation: Slurm resource specification.
        max_parallel: Maximum concurrently running array tasks.
        python_exe: Python executable path for the sbatch script.
    """

    chunk_size: int
    allocation: SlurmAllocation = field(default_factory=SlurmAllocation)
    max_parallel: int = 128
    python_exe: str = "python"
    port_server_addr: str = field(kw_only=True, repr=False)

    # Populated after submit().
    array_job_id: str | None = field(default=None, init=False, repr=False)

    # Internal tracking.
    _known_running: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _known_dead: set[int] = field(default_factory=set, init=False, repr=False)
    _last_poll: list[ArrayTaskInfo] = field(
        default_factory=list, init=False, repr=False
    )
    _last_poll_ok: bool = field(default=True, init=False, repr=False)

    def submit(self) -> str:  # noqa: C901
        """Build and submit the sbatch job array script.

        Returns the array job ID assigned by Slurm.
        """
        array_spec = f"0-{self.chunk_size - 1}"
        if self.max_parallel < self.chunk_size:
            array_spec += f"%{self.max_parallel}"

        output_path = "slurm-%A_%a.out"
        error_path = "slurm-%A_%a.err"
        if self.allocation.worker_logs_base_path:
            output_path = os.path.join(
                self.allocation.worker_logs_base_path, "slurm-%A_%a.out"
            )
            error_path = os.path.join(
                self.allocation.worker_logs_base_path, "slurm-%A_%a.err"
            )

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.allocation.job_name}",
            f"#SBATCH --array={array_spec}",
            "#SBATCH --nodes=1",
            f"#SBATCH --cpus-per-task={self.allocation.cpus_per_task}",
            f"#SBATCH --time={self.allocation.time_limit}",
            f"#SBATCH --output={output_path}",
            f"#SBATCH --error={error_path}",
        ]
        if self.allocation.partition:
            lines.append(f"#SBATCH --partition={self.allocation.partition}")
        if self.allocation.gpus_per_node:
            lines.append(f"#SBATCH --gpus-per-node={self.allocation.gpus_per_node}")
        if self.allocation.exclusive or self.allocation.whole_node:
            lines.append("#SBATCH --exclusive")
        if self.allocation.qos:
            lines.append(f"#SBATCH --qos={self.allocation.qos}")
        if self.allocation.account:
            lines.append(f"#SBATCH --account={self.allocation.account}")
        if self.allocation.mem:
            lines.append(f"#SBATCH --mem={self.allocation.mem}")
        for arg in self.allocation.extra_slurm_args:
            lines.append(f"#SBATCH {arg}")

        lines.append("")
        # Give each array task a unique Monarch log path so that workers sharing
        # a physical node do not clobber each other's /tmp/${USER}/monarch_log.log.
        # Monarch constructs the path as monarch_log_{suffix}.log, so the suffix
        # must NOT include a leading dot or separator.
        # Result: /tmp/${USER}/monarch_log_task_${SLURM_ARRAY_TASK_ID}.log
        lines.append('export MONARCH_LOG_SUFFIX="task_${SLURM_ARRAY_TASK_ID}"')
        extra_srun_args = self.allocation.extra_srun_args
        whole_node_srun_args: tuple[str, ...] = ()
        if self.allocation.whole_node:
            # An exclusive job owns the node, but its worker step would still be
            # pinned to cpus_per_task and would not receive GPU device nodes
            # unless it requests the allocation's GRES.  Build an array so the
            # optional GPU argument disappears cleanly on CPU-only nodes.
            lines.extend(
                [
                    (
                        '_agentbox_whole_node_srun_args=("--cpu-bind=none" '
                        '"--cpus-per-task=${SLURM_CPUS_ON_NODE}")'
                    ),
                    (
                        '_agentbox_whole_node_gpus="${SLURM_GPUS_ON_NODE:-'
                        "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')}\""
                    ),
                    'if [ "${_agentbox_whole_node_gpus:-0}" -gt 0 ] 2>/dev/null; then',
                    (
                        "    _agentbox_whole_node_srun_args+=("
                        '"--gpus-per-node=${_agentbox_whole_node_gpus}")'
                    ),
                    "fi",
                ]
            )
            whole_node_srun_args = ('"${_agentbox_whole_node_srun_args[@]}"',)
        srun = " ".join(("srun", *whole_node_srun_args, *extra_srun_args))
        # Workers bind an OS-assigned port and register it with this server.
        lines.append(f'export AGENTBOX_PORT_SERVER="{self.port_server_addr}"')
        lines.append(f"{srun} {self.python_exe} -m agentbox.worker_bootstrap")

        script = "\n".join(lines) + "\n"
        logger.debug("sbatch script:\n%s", textwrap.indent(script, "  "))

        with scrubbed_slurm_env():
            result = subprocess.run(
                ["sbatch", "--parsable"],
                input=script,
                capture_output=True,
                text=True,
                check=False,
                timeout=_SLURM_CMD_TIMEOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch --parsable failed (exit {result.returncode}).\n"
                f"stderr: {result.stderr.strip() or '<empty>'}\n"
                f"stdout: {result.stdout.strip() or '<empty>'}\n"
                f"script:\n{textwrap.indent(script, '  ')}"
            )
        # --parsable output is "job_id" (or "job_id;cluster" on federated setups).
        self.array_job_id = result.stdout.strip().split(";")[0]

        logger.info(
            "Submitted array job %s (chunk_size=%d, max_parallel=%d)",
            self.array_job_id,
            self.chunk_size,
            self.max_parallel,
        )
        return self.array_job_id

    def poll_tasks(self) -> list[ArrayTaskInfo]:
        """Query Slurm for the current state of all array tasks.

        Uses ``squeue --json`` for structured output.  Non-blocking (single
        query, no polling loop).

        Returns a list of :class:`ArrayTaskInfo` for every task in the array.
        """
        if self.array_job_id is None:
            raise RuntimeError("Cannot poll before submit()")
        return self._poll_tasks_impl()

    def _poll_tasks_impl(self) -> list[ArrayTaskInfo]:  # noqa: C901
        assert self.array_job_id is not None  # guarded by poll_tasks()
        tasks: list[ArrayTaskInfo] = []
        try:
            result = subprocess.run(
                ["squeue", "--job", self.array_job_id, "--json"],
                capture_output=True,
                text=True,
                timeout=_SLURM_CMD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "squeue timed out after %.0fs; treating as a failed poll",
                _SLURM_CMD_TIMEOUT,
            )
            self._last_poll_ok = False
            self._last_poll = tasks
            return tasks

        if result.returncode != 0:
            logger.warning(
                "squeue failed (rc=%d): %s", result.returncode, result.stderr.strip()
            )
            self._last_poll_ok = False
            self._last_poll = tasks
            return tasks

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("squeue returned invalid JSON: %s", result.stdout[:200])
            self._last_poll_ok = False
            self._last_poll = tasks
            return tasks

        for job_entry in data.get("jobs", []):
            task_id = job_entry.get("array_task_id")
            if task_id is None:
                continue
            # Newer Slurm versions return array_task_id as a dict
            # e.g. {"set": "...", "infinite": false, "number": 5}
            if isinstance(task_id, dict):
                task_id = task_id.get("number")
                if task_id is None:
                    continue

            state = job_entry.get("job_state", ["UNKNOWN"])
            if isinstance(state, list):
                state = state[0] if state else "UNKNOWN"

            hostname: str | None = None
            if state == "RUNNING":
                # Try to extract hostname from job_resources or nodes field.
                nodes = job_entry.get("nodes", "")
                if nodes:
                    hostname = nodes
                else:
                    resources = job_entry.get("job_resources", {})
                    allocation = resources.get("nodes", {}).get("allocation", [])
                    if allocation:
                        hostname = allocation[0].get("name")

            tasks.append(
                ArrayTaskInfo(
                    array_job_id=self.array_job_id,
                    array_task_id=int(task_id),
                    status=state,
                    hostname=hostname,
                )
            )

        self._last_poll_ok = True
        self._last_poll = tasks
        return tasks

    def get_newly_running(
        self, port_lookup: Callable[[str, int], tuple[str, int] | None]
    ) -> list[tuple[int, str, int]]:
        """Return array tasks that have become RUNNING since the last check.

        Calls :meth:`poll_tasks` internally and diffs against the internal
        ``_known_running`` set.

        Args:
            port_lookup: Callback ``(job_id, task_id) → (hostname, port)``
                returning the registered worker port. Tasks whose port is not
                yet registered (callback returns ``None``) are picked up on a
                later poll.

        Returns a list of ``(task_id, hostname, port)`` triples for newly running
        tasks.
        """
        tasks = self.poll_tasks()
        new_running: list[tuple[int, str, int]] = []
        for t in tasks:
            if (
                t.status == "RUNNING"
                and t.hostname
                and t.array_task_id not in self._known_running
            ):
                result = port_lookup(self.array_job_id or "", t.array_task_id)
                if result is None:
                    # Worker hasn't registered yet; defer to next poll.
                    continue
                _hostname, port = result
                self._known_running[t.array_task_id] = t.hostname
                new_running.append((t.array_task_id, t.hostname, port))
        return new_running

    def get_dead_tasks(self) -> list[tuple[int, str | None]]:
        """Return array tasks that have entered a terminal state since the last check.

        Returns a list of ``(task_id, hostname_or_none)`` pairs.
        """
        tasks = self.poll_tasks()
        newly_dead: list[tuple[int, str | None]] = []

        for t in tasks:
            if (
                t.status in _SLURM_TERMINAL_STATES
                and t.array_task_id not in self._known_dead
            ):
                self._known_dead.add(t.array_task_id)
                hostname = self._known_running.get(t.array_task_id, t.hostname)
                newly_dead.append((t.array_task_id, hostname))

        # Also detect tasks that disappeared from squeue (already cleaned up by Slurm).
        # Only apply this heuristic when the poll succeeded; a squeue failure
        # returning an empty list must not be mistaken for "all tasks finished".
        if self._last_poll_ok:
            current_ids = {t.array_task_id for t in tasks}
            for task_id in list(self._known_running):
                if task_id not in current_ids and task_id not in self._known_dead:
                    self._known_dead.add(task_id)
                    newly_dead.append((task_id, self._known_running.get(task_id)))

        return newly_dead

    def cancel_task(self, task_id: int) -> None:
        """Cancel a specific array task."""
        if self.array_job_id is None:
            raise RuntimeError("Cannot cancel before submit()")

        subprocess.run(
            ["scancel", f"{self.array_job_id}_{task_id}"],
            check=True,
            timeout=_SLURM_CMD_TIMEOUT,
        )
        logger.info("Cancelled array task %s_%d", self.array_job_id, task_id)

    def cancel_all(self) -> None:
        """Cancel the entire array job."""
        if self.array_job_id is None:
            return

        self._cancel_all_impl()

    def _cancel_all_impl(self) -> None:
        assert self.array_job_id is not None  # guarded by cancel_all()
        try:
            result = subprocess.run(
                ["scancel", self.array_job_id],
                capture_output=True,
                text=True,
                timeout=_SLURM_CMD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "scancel %s timed out after %.0fs",
                self.array_job_id,
                _SLURM_CMD_TIMEOUT,
            )
            return
        if result.returncode != 0:
            logger.warning(
                "scancel %s returned rc=%d: %s",
                self.array_job_id,
                result.returncode,
                result.stderr.strip(),
            )
        else:
            logger.info("Cancelled all tasks for array job %s", self.array_job_id)

    @property
    def pending_count(self) -> int:
        """Estimated number of tasks that have not yet started.

        Computed from internal accounting rather than ``_last_poll`` because
        Slurm's JSON API aggregates all PENDING array tasks into a single
        ``squeue`` entry (``array_task_id`` is a dict like
        ``{"set": "0-7", "number": 0, ...}``), making it impossible to count
        individual pending tasks from poll results.  Instead we subtract the
        tasks we have *seen* (ever ran or died) from the total chunk size.
        """
        seen = self._known_dead | set(self._known_running.keys())
        return max(0, self.chunk_size - len(seen))

    @property
    def running_count(self) -> int:
        """Number of tasks currently running (accounting-based).

        Uses internal tracking rather than ``_last_poll`` so that a transient
        squeue failure (empty poll) cannot falsely report 0 running tasks and
        trigger premature ``is_done`` detection.
        """
        return len(set(self._known_running.keys()) - self._known_dead)

    @property
    def is_done(self) -> bool:
        """True when this chunk has no pending or running tasks and has been polled.

        A chunk is considered done once at least one task has been seen as
        running or dead (i.e. it has been polled) and there are no more
        pending or running tasks in the last poll.  This guards against
        falsely marking a freshly submitted chunk as done before squeue
        reflects its tasks.
        """
        if not (self._known_running or self._known_dead):
            return False  # Not yet substantively polled
        return self.pending_count == 0 and self.running_count == 0
