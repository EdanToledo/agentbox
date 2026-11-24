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

"""Tests for SlurmArrayChunk and SlurmArrayResourceManager."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _endpoints import run_endpoint
from agentbox.actors.resource_managers.slurm import SlurmResourceManager
from agentbox.actors.resource_managers.slurm_array import (
    ArrayTaskInfo,
    SlurmAllocation,
    SlurmArrayChunk,
)


_PORT_SERVER = "login-node:43210"


def _make_chunk(
    chunk_size: int = 4,
    max_parallel: int = 2,
    job_id: str = "12345",
    port_server_addr: str = _PORT_SERVER,
) -> SlurmArrayChunk:
    """Create a SlurmArrayChunk with a pre-set job ID (as if submitted)."""
    chunk = SlurmArrayChunk(
        chunk_size=chunk_size,
        max_parallel=max_parallel,
        port_server_addr=port_server_addr,
    )
    chunk.array_job_id = job_id
    return chunk


def _make_squeue_json(tasks: list[dict[str, Any]]) -> str:
    """Build a mock squeue --json response."""
    jobs = []
    for t in tasks:
        entry: dict[str, Any] = {
            "array_task_id": t["task_id"],
            "job_state": [t["state"]],
        }
        if t.get("hostname"):
            entry["nodes"] = t["hostname"]
        jobs.append(entry)
    return json.dumps({"jobs": jobs})


def _make_array_manager(
    hosts: list[str] | None = None,
    chunk: SlurmArrayChunk | None = None,
) -> SlurmResourceManager:
    """Construct a SlurmResourceManager without submitting jobs or binding a port.

    *hosts* are added as running workers of *chunk* the way ``poll_pending``
    records them, with slot keys ``{job_id}_{task_index}``.
    """
    with (
        patch("agentbox.port_server.PortRegistrationServer") as server_cls,
        patch.object(SlurmResourceManager, "_submit_chunk"),
    ):
        server_cls.return_value.address = "localhost:0"
        mgr = SlurmResourceManager(chunk_size=4)

    if chunk is not None:
        mgr._chunks.append(chunk)
    if hosts:
        job_id = chunk.array_job_id if chunk else "00000"
        for i, h in enumerate(hosts):
            slot_key = f"{job_id}_{i}"
            mgr._available.append(slot_key)
            if chunk is not None:
                mgr._slot_to_host[slot_key] = h
                mgr._slot_to_chunk[slot_key] = chunk
                mgr._slot_to_task_id[slot_key] = i
                mgr._slot_to_port[slot_key] = 22222
    return mgr


class TestSubmit:
    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_script_has_array_directive(self, mock_run, _mock_ctx):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)

        chunk = SlurmArrayChunk(
            chunk_size=4, max_parallel=2, port_server_addr=_PORT_SERVER
        )
        job_id = chunk.submit()

        assert job_id == "12345"
        assert chunk.array_job_id == "12345"

        # Check the sbatch script passed via stdin.
        call_kwargs = mock_run.call_args
        script = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input", "")
        assert "--array=0-3%2" in script
        assert "--nodes=1" in script
        assert "--cpus-per-task=4" in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_script_with_qos_and_account(self, mock_run, _mock_ctx):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)

        alloc = SlurmAllocation(qos="high", account="myteam", partition="gpu")
        chunk = SlurmArrayChunk(
            chunk_size=2, allocation=alloc, port_server_addr=_PORT_SERVER
        )
        chunk.submit()

        script = mock_run.call_args.kwargs.get("input") or mock_run.call_args[1].get(
            "input", ""
        )
        assert "--qos=high" in script
        assert "--account=myteam" in script
        assert "--partition=gpu" in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_script_uses_worker_logs_base_path(self, mock_run, _mock_ctx):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)
        alloc = SlurmAllocation(worker_logs_base_path="/shared/worker-logs")

        SlurmArrayChunk(
            chunk_size=2, allocation=alloc, port_server_addr=_PORT_SERVER
        ).submit()

        script = mock_run.call_args.kwargs["input"]
        assert "#SBATCH --output=/shared/worker-logs/slurm-%A_%a.out" in script
        assert "#SBATCH --error=/shared/worker-logs/slurm-%A_%a.err" in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_no_throttle_when_max_ge_size(self, mock_run, _mock_ctx):
        mock_run.return_value = MagicMock(stdout="99999\n", returncode=0)

        chunk = SlurmArrayChunk(
            chunk_size=3, max_parallel=10, port_server_addr=_PORT_SERVER
        )
        chunk.submit()

        script = mock_run.call_args.kwargs["input"]
        assert "#SBATCH --array=0-2\n" in script
        # Should NOT have %M when max_parallel >= chunk_size
        assert "%10" not in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_uses_worker_bootstrap_when_port_server_set(
        self, mock_run, _mock_ctx
    ):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)

        chunk = SlurmArrayChunk(chunk_size=2, port_server_addr=_PORT_SERVER)
        chunk.submit()

        script = mock_run.call_args.kwargs.get("input") or mock_run.call_args[1].get(
            "input", ""
        )
        assert "AGENTBOX_PORT_SERVER" in script
        assert "login-node:43210" in script
        assert "-m agentbox.worker_bootstrap" in script
        assert "srun python" in script
        assert "srun --help" not in script
        assert "--cpu-bind" not in script
        assert "SLURM_GPUS_ON_NODE" not in script
        # Should NOT contain the legacy inline hash command.
        assert "hashlib" not in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_passes_extra_srun_args_to_worker_step(self, mock_run, _mock_ctx):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)
        allocation = SlurmAllocation(
            exclusive=True,
            extra_srun_args=(
                "--cpu-bind=none",
                "${SLURM_GPUS_ON_NODE:+--gres=gpu:${SLURM_GPUS_ON_NODE}}",
            ),
        )

        SlurmArrayChunk(
            chunk_size=2,
            allocation=allocation,
            port_server_addr=_PORT_SERVER,
        ).submit()

        script = mock_run.call_args.kwargs["input"]
        assert "#SBATCH --exclusive" in script
        assert (
            "srun --cpu-bind=none "
            "${SLURM_GPUS_ON_NODE:+--gres=gpu:${SLURM_GPUS_ON_NODE}} python"
        ) in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    @patch("subprocess.run")
    def test_submit_whole_node_configures_allocation_and_worker_step(
        self, mock_run, _mock_ctx
    ):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)

        SlurmArrayChunk(
            chunk_size=2,
            allocation=SlurmAllocation(whole_node=True),
            port_server_addr=_PORT_SERVER,
        ).submit()

        script = mock_run.call_args.kwargs["input"]
        assert "#SBATCH --exclusive" in script
        assert (
            '_agentbox_whole_node_srun_args=("--cpu-bind=none" '
            '"--cpus-per-task=${SLURM_CPUS_ON_NODE}")'
        ) in script
        assert "SLURM_GPUS_ON_NODE" in script
        assert "nvidia-smi -L" in script
        assert "--gres" not in script
        assert ('srun "${_agentbox_whole_node_srun_args[@]}" python') in script

    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    def test_whole_node_worker_step_uses_all_allocated_cpus(self, _mock_ctx):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)
            SlurmArrayChunk(
                chunk_size=1,
                allocation=SlurmAllocation(whole_node=True),
                port_server_addr=_PORT_SERVER,
            ).submit()
            script = mock_run.call_args.kwargs["input"]

        fake_commands = (
            "srun() {\n"
            '    if [[ $1 == "--help" ]]; then return 1; fi\n'
            '    printf "%s\\n" "$*"\n'
            "}\n"
            "nvidia-smi() { return 0; }\n"
        )
        env = {
            **os.environ,
            "SLURM_CPUS_ON_NODE": "64",
            "SLURM_CPUS_PER_TASK": "4",
        }
        env.pop("SLURM_GPUS_ON_NODE", None)

        result = subprocess.run(
            ["bash"],
            input=fake_commands + script,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        assert result.stdout.strip().startswith("--cpu-bind=none --cpus-per-task=64")

    @pytest.mark.parametrize(
        ("slurm_gpus", "nvidia_smi", "expected_gpu_arg"),
        [
            ("8", "GPU 0\nGPU 1\n", "--gpus-per-node=8"),
            (None, "GPU 0\nGPU 1\n", "--gpus-per-node=2"),
            (None, "", None),
        ],
    )
    @patch("agentbox.actors.resource_managers.slurm_array.scrubbed_slurm_env")
    def test_whole_node_worker_step_resolves_gpu_count(
        self, _mock_ctx, slurm_gpus, nvidia_smi, expected_gpu_arg
    ):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)
            SlurmArrayChunk(
                chunk_size=1,
                allocation=SlurmAllocation(
                    whole_node=True,
                    gpus_per_node=int(slurm_gpus) if slurm_gpus else 0,
                ),
                port_server_addr=_PORT_SERVER,
            ).submit()
            script = mock_run.call_args.kwargs["input"]

        fake_commands = (
            "srun() {\n"
            '    if [[ $1 == "--help" ]]; then return 1; fi\n'
            '    if [[ -n ${SLURM_GPUS_PER_NODE:-} && $* == *"--gres="* ]]; then\n'
            "        return 2\n"
            "    fi\n"
            '    printf "%s\\n" "$*"\n'
            "}\n"
            "nvidia-smi() {\n"
            '    printf "%s" "$FAKE_NVIDIA_SMI"\n'
            "}\n"
        )
        env = {
            **os.environ,
            "FAKE_NVIDIA_SMI": nvidia_smi,
            "SLURM_CPUS_ON_NODE": "64",
        }
        if slurm_gpus is None:
            env.pop("SLURM_GPUS_ON_NODE", None)
            env.pop("SLURM_GPUS_PER_NODE", None)
        else:
            env["SLURM_GPUS_ON_NODE"] = slurm_gpus
            env["SLURM_GPUS_PER_NODE"] = slurm_gpus

        result = subprocess.run(
            ["bash"],
            input=fake_commands + script,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        worker_step = result.stdout.strip()
        assert worker_step.startswith("--cpu-bind=none --cpus-per-task=64")
        assert "--gres=" not in worker_step
        if expected_gpu_arg is None:
            assert "--gpus-per-node=" not in worker_step
        else:
            assert expected_gpu_arg in worker_step


class TestPollTasks:
    @patch("subprocess.run")
    def test_poll_tasks_parses_array_entries(self, mock_run):
        chunk = _make_chunk(chunk_size=3)
        squeue_json = _make_squeue_json(
            [
                {"task_id": 0, "state": "RUNNING", "hostname": "node-001"},
                {"task_id": 1, "state": "PENDING"},
                {"task_id": 2, "state": "RUNNING", "hostname": "node-002"},
            ]
        )
        mock_run.return_value = MagicMock(stdout=squeue_json, returncode=0)

        tasks = chunk.poll_tasks()

        assert len(tasks) == 3
        running = [t for t in tasks if t.status == "RUNNING"]
        assert len(running) == 2
        assert {t.hostname for t in running} == {"node-001", "node-002"}

    @patch("subprocess.run")
    def test_poll_tasks_handles_squeue_failure(self, mock_run):
        chunk = _make_chunk()
        mock_run.return_value = MagicMock(stdout="", stderr="error", returncode=1)

        tasks = chunk.poll_tasks()
        assert tasks == []

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["squeue"], 60.0))
    def test_poll_tasks_handles_squeue_timeout(self, mock_run):
        # A wedged squeue must not propagate out of poll_tasks (which would hang
        # the ResourceManager actor); it's treated as a failed poll.
        chunk = _make_chunk()
        tasks = chunk.poll_tasks()
        assert tasks == []
        assert chunk._last_poll_ok is False


class TestGetNewlyRunning:
    @patch("subprocess.run")
    def test_get_newly_running_is_incremental(self, mock_run):
        chunk = _make_chunk(chunk_size=3)

        # First poll: tasks 0 and 1 are RUNNING.
        squeue_json = _make_squeue_json(
            [
                {"task_id": 0, "state": "RUNNING", "hostname": "node-001"},
                {"task_id": 1, "state": "RUNNING", "hostname": "node-002"},
                {"task_id": 2, "state": "PENDING"},
            ]
        )
        mock_run.return_value = MagicMock(stdout=squeue_json, returncode=0)

        def lookup(jid: str, tid: int) -> tuple[str, int]:
            assert jid == "12345"
            return (f"node-00{tid + 1}", 40000 + tid)

        new = chunk.get_newly_running(lookup)
        assert sorted(new) == [(0, "node-001", 40000), (1, "node-002", 40001)]

        # Second poll: task 2 also running now.
        squeue_json = _make_squeue_json(
            [
                {"task_id": 0, "state": "RUNNING", "hostname": "node-001"},
                {"task_id": 1, "state": "RUNNING", "hostname": "node-002"},
                {"task_id": 2, "state": "RUNNING", "hostname": "node-003"},
            ]
        )
        mock_run.return_value = MagicMock(stdout=squeue_json, returncode=0)

        new = chunk.get_newly_running(lookup)
        assert new == [(2, "node-003", 40002)]

    @patch("subprocess.run")
    def test_get_newly_running_with_port_lookup(self, mock_run):
        chunk = _make_chunk(chunk_size=2)

        squeue_json = _make_squeue_json(
            [
                {"task_id": 0, "state": "RUNNING", "hostname": "node-001"},
                {"task_id": 1, "state": "RUNNING", "hostname": "node-002"},
            ]
        )
        mock_run.return_value = MagicMock(stdout=squeue_json, returncode=0)

        # port_lookup returns port for task 0, None for task 1 (not registered yet).
        def lookup(jid: str, tid: int) -> tuple[str, int] | None:
            if tid == 0:
                return ("node-001", 55555)
            return None

        new = chunk.get_newly_running(port_lookup=lookup)
        assert new == [(0, "node-001", 55555)]

        # Task 1 is deferred, then picked up once its port is registered.
        assert 1 not in chunk._known_running
        new = chunk.get_newly_running(port_lookup=lambda jid, tid: ("node-002", 44444))
        assert new == [(1, "node-002", 44444)]


class TestGetDeadTasks:
    @patch("subprocess.run")
    def test_get_dead_tasks_detects_preemption(self, mock_run):
        chunk = _make_chunk(chunk_size=2)
        # First: both running.
        chunk._known_running = {0: "node-001", 1: "node-002"}

        squeue_json = _make_squeue_json(
            [
                {"task_id": 0, "state": "RUNNING", "hostname": "node-001"},
                {"task_id": 1, "state": "PREEMPTED", "hostname": "node-002"},
            ]
        )
        mock_run.return_value = MagicMock(stdout=squeue_json, returncode=0)

        dead = chunk.get_dead_tasks()
        assert len(dead) == 1
        assert dead[0] == (1, "node-002")

        # Second call should not re-report task 1.
        dead = chunk.get_dead_tasks()
        assert len(dead) == 0

    @patch("subprocess.run")
    def test_get_dead_tasks_no_false_eviction_on_squeue_failure(self, mock_run):
        """A failed squeue call must not cause running tasks to be reported dead."""
        chunk = _make_chunk(chunk_size=2)
        chunk._known_running = {0: "node-001", 1: "node-002"}

        # squeue fails with a non-zero exit code.
        mock_run.return_value = MagicMock(stdout="", returncode=1, stderr="slurm down")

        dead = chunk.get_dead_tasks()

        assert dead == [], "squeue failure must not falsely evict running tasks"
        assert chunk._known_dead == set(), (
            "no tasks should be marked dead after failed poll"
        )

    @patch("subprocess.run")
    def test_get_dead_tasks_no_false_eviction_on_invalid_json(self, mock_run):
        """Invalid squeue JSON must not cause running tasks to be reported dead."""
        chunk = _make_chunk(chunk_size=2)
        chunk._known_running = {0: "node-001", 1: "node-002"}

        mock_run.return_value = MagicMock(stdout="not json", returncode=0, stderr="")

        dead = chunk.get_dead_tasks()

        assert dead == [], "JSON parse failure must not falsely evict running tasks"

    @patch("subprocess.run")
    def test_get_dead_tasks_detects_disappeared_after_successful_poll(self, mock_run):
        """Tasks absent from a successful squeue response ARE reported dead."""
        chunk = _make_chunk(chunk_size=2)
        chunk._known_running = {0: "node-001", 1: "node-002"}

        # Successful poll but task 1 is gone (Slurm already cleaned it up).
        squeue_json = _make_squeue_json(
            [{"task_id": 0, "state": "RUNNING", "hostname": "node-001"}]
        )
        mock_run.return_value = MagicMock(stdout=squeue_json, returncode=0)

        dead = chunk.get_dead_tasks()

        assert len(dead) == 1
        assert dead[0] == (1, "node-002")

    @patch("subprocess.run")
    def test_cancel_task_format(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        chunk = _make_chunk(job_id="12345")

        chunk.cancel_task(3)

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["scancel", "12345_3"]
        # A failed cancel of one task must surface, and must not hang.
        assert mock_run.call_args.kwargs["check"] is True
        assert mock_run.call_args.kwargs["timeout"] is not None

    @patch("subprocess.run")
    def test_cancel_all(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        chunk = _make_chunk(job_id="12345")

        chunk.cancel_all()

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["scancel", "12345"]
        # Best-effort: cancelling the whole array never raises, and never hangs.
        assert not mock_run.call_args.kwargs.get("check")
        assert mock_run.call_args.kwargs["timeout"] is not None


class TestPollPending:
    def test_poll_pending_adds_hosts(self):
        chunk = _make_chunk(chunk_size=2)
        mgr = _make_array_manager(chunk=chunk)

        # Mock get_newly_running to return two hosts (3-tuples with port).
        chunk.get_newly_running = MagicMock(
            return_value=[(0, "node-001", 22222), (1, "node-002", 22222)]
        )
        chunk._last_poll = [
            ArrayTaskInfo("12345", 0, "RUNNING", "node-001"),
            ArrayTaskInfo("12345", 1, "RUNNING", "node-002"),
        ]

        result = run_endpoint(mgr, "poll_pending")

        new_hostnames = [h["hostname"] for h in result["new_hosts"]]
        assert set(new_hostnames) == {"node-001", "node-002"}
        assert len(mgr._available) == 2
        assert mgr._slot_to_chunk["12345_0"] is chunk
        assert mgr._slot_to_task_id["12345_0"] == 0


class TestArrayAcquireRelease:
    def test_acquire_release(self):
        chunk = _make_chunk()
        mgr = _make_array_manager(hosts=["node-001", "node-002"], chunk=chunk)

        handle = run_endpoint(mgr, "acquire", 1)
        assert len(handle.hostnames) == 1
        assert len(mgr._available) == 1

        run_endpoint(mgr, "release", handle.handle_id)
        assert len(mgr._available) == 2

    def test_acquire_too_many_raises(self):
        chunk = _make_chunk()
        mgr = _make_array_manager(hosts=["node-001"], chunk=chunk)
        with pytest.raises(RuntimeError, match="only 1 available"):
            run_endpoint(mgr, "acquire", 2)


class TestArrayHealthCheck:
    def test_health_check_evicts_per_task(self):
        chunk = _make_chunk(chunk_size=2)
        mgr = _make_array_manager(hosts=["node-001", "node-002"], chunk=chunk)

        # Mock: task 1 (node-002) died.
        chunk.get_dead_tasks = MagicMock(return_value=[(1, "node-002")])

        result = run_endpoint(mgr, "health_check")

        assert result.evicted_hosts == ["node-002"]
        assert len(mgr._available) == 1
        assert len(result.extras["dead_tasks"]) == 1
        assert result.extras["dead_tasks"][0]["task_id"] == 1


class TestArrayScaleDown:
    def test_scale_down_cancels_individual_tasks(self):
        chunk = _make_chunk(chunk_size=2)
        mgr = _make_array_manager(hosts=["node-001", "node-002"], chunk=chunk)

        chunk.cancel_task = MagicMock()
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "scale_by", -1)

        assert len(mgr._available) == 1
        chunk.cancel_task.assert_called_once()

    def test_scale_down_removes_empty_chunk(self):
        chunk = _make_chunk(chunk_size=1)
        mgr = _make_array_manager(hosts=["node-001"], chunk=chunk)

        chunk.cancel_task = MagicMock()
        chunk.cancel_all = MagicMock()

        run_endpoint(mgr, "scale_by", -1)

        assert len(mgr._chunks) == 0
        chunk.cancel_all.assert_called_once()

    def test_scale_down_too_many_raises(self):
        chunk = _make_chunk()
        mgr = _make_array_manager(hosts=["node-001"], chunk=chunk)
        with pytest.raises(RuntimeError, match="only 1 idle"):
            run_endpoint(mgr, "scale_by", -2)


class TestArrayShutdown:
    def test_shutdown_cancels_all_chunks(self):
        chunk1 = _make_chunk(job_id="111")
        chunk2 = _make_chunk(job_id="222")
        mgr = _make_array_manager(hosts=["node-001"], chunk=chunk1)
        mgr._chunks.append(chunk2)
        slot_key2 = f"{chunk2.array_job_id}_0"
        mgr._slot_to_host[slot_key2] = "node-002"
        mgr._slot_to_chunk[slot_key2] = chunk2
        mgr._slot_to_task_id[slot_key2] = 0
        mgr._slot_to_port[slot_key2] = 22222
        mgr._available.append(slot_key2)

        chunk1.cancel_all = MagicMock()
        chunk2.cancel_all = MagicMock()

        run_endpoint(mgr, "shutdown")

        chunk1.cancel_all.assert_called_once()
        chunk2.cancel_all.assert_called_once()
        assert len(mgr._chunks) == 0
        assert len(mgr._available) == 0
        assert len(mgr._handles) == 0

    def test_shutdown_tolerates_cancel_failure(self):
        chunk = _make_chunk()
        mgr = _make_array_manager(hosts=["node-001"], chunk=chunk)
        chunk.cancel_all = MagicMock(side_effect=RuntimeError("scancel failed"))

        run_endpoint(mgr, "shutdown")  # Should not raise.
        assert len(mgr._chunks) == 0
