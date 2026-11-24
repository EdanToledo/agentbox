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

"""Tests for agentbox.actors.resource_managers.local."""

from __future__ import annotations

import itertools
import sys
import types
from unittest.mock import MagicMock, patch

import agentbox.actors.resource_managers.local as local_module
import pytest
from _endpoints import run_endpoint
from agentbox.actors.resource_managers.local import LocalResourceManager


@pytest.fixture
def local_tmpdir(tmp_path, monkeypatch):
    """Make LocalResourceManager create its worker directory under *tmp_path*."""
    monkeypatch.setattr(
        local_module,
        "tempfile",
        types.SimpleNamespace(mkdtemp=lambda prefix: str(tmp_path)),
    )
    monkeypatch.setattr(
        LocalResourceManager, "_register_worker_for_supervision", MagicMock()
    )
    return tmp_path


@pytest.fixture
def make_local_manager(local_tmpdir):
    """Build a LocalResourceManager whose workers are mock (alive) processes."""
    pids = itertools.count(10000)

    def spawn(*args, **kwargs) -> MagicMock:
        proc = MagicMock(pid=next(pids))
        proc.poll.return_value = None  # alive by default
        return proc

    def make(n: int) -> LocalResourceManager:
        with patch.object(local_module.subprocess, "Popen", side_effect=spawn):
            return LocalResourceManager(initial_nodes=n)

    return make


class TestSpawnWorkerStdio:
    """Workers must get their own stdio rather than inheriting the parent's.

    A detached worker that inherits (and holds) the parent's stdout/stderr
    *pipe* keeps it open after the parent exits, hanging any parent that drains
    the pipe to EOF before exiting (notably ``conda run``).
    """

    @patch("agentbox.actors.resource_managers.local.subprocess.Popen")
    def test_workers_redirect_stdio_off_parent(self, mock_popen, local_tmpdir):
        import subprocess as _sp

        mock_popen.return_value = MagicMock(pid=4321)

        LocalResourceManager(initial_nodes=1)

        assert mock_popen.call_count == 1
        kwargs = mock_popen.call_args.kwargs
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == _sp.DEVNULL
        # stdout/stderr must be the worker's own sink, never None (which would
        # inherit the parent's descriptors).
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] is not None
        assert (local_tmpdir / "local_0.log").exists()
        command = mock_popen.call_args.args[0]
        assert command == [
            sys.executable,
            "-m",
            "agentbox.local_worker_entry",
            "--agentbox-local-worker",
        ]
        env = mock_popen.call_args.kwargs["env"]
        assert env["_AGENTBOX_LOCAL_WORKER_ADDRESS"].endswith("/local_0")


class TestLocalAcquireRelease:
    def test_acquire_single(self, make_local_manager):
        mgr = make_local_manager(2)
        handle = run_endpoint(mgr, "acquire", 1)

        assert len(handle.hostnames) == 1
        assert handle.handle_id

    def test_acquire_all(self, make_local_manager):
        mgr = make_local_manager(2)
        handle = run_endpoint(mgr, "acquire", 2)

        assert len(handle.hostnames) == 2

    def test_acquire_too_many_raises(self, make_local_manager):
        mgr = make_local_manager(1)
        with pytest.raises(RuntimeError, match="only 1 available"):
            run_endpoint(mgr, "acquire", 2)

    def test_acquire_zero_raises(self, make_local_manager):
        mgr = make_local_manager(1)
        with pytest.raises(ValueError, match="n must be >= 1"):
            run_endpoint(mgr, "acquire", 0)

    def test_release_returns_to_pool(self, make_local_manager):
        mgr = make_local_manager(2)
        handle = run_endpoint(mgr, "acquire", 2)
        assert len(mgr._available) == 0

        run_endpoint(mgr, "release", handle.handle_id)
        assert len(mgr._available) == 2

    def test_release_unknown_handle_raises(self, make_local_manager):
        mgr = make_local_manager(1)
        with pytest.raises(KeyError, match="Unknown handle"):
            run_endpoint(mgr, "release", "nonexistent")

    def test_double_release_raises(self, make_local_manager):
        mgr = make_local_manager(1)
        handle = run_endpoint(mgr, "acquire", 1)
        run_endpoint(mgr, "release", handle.handle_id)
        with pytest.raises(KeyError):
            run_endpoint(mgr, "release", handle.handle_id)


class TestLocalPoolStatus:
    def test_initial_status(self, make_local_manager):
        mgr = make_local_manager(2)
        status = run_endpoint(mgr, "pool_status")

        assert status.available == 2
        assert status.in_use == 0
        assert status.total == 2

    def test_status_after_acquire(self, make_local_manager):
        mgr = make_local_manager(2)
        run_endpoint(mgr, "acquire", 1)
        status = run_endpoint(mgr, "pool_status")

        assert status.available == 1
        assert status.in_use == 1
        assert status.total == 2

    def test_status_after_release(self, make_local_manager):
        mgr = make_local_manager(2)
        handle = run_endpoint(mgr, "acquire", 2)
        run_endpoint(mgr, "release", handle.handle_id)
        status = run_endpoint(mgr, "pool_status")

        assert status.available == 2
        assert status.in_use == 0


class TestLocalScaleDown:
    def test_scale_down_removes_idle(self, make_local_manager):
        mgr = make_local_manager(3)
        # The worker that will be scaled down (LIFO: last available popped).
        killed = mgr._workers[mgr._available[-1]].process
        run_endpoint(mgr, "scale_by", -1)

        assert len(mgr._available) == 2
        assert len(mgr._workers) == 2
        # The removed worker is terminated and reaped (no zombie).
        killed.terminate.assert_called_once()
        killed.wait.assert_called()

    def test_scale_down_too_many_raises(self, make_local_manager):
        mgr = make_local_manager(1)
        with pytest.raises(RuntimeError, match="only 1 idle"):
            run_endpoint(mgr, "scale_by", -2)

    def test_scale_down_zero_raises(self, make_local_manager):
        mgr = make_local_manager(1)
        with pytest.raises(ValueError, match="n must be non-zero"):
            run_endpoint(mgr, "scale_by", 0)


class TestLocalShutdown:
    @patch("shutil.rmtree")
    def test_shutdown_kills_all(self, mock_rmtree, make_local_manager):
        mgr = make_local_manager(2)
        procs = [w.process for w in mgr._workers.values()]
        run_endpoint(mgr, "shutdown")

        # Every worker is terminated and reaped (wait() called — no zombies).
        for proc in procs:
            proc.terminate.assert_called_once()
            proc.wait.assert_called()
        assert len(mgr._workers) == 0
        assert len(mgr._available) == 0
        assert len(mgr._handles) == 0
        mock_rmtree.assert_called_once()

    @patch("shutil.rmtree")
    def test_shutdown_tolerates_kill_failure(self, mock_rmtree, make_local_manager):
        mgr = make_local_manager(1)
        for w in mgr._workers.values():
            w.process.terminate.side_effect = OSError("No such process")
        run_endpoint(mgr, "shutdown")  # should not raise

        assert len(mgr._workers) == 0

    @patch("shutil.rmtree")
    def test_shutdown_after_acquire(self, mock_rmtree, make_local_manager):
        mgr = make_local_manager(2)
        run_endpoint(mgr, "acquire", 1)
        run_endpoint(mgr, "shutdown")

        assert len(mgr._handles) == 0


class TestLocalHealthCheck:
    def test_all_healthy(self, make_local_manager):
        mgr = make_local_manager(2)
        result = run_endpoint(mgr, "health_check")

        assert result.evicted_hosts == []
        assert len(result.healthy_hosts) == 2
        assert result.extras["dead_pids"] == []

    def test_dead_worker_evicted(self, make_local_manager):
        mgr = make_local_manager(2)
        # First worker alive (poll -> None), second worker dead (poll -> exit code).
        workers = list(mgr._workers.values())
        workers[0].process.poll.return_value = None
        workers[1].process.poll.return_value = 1

        result = run_endpoint(mgr, "health_check")

        assert len(result.evicted_hosts) == 1
        assert len(result.healthy_hosts) == 1
        assert len(result.extras["dead_pids"]) == 1
        assert len(mgr._workers) == 1
        assert len(mgr._available) == 1


class TestLocalHandleMetadata:
    def test_metadata_contains_pids(self, make_local_manager):
        mgr = make_local_manager(1)
        handle = run_endpoint(mgr, "acquire", 1)

        assert "pids" in handle.metadata
        assert len(handle.metadata["pids"]) == 1
        # The key should be the address, value should be the pid
        for addr, pid in handle.metadata["pids"].items():
            assert addr.startswith("ipc://")
            assert isinstance(pid, int)
