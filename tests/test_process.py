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

"""Tests for agentbox.process."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch  # noqa: E402

import agentbox.process as _proc_mod  # noqa: E402
import pytest
from agentbox.process import (
    reap_monarch_descendants,  # noqa: E402
    terminate_process,
    track_monarch_descendants,
)


@pytest.fixture(autouse=True)
def _fresh_tracked_pids(monkeypatch):
    """Give every test its own tracked-pid set so none leaks into the next."""
    monkeypatch.setattr(_proc_mod, "_tracked_monarch_pids", set())


def _make_proc(**kwargs) -> MagicMock:
    """Create a mock subprocess.Popen."""
    return MagicMock(spec=subprocess.Popen, **kwargs)


def test_terminate_clean_exit():
    """Process exits within soft_timeout — kill() should NOT be called."""
    proc = _make_proc()
    terminate_process(proc)

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once_with(timeout=10)
    proc.kill.assert_not_called()


def test_terminate_escalates_to_kill():
    """First wait() times out — kill() should be called."""
    proc = _make_proc()
    proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]

    terminate_process(proc)

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2


def test_terminate_already_dead():
    """terminate() raises OSError — should not crash, wait() still called."""
    proc = _make_proc()
    proc.terminate.side_effect = OSError("No such process")

    terminate_process(proc)

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


def test_kill_already_dead():
    """kill() raises OSError — should not crash, second wait() still called."""
    proc = _make_proc()
    proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
    proc.kill.side_effect = OSError("No such process")

    terminate_process(proc)

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2


def test_custom_timeouts():
    """soft_timeout and hard_timeout are forwarded to wait()."""
    proc = _make_proc()
    proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 3), None]

    terminate_process(proc, soft_timeout=3, hard_timeout=2)

    calls = proc.wait.call_args_list
    assert calls[0].kwargs["timeout"] == 3
    assert calls[1].kwargs["timeout"] == 2


def test_terminate_unkillable_process_does_not_raise(caplog):
    """Both waits time out (process stuck after SIGKILL) — logs, never raises."""
    import logging

    proc = _make_proc(pid=4321)
    # Soft wait and post-kill hard wait both time out.
    proc.wait.side_effect = [
        subprocess.TimeoutExpired("cmd", 10),
        subprocess.TimeoutExpired("cmd", 5),
    ]

    with caplog.at_level(logging.WARNING, logger="agentbox.process"):
        terminate_process(proc)  # must not raise

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2
    assert any("4321" in r.getMessage() for r in caplog.records)


# Monarch descendant reaper


# A fake `ps -ax -o pid,ppid,command` listing. Root process is pid 1000.
_PS_AX = """\
  PID  PPID COMMAND
 1000     1 /usr/bin/python myscript.py
 1001  1000 /env/python -m monarch._src.actor.bootstrap_main
 1002  1001 /env/python -c from monarch.actor import run_worker_loop_forever; ...
 1003  1000 /bin/bash -c something
 1004  1000 /env/python -m monarch._src.job._job_sidecar_worker /tmp/sidecar.sock 4
 1005  1004 /env/python -m monarch._src.actor.bootstrap_main
 1006  1005 /env/python -c from monarch.actor import run_worker_loop_forever; ...
 1007  1000 /env/python -m agentbox.local_worker_entry --agentbox-local-worker
 2000     1 /env/python -m monarch._src.actor.bootstrap_main
"""

# Per-pid command lines for the `ps -o command= -p <pid>` lookups.
_PROC_CMD = {
    "3001": "/env/python -m monarch._src.actor.bootstrap_main",
    "3002": "/usr/bin/vim notes.txt",
}


def _fake_ps(cmd, **kwargs):
    if "-ax" in cmd:
        return MagicMock(stdout=_PS_AX)
    if "-p" in cmd:  # ps -o command= -p <pid>
        pid = cmd[cmd.index("-p") + 1]
        return MagicMock(
            stdout=_PROC_CMD.get(pid, "") + ("\n" if pid in _PROC_CMD else "")
        )
    return MagicMock(stdout="")


def test_reap_kills_only_marked_descendants():
    with (
        patch("agentbox.process.os.getpid", return_value=1000),
        patch("agentbox.process.subprocess.run", side_effect=_fake_ps),
        patch("agentbox.process.os.kill") as mock_kill,
    ):
        killed = reap_monarch_descendants()

    sent = sorted(c.args[0] for c in mock_kill.call_args_list)
    # 1001 (bootstrap child), 1002 (worker grandchild), and 1007 (local worker);
    # not 1003 (bash), the persistent job-sidecar subtree 1004-1006, or 2000
    # (not a descendant).
    assert sent == [1001, 1002, 1007]
    assert killed == 3


def test_reap_includes_tracked_orphan_but_skips_reused_pid():
    _proc_mod._tracked_monarch_pids.update({3001, 3002})
    with (
        patch("agentbox.process.os.getpid", return_value=1000),
        patch("agentbox.process.subprocess.run", side_effect=_fake_ps),
        patch("agentbox.process.os.kill") as mock_kill,
    ):
        reap_monarch_descendants()

    sent = sorted(c.args[0] for c in mock_kill.call_args_list)
    # 1001/1002/1007 = live descendants; 3001 = tracked orphan still a monarch
    # proc; 3002 = tracked pid now reused by an unrelated process -> skipped.
    assert sent == [1001, 1002, 1007, 3001]


def test_track_records_only_marked_descendants():
    with (
        patch("agentbox.process.os.getpid", return_value=1000),
        patch("agentbox.process.subprocess.run", side_effect=_fake_ps),
    ):
        track_monarch_descendants()
    assert _proc_mod._tracked_monarch_pids == {1001, 1002, 1007}
