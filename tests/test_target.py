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

"""Tests for the ExecutionTarget command/file API on Machine and Container."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from unittest.mock import AsyncMock, call, MagicMock, patch

import pytest
from agentbox.actors.container import (
    _HOST_GIVE,
    _HOST_TAKE,
    _PLACE,
    _STAGE,
    _STAGED_DIRECTORY,
    _UNPACK,
    Container,
)
from agentbox.actors.machine import Machine
from agentbox.remote import tree_archive
from agentbox.remote.file_transfer import FileCopyError
from agentbox.remote.subprocess import CompletedProcess, PIPE


_EXEC = ["apptainer", "exec", "--cleanenv"]
_EXEC_FLAGS = ["--no-home", "--containall", "instance://inst"]


def _machine() -> Machine:
    machine = Machine(proc_mesh=MagicMock())
    machine._task_pool = MagicMock()
    return machine


def _container(machine: object | None = None, **kwargs: object) -> Container:
    machine = MagicMock() if machine is None else machine
    return Container(machine, image="t.sif", instance_name="inst", **kwargs)


def _result(returncode: int = 0, stdout: str = "") -> CompletedProcess[str]:
    return CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr="boom"
    )


class TestMachineHostArgv:
    def test_passes_argv_and_cwd_through(self):
        assert _machine()._host_argv(["ls", "-l"], None, "/srv") == (
            ["ls", "-l"],
            "/srv",
        )

    def test_env_is_added_with_env_prefix(self):
        argv, cwd = _machine()._host_argv(["printenv"], {"A": "1", "B": "x y"}, None)

        assert argv == ["env", "A=1", "B=x y", "printenv"]
        assert cwd is None


class TestContainerHostArgv:
    def test_wraps_argv_in_apptainer_exec_without_host_cwd(self):
        argv, host_cwd = _container()._host_argv(["ls"], None, None)

        assert argv == [*_EXEC, *_EXEC_FLAGS, "ls"]
        assert host_cwd is None

    def test_env_and_cwd_become_container_flags(self):
        argv, host_cwd = _container()._host_argv(["ls"], {"B": "2"}, "/work")

        assert argv == [
            *_EXEC,
            *["--env", "B=2", "--pwd", "/work"],
            *_EXEC_FLAGS,
            "ls",
        ]
        assert host_cwd is None

    def test_default_cwd_is_container_cwd_and_call_cwd_wins(self):
        container = _container(cwd="/workspace")

        default, _ = container._host_argv(["ls"], None, None)
        override, _ = container._host_argv(["ls"], None, "/other")

        assert default[default.index("--pwd") + 1] == "/workspace"
        assert override[override.index("--pwd") + 1] == "/other"


class TestRunDispatch:
    @pytest.mark.parametrize(
        ("cmd", "argv"),
        [
            ("echo $HOME", ["/bin/sh", "-c", "echo $HOME"]),
            (["echo", "hi"], ["echo", "hi"]),
        ],
    )
    def test_machine_run_captures_output_through_task_pool(self, cmd, argv):
        machine = _machine()

        with patch("agentbox.remote.subprocess._run") as run:
            result = machine.run(cmd, cwd="/tmp", input="in", timeout=3.0, check=True)

        assert result is run.return_value
        run.assert_called_once_with(
            argv,
            proc_mesh=machine.proc_mesh,
            task_pool=machine._task_pool,
            cwd="/tmp",
            capture_output=True,
            input="in",
            timeout=3.0,
            check=True,
            text=True,
        )

    async def test_machine_arun_applies_env_prefix(self):
        machine = _machine()

        with patch("agentbox.remote.subprocess._arun", new_callable=AsyncMock) as arun:
            await machine.arun(["true"], env={"A": "1"}, text=False)

        arun.assert_awaited_once()
        assert arun.await_args.args == (["env", "A=1", "true"],)
        assert arun.await_args.kwargs["task_pool"] is machine._task_pool
        assert arun.await_args.kwargs["capture_output"] is True
        assert arun.await_args.kwargs["text"] is False

    def test_container_run_uses_machine_mesh_pool_and_no_host_cwd(self):
        machine = MagicMock()
        container = _container(machine)

        with patch("agentbox.remote.subprocess._run") as run:
            container.run("pwd", cwd="/work")

        assert run.call_args.args[0][-3:] == ["/bin/sh", "-c", "pwd"]
        assert run.call_args.args[0][:2] == ["apptainer", "exec"]
        assert run.call_args.kwargs["proc_mesh"] is machine.proc_mesh
        assert run.call_args.kwargs["task_pool"] is machine._ensure_task_pool()
        assert run.call_args.kwargs["cwd"] is None


class TestSpawnDispatch:
    def test_spawn_pipes_all_stdio(self):
        machine = _machine()

        with patch("agentbox.remote.subprocess.Popen") as popen:
            proc = machine.spawn("cat", cwd="/tmp", env={"A": "1"})

        assert proc is popen.return_value
        popen.assert_called_once_with(
            ["env", "A=1", "/bin/sh", "-c", "cat"],
            proc_mesh=machine.proc_mesh,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd="/tmp",
            text=True,
            start_new_session=True,
        )

    def test_spawn_pty_uses_pty_spawn(self):
        machine = _machine()

        with (
            patch("agentbox.remote.pty.spawn") as pty_spawn,
            patch("agentbox.remote.subprocess.Popen") as popen,
        ):
            proc = machine.spawn(["bash"], pty=True, text=False)

        assert proc is pty_spawn.return_value
        pty_spawn.assert_called_once_with(
            ["bash"], proc_mesh=machine.proc_mesh, cwd=None, text=False
        )
        popen.assert_not_called()

    def test_container_spawn_pty_runs_inside_container(self):
        machine = MagicMock()

        with patch("agentbox.remote.pty.spawn") as pty_spawn:
            _container(machine).spawn(["bash"], pty=True)

        argv = pty_spawn.call_args.args[0]
        assert argv == [*_EXEC, *_EXEC_FLAGS, "bash"]
        assert pty_spawn.call_args.kwargs["proc_mesh"] is machine.proc_mesh
        assert pty_spawn.call_args.kwargs["cwd"] is None


class TestContainerFileStaging:
    """Container copies: the Machine copies only to/from the host directory, and
    one host command renames the copy across and runs the in-container step."""

    @staticmethod
    def _staged(returncode: int = 0, stdout: str = "") -> tuple[Container, MagicMock]:
        # One mock records Machine copies and runs, to check their order.
        machine = MagicMock()
        machine.run.return_value = _result(returncode, stdout)
        container = _container(machine)
        container._staging_dir, container._host_dir = "/staging", "/host"
        return container, machine

    @staticmethod
    def _give_argv(held, staged, script, dst):
        inner = [*_EXEC, *_EXEC_FLAGS, "sh", "-c", script, "sh", staged, dst]
        return ["sh", "-c", _HOST_GIVE, "sh", held, staged, *inner]

    @staticmethod
    def _take_argv(staged, held, src):
        inner = [*_EXEC, "--pwd", "/", *_EXEC_FLAGS, "sh", "-c", _STAGE]
        return ["sh", "-c", _HOST_TAKE, "sh", staged, held, *inner, "sh", src, staged]

    def test_upload_copies_to_host_dir_then_gives_with_one_run(self):
        container, machine = self._staged()

        container.upload("/local/cfg.json", "/work/cfg.json")

        held = machine.upload.call_args.args[1]
        assert held.startswith("/host/") and held.endswith("-cfg.json")
        staged = "/staging/" + os.path.basename(held)
        assert machine.mock_calls == [
            call.upload("/local/cfg.json", held),
            call.run(self._give_argv(held, staged, _PLACE, "/work/cfg.json")),
        ]

    def test_same_basename_uploads_get_distinct_paths(self):
        container, machine = self._staged()

        container.upload("/a/cfg.json", "/x/cfg.json")
        container.upload("/b/cfg.json", "/y/cfg.json")

        first, second = (c.args[1] for c in machine.upload.call_args_list)
        assert first != second

    def test_upload_failed_place_raises(self):
        container, _ = self._staged(returncode=1)

        with pytest.raises(FileCopyError, match="boom"):
            container.upload("/local/cfg.json", "/work/cfg.json")

    @pytest.mark.parametrize("failure", ["transfer", "place"])
    def test_failed_upload_removes_both_copies(self, failure):
        container, machine = self._staged(returncode=1)
        if failure == "transfer":
            machine.upload.side_effect = RuntimeError("transfer failed")

        with pytest.raises((FileCopyError, RuntimeError)):
            container.upload("/local/cfg.json", "/work/cfg.json")

        held = machine.upload.call_args.args[1]
        staged = "/staging/" + os.path.basename(held)
        assert machine.run.call_args == call(["rm", "-rf", staged, held])

    async def test_failed_async_upload_removes_both_copies(self):
        machine = MagicMock()
        machine.aupload = AsyncMock()
        machine.arun = AsyncMock(return_value=_result(1))
        container = _container(machine)
        container._staging_dir, container._host_dir = "/staging", "/host"

        with pytest.raises(FileCopyError):
            await container.aupload("/local/cfg.json", "/work/cfg.json")

        held = machine.aupload.await_args.args[1]
        staged = "/staging/" + os.path.basename(held)
        assert machine.arun.await_args == call(["rm", "-rf", staged, held])

    def test_copies_before_start_raise(self):
        container = _container()

        with pytest.raises(RuntimeError, match="not started"):
            container.upload("/local/cfg.json", "/work/cfg.json")

    def test_download_takes_transfers_and_cleans_up(self):
        container, machine = self._staged()

        container.download("/work/out.txt", "/local/out.txt")

        held = machine.download.call_args.args[0]
        assert held.startswith("/host/") and held.endswith("-out.txt")
        staged = "/staging/" + os.path.basename(held)
        assert machine.mock_calls == [
            call.run(self._take_argv(staged, held, "/work/out.txt")),
            call.download(held, "/local/out.txt"),
            call.run(["rm", "-rf", staged, held]),
        ]

    @pytest.mark.parametrize("failure", ["take", "transfer"])
    def test_download_cleans_up_on_failure(self, failure):
        container, machine = self._staged(returncode=1 if failure == "take" else 0)
        machine.download.side_effect = RuntimeError("transfer failed")

        with pytest.raises((FileCopyError, RuntimeError)):
            container.download("/work/out.txt", "/local/out.txt")

        staged, held = machine.run.call_args_list[0].args[0][4:6]
        assert machine.run.call_args_list[-1] == call(["rm", "-rf", staged, held])
        if failure == "take":
            machine.download.assert_not_called()

    async def test_async_upload_and_download_follow_same_sequence(self):
        machine = MagicMock()
        machine.aupload = AsyncMock()
        machine.adownload = AsyncMock()
        machine.arun = AsyncMock(return_value=_result())
        container = _container(machine)
        container._staging_dir, container._host_dir = "/staging", "/host"

        await container.aupload("/local/in.txt", "/work/in.txt")
        await container.adownload("/work/out.txt", "/local/out.txt")

        up_held = machine.aupload.await_args.args[1]
        down_held = machine.adownload.await_args.args[0]
        up_staged = "/staging/" + os.path.basename(up_held)
        down_staged = "/staging/" + os.path.basename(down_held)
        assert machine.arun.await_args_list == [
            call(self._give_argv(up_held, up_staged, _PLACE, "/work/in.txt")),
            call(self._take_argv(down_staged, down_held, "/work/out.txt")),
            call(["rm", "-rf", down_staged, down_held]),
        ]
        machine.adownload.assert_awaited_once_with(down_held, "/local/out.txt")

    def test_upload_directory_packs_locally_and_unpacks_in_container(self, tmp_path):
        container, machine = self._staged()
        (tmp_path / "tree" / "sub").mkdir(parents=True)
        (tmp_path / "tree" / "sub" / "f").write_text("x")
        uploaded = {}

        def record(src, dst):
            with tarfile.open(src) as tar:
                uploaded["names"] = sorted(tar.getnames())

        machine.upload.side_effect = record

        container.upload(str(tmp_path / "tree"), "/work/tree")

        held = machine.upload.call_args.args[1]
        staged = "/staging/" + os.path.basename(held)
        assert uploaded["names"] == [".", "sub", "sub/f"]
        machine.run.assert_called_once_with(
            self._give_argv(held, staged, _UNPACK, "/work/tree")
        )

    @pytest.mark.parametrize("stdout", ["", "INFO: wrapper notice\ndirectory\n"])
    def test_download_directory_unpacks_container_archive_locally(
        self, tmp_path, stdout
    ):
        # The exit status alone marks a directory, whatever stdout holds.
        container, machine = self._staged(_STAGED_DIRECTORY, stdout)
        (tmp_path / "remote").mkdir()
        (tmp_path / "remote" / "f").write_text("from container")
        tree_archive.pack(str(tmp_path / "remote"), str(tmp_path / "remote.tar"))
        machine.download.side_effect = lambda src, dst: shutil.copy(
            tmp_path / "remote.tar", dst
        )

        container.download("/work/tree", str(tmp_path / "local"))

        assert (tmp_path / "local" / "f").read_text() == "from container"
        held = machine.download.call_args.args[0]
        assert held.startswith("/host/")

    async def test_async_directory_copies_match_sync(self, tmp_path):
        machine = MagicMock()
        machine.aupload = AsyncMock()
        # Upload's host command, download's host command, download's cleanup.
        machine.arun = AsyncMock(
            side_effect=[_result(), _result(_STAGED_DIRECTORY), _result()]
        )
        (tmp_path / "remote").mkdir()
        (tmp_path / "remote" / "f").write_text("from container")
        tree_archive.pack(str(tmp_path / "remote"), str(tmp_path / "remote.tar"))

        async def fake_download(src, dst):
            shutil.copy(tmp_path / "remote.tar", dst)

        machine.adownload = AsyncMock(side_effect=fake_download)
        container = _container(machine)
        container._staging_dir, container._host_dir = "/staging", "/host"

        await container.aupload(str(tmp_path / "remote"), "/work/tree")
        await container.adownload("/work/tree", str(tmp_path / "local"))

        held = machine.aupload.await_args.args[1]
        staged = "/staging/" + os.path.basename(held)
        assert machine.arun.await_args_list[0] == call(
            self._give_argv(held, staged, _UNPACK, "/work/tree")
        )
        assert (tmp_path / "local" / "f").read_text() == "from container"


class TestHostScripts:
    """Run the host side of container copies for real, with a shell command
    standing in for the in-container step (which the container controls)."""

    @staticmethod
    def _run(script, first, second, *step):
        return subprocess.run(
            ["sh", "-c", script, "sh", str(first), str(second), *step],
            capture_output=True,
            text=True,
        )

    @pytest.fixture
    def dirs(self, tmp_path):
        staging, host = tmp_path / "staging", tmp_path / "host"
        staging.mkdir()
        host.mkdir()
        secret = tmp_path / "secret"
        secret.write_text("host secret")
        return staging, host, secret

    def test_take_moves_a_staged_file_out_of_reach(self, dirs):
        staging, host, _ = dirs
        staged, held = staging / "f", host / "f"

        result = self._run(
            _HOST_TAKE, staged, held, "sh", "-c", 'echo data > "$0"', str(staged)
        )

        assert result.returncode == 0, result.stderr
        assert held.read_text() == "data\n"
        assert not staged.exists()

    def test_take_passes_the_directory_status_through(self, dirs):
        staging, host, _ = dirs
        staged, held = staging / "f", host / "f"

        result = self._run(
            _HOST_TAKE,
            staged,
            held,
            "sh",
            "-c",
            f'touch "$0"; exit {_STAGED_DIRECTORY}',
            str(staged),
        )

        assert result.returncode == _STAGED_DIRECTORY, result.stderr
        assert held.exists()

    def test_stage_signals_directories_with_its_exit_status(self, tmp_path):
        tree, staged = tmp_path / "tree", tmp_path / "staged"
        tree.mkdir()
        (tree / "f").write_text("x")
        (tmp_path / "file").write_text("y")

        as_dir = subprocess.run(["sh", "-c", _STAGE, "sh", str(tree), str(staged)])
        as_file = subprocess.run(
            ["sh", "-c", _STAGE, "sh", str(tmp_path / "file"), str(tmp_path / "copy")]
        )

        assert as_dir.returncode == _STAGED_DIRECTORY
        assert tarfile.is_tarfile(staged)
        assert as_file.returncode == 0
        assert (tmp_path / "copy").read_text() == "y"

    @pytest.mark.parametrize("swap", ["symlink", "directory", "fifo"])
    def test_take_refuses_a_swapped_staged_file(self, dirs, swap):
        staging, host, secret = dirs
        staged, held = staging / "f", host / "f"
        plant = {
            "symlink": f'ln -s {secret} "$0"',
            "directory": 'mkdir "$0"',
            "fifo": 'mkfifo "$0"',
        }[swap]

        result = self._run(_HOST_TAKE, staged, held, "sh", "-c", plant, str(staged))

        assert result.returncode != 0
        assert "not a regular file" in result.stderr
        # Whatever was planted was moved, never followed.
        if swap == "symlink":
            assert held.is_symlink()
        assert secret.read_text() == "host secret"

    def test_take_stops_when_the_step_fails(self, dirs):
        staging, host, _ = dirs

        result = self._run(_HOST_TAKE, staging / "f", host / "f", "false")

        assert result.returncode != 0
        assert not (host / "f").exists()

    def test_give_replaces_a_planted_directory_symlink(self, dirs):
        staging, host, secret = dirs
        staged, held = staging / "f", host / "f"
        held.write_text("upload")
        target = secret.parent / "host-dir"
        target.mkdir()
        os.symlink(target, staged)

        result = self._run(_HOST_GIVE, held, staged, "true")

        assert result.returncode == 0, result.stderr
        assert os.listdir(target) == []
        assert not staged.is_symlink() and staged.read_text() == "upload"

    def test_rename_uses_bsd_flag_without_gnu_mv(self, dirs, tmp_path):
        staging, host, _ = dirs
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log = tmp_path / "mv.log"
        fake_mv = bin_dir / "mv"
        # Like BSD mv: no --version; record the arguments of the real call.
        fake_mv.write_text(
            f'#!/bin/sh\n[ "$1" = --version ] && exit 64\necho "$@" > {log}\n'
        )
        fake_mv.chmod(0o755)

        result = subprocess.run(
            ["sh", "-c", _HOST_GIVE, "sh", str(host / "f"), str(staging / "f"), "true"],
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert log.read_text().split() == [
            "-f",
            "-h",
            "--",
            *map(str, (host / "f", staging / "f")),
        ]

    def test_give_replaces_a_planted_symlink_without_following_it(self, dirs):
        staging, host, secret = dirs
        staged, held = staging / "f", host / "f"
        held.write_text("upload")
        os.symlink(secret, staged)  # The container guessed or raced the name.

        result = self._run(
            _HOST_GIVE, held, staged, "sh", "-c", 'cat "$0"', str(staged)
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == "upload"
        assert secret.read_text() == "host secret"
        assert not staged.is_symlink()
