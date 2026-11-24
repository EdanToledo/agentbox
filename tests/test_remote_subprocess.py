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

"""Tests for agentbox.remote.subprocess."""

from __future__ import annotations

import asyncio
import inspect
import os
import queue
import signal
import subprocess as _subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _endpoints import endpoint_method
from _mock_streaming import MockPort, MockPortReceiver
from agentbox.remote.streaming import (
    ControlSignal,
    StreamControlReady,
    StreamData,
    StreamExit,
    StreamStdinReady,
    StreamType,
)
from agentbox.remote.subprocess import (
    _arun,
    _arun_command_impl,
    _PopenStreamReader,
    _PopenStreamWriter,
    _run,
    _run_command_impl,
    _SubprocessActor,
    CalledProcessError,
    CompletedProcess,
    DEVNULL,
    PIPE,
    Popen,
    STDOUT,
    TimeoutExpired,
)


def _mock_proc_mesh(result_dict: dict) -> MagicMock:
    """Build a mock ProcMesh whose spawn -> choose -> get returns *result_dict*."""
    mock_future = MagicMock()
    mock_future.get.return_value = result_dict

    mock_endpoint = MagicMock()
    mock_endpoint.choose.return_value = mock_future

    mock_actor_mesh = MagicMock()
    mock_actor_mesh.run_command = mock_endpoint

    mock_mesh = MagicMock()
    mock_mesh.spawn.return_value = mock_actor_mesh
    return mock_mesh


def _mock_async_proc_mesh(future: asyncio.Future) -> tuple[MagicMock, asyncio.Event]:
    """Build a mock ProcMesh with an awaitable command and actor stop future."""
    stopped = asyncio.Event()

    async def stop_actor() -> None:
        stopped.set()

    mock_actor_mesh = MagicMock()
    mock_actor_mesh.run_command.choose.return_value = future
    mock_actor_mesh.stop.side_effect = stop_actor

    mock_mesh = MagicMock()
    mock_mesh.spawn.return_value = mock_actor_mesh
    return mock_mesh, stopped


def _setup_popen_stream_attrs(proc: Popen) -> None:
    """Add stream queue/attribute fields to a Popen created via __new__."""
    if not hasattr(proc, "_encoding"):
        proc._encoding = "utf-8" if proc._text else None
    if not hasattr(proc, "_errors"):
        proc._errors = "strict"
    encoding = proc._encoding
    errors = proc._errors
    if proc._stdout_piped:
        proc._stdout_queue = queue.Queue()
        proc.stdout = _PopenStreamReader(
            proc._stdout_queue, encoding=encoding, errors=errors
        )
    else:
        proc._stdout_queue = None
        proc.stdout = None
    if proc._stderr_piped:
        proc._stderr_queue = queue.Queue()
        proc.stderr = _PopenStreamReader(
            proc._stderr_queue, encoding=encoding, errors=errors
        )
    else:
        proc._stderr_queue = None
        proc.stderr = None
    if proc._stdin_enabled:
        proc.stdin = _PopenStreamWriter(
            proc._stdin_ready,
            lambda: proc._stdin_port,
            encoding=encoding,
            errors=errors,
        )
    else:
        proc.stdin = None


def _wait_until_dead(pid: int, timeout: float = 5.0) -> bool:
    """Return whether *pid* exits (or is a zombie) within *timeout* seconds.

    A killed grandchild is reaped by init, not by us, so it can linger as a
    zombie that ``os.kill(pid, 0)`` still finds; a zombie is already dead.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        state = _subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _run_command_args(mesh: MagicMock) -> dict[str, Any]:
    """Name the arguments the last run_command.choose() call sent the actor."""
    call = mesh.spawn.return_value.run_command.choose.call_args
    signature = inspect.signature(endpoint_method(_SubprocessActor, "run_command"))
    return signature.bind(None, *call.args, **call.kwargs).arguments


def _make_popen_with_messages(
    messages: list[Any],
    stdin: int | None = None,
    stdout: int | None = PIPE,
    stderr: int | None = PIPE,
    text: bool = True,
    encoding: str | None = None,
    errors: str | None = None,
    receiver: MockPortReceiver | None = None,
) -> Popen:
    """Create a Popen with a mocked streaming infrastructure and pre-loaded messages.

    Pass *receiver* to push more messages after the drain thread has started.
    """
    if receiver is None:
        receiver = MockPortReceiver()
    for msg in messages:
        receiver.push(msg)

    proc = Popen.__new__(Popen)
    proc.args = ["test_cmd"]
    proc._text = text
    proc._returncode = None
    proc._stdin_enabled = stdin == PIPE
    proc._stdout_piped = stdout == PIPE
    proc._stderr_piped = stderr == PIPE or stderr == STDOUT
    proc._exited = False
    proc._input_sent = False
    proc._communicated = False
    proc._streaming = False
    proc._stdin_port = None
    proc._control_port = None
    proc._stdin_ready = threading.Event()
    proc._control_ready = threading.Event()
    proc._drain_complete = threading.Event()
    proc._stdout_parts = []
    proc._stderr_parts = []
    proc._stdout_cached = None
    proc._stderr_cached = None
    proc._output_receiver = receiver
    proc._actor_mesh = MagicMock()
    if encoding is not None:
        proc._encoding = encoding
    if errors is not None:
        proc._errors = errors

    # Set up stream queues and file-like attributes
    _setup_popen_stream_attrs(proc)

    # Start the drain thread
    proc._drain_thread = threading.Thread(target=proc._drain_messages, daemon=True)
    proc._drain_thread.start()

    return proc


class TestConstants:
    """Verify PIPE, STDOUT, DEVNULL match the standard subprocess module."""

    def test_pipe_matches_stdlib(self):
        assert PIPE == _subprocess.PIPE

    def test_stdout_matches_stdlib(self):
        assert STDOUT == _subprocess.STDOUT

    def test_devnull_matches_stdlib(self):
        assert DEVNULL == _subprocess.DEVNULL


class TestCompletedProcess:
    def test_check_returncode_zero(self):
        cp = CompletedProcess(args=["echo"], returncode=0, stdout="hi\n", stderr="")
        cp.check_returncode()  # should not raise

    def test_check_returncode_nonzero(self):
        cp = CompletedProcess(args="exit 1", returncode=1, stdout="", stderr="fail")
        with pytest.raises(CalledProcessError) as exc_info:
            cp.check_returncode()
        assert exc_info.value.returncode == 1
        assert exc_info.value.cmd == "exit 1"
        assert exc_info.value.stderr == "fail"

    def test_binary_output(self):
        cp = CompletedProcess(
            args=["cat"], returncode=1, stdout=b"output", stderr=b"failure"
        )

        assert cp.stdout == b"output"
        with pytest.raises(CalledProcessError) as exc_info:
            cp.check_returncode()
        assert exc_info.value.stdout == b"output"
        assert exc_info.value.stderr == b"failure"


class TestRun:
    def test_simple_command(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": "hello\n",
                "stderr": "",
            }
        )
        result = _run(
            ["echo", "hello"], proc_mesh=mesh, task_pool=None, capture_output=True
        )
        assert result.returncode == 0
        assert result.stdout == "hello\n"
        assert result.args == ["echo", "hello"]

    def test_check_raises_on_failure(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 1,
                "stdout": "",
                "stderr": "error",
            }
        )
        with pytest.raises(CalledProcessError) as exc_info:
            _run(["false"], proc_mesh=mesh, task_pool=None, check=True)
        assert exc_info.value.returncode == 1

    def test_check_false_no_raise(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 1,
                "stdout": "",
                "stderr": "error",
            }
        )
        result = _run(["false"], proc_mesh=mesh, task_pool=None, check=False)
        assert result.returncode == 1

    def test_timeout_expired(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": True,
                "cmd": ["sleep", "100"],
                "timeout": 1.0,
                "stdout": None,
                "stderr": None,
            }
        )
        with pytest.raises(TimeoutExpired) as exc_info:
            _run(["sleep", "100"], proc_mesh=mesh, task_pool=None, timeout=1.0)
        assert exc_info.value.timeout == 1.0

    def test_shell_mode(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": "ok",
                "stderr": "",
            }
        )
        result = _run(
            "echo ok", proc_mesh=mesh, task_pool=None, shell=True, capture_output=True
        )
        assert result.stdout == "ok"

    def test_spawn_called_with_actor(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": None,
                "stderr": None,
            }
        )
        _run(["true"], proc_mesh=mesh, task_pool=None)
        mesh.spawn.assert_called_once()
        _, actor_class = mesh.spawn.call_args[0]
        assert actor_class is _SubprocessActor

    def test_stops_actor_mesh_on_success(self):
        mesh = _mock_proc_mesh(
            {"timeout_expired": False, "returncode": 0, "stdout": None, "stderr": None}
        )

        _run(["true"], proc_mesh=mesh, task_pool=None)

        actor_mesh = mesh.spawn.return_value
        actor_mesh.stop.assert_called_once_with()
        actor_mesh.stop.return_value.get.assert_called_once_with(timeout=10.0)

    def test_stops_actor_mesh_on_timeout(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": True,
                "cmd": ["sleep", "100"],
                "timeout": 1.0,
                "stdout": None,
                "stderr": None,
            }
        )

        with pytest.raises(TimeoutExpired):
            _run(["sleep", "100"], proc_mesh=mesh, task_pool=None, timeout=1.0)

        mesh.spawn.return_value.stop.assert_called_once_with()

    def test_stops_actor_mesh_on_endpoint_error(self):
        mesh = _mock_proc_mesh({})
        mesh.spawn.return_value.run_command.choose.return_value.get.side_effect = (
            RuntimeError("endpoint failed")
        )

        with pytest.raises(RuntimeError, match="endpoint failed"):
            _run(["true"], proc_mesh=mesh, task_pool=None)

        mesh.spawn.return_value.stop.assert_called_once_with()

    def test_stops_actor_mesh_when_dispatch_raises(self):
        mesh = _mock_proc_mesh({})
        mesh.spawn.return_value.run_command.choose.side_effect = RuntimeError(
            "dispatch failed"
        )

        with pytest.raises(RuntimeError, match="dispatch failed"):
            _run(["true"], proc_mesh=mesh, task_pool=None)

        mesh.spawn.return_value.stop.assert_called_once_with()

    def test_dispatches_through_supplied_task_pool(self):
        proc_mesh = MagicMock()
        task_pool = MagicMock()
        task_pool.submit.return_value = {
            "timeout_expired": False,
            "returncode": 0,
            "stdout": None,
            "stderr": None,
        }

        _run(["true"], proc_mesh=proc_mesh, task_pool=task_pool)

        proc_mesh.spawn.assert_not_called()
        (task,) = task_pool.submit.call_args.args
        assert task.kind == "subprocess"
        assert task.to_payload()[0] == ["true"]

    def test_env_and_cwd_passed_through(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": None,
                "stderr": None,
            }
        )
        _run(["ls"], proc_mesh=mesh, task_pool=None, cwd="/tmp", env={"FOO": "bar"})
        sent = _run_command_args(mesh)
        assert sent["cwd"] == "/tmp"
        assert sent["env"] == {"FOO": "bar"}

    def test_input_passed_through(self):
        """_run() passes input to the actor endpoint."""
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": "echoed",
                "stderr": "",
            }
        )
        result = _run(
            ["cat"], proc_mesh=mesh, task_pool=None, input="hello", capture_output=True
        )
        assert _run_command_args(mesh)["input"] == "hello"
        assert result.stdout == "echoed"

    def test_stdout_pipe(self):
        """_run() passes stdout=PIPE to the actor."""
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": "data",
                "stderr": None,
            }
        )
        result = _run(["echo", "hi"], proc_mesh=mesh, task_pool=None, stdout=PIPE)
        assert _run_command_args(mesh)["stdout"] == PIPE
        assert result.stdout == "data"

    def test_stderr_pipe(self):
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": None,
                "stderr": "warn",
            }
        )
        result = _run(["cmd"], proc_mesh=mesh, task_pool=None, stderr=PIPE)
        assert _run_command_args(mesh)["stderr"] == PIPE
        assert result.stderr == "warn"

    def test_stderr_stdout_redirect(self):
        """stderr=STDOUT merges stderr into stdout."""
        mesh = _mock_proc_mesh(
            {
                "timeout_expired": False,
                "returncode": 0,
                "stdout": "merged",
                "stderr": None,
            }
        )
        result = _run(
            ["cmd"], proc_mesh=mesh, task_pool=None, stdout=PIPE, stderr=STDOUT
        )
        sent = _run_command_args(mesh)
        assert sent["stdout"] == PIPE
        assert sent["stderr"] == STDOUT
        assert result.stdout == "merged"

    def test_stdin_devnull(self):
        mesh = _mock_proc_mesh(
            {"timeout_expired": False, "returncode": 0, "stdout": None, "stderr": None}
        )
        _run(["cmd"], proc_mesh=mesh, task_pool=None, stdin=DEVNULL)
        assert _run_command_args(mesh)["stdin"] == DEVNULL

    def test_stdout_devnull(self):
        mesh = _mock_proc_mesh(
            {"timeout_expired": False, "returncode": 0, "stdout": None, "stderr": None}
        )
        _run(["cmd"], proc_mesh=mesh, task_pool=None, stdout=DEVNULL)
        assert _run_command_args(mesh)["stdout"] == DEVNULL


class TestARun:
    async def test_stops_actor_mesh_on_success(self):
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {"timeout_expired": False, "returncode": 0, "stdout": None, "stderr": None}
        )
        mesh, stopped = _mock_async_proc_mesh(future)

        result = await _arun(["true"], proc_mesh=mesh, task_pool=None)

        assert result.returncode == 0
        assert stopped.is_set()
        mesh.spawn.return_value.stop.assert_called_once_with()

    async def test_stops_actor_mesh_on_timeout(self):
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "timeout_expired": True,
                "cmd": ["sleep", "100"],
                "timeout": 1.0,
                "stdout": None,
                "stderr": None,
            }
        )
        mesh, stopped = _mock_async_proc_mesh(future)

        with pytest.raises(TimeoutExpired):
            await _arun(["sleep", "100"], proc_mesh=mesh, task_pool=None, timeout=1.0)

        assert stopped.is_set()

    async def test_stops_actor_mesh_on_endpoint_error(self):
        future = asyncio.get_running_loop().create_future()
        future.set_exception(RuntimeError("endpoint failed"))
        mesh, stopped = _mock_async_proc_mesh(future)

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await _arun(["true"], proc_mesh=mesh, task_pool=None)

        assert stopped.is_set()

    async def test_stops_actor_mesh_when_dispatch_raises(self):
        future = asyncio.get_running_loop().create_future()
        proc_mesh, stopped = _mock_async_proc_mesh(future)
        proc_mesh.spawn.return_value.run_command.choose.side_effect = RuntimeError(
            "dispatch failed"
        )

        with pytest.raises(RuntimeError, match="dispatch failed"):
            await _arun(["true"], proc_mesh=proc_mesh, task_pool=None)

        assert stopped.is_set()

    async def test_stops_actor_mesh_on_cancellation(self):
        future = asyncio.get_running_loop().create_future()
        mesh, stopped = _mock_async_proc_mesh(future)
        task = asyncio.create_task(_arun(["true"], proc_mesh=mesh, task_pool=None))
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert stopped.is_set()

    async def test_dispatches_through_supplied_task_pool(self):
        task_pool = MagicMock()
        task_pool.asubmit = AsyncMock(
            return_value=(
                {
                    "timeout_expired": False,
                    "returncode": 0,
                    "stdout": None,
                    "stderr": None,
                }
            )
        )
        proc_mesh = MagicMock()

        await _arun(["true"], proc_mesh=proc_mesh, task_pool=task_pool)

        proc_mesh.spawn.assert_not_called()
        (task,) = task_pool.asubmit.await_args.args
        assert task.kind == "subprocess"
        assert task.to_payload()[0] == ["true"]


class TestPopen:
    @pytest.mark.parametrize("method", ["terminate", "kill"])
    def test_signal_timeout_preserves_a_running_process(self, method):
        control_port = MockPort()
        proc = _make_popen_with_messages([])
        try:
            with pytest.raises(TimeoutExpired):
                getattr(proc, method)(timeout=0.01)
            assert proc.poll() is None
            assert control_port.sent == []
            proc._output_receiver.push(StreamControlReady(control_port=control_port))
            getattr(proc, method)(timeout=1.0)
            expected = signal.SIGTERM if method == "terminate" else signal.SIGKILL
            assert [msg.signal_num for msg in control_port.sent] == [expected]
        finally:
            proc._output_receiver.push(StreamExit(returncode=0))
            proc.wait(timeout=1.0)
            proc._drain_thread.join(timeout=1.0)

    def test_signal_is_noop_when_control_channel_closes_before_readiness(self):
        proc = _make_popen_with_messages([])
        try:
            proc._control_ready.set()
            proc.terminate(timeout=0.1)
            assert proc.poll() is None
            assert proc._control_port is None
        finally:
            proc._output_receiver.push(StreamExit(returncode=0))
            proc.wait(timeout=1.0)
            proc._drain_thread.join(timeout=1.0)

    @pytest.mark.parametrize("payload", [b"result", b"\xc3", b""])
    @pytest.mark.parametrize("text", [False, True])
    @pytest.mark.parametrize(
        "method,size", [("read", -1), ("read", 32), ("readline", -1)]
    )
    def test_closing_stdout_releases_a_blocked_reader(
        self, monkeypatch, payload, text, method, size
    ):
        proc = _make_popen_with_messages([], text=text)
        waiting = threading.Event()
        get = proc._stdout_queue.get

        def read_chunk():
            waiting.set()
            return get(timeout=2.0)

        monkeypatch.setattr(proc._stdout_queue, "get", read_chunk)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(getattr(proc.stdout, method), size)
            try:
                assert waiting.wait(timeout=1.0)
                proc._stdout_queue.put(payload)
                proc.stdout.close()
                if text and payload == b"\xc3":
                    with pytest.raises(UnicodeDecodeError):
                        result.result(timeout=1.0)
                else:
                    expected = payload.decode() if text else payload
                    assert result.result(timeout=1.0) == expected
                for read_method in (proc.stdout.read, proc.stdout.readline):
                    with pytest.raises(ValueError, match="closed file"):
                        read_method()
            finally:
                proc.stdout.close()
                proc._output_receiver.push(StreamExit(returncode=0))
                proc.wait(timeout=1.0)
                proc._drain_thread.join(timeout=1.0)

    def test_closing_stdout_preserves_communicate_output(self):
        proc = _make_popen_with_messages([])
        proc.stdout.close()
        proc._output_receiver.push(StreamData(stream=StreamType.STDOUT, data=b"result"))
        proc._output_receiver.push(StreamExit(returncode=0))
        assert proc.communicate(timeout=1.0) == ("result", "")
        with pytest.raises(ValueError):
            proc.stdout.read()
        proc._drain_thread.join(timeout=1.0)

    def test_communicate(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"output"),
            StreamData(stream=StreamType.STDERR, data=b""),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        stdout, stderr = proc.communicate()
        assert stdout == "output"
        assert stderr == ""
        assert proc.returncode == 0

    def test_communicate_idempotent(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"output"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        proc.communicate()
        stdout, stderr = proc.communicate()
        assert stdout == "output"

    def test_wait(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=42),
        ]
        proc = _make_popen_with_messages(messages)
        rc = proc.wait()
        assert rc == 42

    def test_pid_is_none(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.pid is None

    def test_communicate_timeout(self):
        """communicate() raises TimeoutExpired when process doesn't finish in time."""
        # Don't send StreamExit — process never finishes
        control_port = MockPort()
        receiver = MockPortReceiver()
        receiver.push(StreamControlReady(control_port=control_port))
        # Don't push StreamExit — drain will block

        proc = _make_popen_with_messages([], receiver=receiver)

        with pytest.raises(TimeoutExpired) as exc_info:
            proc.communicate(timeout=0.1)
        assert exc_info.value.timeout == 0.1

    def test_communicate_with_input(self):
        """communicate(input=...) sends data to stdin."""
        mock_stdin_port = MockPort()
        control_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"echoed"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdin=PIPE)
        stdout, stderr = proc.communicate(input="hello")
        assert stdout == "echoed"
        # Verify stdin data was sent
        from agentbox.remote.streaming import StdinClose, StdinData

        assert len(mock_stdin_port.sent) == 2
        assert isinstance(mock_stdin_port.sent[0], StdinData)
        assert mock_stdin_port.sent[0].data == b"hello"
        assert isinstance(mock_stdin_port.sent[1], StdinClose)

    def test_kill(self):
        """kill() sends SIGKILL via the control port."""
        control_port = MockPort()
        receiver = MockPortReceiver()
        # Push control ready but NOT StreamExit — process is still running
        receiver.push(StreamControlReady(control_port=control_port))

        proc = _make_popen_with_messages([], receiver=receiver)

        proc._control_ready.wait(timeout=2)
        proc.kill()
        assert len(control_port.sent) == 1
        assert isinstance(control_port.sent[0], ControlSignal)
        assert control_port.sent[0].signal_num == signal.SIGKILL

        # Now let the process "die"
        receiver.push(StreamExit(returncode=-9))
        proc.wait()

    def test_terminate(self):
        """terminate() sends SIGTERM via the control port."""
        control_port = MockPort()
        receiver = MockPortReceiver()
        receiver.push(StreamControlReady(control_port=control_port))

        proc = _make_popen_with_messages([], receiver=receiver)

        proc._control_ready.wait(timeout=2)
        proc.terminate()
        assert len(control_port.sent) == 1
        assert isinstance(control_port.sent[0], ControlSignal)
        assert control_port.sent[0].signal_num == signal.SIGTERM

        receiver.push(StreamExit(returncode=-15))
        proc.wait()

    def test_send_signal(self):
        """send_signal() sends the given signal via the control port."""
        control_port = MockPort()
        receiver = MockPortReceiver()
        receiver.push(StreamControlReady(control_port=control_port))

        proc = _make_popen_with_messages([], receiver=receiver)

        proc._control_ready.wait(timeout=2)
        proc.send_signal(signal.SIGUSR1)
        assert control_port.sent[0].signal_num == signal.SIGUSR1

        receiver.push(StreamExit(returncode=0))
        proc.wait()

    def test_send_signal_noop_after_exit(self):
        """send_signal() does nothing if process already exited."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        proc.wait()  # drain fully
        proc.send_signal(signal.SIGTERM)
        assert len(control_port.sent) == 0

    def test_context_manager(self):
        """Popen can be used as a context manager."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        with _make_popen_with_messages(messages) as proc:
            stdout, stderr = proc.communicate()
        assert stdout == "hello"
        assert proc.returncode == 0


class TestPopenStdioParams:
    def test_stdout_pipe_captures(self):
        """stdout=PIPE captures output."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        assert stdout == "hello"
        assert stderr == ""

    def test_stdout_none_returns_none(self):
        """stdout=None means communicate returns None for stdout."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=None, stderr=PIPE)
        stdout, stderr = proc.communicate()
        assert stdout is None
        assert stderr == ""

    def test_stderr_none_returns_none(self):
        """stderr=None means communicate returns None for stderr."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=PIPE, stderr=None)
        stdout, stderr = proc.communicate()
        assert stdout == "data"
        assert stderr is None

    def test_both_none_returns_none(self):
        """No pipes → communicate returns (None, None)."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=None, stderr=None)
        stdout, stderr = proc.communicate()
        assert stdout is None
        assert stderr is None

    def test_stderr_stdout_treated_as_piped(self):
        """stderr=STDOUT is treated as piped (merged into stdout)."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"merged"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=PIPE, stderr=STDOUT)
        stdout, stderr = proc.communicate()
        assert stdout == "merged"
        # stderr=STDOUT means stderr data is merged into stdout,
        # but the "stderr" return value still reflects what was captured
        assert stderr == ""

    def test_communicate_idempotent_with_none_stdout(self):
        """communicate() is idempotent even when stdout is None."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=None, stderr=None)
        proc.communicate()
        stdout, stderr = proc.communicate()
        assert stdout is None
        assert stderr is None


class TestRunCommandImpl:
    """Test _run_command_impl directly (in-process, no Monarch)."""

    def _run(self, **kwargs):
        defaults = {
            "args": ["echo", "hello"],
            "shell": False,
            "cwd": None,
            "env": None,
            "capture_output": True,
            "timeout": None,
            "text": True,
        }
        defaults.update(kwargs)
        return _run_command_impl(**defaults)

    def test_echo(self):
        result = self._run()
        assert result["timeout_expired"] is False
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_shell_command(self):
        result = self._run(args="echo shell_test", shell=True)
        assert result["returncode"] == 0
        assert "shell_test" in result["stdout"]

    def test_string_without_shell_splits(self):
        result = self._run(args="echo split_test", shell=False)
        assert result["returncode"] == 0
        assert "split_test" in result["stdout"]

    def test_nonzero_exit(self):
        result = self._run(args="exit 2", shell=True)
        assert result["returncode"] == 2

    def test_timeout(self):
        result = self._run(args=["sleep", "10"], timeout=0.1, capture_output=False)
        assert result["timeout_expired"] is True

    def test_cwd(self, tmp_path):
        result = self._run(args=["pwd"], cwd=str(tmp_path))
        assert str(tmp_path) in result["stdout"]

    def test_env(self):
        import os

        env = {**os.environ, "AGENTBOX_TEST_VAR": "yes"}
        result = self._run(args="echo $AGENTBOX_TEST_VAR", shell=True, env=env)
        assert "yes" in result["stdout"]

    def test_input(self):
        """input= passes data to the process's stdin."""
        result = self._run(args=["cat"], input="hello from stdin", capture_output=True)
        assert result["returncode"] == 0
        assert "hello from stdin" in result["stdout"]

    def test_input_bytes(self):
        """input= works with bytes."""
        result = self._run(
            args=["cat"], input=b"byte data", capture_output=True, text=False
        )
        assert result["returncode"] == 0
        assert b"byte data" in result["stdout"]


class TestAsyncRunCommandImpl:
    """Compare the async worker primitive with the synchronous reference."""

    @staticmethod
    def _kwargs(**overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "args": [sys.executable, "-c", "print('hello')"],
            "shell": False,
            "cwd": None,
            "env": None,
            "capture_output": True,
            "timeout": None,
            "text": True,
        }
        kwargs.update(overrides)
        return kwargs

    @pytest.mark.parametrize(
        "overrides",
        [
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)",
                ]
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'bytes')",
                ],
                "text": False,
            },
            {"args": "printf shell", "shell": True},
            {"args": ["printf", "ignored"], "shell": True},
            {"args": "printf split", "shell": False},
            {"args": [sys.executable, "-c", "raise SystemExit(7)"]},
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.write(sys.stdin.read())",
                ],
                "input": "text input",
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
                ],
                "input": b"binary input",
                "text": False,
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)",
                ],
                "capture_output": False,
                "stdout": PIPE,
                "stderr": STDOUT,
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'caf\\xe9')",
                ],
                "encoding": "latin-1",
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'\\xff')",
                ],
                "encoding": "utf-8",
                "errors": "replace",
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'a\\r\\nb\\r')",
                ]
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.write(sys.stdin.read())",
                ],
                "capture_output": False,
                "stdin": DEVNULL,
                "stdout": PIPE,
            },
            {
                "args": [
                    sys.executable,
                    "-c",
                    "import sys,time; print('partial', flush=True); time.sleep(1)",
                ],
                "timeout": 0.05,
            },
        ],
        ids=[
            "text-output",
            "binary-output",
            "shell",
            "shell-list",
            "split-string",
            "nonzero",
            "text-input",
            "binary-input",
            "stderr-to-stdout",
            "encoding",
            "decode-errors",
            "newline-translation",
            "stdin-devnull",
            "timeout-partial-output",
        ],
    )
    async def test_matches_sync_result(self, overrides: dict[str, Any]) -> None:
        kwargs = self._kwargs(**overrides)
        sync_result = await asyncio.to_thread(_run_command_impl, **kwargs)
        async_result = await _arun_command_impl(**kwargs)

        if kwargs["timeout"] is None:
            assert async_result == sync_result
        else:
            # Whether the child reaches its first flushed write before a very
            # short deadline is scheduler-dependent. Both implementations must
            # preserve any output they did observe as bytes.
            for result in (sync_result, async_result):
                assert result["timeout_expired"] is True
                assert result["stdout"] in (None, b"partial\n")
                assert result["stderr"] is None
                assert result["cmd"] == kwargs["args"]
                assert result["timeout"] == kwargs["timeout"]

    async def test_matches_sync_cwd_and_environment(self, tmp_path) -> None:
        env = {**os.environ, "AGENTBOX_ASYNC_PARITY": "present"}
        kwargs = self._kwargs(
            args=[
                sys.executable,
                "-c",
                "import os; print(os.getcwd()); print(os.environ['AGENTBOX_ASYNC_PARITY'])",
            ],
            cwd=str(tmp_path),
            env=env,
        )

        sync_result = await asyncio.to_thread(_run_command_impl, **kwargs)
        async_result = await _arun_command_impl(**kwargs)

        assert async_result == sync_result

    async def test_matches_missing_executable_error(self, tmp_path) -> None:
        kwargs = self._kwargs(args=[str(tmp_path / "does-not-exist")])

        with pytest.raises(FileNotFoundError) as sync_error:
            await asyncio.to_thread(_run_command_impl, **kwargs)
        with pytest.raises(FileNotFoundError) as async_error:
            await _arun_command_impl(**kwargs)

        assert async_error.value.filename == sync_error.value.filename

    @pytest.mark.parametrize(
        ("text", "input_value", "error_type"),
        [(True, b"bytes", AttributeError), (False, "text", TypeError)],
    )
    async def test_matches_input_type_error(
        self,
        text: bool,
        input_value: str | bytes,
        error_type: type[Exception],
    ) -> None:
        kwargs = self._kwargs(
            args=[sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
            text=text,
            input=input_value,
        )

        with pytest.raises(error_type):
            await asyncio.to_thread(_run_command_impl, **kwargs)
        with pytest.raises(error_type):
            await _arun_command_impl(**kwargs)

    @pytest.mark.parametrize("stream", ["stdout", "stderr"])
    async def test_matches_capture_output_validation(self, stream: str) -> None:
        kwargs = self._kwargs(**{stream: PIPE})

        with pytest.raises(ValueError, match="capture_output"):
            await asyncio.to_thread(_run_command_impl, **kwargs)
        with pytest.raises(ValueError, match="capture_output"):
            await _arun_command_impl(**kwargs)

    async def test_commands_overlap_on_one_event_loop(self, tmp_path) -> None:
        script = """
import pathlib
import sys
import time

ready = pathlib.Path(sys.argv[1])
ready.write_text("ready")
deadline = time.monotonic() + 30
while len(list(ready.parent.glob("ready-*"))) < 3:
    if time.monotonic() >= deadline:
        raise SystemExit(9)
    time.sleep(0.01)
"""
        commands = [
            _arun_command_impl(
                **self._kwargs(
                    args=[
                        sys.executable,
                        "-c",
                        script,
                        str(tmp_path / f"ready-{index}"),
                    ]
                )
            )
            for index in range(3)
        ]

        results = await asyncio.gather(*commands)

        assert [result["returncode"] for result in results] == [0, 0, 0]

    @pytest.mark.skipif(os.name != "posix", reason="uses POSIX pid probing")
    async def test_cancellation_kills_and_reaps_child(self, tmp_path) -> None:
        pid_file = tmp_path / "pid"
        script = """
import os
import pathlib
import sys
import time

pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(30)
"""
        task = asyncio.create_task(
            _arun_command_impl(
                **self._kwargs(args=[sys.executable, "-c", script, str(pid_file)])
            )
        )
        for _ in range(200):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_file.exists()
        pid = int(pid_file.read_text())

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    @pytest.mark.skipif(os.name != "posix", reason="uses POSIX process groups")
    async def test_timeout_kills_command_forked_by_shell(self, tmp_path) -> None:
        # /bin/sh may fork the real command; killing only the shell left it
        # holding the pipes, so a timeout waited for the command to finish.
        pid_file = tmp_path / "pid"
        started = time.monotonic()
        result = await _arun_command_impl(
            **self._kwargs(
                args=["/bin/sh", "-c", f"sleep 30 & echo $! > {pid_file}; wait"],
                timeout=1.0,
            )
        )
        assert result["timeout_expired"]
        assert time.monotonic() - started < 10
        assert _wait_until_dead(int(pid_file.read_text()))

    @pytest.mark.skipif(os.name != "posix", reason="uses POSIX pid probing")
    async def test_actor_endpoint_cancellation_kills_child(self, tmp_path) -> None:
        # Stopping the actor cancels run_command. A blocking endpoint could
        # not be cancelled, so its command was orphaned on lease release.
        pid_file = tmp_path / "pid"
        script = (
            "import os, pathlib, sys, time; "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
        )
        run_command = endpoint_method(_SubprocessActor, "run_command")
        kwargs = self._kwargs(args=[sys.executable, "-c", script, str(pid_file)])
        task = asyncio.ensure_future(run_command(None, **kwargs))
        for _ in range(200):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_file.exists()
        pid = int(pid_file.read_text())

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class TestStdlibDocExamples:
    """Tests that mirror the examples from the Python subprocess docs,
    adapted for the AgentBox remote subprocess API.
    """

    def test_popen_communicate_with_timeout_and_kill(self):
        """Mirrors the stdlib pattern:
        proc = subprocess.Popen(...)
        try:
            outs, errs = proc.communicate(timeout=15)
        except TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
        """
        control_port = MockPort()
        receiver = MockPortReceiver()
        receiver.push(StreamControlReady(control_port=control_port))

        proc = _make_popen_with_messages([], receiver=receiver)

        try:
            outs, errs = proc.communicate(timeout=0.1)
        except TimeoutExpired:
            proc.kill()
            # Simulate process dying after kill
            receiver.push(StreamData(stream=StreamType.STDOUT, data=b"partial"))
            receiver.push(StreamExit(returncode=-9))
            outs, errs = proc.communicate()

        assert proc.returncode == -9
        assert outs == "partial"

    def test_called_process_error_attributes(self):
        """Mirrors: CalledProcessError attributes — returncode, cmd, output, stdout, stderr."""
        err = CalledProcessError(
            returncode=1, cmd=["bad_cmd"], output="out", stderr="err"
        )
        assert err.returncode == 1
        assert err.cmd == ["bad_cmd"]
        assert err.output == "out"
        assert err.stdout == "out"  # alias
        assert err.stderr == "err"

    def test_timeout_expired_attributes(self):
        """Mirrors: TimeoutExpired attributes — cmd, timeout, output, stdout, stderr."""
        err = TimeoutExpired(
            cmd=["slow_cmd"], timeout=5.0, output="partial", stderr="warn"
        )
        assert err.cmd == ["slow_cmd"]
        assert err.timeout == 5.0
        assert err.output == "partial"
        assert err.stdout == "partial"  # alias
        assert err.stderr == "warn"


class TestRunCommandImplStdio:
    """Integration tests for stdin/stdout/stderr support in _run_command_impl."""

    def test_stdout_pipe(self):
        result = _run_command_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            capture_output=False,
            timeout=5,
            text=True,
            stdout=PIPE,
        )
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_stderr_pipe(self):
        result = _run_command_impl(
            args="echo err >&2",
            shell=True,
            cwd=None,
            env=None,
            capture_output=False,
            timeout=5,
            text=True,
            stderr=PIPE,
        )
        assert result["returncode"] == 0
        assert "err" in result["stderr"]

    def test_stderr_stdout_redirect(self):
        """stderr=STDOUT merges stderr into stdout pipe."""
        result = _run_command_impl(
            args="echo out && echo err >&2",
            shell=True,
            cwd=None,
            env=None,
            capture_output=False,
            timeout=5,
            text=True,
            stdout=PIPE,
            stderr=STDOUT,
        )
        assert result["returncode"] == 0
        assert "out" in result["stdout"]
        assert "err" in result["stdout"]
        assert result["stderr"] is None

    def test_stdout_devnull(self):
        result = _run_command_impl(
            args=["echo", "suppressed"],
            shell=False,
            cwd=None,
            env=None,
            capture_output=False,
            timeout=5,
            text=True,
            stdout=DEVNULL,
        )
        assert result["returncode"] == 0
        assert result["stdout"] is None

    def test_stdin_devnull(self):
        """stdin=DEVNULL provides empty stdin to the process."""
        result = _run_command_impl(
            args=["cat"],
            shell=False,
            cwd=None,
            env=None,
            capture_output=False,
            timeout=5,
            text=True,
            stdin=DEVNULL,
            stdout=PIPE,
        )
        assert result["returncode"] == 0
        assert result["stdout"] == ""

    def test_capture_output_conflicts_with_stdout(self):
        """capture_output=True + stdout=PIPE raises ValueError."""
        with pytest.raises(ValueError, match="capture_output"):
            _run_command_impl(
                args=["echo", "x"],
                shell=False,
                cwd=None,
                env=None,
                capture_output=True,
                timeout=5,
                text=True,
                stdout=PIPE,
            )

    def test_capture_output_conflicts_with_stderr(self):
        with pytest.raises(ValueError, match="capture_output"):
            _run_command_impl(
                args=["echo", "x"],
                shell=False,
                cwd=None,
                env=None,
                capture_output=True,
                timeout=5,
                text=True,
                stderr=PIPE,
            )


class TestPopenStreamAttributes:
    """Tests for Popen.stdin, .stdout, .stderr file-like stream attributes."""

    def test_stdout_read_returns_full_output(self):
        """proc.stdout.read() returns all output."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"hello "),
            StreamData(stream=StreamType.STDOUT, data=b"world"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        data = proc.stdout.read()
        assert data == "hello world"

    def test_stdout_iterate_lines(self):
        """for line in proc.stdout iterates lines."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"line1\nline2\nline3\n"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        lines = list(proc.stdout)
        assert lines == ["line1\n", "line2\n", "line3\n"]

    def test_stdout_readline(self):
        """proc.stdout.readline() returns one line."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"first\nsecond\n"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        line1 = proc.stdout.readline()
        assert line1 == "first\n"
        line2 = proc.stdout.readline()
        assert line2 == "second\n"

    def test_stderr_read(self):
        """proc.stderr.read() returns stderr output."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDERR, data=b"error output"),
            StreamExit(returncode=1),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stderr is not None
        data = proc.stderr.read()
        assert data == "error output"

    def test_stdin_write_and_close(self):
        """proc.stdin.write() + proc.stdin.close() sends data and closes."""
        mock_stdin_port = MockPort()
        control_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"echoed"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdin=PIPE)
        assert proc.stdin is not None
        proc.stdin.write(b"data")
        proc.stdin.close()

        from agentbox.remote.streaming import StdinClose, StdinData

        assert len(mock_stdin_port.sent) == 2
        assert isinstance(mock_stdin_port.sent[0], StdinData)
        assert mock_stdin_port.sent[0].data == b"data"
        assert isinstance(mock_stdin_port.sent[1], StdinClose)

    def test_stdout_is_none_when_not_pipe(self):
        """proc.stdout is None when stdout != PIPE."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=None, stderr=PIPE)
        assert proc.stdout is None

    def test_stdin_is_none_when_not_pipe(self):
        """proc.stdin is None when stdin != PIPE."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdin is None

    def test_stderr_is_none_when_not_pipe(self):
        """proc.stderr is None when stderr != PIPE."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdout=PIPE, stderr=None)
        assert proc.stderr is None

    def test_stdout_read_bytes_mode(self):
        """proc.stdout.read() returns bytes when text=False."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"raw bytes"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, text=False)
        assert proc.stdout is not None
        data = proc.stdout.read()
        assert data == b"raw bytes"
        assert isinstance(data, bytes)

    def test_stdout_readable_writable(self):
        """proc.stdout.readable() is True, writable() is False."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        assert proc.stdout.readable() is True
        assert proc.stdout.writable() is False

    def test_stdin_readable_writable(self):
        """proc.stdin.writable() is True, readable() is False."""
        mock_stdin_port = MockPort()
        control_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdin=PIPE)
        assert proc.stdin is not None
        assert proc.stdin.writable() is True
        assert proc.stdin.readable() is False

    def test_stdout_fileno_raises(self):
        """proc.stdout.fileno() raises UnsupportedOperation."""
        import io

        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        with pytest.raises(io.UnsupportedOperation):
            proc.stdout.fileno()

    def test_stdout_close(self):
        """proc.stdout.close() sets closed."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        assert proc.stdout.closed is False
        proc.stdout.close()
        assert proc.stdout.closed is True

    def test_stdout_read_after_close_raises(self):
        """Reading from a closed stream raises ValueError."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        proc.stdout.close()
        with pytest.raises(ValueError, match="closed"):
            proc.stdout.read()

    def test_partial_iteration_then_communicate(self):
        """Reading some lines, then communicate() returns the remainder."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"line1\nline2\nline3\n"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        # Read one line
        line1 = proc.stdout.readline()
        assert line1 == "line1\n"
        # communicate() should still work (reads from _stdout_parts)
        stdout, stderr = proc.communicate()
        assert stdout == "line1\nline2\nline3\n"

    def test_stdout_read_with_size(self):
        """proc.stdout.read(5) reads exactly 5 bytes."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"hello world"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages)
        assert proc.stdout is not None
        chunk = proc.stdout.read(5)
        assert chunk == "hello"
        rest = proc.stdout.read()
        assert rest == " world"

    def test_stdin_write_str(self):
        """proc.stdin.write(str) encodes and sends."""
        mock_stdin_port = MockPort()
        control_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, stdin=PIPE)
        assert proc.stdin is not None
        n = proc.stdin.write("hello")
        assert n == 5
        assert mock_stdin_port.sent[0].data == b"hello"


class TestPopenStreamReaderDecoding:
    """Text-mode reads must never split a multibyte char.

    Slicing the raw byte buffer and decoding the slice in read(size) or
    readline(size) would raise UnicodeDecodeError when a boundary landed
    mid-character.
    """

    @staticmethod
    def _reader(chunks, encoding="utf-8", errors="strict"):
        q: queue.Queue = queue.Queue()
        for c in chunks:
            q.put(c)
        q.put(None)  # _EOF_SENTINEL
        return _PopenStreamReader(q, encoding=encoding, errors=errors)

    def test_read_size_does_not_split_char_single_chunk(self):
        # The original crashing case: 'é' is 2 bytes; read(2) must yield "hé".
        r = self._reader([b"h\xc3\xa9llo"])
        assert r.read(2) == "hé"
        assert r.read() == "llo"

    def test_read_size_with_char_split_across_chunks(self):
        # 'é' (0xc3 0xa9) split across two queue chunks.
        r = self._reader([b"h\xc3", b"\xa9llo"])
        assert r.read(2) == "hé"
        assert r.read() == "llo"

    def test_readline_with_char_split_across_chunks(self):
        r = self._reader([b"caf\xc3", b"\xa9\nnext"])
        assert r.readline() == "café\n"
        assert r.read() == "next"

    def test_read_all_multibyte(self):
        r = self._reader([b"\xe2\x9c", b"\x93 done"])  # '✓' (3 bytes) split
        assert r.read() == "✓ done"

    def test_binary_mode_still_counts_bytes(self):
        r = self._reader([b"hello world"], encoding=None)
        assert r.read(5) == b"hello"
        assert r.read() == b" world"


class TestEncoding:
    """Tests for encoding= and errors= parameters."""

    def test_run_command_impl_encoding_latin1(self):
        """_run_command_impl passes encoding to subprocess.run."""
        # Octal escapes (\351 == 0xe9) work with both GNU and BSD printf, unlike
        # \xHH which BSD printf (macOS /usr/bin/printf) does not interpret.
        result = _run_command_impl(
            args=["printf", "caf\\351"],
            shell=False,
            cwd=None,
            env=None,
            capture_output=True,
            timeout=5,
            text=True,
            encoding="latin-1",
        )
        assert result["returncode"] == 0
        assert "caf\xe9" in result["stdout"]

    def test_run_encoding_implies_text(self):
        """_run() with encoding= implies text=True."""
        mesh = _mock_proc_mesh(
            {"timeout_expired": False, "returncode": 0, "stdout": "hello", "stderr": ""}
        )
        _run(
            ["echo", "hello"],
            proc_mesh=mesh,
            task_pool=None,
            capture_output=True,
            encoding="utf-8",
            text=False,
        )
        # encoding= should override text=False
        assert _run_command_args(mesh)["text"] is True

    def test_run_errors_replace(self):
        """errors='replace' replaces invalid bytes."""
        # Octal escapes (\377\376 == 0xff 0xfe) work with both GNU and BSD printf,
        # unlike \xHH which BSD printf (macOS /usr/bin/printf) does not interpret.
        result = _run_command_impl(
            args=["printf", "\\377\\376"],
            shell=False,
            cwd=None,
            env=None,
            capture_output=True,
            timeout=5,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result["returncode"] == 0
        # Invalid UTF-8 bytes should be replaced with the replacement character
        assert "\ufffd" in result["stdout"]

    def test_run_encoding_passed_to_actor(self):
        """_run() passes encoding and errors to the actor."""
        mesh = _mock_proc_mesh(
            {"timeout_expired": False, "returncode": 0, "stdout": "data", "stderr": ""}
        )
        _run(
            ["echo"],
            proc_mesh=mesh,
            task_pool=None,
            encoding="latin-1",
            errors="replace",
        )
        sent = _run_command_args(mesh)
        assert sent["encoding"] == "latin-1"
        assert sent["errors"] == "replace"

    def test_popen_encoding_latin1(self):
        """Popen with encoding='latin-1' decodes output accordingly."""
        control_port = MockPort()
        # Latin-1 encoded "café"
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"caf\xe9"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, encoding="latin-1", errors="strict")
        stdout, stderr = proc.communicate()
        assert stdout == "caf\xe9"

    def test_popen_errors_replace(self):
        """Popen with errors='replace' replaces invalid bytes."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"\xff\xfe"),
            StreamExit(returncode=0),
        ]
        proc = _make_popen_with_messages(messages, encoding="utf-8", errors="replace")
        stdout, stderr = proc.communicate()
        assert "\ufffd" in stdout


class TestSubprocessExceptionPickle:
    """Verify that remote subprocess exceptions survive pickle round-trips.

    Monarch serialises exceptions; without __reduce__ overrides the default
    args-tuple form only passes the message string to __init__, which fails
    for constructors that require typed positional arguments.
    """

    def test_called_process_error_pickle(self):
        import pickle

        from agentbox.remote.subprocess import CalledProcessError

        exc = CalledProcessError(
            returncode=2, cmd="ls /bad", output="out", stderr="err"
        )
        exc2 = pickle.loads(pickle.dumps(exc))

        assert exc2.returncode == 2
        assert exc2.cmd == "ls /bad"
        assert exc2.output == "out"
        assert exc2.stdout == "out"
        assert exc2.stderr == "err"
        assert "ls /bad" in str(exc2)

    def test_timeout_expired_pickle(self):
        import pickle

        from agentbox.remote.subprocess import TimeoutExpired

        exc = TimeoutExpired(
            cmd=["sleep", "9999"], timeout=5.0, output="out", stderr="err"
        )
        exc2 = pickle.loads(pickle.dumps(exc))

        assert exc2.cmd == ["sleep", "9999"]
        assert exc2.timeout == 5.0
        assert exc2.output == "out"
        assert exc2.stdout == "out"
        assert exc2.stderr == "err"

    def test_called_process_error_pickle_optional_none(self):
        import pickle

        from agentbox.remote.subprocess import CalledProcessError

        exc = CalledProcessError(returncode=1, cmd="exit 1")
        exc2 = pickle.loads(pickle.dumps(exc))

        assert exc2.returncode == 1
        assert exc2.output is None
        assert exc2.stderr is None
