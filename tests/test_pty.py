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

"""Tests for remote PTY support."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
import types
from typing import Any
from unittest.mock import MagicMock, patch

import agentbox.remote.pty as pty_module
import pytest
from _endpoints import endpoint_method
from _mock_streaming import MockPort, MockPortReceiver
from agentbox.remote.pty import _run_pty_impl, PtyProcess
from agentbox.remote.streaming import (
    ControlSignal,
    StdinClose,
    StdinData,
    StreamControlReady,
    StreamData,
    StreamError,
    StreamExit,
    StreamStdinReady,
    StreamType,
)


class TestRunPtyImpl:
    def test_pty_echo(self):
        """Run a real echo command via PTY and verify output."""
        output_port = MockPort()
        _run_pty_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]

        assert len(data_msgs) >= 1
        # PTY output includes the echoed text
        all_data = b"".join(m.data for m in data_msgs)
        assert b"hello" in all_data

        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == 0
        assert isinstance(output_port.sent[-1], StreamExit)

    def test_pty_drains_output_before_closing_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A fast child may exit before the reader consumes buffered output."""
        output_port = MockPort()
        read_pty_controller = pty_module._read_pty_controller

        def delayed_read(
            fd: int, size: int, port: Any, stop_event: threading.Event
        ) -> None:
            time.sleep(0.05)
            read_pty_controller(fd, size, port, stop_event)

        monkeypatch.setattr(pty_module, "_read_pty_controller", delayed_read)

        _run_pty_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
        assert b"hello" in b"".join(m.data for m in data_msgs)
        assert isinstance(output_port.sent[-1], StreamExit)

    def test_pty_reader_stop_with_terminal_still_open(self):
        """Stopping the reader must not depend on terminal EOF."""
        controller_fd, terminal_fd = os.openpty()
        output_port = MockPort()
        stop_event = threading.Event()
        reader = threading.Thread(
            target=pty_module._read_pty_controller,
            args=(controller_fd, 65536, output_port, stop_event),
        )

        try:
            os.write(terminal_fd, b"buffered output\n")
            reader.start()
            stop_event.set()
            reader.join(timeout=1)

            assert not reader.is_alive()
            data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
            assert b"buffered output" in b"".join(m.data for m in data_msgs)
            with pytest.raises(OSError):
                os.fstat(controller_fd)
        finally:
            stop_event.set()
            os.close(terminal_fd)
            if reader.ident is not None:
                reader.join(timeout=1)
            try:
                os.close(controller_fd)
            except OSError:
                pass

    def test_pty_reader_drain_deadline_excludes_forwarding_time(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Slow forwarding must not consume the post-exit drain timeout."""
        clock = [0.0]

        class SlowPort(MockPort):
            def send(self, msg: Any) -> None:
                # Each send outlasts the whole drain window on the fake clock.
                clock[0] += 1.0
                super().send(msg)

        monkeypatch.setattr(
            pty_module, "time", types.SimpleNamespace(monotonic=lambda: clock[0])
        )
        monkeypatch.setattr(pty_module, "_POST_EXIT_IDLE_SECONDS", 0.02)
        monkeypatch.setattr(pty_module, "_POST_EXIT_DRAIN_SECONDS", 0.5)
        controller_fd, terminal_fd = os.pipe()
        output_port = SlowPort()
        stop_event = threading.Event()
        payload = b"x" * 4096

        try:
            assert os.write(terminal_fd, payload) == len(payload)
            os.close(terminal_fd)
            terminal_fd = -1
            stop_event.set()

            pty_module._read_pty_controller(
                controller_fd, 1024, output_port, stop_event
            )

            data_msgs = [
                message
                for message in output_port.sent
                if isinstance(message, StreamData)
            ]
            assert b"".join(message.data for message in data_msgs) == payload
        finally:
            if terminal_fd >= 0:
                os.close(terminal_fd)
            try:
                os.close(controller_fd)
            except OSError:
                pass

    def test_pty_reader_stop_with_continuously_readable_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """An escaped writer cannot postpone StreamExit indefinitely."""

        clock = [0.0]

        class AlwaysReadableSelector:
            def register(self, fd: int, events: int) -> None:
                pass

            def select(self, timeout: float) -> list[object]:
                clock[0] += min(timeout, 0.001)
                return [object()]

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            pty_module.selectors, "DefaultSelector", AlwaysReadableSelector
        )
        monkeypatch.setattr(pty_module.os, "read", lambda fd, size: b"x")
        monkeypatch.setattr(pty_module.os, "close", lambda fd: None)
        monkeypatch.setattr(
            pty_module, "time", types.SimpleNamespace(monotonic=lambda: clock[0])
        )
        monkeypatch.setattr(pty_module, "_POST_EXIT_IDLE_SECONDS", 0.005)
        monkeypatch.setattr(pty_module, "_POST_EXIT_DRAIN_SECONDS", 0.02)
        output_port = MockPort()
        stop_event = threading.Event()
        stop_event.set()

        reader = threading.Thread(
            target=pty_module._read_pty_controller,
            args=(-1, 1024, output_port, stop_event),
            daemon=True,
        )
        reader.start()
        reader.join(timeout=10)

        assert not reader.is_alive()
        assert output_port.sent

    def test_pty_exit_code(self):
        """Verify StreamExit carries the correct non-zero returncode."""
        output_port = MockPort()
        _run_pty_impl(
            args="exit 42",
            shell=True,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == 42

    def test_pty_stdin_write(self, monkeypatch: pytest.MonkeyPatch):
        """Write input to PTY and verify echoed output (PTY echoes by default)."""
        output_port = MockPort()
        stdin_port = MockPort()
        stdin_receiver = MockPortReceiver()

        def stdin_factory():
            return stdin_port, stdin_receiver

        monkeypatch.setattr(pty_module, "_open_stdin_channel", stdin_factory)

        def feed_stdin():
            time.sleep(0.1)
            stdin_receiver.push(StdinData(data=b"pty_test_value\n"))
            time.sleep(0.1)
            stdin_receiver.push(StdinClose())

        feeder = threading.Thread(target=feed_stdin, daemon=True)
        feeder.start()

        _run_pty_impl(
            args="cat",
            shell=True,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=True,
            chunk_size=65536,
        )

        feeder.join(timeout=10)

        # Check that StreamStdinReady was sent
        stdin_ready = [m for m in output_port.sent if isinstance(m, StreamStdinReady)]
        assert len(stdin_ready) == 1
        assert stdin_ready[0].stdin_port is stdin_port

        # PTY echoes input, so the output should contain what we wrote
        data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
        all_data = b"".join(m.data for m in data_msgs)
        assert b"pty_test_value" in all_data

        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == 0

    def test_pty_env(self):
        """Verify environment variables are passed to the PTY child."""
        output_port = MockPort()
        _run_pty_impl(
            args="echo $MY_PTY_VAR",
            shell=True,
            cwd=None,
            env={"MY_PTY_VAR": "pty_env_value"},
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
        all_data = b"".join(m.data for m in data_msgs)
        assert b"pty_env_value" in all_data

        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == 0

    def test_pty_control_kill(self, monkeypatch: pytest.MonkeyPatch):
        """Verify SIGKILL via control channel terminates the PTY process."""
        output_port = MockPort()
        control_port = MockPort()
        control_receiver = MockPortReceiver()

        def control_factory():
            return control_port, control_receiver

        monkeypatch.setattr(pty_module, "_open_control_channel", control_factory)

        def send_kill():
            time.sleep(0.2)
            control_receiver.push(ControlSignal(signal_num=signal.SIGKILL))

        killer = threading.Thread(target=send_kill, daemon=True)
        killer.start()

        _run_pty_impl(
            args=["sleep", "30"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            enable_control=True,
        )

        killer.join(timeout=5)

        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == -signal.SIGKILL

    def test_pty_bad_command(self):
        """Verify StreamError is sent for a bad command."""
        output_port = MockPort()
        _run_pty_impl(
            args=["/nonexistent/command/that/does/not/exist"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        error_msgs = [m for m in output_port.sent if isinstance(m, StreamError)]
        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(error_msgs) == 1
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == -1

    def test_pty_bad_command_does_not_leak_fds(self):
        """A failed spawn must close the PTY pair (regression: openpty fds
        leaked when Popen raised, exhausting the worker fd table)."""
        import os

        def count_fds() -> int:
            try:
                return len(os.listdir("/dev/fd"))
            except FileNotFoundError:
                return len(os.listdir("/proc/self/fd"))

        before = count_fds()
        for _ in range(50):
            _run_pty_impl(
                args=["/nonexistent/command/that/does/not/exist"],
                shell=False,
                cwd=None,
                env=None,
                output_port=MockPort(),
                enable_stdin=False,
                chunk_size=65536,
            )
        # A leak would grow by ~2 fds per failed spawn (controller + terminal).
        assert count_fds() - before <= 2

    def test_pty_all_output_is_stdout(self):
        """PTY merges all output into STDOUT stream type (no separate STDERR)."""
        output_port = MockPort()
        _run_pty_impl(
            args="echo out && echo err >&2",
            shell=True,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
        # All data messages should be STDOUT (PTY merges streams)
        for msg in data_msgs:
            assert msg.stream == StreamType.STDOUT

        all_data = b"".join(m.data for m in data_msgs)
        assert b"out" in all_data
        assert b"err" in all_data

    def test_pty_control_channel_setup(self, monkeypatch: pytest.MonkeyPatch):
        """Verify control channel is set up when enable_control=True."""
        output_port = MockPort()
        control_port = MockPort()
        control_receiver = MockPortReceiver()

        def control_factory():
            return control_port, control_receiver

        monkeypatch.setattr(pty_module, "_open_control_channel", control_factory)

        _run_pty_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            enable_control=True,
        )

        control_ready = [
            m for m in output_port.sent if isinstance(m, StreamControlReady)
        ]
        assert len(control_ready) == 1
        assert control_ready[0].control_port is control_port

    def test_pty_no_control_channel_by_default(self):
        """When enable_control is False (default), no control channel is created."""
        output_port = MockPort()
        _run_pty_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        control_ready = [
            m for m in output_port.sent if isinstance(m, StreamControlReady)
        ]
        assert len(control_ready) == 0


class TestPtyActor:
    """Stopping the actor (e.g. releasing a lease with a live PTY) cancels run_pty.

    A blocking endpoint could not be cancelled, so Monarch's cleanup deadline
    elapsed and the client crashed.
    """

    @staticmethod
    def _start(
        monkeypatch: pytest.MonkeyPatch, gate: threading.Event | None = None
    ) -> tuple[asyncio.Future[None], list[Any], MockPort]:
        """Start run_pty on a process group; *gate* holds the thread before spawn."""
        started: list[Any] = []
        real_impl = pty_module._run_pty_impl

        def recording_impl(*args: Any) -> None:
            *rest, on_started = args

            def record(proc: Any) -> None:
                started.append(proc)
                on_started(proc)

            if gate is not None:
                gate.wait(timeout=10)
            real_impl(*rest, record)

        monkeypatch.setattr(pty_module, "_run_pty_impl", recording_impl)
        run_pty = endpoint_method(pty_module._PtyActor, "run_pty")
        output_port = MockPort()
        task = asyncio.ensure_future(
            run_pty(
                None,
                ["sh", "-c", "sleep 600 & wait"],
                False,
                None,
                None,
                output_port,
                False,
                65536,
            )
        )
        return task, started, output_port

    @staticmethod
    async def _assert_cancelled_and_killed(
        task: asyncio.Future[None], started: list[Any], output_port: MockPort
    ) -> None:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert started, "PTY process did not start"
        with pytest.raises(ProcessLookupError):
            os.killpg(started[0].pid, 0)
        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert [m.returncode for m in exit_msgs] == [-signal.SIGKILL]

    async def test_cancel_kills_process_group(self, monkeypatch: pytest.MonkeyPatch):
        task, started, output_port = self._start(monkeypatch)
        deadline = time.monotonic() + 10
        while not started and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert started, "PTY process did not start"

        task.cancel()
        await self._assert_cancelled_and_killed(task, started, output_port)

    async def test_cancel_before_spawn_kills_late_process(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A cancel that arrives before the child starts still kills it."""
        gate = threading.Event()
        task, started, output_port = self._start(monkeypatch, gate)
        await asyncio.sleep(0.1)  # let the thread reach the gate
        assert not started

        task.cancel()
        await asyncio.sleep(0)  # deliver the cancellation before the spawn
        gate.set()
        await self._assert_cancelled_and_killed(task, started, output_port)


def _make_pty_process_with_messages(
    messages: list[Any],
    text: bool = False,
    stdin_port: Any = None,
    control_port: Any = None,
) -> PtyProcess:
    """Create a PtyProcess over a mock mesh whose output replays *messages*.

    The actor's port announcements come first, as they do from a real actor,
    so the constructor's startup handshake completes.
    """
    receiver = MockPortReceiver()
    receiver.push(StreamStdinReady(stdin_port=stdin_port or MockPort()))
    receiver.push(StreamControlReady(control_port=control_port or MockPort()))
    for msg in messages:
        receiver.push(msg)

    with patch("agentbox.remote.pty.Channel") as channel:
        channel.__getitem__.return_value.open.return_value = (MagicMock(), receiver)
        return PtyProcess(["bash"], proc_mesh=MagicMock(), text=text)


class TestPtyProcessIteration:
    def test_stdout_iteration(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello "),
            StreamData(stream=StreamType.STDOUT, data=b"world"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages)
        chunks = list(proc.stdout)
        assert chunks == [b"hello ", b"world"]
        assert proc.returncode == 0

    def test_stdout_text_mode(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, text=True)
        chunks = list(proc.stdout)
        assert chunks == ["hello"]

    def test_read_collects_all(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"part1"),
            StreamData(stream=StreamType.STDOUT, data=b"part2"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages)
        result = proc.read()
        assert result == b"part1part2"

    def test_read_text_mode(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, text=True)
        result = proc.read()
        assert result == "hello"

    def test_wait_returns_exit_code(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=3),
        ]
        proc = _make_pty_process_with_messages(messages)
        code = proc.wait()
        assert code == 3

    def test_pid_is_none(self):
        messages = [StreamExit(returncode=0)]
        proc = _make_pty_process_with_messages(messages)
        assert proc.pid is None


class TestPtyProcessSignals:
    def test_kill_sends_sigkill(self):
        control_port = MockPort()
        messages = [
            StreamExit(returncode=-9),
        ]
        proc = _make_pty_process_with_messages(messages, control_port=control_port)
        proc.kill()
        assert len(control_port.sent) == 1
        assert isinstance(control_port.sent[0], ControlSignal)
        assert control_port.sent[0].signal_num == signal.SIGKILL

    def test_terminate_sends_sigterm(self):
        control_port = MockPort()
        messages = [
            StreamExit(returncode=-15),
        ]
        proc = _make_pty_process_with_messages(messages, control_port=control_port)
        proc.terminate()
        assert len(control_port.sent) == 1
        assert control_port.sent[0].signal_num == signal.SIGTERM

    def test_send_signal_noop_after_exit(self):
        control_port = MockPort()
        messages = [
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, control_port=control_port)
        proc.wait()
        proc.send_signal(signal.SIGTERM)
        assert len(control_port.sent) == 0


class TestPtyProcessContextManager:
    def test_context_manager_waits_on_exit(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=5),
        ]
        with _make_pty_process_with_messages(messages) as proc:
            pass
        assert proc.returncode == 5


class TestPtyProcessCommunicate:
    def test_communicate_collects_output(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, stdin_port=mock_stdin_port)
        output, empty = proc.communicate(input=b"test")
        assert output == b"hello"
        assert empty == b""
        assert proc.returncode == 0

    def test_communicate_text_mode(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(
            messages, text=True, stdin_port=mock_stdin_port
        )
        output, empty = proc.communicate(input="test")
        assert output == "hello"
        assert empty == ""

    def test_communicate_no_input(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"output"),
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, stdin_port=mock_stdin_port)
        output, empty = proc.communicate()
        assert output == b"output"
        # StdinClose should have been sent
        assert any(isinstance(m, StdinClose) for m in mock_stdin_port.sent)


class TestPtyProcessWrite:
    def test_write_sends_stdin_data(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, stdin_port=mock_stdin_port)
        proc.write(b"hello")
        assert len(mock_stdin_port.sent) == 1
        assert isinstance(mock_stdin_port.sent[0], StdinData)
        assert mock_stdin_port.sent[0].data == b"hello"

    def test_write_str_encodes(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamExit(returncode=0),
        ]
        proc = _make_pty_process_with_messages(messages, stdin_port=mock_stdin_port)
        proc.write("hello")
        assert mock_stdin_port.sent[0].data == b"hello"

    def test_write_raises_after_exit(self):
        mock_stdin_port = MockPort()
        messages = [StreamExit(returncode=0)]
        proc = _make_pty_process_with_messages(messages, stdin_port=mock_stdin_port)
        proc.wait()
        with pytest.raises(RuntimeError, match="process has exited"):
            proc.write(b"data")
