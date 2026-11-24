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

"""Tests for streaming subprocess support."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import threading
import types
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _mock_streaming import MockPort
from agentbox.remote import streaming
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
from agentbox.remote.subprocess import (
    _arun_streaming_impl,
    _run_streaming_impl,
    Popen,
    TimeoutExpired,
)


class TestMessageProtocol:
    def test_stream_data_frozen(self):
        msg = StreamData(stream=StreamType.STDOUT, data=b"hello")
        with pytest.raises(AttributeError):
            msg.data = b"world"  # type: ignore[misc]

    def test_control_signal_frozen(self):
        msg = ControlSignal(signal_num=signal.SIGKILL)
        with pytest.raises(AttributeError):
            msg.signal_num = signal.SIGTERM  # type: ignore[misc]


class MockFuture:
    """Mimics a Monarch Future.

    The real ``recv()`` returns a future *immediately*; the wait happens in
    ``.get(timeout)``, which raises :class:`TimeoutError` if no message arrives
    within *timeout*.  Modelling that split is essential: the streaming impl
    polls with ``recv().get(timeout=0.05)`` and relies on the timeout to fall
    through and re-check ``proc.poll()``.  A fake whose ``recv()`` blocks would
    deadlock that loop.
    """

    def __init__(self, receiver: MockPortReceiver):
        self._receiver = receiver

    def get(self, timeout: float | None = None) -> Any:
        return self._receiver._take(timeout)


class MockPortReceiver:
    """Mimics PortReceiver — recv() returns a future immediately."""

    def __init__(self) -> None:
        self._messages: deque[Any] = deque()
        self._cond = threading.Condition()

    def push(self, msg: Any) -> None:
        with self._cond:
            self._messages.append(msg)
            self._cond.notify()

    def recv(self) -> MockFuture:
        return MockFuture(self)

    def _take(self, timeout: float | None) -> Any:
        with self._cond:
            if not self._messages:
                self._cond.wait(timeout)
            if not self._messages:
                raise TimeoutError
            return self._messages.popleft()


class _ReceiveSlotFuture:
    """Future over a single :class:`_TrackingReceiver` receive slot."""

    def __init__(self, receiver: "_TrackingReceiver", slot: list[Any]) -> None:
        self._receiver = receiver
        self._slot = slot

    def get(self, timeout: float | None = None) -> Any:
        with self._receiver._cond:
            if not self._slot[0]:
                self._receiver._cond.wait(timeout)
            if not self._slot[0]:
                raise TimeoutError
            return self._slot[1]


class _TrackingReceiver:
    """PortReceiver that tracks outstanding receives.

    Each ``recv()`` registers a *distinct* pending receive, and a timed-out
    ``get()`` does **not** cancel that registration; a pushed message is
    delivered FIFO to the oldest still-pending receive.  Under these semantics
    a poll loop that calls ``recv()`` afresh every iteration and abandons the
    future on timeout leaks registrations, so a message arriving after an idle
    gap (e.g. the first command after a shell-command timeout) is handed to a
    long-abandoned receive and lost.  The correct loop holds a *single* pending
    receive and re-``get()``s it, so ``recv_calls`` stays at 1 while idle.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: deque[list[Any]] = deque()  # slots: [ready: bool, msg]
        self._inbox: deque[Any] = deque()
        self.recv_calls = 0
        self.max_pending = 0

    def push(self, msg: Any) -> None:
        with self._cond:
            if self._pending:
                slot = self._pending.popleft()
                slot[0] = True
                slot[1] = msg
            else:
                self._inbox.append(msg)
            self._cond.notify_all()

    def recv(self) -> _ReceiveSlotFuture:
        with self._cond:
            self.recv_calls += 1
            slot: list[Any] = [False, None]
            if self._inbox:
                slot[0] = True
                slot[1] = self._inbox.popleft()
            else:
                self._pending.append(slot)
                self.max_pending = max(self.max_pending, len(self._pending))
            return _ReceiveSlotFuture(self, slot)


class _GatedFuture(Future[Any]):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.get_calls = 0

    def get(self, timeout: float | None = None) -> Any:
        self.get_calls += 1
        self.entered.set()
        return self.result(timeout=timeout)


def _make_streaming_popen_with_messages(
    messages: list[Any],
    stdin: bool = False,
    text: bool = False,
    stdout_piped: bool = False,
    stderr_piped: bool = False,
):
    """Create a streaming-mode Popen with a mocked Channel and pre-loaded messages."""
    receiver = MockPortReceiver()
    for msg in messages:
        receiver.push(msg)

    mock_port = MockPort()

    with (
        patch("agentbox.remote.subprocess.Channel") as mock_channel_cls,
        patch("agentbox.remote.subprocess.ProcMesh"),
    ):
        mock_channel_cls.__class_getitem__ = lambda cls, item: cls
        mock_channel_cls.open.return_value = (mock_port, receiver)

        mock_proc_mesh = MagicMock()
        mock_actor_mesh = MagicMock()
        mock_proc_mesh.spawn.return_value = mock_actor_mesh

        sp = Popen.__new__(Popen)
        sp.args = ["echo", "hello"]
        sp._text = text
        sp._encoding = "utf-8" if text else None
        sp._errors = "strict"
        sp._returncode = None
        sp._stdin_enabled = stdin
        sp._stdin_port = None
        sp._control_port = None
        sp._exited = False
        sp._input_sent = False
        sp._communicated = False
        sp._streaming = True
        sp._pending_receive = None
        sp._pending_messages = deque()
        sp._receive_condition = threading.Condition()
        sp._receiving = False
        sp._stdout_piped = stdout_piped
        sp._stderr_piped = stderr_piped
        sp._output_receiver = receiver
        sp._actor_mesh = mock_actor_mesh

        # Set up stdout/stderr iterators when piped
        if stdout_piped:
            sp.stdout = sp._iter_stdout()
        else:
            sp.stdout = None
        if stderr_piped:
            sp.stderr = sp._iter_stderr()
        else:
            sp.stderr = None

    return sp


class TestStreamingPopenIteration:
    def test_stdout_iteration(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello "),
            StreamData(stream=StreamType.STDERR, data=b"err"),
            StreamData(stream=StreamType.STDOUT, data=b"world"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        chunks = list(sp._iter_stdout())
        assert chunks == [b"hello ", b"world"]

    def test_stdout_attr_iteration(self):
        """Popen(streaming=True, stdout=PIPE).stdout is iterable."""
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello "),
            StreamData(stream=StreamType.STDOUT, data=b"world"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, stdout_piped=True)
        chunks = list(sp.stdout)
        assert chunks == [b"hello ", b"world"]
        assert sp.returncode == 0

    def test_stderr_iteration(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"out"),
            StreamData(stream=StreamType.STDERR, data=b"err1"),
            StreamData(stream=StreamType.STDERR, data=b"err2"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        chunks = list(sp._iter_stderr())
        assert chunks == [b"err1", b"err2"]

    def test_stderr_attr_iteration(self):
        """Popen(streaming=True, stderr=PIPE).stderr is iterable."""
        messages = [
            StreamData(stream=StreamType.STDERR, data=b"err1"),
            StreamData(stream=StreamType.STDERR, data=b"err2"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, stderr_piped=True)
        chunks = list(sp.stderr)
        assert chunks == [b"err1", b"err2"]

    def test_stdout_stderr_none_when_not_piped(self):
        """Without piped flags, stdout/stderr are None in streaming mode."""
        messages = [StreamExit(returncode=0)]
        sp = _make_streaming_popen_with_messages(messages)
        assert sp.stdout is None
        assert sp.stderr is None

    def test_interleaved_output(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"out1"),
            StreamData(stream=StreamType.STDERR, data=b"err1"),
            StreamData(stream=StreamType.STDOUT, data=b"out2"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        chunks = list(sp.output)
        assert len(chunks) == 3
        assert chunks[0] == StreamData(stream=StreamType.STDOUT, data=b"out1")
        assert chunks[1] == StreamData(stream=StreamType.STDERR, data=b"err1")
        assert chunks[2] == StreamData(stream=StreamType.STDOUT, data=b"out2")

    def test_returncode_set_after_iteration(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=7),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        assert sp.returncode is None
        list(sp._iter_stdout())
        assert sp.returncode == 7

    def test_communicate_collects_all_output(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello "),
            StreamData(stream=StreamType.STDERR, data=b"warn "),
            StreamData(stream=StreamType.STDOUT, data=b"world"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        stdout, stderr = sp.communicate()
        assert stdout == b"hello world"
        assert stderr == b"warn "
        assert sp.returncode == 0

    def test_communicate_text_mode(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamData(stream=StreamType.STDERR, data=b"err"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, text=True)
        stdout, stderr = sp.communicate()
        assert stdout == "hello"
        assert stderr == "err"

    def test_stdout_text_mode(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, text=True)
        chunks = list(sp._iter_stdout())
        assert chunks == ["hello"]

    def test_stream_error_captured(self):
        messages = [
            StreamError(error="something broke"),
            StreamExit(returncode=-1),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        # output iterator skips StreamError, but still gets to StreamExit
        chunks = list(sp.output)
        assert chunks == []
        assert sp.returncode == -1

    def test_wait_returns_exit_code(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=3),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        code = sp.wait()
        assert code == 3

    def test_pid_is_none(self):
        messages = [StreamExit(returncode=0)]
        sp = _make_streaming_popen_with_messages(messages)
        assert sp.pid is None

    def test_control_ready_skipped_in_iteration(self):
        """StreamControlReady messages are handled internally, not yielded."""
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        chunks = list(sp._iter_stdout())
        assert chunks == [b"data"]
        assert sp._control_port is control_port


class TestStreamingPopenStdin:
    def test_write_stdin_raises_when_disabled(self):
        messages = [StreamExit(returncode=0)]
        sp = _make_streaming_popen_with_messages(messages, stdin=False)
        with pytest.raises(RuntimeError, match="stdin is not enabled"):
            sp.write_stdin(b"data")

    def test_write_stdin_raises_after_exit(self):
        messages = [StreamExit(returncode=0)]
        sp = _make_streaming_popen_with_messages(messages, stdin=True)
        # Drain to set _exited
        sp.wait()
        with pytest.raises(RuntimeError, match="process has exited"):
            sp.write_stdin(b"data")

    def test_close_stdin_sends_close_message(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, stdin=True)
        sp.close_stdin()
        assert len(mock_stdin_port.sent) == 1
        assert isinstance(mock_stdin_port.sent[0], StdinClose)

    def test_write_stdin_sends_data(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, stdin=True)
        sp.write_stdin(b"hello")
        assert len(mock_stdin_port.sent) == 1
        assert isinstance(mock_stdin_port.sent[0], StdinData)
        assert mock_stdin_port.sent[0].data == b"hello"

    def test_write_stdin_str_encodes_to_bytes(self):
        mock_stdin_port = MockPort()
        messages = [
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, stdin=True)
        sp.write_stdin("hello")
        assert mock_stdin_port.sent[0].data == b"hello"

    def test_close_stdin_raises_when_disabled(self):
        messages = [StreamExit(returncode=0)]
        sp = _make_streaming_popen_with_messages(messages, stdin=False)
        with pytest.raises(RuntimeError, match="stdin is not enabled"):
            sp.close_stdin()

    def test_write_stdin_skips_control_ready(self):
        """write_stdin handles StreamControlReady while waiting for stdin port."""
        mock_stdin_port = MockPort()
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamStdinReady(stdin_port=mock_stdin_port),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages, stdin=True)
        sp.write_stdin(b"hello")
        assert sp._control_port is control_port
        assert mock_stdin_port.sent[0].data == b"hello"

    def test_close_stdin_returns_for_buffered_exit(self):
        sp = _make_streaming_popen_with_messages([StreamExit(returncode=4)], stdin=True)
        sp.terminate()

        sp.close_stdin()

        assert sp.returncode == 4


class TestStreamingPopenSignals:
    def test_signal_timeout_does_not_share_output_receive(self):
        control_port = MockPort()
        first = _GatedFuture()
        sp = _make_streaming_popen_with_messages(
            [
                StreamControlReady(control_port=control_port),
                StreamData(stream=StreamType.STDERR, data=b"second"),
                StreamExit(returncode=0),
            ]
        )
        receiver = sp._output_receiver
        with (
            patch.object(
                receiver,
                "recv",
                side_effect=[first, receiver.recv(), receiver.recv(), receiver.recv()],
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            output = pool.submit(lambda: list(sp.output))
            try:
                assert first.entered.wait(timeout=2)
                with pytest.raises(TimeoutExpired):
                    pool.submit(sp.terminate, timeout=0.01).result(timeout=2)
                assert first.get_calls == 1
                assert control_port.sent == []
            finally:
                first.set_result(StreamData(stream=StreamType.STDOUT, data=b"first"))
            assert [(msg.stream, msg.data) for msg in output.result(timeout=2)] == [
                (StreamType.STDOUT, b"first"),
                (StreamType.STDERR, b"second"),
            ]
        assert sp.returncode == 0

    def test_ready_ports_do_not_wait_for_idle_output(self):
        control_port, stdin_port = MockPort(), MockPort()
        idle = _GatedFuture()
        sp = _make_streaming_popen_with_messages(
            [
                StreamControlReady(control_port=control_port),
                StreamStdinReady(stdin_port=stdin_port),
            ],
            stdin=True,
        )
        receiver = sp._output_receiver
        with (
            patch.object(
                receiver, "recv", side_effect=[receiver.recv(), receiver.recv(), idle]
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            output = pool.submit(lambda: list(sp.output))

            def send() -> None:
                sp.write_stdin(b"input")
                sp.close_stdin()
                sp.terminate(timeout=1)

            try:
                assert idle.entered.wait(timeout=2)
                pool.submit(send).result(timeout=2)
                assert stdin_port.sent == [StdinData(data=b"input"), StdinClose()]
                assert control_port.sent == [ControlSignal(signal_num=signal.SIGTERM)]
            finally:
                idle.set_result(StreamExit(returncode=-15))
            assert output.result(timeout=2) == []
        assert sp.returncode == -15

    @pytest.mark.parametrize("operation", ["terminate", "write_stdin", "close_stdin"])
    @pytest.mark.parametrize("exits", [False, True])
    @pytest.mark.parametrize("control_first", [False, True])
    def test_port_waiters_and_output_share_startup(
        self, operation, exits, control_first
    ):
        control_port, stdin_port = MockPort(), MockPort()
        first = _GatedFuture()
        last = _GatedFuture()
        sp = _make_streaming_popen_with_messages(
            [
                StreamStdinReady(stdin_port=stdin_port),
                StreamControlReady(control_port=control_port),
            ],
            stdin=True,
        )
        receiver = sp._output_receiver

        def control() -> None:
            if operation == "write_stdin":
                sp.write_stdin(b"input")
            else:
                getattr(sp, operation)()

        with (
            patch.object(
                receiver,
                "recv",
                side_effect=[first, receiver.recv(), receiver.recv(), last],
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            try:
                if control_first:
                    waiter = pool.submit(control)
                else:
                    output = pool.submit(lambda: list(sp.output))
                assert first.entered.wait(timeout=2)
                if control_first:
                    output = pool.submit(lambda: list(sp.output))
                else:
                    waiter = pool.submit(control)
                first.set_result(
                    StreamExit(returncode=3)
                    if exits
                    else StreamData(stream=StreamType.STDOUT, data=b"first")
                )
                if operation == "write_stdin" and exits:
                    with pytest.raises(RuntimeError, match="process has exited"):
                        waiter.result(timeout=2)
                else:
                    waiter.result(timeout=2)
                if not exits:
                    assert control_port.sent == (
                        [ControlSignal(signal_num=signal.SIGTERM)]
                        if operation == "terminate"
                        else []
                    )
                    assert stdin_port.sent == (
                        [StdinData(data=b"input")]
                        if operation == "write_stdin"
                        else [StdinClose()]
                        if operation == "close_stdin"
                        else []
                    )
                last.set_result(StreamExit(returncode=3))
                assert [msg.data for msg in output.result(timeout=2)] == (
                    [] if exits else [b"first"]
                )
            finally:
                if not first.done():
                    first.set_result(StreamExit(returncode=3))
                if not last.done():
                    last.set_result(StreamExit(returncode=3))
        assert sp.returncode == 3
        if exits:
            assert control_port.sent == stdin_port.sent == []

    def test_paused_output_keeps_receives_lazy_and_control_usable(self):
        control_port = MockPort()
        sp = _make_streaming_popen_with_messages(
            [
                StreamData(stream=StreamType.STDOUT, data=b"first"),
                StreamControlReady(control_port=control_port),
                StreamExit(returncode=0),
            ]
        )
        # The test double's queue shows what has been taken off the wire.
        unread = sp._output_receiver._messages
        with ThreadPoolExecutor(max_workers=1) as pool:
            output = sp.output
            assert next(output).data == b"first"
            # Lazy: nothing beyond the yielded chunk has been consumed.
            assert len(unread) == 2
            try:
                pool.submit(sp.terminate, timeout=1).result(timeout=2)
                # terminate() takes only the control port, leaving the exit.
                assert len(unread) == 1
                assert control_port.sent == [ControlSignal(signal_num=signal.SIGTERM)]
            finally:
                assert list(output) == []

    @pytest.mark.parametrize("control_first", [False, True])
    def test_failed_receive_releases_other_caller(self, control_first):
        control_port = MockPort()
        first = _GatedFuture()
        failure = RuntimeError("receive failed")
        sp = _make_streaming_popen_with_messages(
            [StreamControlReady(control_port=control_port), StreamExit(returncode=0)]
        )
        receiver = sp._output_receiver
        with (
            patch.object(
                receiver, "recv", side_effect=[first, receiver.recv(), receiver.recv()]
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            try:
                if control_first:
                    owner = pool.submit(sp.terminate, timeout=1)
                else:
                    owner = pool.submit(lambda: list(sp.output))
                assert first.entered.wait(timeout=2)
                observer = pool.submit(
                    (lambda: list(sp.output)) if control_first else sp.terminate
                )
            finally:
                first.set_exception(failure)
            with pytest.raises(RuntimeError) as caught:
                owner.result(timeout=2)
            assert caught.value is failure
            observer.result(timeout=2)
            assert first.get_calls == 1
        if control_first:
            assert control_port.sent == []
            assert sp.returncode == 0
        else:
            assert control_port.sent == [ControlSignal(signal_num=signal.SIGTERM)]
            assert sp.communicate() == (b"", b"")

    def test_signal_deadline_also_bounds_readily_available_output(self):
        control_port = MockPort()
        sp = _make_streaming_popen_with_messages(
            [
                StreamData(stream=StreamType.STDOUT, data=b"output"),
                StreamControlReady(control_port=control_port),
                StreamExit(returncode=0),
            ]
        )
        clock = [0.0]
        receive = sp._receive_message

        def slow_receive(timeout: float | None) -> Any:
            message = receive(timeout)
            clock[0] += 2.0  # Receiving the output outlasts the deadline.
            return message

        sp._receive_message = slow_receive
        with patch(
            "agentbox.remote.subprocess.time",
            types.SimpleNamespace(monotonic=lambda: clock[0]),
        ):
            with pytest.raises(TimeoutExpired):
                sp.terminate(timeout=1.0)
        assert control_port.sent == []
        assert sp.communicate() == (b"output", b"")

    def test_signal_timeout_retains_pending_receive_and_output(self):
        sp = _make_streaming_popen_with_messages([])
        receiver = _TrackingReceiver()
        sp._output_receiver = receiver
        control_port = MockPort()
        receiver.push(StreamData(stream=StreamType.STDOUT, data=b"before control"))
        with pytest.raises(TimeoutExpired):
            sp.terminate(timeout=0.01)
        assert sp.poll() is None
        receiver.push(StreamControlReady(control_port=control_port))
        sp.terminate(timeout=1.0)
        receiver.push(StreamExit(returncode=-15))
        assert sp.communicate() == (b"before control", b"")
        assert [msg.signal_num for msg in control_port.sent] == [signal.SIGTERM]
        assert receiver.max_pending == 1

    @pytest.mark.parametrize("timeout", [None, 1.0])
    @pytest.mark.parametrize("retry_signal", [False, True])
    def test_receive_failure_does_not_poison_later_operations(
        self, timeout, retry_signal
    ):
        control_port = MockPort()
        sp = _make_streaming_popen_with_messages(
            [
                StreamControlReady(control_port=control_port),
                StreamData(stream=StreamType.STDOUT, data=b"after failure"),
                StreamExit(returncode=0),
            ]
        )
        failure = RuntimeError("receive failed")
        failed_future = MagicMock()
        failed_future.get.side_effect = failure
        receiver = sp._output_receiver
        with patch.object(
            receiver,
            "recv",
            side_effect=[
                failed_future,
                receiver.recv(),
                receiver.recv(),
                receiver.recv(),
            ],
        ):
            with pytest.raises(RuntimeError) as caught:
                sp.terminate(timeout=timeout)
            assert caught.value is failure
            assert control_port.sent == []

            if retry_signal:
                sp.terminate(timeout=timeout)
            assert sp.communicate() == (b"after failure", b"")
            assert sp.returncode == 0
            assert [msg.signal_num for msg in control_port.sent] == (
                [signal.SIGTERM] if retry_signal else []
            )

    def test_exit_before_control_is_not_signaled(self):
        sp = _make_streaming_popen_with_messages([StreamExit(returncode=7)])
        sp.kill(timeout=0.1)
        assert sp.returncode == 7

    def test_buffered_exit_is_not_signaled_when_control_is_ready(self):
        control_port = MockPort()
        sp = _make_streaming_popen_with_messages(
            [StreamControlReady(control_port=control_port), StreamExit(returncode=7)],
            stdin=True,
        )
        sp.close_stdin()

        sp.kill(timeout=0.1)

        assert sp.returncode == 7
        assert control_port.sent == []

    def test_kill_sends_sigkill(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=-9),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        sp.kill()
        assert len(control_port.sent) == 1
        assert isinstance(control_port.sent[0], ControlSignal)
        assert control_port.sent[0].signal_num == signal.SIGKILL

    def test_terminate_sends_sigterm(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=-15),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        sp.terminate()
        assert len(control_port.sent) == 1
        assert control_port.sent[0].signal_num == signal.SIGTERM

    def test_send_signal_custom(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        sp.send_signal(signal.SIGUSR1)
        assert control_port.sent[0].signal_num == signal.SIGUSR1

    def test_send_signal_noop_after_exit(self):
        control_port = MockPort()
        messages = [
            StreamControlReady(control_port=control_port),
            StreamExit(returncode=0),
        ]
        sp = _make_streaming_popen_with_messages(messages)
        sp.wait()
        sp.send_signal(signal.SIGTERM)
        assert len(control_port.sent) == 0


class TestStreamingPopenContextManager:
    def test_context_manager_basic(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"hello"),
            StreamExit(returncode=0),
        ]
        with _make_streaming_popen_with_messages(messages) as sp:
            stdout, stderr = sp.communicate()
        assert stdout == b"hello"
        assert sp.returncode == 0

    def test_context_manager_waits_on_exit(self):
        messages = [
            StreamData(stream=StreamType.STDOUT, data=b"data"),
            StreamExit(returncode=5),
        ]
        with _make_streaming_popen_with_messages(messages) as sp:
            pass  # don't explicitly communicate
        assert sp.returncode == 5


class TestRunStreamingImpl:
    @pytest.mark.asyncio
    async def test_cancellation_kills_owned_process_group(self, tmp_path):
        """Actor cancellation must clean up without a client control port."""
        output_port = MockPort()
        control_receiver = MockPortReceiver()
        pid_path = tmp_path / "process-group-pid"
        command = f"sleep 30 & echo $$ > {shlex.quote(str(pid_path))}; wait"

        task = asyncio.create_task(
            _arun_streaming_impl(
                args=["bash", "-c", command],
                shell=False,
                cwd=None,
                env=None,
                output_port=output_port,
                enable_stdin=False,
                chunk_size=65536,
                enable_control=True,
                start_new_session=True,
                _control_channel_factory=lambda: (MockPort(), control_receiver),
            )
        )
        process_group: int | None = None
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            while not pid_path.exists():
                if loop.time() >= deadline:
                    pytest.fail("streaming subprocess did not start")
                await asyncio.sleep(0.01)
            process_group = int(pid_path.read_text())

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            deadline = loop.time() + 2
            while True:
                try:
                    os.killpg(process_group, 0)
                except ProcessLookupError:
                    break
                if loop.time() >= deadline:
                    pytest.fail("cancelled streaming process group survived cleanup")
                await asyncio.sleep(0.01)

            # The client handle must learn the process is gone, or it never exits.
            exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
            assert [m.returncode for m in exit_msgs] == [-signal.SIGKILL]
        finally:
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            if process_group is not None:
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_streaming_echo(self):
        """Run a real echo command and verify StreamData + StreamExit sequence."""
        output_port = MockPort()
        _run_streaming_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        # Should have at least one StreamData(STDOUT) and one StreamExit
        data_msgs = [m for m in output_port.sent if isinstance(m, StreamData)]
        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]

        assert len(data_msgs) >= 1
        assert data_msgs[0].stream == StreamType.STDOUT
        assert b"hello" in data_msgs[0].data

        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == 0

        # StreamExit is the last message
        assert isinstance(output_port.sent[-1], StreamExit)

    def test_streaming_exit_code(self):
        """Verify StreamExit carries the correct non-zero returncode."""
        output_port = MockPort()
        _run_streaming_impl(
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

    def test_streaming_stderr(self):
        """Verify stderr output is captured as StreamData(STDERR)."""
        output_port = MockPort()
        _run_streaming_impl(
            args="echo error >&2",
            shell=True,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
        )

        stderr_msgs = [
            m
            for m in output_port.sent
            if isinstance(m, StreamData) and m.stream == StreamType.STDERR
        ]
        assert len(stderr_msgs) >= 1
        assert b"error" in stderr_msgs[0].data

    def test_streaming_with_stdin(self):
        """Verify stdin channel is set up and data flows through."""
        output_port = MockPort()
        stdin_port = MockPort()
        stdin_receiver = MockPortReceiver()

        def stdin_factory():
            return stdin_port, stdin_receiver

        # Feed stdin data then close in a separate thread
        def feed_stdin():
            import time

            time.sleep(0.1)
            stdin_receiver.push(StdinData(data=b"hello from stdin\n"))
            stdin_receiver.push(StdinClose())

        feeder = threading.Thread(target=feed_stdin, daemon=True)
        feeder.start()

        _run_streaming_impl(
            args="cat",
            shell=True,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=True,
            chunk_size=65536,
            _stdin_channel_factory=stdin_factory,
        )

        feeder.join(timeout=5)

        # Check that StreamStdinReady was sent
        stdin_ready = [m for m in output_port.sent if isinstance(m, StreamStdinReady)]
        assert len(stdin_ready) == 1
        assert stdin_ready[0].stdin_port is stdin_port

        # Check stdout received the echoed input
        data_msgs = [
            m
            for m in output_port.sent
            if isinstance(m, StreamData) and m.stream == StreamType.STDOUT
        ]
        stdout_data = b"".join(m.data for m in data_msgs)
        assert b"hello from stdin" in stdout_data

        # Check exit
        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == 0

    def test_streaming_bad_command(self):
        """Verify StreamError is sent for a bad command."""
        output_port = MockPort()
        _run_streaming_impl(
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

    def test_streaming_large_output(self):
        """Verify large output is chunked correctly."""
        output_port = MockPort()
        # Generate 1MB of output
        _run_streaming_impl(
            args="dd if=/dev/zero bs=1024 count=1024 2>/dev/null",
            shell=True,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=4096,  # Small chunks
        )

        data_msgs = [
            m
            for m in output_port.sent
            if isinstance(m, StreamData) and m.stream == StreamType.STDOUT
        ]
        total = sum(len(m.data) for m in data_msgs)
        assert total == 1024 * 1024  # 1 MB
        # With 4K chunks we expect multiple messages
        assert len(data_msgs) > 1

    def test_streaming_with_control_channel(self):
        """Verify control channel is set up when enable_control=True."""
        output_port = MockPort()
        control_port = MockPort()
        control_receiver = MockPortReceiver()

        def control_factory():
            return control_port, control_receiver

        _run_streaming_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            enable_control=True,
            _control_channel_factory=control_factory,
        )

        # Check StreamControlReady was sent
        control_ready = [
            m for m in output_port.sent if isinstance(m, StreamControlReady)
        ]
        assert len(control_ready) == 1
        assert control_ready[0].control_port is control_port

    def test_streaming_control_kill(self):
        """Verify kill signal via control channel terminates the process."""
        output_port = MockPort()
        control_port = MockPort()
        control_receiver = MockPortReceiver()

        def control_factory():
            return control_port, control_receiver

        # Send SIGKILL after a short delay
        def send_kill():
            import time

            time.sleep(0.2)
            control_receiver.push(ControlSignal(signal_num=signal.SIGKILL))

        killer = threading.Thread(target=send_kill, daemon=True)
        killer.start()

        _run_streaming_impl(
            args=["sleep", "30"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            enable_control=True,
            _control_channel_factory=control_factory,
        )

        killer.join(timeout=5)

        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        # Process killed by SIGKILL returns -9
        assert exit_msgs[0].returncode == -signal.SIGKILL

    def test_streaming_control_kills_process_group(self):
        import time

        output_port = MockPort()
        control_receiver = MockPortReceiver()

        def send_kill():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if any(
                    isinstance(message, StreamData) and b"child-ready" in message.data
                    for message in output_port.sent
                ):
                    break
                time.sleep(0.01)
            control_receiver.push(ControlSignal(signal_num=signal.SIGKILL))

        killer = threading.Thread(target=send_kill, daemon=True)
        killer.start()
        started_at = time.monotonic()
        _run_streaming_impl(
            args=["bash", "-c", "sleep 30 & echo child-ready; wait"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            enable_control=True,
            start_new_session=True,
            _control_channel_factory=lambda: (MockPort(), control_receiver),
        )
        elapsed = time.monotonic() - started_at
        killer.join(timeout=5)

        exit_messages = [
            message for message in output_port.sent if isinstance(message, StreamExit)
        ]
        assert len(exit_messages) == 1
        # macOS bash may reap the killed background job and report 0 before
        # its own group signal is observed.  The bounded elapsed time proves
        # that the descendant did not survive for the full 30 seconds.
        assert elapsed < 5
        assert exit_messages[0].returncode in (0, -signal.SIGKILL)

    def test_streaming_cleans_process_group_after_leader_exits(self):
        import time

        output_port = MockPort()
        started_at = time.monotonic()
        _run_streaming_impl(
            args=["bash", "-c", "sleep 30 & echo child-ready"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            start_new_session=True,
        )
        elapsed = time.monotonic() - started_at

        exit_messages = [
            message for message in output_port.sent if isinstance(message, StreamExit)
        ]
        assert elapsed < 5
        assert len(exit_messages) == 1
        assert exit_messages[0].returncode == 0

    def test_streaming_control_terminate(self):
        """Verify SIGTERM via control channel terminates the process."""
        output_port = MockPort()
        control_port = MockPort()
        control_receiver = MockPortReceiver()

        def control_factory():
            return control_port, control_receiver

        def send_term():
            import time

            time.sleep(0.2)
            control_receiver.push(ControlSignal(signal_num=signal.SIGTERM))

        terminator = threading.Thread(target=send_term, daemon=True)
        terminator.start()

        _run_streaming_impl(
            args=["sleep", "30"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            enable_control=True,
            _control_channel_factory=control_factory,
        )

        terminator.join(timeout=5)

        exit_msgs = [m for m in output_port.sent if isinstance(m, StreamExit)]
        assert len(exit_msgs) == 1
        assert exit_msgs[0].returncode == -signal.SIGTERM

    def test_streaming_no_control_channel_by_default(self):
        """When enable_control is False (default), no control channel is created."""
        output_port = MockPort()
        _run_streaming_impl(
            args=["echo", "hello"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=False,
            chunk_size=65536,
            # enable_control defaults to False
        )

        control_ready = [
            m for m in output_port.sent if isinstance(m, StreamControlReady)
        ]
        assert len(control_ready) == 0

    def test_stdin_poll_reuses_single_receive_while_idle(self):
        """Regression: the stdin poll loop must hold ONE pending recv() across
        timeouts rather than issuing a fresh recv() every poll.

        When a timed-out ``get()`` does not cancel the underlying receive, a
        fresh ``recv()`` per poll leaks pending receives; a stdin message that
        arrives after an idle gap (the first command after a shell-command
        timeout) is then delivered to an abandoned receive and lost, hanging the
        shell.  ``sleep`` ignores stdin, so the loop polls with no messages for
        the process's lifetime — the correct loop issues exactly one ``recv()``.
        """
        output_port = MockPort()
        receiver = _TrackingReceiver()

        _run_streaming_impl(
            args=["sleep", "0.4"],
            shell=False,
            cwd=None,
            env=None,
            output_port=output_port,
            enable_stdin=True,
            chunk_size=65536,
            _stdin_channel_factory=lambda: (MockPort(), receiver),
        )

        # ~8 poll iterations elapsed over 0.4s with no stdin messages; the loop
        # must have issued a single recv() and re-get()'d it, never leaking
        # additional pending receives.  (Pre-fix: recv_calls == number of polls.)
        assert receiver.recv_calls == 1
        assert receiver.max_pending == 1


def test_stream_wire_codec_round_trips_and_stays_small():
    import cloudpickle

    for msg in (StreamData(stream=StreamType.STDERR, data=b"x"), StdinData(data=b"y")):
        wire = streaming.encode(msg)
        assert isinstance(wire, tuple)
        assert streaming.decode(wire) == msg
        # By-value dataclasses (and the StreamType enum) pickle their classes,
        # which the wire form avoids.
        assert len(cloudpickle.dumps(wire)) < len(cloudpickle.dumps(msg))
    exit_msg = StreamExit(returncode=3)
    assert streaming.encode(exit_msg) is exit_msg
    assert streaming.decode(exit_msg) is exit_msg
