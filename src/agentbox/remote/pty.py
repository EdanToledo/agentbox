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

"""Remote PTY execution on Monarch processes.

Mirrors Python's ``pty`` module but runs commands on remote Monarch
ProcMesh workers via an internal actor, providing a PTY-based
alternative to the pipe-based :mod:`agentbox.remote.subprocess`.

A PTY merges stdout and stderr into a single stream (as a real
terminal would), enables line-editing, and honours terminal escape
sequences (colours, cursor movement, etc.).

Example::

    from monarch.actor import this_host
    from agentbox.remote import pty

    host = this_host()
    procs = host.spawn_procs()

    proc = pty.spawn("bash", proc_mesh=procs)
    proc.write(b"echo hello\\n")
    for chunk in proc.stdout:
        print(chunk, end="")
    proc.wait()
"""

from __future__ import annotations

import asyncio
import io
import itertools
import os
import queue
import selectors
import shlex
import signal as _signal
import subprocess as _subprocess
import sys
import threading
import time
from typing import Any, Callable, cast, Iterator, TYPE_CHECKING

import cloudpickle
from agentbox.remote.streaming import (
    ControlSignal,
    decode as _decode_stream,
    encode as _encode_stream,
    StdinClose,
    StdinData,
    StdinMessage,
    StreamControlReady,
    StreamData,
    StreamError,
    StreamExit,
    StreamMessage,
    StreamStdinReady,
    StreamType,
)
from monarch.actor import Actor, Channel, endpoint, Port, PortReceiver, ProcMesh


if TYPE_CHECKING:
    from monarch._src.actor.actor_mesh import ActorMesh


_counter = itertools.count()

_DEFAULT_CHUNK_SIZE = 65536  # 64 KiB
_CONTROLLER_POLL_SECONDS = 0.05
_POST_EXIT_IDLE_SECONDS = 0.05
_POST_EXIT_DRAIN_SECONDS = 0.1
# How long a cancelled endpoint waits for its PTY thread after killing the child.
_CANCEL_JOIN_SECONDS = 5.0

# Sentinel pushed onto the output queue by the drain thread once the process
# has exited, so consumers (stdout iteration / read()) know the stream is done.
_EOF = object()


def _open_stdin_channel() -> tuple[Port[StdinMessage], PortReceiver[StdinMessage]]:
    """Open the channel used to forward stdin to the PTY."""
    return cast(
        tuple[Port[StdinMessage], PortReceiver[StdinMessage]],
        Channel[StdinMessage].open(),
    )


def _open_control_channel() -> tuple[Port[ControlSignal], PortReceiver[ControlSignal]]:
    """Open the channel used to forward signals to the PTY process."""
    return cast(
        tuple[Port[ControlSignal], PortReceiver[ControlSignal]],
        Channel[ControlSignal].open(),
    )


def _read_pty_controller(
    controller_fd: int,
    chunk_size: int,
    output_port: Port[StreamMessage],
    stop_event: threading.Event,
) -> None:
    """Forward output, bounding reads after stop without timing port sends."""
    buffered_after_stop: list[bytes] = []
    drain_deadline: float | None = None
    selector = selectors.DefaultSelector()
    try:
        selector.register(controller_fd, selectors.EVENT_READ)
        while True:
            if stop_event.is_set() and drain_deadline is None:
                drain_deadline = time.monotonic() + _POST_EXIT_DRAIN_SECONDS

            if drain_deadline is None:
                timeout = _CONTROLLER_POLL_SECONDS
            else:
                remaining = drain_deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(_POST_EXIT_IDLE_SECONDS, remaining)

            try:
                events = selector.select(timeout)
            except InterruptedError:
                continue
            except OSError:
                break

            if not events:
                if stop_event.is_set():
                    # If stop was set during the regular poll, start the
                    # bounded drain before treating the PTY as idle.
                    if drain_deadline is None:
                        continue
                    break
                continue

            try:
                data = os.read(controller_fd, chunk_size)
            except OSError:
                break
            if not data:
                break
            if drain_deadline is None:
                output_port.send(
                    _encode_stream(StreamData(stream=StreamType.STDOUT, data=data))
                )
            else:
                # Capture first, then forward after closing the fd. This keeps
                # slow port sends outside the absolute read-drain deadline.
                buffered_after_stop.append(data)
    except Exception:
        pass  # fd closed or broken
    finally:
        selector.close()
        try:
            os.close(controller_fd)
        except OSError:
            pass

    try:
        for data in buffered_after_stop:
            output_port.send(
                _encode_stream(StreamData(stream=StreamType.STDOUT, data=data))
            )
    except Exception:
        pass  # port closed or broken


# PTY actor implementation (testable standalone)


def _run_pty_impl(  # noqa: C901
    args: str | list[str],
    shell: bool,
    cwd: str | None,
    env: dict[str, str] | None,
    output_port: Port[StreamMessage],
    enable_stdin: bool,
    chunk_size: int,
    enable_control: bool = False,
    on_started: Callable[[_subprocess.Popen[bytes]], None] | None = None,
) -> None:
    """Core PTY subprocess logic, factored out for testability.

    Allocates a PTY pair via :func:`os.openpty`, launches the child
    process with the terminal side as stdin/stdout/stderr, and streams
    output from the controller side over the Monarch port.

    Args:
        args: Command to run (string or list).
        shell: If True, run through the shell.
        cwd: Working directory for the child.
        env: Environment variables for the child.
        output_port: Monarch port for sending stream messages.
        enable_stdin: If True, create a stdin channel for writing to the PTY.
        chunk_size: Read chunk size for the controller fd.
        enable_control: If True, create a control channel for signals.
        on_started: Called with the child process right after it starts, so
            the caller can kill it if it has to abandon this function.
    """
    cmd_args: str | list[str]
    if isinstance(args, str) and not shell:
        cmd_args = shlex.split(args)
    else:
        cmd_args = args

    stdin_receiver: PortReceiver[StdinMessage] | None = None
    control_receiver: PortReceiver[ControlSignal] | None = None

    try:
        # Set up stdin channel if requested
        if enable_stdin:
            stdin_port, stdin_receiver = _open_stdin_channel()
            output_port.send(StreamStdinReady(stdin_port=stdin_port))

        # Set up control channel if requested
        if enable_control:
            control_port, control_receiver = _open_control_channel()
            output_port.send(StreamControlReady(control_port=control_port))

        # Allocate PTY pair
        controller_fd, terminal_fd = os.openpty()

        try:
            proc = _subprocess.Popen(
                cmd_args,
                shell=shell,
                cwd=cwd,
                env=env,
                stdin=terminal_fd,
                stdout=terminal_fd,
                stderr=terminal_fd,
                start_new_session=True,
            )
        except Exception:
            # Popen failed (bad command, cwd, etc.) — the child never took
            # ownership of the fds, so close both here to avoid leaking them
            # on the worker (a failed spawn must not exhaust the fd table).
            for fd in (controller_fd, terminal_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

        # Close the terminal endpoint in the parent — the child owns it now.
        os.close(terminal_fd)
        if on_started is not None:
            on_started(proc)
    except Exception as exc:
        output_port.send(StreamError(error=str(exc)))
        output_port.send(StreamExit(returncode=-1))
        return

    def _write_stdin(receiver: PortReceiver[StdinMessage]) -> None:
        """Write stdin data to the PTY controller fd."""
        try:
            while True:
                msg = _decode_stream(receiver.recv().get())
                if isinstance(msg, StdinClose):
                    # Send EOF character (Ctrl+D) to the PTY to signal end-of-input
                    try:
                        os.write(controller_fd, b"\x04")
                    except OSError:
                        pass
                    break
                if isinstance(msg, StdinData):
                    os.write(controller_fd, msg.data)
        except Exception:
            pass  # port closed or process exited

    def _handle_control(receiver: PortReceiver[ControlSignal], process: Any) -> None:
        """Handle control signals (kill, terminate, etc.)."""
        try:
            while True:
                msg = _decode_stream(receiver.recv().get())
                if isinstance(msg, ControlSignal):
                    try:
                        process.send_signal(msg.signal_num)
                    except OSError:
                        pass  # process already exited
        except Exception:
            pass  # channel closed or process done

    # The reader owns the controller fd. After the child exits, the stop event
    # wakes its polling loop so it can drain buffered output and close the fd
    # itself; cross-thread close does not reliably interrupt read() on Linux.
    reader_stop = threading.Event()
    reader_thread = threading.Thread(
        target=_read_pty_controller,
        args=(controller_fd, chunk_size, output_port, reader_stop),
        daemon=True,
    )
    reader_thread.start()

    # Start stdin writer thread if enabled
    stdin_thread: threading.Thread | None = None
    if enable_stdin and stdin_receiver is not None:
        stdin_thread = threading.Thread(
            target=_write_stdin, args=(stdin_receiver,), daemon=True
        )
        stdin_thread.start()

    # Start control handler thread if enabled
    control_thread: threading.Thread | None = None
    if enable_control and control_receiver is not None:
        control_thread = threading.Thread(
            target=_handle_control, args=(control_receiver, proc), daemon=True
        )
        control_thread.start()

    # Wait for process to exit
    returncode = proc.wait()

    # StreamExit is the protocol terminator. Wait until the reader has stopped
    # so all captured output precedes it. An idle timeout handles descendants
    # holding the terminal open, while an absolute read deadline handles an
    # escaped descendant that keeps writing continuously.
    reader_stop.set()
    reader_thread.join()

    output_port.send(StreamExit(returncode=returncode))


class _PtyActor(Actor):
    """Runs a PTY subprocess on a remote Monarch process."""

    @endpoint
    async def run_pty(
        self,
        args: str | list[str],
        shell: bool,
        cwd: str | None,
        env: dict[str, str] | None,
        output_port: Port[StreamMessage],
        enable_stdin: bool,
        chunk_size: int,
        enable_control: bool = False,
    ) -> None:
        """Run the PTY on a thread so stopping the actor can cancel this endpoint.

        A synchronous endpoint blocked in ``proc.wait()`` cannot be stopped:
        Monarch's actor cleanup deadline elapses and the client receives an
        unhandled actor failure. On cancellation, kill the child's process
        group (it runs in its own session), which unblocks the thread. The
        cancel can arrive before the thread has started the child, so the
        start callback also kills a child that starts after cancellation.
        """
        lock = threading.Lock()
        started: list[_subprocess.Popen[bytes]] = []
        cancelled = False

        def kill_group(proc: _subprocess.Popen[bytes]) -> None:
            try:
                os.killpg(proc.pid, _signal.SIGKILL)
            except OSError:
                pass  # process group already exited

        def on_started(proc: _subprocess.Popen[bytes]) -> None:
            with lock:
                started.append(proc)
                kill = cancelled
            if kill:
                kill_group(proc)

        work = asyncio.ensure_future(
            asyncio.to_thread(
                _run_pty_impl,
                args,
                shell,
                cwd,
                env,
                output_port,
                enable_stdin,
                chunk_size,
                enable_control,
                on_started,
            )
        )
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            with lock:
                cancelled = True
                running = list(started)
            for proc in running:
                kill_group(proc)
            try:
                await asyncio.wait_for(work, timeout=_CANCEL_JOIN_SECONDS)
            except BaseException:  # noqa: B036 - preserve the active cancellation
                pass
            raise


# Register so cloudpickle serializes the actor class by value to remote workers.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _PtyStdinWriter:
    """Writable file-like wrapper for :class:`PtyProcess` stdin.

    Provides ``write()`` / ``close()`` so that ``proc.stdin.write(data)``
    mirrors the subprocess API.
    """

    def __init__(self, owner: PtyProcess) -> None:
        self._owner = owner
        self._closed = False

    def write(self, data: bytes | str) -> int:
        """Write *data* to the remote PTY controller fd."""
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        self._owner.write(data)
        if isinstance(data, str):
            return len(
                data.encode(self._owner._encoding or "utf-8", self._owner._errors)
            )
        return len(data)

    def close(self) -> None:
        """Close the remote PTY stdin channel."""
        if not self._closed:
            self._closed = True
            self._owner._close_stdin()

    def flush(self) -> None:
        pass  # data sent immediately

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")

    @property
    def closed(self) -> bool:
        return self._closed


class PtyProcess:
    """Remote PTY process handle.

    Returned by :func:`spawn`. Provides bidirectional I/O with a remote
    process running inside a PTY.

    The PTY merges stdout and stderr into a single stream, just like a
    real terminal. Output is available via :attr:`stdout` (an iterator)
    or :meth:`read`/:meth:`communicate`.
    """

    DEFAULT_CHUNK_SIZE = _DEFAULT_CHUNK_SIZE

    def __init__(
        self,
        args: str | list[str],
        *,
        proc_mesh: ProcMesh,
        shell: bool = False,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        if encoding is not None:
            text = True
        self.args = args
        self._text = text
        self._encoding = encoding or ("utf-8" if text else None)
        self._errors = errors or "strict"
        self._returncode: int | None = None
        self._exited = False

        # A single background drain thread pumps the output receiver into
        # ``_out_queue`` and flips ``_exit_event`` on StreamExit.  This lets
        # multiple consumers coexist safely — e.g. a thread iterating
        # :attr:`stdout` while another calls :meth:`wait` — without racing for
        # messages on the single-consumer receiver (which would deadlock
        # whichever caller lost the race for the StreamExit message).
        self._out_queue: queue.Queue[Any] = queue.Queue()
        self._exit_event = threading.Event()
        self._drain_error: str | None = None
        self._drain_thread: threading.Thread | None = None
        self._drain_started = False
        self._drain_lock = threading.Lock()

        # Ports (set when received from actor)
        self._stdin_port: Port[StdinMessage] | None = None
        self._control_port: Port[ControlSignal] | None = None

        # Stdin wrapper
        self.stdin: _PtyStdinWriter | None = _PtyStdinWriter(self)

        # Create output channel
        output_port, self._output_receiver = Channel[StreamMessage].open()

        # Spawn actor and start PTY
        from agentbox.actors.mesh_schema import make_name

        name = make_name("pty", str(next(_counter)))
        self._actor_mesh: ActorMesh[_PtyActor] = proc_mesh.spawn(name, _PtyActor)
        self._actor_mesh.run_pty.broadcast(
            args,
            shell,
            cwd,
            env,
            output_port,
            True,  # enable_stdin — always on for PTY
            chunk_size,
            True,  # enable_control — always on for kill/terminate
        )

        # Wait for stdin and control ports before returning, so that
        # write()/kill() never race with the stdout reader for messages.
        ports_needed = 2  # stdin + control
        ports_found = 0
        while ports_found < ports_needed:
            msg = _decode_stream(self._output_receiver.recv().get())
            if isinstance(msg, StreamStdinReady):
                self._stdin_port = msg.stdin_port
                ports_found += 1
            elif isinstance(msg, StreamControlReady):
                self._control_port = msg.control_port
                ports_found += 1
            elif isinstance(msg, StreamError):
                raise RuntimeError(f"PTY actor failed during startup: {msg.error}")
            elif isinstance(msg, StreamExit):
                self._returncode = msg.returncode
                self._exited = True
                raise RuntimeError(
                    f"PTY process exited during startup with code {msg.returncode}"
                )

    def _drain_loop(self) -> None:
        """Pump the output receiver into ``_out_queue`` until StreamExit.

        Runs on a single background thread so that ``stdout`` iteration,
        :meth:`read`, and :meth:`wait` all read from shared state instead of
        each calling ``recv()`` directly (which lets concurrent consumers
        steal each other's messages and deadlock).  Raw ``bytes`` are queued;
        decoding happens in the consumer so multibyte characters split across
        chunks are not corrupted.
        """
        while True:
            msg = _decode_stream(self._output_receiver.recv().get())
            if isinstance(msg, (StreamStdinReady, StreamControlReady)):
                continue
            if isinstance(msg, StreamData):
                self._out_queue.put(msg.data)
            elif isinstance(msg, StreamError):
                self._drain_error = msg.error
            elif isinstance(msg, StreamExit):
                self._returncode = msg.returncode
                self._exited = True
                self._exit_event.set()
                self._out_queue.put(_EOF)
                break

    def _ensure_draining(self) -> None:
        """Start the drain thread on first use (idempotent, thread-safe)."""
        with self._drain_lock:
            if self._drain_started:
                return
            self._drain_started = True
            self._drain_thread = threading.Thread(
                target=self._drain_loop,
                name=f"pty-drain-{id(self):x}",
                daemon=True,
            )
            self._drain_thread.start()

    def _iter_raw(self) -> Iterator[bytes]:
        """Yield raw output chunks until the process exits."""
        self._ensure_draining()
        while True:
            item = self._out_queue.get()
            if item is _EOF:
                self._out_queue.put(_EOF)  # keep sentinel sticky for later consumers
                return
            yield item

    def _iter_output(self) -> Iterator[bytes] | Iterator[str]:
        """Iterate over output chunks. Yields ``str`` if ``text=True``."""
        for data in self._iter_raw():
            yield (
                data.decode(self._encoding or "utf-8", self._errors)
                if self._text
                else data
            )

    @property
    def stdout(self) -> Iterator[bytes] | Iterator[str]:
        """Iterator over output chunks from the PTY.

        Since a PTY merges stdout/stderr, this yields all output.
        """
        return self._iter_output()

    def write(self, data: bytes | str) -> None:
        """Write *data* to the PTY controller fd on the remote process."""
        if self._exited:
            raise RuntimeError("Cannot write to PTY after process has exited")
        if isinstance(data, str):
            data = data.encode(self._encoding or "utf-8", self._errors)
        self._stdin_port.send(_encode_stream(StdinData(data=data)))  # type: ignore[union-attr]

    def _close_stdin(self) -> None:
        """Close the stdin channel to the PTY."""
        self._stdin_port.send(StdinClose())  # type: ignore[union-attr]

    def read(self) -> bytes | str:
        """Read all remaining output from the PTY until the process exits.

        Returns ``str`` if ``text=True``, ``bytes`` otherwise.
        """
        result = b"".join(self._iter_raw())
        if self._text:
            return result.decode(self._encoding or "utf-8", self._errors)
        return result

    def communicate(
        self, input: bytes | str | None = None
    ) -> tuple[bytes | str, bytes | str]:
        """Send *input* to the PTY and collect all output.

        Args:
            input: Data to write to the PTY. If None, no input is sent
                but stdin is closed to signal EOF.

        Returns:
            ``(output, "")``.  The second element is always empty since
            a PTY merges stdout and stderr.
        """
        if input is not None:
            self.write(input)
        self._close_stdin()

        output = self.read()
        empty: bytes | str = "" if self._text else b""
        return output, empty

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the remote process to exit and return the exit code.

        Safe to call concurrently with :attr:`stdout` iteration or :meth:`read`
        — they share a single drain thread.  Raises :class:`TimeoutError` if
        *timeout* elapses before the process exits.
        """
        if self._returncode is not None:
            return self._returncode
        self._ensure_draining()
        if not self._exit_event.wait(timeout):
            raise TimeoutError(f"PTY process did not exit within {timeout} seconds")
        assert self._returncode is not None
        return self._returncode

    def poll(self) -> int | None:
        """Check if the remote process has finished. Returns returncode or ``None``."""
        self._ensure_draining()
        return self._returncode

    def send_signal(self, sig: int) -> None:
        """Send a signal to the remote process."""
        if self._exited:
            return
        self._control_port.send(ControlSignal(signal_num=sig))  # type: ignore[union-attr]

    def terminate(self) -> None:
        """Terminate the remote process by sending SIGTERM."""
        self.send_signal(_signal.SIGTERM)

    def kill(self) -> None:
        """Kill the remote process by sending SIGKILL."""
        self.send_signal(_signal.SIGKILL)

    @property
    def returncode(self) -> int | None:
        """The exit code of the remote process, or ``None`` if not yet finished."""
        return self._returncode

    @property
    def pid(self) -> None:
        """Not available for remote processes."""
        return None

    def __enter__(self) -> PtyProcess:
        return self

    def __exit__(self, *args: Any) -> None:
        self.wait()


def spawn(
    argv: str | list[str],
    *,
    proc_mesh: ProcMesh,
    shell: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    text: bool = True,
    encoding: str | None = None,
    errors: str | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> PtyProcess:
    """Spawn a command in a remote PTY.

    Mirrors CPython's :func:`pty.spawn` but returns a :class:`PtyProcess`
    handle instead of an exit code, since we need bidirectional I/O over
    Monarch channels.

    Args:
        argv: Command to run (string or list of arguments).
        proc_mesh: Monarch ProcMesh to execute on.
        shell: If True, run through the shell.
        cwd: Working directory on the remote process.
        env: Environment variables for the remote process.
        text: If True (default), decode output as strings.
        encoding: Character encoding for text mode.
        errors: Error handling for text decoding.
        chunk_size: Read chunk size for the PTY controller fd.

    Returns:
        A :class:`PtyProcess` handle for the remote PTY process.
    """
    return PtyProcess(
        argv,
        proc_mesh=proc_mesh,
        shell=shell,
        cwd=cwd,
        env=env,
        text=text,
        encoding=encoding,
        errors=errors,
        chunk_size=chunk_size,
    )


__all__ = ["spawn", "PtyProcess"]
