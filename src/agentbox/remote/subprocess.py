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

"""Remote subprocess execution on Monarch processes.

Mirrors Python's subprocess API but runs commands on remote Monarch
ProcMesh workers via an internal actor.

Example::

    from monarch.actor import this_host
    from agentbox.remote import subprocess

    host = this_host()
    procs = host.spawn_procs()

    result = subprocess.run(["ls", "-la"], proc_mesh=procs, capture_output=True)
    print(result.stdout)
"""

from __future__ import annotations

import asyncio
import codecs
import io
import itertools
import locale
import logging
import os
import queue
import shlex
import signal as _signal
import subprocess as _subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, cast, Generic, Iterator, TYPE_CHECKING, TypeVar

import cloudpickle
from agentbox.remote.streaming import (
    ControlSignal,
    decode as _decode_stream,
    encode as _encode_stream,
    StdinClose,  # noqa: E402
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
from agentbox.tasks.api import Task, TaskPool
from agentbox.tasks.registry import default_task_handlers
from monarch.actor import Actor, Channel, endpoint, Port, PortReceiver, ProcMesh


if TYPE_CHECKING:
    from monarch._src.actor.actor_mesh import ActorMesh


logger = logging.getLogger(__name__)


# Constants (mirroring subprocess.PIPE, subprocess.STDOUT, subprocess.DEVNULL)

PIPE = _subprocess.PIPE
STDOUT = _subprocess.STDOUT
DEVNULL = _subprocess.DEVNULL


class SubprocessError(Exception):
    """Base class for remote subprocess exceptions."""


class CalledProcessError(SubprocessError):
    """Raised when a remote process returns a non-zero exit code."""

    def __init__(
        self,
        returncode: int,
        cmd: str | list[str],
        output: str | bytes | None = None,
        stderr: str | bytes | None = None,
    ) -> None:
        self.returncode = returncode
        self.cmd = cmd
        self.output = output
        self.stdout = output
        self.stderr = stderr
        super().__init__(f"Command {cmd!r} returned non-zero exit status {returncode}.")

    def __reduce__(self) -> tuple:
        return (self.__class__, (self.returncode, self.cmd, self.output, self.stderr))


class TimeoutExpired(SubprocessError):
    """Raised when a remote process exceeds its timeout."""

    def __init__(
        self,
        cmd: str | list[str],
        timeout: float,
        output: str | bytes | None = None,
        stderr: str | bytes | None = None,
    ) -> None:
        self.cmd = cmd
        self.timeout = timeout
        self.output = output
        self.stdout = output
        self.stderr = stderr
        super().__init__(f"Command {cmd!r} timed out after {timeout} seconds.")

    def __reduce__(self) -> tuple:
        return (self.__class__, (self.cmd, self.timeout, self.output, self.stderr))


_OutputT = TypeVar("_OutputT", str, bytes)


@dataclass
class CompletedProcess(Generic[_OutputT]):
    """Result of a completed remote subprocess, mirroring subprocess.CompletedProcess.

    ``stdout`` and ``stderr`` are the captured output: strings in text mode
    and bytes in binary mode.
    """

    args: str | list[str]
    returncode: int
    stdout: _OutputT
    stderr: _OutputT

    def check_returncode(self) -> None:
        """Raise CalledProcessError if returncode is non-zero."""
        if self.returncode != 0:
            raise CalledProcessError(
                self.returncode, self.args, output=self.stdout, stderr=self.stderr
            )


_counter = itertools.count()
_ACTOR_STOP_TIMEOUT = 10.0


def _run_command_impl(  # noqa: C901
    args: str | list[str],
    shell: bool,
    cwd: str | None,
    env: dict[str, str] | None,
    capture_output: bool,
    timeout: float | None,
    text: bool,
    input: str | bytes | None = None,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    encoding: str | None = None,
    errors: str | None = None,
) -> dict[str, Any]:
    """Execute a subprocess command and return a serializable result dict."""
    cmd_args: str | list[str]
    if isinstance(args, str) and not shell:
        cmd_args = shlex.split(args)
    else:
        cmd_args = args

    kwargs: dict[str, Any] = {
        "shell": shell,
        "cwd": cwd,
        "env": env,
        "text": text,
    }

    if encoding is not None:
        kwargs["encoding"] = encoding
    if errors is not None:
        kwargs["errors"] = errors

    # capture_output is shorthand for stdout=PIPE, stderr=PIPE
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("capture_output cannot be used with stdout or stderr")
        kwargs["stdout"] = _subprocess.PIPE
        kwargs["stderr"] = _subprocess.PIPE
    else:
        if stdout is not None:
            kwargs["stdout"] = stdout
        if stderr is not None:
            kwargs["stderr"] = stderr

    # input implies stdin=PIPE (match stdlib behaviour)
    if input is not None:
        kwargs["input"] = input
    elif stdin is not None:
        kwargs["stdin"] = stdin

    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        result = _subprocess.run(cmd_args, **kwargs)
    except _subprocess.TimeoutExpired as exc:
        return {
            "timeout_expired": True,
            "cmd": args,
            "timeout": timeout,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }

    return {
        "timeout_expired": False,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _decode_async_output(
    data: bytes | None,
    *,
    text: bool,
    encoding: str | None,
    errors: str | None,
) -> str | bytes | None:
    """Decode asyncio subprocess output using ``subprocess.run`` semantics."""
    if data is None or not (text or encoding is not None or errors is not None):
        return data
    decoded = data.decode(encoding or locale.getencoding(), errors or "strict")
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


def _kill_command(process: asyncio.subprocess.Process) -> None:
    """Kill *process* and, on POSIX, every process left in its group.

    A ``str`` command runs under ``/bin/sh -c``, which may fork the real
    command; killing only the shell would leave that child running and
    holding the output pipes open, so draining them would never finish.
    """
    if os.name == "posix":
        try:
            os.killpg(process.pid, _signal.SIGKILL)
            return
        except ProcessLookupError:
            return  # the whole group has already exited
        except PermissionError:
            pass  # pid reused outside our group; fall back to the process
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass


async def _akill_and_communicate(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Kill *process* and its group, then drain its pipes and reap it."""
    _kill_command(process)
    return await asyncio.shield(communication)


async def _arun_command_impl(  # noqa: C901
    args: str | list[str],
    shell: bool,
    cwd: str | None,
    env: dict[str, str] | None,
    capture_output: bool,
    timeout: float | None,
    text: bool,
    input: str | bytes | None = None,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    encoding: str | None = None,
    errors: str | None = None,
) -> dict[str, Any]:
    """Async counterpart of :func:`_run_command_impl`.

    The child process and its pipe I/O are supervised by the current asyncio
    loop, allowing one thread to drive multiple commands concurrently.
    """
    cmd_args: str | list[str]
    if isinstance(args, str) and not shell:
        cmd_args = shlex.split(args)
    else:
        cmd_args = args

    if capture_output and (stdout is not None or stderr is not None):
        raise ValueError("capture_output cannot be used with stdout or stderr")

    child_stdin = _subprocess.PIPE if input is not None else stdin
    child_stdout = _subprocess.PIPE if capture_output else stdout
    child_stderr = _subprocess.PIPE if capture_output else stderr
    encoded_input: Any = input
    if input is not None and (text or encoding is not None or errors is not None):
        # Deliberately let bytes input raise AttributeError, matching
        # subprocess.run(..., text=True).
        encoded_input = input.encode(  # type: ignore[union-attr]
            encoding or locale.getencoding(), errors or "strict"
        )
    elif input is not None and not isinstance(input, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"memoryview: a bytes-like object is required, not '{type(input).__name__}'"
        )

    spawn_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdin": child_stdin,
        "stdout": child_stdout,
        "stderr": child_stderr,
        # Lead a new process group so cancellation and timeouts can kill
        # everything the command started (see _kill_command).
        "start_new_session": os.name == "posix",
    }

    if shell:
        if isinstance(cmd_args, str):
            process = await asyncio.create_subprocess_shell(cmd_args, **spawn_kwargs)
        elif os.name == "posix":
            process = await asyncio.create_subprocess_exec(
                "/bin/sh", "-c", *cmd_args, **spawn_kwargs
            )
        else:
            process = await asyncio.create_subprocess_shell(
                _subprocess.list2cmdline(cmd_args), **spawn_kwargs
            )
    else:
        if isinstance(cmd_args, str):
            raise TypeError("command arguments were not normalized")
        process = await asyncio.create_subprocess_exec(*cmd_args, **spawn_kwargs)

    communication = asyncio.create_task(process.communicate(encoded_input))
    stdout_data: bytes | None
    stderr_data: bytes | None
    try:
        if timeout is None:
            stdout_data, stderr_data = await asyncio.shield(communication)
        else:
            stdout_data, stderr_data = await asyncio.wait_for(
                asyncio.shield(communication), timeout=timeout
            )
    except TimeoutError:
        stdout_data, stderr_data = await _akill_and_communicate(process, communication)
        # TimeoutExpired exposes captured output as bytes even in text mode.
        return {
            "timeout_expired": True,
            "cmd": args,
            "timeout": timeout,
            "stdout": stdout_data or None,
            "stderr": stderr_data or None,
        }
    except asyncio.CancelledError:
        try:
            await _akill_and_communicate(process, communication)
        except BaseException:  # noqa: B036 - preserve the active cancellation
            logger.debug("Error cleaning up cancelled async subprocess", exc_info=True)
        raise
    except BaseException:
        _kill_command(process)
        await process.wait()
        raise

    return {
        "timeout_expired": False,
        "returncode": process.returncode,
        "stdout": _decode_async_output(
            stdout_data, text=text, encoding=encoding, errors=errors
        ),
        "stderr": _decode_async_output(
            stderr_data, text=text, encoding=encoding, errors=errors
        ),
    }


class _SubprocessActor(Actor):
    """Runs subprocess commands on a remote Monarch process."""

    @endpoint
    async def run_command(
        self,
        args: str | list[str],
        shell: bool,
        cwd: str | None,
        env: dict[str, str] | None,
        capture_output: bool,
        timeout: float | None,
        text: bool,
        input: str | bytes | None = None,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> dict[str, Any]:
        # Async so stopping the actor cancels the command, which kills it; a
        # blocking endpoint could not be stopped and orphaned the process.
        return await _arun_command_impl(
            args,
            shell,
            cwd,
            env,
            capture_output,
            timeout,
            text,
            input,
            stdin,
            stdout,
            stderr,
            encoding,
            errors,
        )


@dataclass
class _SubprocessTask(Task[dict[str, Any]]):
    kind = "subprocess"

    args: str | list[str]
    shell: bool
    cwd: str | None
    env: dict[str, str] | None
    capture_output: bool
    timeout: float | None
    text: bool
    input: str | bytes | None
    stdin: int | None
    stdout: int | None
    stderr: int | None
    encoding: str | None
    errors: str | None

    def to_payload(self) -> tuple[Any, ...]:
        """Return a built-in-only envelope that avoids cloudpickle class state."""
        return (
            self.args,
            self.shell,
            self.cwd,
            self.env,
            self.capture_output,
            self.timeout,
            self.text,
            self.input,
            self.stdin,
            self.stdout,
            self.stderr,
            self.encoding,
            self.errors,
        )


default_task_handlers.register(_SubprocessTask.kind, _arun_command_impl)


# Register so cloudpickle serializes the actor class by value to remote workers.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _run(
    args: str | list[str],
    *,
    proc_mesh: ProcMesh,
    task_pool: TaskPool | None,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    shell: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    input: str | bytes | None = None,
    timeout: float | None = None,
    check: bool = False,
    text: bool = True,
    encoding: str | None = None,
    errors: str | None = None,
) -> CompletedProcess[str] | CompletedProcess[bytes]:
    """Internal :func:`run` implementation with an optional shared task pool."""
    if encoding is not None:
        text = True

    request = _SubprocessTask(
        args,
        shell,
        cwd,
        env,
        capture_output,
        timeout,
        text,
        input,
        stdin,
        stdout,
        stderr,
        encoding,
        errors,
    )
    if task_pool is not None:
        pooled_result = task_pool.submit(request)
        return _finalize_run(args, pooled_result, check)

    # Low-level calls without a Machine-owned task pool retain the isolated
    # actor behavior and always release their per-command actor state.
    actor_mesh = _spawn_run_actor(proc_mesh)
    try:
        future = _issue_run_command(actor_mesh, request)
        result: dict[str, Any] = future.get()
        return _finalize_run(args, result, check)
    finally:
        _stop_run_actor(actor_mesh)


def _spawn_run_actor(proc_mesh: ProcMesh) -> ActorMesh[_SubprocessActor]:
    """Spawn an isolated actor for one command."""
    from agentbox.actors.mesh_schema import make_name

    name = make_name("subprocess", str(next(_counter)))
    return proc_mesh.spawn(name, _SubprocessActor)


def _issue_run_command(
    actor_mesh: ActorMesh[_SubprocessActor], request: _SubprocessTask
) -> Any:
    """Dispatch one command to an already-owned isolated actor.

    Shared by the synchronous :func:`run` and the async :func:`arun` so the
    endpoint argument list is written once. The caller owns *actor_mesh* before
    entering this function, so synchronous dispatch errors are covered by its
    teardown ``finally`` block.
    """
    return actor_mesh.run_command.choose(
        request.args,
        request.shell,
        request.cwd,
        request.env,
        request.capture_output,
        request.timeout,
        request.text,
        request.input,
        request.stdin,
        request.stdout,
        request.stderr,
        request.encoding,
        request.errors,
    )


def _stop_run_actor(
    actor_mesh: ActorMesh[_SubprocessActor],
    *,
    timeout: float = _ACTOR_STOP_TIMEOUT,
) -> None:
    """Best-effort, bounded teardown for a one-shot subprocess actor."""
    try:
        actor_mesh.stop().get(timeout=timeout)
    except Exception:
        logger.debug("Error stopping one-shot subprocess actor", exc_info=True)


async def _astop_run_actor(
    actor_mesh: ActorMesh[_SubprocessActor],
    *,
    timeout: float = _ACTOR_STOP_TIMEOUT,
) -> None:
    """Async-native counterpart of :func:`_stop_run_actor`."""
    try:
        await asyncio.wait_for(actor_mesh.stop(), timeout=timeout)
    except Exception:
        logger.debug("Error stopping one-shot subprocess actor", exc_info=True)


def _finalize_run(
    args: str | list[str], result: dict[str, Any], check: bool
) -> CompletedProcess[str] | CompletedProcess[bytes]:
    """Turn a ``run_command`` result dict into a :class:`CompletedProcess`.

    Raises :class:`TimeoutExpired` when the remote command timed out, and
    :class:`CalledProcessError` when *check* is set and the exit code is
    non-zero.  Shared by :func:`run` and :func:`arun`.
    """
    if result.get("timeout_expired"):
        raise TimeoutExpired(
            cmd=result["cmd"],
            timeout=result["timeout"],
            output=result.get("stdout"),
            stderr=result.get("stderr"),
        )

    completed = CompletedProcess(
        args=args,
        returncode=result["returncode"],
        stdout=result["stdout"],
        stderr=result["stderr"],
    )

    if check:
        completed.check_returncode()

    return completed


async def _arun(
    args: str | list[str],
    *,
    proc_mesh: ProcMesh,
    task_pool: TaskPool | None,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    shell: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    input: str | bytes | None = None,
    timeout: float | None = None,
    check: bool = False,
    text: bool = True,
    encoding: str | None = None,
    errors: str | None = None,
) -> CompletedProcess[str] | CompletedProcess[bytes]:
    """Internal :func:`arun` implementation with an optional shared task pool."""
    if encoding is not None:
        text = True

    request = _SubprocessTask(
        args,
        shell,
        cwd,
        env,
        capture_output,
        timeout,
        text,
        input,
        stdin,
        stdout,
        stderr,
        encoding,
        errors,
    )
    if task_pool is not None:
        pooled_result = await task_pool.asubmit(request)
        return _finalize_run(args, pooled_result, check)

    # Low-level calls without a Machine-owned task pool retain the isolated
    # actor behavior and always release their per-command actor state.
    actor_mesh = _spawn_run_actor(proc_mesh)
    try:
        future = _issue_run_command(actor_mesh, request)
        result: dict[str, Any] = await future
        return _finalize_run(args, result, check)
    finally:
        await _astop_run_actor(actor_mesh)


# Convenience wrappers (mirroring subprocess.call / check_call / check_output)


# Streaming message protocol (re-exported from agentbox.remote.streaming)


async def _arun_streaming_impl(  # noqa: C901
    args: str | list[str],
    shell: bool,
    cwd: str | None,
    env: dict[str, str] | None,
    output_port: Port[StreamMessage],
    enable_stdin: bool,
    chunk_size: int,
    enable_control: bool = False,
    stdout: int | None = PIPE,
    stderr: int | None = PIPE,
    start_new_session: bool = False,
    *,
    _stdin_channel_factory: Any = None,
    _control_channel_factory: Any = None,
) -> None:
    """Async core for a streaming subprocess actor endpoint.

    Args:
        enable_control: If True, create a control channel for kill/terminate.
        stdout: How to handle stdout — ``PIPE`` (default) to stream, ``DEVNULL``
            to suppress, or ``None`` to inherit.
        stderr: How to handle stderr — ``PIPE`` (default) to stream, ``DEVNULL``
            to suppress, ``STDOUT`` to merge into stdout, or ``None`` to inherit.
        start_new_session: Start the child in a new process session so control
            signals can terminate its complete process group. Remaining group
            members are killed before the streaming actor exits.
        _stdin_channel_factory: If provided, called with no args to return
            ``(port, receiver)`` instead of using ``Channel``. For testing only.
        _control_channel_factory: If provided, called with no args to return
            ``(port, receiver)`` instead of using ``Channel``. For testing only.
    """
    import time as _time

    _t0 = _time.monotonic()

    cmd_args: str | list[str]
    if isinstance(args, str) and not shell:
        cmd_args = shlex.split(args)
    else:
        cmd_args = args

    logger.info("Streaming subprocess starting: %s (shell=%s)", cmd_args, shell)

    stdin_receiver: PortReceiver[StdinMessage] | None = None
    control_receiver: PortReceiver[ControlSignal] | None = None

    try:
        # Set up stdin channel if requested
        if enable_stdin:
            if _stdin_channel_factory is not None:
                stdin_port, stdin_receiver = _stdin_channel_factory()
            else:
                stdin_port, stdin_receiver = Channel[StdinMessage].open()
            output_port.send(StreamStdinReady(stdin_port=stdin_port))
            logger.info("Stdin channel ready")

        # Set up control channel if requested
        if enable_control:
            if _control_channel_factory is not None:
                control_port, control_receiver = _control_channel_factory()
            else:
                control_port, control_receiver = Channel[ControlSignal].open()
            output_port.send(StreamControlReady(control_port=control_port))
            logger.info("Control channel ready")

        proc = _subprocess.Popen(
            cmd_args,
            shell=shell,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            stdin=_subprocess.PIPE if enable_stdin else None,
            start_new_session=start_new_session,
        )
        logger.info("Subprocess started, pid=%s", proc.pid)
    except Exception as exc:
        logger.exception("Streaming subprocess setup failed: %s", exc)
        output_port.send(StreamError(error=str(exc)))
        output_port.send(StreamExit(returncode=-1))
        return

    def _send_process_signal(signal_num: int) -> None:
        try:
            if start_new_session:
                os.killpg(proc.pid, signal_num)
            else:
                proc.send_signal(signal_num)
        except OSError:
            pass  # process or process group already exited

    def _read_stream(pipe: Any, stream_type: StreamType) -> None:
        try:
            while True:
                # Use read1() to return as soon as any data is available
                # rather than blocking until the full chunk_size is filled.
                # This is critical for streaming use-cases like tail -f where
                # data arrives in small increments.
                data = pipe.read1(chunk_size)
                if not data:
                    break
                output_port.send(
                    _encode_stream(StreamData(stream=stream_type, data=data))
                )
        except Exception:
            pass  # pipe closed or broken

    async def _await_receive(future: Any) -> Any:
        """Await a Monarch receive, with a polling fallback for test doubles."""
        if hasattr(future, "__await__"):
            return await future
        while True:
            try:
                return future.get(timeout=0)
            except TimeoutError:
                await asyncio.sleep(0.01)

    async def _join_reader(thread: threading.Thread | None) -> None:
        if thread is None:
            return
        while thread.is_alive():
            await asyncio.sleep(0.01)
        thread.join()

    def _force_cleanup() -> None:
        """Best-effort bounded cleanup when the endpoint is cancelled or fails."""
        _send_process_signal(_signal.SIGKILL)
        try:
            proc.wait(timeout=1.0)
        except Exception:
            logger.debug("Error reaping streaming subprocess", exc_info=True)
        for thread in (stdout_thread, stderr_thread):
            if thread is not None:
                try:
                    thread.join(timeout=1.0)
                except RuntimeError:
                    logger.debug("Error joining streaming reader", exc_info=True)

    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    stdin_task: asyncio.Task[Any] | None = None
    control_task: asyncio.Task[Any] | None = None
    cleanup_complete = False

    try:
        if proc.stdout is not None:
            stdout_thread = threading.Thread(
                target=_read_stream, args=(proc.stdout, StreamType.STDOUT), daemon=True
            )
            stdout_thread.start()

        if proc.stderr is not None:
            stderr_thread = threading.Thread(
                target=_read_stream,
                args=(proc.stderr, StreamType.STDERR),
                daemon=True,
            )
            stderr_thread.start()

        # Handle stdin and control on the actor's asyncio loop. Keeping the
        # endpoint async is important: ActorMesh.stop() can then cancel it and
        # run the process cleanup below even if the client never received the
        # control port. Each channel keeps one pending receive task across poll
        # intervals so messages cannot be delivered to abandoned receives.
        stdin_open = (
            enable_stdin and stdin_receiver is not None and proc.stdin is not None
        )
        control_open = enable_control and control_receiver is not None
        control_was_enabled = control_open

        while stdin_open or control_open:
            if proc.poll() is not None:
                break

            if stdin_open and stdin_task is None:
                stdin_task = asyncio.create_task(
                    _await_receive(stdin_receiver.recv())  # type: ignore[union-attr]
                )
            if control_open and control_task is None:
                control_task = asyncio.create_task(
                    _await_receive(control_receiver.recv())  # type: ignore[union-attr]
                )

            pending = {task for task in (stdin_task, control_task) if task is not None}
            done, _ = await asyncio.wait(
                pending, timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )

            if stdin_task is not None and stdin_task in done:
                completed_stdin_task = stdin_task
                stdin_task = None
                try:
                    msg = _decode_stream(completed_stdin_task.result())
                except Exception:
                    stdin_open = False
                else:
                    # The child can exit between the poll() check above and
                    # these writes. Stop feeding stdin instead of aborting the
                    # actor and leaving its client blocked.
                    if isinstance(msg, StdinClose):
                        try:
                            proc.stdin.close()  # type: ignore[union-attr]
                        except OSError:
                            pass
                        stdin_open = False
                    elif isinstance(msg, StdinData):
                        try:
                            proc.stdin.write(msg.data)  # type: ignore[union-attr]
                            proc.stdin.flush()  # type: ignore[union-attr]
                        except OSError:
                            stdin_open = False

            if control_task is not None and control_task in done:
                completed_control_task = control_task
                control_task = None
                try:
                    msg = completed_control_task.result()
                except Exception:
                    control_open = False
                else:
                    if isinstance(msg, ControlSignal):
                        _send_process_signal(msg.signal_num)

        if not control_was_enabled:
            # With no control channel there is nothing useful to receive. Wait
            # asynchronously so actor shutdown can still cancel this endpoint.
            while proc.poll() is None:
                await asyncio.sleep(0.05)

        if start_new_session:
            # A session leader can exit while descendants keep running and
            # holding its output pipes open. No client can manage the group
            # after this endpoint returns, so clean it before joining readers.
            _send_process_signal(_signal.SIGKILL)

        await _join_reader(stdout_thread)
        await _join_reader(stderr_thread)

        returncode = proc.wait()
        cleanup_complete = True
        logger.info("Subprocess exited, returncode=%s, pid=%s", returncode, proc.pid)
        output_port.send(StreamExit(returncode=returncode))
    finally:
        for task in (stdin_task, control_task):
            if task is not None:
                task.cancel()
        if not cleanup_complete:
            _force_cleanup()
            # Tell the client the process is gone (e.g. its actor is being
            # stopped on lease release); otherwise its handle never exits.
            try:
                output_port.send(
                    StreamExit(
                        returncode=(
                            proc.returncode
                            if proc.returncode is not None
                            else -_signal.SIGKILL
                        )
                    )
                )
            except Exception:
                logger.debug("Error reporting cancelled subprocess exit", exc_info=True)


def _run_streaming_impl(
    args: str | list[str],
    shell: bool,
    cwd: str | None,
    env: dict[str, str] | None,
    output_port: Port[StreamMessage],
    enable_stdin: bool,
    chunk_size: int,
    enable_control: bool = False,
    stdout: int | None = PIPE,
    stderr: int | None = PIPE,
    start_new_session: bool = False,
    *,
    _stdin_channel_factory: Any = None,
    _control_channel_factory: Any = None,
) -> None:
    """Synchronous wrapper around the async streaming implementation."""
    asyncio.run(
        _arun_streaming_impl(
            args,
            shell,
            cwd,
            env,
            output_port,
            enable_stdin,
            chunk_size,
            enable_control,
            stdout,
            stderr,
            start_new_session,
            _stdin_channel_factory=_stdin_channel_factory,
            _control_channel_factory=_control_channel_factory,
        )
    )


class _StreamingSubprocessActor(Actor):
    """Runs a subprocess with streaming output on a remote Monarch process."""

    @endpoint
    async def run_streaming(
        self,
        args: str | list[str],
        shell: bool,
        cwd: str | None,
        env: dict[str, str] | None,
        output_port: Port[StreamMessage],
        enable_stdin: bool,
        chunk_size: int,
        enable_control: bool = False,
        stdout: int | None = PIPE,
        stderr: int | None = PIPE,
        start_new_session: bool = False,
    ) -> None:
        await _arun_streaming_impl(
            args,
            shell,
            cwd,
            env,
            output_port,
            enable_stdin,
            chunk_size,
            enable_control,
            stdout,
            stderr,
            start_new_session,
        )


_EOF_SENTINEL = None


class _PopenStreamReader:
    """Readable file-like wrapper around a ``queue.Queue`` fed by the drain thread.

    Supports ``read()``, ``readline()``, iteration (``for line in reader``),
    and the usual ``readable()``/``writable()``/``fileno()`` introspection.
    """

    def __init__(
        self,
        q: queue.Queue[bytes | None],
        encoding: str | None = None,
        errors: str | None = None,
    ) -> None:
        self._queue = q
        self._encoding = encoding
        self._errors = errors or "strict"
        # In text mode the queue's raw byte chunks are fed through an
        # incremental decoder, which holds back a trailing partial multibyte
        # sequence until the continuation bytes arrive.  This keeps the internal
        # buffer a fully-decoded ``str`` so that slicing by ``size`` (or at a
        # newline) can never split a character — the previous "slice bytes, then
        # decode the slice" approach raised ``UnicodeDecodeError`` whenever a
        # boundary landed mid-character.  Binary mode keeps a ``bytes`` buffer.
        self._decoder = (
            codecs.getincrementaldecoder(encoding)(self._errors) if encoding else None
        )
        self._buffer: bytes | str = "" if encoding else b""
        self._newline: bytes | str = "\n" if encoding else b"\n"
        self._eof = False
        self._closed = False

    def _fill_buffer(self) -> None:
        """Block until more decoded data (or EOF) is available."""
        if self._eof:
            return
        while True:
            chunk = self._queue.get()
            if chunk is _EOF_SENTINEL:
                self._eof = True
                if self._decoder is not None:
                    # Flush any bytes the decoder was holding back.  With
                    # errors="strict" a truncated trailing sequence raises here,
                    # matching stdlib text-stream behaviour at EOF.
                    assert isinstance(self._buffer, str)
                    self._buffer += self._decoder.decode(b"", final=True)
                return
            if self._decoder is not None:
                decoded = self._decoder.decode(chunk)
                if not decoded:
                    # Whole chunk was a partial multibyte prefix — keep reading
                    # until the continuation bytes complete at least one char.
                    continue
                assert isinstance(self._buffer, str)
                self._buffer += decoded
                return
            assert isinstance(self._buffer, bytes)
            assert isinstance(chunk, bytes)
            self._buffer += chunk
            return

    def _drain_all(self) -> None:
        """Drain the queue into the internal buffer until EOF."""
        while not self._eof:
            self._fill_buffer()

    def read(self, size: int = -1) -> bytes | str:
        """Read up to *size* units. ``read()`` / ``read(-1)`` reads until EOF.

        *size* counts characters in text mode and bytes in binary mode.
        """
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if size is None or size < 0:
            self._drain_all()
            data = self._buffer
            self._buffer = self._buffer[:0]
            return data
        # Specific size requested
        while len(self._buffer) < size and not self._eof:
            self._fill_buffer()
        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data

    def readline(self, size: int = -1) -> bytes | str:
        """Read a single line (up to and including ``\\n``)."""
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        while True:
            # Branch split solely so mypy can narrow buffer/newline to matching
            # str-or-bytes; both arms call the same find().  Text vs binary is
            # fixed at construction (see _decoder/_buffer init), so it never flips.
            if isinstance(self._buffer, str):
                assert isinstance(self._newline, str)
                nl = self._buffer.find(self._newline)
            else:
                assert isinstance(self._buffer, bytes)
                assert isinstance(self._newline, bytes)
                nl = self._buffer.find(self._newline)
            if nl >= 0:
                end = nl + 1
                if size is not None and size >= 0:
                    end = min(end, size)
                line = self._buffer[:end]
                self._buffer = self._buffer[end:]
                return line
            if self._eof:
                if size is not None and size >= 0:
                    line = self._buffer[:size]
                    self._buffer = self._buffer[size:]
                else:
                    line = self._buffer
                    self._buffer = self._buffer[:0]
                return line
            self._fill_buffer()

    def __iter__(self) -> Iterator[bytes | str]:
        return self

    def __next__(self) -> bytes | str:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(_EOF_SENTINEL)

    @property
    def closed(self) -> bool:
        return self._closed


class _PopenStreamWriter:
    """Writable file-like wrapper that sends data to the remote process's stdin.

    Data is sent immediately via the Monarch port; ``flush()`` is a no-op.
    """

    def __init__(
        self,
        stdin_ready: threading.Event,
        get_stdin_port: Any,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> None:
        self._stdin_ready = stdin_ready
        self._get_stdin_port = get_stdin_port  # callable returning Port | None
        self._encoding = encoding
        self._errors = errors or "strict"
        self._closed = False

    def write(self, data: bytes | str) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        self._stdin_ready.wait()
        port = self._get_stdin_port()
        if port is None:
            raise RuntimeError("stdin port not available")
        if isinstance(data, str):
            data = data.encode(self._encoding or "utf-8", self._errors)
        port.send(_encode_stream(StdinData(data=data)))
        return len(data)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stdin_ready.wait()
            port = self._get_stdin_port()
            if port is not None:
                port.send(StdinClose())

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


class Popen:
    """Remote process handle, mirroring :class:`subprocess.Popen`.

    The command starts executing on the remote Monarch process immediately.
    Call :meth:`communicate` or :meth:`wait` to block until completion.
    Use :meth:`kill`, :meth:`terminate`, or :meth:`send_signal` for
    remote process control.

    When ``streaming=True``, stdout/stderr become iterators that yield
    chunks incrementally (like the old ``StreamingPopen``).

    Can be used as a context manager::

        with Popen(["cmd"], proc_mesh=mesh, stdout=PIPE) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
    """

    DEFAULT_CHUNK_SIZE = 65536  # 64 KiB

    def __init__(
        self,
        args: str | list[str],
        *,
        proc_mesh: ProcMesh,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
        shell: bool = False,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        streaming: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        start_new_session: bool = False,
    ) -> None:
        if encoding is not None:
            text = True
        self.args = args
        self._text = text
        self._encoding = encoding or ("utf-8" if text else None)
        self._errors = errors or "strict"
        self._returncode: int | None = None
        self._stdin_enabled = stdin == PIPE
        self._stdout_piped = stdout == PIPE
        self._stderr_piped = stderr == PIPE or stderr == STDOUT
        self._exited = False
        self._input_sent = False
        self._communicated = False
        self._streaming = streaming
        self._pending_receive: Any = None
        self._pending_messages: deque[StreamMessage] = deque()
        self._receive_condition = threading.Condition()
        self._receiving = False

        # Ports (set by drain thread or inline for streaming)
        self._stdin_port: Port[StdinMessage] | None = None
        self._control_port: Port[ControlSignal] | None = None

        if streaming:
            # -- Streaming mode: no drain thread, iterators on stdout/stderr --
            self.stdin: _StreamingStdinWriter | _PopenStreamWriter | None = (
                _StreamingStdinWriter(self) if self._stdin_enabled else None
            )
            # Lazy iterators for stdout/stderr (share _recv_messages, so
            # only one of stdout/stderr/output should be consumed per process;
            # use .output for interleaved access).
            self.stdout: _PopenStreamReader | Iterator[bytes] | Iterator[str] | None = (
                self._iter_stdout() if self._stdout_piped else None
            )
            self.stderr: _PopenStreamReader | Iterator[bytes] | Iterator[str] | None = (
                self._iter_stderr() if self._stderr_piped else None
            )

            # Create output channel
            output_port, self._output_receiver = Channel[StreamMessage].open()

            # Resolve stdout/stderr for the streaming actor.
            actor_stdout = _subprocess.PIPE if stdout == PIPE else stdout
            actor_stderr: int | None
            if stderr == STDOUT:
                actor_stderr = _subprocess.STDOUT
            elif stderr == PIPE:
                actor_stderr = _subprocess.PIPE
            else:
                actor_stderr = stderr

            from agentbox.actors.mesh_schema import make_name

            name = make_name("streaming", str(next(_counter)))
            logger.info("Spawning streaming subprocess actor: %s", name)
            self._actor_mesh: ActorMesh[_StreamingSubprocessActor] = proc_mesh.spawn(
                name, _StreamingSubprocessActor
            )
            self._actor_mesh.run_streaming.broadcast(
                args,
                shell,
                cwd,
                env,
                output_port,
                self._stdin_enabled,
                chunk_size,
                True,  # enable_control
                actor_stdout,
                actor_stderr,
                start_new_session,
            )
        else:
            # -- Non-streaming mode: drain thread + buffered output --
            # Synchronisation events
            self._stdin_ready = threading.Event()
            self._control_ready = threading.Event()
            self._drain_complete = threading.Event()

            # Output buffers (written by drain thread, read after drain_complete)
            self._stdout_parts: list[bytes] = []
            self._stderr_parts: list[bytes] = []
            self._stdout_cached: str | bytes | None = None
            self._stderr_cached: str | bytes | None = None

            # Stream queues for file-like .stdout / .stderr / .stdin attributes
            self._stdout_queue: queue.Queue[bytes | None] | None = None
            self._stderr_queue: queue.Queue[bytes | None] | None = None

            if self._stdout_piped:
                self._stdout_queue = queue.Queue()
                self.stdout = _PopenStreamReader(  # type: ignore[assignment]
                    self._stdout_queue, encoding=self._encoding, errors=self._errors
                )
            else:
                self.stdout = None

            if self._stderr_piped:
                self._stderr_queue = queue.Queue()
                self.stderr = _PopenStreamReader(  # type: ignore[assignment]
                    self._stderr_queue, encoding=self._encoding, errors=self._errors
                )
            else:
                self.stderr = None

            if self._stdin_enabled:
                self.stdin = _PopenStreamWriter(
                    self._stdin_ready,
                    lambda: self._stdin_port,
                    encoding=self._encoding,
                    errors=self._errors,
                )
            else:
                self.stdin = None

            # Create output channel
            output_port, self._output_receiver = Channel[StreamMessage].open()

            # Resolve stdout/stderr for the streaming actor.
            actor_stdout = _subprocess.PIPE if stdout == PIPE else stdout
            actor_stderr: int | None = None  # type: ignore[no-redef]
            if stderr == STDOUT:
                actor_stderr = _subprocess.STDOUT
            elif stderr == PIPE:
                actor_stderr = _subprocess.PIPE
            else:
                actor_stderr = stderr

            from agentbox.actors.mesh_schema import make_name

            name = make_name("streaming", str(next(_counter)))
            self._actor_mesh = proc_mesh.spawn(name, _StreamingSubprocessActor)
            self._actor_mesh.run_streaming.broadcast(
                args,
                shell,
                cwd,
                env,
                output_port,
                self._stdin_enabled,
                chunk_size,
                True,  # enable_control — always on for kill/terminate support
                actor_stdout,
                actor_stderr,
                start_new_session,
            )

            # Background thread drains all messages from the actor
            self._drain_thread = threading.Thread(
                target=self._drain_messages, daemon=True
            )
            self._drain_thread.start()

    def _drain_messages(self) -> None:  # noqa: C901
        """Background thread that receives all messages from the streaming actor."""
        try:
            while True:
                msg = _decode_stream(self._output_receiver.recv().get())
                if isinstance(msg, StreamControlReady):
                    self._control_port = msg.control_port
                    self._control_ready.set()
                elif isinstance(msg, StreamStdinReady):
                    self._stdin_port = msg.stdin_port
                    self._stdin_ready.set()
                elif isinstance(msg, StreamData):
                    if msg.stream == StreamType.STDOUT:
                        self._stdout_parts.append(msg.data)
                        if self._stdout_queue is not None:
                            self._stdout_queue.put(msg.data)
                    else:
                        self._stderr_parts.append(msg.data)
                        if self._stderr_queue is not None:
                            self._stderr_queue.put(msg.data)
                elif isinstance(msg, StreamExit):
                    self._returncode = msg.returncode
                    self._exited = True
                    # Signal EOF on stream queues
                    if self._stdout_queue is not None:
                        self._stdout_queue.put(_EOF_SENTINEL)
                    if self._stderr_queue is not None:
                        self._stderr_queue.put(_EOF_SENTINEL)
                    break
                elif isinstance(msg, StreamError):
                    continue
        except Exception:
            pass  # channel closed
        finally:
            self._drain_complete.set()
            # Unblock anything waiting on ports that never arrived
            self._control_ready.set()
            self._stdin_ready.set()

    def _receive_message(self, timeout: float | None = None) -> StreamMessage:
        """Consume the shared receive slot, retaining it only across timeouts."""
        # A timed-out Monarch Future retains its receive slot. Reuse it until
        # consumed, including when the next observer is an output iterator.
        if self._pending_receive is None:
            self._pending_receive = self._output_receiver.recv()
        try:
            message = cast(
                StreamMessage,
                _decode_stream(self._pending_receive.get(timeout=timeout)),
            )
        except TimeoutError:
            raise
        except Exception:
            self._pending_receive = None
            raise
        self._pending_receive = None
        return message

    def _receive_until(
        self, ready: Callable[[], bool], timeout: float | None = None
    ) -> None:
        """Receive until *ready*, with one owner publishing under the condition."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._receive_condition:
                if ready():
                    return
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining < 0:
                    raise TimeoutError
                if self._receiving:
                    self._receive_condition.wait(timeout=remaining)
                    continue
                self._receiving = True
            try:
                # One caller owns receipt through publication. Other callers can
                # observe ready ports or time out while its receive is blocked.
                msg = self._receive_message(remaining)
                with self._receive_condition:
                    if isinstance(msg, StreamControlReady):
                        self._control_port = msg.control_port
                    elif isinstance(msg, StreamStdinReady):
                        self._stdin_port = msg.stdin_port
                    else:
                        self._pending_messages.append(msg)
                        if isinstance(msg, StreamExit):
                            self._returncode = msg.returncode
                            self._exited = True
            finally:
                with self._receive_condition:
                    self._receiving = False
                    self._receive_condition.notify_all()

    def _recv_messages(self) -> Iterator[StreamMessage]:
        """Iterate over all messages from the output receiver until StreamExit.

        Used only in streaming mode.
        """
        while True:
            self._receive_until(lambda: bool(self._pending_messages) or self._exited)
            with self._receive_condition:
                if not self._pending_messages:
                    return
                msg = self._pending_messages.popleft()
            yield msg
            if isinstance(msg, StreamExit):
                break

    def _iter_stdout(self) -> Iterator[bytes] | Iterator[str]:
        """Iterate over stdout chunks (streaming mode). Yields ``str`` if ``text=True``."""
        for msg in self._recv_messages():
            if isinstance(msg, StreamData) and msg.stream == StreamType.STDOUT:
                yield (
                    msg.data.decode(self._encoding or "utf-8", self._errors)
                    if self._text
                    else msg.data
                )

    def _iter_stderr(self) -> Iterator[bytes] | Iterator[str]:
        """Iterate over stderr chunks (streaming mode). Yields ``str`` if ``text=True``."""
        for msg in self._recv_messages():
            if isinstance(msg, StreamData) and msg.stream == StreamType.STDERR:
                yield (
                    msg.data.decode(self._encoding or "utf-8", self._errors)
                    if self._text
                    else msg.data
                )

    @property
    def output(self) -> Iterator[StreamData]:
        """Iterate over all output chunks (stdout and stderr interleaved).

        Only available in streaming mode (``streaming=True``).
        """
        if not self._streaming:
            raise AttributeError(
                "output iterator is only available in streaming mode (streaming=True)"
            )
        for msg in self._recv_messages():
            if isinstance(msg, StreamData):
                yield msg

    def write_stdin(self, data: bytes | str) -> None:
        """Send data to the remote process's stdin (streaming mode)."""
        if not self._stdin_enabled:
            raise RuntimeError("stdin is not enabled for this Popen instance")
        self._receive_until(lambda: self._stdin_port is not None or self._exited)
        with self._receive_condition:
            if self._exited:
                raise RuntimeError("Cannot write to stdin after process has exited")
            stdin_port = self._stdin_port
        assert stdin_port is not None
        if isinstance(data, str):
            data = data.encode(self._encoding or "utf-8", self._errors)
        stdin_port.send(_encode_stream(StdinData(data=data)))

    def close_stdin(self) -> None:
        """Close the remote process's stdin (streaming mode)."""
        if not self._stdin_enabled:
            raise RuntimeError("stdin is not enabled for this Popen instance")
        self._receive_until(lambda: self._stdin_port is not None or self._exited)
        with self._receive_condition:
            if self._exited:
                return
            stdin_port = self._stdin_port
        assert stdin_port is not None
        stdin_port.send(StdinClose())

    def communicate(
        self, input: str | bytes | None = None, timeout: float | None = None
    ) -> tuple[str | bytes | None, str | bytes | None]:
        """Wait for the remote process and return ``(stdout, stderr)``.

        If *input* is given and ``stdin=PIPE`` was set, the data is
        sent to the process's stdin before waiting for output.

        Returns ``None`` for streams that were not opened with ``PIPE``
        (non-streaming mode only; streaming mode always returns ``bytes``
        or ``str``).
        """
        if self._streaming:
            return self._communicate_streaming(input)
        return self._communicate_buffered(input, timeout)

    def _communicate_streaming(
        self, input: str | bytes | None = None
    ) -> tuple[bytes | str, bytes | str]:
        """Streaming-mode communicate: drain inline."""
        if input is not None:
            self.write_stdin(input)
            self.close_stdin()

        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        for msg in self._recv_messages():
            if isinstance(msg, StreamData):
                if msg.stream == StreamType.STDOUT:
                    stdout_parts.append(msg.data)
                else:
                    stderr_parts.append(msg.data)

        stdout_bytes = b"".join(stdout_parts)
        stderr_bytes = b"".join(stderr_parts)

        if self._text:
            return stdout_bytes.decode(
                self._encoding or "utf-8", self._errors
            ), stderr_bytes.decode(self._encoding or "utf-8", self._errors)
        return stdout_bytes, stderr_bytes

    def _communicate_buffered(
        self, input: str | bytes | None = None, timeout: float | None = None
    ) -> tuple[str | bytes | None, str | bytes | None]:
        """Non-streaming (buffered) communicate."""
        if self._communicated:
            return self._stdout_cached, self._stderr_cached

        # Send input to stdin (only once)
        if input is not None and self._stdin_enabled and not self._input_sent:
            self._stdin_ready.wait()
            if self._stdin_port is not None:
                send_data = (
                    input.encode(self._encoding or "utf-8", self._errors)
                    if isinstance(input, str)
                    else input
                )
                self._stdin_port.send(_encode_stream(StdinData(data=send_data)))
                self._stdin_port.send(StdinClose())
            self._input_sent = True

        # Wait for process to finish
        if not self._drain_complete.wait(timeout=timeout):
            raise TimeoutExpired(cmd=self.args, timeout=timeout or 0)

        # Build final output — None for streams not opened with PIPE
        if self._stdout_piped:
            stdout_bytes = b"".join(self._stdout_parts)
            self._stdout_cached = (
                stdout_bytes.decode(self._encoding or "utf-8", self._errors)
                if self._text
                else stdout_bytes
            )
        else:
            self._stdout_cached = None

        if self._stderr_piped:
            stderr_bytes = b"".join(self._stderr_parts)
            self._stderr_cached = (
                stderr_bytes.decode(self._encoding or "utf-8", self._errors)
                if self._text
                else stderr_bytes
            )
        else:
            self._stderr_cached = None

        self._communicated = True
        return self._stdout_cached, self._stderr_cached

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the remote process and return the exit code."""
        if self._streaming:
            if self._returncode is not None:
                return self._returncode
            for _msg in self._recv_messages():
                pass
            assert self._returncode is not None
            return self._returncode
        self.communicate(timeout=timeout)
        assert self._returncode is not None
        return self._returncode

    def poll(self) -> int | None:
        """Check if the remote process has finished. Returns returncode or ``None``."""
        return self._returncode

    def send_signal(self, sig: int, *, timeout: float | None = None) -> None:
        """Send a signal to the remote process.

        Does nothing if the process has already exited. ``timeout`` bounds
        waiting for its control channel, not process exit; use ``wait`` to
        confirm exit. A timeout does not cancel or replay the signal.
        """
        if self._exited:
            return
        if self._streaming:
            try:
                self._receive_until(
                    lambda: self._control_port is not None or self._exited, timeout
                )
            except TimeoutError as exc:
                raise TimeoutExpired(cmd=self.args, timeout=timeout or 0) from exc
            with self._receive_condition:
                if self._exited:
                    return
                control_port = self._control_port
            assert control_port is not None
            control_port.send(ControlSignal(signal_num=sig))
        else:
            # Non-streaming mode: wait on event
            if not self._control_ready.wait(timeout=timeout):
                raise TimeoutExpired(cmd=self.args, timeout=timeout or 0)
            if self._control_port is not None and not self._exited:
                self._control_port.send(ControlSignal(signal_num=sig))

    def terminate(self, *, timeout: float | None = None) -> None:
        """Send SIGTERM, optionally bounding the wait for its control channel."""
        self.send_signal(_signal.SIGTERM, timeout=timeout)

    def kill(self, *, timeout: float | None = None) -> None:
        """Send SIGKILL, optionally bounding the wait for its control channel."""
        self.send_signal(_signal.SIGKILL, timeout=timeout)

    @property
    def returncode(self) -> int | None:
        """The exit code of the remote process, or ``None`` if not yet finished."""
        return self._returncode

    @property
    def pid(self) -> None:
        """Not available for remote processes."""
        return None

    def __enter__(self) -> Popen:
        return self

    def __exit__(self, *args: Any) -> None:
        self.wait()


class _StreamingStdinWriter:
    """Writable file-like wrapper for :class:`Popen` streaming-mode stdin.

    Provides ``write()`` / ``close()`` so that ``proc.stdin.write(data)``
    and ``proc.stdin.close()`` mirror ``subprocess.Popen``.
    """

    def __init__(self, owner: Popen) -> None:
        self._owner = owner
        self._closed = False

    def write(self, data: bytes | str) -> int:
        """Write *data* to the remote process's stdin."""
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        self._owner.write_stdin(data)
        if isinstance(data, str):
            return len(
                data.encode(self._owner._encoding or "utf-8", self._owner._errors)
            )
        return len(data)

    def close(self) -> None:
        """Close the remote process's stdin."""
        if not self._closed:
            self._closed = True
            self._owner.close_stdin()

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


__all__ = [
    "PIPE",
    "STDOUT",
    "DEVNULL",
    "Popen",
    "CompletedProcess",
    "SubprocessError",
    "CalledProcessError",
    "TimeoutExpired",
    "StreamType",
    "StreamData",
    "StreamExit",
    "StreamError",
    "ControlSignal",
    "StreamControlReady",
]
