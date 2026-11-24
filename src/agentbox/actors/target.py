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

"""The command and file API shared by :class:`Machine` and :class:`Container`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, overload, TYPE_CHECKING

from agentbox.remote import pty as _pty, subprocess as _rsp


if TYPE_CHECKING:
    from agentbox.remote.pty import PtyProcess
    from agentbox.remote.subprocess import CompletedProcess, Popen
    from agentbox.tasks.api import TaskPool
    from monarch.actor import ProcMesh


type Command = str | list[str]
"""A ``str`` runs through ``/bin/sh -c``; a ``list`` is executed as argv."""


def _argv(cmd: Command) -> list[str]:
    return ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else list(cmd)


class ExecutionTarget(ABC):
    """Where commands run: a :class:`Machine` host or a :class:`Container`.

    ``env`` always *adds* variables to the target's environment, and ``cwd``
    is a path on the target. Subclasses decide how a command reaches the
    target by implementing :meth:`_host_argv`.
    """

    @property
    @abstractmethod
    def _proc_mesh_for_exec(self) -> ProcMesh:
        """ProcMesh of the host that executes commands."""

    @abstractmethod
    def _exec_task_pool(self) -> TaskPool:
        """Worker-side task pool that runs one-shot commands."""

    @abstractmethod
    def _host_argv(
        self, argv: list[str], env: dict[str, str] | None, cwd: str | None
    ) -> tuple[list[str], str | None]:
        """Return the host argv and host working directory that run *argv*."""

    @overload
    def run(
        self,
        cmd: Command,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = None,
        check: bool = False,
        text: Literal[True] = True,
    ) -> CompletedProcess[str]: ...

    @overload
    def run(
        self,
        cmd: Command,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = None,
        check: bool = False,
        text: Literal[False],
    ) -> CompletedProcess[bytes]: ...

    def run(
        self,
        cmd: Command,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = None,
        check: bool = False,
        text: bool = True,
    ) -> CompletedProcess[str] | CompletedProcess[bytes]:
        """Run *cmd* to completion and return its captured output.

        Raises:
            TimeoutExpired: If the command outlives *timeout* seconds.
            CalledProcessError: If *check* is set and the command fails.
        """
        argv, host_cwd = self._host_argv(_argv(cmd), env, cwd)
        return _rsp._run(
            argv,
            proc_mesh=self._proc_mesh_for_exec,
            task_pool=self._exec_task_pool(),
            cwd=host_cwd,
            capture_output=True,
            input=input,
            timeout=timeout,
            check=check,
            text=text,
        )

    @overload
    async def arun(
        self,
        cmd: Command,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = None,
        check: bool = False,
        text: Literal[True] = True,
    ) -> CompletedProcess[str]: ...

    @overload
    async def arun(
        self,
        cmd: Command,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = None,
        check: bool = False,
        text: Literal[False],
    ) -> CompletedProcess[bytes]: ...

    async def arun(
        self,
        cmd: Command,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        timeout: float | None = None,
        check: bool = False,
        text: bool = True,
    ) -> CompletedProcess[str] | CompletedProcess[bytes]:
        """Async-native counterpart of :meth:`run`."""
        argv, host_cwd = self._host_argv(_argv(cmd), env, cwd)
        return await _rsp._arun(
            argv,
            proc_mesh=self._proc_mesh_for_exec,
            task_pool=self._exec_task_pool(),
            cwd=host_cwd,
            capture_output=True,
            input=input,
            timeout=timeout,
            check=check,
            text=text,
        )

    @overload
    def spawn(
        self,
        cmd: Command,
        *,
        pty: Literal[False] = False,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
    ) -> Popen: ...

    @overload
    def spawn(
        self,
        cmd: Command,
        *,
        pty: Literal[True],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
    ) -> PtyProcess: ...

    def spawn(
        self,
        cmd: Command,
        *,
        pty: bool = False,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
    ) -> Popen | PtyProcess:
        """Start a long-running process with live I/O.

        With ``pty=False`` the process gets stdin/stdout/stderr pipes
        (:class:`~agentbox.remote.subprocess.Popen`); with ``pty=True``
        it runs attached to a pseudo-terminal
        (:class:`~agentbox.remote.pty.PtyProcess`), e.g.
        ``spawn(["bash"], pty=True)`` for an interactive shell.

        Either way the process leads its own process group, which is killed
        when the handle's actor is stopped (e.g. when the lease is released).
        With ``pty=False``, signals reach the whole group and the group is
        also killed when the process exits, so background children do not
        outlive it.
        """
        argv, host_cwd = self._host_argv(_argv(cmd), env, cwd)
        mesh = self._proc_mesh_for_exec
        if pty:
            return _pty.spawn(argv, proc_mesh=mesh, cwd=host_cwd, text=text)
        return _rsp.Popen(
            argv,
            proc_mesh=mesh,
            stdin=_rsp.PIPE,
            stdout=_rsp.PIPE,
            stderr=_rsp.PIPE,
            cwd=host_cwd,
            text=text,
            start_new_session=True,
        )

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """Copy a local file or directory to *remote_path* on the target.

        A file replaces *remote_path*. A directory's contents are copied into
        *remote_path*, which is created if needed; entries already there are
        overwritten or kept, as with ``tar -x``. Directory copies keep
        regular files, subdirectories, and symlinks (as links) with their
        modes and mtimes, skip other file types, and never write outside
        *remote_path*. Modes lose setuid, setgid, sticky, and group/other
        write bits, and every entry stays writable by its owner.
        :class:`Container` uploads extract with the image's ``tar``, so inside
        a container the image's ``tar`` decides how existing entries (for
        example a symlink where a directory is expected) are treated.

        Raises:
            FileCopyError: If the copy fails.
        """

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> None:
        """Copy a file or directory on the target to *local_path*.

        The reverse of :meth:`upload`, with the same semantics.
        """

    @abstractmethod
    async def aupload(self, local_path: str, remote_path: str) -> None:
        """Async-native counterpart of :meth:`upload`."""

    @abstractmethod
    async def adownload(self, remote_path: str, local_path: str) -> None:
        """Async-native counterpart of :meth:`download`."""
