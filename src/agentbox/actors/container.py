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

"""Machine-like wrapper for Apptainer containers.

Provides :class:`Container` for executing commands inside an Apptainer
instance via ``apptainer exec`` from the host side, using a
:class:`~agentbox.actors.machine.Machine` for remote execution.  Monarch
runs on the host — the container image does not need Monarch installed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shlex
import tempfile
import time
import uuid
import warnings
from typing import Literal, TYPE_CHECKING

from agentbox.actors.machine import Machine
from agentbox.actors.target import ExecutionTarget
from agentbox.remote import tree_archive
from agentbox.remote.file_transfer import FileCopyError
from monarch.actor import Actor, endpoint, ProcMesh


if TYPE_CHECKING:
    from agentbox.remote.subprocess import CompletedProcess
    from agentbox.tasks.api import TaskPool
    from monarch._src.actor.actor_mesh import ActorMesh


logger = logging.getLogger(__name__)

# Moves a staged upload ($1) to its destination ($2), creating parent dirs.
_PLACE = 'mkdir -p "$(dirname "$2")" && mv -f "$1" "$2"'

# Unpacks a staged directory archive ($1) into $2, then removes the archive.
_UNPACK = 'mkdir -p "$2" && tar -xf "$1" -C "$2"; status=$?; rm -f "$1"; exit $status'

# Exit status of _STAGE after archiving a directory. An exit status, unlike a
# marker on stdout, cannot be disturbed by what Apptainer or a wrapper prints.
_STAGED_DIRECTORY = 10

# Stages $1 at $2 for download: a file is copied; a directory is archived and
# the step exits with _STAGED_DIRECTORY.
_STAGE = (
    'if [ -d "$1" ]; then tar -cf "$2" -C "$1" . || exit; '
    f"exit {_STAGED_DIRECTORY}; "
    'else cp "$1" "$2"; fi'
)

# Defines rename SRC DST: a plain rename, even when DST is a symlink to a
# directory (plain mv would move SRC into that directory). GNU mv spells this
# -T and BSD/macOS mv -h.
_RENAME = (
    "if mv --version >/dev/null 2>&1; then nofollow=-T; else nofollow=-h; fi; "
    'rename() { mv -f "$nofollow" -- "$1" "$2"; }; '
)

# Host side of an upload: moves the received file $1 into the staging directory
# as $2, then runs the in-container step given as the remaining arguments.
_HOST_GIVE = _RENAME + 'rename "$1" "$2" && shift 2 && exec "$@"'

# Host side of a download: runs the in-container step given after $1 and $2,
# then moves the staged $1 out of the container's reach to $2 and checks that
# it is still a regular file before the host reads it. Exits with the step's
# status (0 or _STAGED_DIRECTORY) on success.
_HOST_TAKE = _RENAME + (
    'src=$1 dst=$2; shift 2; "$@"; status=$?; '
    f"case $status in 0|{_STAGED_DIRECTORY}) ;; *) exit $status;; esac; "
    'rename "$src" "$dst" || exit; '
    'if [ -L "$dst" ] || [ ! -f "$dst" ]; then '
    'echo "staged copy is not a regular file" >&2; exit 1; fi; '
    'exit "$status"'
)


def _subprocess_output_text(stderr: str | bytes | None) -> str:
    """Return captured subprocess stderr as text for diagnostics."""
    if isinstance(stderr, bytes):
        return stderr.decode(errors="replace")
    return stderr if stderr is not None else "(no stderr)"


class ApptainerNotFoundError(RuntimeError):
    """Raised when the ``apptainer`` binary is not found on the target node."""


@dataclasses.dataclass
class BindMount:
    """A bind-mount specification for Apptainer (``--bind src[:dst[:opts]]``).

    Args:
        src: Source path on the **remote compute node** to mount into the
            container.
        dst: Destination path inside the container.  Defaults to ``src``
            (same-path mount) when ``None``.
        options: Mount access mode: ``"ro"`` for read-only or ``"rw"`` for
            read-write (Apptainer default when omitted).
    """

    src: str
    dst: str | None = None
    options: Literal["ro", "rw"] | None = None

    def to_arg(self) -> str:
        """Render to an Apptainer ``--bind`` argument string."""
        if self.options is not None:
            dst = self.dst if self.dst is not None else self.src
            return f"{self.src}:{dst}:{self.options}"
        if self.dst is not None:
            return f"{self.src}:{self.dst}"
        return self.src


def _normalize_bind(mount: str | BindMount) -> str:
    """Normalize a bind-mount spec to a raw ``--bind`` argument string."""
    if isinstance(mount, BindMount):
        return mount.to_arg()
    return mount


def _bind_dst(bind: str) -> str:
    """Return the container-side destination path of a ``--bind`` spec.

    Apptainer bind syntax is ``src[:dst[:opts]]``; the destination defaults to
    the source when no ``dst`` is given.  Used to detect whether a destination
    path is already bound (exact match — not a substring, which would let
    ``/workspace`` falsely satisfy a request for ``/work``).
    """
    parts = bind.split(":")
    return parts[1] if len(parts) >= 2 else parts[0]


def _format_apptainer_command_for_log(cmd: list[str]) -> str:
    """Shell-format an Apptainer command without exposing ``--env`` values."""
    redacted: list[str] = []
    redact_next = False
    for arg in cmd:
        if redact_next:
            key, separator, _ = arg.partition("=")
            redacted.append(f"{key}=<redacted>" if separator else "<redacted>")
            redact_next = False
        elif arg == "--env":
            redacted.append(arg)
            redact_next = True
        elif arg.startswith("--env="):
            key, separator, _ = arg[len("--env=") :].partition("=")
            redacted.append(
                f"--env={key}=<redacted>" if separator else "--env=<redacted>"
            )
        else:
            redacted.append(arg)
    return shlex.join(redacted)


@dataclasses.dataclass
class ApptainerOptions:
    """Named options for Apptainer container execution.

    Covers the most commonly used flags.  For anything not listed here,
    use ``extra_args`` to pass raw arguments to ``apptainer``.

    Args:
        nv: Pass ``--nv`` for NVIDIA GPU access (via NVML).
        userns: Pass ``--userns`` to run in a new user namespace.
        nvccli: Pass ``--nvccli`` for NVIDIA Container CLI GPU access.
        rocm: Pass ``--rocm`` for AMD ROCm GPU access.
        writable_tmpfs: Pass ``--writable-tmpfs`` to overlay a writable tmpfs
            on the read-only SIF image (required for ``pip install`` etc.).
        no_home: Pass ``--no-home`` to prevent the host home directory from
            being mounted inside the container.  Enabled by default.
        contain: Pass ``--contain`` for a minimal ``/dev`` and empty home/tmp.
        containall: Pass ``--containall`` for ``--contain`` plus PID and IPC
            namespace isolation.  Enabled by default and supersedes ``contain``
            when both are set.  Setting this to ``False`` shares the worker's
            PID and IPC namespaces and emits a security warning.
        fakeroot: Pass ``--fakeroot`` to run as a fake root user inside the
            container.
        network: Create a network namespace at instance startup with
            ``--net --network <name>``. Use ``"none"`` for a loopback-only
            namespace suitable for untrusted workloads.
        overlay: Persistent overlay paths passed as ``--overlay <path>`` to
            both ``instance start`` and every ``exec``.  Each entry may be a
            plain path (``/tmp/overlay.img``) or a colon-suffixed read-only
            mount (``/tmp/overlay.img:ro``).
        writable: Pass ``--writable`` to modify an overlay embedded in the SIF.
            Mutually exclusive with ``writable_tmpfs``; when both are set
            ``writable`` takes precedence.
        extra_args: Raw extra arguments appended after all named flags.
    """

    # GPU passthrough
    nv: bool = False
    nvccli: bool = False
    rocm: bool = False

    # User namespace isolation
    userns: bool = False

    # Filesystem / environment isolation
    writable_tmpfs: bool = False
    no_home: bool = True
    contain: bool = False
    containall: bool = True

    # Privilege
    fakeroot: bool = False

    # Network namespaces are instance state, so render this only for startup.
    network: str | None = None

    # Persistent overlays: each entry is an overlay path or ``path:ro`` string.
    # Passed as one ``--overlay <path>`` flag per entry to ``instance start``
    # only — the overlay is already mounted as part of the running instance and
    # must NOT be re-passed to ``apptainer exec instance://``.
    overlay: list[str] = dataclasses.field(default_factory=list)

    # Modify an overlay that is embedded in the SIF (``--writable``).
    # Mutually exclusive with ``writable_tmpfs``; ``writable`` takes precedence
    # when both are set (Apptainer ignores ``--writable-tmpfs`` with ``--writable``).
    # Like ``overlay``, only passed to ``instance start``, not to ``exec``.
    writable: bool = False

    # Escape hatch
    extra_args: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.containall:
            warnings.warn(
                "ApptainerOptions(containall=False) shares the worker's PID and IPC "
                "namespaces. Without separate filesystem containment, host-backed "
                "paths such as /tmp may also be visible. This is unsafe for untrusted "
                "workloads; use containall=True unless host namespace sharing is "
                "explicitly required.",
                UserWarning,
                stacklevel=2,
            )

    def _common_args(self) -> list[str]:
        """Flags valid for both ``instance start`` and ``exec instance://``."""
        args: list[str] = []
        if self.userns:
            args.append("--userns")
        if self.nv:
            args.append("--nv")
        if self.nvccli:
            args.append("--nvccli")
        if self.rocm:
            args.append("--rocm")
        if self.no_home:
            args.append("--no-home")
        if self.containall:
            args.append("--containall")
        elif self.contain:
            args.append("--contain")
        args.extend(self.extra_args)
        return args

    def to_exec_args(self) -> list[str]:
        """Render flags valid for ``apptainer exec instance://``.

        Excludes instance-scoped filesystem, privilege, and network flags:
        they are established by ``instance start`` and must not be re-passed
        on each ``exec``. ``--fakeroot`` is also silently
        ignored by Apptainer when joining a running instance and produces an
        INFO message on stderr that causes false ``FileCopyError`` failures.
        """
        args = self._common_args()
        if self.writable_tmpfs:
            args.insert(0, "--writable-tmpfs")
        return args

    def to_start_args(self) -> list[str]:
        """Render flags for ``apptainer instance start`` (superset of exec args).

        Includes ``--overlay``, ``--writable``, and ``--fakeroot`` which are
        applied once at instance startup.  ``--writable`` takes precedence over
        ``--writable-tmpfs`` when both are set.
        """
        args = self._common_args()
        if self.network:
            args += ["--net", "--network", self.network]
        if self.fakeroot:
            args.append("--fakeroot")
        if self.writable:
            args.insert(0, "--writable")
        elif self.writable_tmpfs:
            args.insert(0, "--writable-tmpfs")
        for ov in self.overlay:
            args += ["--overlay", ov]
        return args


# Shared Apptainer command builders


def _apptainer_command(*args: str, config_path: str | None = None) -> list[str]:
    """Build an Apptainer command with an optional global config file."""
    cmd = ["apptainer"]
    if config_path is not None:
        cmd += ["--config", config_path]
    cmd.extend(args)
    return cmd


def _build_exec_prefix(
    instance_name: str,
    *,
    apptainer_config_path: str | None = None,
    instance_env: dict[str, str] | None = None,
    apptainer_args: list[str] | None = None,
    bind_mounts: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[str]:
    """Build an ``apptainer exec`` command prefix.

    Pure string logic — no RPC, no subprocess calls.  Used by
    :class:`Container`.

    Args:
        instance_name: Apptainer instance name (used as ``instance://<name>``).
        apptainer_config_path: Config file path on the remote worker.
        instance_env: Base environment variables configured on the instance.
        apptainer_args: Extra arguments (e.g. ``["--nv"]``).
        bind_mounts: Bind-mount specs (``"src:dst"`` strings).
        env: Additional per-call environment variables.
        cwd: Working directory inside the container (``--pwd``).
    """
    cmd = _apptainer_command("exec", "--cleanenv", config_path=apptainer_config_path)

    # Merge env layers: instance env first, then per-call env.
    # Do NOT inject a default PATH — let the SIF's %environment handle it.
    # Docker-converted SIFs preserve the Docker ENV PATH, and conda-based
    # images have their PATH set in %environment. Overriding it here would
    # shadow the SIF's own environment (breaking conda images, etc.).
    # Users who need a specific PATH can pass env={"PATH": "..."} to Container.
    merged_env: dict[str, str] = {}
    if instance_env:
        merged_env.update(instance_env)
    if env:
        merged_env.update(env)

    for key, value in merged_env.items():
        cmd += ["--env", f"{key}={value}"]

    if cwd is not None:
        cmd += ["--pwd", cwd]

    if bind_mounts:
        for mount in bind_mounts:
            cmd += ["--bind", mount]

    if apptainer_args:
        cmd += apptainer_args

    cmd.append(f"instance://{instance_name}")
    return cmd


def _build_start_cmd(
    image: str,
    name: str,
    apptainer_args: list[str],
    env: dict[str, str],
    binds: list[str],
    *,
    apptainer_config_path: str | None = None,
) -> list[str]:
    """Build an ``apptainer instance start`` command.

    Pure string logic — no RPC, no subprocess calls.  Mirrors the command
    constructed inside :meth:`_ContainerActor.start_instance` so the client
    can log the equivalent shell command before dispatching the RPC.
    """
    cmd = _apptainer_command(
        "instance", "start", "--cleanenv", config_path=apptainer_config_path
    )

    for mount in binds:
        cmd += ["--bind", mount]

    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]

    cmd += apptainer_args
    cmd += [image, name]
    return cmd


class _ContainerActor(Actor):
    """Monarch actor that manages an Apptainer instance on a remote compute node.

    All methods execute on the remote node where the ProcMesh worker lives.
    The actor handles instance lifecycle and staging directory management.
    """

    @endpoint
    def start_instance(
        self,
        image: str,
        name: str,
        start_args: list[str],
        env: dict[str, str],
        binds: list[str],
        timeout: float | None = None,
        apptainer_config_path: str | None = None,
    ) -> None:
        """Start an Apptainer instance on the remote node.

        Args:
            image: Image path or URI passed to ``apptainer instance start``.
            name: Name assigned to the Apptainer instance.
            start_args: Effective arguments passed to ``instance start``.
            env: Variables explicitly injected into the container via ``--env``.
            binds: Bind-mount specifications passed via ``--bind``.
            apptainer_config_path: Config file path on the remote worker.
        """
        import subprocess as _sp
        import time as _time

        cmd = _build_start_cmd(
            image,
            name,
            start_args,
            env,
            binds,
            apptainer_config_path=apptainer_config_path,
        )
        logger.info(
            "Container phase started phase=boot instance=%s "
            "timeout_seconds=%s command=%s",
            name,
            timeout,
            _format_apptainer_command_for_log(cmd),
        )
        started_at = _time.monotonic()
        try:
            # Keep the host environment for Apptainer/GPU/module setup;
            # the command's --cleanenv prevents it from entering the container.
            _sp.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise ApptainerNotFoundError(
                "apptainer binary not found on the target node. "
                "Ensure Apptainer is installed and available in PATH."
            ) from None
        except _sp.TimeoutExpired:
            elapsed = _time.monotonic() - started_at
            try:
                _sp.run(
                    _apptainer_command(
                        "instance",
                        "stop",
                        name,
                        config_path=apptainer_config_path,
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, _sp.SubprocessError) as stop_error:
                logger.warning(
                    "Unable to stop instance after boot timeout instance=%s error=%s",
                    name,
                    stop_error,
                )
            logger.error(  # noqa: TRY400 - TimeoutExpired contains raw --env values.
                "Container phase failed phase=boot instance=%s "
                "elapsed_seconds=%.3f timeout_seconds=%s reason=timeout",
                name,
                elapsed,
                timeout,
            )
            raise RuntimeError(
                f"Apptainer instance start timed out after {elapsed:.1f}s "
                f"(timeout={timeout}s) for {name}"
            ) from None
        except _sp.CalledProcessError as e:
            elapsed = _time.monotonic() - started_at
            logger.exception(
                "Container phase failed phase=boot instance=%s "
                "elapsed_seconds=%.3f timeout_seconds=%s reason=exit_%d\n"
                "stdout: %s\nstderr: %s",
                name,
                elapsed,
                timeout,
                e.returncode,
                e.stdout,
                e.stderr,
            )
            # Re-raise as a RuntimeError whose message includes stderr.
            # CalledProcessError.__str__ only shows command + rc, so the
            # stderr is invisible across the Monarch actor boundary
            # (which only carries the exception message, not logger
            # output). RuntimeError carries the full message verbatim.
            stdout_s = (e.stdout or "").strip() or "(empty)"
            stderr_s = (e.stderr or "").strip() or "(empty)"
            raise RuntimeError(
                f"apptainer instance start failed (rc={e.returncode})\n"
                f"stdout:\n{stdout_s}\nstderr:\n{stderr_s}"
            ) from e
        elapsed = _time.monotonic() - started_at
        logger.info(
            "Container phase completed phase=boot instance=%s "
            "elapsed_seconds=%.3f timeout_seconds=%s",
            name,
            elapsed,
            timeout,
        )

    @endpoint
    def stop_instance(
        self,
        name: str,
        apptainer_config_path: str | None = None,
    ) -> None:
        """Stop an instance and wait until it is no longer listed."""
        import json as _json
        import subprocess as _sp

        cmd = _apptainer_command(
            "instance", "stop", name, config_path=apptainer_config_path
        )
        logger.info("Stopping Apptainer instance: %s", name)
        timeout = 60.0
        deadline = time.monotonic() + timeout
        try:
            stopped = _sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        except _sp.TimeoutExpired as exc:
            raise RuntimeError(
                f"Apptainer shutdown timed out after {timeout:g}s for {name}"
                f"\nstop stderr: {_subprocess_output_text(exc.stderr)}"
            ) from exc

        last_status_error: str | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = (
                    f"\nlast status error: {last_status_error}"
                    if last_status_error is not None
                    else ""
                )
                raise RuntimeError(
                    f"Apptainer shutdown timed out after {timeout:g}s for {name}"
                    f"\nstop stderr: {_subprocess_output_text(stopped.stderr)}"
                    f"{detail}"
                )

            try:
                result = _sp.run(
                    _apptainer_command(
                        "instance",
                        "list",
                        "--json",
                        config_path=apptainer_config_path,
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                )
                payload = _json.loads(result.stdout)
                instances = payload["instances"]
                if not isinstance(instances, list) or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("instance"), str)
                    for row in instances
                ):
                    raise ValueError("invalid instances")
            except _sp.TimeoutExpired as exc:
                last_status_error = (
                    f"instance list timed out: {exc}; stderr: "
                    f"{_subprocess_output_text(exc.stderr)}"
                )
            except _sp.CalledProcessError as exc:
                last_status_error = (
                    f"instance list exited with rc={exc.returncode}: "
                    f"{_subprocess_output_text(exc.stderr)}"
                )
            except (_json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_status_error = f"invalid instance list: {exc}"
            else:
                last_status_error = None
                if not any(row["instance"] == name for row in instances):
                    if stopped.returncode:
                        logger.warning(
                            "Instance %s is absent after stop exited with rc=%s: %s",
                            name,
                            stopped.returncode,
                            _subprocess_output_text(stopped.stderr),
                        )
                    return

            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    @endpoint
    def is_running(self, name: str, apptainer_config_path: str | None = None) -> bool:
        """Check whether the named instance is listed by ``apptainer instance list``."""
        import json as _json
        import subprocess as _sp

        try:
            result = _sp.run(
                _apptainer_command(
                    "instance",
                    "list",
                    "--json",
                    config_path=apptainer_config_path,
                ),
                capture_output=True,
                text=True,
                check=True,
            )
            data = _json.loads(result.stdout)
            instances = data.get("instances") or []
            return any(inst.get("instance") == name for inst in instances)
        except (_sp.CalledProcessError, _json.JSONDecodeError):
            return False

    @endpoint
    def create_staging_dir(self, prefix: str) -> str:
        """Create a temporary directory on the remote node and return its path."""
        import os as _os
        import sys as _sys
        import tempfile as _tmp

        # On macOS, Apptainer runs inside a Lima VM that mounts $HOME but not
        # /var/folders (the default macOS TMPDIR), so a staging dir under
        # TMPDIR cannot be bind-mounted into the container. Place it under
        # ~/.agentbox/tmp/ which is Lima-visible.
        parent: str | None = None
        if _sys.platform == "darwin":
            parent = _os.path.join(_os.path.expanduser("~"), ".agentbox", "tmp")
            _os.makedirs(parent, exist_ok=True)
        return _tmp.mkdtemp(prefix=prefix, dir=parent)

    @endpoint
    def cleanup_staging_dir(self, path: str) -> None:
        """Remove a staging directory on the remote node."""
        import shutil as _shutil

        _shutil.rmtree(path, ignore_errors=True)


class Container(ExecutionTarget):
    """An Apptainer instance on a :class:`Machine`, running every command inside it.

    ``run``, ``spawn``, ``upload``, and ``download`` behave as on a Machine but
    execute inside the instance via ``apptainer exec``; Monarch itself stays on
    the host, so images need no AgentBox dependencies.

    **Environment.** Both ``instance start`` and ``exec`` use ``--cleanenv``:
    the container sees only the image's ``%environment``, then the
    constructor's ``env``, then per-call ``env`` (later wins). ``PATH`` is not
    injected, so the image's own ``PATH`` applies.

    **Isolation.** :class:`ApptainerOptions` defaults to ``--containall`` and
    ``--no-home``; set ``network="none"`` for a loopback-only network.

    **Files.** :meth:`start` binds a private worker-side staging directory at
    the same path inside the container; uploads and downloads move through it.
    The host reads and writes copies only in a second directory the container
    cannot reach, and moves them across with a rename, which never follows a
    symlink the container may have planted.
    Directories cross as one tar archive, packed and unpacked by the image's
    ``tar``, so the host never walks a tree the container can change.

    **Working directory.** ``cwd`` becomes ``--pwd`` for every command unless
    a call passes its own. When the path is not already a bind target, a fresh
    writable worker directory is bound there, because image filesystems are
    read-only.

    Args:
        machine: A :class:`Machine` whose ProcMesh is on the target compute node.
        image: Path to the SIF image or Docker URI on the remote node.
        instance_name: Name for the Apptainer instance (auto-generated if ``None``).
        options: Named Apptainer options (GPU, isolation, etc.).  Use
            ``options`` for all Apptainer flags.
        apptainer_config_path: Path to an Apptainer config file on the remote
            worker. Passed as the global ``--config`` option for every command
            in this container's lifecycle.
        env: Environment variables propagated into the container.  Passed via
            ``--env`` to both ``instance start`` and every ``exec``; the host
            environment is fully scrubbed by ``--cleanenv`` before these are
            applied.  Does not include ``PATH`` by default — see above.
        bind_mounts: Bind-mount specs as ``"src:dst"`` strings or
            :class:`BindMount` objects (which also expose the ``ro``/``rw``
            access option).  ``src`` must exist on the remote compute node.
        cwd: Default working directory inside the container.
    """

    def __init__(
        self,
        machine: Machine,
        image: str,
        *,
        instance_name: str | None = None,
        options: ApptainerOptions | None = None,
        apptainer_config_path: str | None = None,
        env: dict[str, str] | None = None,
        bind_mounts: list[str | BindMount] | None = None,
        cwd: str | None = None,
        start_timeout: float | None = None,
    ) -> None:
        self._machine = machine
        self._image = image
        self._instance_name = instance_name or f"agentbox-ctr-{uuid.uuid4().hex[:8]}"
        self._options = options or ApptainerOptions()
        self._apptainer_config_path = apptainer_config_path
        self._apptainer_args: list[str] = self._options.to_exec_args()
        self._apptainer_start_args: list[str] = self._options.to_start_args()
        self._env: dict[str, str] = dict(env) if env else {}
        # Warn about env vars that apptainer cannot handle via --env:
        #   - names containing "%%" (bash exported functions, e.g. BASH_FUNC_foo%%)
        #   - values containing newlines (break the key=value format on the command line)
        _bad_env_keys = [k for k in self._env if "%%" in k or "\n" in self._env[k]]
        if _bad_env_keys:
            raise ValueError(
                f"Container env contains {len(_bad_env_keys)} variable(s) that apptainer will reject "
                f"via --env (names with '%%' or values with newlines): {_bad_env_keys}. "
                "Filter these out before constructing Container (e.g. pass only "
                "POSIX-identifier keys whose values are single-line)."
            )
        self._bind_mounts: list[str] = [_normalize_bind(m) for m in bind_mounts or []]
        # User binds plus the staging and cwd binds added by start().
        self._bind_mounts_effective: list[str] = list(self._bind_mounts)
        self._cwd = cwd
        self._staging_dir: str | None = None
        self._host_dir: str | None = None  # worker-only side of every file copy
        self._cwd_dir: str | None = None  # worker-side tmpdir bound to self._cwd
        self._running = False
        self._start_timeout = start_timeout

        # Lazily spawned actor on first start().
        self._actor_mesh: ActorMesh[_ContainerActor] | None = None

    @property
    def staging_dir(self) -> str:
        """The staging directory on the remote node (available after :meth:`start`)."""
        if self._staging_dir is None:
            raise RuntimeError("Container not started — call start() first")
        return self._staging_dir

    @property
    def instance_name(self) -> str:
        """The Apptainer instance name."""
        return self._instance_name

    def _start_cmd(self, start_args: list[str] | None = None) -> list[str]:
        """Build the effective ``apptainer instance start`` command."""
        return _build_start_cmd(
            self._image,
            self._instance_name,
            self._apptainer_start_args if start_args is None else start_args,
            self._env,
            self._bind_mounts_effective,
            apptainer_config_path=self._apptainer_config_path,
        )

    def _ensure_actor(self) -> ActorMesh[_ContainerActor]:
        if self._actor_mesh is None:
            from agentbox.actors.mesh_schema import make_name

            name = make_name("container", uuid.uuid4().hex[:8])
            self._actor_mesh = self._machine.proc_mesh.spawn(name, _ContainerActor)
        return self._actor_mesh

    def start(self) -> None:
        """Start the Apptainer instance on the remote node."""

        actor = self._ensure_actor()

        # Create staging dir on the remote node.
        self._staging_dir = actor.create_staging_dir.choose("agentbox-staging-").get()
        self._host_dir = actor.create_staging_dir.choose("agentbox-host-").get()

        # Add staging dir bind mount.
        staging_bind = f"{self._staging_dir}:{self._staging_dir}"
        effective_binds = list(self._bind_mounts)
        if staging_bind not in effective_binds:
            effective_binds.append(staging_bind)

        # If a working directory was requested, bind a writable tmpdir from the
        # worker node to self._cwd.  The base SIF filesystem is read-only squashfs
        # and --fakeroot is ignored on apptainer exec into a running instance, so
        # mkdir inside the container fails.  A host-side bind mount is the fix.
        if self._cwd and not any(_bind_dst(b) == self._cwd for b in effective_binds):
            self._cwd_dir = actor.create_staging_dir.choose("agentbox-cwd-").get()
            effective_binds.append(f"{self._cwd_dir}:{self._cwd}")
        self._bind_mounts_effective = effective_binds
        start_args = list(self._apptainer_start_args)
        logger.info(
            "Starting container: %s",
            _format_apptainer_command_for_log(self._start_cmd(start_args)),
        )

        actor.start_instance.choose(
            self._image,
            self._instance_name,
            start_args,
            self._env,
            effective_binds,
            self._start_timeout,
            self._apptainer_config_path,
        ).get()
        self._running = True
        self._register_with_machine()

    async def astart(self) -> None:
        """Async-native counterpart of :meth:`start` (awaits the lifecycle RPCs)."""

        actor = self._ensure_actor()

        self._staging_dir = await actor.create_staging_dir.choose("agentbox-staging-")
        self._host_dir = await actor.create_staging_dir.choose("agentbox-host-")

        staging_bind = f"{self._staging_dir}:{self._staging_dir}"
        effective_binds = list(self._bind_mounts)
        if staging_bind not in effective_binds:
            effective_binds.append(staging_bind)

        if self._cwd and not any(_bind_dst(b) == self._cwd for b in effective_binds):
            self._cwd_dir = await actor.create_staging_dir.choose("agentbox-cwd-")
            effective_binds.append(f"{self._cwd_dir}:{self._cwd}")
        self._bind_mounts_effective = effective_binds
        start_args = list(self._apptainer_start_args)
        logger.info(
            "Starting container: %s",
            _format_apptainer_command_for_log(self._start_cmd(start_args)),
        )

        await actor.start_instance.choose(
            self._image,
            self._instance_name,
            start_args,
            self._env,
            effective_binds,
            self._start_timeout,
            self._apptainer_config_path,
        )
        self._running = True
        self._register_with_machine()

    def _register_with_machine(self) -> None:
        # Releasing the Machine stops containers left running (see release()).
        with self._machine._lifecycle_lock:
            self._machine._containers.add(self)

    def _unregister_from_machine(self) -> None:
        with self._machine._lifecycle_lock:
            self._machine._containers.discard(self)

    def stop(self) -> None:
        """Stop the Apptainer instance and clean up the staging directories.

        For a running container, confirm shutdown before staging cleanup. Retain
        its state on stop failure so shutdown can be retried. Directory cleanup
        remains best-effort; attempt both RPCs and retain paths on reported errors.
        """
        if self._actor_mesh is None:
            return
        actor = self._actor_mesh

        first_error: Exception | None = None

        if self._running:
            actor.stop_instance.choose(
                self._instance_name,
                self._apptainer_config_path,
            ).get()
            self._running = False
            self._unregister_from_machine()

        if self._staging_dir is not None:
            try:
                actor.cleanup_staging_dir.choose(self._staging_dir).get()
            except Exception as exc:
                first_error = first_error or exc
            else:
                self._staging_dir = None

        if self._cwd_dir is not None:
            try:
                actor.cleanup_staging_dir.choose(self._cwd_dir).get()
            except Exception as exc:
                first_error = first_error or exc
            else:
                self._cwd_dir = None

        if self._host_dir is not None:
            try:
                actor.cleanup_staging_dir.choose(self._host_dir).get()
            except Exception as exc:
                first_error = first_error or exc
            else:
                self._host_dir = None

        if first_error is not None:
            raise first_error

    async def astop(self) -> None:
        """Async-native counterpart of :meth:`stop` (awaits the cleanup RPCs)."""
        if self._actor_mesh is None:
            return
        actor = self._actor_mesh

        first_error: Exception | None = None

        if self._running:
            await actor.stop_instance.choose(
                self._instance_name,
                self._apptainer_config_path,
            )
            self._running = False
            self._unregister_from_machine()

        if self._staging_dir is not None:
            try:
                await actor.cleanup_staging_dir.choose(self._staging_dir)
            except Exception as exc:
                first_error = first_error or exc
            else:
                self._staging_dir = None

        if self._cwd_dir is not None:
            try:
                await actor.cleanup_staging_dir.choose(self._cwd_dir)
            except Exception as exc:
                first_error = first_error or exc
            else:
                self._cwd_dir = None

        if self._host_dir is not None:
            try:
                await actor.cleanup_staging_dir.choose(self._host_dir)
            except Exception as exc:
                first_error = first_error or exc
            else:
                self._host_dir = None

        if first_error is not None:
            raise first_error

    def _exec_prefix(
        self,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        """Build the ``apptainer exec`` prefix for this container."""
        return _build_exec_prefix(
            self._instance_name,
            apptainer_config_path=self._apptainer_config_path,
            instance_env=self._env,
            apptainer_args=self._apptainer_args,
            bind_mounts=self._bind_mounts_effective,
            env=env,
            cwd=cwd,
        )

    @property
    def _proc_mesh_for_exec(self) -> ProcMesh:
        return self._machine.proc_mesh

    def _exec_task_pool(self) -> TaskPool:
        return self._machine._ensure_task_pool()

    def _host_argv(
        self, argv: list[str], env: dict[str, str] | None, cwd: str | None
    ) -> tuple[list[str], str | None]:
        # env/cwd apply inside the container via ``--env``/``--pwd``; the host
        # ``apptainer`` launcher keeps the worker's own environment.
        cmd = self._exec_prefix(env=env, cwd=cwd or self._cwd) + argv
        logger.debug(
            "Executing Apptainer command: %s", _format_apptainer_command_for_log(cmd)
        )
        return cmd, None

    def is_running(self) -> bool:
        """Check whether the Apptainer instance is running on the remote node."""
        actor = self._ensure_actor()
        return bool(
            actor.is_running.choose(
                self._instance_name, self._apptainer_config_path
            ).get()
        )

    # Files cross the container boundary through the staging directory, which
    # is the same path on the worker host and inside the container. The
    # container can replace anything in it at any moment, so the Machine only
    # copies to and from the host directory, and one host command renames the
    # copy across and runs the ``apptainer exec`` step that moves it between
    # the staging directory and its in-container path. A directory is packed
    # on the client or in the container, never by the host.

    def _copy_paths(self, path: str) -> tuple[str, str]:
        """Return the (staging, host) paths for one copy of *path*."""
        if self._host_dir is None:
            raise RuntimeError("Container not started — call start() first")
        # A unique name keeps concurrent copies of same-named files apart.
        name = f"{uuid.uuid4().hex}-{os.path.basename(path)}"
        return (
            os.path.join(self.staging_dir, name),
            os.path.join(self._host_dir, name),
        )

    def _via_host(
        self,
        script: str,
        first: str,
        second: str,
        argv: list[str],
        cwd: str | None = None,
    ) -> list[str]:
        """Return a host command running *script* around in-container *argv*."""
        host_argv, _ = self._host_argv(argv, None, cwd)
        return ["sh", "-c", script, "sh", first, second, *host_argv]

    def _give(self, held: str, staged: str, script: str, remote_path: str) -> list[str]:
        return self._via_host(
            _HOST_GIVE, held, staged, ["sh", "-c", script, "sh", staged, remote_path]
        )

    def _take(self, remote_path: str, staged: str, held: str) -> list[str]:
        return self._via_host(
            _HOST_TAKE,
            staged,
            held,
            ["sh", "-c", _STAGE, "sh", remote_path, staged],
            cwd="/",
        )

    @staticmethod
    def _check_copy(result: CompletedProcess[str] | CompletedProcess[bytes]) -> None:
        if result.returncode != 0:
            raise FileCopyError(
                f"container file copy failed: {_subprocess_output_text(result.stderr)}"
            )

    def upload(self, local_path: str, remote_path: str) -> None:
        staged, held = self._copy_paths(local_path)
        try:
            if not os.path.isdir(local_path):
                self._machine.upload(local_path, held)
                script = _PLACE
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    archive = os.path.join(tmp, "upload.tar")
                    tree_archive.pack(local_path, archive)
                    self._machine.upload(archive, held)
                script = _UNPACK
            self._check_copy(
                self._machine.run(self._give(held, staged, script, remote_path))
            )
        except BaseException:
            # A successful step consumed both copies; a failed one may not have.
            try:
                self._machine.run(["rm", "-rf", staged, held])
            except Exception:
                logger.warning("Could not remove failed upload %s", held, exc_info=True)
            raise

    def download(self, remote_path: str, local_path: str) -> None:
        staged, held = self._copy_paths(remote_path)
        try:
            result = self._machine.run(self._take(remote_path, staged, held))
            if result.returncode != _STAGED_DIRECTORY:
                self._check_copy(result)
                self._machine.download(held, local_path)
                return
            with tempfile.TemporaryDirectory() as tmp:
                archive = os.path.join(tmp, "download.tar")
                self._machine.download(held, archive)
                tree_archive.unpack(archive, local_path)
        finally:
            self._machine.run(["rm", "-rf", staged, held])

    async def aupload(self, local_path: str, remote_path: str) -> None:
        """Async-native counterpart of :meth:`upload`."""
        staged, held = self._copy_paths(local_path)
        try:
            if not os.path.isdir(local_path):
                await self._machine.aupload(local_path, held)
                script = _PLACE
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    archive = os.path.join(tmp, "upload.tar")
                    await asyncio.to_thread(tree_archive.pack, local_path, archive)
                    await self._machine.aupload(archive, held)
                script = _UNPACK
            self._check_copy(
                await self._machine.arun(self._give(held, staged, script, remote_path))
            )
        except BaseException:
            try:
                await self._machine.arun(["rm", "-rf", staged, held])
            except Exception:
                logger.warning("Could not remove failed upload %s", held, exc_info=True)
            raise

    async def adownload(self, remote_path: str, local_path: str) -> None:
        """Async-native counterpart of :meth:`download`."""
        staged, held = self._copy_paths(remote_path)
        try:
            result = await self._machine.arun(self._take(remote_path, staged, held))
            if result.returncode != _STAGED_DIRECTORY:
                self._check_copy(result)
                await self._machine.adownload(held, local_path)
                return
            with tempfile.TemporaryDirectory() as tmp:
                archive = os.path.join(tmp, "download.tar")
                await self._machine.adownload(held, archive)
                await asyncio.to_thread(tree_archive.unpack, archive, local_path)
        finally:
            await self._machine.arun(["rm", "-rf", staged, held])

    def __enter__(self) -> Container:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    async def __aenter__(self) -> Container:
        await self.astart()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.astop()
