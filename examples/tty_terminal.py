#!/usr/bin/env python3
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

"""Interactive remote terminal backed by a PTY.

Opens a pseudo-terminal on an AgentBox worker (optionally inside an Apptainer
container) and bridges it to the local terminal, so colours, line editing, and
interactive programs (vim, htop, a Python REPL) work as they would locally.
A PTY merges stdout and stderr into one stream exactly as a real terminal
does.

Usage:
    # Interactive bash on a local worker:
    python examples/tty_terminal.py

    # Inside an Apptainer container:
    python examples/tty_terminal.py --image docker://ubuntu:24.04

    # Run one command and exit (non-interactive; usable in scripts):
    python examples/tty_terminal.py --command "python3 --version"
    python examples/tty_terminal.py --image ./my-image.sif --command "uname -a"

    # GPU passthrough (requires a compatible image):
    python examples/tty_terminal.py --image ./my-image.sif --nv

How it works:
    1. ``target.spawn(argv, pty=True)`` starts bash in a remote PTY, where
       ``target`` is the acquired :class:`Machine` or a :class:`Container`.
    2. The local terminal switches to raw mode so every keypress (arrow keys,
       Ctrl+C, ...) is forwarded to the remote PTY.
    3. A background thread forwards local stdin; the main thread streams PTY
       output to local stdout.
    4. On exit the local terminal settings are restored.

Limitation: terminal resizes (SIGWINCH) are not propagated.

Success: with ``--command`` the script exits with the remote command's status.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import termios
import threading
import tty

from agentbox import AgentBoxManager, ApptainerOptions, Container, Machine, PtyProcess


logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def _bash_argv(command: str | None) -> list[str]:
    """Return an interactive bash, or ``bash -c <command>`` when given."""
    if command is not None:
        return ["bash", "-c", command]
    return ["bash", "--norc", "--noprofile"]


def _run_interactive(proc: PtyProcess) -> int:
    """Bridge the local terminal to the remote PTY in raw mode.

    Puts the local tty into raw mode, then runs two concurrent tasks:
    - **Background thread**: reads raw bytes from local stdin and writes them
      to the remote PTY (so keypresses, arrow keys, Ctrl+C all pass through).
    - **Main thread**: reads output from the remote PTY and writes it to local
      stdout.

    Returns the remote process exit code.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    stop = threading.Event()

    def _stdin_to_pty() -> None:
        """Read local stdin bytes and forward to the remote PTY."""
        while not stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            except (ValueError, OSError):
                break
            if not ready:
                continue
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            try:
                proc.write(data)
            except RuntimeError:
                # PTY exited
                break

    try:
        tty.setraw(fd)
        stdin_thread = threading.Thread(target=_stdin_to_pty, daemon=True)
        stdin_thread.start()

        # Stream remote PTY output to local stdout.
        for chunk in proc.stdout:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    finally:
        stop.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return proc.wait()


def _run_command(proc: PtyProcess) -> int:
    """Stream output from a non-interactive PTY command until the process exits.

    The command was already baked into the PTY's argv via ``bash -c <cmd>``,
    so there is nothing to write — just drain the output and wait.

    Returns the remote process exit code.
    """
    for chunk in proc.stdout:
        if isinstance(chunk, str):
            sys.stdout.write(chunk)
        else:
            sys.stdout.buffer.write(chunk)
        sys.stdout.flush()

    return proc.wait()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgentBox TTY Terminal — interactive remote shell via PTY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Container options
    parser.add_argument(
        "--image",
        metavar="IMAGE",
        help=(
            "Apptainer SIF path or Docker URI (e.g. docker://ubuntu:22.04). "
            "If omitted, bash runs on the bare remote host."
        ),
    )
    parser.add_argument(
        "--nv", action="store_true", help="Pass --nv to Apptainer for NVIDIA GPU access"
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Extra bind mount (repeatable, e.g. /data:/data)",
    )

    # Session options
    parser.add_argument(
        "--command",
        "-c",
        metavar="CMD",
        help="Execute a single command and exit (non-interactive mode)",
    )

    return parser.parse_args()


def _run_session(target: Machine | Container, command: str | None) -> int:
    argv = _bash_argv(command)
    print(f"Spawning remote PTY: {argv}")
    proc = target.spawn(argv, pty=True, text=False)

    if command is not None:
        print("-" * 60, flush=True)
        rc = _run_command(proc)
        print("-" * 60)
        print(f"Exit code: {rc}")
        return rc

    print("Interactive session; type 'exit' or press Ctrl+D to quit.\n")
    rc = _run_interactive(proc)
    print(f"\nSession ended with exit code {rc}.")
    return rc


def main() -> int:
    args = get_args()

    with AgentBoxManager.create_local(num_workers=1) as manager:
        with manager.acquire_machine() as machine:
            print(f"Worker host: {machine.hostname()}")
            if not args.image:
                return _run_session(machine, args.command)
            with Container(
                machine,
                image=args.image,
                options=ApptainerOptions(nv=args.nv),
                bind_mounts=args.bind or None,
            ) as container:
                print(f"Container instance: {container.instance_name}")
                return _run_session(container, args.command)


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rc = 130
    raise SystemExit(rc)
