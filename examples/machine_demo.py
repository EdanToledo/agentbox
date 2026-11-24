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

"""Run commands, processes, and file transfers on an AgentBox worker host.

Tours the execution API that :class:`Machine` and :class:`Container` share
(:class:`agentbox.ExecutionTarget`) on a local worker, so it needs no
scheduler and no Apptainer:

1. ``run`` / ``arun``: one-shot commands with captured output, ``env``,
   ``cwd``, ``input``, ``check``, and ``timeout``.
2. ``spawn``: a pipe-backed process and a PTY-backed interactive shell.
3. ``upload`` / ``download``: a file, then a directory tree.

Usage:
    python examples/machine_demo.py

Swap ``create_local`` for ``create_slurm(allocation=SlurmAllocation(...))`` to
run the same code on Slurm workers. Every step asserts its result; the script
prints ``All machine demos passed.`` on success.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile

from agentbox import AgentBoxManager, CalledProcessError, Machine, TimeoutExpired


def demo_run(machine: Machine) -> None:
    print("--- run ---")
    # A str runs through /bin/sh -c; a list is executed as argv.
    result = machine.run("echo hello from $(hostname)")
    print(f"  shell string: {result.stdout.strip()}")
    assert result.returncode == 0

    result = machine.run(["printf", "%s\n", "argv", "is", "not", "re-parsed"])
    assert result.stdout.split() == ["argv", "is", "not", "re-parsed"]

    # env adds variables; cwd is a path on the worker.
    result = machine.run('echo "$GREETING from $PWD"', env={"GREETING": "hi"}, cwd="/")
    assert result.stdout.strip() == "hi from /", result.stdout
    print(f"  env + cwd:    {result.stdout.strip()}")

    result = machine.run(["tr", "a-z", "A-Z"], input="piped input\n")
    assert result.stdout == "PIPED INPUT\n"

    result = machine.run("echo oops >&2; exit 3")
    assert (result.returncode, result.stderr.strip()) == (3, "oops")
    try:
        machine.run("exit 4", check=True)
    except CalledProcessError as error:
        print(f"  check=True:   CalledProcessError(returncode={error.returncode})")
    else:
        raise AssertionError("check=True did not raise")

    try:
        machine.run(["sleep", "30"], timeout=1)
    except TimeoutExpired:
        print("  timeout=1:    TimeoutExpired")
    else:
        raise AssertionError("timeout did not raise")


async def demo_arun(machine: Machine) -> None:
    print("--- arun (concurrent) ---")
    results = await asyncio.gather(
        *(machine.arun(f"sleep 1; echo task {i}") for i in range(4))
    )
    outputs = [r.stdout.strip() for r in results]
    assert outputs == [f"task {i}" for i in range(4)], outputs
    print(f"  {outputs}")


def demo_spawn(machine: Machine) -> None:
    print("--- spawn ---")
    proc = machine.spawn(["sort"])
    stdout, _ = proc.communicate(input="pear\napple\nfig\n")
    assert stdout == "apple\nfig\npear\n", stdout
    print(f"  pipes: sorted -> {stdout.split()}")

    shell = machine.spawn(["bash", "--norc", "--noprofile"], pty=True)
    output, _ = shell.communicate(input="tty && echo pty-ok\nexit\n")
    assert "pty-ok" in output and "/dev/pts/" in output, output
    print(f"  pty:   exit code {shell.returncode}, saw a /dev/pts terminal")


def demo_files(machine: Machine) -> None:
    print("--- upload / download ---")
    with tempfile.TemporaryDirectory() as local_dir:
        src = os.path.join(local_dir, "payload.txt")
        with open(src, "w") as f:
            f.write("round trip\n")
        remote_dir = machine.run(["mktemp", "-d"], check=True).stdout.strip()
        try:
            remote = f"{remote_dir}/payload.txt"
            machine.upload(src, remote)
            machine.run(["sh", "-c", 'echo "edited remotely" >> "$1"', "sh", remote])
            back = os.path.join(local_dir, "back.txt")
            machine.download(remote, back)
            with open(back) as f:
                contents = f.read()
        finally:
            machine.run(["rm", "-rf", remote_dir])
    assert contents == "round trip\nedited remotely\n", contents
    print(f"  file: {contents.splitlines()}")

    # A directory copies as a tree; symlinks stay links.
    with tempfile.TemporaryDirectory() as local_dir:
        tree = os.path.join(local_dir, "tree")
        os.makedirs(os.path.join(tree, "src"))
        with open(os.path.join(tree, "src", "main.py"), "w") as f:
            f.write("print('hi')\n")
        os.symlink("src/main.py", os.path.join(tree, "entry"))
        remote_dir = machine.run(["mktemp", "-d"], check=True).stdout.strip()
        try:
            machine.upload(tree, f"{remote_dir}/tree")
            listing = machine.run(["find", ".", "-print"], cwd=f"{remote_dir}/tree")
            machine.run(["touch", f"{remote_dir}/tree/src/new.txt"], check=True)
            back = os.path.join(local_dir, "back")
            machine.download(f"{remote_dir}/tree", back)
        finally:
            machine.run(["rm", "-rf", remote_dir])
        assert sorted(listing.stdout.split()) == [
            ".",
            "./entry",
            "./src",
            "./src/main.py",
        ], listing.stdout
        assert sorted(os.listdir(os.path.join(back, "src"))) == ["main.py", "new.txt"]
        assert os.readlink(os.path.join(back, "entry")) == "src/main.py"
    print("  directory: uploaded, edited remotely, downloaded")


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    with AgentBoxManager.create_local(num_workers=1) as manager:
        with manager.acquire_machine() as machine:
            print(f"Acquired worker {machine.hostname()}")
            demo_run(machine)
            asyncio.run(demo_arun(machine))
            demo_spawn(machine)
            demo_files(machine)

    print("All machine demos passed.")


if __name__ == "__main__":
    main()
