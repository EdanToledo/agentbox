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

"""AgentBox basic CI smoketest.

Exercises the core workflow against the CI-built SIF:
  1. ``AgentBoxManager.create_local(num_workers=1)`` → ``acquire_machine``.
  2. Release a lease while a PTY process is still running, then reuse the worker.
  3. ``Container(machine, image=...)``.
  4. Shell commands via ``container.run(...)``.
  5. ``container.upload(...)`` host→container, then execute the file.
  6. A directory round trip: ``upload`` a tree, edit it inside, ``download`` it.
"""

import os
import sys
import tempfile

from agentbox import AgentBoxManager, ApptainerOptions, Container


if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    __package__ = "examples.ci"

from ._common import resolve_image, run_smoketest


def main(local: bool = False) -> None:
    image = resolve_image(local=local)
    print(f"Starting AgentBox basic smoketest (image={image})...")

    options = ApptainerOptions(writable_tmpfs=True, no_home=True)

    with AgentBoxManager.create_local(num_workers=1) as manager:
        print("Manager created.")
        with manager.acquire_machine() as machine:
            # Releasing a lease with a live PTY must not crash the client.
            machine.spawn(["sleep", "600"], pty=True)
        print("  [PASS] release with a live PTY")

        with manager.acquire_machine() as machine:
            print("Machine acquired.")
            with Container(machine, image=image, options=options) as container:
                print("Container started.")

                _check(
                    container,
                    ["echo", "Hello from AgentBox CI"],
                    "Hello from AgentBox CI",
                    label="echo",
                )
                _check(
                    container,
                    ["python3", "-c", "print(2 + 2)"],
                    "4",
                    label="python arithmetic",
                )
                _check(
                    container,
                    ["sh", "-c", "mkdir -p /tmp/workspace && cd /tmp/workspace && pwd"],
                    "/tmp/workspace",
                    label="working directory",
                )

                script = "#!/usr/bin/env python3\nprint('smoketest-file-copy-ok')\n"
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as tmp:
                    tmp.write(script)
                    local_path = tmp.name
                try:
                    container.upload(local_path, "/tmp/smoketest.py")
                    print("  File uploaded into container.")
                    _check(
                        container,
                        ["python3", "/tmp/smoketest.py"],
                        "smoketest-file-copy-ok",
                        label="file copy + execute",
                    )
                finally:
                    os.unlink(local_path)

                _check_directory_round_trip(container)

    print("All AgentBox basic smoketest checks passed.")


def _check_directory_round_trip(container: Container) -> None:
    with tempfile.TemporaryDirectory() as local_dir:
        tree = os.path.join(local_dir, "tree")
        os.makedirs(os.path.join(tree, "pkg"))
        with open(os.path.join(tree, "pkg", "mod.py"), "w") as f:
            f.write("VALUE = 'smoketest-dir-ok'\n")
        os.symlink("pkg/mod.py", os.path.join(tree, "link.py"))
        container.upload(tree, "/tmp/tree")
        _check(
            container,
            [
                "python3",
                # -B: no __pycache__, so the downloaded tree matches exactly.
                "-B",
                "-c",
                "import sys; sys.path[:0] = ['/tmp/tree']; "
                "import pkg.mod; print(pkg.mod.VALUE)",
            ],
            "smoketest-dir-ok",
            label="directory upload",
        )
        _check(container, ["touch", "/tmp/tree/pkg/new.txt"], "", label="edit tree")
        back = os.path.join(local_dir, "back")
        container.download("/tmp/tree", back)
        names = sorted(os.listdir(os.path.join(back, "pkg")))
        if (
            names != ["mod.py", "new.txt"]
            or os.readlink(os.path.join(back, "link.py")) != "pkg/mod.py"
        ):
            raise AssertionError(f"directory download: unexpected tree {names}")
        print("  [PASS] directory download")


def _check(container: Container, argv: list[str], expected: str, *, label: str) -> None:
    result = container.run(argv)
    if result.returncode != 0:
        raise AssertionError(
            f"{label}: nonzero exit {result.returncode}; stderr={result.stderr!r}"
        )
    stdout = result.stdout or ""
    if expected not in stdout:
        raise AssertionError(
            f"{label}: expected {expected!r} in stdout, got {result.stdout!r}"
        )
    print(f"  [PASS] {label}")


if __name__ == "__main__":
    run_smoketest("AgentBox basic shell + file copy smoketest", main)
