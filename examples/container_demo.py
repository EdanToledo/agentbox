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

"""Container demo: commands, an interactive shell, files, and overlays.

Monarch runs only on the worker host and drives an Apptainer container through
``apptainer exec``, so the image does **not** need Monarch or AgentBox
installed.

Usage:
    # Basic — uses a standard Docker image (pulled automatically):
    python examples/container_demo.py --image docker://ubuntu:22.04

    # With a pre-built SIF image:
    python examples/container_demo.py --image ./my-image.sif

    # With GPU passthrough:
    python examples/container_demo.py --image docker://ubuntu:22.04 --nv

    # With a persistent overlay (auto-created 256 MiB ext3 image):
    python examples/container_demo.py --image docker://ubuntu:22.04 --create-overlay

    # With an existing overlay image:
    python examples/container_demo.py --image ./my.sif --overlay /tmp/my_overlay.img

    # With a SIF that has an embedded writable overlay partition:
    python examples/container_demo.py --image ./my.sif --writable

Prerequisites:
    - Apptainer installed and on PATH
    - ``pip install -e .``

Success: the script ends with "All Container demos completed successfully!".
"""

import argparse
import logging
import os
import subprocess
import tempfile

from agentbox import AgentBoxManager, ApptainerOptions, Container


logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default="docker://python:3.12-slim",
        help="Apptainer SIF image or Docker URI (default: docker://python:3.12-slim)",
    )
    parser.add_argument(
        "--nv",
        action="store_true",
        help="Pass --nv to Apptainer for GPU access",
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Extra bind mounts (src:dst), may be repeated",
    )
    overlay_group = parser.add_mutually_exclusive_group()
    overlay_group.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="PATH",
        help="Persistent overlay image or directory (may be repeated; append :ro for read-only)",
    )
    overlay_group.add_argument(
        "--create-overlay",
        action="store_true",
        help="Auto-create a temporary 256 MiB ext3 overlay image for the demo",
    )
    overlay_group.add_argument(
        "--writable",
        action="store_true",
        help="Pass --writable to modify an overlay embedded in the SIF",
    )
    return parser.parse_args()


def demo_one_shot_commands(container: Container) -> None:
    """Run one-shot commands inside the container."""
    print("\n--- One-shot commands ---")

    # A str runs through /bin/sh -c inside the container.
    result = container.run("echo 'Hello from the container!'", check=True)
    print(f"  echo: {result.stdout.strip()}")

    # A list is executed as argv, with no shell parsing.
    result = container.run(["uname", "-a"], check=True)
    print(f"  uname: {result.stdout.strip()}")

    result = container.run("head -2 /etc/os-release", check=True)
    print("  os-release:\n    " + result.stdout.strip().replace("\n", "\n    "))

    # Container-level env from the constructor, plus per-call additions.
    result = container.run('echo "$MY_VAR $EXTRA"', env={"EXTRA": "+ per-call"})
    assert result.stdout.strip() == "set_from_host + per-call", result.stdout
    print(f"  env: {result.stdout.strip()!r}")


def demo_interactive_shell(container: Container) -> None:
    """Drive one bash process in a PTY; its state persists across commands."""
    print("\n--- Interactive shell (spawn with pty=True) ---")
    shell = container.spawn(["bash", "--norc", "--noprofile"], pty=True)
    output, _ = shell.communicate(
        input='export GREETING=persisted\ncd /tmp\necho "$GREETING in $PWD"\nexit\n'
    )
    assert "persisted in /tmp" in output, output
    print(f"  bash exited with {shell.returncode}; state persisted across commands")


def demo_overlay(container: Container) -> None:
    """Demonstrate a persistent overlay by writing to a normally read-only path.

    Without an overlay the container filesystem is immutable; writes to paths
    like ``/usr/local/bin/`` fail.  With an overlay layer those writes succeed
    and persist as long as the overlay image is reused.
    """
    print("\n--- Persistent overlay ---")

    marker = "/usr/local/bin/agentbox_overlay_marker"

    # Confirm the path doesn't exist yet (or was left from a previous run).
    result = container.run(["test", "-f", marker])
    if result.returncode == 0:
        print(
            f"  {marker} already exists (leftover from a previous run — overlay is persistent!)"
        )
    else:
        print(f"  {marker} does not exist yet — writing it now via overlay …")
        result = container.run(f"echo 'written via overlay' > {marker}")
        if result.returncode != 0:
            print(f"  Write failed (rc={result.returncode}): {result.stderr.strip()}")
            print("  (Is the overlay writable?  Try --fakeroot if needed.)")
            return
        print(f"  Write succeeded (rc={result.returncode})")

    # Read it back.
    result = container.run(["cat", marker])
    print(f"  Contents: {result.stdout.strip()!r}")
    print(
        "  Re-run with the same --overlay path to confirm the file persists across container restarts."
    )


def demo_file_copy(container: Container) -> None:
    """Demonstrate copying files to/from the container."""
    print("\n--- File copy ---")

    # Create a local temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This file was created on the host.\n")
        local_src = f.name

    try:
        # Copy into the container
        # ``--containall`` (the default) gives the instance a private /tmp.
        container_dest = "/tmp/demo-file.txt"
        container.upload(local_src, container_dest)
        print(f"  Copied {local_src} -> container:{container_dest}")

        result = container.run(["cat", container_dest], check=True)
        print(f"  Contents inside container: {result.stdout.strip()!r}")

        container.run(
            f"echo 'Modified inside container.' >> {container_dest}", check=True
        )

        # Copy back
        local_dest = local_src + ".from_container"
        container.download(container_dest, local_dest)
        with open(local_dest) as f:
            contents = f.read()
        print(f"  Copied back to host, contents: {contents.strip()!r}")
        assert contents.endswith("Modified inside container.\n"), contents

        os.unlink(local_dest)
    finally:
        os.unlink(local_src)


def main() -> None:
    args = get_args()

    # --create-overlay: build a temporary ext3 overlay image on the fly.
    _tmp_overlay_path: str | None = None
    overlay_paths: list[str] = list(args.overlay)
    if args.create_overlay:
        fd, _tmp_overlay_path = tempfile.mkstemp(suffix=".img")
        os.close(fd)
        os.unlink(
            _tmp_overlay_path
        )  # apptainer overlay create requires a non-existent path
        overlay_img = _tmp_overlay_path
        print(f"Creating temporary overlay image: {overlay_img} …")
        subprocess.run(
            ["apptainer", "overlay", "create", "--size", "256", overlay_img], check=True
        )
        overlay_paths = [overlay_img]

    print("=" * 60)
    print("Container Demo")
    print("=" * 60)
    print(f"  Image: {args.image}")
    if args.nv:
        print("  GPU: --nv enabled")
    if overlay_paths:
        print(f"  Overlay(s): {overlay_paths}")
    if args.writable:
        print("  Mode: --writable (SIF-embedded overlay)")
    print("=" * 60)

    options = ApptainerOptions(
        nv=bool(args.nv),
        overlay=overlay_paths,
        writable=bool(args.writable),
    )

    try:
        with (
            AgentBoxManager.create_local(num_workers=1) as manager,
            manager.acquire_machine() as machine,
            Container(
                machine,
                image=args.image,
                options=options,
                env={"MY_VAR": "set_from_host"},
                bind_mounts=args.bind or None,
            ) as container,
        ):
            print(f"\n  Instance: {container.instance_name}")
            print(f"  Running: {container.is_running()}")

            demo_one_shot_commands(container)
            demo_interactive_shell(container)
            demo_file_copy(container)
            if overlay_paths or args.writable:
                demo_overlay(container)

            print("\n" + "=" * 60)
            print("All Container demos completed successfully!")
            print("=" * 60)
    finally:
        if _tmp_overlay_path is not None and os.path.exists(_tmp_overlay_path):
            os.unlink(_tmp_overlay_path)

    print("\nContainer stopped and cleaned up.")


if __name__ == "__main__":
    main()
