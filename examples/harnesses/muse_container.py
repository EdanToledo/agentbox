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

r"""Install and optionally run Muse Code inside an AgentBox container.

Muse Code is installed with Meta's installer under a throwaway ``HOME`` in the
container's staging directory (a host directory bind-mounted into the container
and removed when it closes), so its binary, configuration, and session state
never touch the host home directory.

With ``--prompt``, ``muse exec`` calls the Meta Model API directly and
authenticates with ``META_API_KEY`` (see
https://dev.meta.ai/docs/muse-code/auth), so the key must be available inside
the container. It is written to the process's stdin and exported only into the
``muse exec`` process, keeping it out of every command line and the container's
base environment. Muse Code and the commands it runs can still read it, so use
a scoped key you can revoke for untrusted tasks.

Usage:
    # Install Muse Code and print its version (no API key required):
    python examples/harnesses/muse_container.py

    # Run a headless prompt (--model is optional; Muse Code picks its default):
    META_API_KEY=... python examples/harnesses/muse_container.py \
        --prompt "Reply with exactly: Hello from Muse"

Prerequisites:
    - Apptainer on PATH (on macOS, inside the Lima VM)
    - ``pip install -e .``
    - ``curl`` and ``bash`` inside the image (the default image has both)
    - ``META_API_KEY`` when ``--prompt`` is used; see https://dev.meta.ai/ for
      API keys and model ids

Success: the script prints ``Installed: <version>`` and, with ``--prompt``,
Muse Code's reply; it exits with Muse Code's exit status.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentbox import AgentBoxManager, ApptainerOptions, Container
from host_llm_bridge import proxy_env, redact


DEFAULT_IMAGE = "docker://python:3.12-bookworm"
INSTALLER_URL = "https://dev.meta.ai/install.sh"

# Apptainer does not let ``--env`` override HOME, so set it inside the
# container. When stdin is used, read the first line into META_API_KEY, then
# detach stdin from the command.
_SET_HOME = 'HOME="$1"; export HOME; shift; exec "$@"'
_SET_HOME_AND_KEY = (
    'HOME="$1"; export HOME; shift; '
    "IFS= read -r META_API_KEY; export META_API_KEY; "
    'exec "$@" </dev/null'
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="SIF path or URI")
    parser.add_argument("--prompt", help="Run this prompt with `muse exec`")
    parser.add_argument(
        "--model", help="Muse model id (default: Muse Code's default model)"
    )
    parser.add_argument("--max-model-steps", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.max_model_steps < 1:
        parser.error("--max-model-steps must be at least 1")
    return args


def _with_home(home: str, argv: list[str], *, key_on_stdin: bool = False) -> list[str]:
    script = _SET_HOME_AND_KEY if key_on_stdin else _SET_HOME
    return ["sh", "-c", script, "sh", home, *argv]


def _muse_argv(muse_bin: str, args: argparse.Namespace) -> list[str]:
    argv = [
        muse_bin,
        "exec",
        "--workspace",
        "/workspace",
        "--disable-approval",
        # The container is the sandbox boundary; Muse Code's bubblewrap
        # sandbox does not nest inside Apptainer.
        "--disable-sandbox",
        "--max-model-steps",
        str(args.max_model_steps),
    ]
    if args.model:
        argv += ["--model", args.model]
    return [*argv, "--", args.prompt]


async def _amain(args: argparse.Namespace) -> int:
    api_key = os.environ.get("META_API_KEY")
    if args.prompt is not None and not api_key:
        raise SystemExit("META_API_KEY must be set when --prompt is used")

    loopback_host = "host.lima.internal" if sys.platform == "darwin" else None

    async with AgentBoxManager.create_local(num_workers=1) as manager:
        async with manager.aacquire_machine() as machine:
            async with Container(
                machine,
                image=args.image,
                options=ApptainerOptions(no_home=True),
                env=proxy_env("127.0.0.1", "localhost", loopback_host=loopback_host),
                cwd="/workspace",
            ) as container:
                staging = container.staging_dir
                home = os.path.join(staging, "home")
                # The installer's default location is ~/.local/bin.
                install_dir = os.path.join(home, ".local", "bin")
                muse_bin = os.path.join(install_dir, "muse")
                env = {
                    "PATH": f"{install_dir}:/usr/local/bin:/usr/bin:/bin",
                    "TMPDIR": staging,
                }
                await container.arun(["mkdir", "-p", home], check=True)

                print("Installing Muse Code...")
                install = await container.arun(
                    _with_home(
                        home, ["bash", "-c", f"curl -fsSL {INSTALLER_URL} | bash"]
                    ),
                    env=env,
                )
                if install.returncode != 0:
                    print(install.stderr or install.stdout, file=sys.stderr)
                    return install.returncode

                version = await container.arun(
                    _with_home(home, [muse_bin, "--version"]), env=env, check=True
                )
                print(f"Installed: {version.stdout.strip()}")
                if args.prompt is None:
                    return 0

                assert api_key is not None
                print(f"Running Muse Code with {args.model or 'its default model'}...")
                result = await container.arun(
                    _with_home(home, _muse_argv(muse_bin, args), key_on_stdin=True),
                    env=env,
                    input=api_key + "\n",
                    timeout=args.timeout,
                )
                secrets = [api_key]
                print(redact(result.stderr, secrets), file=sys.stderr, end="")
                print(redact(result.stdout, secrets), end="")
                return result.returncode


def main() -> int:
    return asyncio.run(_amain(get_args()))


if __name__ == "__main__":
    raise SystemExit(main())
