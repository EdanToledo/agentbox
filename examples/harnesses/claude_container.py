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

r"""Install and optionally run Claude Code inside an AgentBox container.

Claude Code is installed with Anthropic's native installer into the
container's staging directory (a host directory bind-mounted into the container
and removed when it closes), so nothing is written to the host home directory.
The staging directory also avoids ``--containall``'s ``/tmp``, which Apptainer
caps at ``sessiondir max size`` (64 MiB by default), too small for Claude Code.

The real ``ANTHROPIC_API_KEY`` never enters the container. When ``--prompt``
is given, a LiteLLM proxy (``host_llm_bridge.HostBridge``) runs as a
worker-host process outside the container's PID namespace and holds the key;
Claude Code only sees a random per-run bridge credential that stops working
when the run ends.

Usage:
    # Install Claude Code and print its version (no API key required):
    python examples/harnesses/claude_container.py

    # Run a headless prompt:
    ANTHROPIC_API_KEY=... python examples/harnesses/claude_container.py \
        --prompt "Reply with exactly: Hello from Claude"

Prerequisites:
    - Apptainer on PATH (on macOS, inside the Lima VM)
    - ``pip install -e .``
    - ``curl`` and ``bash`` inside the image (the default image has both)
    - ``ANTHROPIC_API_KEY`` when ``--prompt`` is used

Success: the script prints ``Installed: <version>`` and, with ``--prompt``,
Claude Code's reply; it exits with Claude Code's exit status.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentbox import AgentBoxManager, ApptainerOptions, Container
from host_llm_bridge import HostBridge, litellm_config, proxy_env, redact


DEFAULT_IMAGE = "docker://python:3.12-bookworm"
DEFAULT_MODEL = "claude-opus-5"
INSTALLER_URL = "https://claude.ai/install.sh"
LITELLM_SPEC = "litellm[proxy]>=1.63.8,<2"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="SIF path or URI")
    parser.add_argument(
        "--claude-version",
        default="stable",
        help="Claude Code release channel or exact version (default: stable)",
    )
    parser.add_argument("--prompt", help="Run this prompt with `claude -p`")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.max_turns < 1:
        parser.error("--max-turns must be at least 1")
    return args


def _claude_argv(claude_bin: str, args: argparse.Namespace) -> list[str]:
    return [
        claude_bin,
        "--bare",
        "-p",
        "--model",
        args.model,
        # The container is the sandbox boundary.
        "--permission-mode",
        "bypassPermissions",
        "--max-turns",
        str(args.max_turns),
        "--no-session-persistence",
        "--",
        args.prompt,
    ]


async def _amain(args: argparse.Namespace) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.prompt is not None and not api_key:
        raise SystemExit("ANTHROPIC_API_KEY must be set when --prompt is used")

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
                claude_home = os.path.join(container.staging_dir, "claude-home")
                claude_bin = os.path.join(claude_home, ".local", "bin", "claude")
                env = {
                    "DISABLE_AUTOUPDATER": "1",
                    "PATH": f"{os.path.dirname(claude_bin)}:/usr/local/bin:/usr/bin:/bin",
                    "TMPDIR": container.staging_dir,
                }
                # Apptainer refuses to override HOME through --env, so set it
                # on each command instead.
                with_home = ["env", f"HOME={claude_home}"]

                print(f"Installing Claude Code ({args.claude_version})...")
                install = await container.arun(
                    [
                        *with_home,
                        "bash",
                        "-c",
                        f'mkdir -p "$HOME" && curl -fsSL {INSTALLER_URL} '
                        '| bash -s -- "$1"',
                        "install",
                        args.claude_version,
                    ],
                    env=env,
                )
                if install.returncode != 0:
                    print(install.stderr or install.stdout, file=sys.stderr)
                    return install.returncode

                version = await container.arun(
                    [*with_home, claude_bin, "--version"], env=env, check=True
                )
                print(f"Installed: {version.stdout.strip()}")
                if args.prompt is None:
                    return 0

                assert api_key is not None
                print(f"Running Claude Code with {args.model}...")
                async with HostBridge(
                    machine,
                    config=litellm_config(
                        model=args.model,
                        upstream_model=f"anthropic/{args.model}",
                        key_env_name="ANTHROPIC_API_KEY",
                    ),
                    key_env={"ANTHROPIC_API_KEY": api_key},
                    litellm_spec=LITELLM_SPEC,
                ) as bridge:
                    result = await container.arun(
                        [*with_home, *_claude_argv(claude_bin, args)],
                        env={
                            **env,
                            **bridge.container_env(),
                            "ANTHROPIC_AUTH_TOKEN": bridge.bridge_key,
                            "ANTHROPIC_BASE_URL": bridge.base_url,
                        },
                        timeout=args.timeout,
                    )
                secrets = [bridge.bridge_key, api_key]
                print(redact(result.stderr, secrets), file=sys.stderr, end="")
                print(redact(result.stdout, secrets), end="")
                return result.returncode


def main() -> int:
    return asyncio.run(_amain(get_args()))


if __name__ == "__main__":
    raise SystemExit(main())
