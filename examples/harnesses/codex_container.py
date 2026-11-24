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

r"""Install and optionally run the open-source Codex CLI inside an AgentBox container.

Codex is installed with OpenAI's native installer into the container's staging
directory (a host directory bind-mounted into the container and removed when it
closes), so nothing is written to the host home directory and large installer
archives stay out of ``--containall``'s small ``/tmp``.

The real ``OPENAI_API_KEY`` never enters the container. When ``--prompt`` is
given, a LiteLLM proxy (``host_llm_bridge.HostBridge``) runs as a worker-host
process outside the container's PID namespace and holds the key; Codex only
sees a random per-run bridge credential that stops working when the run ends,
and model-selected shell commands do not inherit even that.

Usage:
    # Install Codex and print its version (no API key required):
    python examples/harnesses/codex_container.py

    # Run a headless prompt with Codex's default model:
    OPENAI_API_KEY=... python examples/harnesses/codex_container.py \
        --prompt "Reply with exactly: Hello from Codex"

    # Pin a Codex release and choose the model:
    OPENAI_API_KEY=... python examples/harnesses/codex_container.py \
        --codex-version 0.101.0 --model <openai-model-id> --prompt "..."

Prerequisites:
    - Apptainer on PATH (on macOS, inside the Lima VM)
    - ``pip install -e .``
    - ``curl``, ``tar``, and a SHA-256 tool inside the image (the default
      image has all three)
    - ``OPENAI_API_KEY`` when ``--prompt`` is used

Success: the script prints ``Installed: <version>`` and, with ``--prompt``,
Codex's reply; it exits with Codex's exit status.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentbox import AgentBoxManager, ApptainerOptions, Container
from host_llm_bridge import HostBridge, litellm_config, proxy_env, redact


DEFAULT_IMAGE = "docker://python:3.12-bookworm"
INSTALLER_URL = "https://chatgpt.com/codex/install.sh"
# 1.63.8 is the first LiteLLM release whose proxy serves the Responses API.
LITELLM_SPEC = "litellm[proxy]>=1.63.8,<2"
BRIDGE_KEY_ENV = "AGENTBOX_BRIDGE_KEY"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="SIF path or URI")
    parser.add_argument(
        "--codex-version",
        default="latest",
        help="Codex release to install (default: latest)",
    )
    parser.add_argument("--prompt", help="Run this prompt with `codex exec`")
    parser.add_argument(
        "--model",
        help="OpenAI model id (default: Codex's default model)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def _codex_argv(codex_bin: str, base_url: str, args: argparse.Namespace) -> list[str]:
    argv = [
        codex_bin,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        # The container is the sandbox boundary; Codex's bubblewrap sandbox
        # does not nest inside Apptainer.
        "--sandbox",
        "danger-full-access",
        "--config",
        'model_provider="agentbox_bridge"',
        "--config",
        'model_providers.agentbox_bridge.name="OpenAI via AgentBox host bridge"',
        "--config",
        f'model_providers.agentbox_bridge.base_url="{base_url}/v1"',
        "--config",
        'model_providers.agentbox_bridge.wire_api="responses"',
        "--config",
        f'model_providers.agentbox_bridge.env_key="{BRIDGE_KEY_ENV}"',
        # Keep the bridge credential out of model-selected shell commands.
        "--config",
        f'shell_environment_policy.exclude=["{BRIDGE_KEY_ENV}"]',
    ]
    if args.model:
        argv += ["--model", args.model]
    return [*argv, "--", args.prompt]


async def _amain(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if args.prompt is not None and not api_key:
        raise SystemExit("OPENAI_API_KEY must be set when --prompt is used")

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
                install_dir = os.path.join(container.staging_dir, "codex-bin")
                codex_bin = os.path.join(install_dir, "codex")
                env = {
                    "CODEX_HOME": os.path.join(container.staging_dir, "codex-home"),
                    "PATH": f"{install_dir}:/usr/local/bin:/usr/bin:/bin",
                    "TMPDIR": container.staging_dir,
                }

                print(f"Installing Codex ({args.codex_version})...")
                install = await container.arun(
                    [
                        "sh",
                        "-c",
                        f'mkdir -p "$CODEX_HOME" && curl -fsSL {INSTALLER_URL} | sh',
                    ],
                    env={
                        **env,
                        "CODEX_INSTALL_DIR": install_dir,
                        "CODEX_NON_INTERACTIVE": "true",
                        "CODEX_RELEASE": args.codex_version,
                    },
                )
                if install.returncode != 0:
                    print(install.stderr or install.stdout, file=sys.stderr)
                    return install.returncode

                version = await container.arun(
                    [codex_bin, "--version"], env=env, check=True
                )
                print(f"Installed: {version.stdout.strip()}")
                if args.prompt is None:
                    return 0

                assert api_key is not None
                print(f"Running Codex with {args.model or 'its default model'}...")
                async with HostBridge(
                    machine,
                    # Forward whichever model Codex requests to OpenAI.
                    config=litellm_config(
                        model="*",
                        upstream_model="openai/*",
                        key_env_name="OPENAI_API_KEY",
                    ),
                    key_env={"OPENAI_API_KEY": api_key},
                    litellm_spec=LITELLM_SPEC,
                ) as bridge:
                    result = await container.arun(
                        _codex_argv(codex_bin, bridge.base_url, args),
                        env={
                            **env,
                            **bridge.container_env(),
                            BRIDGE_KEY_ENV: bridge.bridge_key,
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
