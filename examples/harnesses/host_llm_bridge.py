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

"""Worker-host LiteLLM bridge that keeps provider API keys out of containers.

Imported by the Claude, Codex, and Muse container examples (Muse uses only the
proxy and redaction helpers); not meant to be run directly.

Threat model
------------
An agent CLI running in a container has unrestricted shell access there.
Anything in a container process's environment can be recovered by a
model-selected command via ``/proc/<pid>/environ``, even if the command's own
environment is filtered, because the parent agent process is readable under
the same UID in the same PID namespace.

So the real provider key never enters the container. A LiteLLM proxy runs as
a worker-host process (``machine.spawn``), outside the container's PID
namespace (Apptainer ``--containall``), and holds the key. The container only
sees a random per-run bridge credential that the proxy checks as its
``LITELLM_MASTER_KEY`` and that stops working when the proxy exits.

Networking
----------
The container shares the worker's network namespace, so on Linux the agent
reaches the proxy at ``127.0.0.1:<port>``. When the worker runs on macOS,
Apptainer lives in a Lima VM that does not share the host loopback, so the
proxy binds all interfaces and the agent dials ``host.lima.internal``.

The bridge assumes the worker runs the client's Python interpreter, which
holds for ``AgentBoxManager.create_local``.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import textwrap
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from agentbox import Machine, Popen


_LOG_TAIL_LINES = 40
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")

_FIND_FREE_PORT_PY = textwrap.dedent("""\
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        print(sock.getsockname()[1])
    """)

_WAIT_FOR_PORT_PY = textwrap.dedent("""\
    import socket, sys, time
    port, deadline = int(sys.argv[1]), time.monotonic() + float(sys.argv[2])
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            sys.exit(0)
        except OSError:
            time.sleep(0.25)
    sys.exit(f"port {port} did not accept connections in time")
    """)


def proxy_env(*no_proxy: str, loopback_host: str | None = None) -> dict[str, str]:
    """Forward the client's HTTP(S) proxy settings, bypassing *no_proxy* hosts.

    If *loopback_host* is set, loopback proxy URLs are rewritten to it so a
    Lima-hosted container can reach a proxy listening on the macOS host.
    """
    env = {name: os.environ[name] for name in _PROXY_VARS if os.environ.get(name)}
    if loopback_host:
        for name, url in env.items():
            for loopback in ("localhost", "127.0.0.1", "[::1]"):
                url = url.replace(f"//{loopback}", f"//{loopback_host}")
            env[name] = url
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    hosts = [h for h in existing.split(",") if h] + list(no_proxy)
    merged = ",".join(dict.fromkeys(hosts))
    return {**env, "NO_PROXY": merged, "no_proxy": merged}


def litellm_config(
    *,
    model: str,
    upstream_model: str,
    key_env_name: str,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Build a LiteLLM proxy config that exposes one upstream model.

    Args:
        model: Model name the agent requests from the proxy. ``"*"`` matches
            any requested name.
        upstream_model: LiteLLM model route, e.g. ``"anthropic/claude-opus-5"``.
            With ``model="*"``, a route such as ``"openai/*"`` forwards the
            requested name unchanged.
        key_env_name: Environment variable holding the real provider key. It is
            read inside the host-side proxy process only.
        api_base: Upstream endpoint for OpenAI-compatible providers; ``None``
            uses the provider's default.
    """
    params: dict[str, Any] = {
        "model": upstream_model,
        "api_key": f"os.environ/{key_env_name}",
    }
    if api_base is not None:
        params["api_base"] = api_base
    return proxy_config([{"model_name": model, "litellm_params": params}])


def proxy_config(model_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap LiteLLM ``model_list`` entries in a bridge proxy config.

    The proxy accepts only the per-run bridge credential (``LITELLM_MASTER_KEY``).
    """
    return {
        "model_list": model_list,
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


def redact(text: str | bytes | None, secrets_: list[str]) -> str:
    """Replace every occurrence of each secret with ``<redacted>``."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    result = text or ""
    for secret in secrets_:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


@dataclass(frozen=True)
class BridgeEndpoint:
    """How the agent container reaches a running host-side proxy."""

    host: str
    port: int
    bridge_key: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def container_env(self) -> dict[str, str]:
        """Proxy variables that route the bridge endpoint directly.

        When the container dials a non-loopback bridge host (a Lima VM on
        macOS), loopback proxy URLs are rewritten to that host as well, so the
        agent's other outbound traffic still reaches the client's proxy.
        """
        loopback_host = None if self.host == "127.0.0.1" else self.host
        return proxy_env(
            "127.0.0.1", "localhost", "::1", self.host, loopback_host=loopback_host
        )


class HostBridge:
    """Async context manager running a LiteLLM proxy on the worker host.

    On entry it installs ``litellm[proxy]`` into a private worker directory,
    starts the proxy with the real key in the *host* process environment,
    waits for it to accept connections, and returns a :class:`BridgeEndpoint`
    that carries only the per-run bridge credential. On exit it stops the
    proxy (revoking the credential) and removes the directory.

    Args:
        machine: Worker host that also runs the agent container.
        config: LiteLLM proxy configuration (see :func:`litellm_config`).
        key_env: Real provider credential as ``{ENV_NAME: value}``; present only
            in the host-side proxy environment.
        litellm_spec: pip requirement, e.g. ``"litellm[proxy]>=1.63.8,<2"``.
        start_timeout: Seconds to wait for the proxy to accept connections.
    """

    def __init__(
        self,
        machine: Machine,
        *,
        config: dict[str, Any],
        key_env: dict[str, str],
        litellm_spec: str,
        start_timeout: float = 120.0,
    ) -> None:
        self._machine = machine
        self._config = config
        self._key_env = dict(key_env)
        self._litellm_spec = litellm_spec
        self._start_timeout = start_timeout
        self.bridge_key = "bridge-" + secrets.token_urlsafe(32)
        self._python = sys.executable
        self._host_dir: str | None = None
        self._proxy: Popen | None = None

    def _secrets(self) -> list[str]:
        return [*self._key_env.values(), self.bridge_key]

    async def __aenter__(self) -> BridgeEndpoint:
        on_macos = sys.platform == "darwin"
        agent_host = "host.lima.internal" if on_macos else "127.0.0.1"
        bind_address = "0.0.0.0" if on_macos else "127.0.0.1"
        machine = self._machine
        try:
            made = await machine.arun(
                ["mktemp", "-d", "/tmp/agentbox-bridge-XXXXXXXX"], check=True
            )
            self._host_dir = made.stdout.strip()
            packages = f"{self._host_dir}/packages"
            host_env = proxy_env("127.0.0.1", "localhost", agent_host)

            print("Installing LiteLLM on the worker host (outside the container)...")
            install = await machine.arun(
                [self._python, "-m", "pip", "install", "--quiet"]
                + ["--disable-pip-version-check", "--target", packages]
                + [self._litellm_spec],
                env={**host_env, "PIP_CACHE_DIR": f"{self._host_dir}/pip-cache"},
            )
            if install.returncode != 0:
                raise RuntimeError(
                    "Installing LiteLLM on the worker host failed: "
                    + redact(install.stderr or install.stdout, self._secrets())
                )

            config_path = f"{self._host_dir}/litellm.json"
            await machine.arun(
                ["tee", config_path], input=json.dumps(self._config), check=True
            )
            found = await machine.arun([self._python, "-c", _FIND_FREE_PORT_PY])
            port = int(found.stdout.strip())

            self._proxy = machine.spawn(
                [self._python, "-c", "from litellm import run_server; run_server()"]
                + ["--config", config_path, "--host", bind_address]
                + ["--port", str(port)],
                env={
                    **host_env,
                    "PYTHONPATH": packages,
                    **self._key_env,
                    "LITELLM_MASTER_KEY": self.bridge_key,
                },
            )
            ready = await machine.arun(
                [self._python, "-c", _WAIT_FOR_PORT_PY, str(port)]
                + [str(self._start_timeout)],
                timeout=self._start_timeout + 30.0,
            )
            if ready.returncode != 0 or self._proxy.poll() is not None:
                log = await self._stop_proxy()
                raise RuntimeError(
                    f"LiteLLM bridge did not start: {ready.stderr}\n"
                    + redact(log, self._secrets())
                )
            print(f"LiteLLM bridge ready; container dials {agent_host}:{port}")
            return BridgeEndpoint(
                host=agent_host, port=port, bridge_key=self.bridge_key
            )
        except BaseException:
            await self._stop_proxy()
            await self._remove_host_dir()
            raise

    async def _stop_proxy(self) -> str:
        """Stop the proxy (in a thread; Popen is blocking) and return its log."""
        proxy, self._proxy = self._proxy, None
        if proxy is None:
            return ""

        def stop() -> str:
            if proxy.poll() is None:
                proxy.terminate()
                try:
                    proxy.wait(timeout=5)
                except Exception:
                    proxy.kill()
            stdout, stderr = proxy.communicate(timeout=10)
            return redact(stdout, []) + redact(stderr, [])

        # Shield so a cancelled caller cannot leave the proxy running.
        return await asyncio.shield(asyncio.to_thread(stop))

    async def _remove_host_dir(self) -> None:
        host_dir, self._host_dir = self._host_dir, None
        if host_dir:
            await self._machine.arun(["rm", "-rf", host_dir], check=True)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            log = await self._stop_proxy()
            if exc_type is not None and log.strip():
                tail = "\n".join(log.splitlines()[-_LOG_TAIL_LINES:])
                print(
                    "[host-bridge] proxy log:\n" + redact(tail, self._secrets()),
                    file=sys.stderr,
                )
        finally:
            await self._remove_host_dir()
