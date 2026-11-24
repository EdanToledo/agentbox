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

"""AgentBox: shells, files, and containers on Monarch workers."""

from importlib.metadata import PackageNotFoundError, version

# Public names are imported from their owning modules so each symbol has a
# single home.  All imports must follow ensure_short_tmpdir() above (noqa E402).
from agentbox.actors import (
    AgentBoxManager,  # noqa: E402
    ApptainerNotFoundError,
    ApptainerOptions,
    await_for_pool,
    BindMount,
    Container,
    ExecutionTarget,
    HealthCheckResult,
    LocalResourceManager,
    Machine,
    PoolExhaustedError,
    PoolStatus,
    ResourceHandle,
    ResourceManager,
    SlurmAllocation,
    SlurmResourceManager,
    wait_for_pool,
    WorkerConnectionError,
)

# macOS: point TMPDIR at a short, $HOME-rooted dir before Monarch loads so its
# Unix-domain sockets stay under the SUN_LEN limit and temp dirs are visible
# inside the Lima VM.  No-op on Linux.  Must run before importing
# agentbox.actors (which imports monarch).
from agentbox.env import bound_native_heap, ensure_short_tmpdir
from agentbox.remote.file_transfer import FileCopyError  # noqa: E402
from agentbox.remote.pty import PtyProcess  # noqa: E402
from agentbox.remote.subprocess import (  # noqa: E402
    CalledProcessError,
    CompletedProcess,
    Popen,
    TimeoutExpired,
)


ensure_short_tmpdir()
bound_native_heap()


def _get_version() -> str:
    try:
        return version("agentbox")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _get_version()

__all__ = [
    "__version__",
    "AgentBoxManager",
    "ApptainerNotFoundError",
    "ApptainerOptions",
    "BindMount",
    "CalledProcessError",
    "CompletedProcess",
    "Container",
    "ExecutionTarget",
    "FileCopyError",
    "HealthCheckResult",
    "LocalResourceManager",
    "Machine",
    "PoolExhaustedError",
    "PoolStatus",
    "Popen",
    "PtyProcess",
    "ResourceHandle",
    "ResourceManager",
    "SlurmAllocation",
    "SlurmResourceManager",
    "TimeoutExpired",
    "WorkerConnectionError",
    "await_for_pool",
    "wait_for_pool",
]
