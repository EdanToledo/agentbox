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

"""Resource manager actors and machine abstractions for distributed execution."""

from agentbox.actors.connector import (
    connect,
    Connector,
    ConnectResult,
    DefaultConnector,
    WorkerConnectionError,
)
from agentbox.actors.container import (
    ApptainerNotFoundError,
    ApptainerOptions,
    BindMount,
    Container,
)
from agentbox.actors.machine import Machine
from agentbox.actors.manager import AgentBoxManager
from agentbox.actors.resource_managers import (
    await_for_pool,
    HealthCheckResult,
    LocalResourceManager,
    PoolExhaustedError,
    PoolStatus,
    ResourceHandle,
    ResourceManager,
    SlurmAllocation,
    SlurmResourceManager,
    wait_for_pool,
)
from agentbox.actors.target import ExecutionTarget


__all__ = [
    "AgentBoxManager",
    "ApptainerNotFoundError",
    "ApptainerOptions",
    "BindMount",
    "ConnectResult",
    "Connector",
    "Container",
    "DefaultConnector",
    "ExecutionTarget",
    "HealthCheckResult",
    "LocalResourceManager",
    "Machine",
    "PoolExhaustedError",
    "PoolStatus",
    "ResourceHandle",
    "ResourceManager",
    "SlurmAllocation",
    "SlurmResourceManager",
    "WorkerConnectionError",
    "await_for_pool",
    "connect",
    "wait_for_pool",
]
