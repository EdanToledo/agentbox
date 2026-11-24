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

"""Resource managers for distributed container execution.

Scheduler-agnostic interface for acquiring and releasing compute resources
(machines/nodes) from a pool, with concrete local-subprocess and Slurm
backends.  The base :class:`ResourceManager` uses the Template Method pattern;
backends override hook methods.
"""

from agentbox.actors.resource_managers.base import (
    await_for_pool,
    HealthCheckResult,
    PoolExhaustedError,
    PoolStatus,
    ResourceHandle,
    ResourceManager,
    wait_for_pool,
)
from agentbox.actors.resource_managers.local import (
    LocalResourceManager,
    LocalWorkerState,
)
from agentbox.actors.resource_managers.slurm import SlurmResourceManager
from agentbox.actors.resource_managers.slurm_array import (
    ArrayTaskInfo,
    SlurmAllocation,
    SlurmArrayChunk,
)


__all__ = [
    "ArrayTaskInfo",
    "HealthCheckResult",
    "LocalResourceManager",
    "LocalWorkerState",
    "PoolExhaustedError",
    "PoolStatus",
    "ResourceHandle",
    "ResourceManager",
    "SlurmAllocation",
    "SlurmArrayChunk",
    "SlurmResourceManager",
    "await_for_pool",
    "wait_for_pool",
]
