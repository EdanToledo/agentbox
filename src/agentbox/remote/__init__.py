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

"""Low-level process primitives that execute on Monarch workers.

Most callers use :meth:`ExecutionTarget.run` / :meth:`ExecutionTarget.spawn`
on a :class:`~agentbox.Machine` or :class:`~agentbox.Container`
instead of these modules directly.
"""

from agentbox.remote.pty import PtyProcess
from agentbox.remote.subprocess import (
    CalledProcessError,
    CompletedProcess,
    Popen,
    TimeoutExpired,
)


__all__ = [
    "CalledProcessError",
    "CompletedProcess",
    "Popen",
    "PtyProcess",
    "TimeoutExpired",
]
