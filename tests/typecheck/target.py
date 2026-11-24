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

"""Static checks for the ExecutionTarget overloads (run with mypy, not pytest)."""

from typing import assert_type

from agentbox.actors.target import ExecutionTarget
from agentbox.remote.pty import PtyProcess
from agentbox.remote.subprocess import CompletedProcess, Popen


async def check_execution_target(target: ExecutionTarget) -> None:
    assert_type(target.run("true"), CompletedProcess[str])
    assert_type(target.run(["true"], text=False), CompletedProcess[bytes])
    assert_type(await target.arun("true"), CompletedProcess[str])
    assert_type(await target.arun(["true"], text=False), CompletedProcess[bytes])
    assert_type(target.spawn("cat"), Popen)
    assert_type(target.spawn(["bash"], pty=True), PtyProcess)
