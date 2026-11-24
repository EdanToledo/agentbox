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

"""Public protocols for describing, scheduling, and observing tasks."""

from __future__ import annotations

from typing import Any, Protocol


class Task[ResultT](Protocol):
    """Serializable description of work accepted by a :class:`TaskPool`.

    ``ResultT`` is the result returned by the task's registered handler. Task
    objects stay on the client; only their built-in payload crosses the
    Monarch transport boundary.
    """

    @property
    def kind(self) -> str:
        """Handler key used by the worker-side scheduler."""
        ...

    def to_payload(self) -> tuple[Any, ...]:
        """Encode handler arguments as a transport-safe tuple."""
        ...


class TaskHandle[ResultT](Protocol):
    """Observation and cancellation interface for one accepted task."""

    @property
    def task_id(self) -> str:
        """Stable identifier assigned when the task is scheduled."""
        ...

    def wait_started(self, *, timeout: float | None = None) -> bool:
        """Wait for start disposition; return false if cancelled while queued."""
        ...

    async def await_started(self, *, timeout: float | None = None) -> bool:
        """Async-native counterpart of :meth:`wait_started`."""
        ...

    def cancel(self, *, timeout: float | None = None) -> bool:
        """Request cancellation and report whether the scheduler found the task."""
        ...

    async def acancel(self, *, timeout: float | None = None) -> bool:
        """Async-native counterpart of :meth:`cancel`."""
        ...

    def result(self, *, timeout: float | None = None) -> ResultT:
        """Wait synchronously for the terminal task result."""
        ...

    async def aresult(self, *, timeout: float | None = None) -> ResultT:
        """Wait asynchronously for the terminal task result."""
        ...


class TaskPool(Protocol):
    """Structural interface used by components that share background work."""

    def schedule[ResultT](self, task: Task[ResultT]) -> TaskHandle[ResultT]:
        """Schedule a task and return after the worker accepts it."""
        ...

    async def aschedule[ResultT](self, task: Task[ResultT]) -> TaskHandle[ResultT]:
        """Async-native counterpart of :meth:`schedule`."""
        ...

    def submit[ResultT](self, task: Task[ResultT]) -> ResultT:
        """Synchronously schedule one task and return its result."""
        ...

    async def asubmit[ResultT](self, task: Task[ResultT]) -> ResultT:
        """Asynchronously schedule one task and return its result."""
        ...

    def close(self, *, timeout: float = 10.0, drain: bool = True) -> None:
        """Tear down the worker-side service.

        With *drain*, running tasks finish first; otherwise they are cancelled
        and their waiters fail.
        """
        ...

    async def aclose(self, *, timeout: float = 10.0, drain: bool = True) -> None:
        """Async-native counterpart of :meth:`close`."""
        ...
