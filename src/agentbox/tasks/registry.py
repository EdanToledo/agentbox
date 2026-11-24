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

"""Task-handler registration for worker-local task pools."""

from __future__ import annotations

import threading
from typing import Any, Callable, Coroutine


type TaskHandler = Callable[..., Coroutine[Any, Any, Any]]


class TaskHandlerRegistry:
    """Thread-safe registry of worker-side handlers keyed by task kind.

    Registries are mutable configuration objects. A task pool consumes a
    snapshot, so later registrations affect only pools created afterward.
    """

    def __init__(self, base: TaskHandlerRegistry | None = None) -> None:
        self._handlers = {} if base is None else base.snapshot()
        self._lock = threading.RLock()

    def register(self, kind: str, handler: TaskHandler) -> None:
        """Register *handler* for *kind*, rejecting ambiguous replacements."""
        if not kind:
            raise ValueError("task kind must not be empty")
        if not callable(handler):
            raise TypeError("task handler must be callable")
        with self._lock:
            existing = self._handlers.get(kind)
            if existing is not None and existing is not handler:
                raise ValueError(f"task handler already registered for kind: {kind}")
            self._handlers[kind] = handler

    def snapshot(self) -> dict[str, TaskHandler]:
        """Return an isolated handler mapping for one task-pool instance."""
        with self._lock:
            return self._handlers.copy()


default_task_handlers = TaskHandlerRegistry()
