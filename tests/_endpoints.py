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

"""Call Monarch ``@endpoint`` methods directly on local actor instances.

Unit tests exercise actor logic without spawning a mesh, which means unwrapping
the endpoint descriptor. That relies on Monarch's private ``_method`` attribute,
so it lives here alone: a Monarch change breaks one helper, not every test.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable


def endpoint_method(cls: type, name: str) -> Callable[..., Any]:
    """Return the undecorated function behind endpoint *name* on *cls*."""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]._method
    raise AttributeError(f"{cls.__name__} has no endpoint {name!r}")


def call_endpoint(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call endpoint *name* on *obj*; return its result as-is (sync or awaitable)."""
    return endpoint_method(type(obj), name)(obj, *args, **kwargs)


def run_endpoint(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call endpoint *name* on *obj* from sync code, running it if it is async."""
    result = call_endpoint(obj, name, *args, **kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result
