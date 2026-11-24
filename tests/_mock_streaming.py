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

"""Shared test doubles for Monarch's Future / Port / PortReceiver.

These model the *blocking-recv* semantics used by the pipe-backed subprocess
(:mod:`agentbox.remote.subprocess`) and PTY (:mod:`agentbox.remote.pty`) impls,
where the drain thread calls ``recv().get()`` and is content to block until a
message arrives.

Note: ``test_streaming_subprocess.py`` deliberately keeps its *own* variants
with different semantics — there ``recv()`` must return immediately and the
timeout lives in ``.get(timeout)`` so the streaming poll loop can fall through
and re-check ``proc.poll()`` instead of deadlocking.  Do not unify the two.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

from agentbox.remote import streaming


class MockFuture:
    """Mimics a Monarch Future with a blocking ``.get()``."""

    def __init__(self, value: Any):
        self._value = value

    def get(self, timeout: float | None = None) -> Any:
        return self._value


class MockPortReceiver:
    """Mimics PortReceiver — ``recv()`` blocks until a message is queued."""

    def __init__(self) -> None:
        self._messages: deque[Any] = deque()
        self._event = threading.Event()

    def push(self, msg: Any) -> None:
        self._messages.append(msg)
        self._event.set()

    def recv(self) -> MockFuture:
        while not self._messages:
            self._event.wait()
            self._event.clear()
        return MockFuture(self._messages.popleft())


class MockPort:
    """Mimics Port — ``send()``/``exception()`` append to a list.

    ``sent`` holds what a receiver decodes; ``wire`` the raw sent objects.
    """

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.wire: list[Any] = []

    def send(self, msg: Any) -> None:
        self.wire.append(msg)
        self.sent.append(streaming.decode(msg))

    def exception(self, exc: Exception) -> None:
        self.sent.append(exc)
