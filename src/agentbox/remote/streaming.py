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

"""Streaming message protocol shared by subprocess and PTY remote execution.

Defines the message types exchanged between remote execution actors and
their controllers over Monarch channels.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import auto, Enum
from typing import Union

import cloudpickle
from monarch.actor import Port


class StreamType(Enum):
    """Identifies which standard stream a data chunk belongs to."""

    STDOUT = auto()
    STDERR = auto()


@dataclass(frozen=True, slots=True)
class StreamData:
    """A chunk of output data from stdout or stderr."""

    stream: StreamType
    data: bytes


@dataclass(frozen=True, slots=True)
class StreamExit:
    """Signals process termination with an exit code. Always the last message."""

    returncode: int


@dataclass(frozen=True, slots=True)
class StreamError:
    """Signals an infrastructure error (not a process error)."""

    error: str


@dataclass(frozen=True, slots=True)
class StdinData:
    """A chunk of data to write to the remote process's stdin."""

    data: bytes


@dataclass(frozen=True, slots=True)
class StdinClose:
    """Signal to close the remote process's stdin."""


@dataclass(frozen=True, slots=True)
class StreamStdinReady:
    """Sent by the actor to provide the stdin port back to the controller."""

    stdin_port: Port[Union[StdinData, StdinClose]]


@dataclass(frozen=True, slots=True)
class ControlSignal:
    """Signal to send to the remote process (e.g., SIGTERM, SIGKILL)."""

    signal_num: int


@dataclass(frozen=True, slots=True)
class StreamControlReady:
    """Sent by the actor to provide the control port back to the controller."""

    control_port: Port[ControlSignal]


def encode(message: StreamData | StdinData | object) -> object:
    """Return the wire form of *message* for ``Port.send``.

    This module is pickled by value, so a dataclass instance would carry its
    class (and ``StreamType``, an Enum rebuilt on every unpickle) in each
    message. The per-chunk messages therefore travel as builtin tuples; rare
    control messages are sent unchanged.
    """
    if type(message) is StreamData:
        return ("D", message.stream.value, message.data)
    if type(message) is StdinData:
        return ("I", message.data)
    return message


def decode(wire: object) -> object:
    """Inverse of :func:`encode`; non-tuple messages are returned unchanged."""
    if isinstance(wire, tuple) and len(wire) == 3 and wire[0] == "D":
        return StreamData(stream=StreamType(wire[1]), data=wire[2])
    if isinstance(wire, tuple) and len(wire) == 2 and wire[0] == "I":
        return StdinData(data=wire[1])
    return wire


# Type aliases for clarity
StreamMessage = Union[
    StreamData, StreamExit, StreamError, StreamStdinReady, StreamControlReady
]
StdinMessage = Union[StdinData, StdinClose]

# Register so cloudpickle serializes streaming types by value to remote workers.
# Without this, containers that don't have agentbox installed will fail with
# ModuleNotFoundError when unpickling actors that reference these types.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
