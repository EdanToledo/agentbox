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

"""Tests for coordinator transport address selection."""

from __future__ import annotations

from ipaddress import IPv4Address
from unittest.mock import MagicMock, patch

import pytest
from agentbox.actors.manager import AgentBoxManager


class _CapturingManager(AgentBoxManager):
    """Avoid the real client-side manager wiring in factory tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        ("create_slurm", {}),
    ],
)
def test_slurm_factories_use_routable_ipv4(factory: str, kwargs: dict) -> None:
    host = MagicMock()
    with (
        patch("agentbox.actors.manager.this_host", return_value=host),
        patch("agentbox.actors.manager.enable_transport") as enable_transport,
        patch(
            "agentbox.port_server.routable_ipv4",
            return_value=IPv4Address("10.20.30.40"),
        ) as route,
    ):
        getattr(_CapturingManager, factory)(**kwargs)

    route.assert_called_once_with()
    enable_transport.assert_called_once_with("tcp://10.20.30.40:0")
