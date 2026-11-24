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

"""Tests for AgentBoxManager's signed scaling facade."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agentbox.actors.manager import AgentBoxManager


def _manager() -> AgentBoxManager:
    manager = AgentBoxManager.__new__(AgentBoxManager)
    manager._manager = MagicMock()
    return manager


@pytest.mark.parametrize("n", [2, -2])
def test_scale_by(n: int) -> None:
    manager = _manager()

    result = manager.scale_by(n)
    assert result is None

    manager._manager.scale_by.choose.assert_called_once_with(n)
    manager._manager.scale_by.choose.return_value.get.assert_called_once_with()


def test_scale_by_zero_raises() -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="n must be non-zero"):
        manager.scale_by(0)

    manager._manager.scale_by.choose.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [2, -2])
async def test_ascale_by(n: int) -> None:
    manager = _manager()
    choose = AsyncMock()
    transaction = object()
    choose.return_value = transaction
    manager._manager.scale_by.choose = choose

    result = await manager.ascale_by(n)
    assert result is None

    choose.assert_awaited_once_with(n)
