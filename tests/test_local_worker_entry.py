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

"""Tests for agentbox.local_worker_entry."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from agentbox import local_worker_entry


class TestMain:
    @patch.dict(
        "os.environ",
        {
            local_worker_entry.LOCAL_WORKER_ADDRESS_ENV: (
                "ipc:///tmp/agentbox-local-worker"
            )
        },
        clear=True,
    )
    @patch("agentbox.local_worker_entry.run_worker")
    def test_runs_worker_and_removes_address_from_environment(self, run_worker):
        local_worker_entry.main()

        run_worker.assert_called_once_with("ipc:///tmp/agentbox-local-worker")
        assert local_worker_entry.LOCAL_WORKER_ADDRESS_ENV not in os.environ

    @patch.dict("os.environ", {}, clear=True)
    def test_requires_address(self):
        with pytest.raises(
            RuntimeError, match=local_worker_entry.LOCAL_WORKER_ADDRESS_ENV
        ):
            local_worker_entry.main()
