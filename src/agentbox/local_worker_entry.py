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

"""Entry point for local AgentBox worker subprocesses."""

from __future__ import annotations

import os

from agentbox.process import LOCAL_WORKER_ADDRESS_ENV
from agentbox.worker_bootstrap import run_worker


def main() -> None:
    """Run the local worker configured by the parent process."""

    try:
        address = os.environ.pop(LOCAL_WORKER_ADDRESS_ENV)
    except KeyError:
        raise RuntimeError(f"{LOCAL_WORKER_ADDRESS_ENV} is required") from None
    run_worker(address)


if __name__ == "__main__":
    main()
