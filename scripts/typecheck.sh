#!/usr/bin/env bash
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

# Type-check the package and the examples that CI or users run directly.

set -euo pipefail

cd "$(dirname "$0")/.."

# Example directories are on the path so scripts can import their sibling
# helpers.
MYPYPATH="src:examples:examples/harnesses" mypy \
  src/agentbox \
  examples/container_demo.py \
  examples/machine_demo.py \
  examples/tty_terminal.py \
  examples/ci/ci_basic_smoketest.py \
  examples/ci/ci_fuzz_smoketest.py \
  examples/harnesses/claude_container.py \
  examples/harnesses/codex_container.py \
  examples/harnesses/host_llm_bridge.py \
  examples/harnesses/muse_container.py \
  tests/typecheck/target.py
