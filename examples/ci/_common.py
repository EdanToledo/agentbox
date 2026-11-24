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

"""Shared utilities for AgentBox CI smoketests."""

import argparse
import os
import sys
from typing import Callable


CI_SIF_PATH = "/usr/share/sif/agentbox.ci.sif"


def resolve_image(*, local: bool) -> str:
    """Pick the container image for a smoketest.

    Resolution order:
      1. ``AGENTBOX_CI_IMAGE`` env var (always wins; lets devs override).
      2. ``CI_SIF_PATH`` (the SIF built by ``.github/actions/build_sif``).
      3. If ``local`` is true, fall back to ``docker://python:3.12-slim``.

    Raises:
        FileNotFoundError: In CI mode (not ``--local``) when the SIF is missing.
    """
    env_image = os.environ.get("AGENTBOX_CI_IMAGE")
    if env_image:
        return env_image
    if os.path.exists(CI_SIF_PATH):
        return CI_SIF_PATH
    if local:
        return "docker://python:3.12-slim"
    raise FileNotFoundError(
        f"CI SIF image not found at {CI_SIF_PATH}. "
        "Set AGENTBOX_CI_IMAGE or pass --local to use docker://python:3.12-slim."
    )


def run_smoketest(description: str, main_fn: Callable[..., None]) -> None:
    """Parse ``--local`` and dispatch to ``main_fn(local=...)``.

    Always exits non-zero on any exception so CI surfaces the failure.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Fall back to docker://python:3.12-slim when the CI SIF is absent.",
    )
    args = parser.parse_args()
    try:
        main_fn(local=args.local)
    except Exception as exc:
        print(f"SMOKETEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
