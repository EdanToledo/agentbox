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

"""AgentBox fuzz CI smoketest.

Runs a short, randomised sequence of shell commands + host→container file
copies against a real container — catches regressions hand-written smoketests
miss.

Two passes:
  1. Deterministic (``seed=42``) — stable regression gate.
  2. Randomised — exploratory; seed is printed so failures reproduce locally.
"""

import hashlib
import os
import random
import string
import sys
import tempfile
from dataclasses import dataclass

from agentbox import AgentBoxManager, ApptainerOptions, Container


if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    __package__ = "examples.ci"

from ._common import resolve_image, run_smoketest


DETERMINISTIC_SEED = 42
DETERMINISTIC_LENGTH = 8
RANDOM_LENGTH = 7
FILE_OP_RATIO = 0.3
MIN_FILE_BYTES = 16
MAX_FILE_BYTES = 4096


@dataclass
class ShellOp:
    command: list[str]
    expect_substr: str


@dataclass
class FileOp:
    local_path: str
    remote_path: str
    expected_size: int
    expected_sha256: str


def _gen_shell_op(rng: random.Random) -> ShellOp:
    kind = rng.choice(["echo", "wc", "seq", "pwd"])
    if kind == "echo":
        token = "".join(rng.choices(string.ascii_letters + string.digits, k=8))
        return ShellOp(["echo", token], expect_substr=token)
    if kind == "wc":
        n = rng.randint(1, 50)
        return ShellOp(
            ["sh", "-c", f"printf 'x%.0s' $(seq 1 {n}) | wc -c"], expect_substr=str(n)
        )
    if kind == "seq":
        n = rng.randint(2, 8)
        return ShellOp(
            ["sh", "-c", f"seq 1 {n} | paste -sd,"],
            expect_substr=",".join(str(i) for i in range(1, n + 1)),
        )
    return ShellOp(["pwd"], expect_substr="/")


def _gen_file_op(rng: random.Random, tmpdir: str) -> FileOp:
    size = rng.randint(MIN_FILE_BYTES, MAX_FILE_BYTES)
    payload = rng.randbytes(size)
    local_path = os.path.join(tmpdir, f"fuzz_{rng.randrange(2**32):08x}.bin")
    with open(local_path, "wb") as f:
        f.write(payload)
    remote_path = f"/tmp/fuzz_{os.path.basename(local_path)}"
    return FileOp(
        local_path=local_path,
        remote_path=remote_path,
        expected_size=size,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _gen_sequence(seed: int, length: int, tmpdir: str) -> list[ShellOp | FileOp]:
    rng = random.Random(seed)
    return [
        _gen_file_op(rng, tmpdir)
        if rng.random() < FILE_OP_RATIO
        else _gen_shell_op(rng)
        for _ in range(length)
    ]


def _run_pass(
    container: Container, ops: list[ShellOp | FileOp], *, label: str, seed: int
) -> None:
    for i, op in enumerate(ops, start=1):
        step = f"[{label} seed={seed}] {i}/{len(ops)}"
        if isinstance(op, ShellOp):
            result = container.run(op.command)
            if result.returncode != 0:
                raise AssertionError(
                    f"{step} {op.command!r} exit={result.returncode} stderr={result.stderr!r}"
                )
            stdout = result.stdout or ""
            if op.expect_substr not in stdout:
                raise AssertionError(
                    f"{step} {op.command!r}: expected {op.expect_substr!r} in stdout, got {result.stdout!r}"
                )
            print(f"  {step}: shell {op.command} OK")
        else:
            container.upload(op.local_path, op.remote_path)
            stat = container.run(
                ["sh", "-c", f"stat -c %s '{op.remote_path}'"],
            )
            remote_size = int((stat.stdout or "").strip())
            if remote_size != op.expected_size:
                raise AssertionError(
                    f"{step} size mismatch: expected {op.expected_size}, got {remote_size}"
                )
            sha = container.run(
                ["sh", "-c", f"sha256sum '{op.remote_path}' | cut -d' ' -f1"],
            )
            remote_sha = (sha.stdout or "").strip()
            if remote_sha != op.expected_sha256:
                raise AssertionError(
                    f"{step} sha mismatch: expected {op.expected_sha256}, got {remote_sha}"
                )
            print(f"  {step}: file {op.remote_path} ({op.expected_size} B) OK")


def main(local: bool = False) -> None:
    image = resolve_image(local=local)
    print(f"Starting AgentBox fuzz smoketest (image={image})...")

    options = ApptainerOptions(writable_tmpfs=True, no_home=True)
    random_seed = random.randint(0, 2**32 - 1)
    print(f"Deterministic pass: seed={DETERMINISTIC_SEED}, {DETERMINISTIC_LENGTH} ops")
    print(f"Random pass:        seed={random_seed}, {RANDOM_LENGTH} ops")

    with tempfile.TemporaryDirectory(prefix="agentbox-fuzz-") as tmpdir:
        det_ops = _gen_sequence(DETERMINISTIC_SEED, DETERMINISTIC_LENGTH, tmpdir)
        rand_ops = _gen_sequence(random_seed, RANDOM_LENGTH, tmpdir)

        with AgentBoxManager.create_local(num_workers=1) as manager:
            with manager.acquire_machine() as machine:
                with Container(machine, image=image, options=options) as container:
                    print("\n--- Deterministic pass ---")
                    _run_pass(
                        container,
                        det_ops,
                        label="deterministic",
                        seed=DETERMINISTIC_SEED,
                    )
                    print("\n--- Random pass ---")
                    _run_pass(container, rand_ops, label="random", seed=random_seed)

    print("\nAll AgentBox fuzz smoketest checks passed.")


if __name__ == "__main__":
    run_smoketest("AgentBox fuzz smoketest", main)
