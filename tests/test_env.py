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

"""Tests for agentbox.env."""

from __future__ import annotations

import logging
import os
import sys
import tempfile

from agentbox.env import (
    ensure_short_tmpdir,
    restore_slurm_env,
    scrub_slurm_env,
    scrubbed_slurm_env,
)


# ── Slurm tests ──────────────────────────────────────────────────────


def test_scrub_removes_slurm_vars(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_TMPDIR", "/tmp/slurm")

    saved = scrub_slurm_env()

    assert "SLURM_JOB_ID" not in os.environ
    assert "SLURM_TMPDIR" not in os.environ
    assert saved["SLURM_JOB_ID"] == "12345"
    assert saved["SLURM_TMPDIR"] == "/tmp/slurm"


def test_scrub_keeps_cluster_name(monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mycluster")

    scrub_slurm_env()

    assert os.environ.get("SLURM_CLUSTER_NAME") == "mycluster"


def test_scrub_noop_when_no_slurm_vars(monkeypatch):
    # Remove any Slurm vars that might be set (monkeypatch restores on teardown).
    for k in [k for k in os.environ if k.startswith("SLURM_")]:
        monkeypatch.delenv(k)

    saved = scrub_slurm_env()

    assert saved == {}


def test_restore_puts_vars_back(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "99")

    saved = scrub_slurm_env()
    assert "SLURM_JOB_ID" not in os.environ

    restore_slurm_env(saved)
    assert os.environ["SLURM_JOB_ID"] == "99"


def test_scrub_then_restore_round_trips(monkeypatch):
    """scrub + restore leaves os.environ exactly as it was — no leak across calls."""
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_TMPDIR", "/tmp/s")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "c")
    before = dict(os.environ)

    restore_slurm_env(scrub_slurm_env())

    assert os.environ == before


def test_scrubbed_slurm_env_restores_on_exit(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "5")

    with scrubbed_slurm_env() as saved:
        assert "SLURM_JOB_ID" not in os.environ
        assert saved["SLURM_JOB_ID"] == "5"

    assert os.environ["SLURM_JOB_ID"] == "5"


# ── ensure_short_tmpdir tests ────────────────────────────────────────


def test_ensure_short_tmpdir_warns_when_target_too_long(monkeypatch, tmp_path, caplog):
    # Exercise the macOS-only branch without requiring a macOS test worker.
    monkeypatch.setattr(sys, "platform", "darwin")
    # An AGENTBOX_TMPDIR override longer than the socket-safe limit must still be
    # honored, but should warn so the eventual SUN_LEN error is diagnosable.
    long_dir = tmp_path / ("d" * 60) / "tmp"  # well over the 40-byte limit
    monkeypatch.setenv("AGENTBOX_TMPDIR", str(long_dir))
    monkeypatch.delenv("TMPDIR", raising=False)
    # Snapshot tempfile.tempdir so monkeypatch restores it after the call mutates it.
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)

    with caplog.at_level(logging.WARNING, logger="agentbox.env"):
        result = ensure_short_tmpdir()

    assert result == str(long_dir)
    assert os.environ["TMPDIR"] == str(long_dir)
    assert any("SUN_LEN" in r.getMessage() for r in caplog.records)


class TestBoundNativeHeap:
    def test_pins_glibc_mmap_threshold(self, monkeypatch):
        import ctypes

        from agentbox import env

        calls = []

        class FakeLibc:
            def mallopt(self, param, value):
                calls.append((param, value))
                return 1

        monkeypatch.delenv("MALLOC_MMAP_THRESHOLD_", raising=False)
        monkeypatch.setattr(env.sys, "platform", "linux")
        monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeLibc())
        assert env.bound_native_heap()
        # M_MMAP_THRESHOLD; a fixed value also disables glibc's dynamic raising.
        assert calls == [(-3, 1 << 20)]

    def test_respects_user_tunable_and_non_linux(self, monkeypatch):
        from agentbox import env

        monkeypatch.setenv("MALLOC_MMAP_THRESHOLD_", "65536")
        assert not env.bound_native_heap()
        monkeypatch.delenv("MALLOC_MMAP_THRESHOLD_")
        monkeypatch.setattr(env.sys, "platform", "darwin")
        assert not env.bound_native_heap()
