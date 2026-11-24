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

"""Tests for the agentbox mesh schema, classification, and registry (mesh_schema)."""

from __future__ import annotations

import pytest
from agentbox.actors.mesh_schema import (
    AGENTBOX_PREFIX,
    base_name,
    FAULT_POLICY,
    fault_scope,
    FaultAction,
    is_internal_droppable,
    make_name,
    make_scoped_name,
    MeshRegistry,
    MeshType,
    parse_name,
    strip_fault_scope,
)


def test_make_name_with_and_without_ident():
    assert make_name("coordinator") == f"{AGENTBOX_PREFIX}coordinator"
    assert make_name("worker", "h1") == f"{AGENTBOX_PREFIX}worker_h1"


def test_make_name_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_name("nope", "x")


def test_make_name_sanitizes_dash_in_ident():
    # "-" is Monarch's fault-suffix delimiter and must not appear in an ident.
    assert make_name("worker", "a-b") == f"{AGENTBOX_PREFIX}worker_a_b"


def test_manager_scoped_name_roundtrip():
    scope = "mgr0123456789abcdef"
    name = make_scoped_name("heartbeat", scope, "local_0")

    assert name == "agentbox_heartbeat_mgr0123456789abcdef__local_0"
    assert fault_scope(f"{name}-suffix") == scope
    parsed = parse_name(name)
    assert parsed is not None
    assert strip_fault_scope(parsed.ident, scope) == "local_0"


def test_fault_scope_rejects_malformed_scope():
    with pytest.raises(ValueError, match="invalid manager fault scope"):
        make_scoped_name("worker", "not-a-scope", "handle")
    assert fault_scope(make_name("worker", "legacy_handle")) is None


def test_base_name_strips_monarch_suffix():
    assert base_name("agentbox_worker_h1-AbCdEf") == "agentbox_worker_h1"
    assert base_name("agentbox_coordinator") == "agentbox_coordinator"


def test_parse_name_roundtrip_ident_with_underscores():
    # Heartbeat idents are slot keys like "slurm_99_0"; they must survive the round trip.
    name = make_name("heartbeat", "slurm_99_0")
    parsed = parse_name(f"{name}-RaNd123")
    assert parsed is not None
    assert parsed.kind == "heartbeat"
    assert parsed.ident == "slurm_99_0"
    assert parsed.mesh_type is MeshType.HEARTBEAT


def test_parse_name_worker_carries_handle():
    parsed = parse_name(f"{make_name('worker', 'handle-7')}-sfx")
    assert parsed is not None
    assert parsed.mesh_type is MeshType.WORKER
    assert parsed.ident == "handle_7"  # dash sanitized at make_name time


def test_parse_name_coordinator():
    parsed = parse_name(f"{make_name('coordinator')}-sfx")
    assert parsed is not None
    assert parsed.mesh_type is MeshType.COORDINATOR
    assert parsed.ident == ""


def test_parse_name_unknown_agentbox_kind_is_auxiliary():
    parsed = parse_name("agentbox_somethingnew_42-sfx")
    assert parsed is not None
    assert parsed.mesh_type is MeshType.AUXILIARY


def test_parse_name_aux_kinds():
    for kind in (
        "subprocess",
        "taskpool",
        "streaming",
        "pty",
        "python",
        "container",
        "imagebuilder",
        "filesource",
        "filearchive",
        "admin",
    ):
        parsed = parse_name(f"{make_name(kind, '1')}-sfx")
        assert parsed is not None and parsed.mesh_type is MeshType.AUXILIARY


def test_parse_name_foreign_returns_none():
    assert parse_name("logger-xAaF2vgKqqU") is None
    assert parse_name("something-else") is None
    assert parse_name(None) is None
    assert parse_name("") is None


def test_internal_droppable_allowlist():
    assert is_internal_droppable("log_forwarder-r4u2gLYUUeW")
    assert is_internal_droppable("logger-xAaF2vgKqqU")
    assert not is_internal_droppable("agentbox_worker_h1-sfx")
    assert not is_internal_droppable("random-mesh")
    assert not is_internal_droppable(None)


def test_fault_policy_covers_every_mesh_type():
    # Guarantees the hook's dispatch is total: no MeshType is unhandled.
    for mesh_type in MeshType:
        assert mesh_type in FAULT_POLICY
        assert isinstance(FAULT_POLICY[mesh_type], FaultAction)


def test_registry_register_and_lookup_by_suffixed_name():
    reg = MeshRegistry()
    name = make_name("worker", "h1")
    reg.register(name, MeshType.WORKER, ident="h1", handle_id="h1")

    record = reg.get(f"{name}-monarchsuffix")  # fault carries a suffix
    assert record is not None
    assert record.mesh_type is MeshType.WORKER
    assert record.metadata["handle_id"] == "h1"
    assert record.live is True


def test_registry_mark_failed_sets_flag():
    reg = MeshRegistry()
    name = make_name("worker", "h1")
    reg.register(name, MeshType.WORKER, ident="h1")
    assert reg.get(name).failed is False
    reg.mark_failed(f"{name}-sfx")
    assert reg.get(name).failed is True


def test_registry_mark_failed_unknown_is_noop():
    reg = MeshRegistry()
    reg.mark_failed("agentbox_worker_nope-sfx")  # must not raise


def test_registry_mark_released_clears_live_but_keeps_record():
    reg = MeshRegistry()
    name = make_name("worker", "h1")
    reg.register(name, MeshType.WORKER, ident="h1", handle_id="h1")
    reg.mark_released(name)

    record = reg.get(f"{name}-sfx")  # record persists (with metadata) after release
    assert record is not None
    assert record.live is False
    assert record.metadata["handle_id"] == "h1"


def test_registry_live_by_default():
    reg = MeshRegistry()
    name = make_name("worker", "h1")
    reg.register(name, MeshType.WORKER, ident="h1")
    assert reg.get(f"{name}-sfx").live is True
    assert [r.name for r in reg.live()] == [name]


def test_registry_mark_released_unknown_is_noop():
    reg = MeshRegistry()
    reg.mark_released(make_name("worker", "never"))  # must not raise


def test_registry_not_live_records_bounded_by_cap():
    reg = MeshRegistry(released_cap=4)
    for i in range(10):
        name = make_name("worker", str(i))
        reg.register(name, MeshType.WORKER, ident=str(i))
        reg.mark_released(name)
    # Live records are never evicted; only the not-live overflow is dropped.
    assert len(reg._meshes) == 4
    assert make_name("worker", "5") not in reg._meshes  # oldest not-live evicted
    assert make_name("worker", "9") in reg._meshes  # newest retained


def test_registry_not_live_records_expire_by_ttl():
    reg = MeshRegistry(released_ttl=60.0)
    name = make_name("worker", "old")
    reg.register(name, MeshType.WORKER, ident="old")
    reg.mark_released(name)
    reg._meshes[name].released_monotonic -= 120.0  # simulate the TTL having elapsed
    assert reg.get(f"{name}-sfx") is None  # evicted on access


def test_registry_live_records_are_never_evicted():
    reg = MeshRegistry(released_cap=1)
    live_name = make_name("worker", "held")
    reg.register(live_name, MeshType.WORKER, ident="held")  # stays live
    for i in range(5):
        name = make_name("worker", f"r{i}")
        reg.register(name, MeshType.WORKER, ident=f"r{i}")
        reg.mark_released(name)
    assert live_name in reg._meshes  # a long-held live worker survives the churn
    assert reg.get(live_name).live is True
