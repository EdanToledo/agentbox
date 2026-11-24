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

"""Schema, classification, and controller-side registry for agentbox Monarch meshes.

Every mesh (host mesh, proc mesh, actor) that agentbox spawns is named through
:func:`make_name` so it carries a structured, ``agentbox_``-prefixed identity of
the form ``agentbox_<kind>[_<ident>]``.  Monarch delivers supervision faults at the
**actor** level with a random suffix appended (``<actor_name>-<rand>``), so the name
that reaches the client fault hook looks like ``agentbox_<kind>_<ident>-<rand>``.

:func:`parse_name` inverts that (stripping the Monarch suffix) into a
:class:`ParsedMesh`, and :data:`FAULT_POLICY` maps every :class:`MeshType` to a
:class:`FaultAction`.  Because every agentbox actor is schema-named, the client hook
can classify every fault it owns; the only residual "unknown" is a genuinely foreign
Monarch-internal mesh (e.g. the per-proc ``logger-*`` actor), handled via
:data:`INTERNAL_DROP_PREFIXES`.

:class:`MeshRegistry` lives on the controller (client process, where the fault hook
runs) and tracks the meshes the client creates, with their type + metadata.  It
subsumes the manager's former ``_handle_mesh_names`` / ``_released_mesh_names`` dicts.
"""

from __future__ import annotations

import collections
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


AGENTBOX_PREFIX = "agentbox_"
_FAULT_SCOPE_RE = re.compile(r"mgr[0-9a-f]{16}")
_FAULT_SCOPE_SEPARATOR = "__"

# Monarch-internal mesh names (not agentbox-owned) whose faults are always a
# co-symptom of a proc/host death that an agentbox actor on the same proc reports
# authoritatively.  Monarch's per-proc logging mesh is named ``log_forwarder-*``
# in current releases and ``logger-*`` in older releases. ``host_agent`` is the
# host-mesh drain agent (emits DrainHost/broken-link faults during host teardown).
INTERNAL_DROP_PREFIXES = ("log_forwarder-", "logger-", "host_agent")


class MeshType(Enum):
    """The role of a mesh, which determines its fault-recovery policy."""

    COORDINATOR = "coordinator"  # the controller's own proc mesh (resource manager)
    WORKER = "worker"  # an acquired worker connection (carries a handle_id)
    HEARTBEAT = "heartbeat"  # a worker supervision anchor (carries a slot_key)
    AUXILIARY = (
        "auxiliary"  # any other agentbox actor (shells, file transfer, admin, ...)
    )


# Descriptive name kinds → their MeshType.  Kinds must not contain "_" or "-" so the
# name grammar stays unambiguous.  Several distinct kinds map to AUXILIARY so names
# stay debuggable while sharing the same (drop) policy.
KIND_TO_TYPE: dict[str, MeshType] = {
    "coordinator": MeshType.COORDINATOR,
    "worker": MeshType.WORKER,
    "heartbeat": MeshType.HEARTBEAT,
    "machineinit": MeshType.AUXILIARY,
    "filesource": MeshType.AUXILIARY,
    "filesink": MeshType.AUXILIARY,
    "filetransfer": MeshType.AUXILIARY,
    "filewarmup": MeshType.AUXILIARY,
    "filearchive": MeshType.AUXILIARY,
    "admin": MeshType.AUXILIARY,
    "subprocess": MeshType.AUXILIARY,
    "taskpool": MeshType.AUXILIARY,
    "streaming": MeshType.AUXILIARY,
    "pty": MeshType.AUXILIARY,
    "python": MeshType.AUXILIARY,
    "container": MeshType.AUXILIARY,
    "imagebuilder": MeshType.AUXILIARY,
}


class FaultAction(Enum):
    """What the fault hook does with a classified fault."""

    FATAL = (
        "fatal"  # raise UnhandledProcMeshFault (tears the client down, with context)
    )
    REPORT_HANDLE_AND_DROP = (
        "report_handle_and_drop"  # evict handle_id from the pool, drop
    )
    REPORT_SLOT_AND_DROP = "report_slot_and_drop"  # evict slot_key from the pool, drop
    DROP = "drop"  # log and drop (non-fatal, nothing to report)


FAULT_POLICY: dict[MeshType, FaultAction] = {
    MeshType.COORDINATOR: FaultAction.FATAL,
    MeshType.WORKER: FaultAction.REPORT_HANDLE_AND_DROP,
    MeshType.HEARTBEAT: FaultAction.REPORT_SLOT_AND_DROP,
    MeshType.AUXILIARY: FaultAction.DROP,
}


def make_name(kind: str, ident: str = "") -> str:
    """Build a schema mesh name: ``agentbox_<kind>`` or ``agentbox_<kind>_<ident>``.

    *kind* must be a registered kind (see :data:`KIND_TO_TYPE`).  *ident* must not
    contain ``-`` (Monarch's fault-suffix delimiter); ``-`` is replaced with ``_``.
    """
    if kind not in KIND_TO_TYPE:
        raise ValueError(f"unknown mesh kind {kind!r}; register it in KIND_TO_TYPE")
    if not ident:
        return f"{AGENTBOX_PREFIX}{kind}"
    return f"{AGENTBOX_PREFIX}{kind}_{ident.replace('-', '_')}"


def make_scoped_name(kind: str, scope: str, ident: str = "") -> str:
    """Build a mesh name carrying its owning manager's unique *scope*."""
    if _FAULT_SCOPE_RE.fullmatch(scope) is None:
        raise ValueError(f"invalid manager fault scope {scope!r}")
    scoped_ident = f"{scope}{_FAULT_SCOPE_SEPARATOR}{ident}" if ident else scope
    return make_name(kind, scoped_ident)


def base_name(mesh_name: str) -> str:
    """Strip Monarch's ``-<rand>`` fault suffix, returning the schema base name."""
    return mesh_name.split("-", 1)[0]


@dataclass(frozen=True)
class ParsedMesh:
    """A classified agentbox mesh name."""

    kind: str
    ident: str
    mesh_type: MeshType


def parse_name(mesh_name: Optional[str]) -> Optional[ParsedMesh]:
    """Classify *mesh_name* (a fault's ``mesh_name``) into a :class:`ParsedMesh`.

    Returns ``None`` for a non-agentbox (foreign / Monarch-internal) name.  An
    ``agentbox_``-prefixed name with an unrecognized kind is classified as
    :attr:`MeshType.AUXILIARY` (safe drop) rather than treated as foreign.
    """
    if not mesh_name:
        return None
    base = base_name(mesh_name)
    if not base.startswith(AGENTBOX_PREFIX):
        return None
    parts = base[len(AGENTBOX_PREFIX) :].split("_")
    kind = parts[0]
    ident = "_".join(parts[1:])
    return ParsedMesh(
        kind=kind, ident=ident, mesh_type=KIND_TO_TYPE.get(kind, MeshType.AUXILIARY)
    )


def fault_scope(mesh_name: Optional[str]) -> Optional[str]:
    """Return the manager scope embedded in *mesh_name*, if present."""
    parsed = parse_name(mesh_name)
    if parsed is None:
        return None
    candidate = parsed.ident.split(_FAULT_SCOPE_SEPARATOR, 1)[0]
    return candidate if _FAULT_SCOPE_RE.fullmatch(candidate) is not None else None


def strip_fault_scope(ident: str, scope: str | None) -> str:
    """Remove *scope* from a parsed mesh identifier owned by that manager."""
    if scope is None:
        return ident
    prefix = f"{scope}{_FAULT_SCOPE_SEPARATOR}"
    return ident[len(prefix) :] if ident.startswith(prefix) else ident


def is_internal_droppable(mesh_name: Optional[str]) -> bool:
    """True if *mesh_name* is a known-benign Monarch-internal mesh (e.g. ``logger-*``)."""
    return mesh_name is not None and mesh_name.startswith(INTERNAL_DROP_PREFIXES)


@dataclass
class MeshRecord:
    """A mesh the controller created, with its classification metadata + lifecycle."""

    name: str  # schema base name (make_name output, no Monarch suffix)
    mesh_type: MeshType
    ident: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_monotonic: float = 0.0
    live: bool = True  # cleared when the worker is intentionally released
    released_monotonic: float = 0.0  # when live was cleared (for TTL eviction)
    failed: bool = False  # set once the fault hook has reported this mesh as failed


class MeshRegistry:
    """Thread-safe controller-side registry of the meshes this client created.

    The fault hook runs on a Monarch callback thread while acquire/release run on
    other threads, so all access is guarded by a single lock. Each mesh keeps one
    :class:`MeshRecord` for its whole lifecycle: registered ``live``, cleared to
    not-live on intentional release. The hook drops faults from a not-live record
    (an intentional teardown emits broken-link faults asynchronously — up to a few
    seconds after ``stop()`` returns).

    A not-live record is retained for ``released_ttl`` (default 60s) — comfortably
    longer than the observed teardown-fault latency — then evicted, so its window
    matches how long a fault could plausibly still be a teardown artifact. Live
    records are never evicted. ``released_cap`` is a hard memory backstop if
    not-live records ever outrun the TTL. A worker name is never reused (it embeds
    a fresh uuid handle_id), so a record only ever matches the mesh that produced
    it, and a fault arriving after eviction is a harmless no-op report against an
    already-returned handle.
    """

    def __init__(self, released_ttl: float = 60.0, released_cap: int = 1024) -> None:
        self._meshes: "collections.OrderedDict[str, MeshRecord]" = (
            collections.OrderedDict()
        )
        self._released_ttl = released_ttl
        self._released_cap = released_cap
        self._lock = threading.Lock()

    def register(
        self, name: str, mesh_type: MeshType, ident: str = "", **metadata: Any
    ) -> None:
        """Record *name* (a schema base name) as a live mesh."""
        with self._lock:
            self._meshes[name] = MeshRecord(
                name=name,
                mesh_type=mesh_type,
                ident=ident,
                metadata=dict(metadata),
                created_monotonic=time.monotonic(),
            )

    def mark_released(self, name: str) -> None:
        """Clear a mesh's ``live`` flag; its later teardown faults are then dropped."""
        with self._lock:
            record = self._meshes.get(name)
            if record is None:
                return
            record.live = False
            record.released_monotonic = time.monotonic()
            self._meshes.move_to_end(name)  # most-recently-released last
            self._evict()

    def get(self, mesh_name: str) -> Optional[MeshRecord]:
        """Return the record for *mesh_name* (Monarch suffix stripped), or None."""
        with self._lock:
            self._evict()  # lazily drop TTL-expired not-live records on access
            return self._meshes.get(base_name(mesh_name))

    def mark_failed(self, mesh_name: str) -> None:
        """Flag a mesh's record as failed (evicted from the pool by the fault hook).

        The release path reads this to skip returning a dead worker to the pool.
        """
        with self._lock:
            record = self._meshes.get(base_name(mesh_name))
            if record is not None:
                record.failed = True

    def live(self) -> list[MeshRecord]:
        """Snapshot of currently-live mesh records (for observability)."""
        with self._lock:
            return [r for r in self._meshes.values() if r.live]

    def _evict(self) -> None:
        """Drop not-live records past their TTL, then any beyond the count backstop."""
        now = time.monotonic()
        dead = [
            (name, record)
            for name, record in self._meshes.items()  # OrderedDict → oldest first
            if not record.live
        ]
        expired = {
            name
            for name, record in dead
            if now - record.released_monotonic > self._released_ttl
        }
        overflow = {name for name, _ in dead[: max(0, len(dead) - self._released_cap)]}
        for name in expired | overflow:
            del self._meshes[name]
