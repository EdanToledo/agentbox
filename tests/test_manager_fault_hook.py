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

"""Tests for AgentBoxManager._client_fault_hook (mesh-schema classification).

The hook never calls Monarch's default hook (bare ``sys.exit(1)``). It classifies
each fault by its ``mesh_name`` via the schema (parse_name) + self._mesh_registry +
FAULT_POLICY, so no agentbox-owned mesh is ever unhandled:
  * released (record not live)            -> drop (expected teardown)
  * WORKER (...worker_<scope>__<handle>)   -> report handle to the pool, drop
  * HEARTBEAT (...heartbeat_<scope>__<slot>) -> report slot to the pool, drop
  * AUXILIARY (shells / file transfer / …) -> drop
  * COORDINATOR (...coordinator_<scope>)   -> raise UnhandledProcMeshFault (fatal)
  * logger-* (Monarch-internal allowlist)  -> drop
  * anything else (truly foreign)          -> raise UnhandledProcMeshFault
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import monarch
import pytest
from agentbox.actors.manager import (
    _FaultHookDispatcher,
    _registered_fault_hook,
    AgentBoxManager,
    UnhandledProcMeshFault,
)
from agentbox.actors.mesh_schema import (
    fault_scope,
    make_name,
    make_scoped_name,
    MeshType,
)


def _make_manager(
    fault_scope: str | None = None, actor_module: object | None = None
) -> AgentBoxManager:
    """Build a manager around a mock pool, without reaping or real fault hooks.

    With *actor_module*, the manager registers its fault hook there (standing in
    for ``monarch.actor``); otherwise it registers none.
    """
    with ExitStack() as stack:
        stack.enter_context(patch("agentbox.process.ensure_monarch_reaper"))
        stack.enter_context(patch("agentbox.process.track_monarch_descendants"))
        if actor_module is None:
            stack.enter_context(patch("agentbox.actors.manager._registered_fault_hook"))
        else:
            stack.enter_context(patch.object(monarch, "actor", actor_module))
        return AgentBoxManager(MagicMock(), None, fault_scope=fault_scope)


def _failure(mesh_name: str | None) -> MagicMock:
    f = MagicMock()
    f.mesh_name = mesh_name
    f.report = lambda: "broken link: channel closed"
    return f


def test_coordinator_fault_raises():
    mgr = _make_manager()
    with pytest.raises(UnhandledProcMeshFault):
        mgr._client_fault_hook(_failure(f"{make_name('coordinator')}-sfx"))


def test_foreign_mesh_raises():
    mgr = _make_manager()
    with pytest.raises(UnhandledProcMeshFault):
        mgr._client_fault_hook(_failure("something-foreign"))


def test_logger_internal_fault_is_dropped():
    mgr = _make_manager()
    mgr._client_fault_hook(_failure("logger-xAaF2vgKqqU"))  # must not raise


def test_log_forwarder_internal_fault_is_dropped():
    mgr = _make_manager()
    mgr._client_fault_hook(_failure("log_forwarder-r4u2gLYUUeW"))


def test_worker_fault_reports_handle_and_drops():
    mgr = _make_manager()
    worker_name = make_name("worker", "h7")
    mgr._mesh_registry.register(
        worker_name, MeshType.WORKER, ident="h7", handle_id="h7"
    )

    mgr._client_fault_hook(_failure(f"{worker_name}-sfx"))  # must not raise

    mgr._manager.report_failure.choose.assert_called_once()
    assert mgr._manager.report_failure.choose.call_args.args[0] == "h7"


def test_worker_fault_uses_name_ident_when_unregistered():
    # Even without a registry record, the handle is recoverable from the name.
    mgr = _make_manager()
    mgr._client_fault_hook(_failure(f"{make_name('worker', 'h9')}-sfx"))
    assert mgr._manager.report_failure.choose.call_args.args[0] == "h9"


def test_worker_fault_invokes_on_machine_failure_callback():
    mgr = _make_manager()
    cb = MagicMock()
    mgr._on_machine_failure = cb
    mgr._client_fault_hook(_failure(f"{make_name('worker', 'h1')}-sfx"))
    cb.assert_called_once()
    assert cb.call_args.args[0] == "h1"


def test_heartbeat_fault_reports_slot_and_drops():
    mgr = _make_manager()
    mgr._client_fault_hook(
        _failure(f"{make_name('heartbeat', 'slurm_9_0')}-sfx")
    )  # must not raise
    mgr._manager.report_failure.choose.assert_called_once()
    assert mgr._manager.report_failure.choose.call_args.args[0] == "slurm_9_0"


def test_auxiliary_fault_is_dropped_without_report():
    mgr = _make_manager()
    mgr._client_fault_hook(
        _failure(f"{make_name('subprocess', '3')}-sfx")
    )  # must not raise
    mgr._manager.report_failure.choose.assert_not_called()


def test_worker_fault_marks_record_failed():
    # So the release path skips returning a dead worker to the pool.
    mgr = _make_manager()
    worker_name = make_name("worker", "h5")
    mgr._mesh_registry.register(
        worker_name, MeshType.WORKER, ident="h5", handle_id="h5"
    )

    mgr._client_fault_hook(_failure(f"{worker_name}-sfx"))

    record = mgr._mesh_registry.get(worker_name)
    assert record is not None and record.failed


def test_released_worker_fault_is_dropped():
    mgr = _make_manager()
    worker_name = make_name("worker", "h2")
    mgr._mesh_registry.register(
        worker_name, MeshType.WORKER, ident="h2", handle_id="h2"
    )
    mgr._mesh_registry.mark_released(worker_name)

    mgr._client_fault_hook(
        _failure(f"{worker_name}-sfx")
    )  # expected teardown: no raise, no report

    mgr._manager.report_failure.choose.assert_not_called()


def _make_shutdown_manager() -> tuple[AgentBoxManager, SimpleNamespace, object]:
    previous_hook = object()
    actor_module = SimpleNamespace(unhandled_fault_hook=previous_hook)
    mgr = _make_manager(actor_module=actor_module)
    return mgr, actor_module, previous_hook


def test_process_fault_hook_routes_to_owning_manager_and_restores_once():
    first = _make_manager()
    second = _make_manager()
    first_worker = make_name("worker", "first_handle")
    second_worker = make_name("worker", "second_handle")
    first._mesh_registry.register(
        first_worker,
        MeshType.WORKER,
        ident="first_handle",
        handle_id="first_handle",
    )
    second._mesh_registry.register(
        second_worker,
        MeshType.WORKER,
        ident="second_handle",
        handle_id="second_handle",
    )
    previous_hook = object()
    actor_module = SimpleNamespace(unhandled_fault_hook=previous_hook)
    first_stack = ExitStack()
    second_stack = ExitStack()

    try:
        first_stack.enter_context(_registered_fault_hook(actor_module, first))
        dispatcher = actor_module.unhandled_fault_hook
        second_stack.enter_context(_registered_fault_hook(actor_module, second))

        assert actor_module.unhandled_fault_hook is dispatcher
        dispatcher(_failure(f"{first_worker}-sfx"))
        first._manager.report_failure.choose.assert_called_once()
        second._manager.report_failure.choose.assert_not_called()

        # Managers may shut down in creation order. Removing the first one must
        # leave the shared hook installed for the second manager.
        first_stack.close()
        assert actor_module.unhandled_fault_hook is dispatcher
        dispatcher(_failure(f"{second_worker}-sfx"))
        second._manager.report_failure.choose.assert_called_once()

        second_stack.close()
        assert actor_module.unhandled_fault_hook is previous_hook
    finally:
        second_stack.close()
        first_stack.close()


def test_process_fault_hook_routes_scoped_heartbeat_to_owning_manager():
    first = _make_manager(fault_scope="mgr1111111111111111")
    second = _make_manager(fault_scope="mgr2222222222222222")
    dispatcher = _FaultHookDispatcher(None)
    dispatcher.add(first)
    dispatcher.add(second)

    heartbeat = make_scoped_name("heartbeat", first._fault_scope, "local_0")
    dispatcher(_failure(f"{heartbeat}-sfx"))

    first._manager.report_failure.choose.assert_called_once()
    assert first._manager.report_failure.choose.call_args.args[0] == "local_0"
    second._manager.report_failure.choose.assert_not_called()


def test_process_fault_hook_drops_late_fault_from_removed_manager():
    first = _make_manager(fault_scope="mgr1111111111111111")
    second = _make_manager(fault_scope="mgr2222222222222222")
    dispatcher = _FaultHookDispatcher(None)
    dispatcher.add(first)
    dispatcher.add(second)
    dispatcher.remove(first)

    worker = make_scoped_name("worker", first._fault_scope, "old_handle")
    dispatcher(_failure(f"{worker}-sfx"))

    second._manager.report_failure.choose.assert_not_called()


def test_process_fault_hook_raises_for_unowned_unscoped_mesh():
    first = _make_manager()
    second = _make_manager()
    dispatcher = _FaultHookDispatcher(None)
    dispatcher.add(first)
    dispatcher.add(second)

    # Unscoped names are not routed by guesswork to any live manager.
    with pytest.raises(UnhandledProcMeshFault, match="heartbeat_local_0"):
        dispatcher(_failure(f"{make_name('heartbeat', 'local_0')}-sfx"))

    first._manager.report_failure.choose.assert_not_called()
    second._manager.report_failure.choose.assert_not_called()


def test_process_fault_hook_delegates_unowned_mesh_to_previous_hook():
    previous_hook = MagicMock()
    manager = _make_manager()
    dispatcher = _FaultHookDispatcher(previous_hook)
    dispatcher.add(manager)
    failure = _failure("foreign_mesh")

    dispatcher(failure)

    previous_hook.assert_called_once_with(failure)
    manager._manager.report_failure.choose.assert_not_called()


@pytest.mark.parametrize(
    "kind, ident",
    [
        ("pty", "0"),
        ("streaming", "3"),
        ("subprocess", "7"),
        ("filesink", "ab12cd34"),
        ("container", "ab12cd34"),
    ],
)
def test_process_fault_hook_drops_unscoped_auxiliary_fault(kind, ident):
    # Auxiliary actors are named without a manager scope. Their faults must not
    # reach Monarch's default hook (sys.exit), which would crash the client,
    # e.g. when a lease is released while a PTY actor cannot be stopped.
    previous_hook = MagicMock(side_effect=SystemExit(1))
    first = _make_manager()
    second = _make_manager()
    dispatcher = _FaultHookDispatcher(previous_hook)
    dispatcher.add(first)
    dispatcher.add(second)

    dispatcher(_failure(f"{make_name(kind, ident)}-sfx"))  # must not raise

    previous_hook.assert_not_called()
    first._manager.report_failure.choose.assert_not_called()
    second._manager.report_failure.choose.assert_not_called()


@pytest.mark.parametrize("mesh_name", [None, "<none>"])
def test_process_fault_hook_sends_unnamed_fault_to_last_manager(mesh_name):
    first = _make_manager()
    second = _make_manager()
    first._client_fault_hook = MagicMock()
    second._client_fault_hook = MagicMock()
    dispatcher = _FaultHookDispatcher(None)
    dispatcher.add(first)
    dispatcher.add(second)
    failure = _failure(mesh_name)

    dispatcher(failure)

    first._client_fault_hook.assert_not_called()
    second._client_fault_hook.assert_called_once_with(failure)


def test_manager_generates_unique_fault_scope_when_not_passed():
    with (
        patch("agentbox.process.ensure_monarch_reaper"),
        patch("agentbox.process.track_monarch_descendants"),
        patch("agentbox.actors.manager._registered_fault_hook"),
    ):
        first = AgentBoxManager(MagicMock(), None)
        second = AgentBoxManager(MagicMock(), None)
        explicit = AgentBoxManager(MagicMock(), None, fault_scope="mgr0123456789abcdef")

    assert first._fault_scope != second._fault_scope
    worker = make_scoped_name("worker", first._fault_scope, "handle")
    assert fault_scope(f"{worker}-sfx") == first._fault_scope
    assert first._owns_fault(f"{worker}-sfx")
    assert not second._owns_fault(f"{worker}-sfx")
    assert explicit._fault_scope == "mgr0123456789abcdef"


def test_shutdown_keeps_fault_hook_installed_during_cleanup():
    mgr, actor_module, previous_hook = _make_shutdown_manager()
    active_hook = actor_module.unhandled_fault_hook
    shutdown_future = MagicMock()

    def wait_for_shutdown():
        assert actor_module.unhandled_fault_hook is active_hook

    shutdown_future.get.side_effect = wait_for_shutdown
    mgr._manager.shutdown.choose.return_value = shutdown_future

    mgr.shutdown()

    assert actor_module.unhandled_fault_hook is previous_hook


def test_shutdown_retry_keeps_fault_hook_active_until_cleanup_succeeds():
    mgr, actor_module, previous_hook = _make_shutdown_manager()
    active_hook = actor_module.unhandled_fault_hook
    shutdown_future = MagicMock()
    attempts = 0

    def shutdown_attempt() -> None:
        nonlocal attempts
        assert actor_module.unhandled_fault_hook is active_hook
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cleanup failed")

    shutdown_future.get.side_effect = shutdown_attempt
    mgr._manager.shutdown.choose.return_value = shutdown_future

    with pytest.raises(RuntimeError, match="cleanup failed"):
        mgr.shutdown()

    assert actor_module.unhandled_fault_hook is active_hook
    mgr.shutdown()

    assert attempts == 2
    assert mgr._shutdown_done
    assert actor_module.unhandled_fault_hook is previous_hook


def test_context_manager_preserves_body_exception_when_shutdown_fails():
    mgr, _actor_module, _previous_hook = _make_shutdown_manager()
    shutdown_future = MagicMock()
    shutdown_future.get.side_effect = RuntimeError("cleanup failed")
    mgr._manager.shutdown.choose.return_value = shutdown_future

    with pytest.raises(ValueError, match="body failed") as exc_info:
        with mgr:
            raise ValueError("body failed")

    assert exc_info.value.__notes__ == [
        "AgentBoxManager shutdown also failed: RuntimeError: cleanup failed"
    ]


def test_context_manager_propagates_shutdown_failure_without_body_exception():
    mgr, _actor_module, _previous_hook = _make_shutdown_manager()
    shutdown_future = MagicMock()
    shutdown_future.get.side_effect = RuntimeError("cleanup failed")
    mgr._manager.shutdown.choose.return_value = shutdown_future

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with mgr:
            pass


async def test_ashutdown_keeps_fault_hook_installed_during_cleanup():
    mgr, actor_module, previous_hook = _make_shutdown_manager()
    active_hook = actor_module.unhandled_fault_hook

    async def wait_for_shutdown():
        assert actor_module.unhandled_fault_hook is active_hook

    mgr._manager.shutdown.choose.return_value = wait_for_shutdown()

    await mgr.ashutdown()

    assert actor_module.unhandled_fault_hook is previous_hook


async def test_ashutdown_retry_keeps_fault_hook_active_until_cleanup_succeeds():
    mgr, actor_module, previous_hook = _make_shutdown_manager()
    active_hook = actor_module.unhandled_fault_hook
    attempts = 0

    async def shutdown_attempt() -> None:
        nonlocal attempts
        assert actor_module.unhandled_fault_hook is active_hook
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cleanup failed")

    mgr._manager.shutdown.choose.side_effect = shutdown_attempt

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await mgr.ashutdown()

    assert actor_module.unhandled_fault_hook is active_hook
    await mgr.ashutdown()

    assert attempts == 2
    assert mgr._shutdown_done
    assert actor_module.unhandled_fault_hook is previous_hook


async def test_async_context_manager_preserves_body_exception_when_shutdown_fails():
    mgr, _actor_module, _previous_hook = _make_shutdown_manager()

    async def fail_shutdown():
        raise RuntimeError("cleanup failed")

    mgr._manager.shutdown.choose.return_value = fail_shutdown()

    with pytest.raises(ValueError, match="body failed") as exc_info:
        async with mgr:
            raise ValueError("body failed")

    assert exc_info.value.__notes__ == [
        "AgentBoxManager shutdown also failed: RuntimeError: cleanup failed"
    ]
