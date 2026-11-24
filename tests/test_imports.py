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

import inspect

import pytest


def test_import_nonexistent_submodule_raises():
    """A missing AgentBox submodule raises ImportError (guards the import boundary)."""
    with pytest.raises(ImportError):
        import agentbox.does_not_exist  # noqa: F401


def test_import_agentbox():
    import agentbox

    assert isinstance(agentbox.__version__, str)
    assert agentbox.__version__  # non-empty


def test_import_remote_subprocess():
    from agentbox.remote import subprocess

    # Mirrors the stdlib subprocess surface AgentBox reimplements.
    assert hasattr(subprocess, "PIPE")
    assert inspect.isclass(subprocess.Popen)
    # The module-level run/call/check_* mirrors were removed; targets own run().
    assert not hasattr(subprocess, "run")


def test_import_remote_public_api():
    from agentbox.remote import (
        CalledProcessError,
        CompletedProcess,
        Popen,
        PtyProcess,
        TimeoutExpired,
    )

    for cls in (CalledProcessError, CompletedProcess, Popen, PtyProcess):
        assert inspect.isclass(cls)
    assert issubclass(TimeoutExpired, Exception)


def test_import_container_and_machine():
    from agentbox.actors import Container, Machine

    assert inspect.isclass(Container)
    assert inspect.isclass(Machine)


def test_import_agentbox_public_api():
    import agentbox
    from agentbox import AgentBoxManager, Container, ExecutionTarget, Machine

    assert inspect.isclass(AgentBoxManager)
    assert inspect.isclass(Container)
    assert inspect.isclass(ExecutionTarget)
    assert inspect.isclass(Machine)
    assert issubclass(Machine, ExecutionTarget)
    assert issubclass(Container, ExecutionTarget)
    assert not hasattr(agentbox, "TaskPool")
    assert not hasattr(agentbox, "Shell")
