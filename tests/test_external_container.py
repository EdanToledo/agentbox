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

"""Tests for the Apptainer-backed :class:`Container` and its worker actor."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _endpoints import endpoint_method
from agentbox.actors.container import (
    _ContainerActor,
    _format_apptainer_command_for_log,
    _subprocess_output_text,
    ApptainerNotFoundError,
    ApptainerOptions,
    BindMount,
)


def test_worker_preserves_effective_start_args() -> None:
    endpoint = endpoint_method(_ContainerActor, "start_instance")
    start_args = [
        "--nvccli",
        "--fakeroot",
        "--overlay",
        "/runtime/overlay:rw",
    ]

    with patch("subprocess.run") as run:
        endpoint(
            _ContainerActor(),
            "image.sif",
            "instance-name",
            start_args,
            {},
            [],
            None,
            "/worker/apptainer.conf",
        )

    cmd = run.call_args.args[0]
    assert cmd[:3] == ["apptainer", "--config", "/worker/apptainer.conf"]
    assert cmd[-6:] == start_args + ["image.sif", "instance-name"]


def test_worker_start_command_includes_binds_and_env() -> None:
    start = endpoint_method(_ContainerActor, "start_instance")

    with patch("subprocess.run") as run:
        start(
            _ContainerActor(),
            "image.sif",
            "instance-name",
            ["--containall"],
            {"FOO": "bar"},
            ["/data:/data"],
        )

    assert run.call_args.args[0] == [
        "apptainer",
        "instance",
        "start",
        "--cleanenv",
        "--bind",
        "/data:/data",
        "--env",
        "FOO=bar",
        "--containall",
        "image.sif",
        "instance-name",
    ]


class TestWorkerStopInstance:
    @pytest.fixture
    def shutdown(self):
        clock = [0.0]

        def advance(seconds):
            clock[0] += seconds

        with (
            patch("subprocess.run") as run,
            patch(
                "agentbox.actors.container.time.monotonic",
                side_effect=lambda: clock[0],
            ),
            patch("agentbox.actors.container.time.sleep", side_effect=advance) as sleep,
        ):
            stop = endpoint_method(_ContainerActor, "stop_instance")
            yield (
                lambda apptainer_config_path=None: stop(
                    _ContainerActor(), "inst", apptainer_config_path
                ),
                run,
                clock,
                sleep,
            )

    def test_waits_for_target_to_disappear(self, shutdown):
        stop, run, clock, _ = shutdown
        present = subprocess.CompletedProcess(
            [], 0, '{"instances": [{"instance": "inst"}, {"instance": "sibling"}]}'
        )
        absent = subprocess.CompletedProcess(
            [], 0, '{"instances": [{"instance": "sibling"}]}'
        )
        run.side_effect = [subprocess.CompletedProcess([], 0), present, present, absent]

        stop()

        assert run.call_args_list[0].args[0] == [
            "apptainer",
            "instance",
            "stop",
            "inst",
        ]
        assert [call.args[0] for call in run.call_args_list[1:]] == [
            ["apptainer", "instance", "list", "--json"]
        ] * 3
        assert clock[0] > 0
        assert all(call.kwargs["timeout"] > 0 for call in run.call_args_list)

    def test_uses_custom_config_for_stop_and_status(self, shutdown):
        stop, run, _, _ = shutdown
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0, '{"instances": []}'),
        ]

        stop(apptainer_config_path="/worker/apptainer.conf")

        prefix = ["apptainer", "--config", "/worker/apptainer.conf"]
        assert [call.args[0] for call in run.call_args_list] == [
            [*prefix, "instance", "stop", "inst"],
            [*prefix, "instance", "list", "--json"],
        ]

    @pytest.mark.parametrize("returncode", [0, 1])
    def test_already_absent_is_success_only_after_verification(
        self, shutdown, returncode
    ):
        stop, run, _, _ = shutdown
        run.side_effect = [
            subprocess.CompletedProcess([], returncode, "", "already stopped"),
            subprocess.CompletedProcess([], 0, '{"instances": []}'),
        ]

        stop()

        assert run.call_count == 2
        assert run.call_args.args[0] == ["apptainer", "instance", "list", "--json"]

    def test_stop_launch_failure_propagates(self, shutdown):
        stop, run, _, _ = shutdown
        run.side_effect = OSError("cannot launch")

        with pytest.raises(OSError, match="cannot launch"):
            stop()

    def test_nonzero_stop_waits_until_deadline_before_reporting_stderr(self, shutdown):
        stop, run, clock, sleep = shutdown
        run.side_effect = [
            subprocess.CompletedProcess([], 1, "", "permission denied"),
            subprocess.CompletedProcess([], 0, '{"instances": [{"instance": "inst"}]}'),
        ]
        sleep.side_effect = lambda _: clock.__setitem__(0, 60)

        with pytest.raises(RuntimeError) as exc_info:
            stop()

        assert "timed out" in str(exc_info.value)
        assert "permission denied" in str(exc_info.value)
        assert run.call_count == 2

    def test_transient_status_failures_are_retried(self, shutdown):
        stop, run, _, _ = shutdown
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(
                1,
                ["apptainer", "instance", "list", "--json"],
                stderr="runtime busy",
            ),
            subprocess.CompletedProcess([], 0, "not-json"),
            subprocess.CompletedProcess([], 0, '{"instances": []}'),
        ]

        stop()

        assert run.call_count == 4

    @pytest.mark.parametrize(
        "listing",
        ["not-json", "null", "{}", '{"instances": null}', '{"instances": [{}]}'],
    )
    def test_invalid_status_times_out_instead_of_assuming_stopped(
        self, shutdown, listing
    ):
        stop, run, clock, sleep = shutdown
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0, listing),
        ]
        sleep.side_effect = lambda _: clock.__setitem__(0, 60)

        with pytest.raises(RuntimeError, match="invalid instance list"):
            stop()

        assert run.call_count == 2

    def test_status_command_failure_is_retried_until_deadline(self, shutdown):
        stop, run, clock, sleep = shutdown
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(
                1,
                ["apptainer", "instance", "list", "--json"],
                stderr="cannot list instances",
            ),
        ]
        sleep.side_effect = lambda _: clock.__setitem__(0, 60)

        with pytest.raises(RuntimeError, match="cannot list instances"):
            stop()

        assert run.call_count == 2

    @pytest.mark.parametrize("phase", ["stop", "list"])
    def test_subprocess_timeout_is_bounded_and_reported(self, shutdown, phase):
        stop, run, clock, sleep = shutdown
        timeout = subprocess.TimeoutExpired(
            ["apptainer", "instance", phase], 60, stderr=b"slow runtime"
        )
        run.side_effect = (
            [timeout]
            if phase == "stop"
            else [subprocess.CompletedProcess([], 0), timeout]
        )
        sleep.side_effect = lambda _: clock.__setitem__(0, 60)

        with pytest.raises(RuntimeError, match="timed out") as exc_info:
            stop()

        assert "slow runtime" in str(exc_info.value)
        assert all(0 < call.kwargs["timeout"] <= 60 for call in run.call_args_list)

    def test_stop_and_verification_share_a_deadline(self, shutdown):
        stop, run, clock, sleep = shutdown

        def command(argv, **kwargs):
            if argv[2] == "stop":
                assert kwargs["timeout"] == 60
                clock[0] += 10
                return subprocess.CompletedProcess(argv, 0)
            assert kwargs["timeout"] == 50
            return subprocess.CompletedProcess(
                argv, 0, '{"instances": [{"instance": "inst"}]}'
            )

        run.side_effect = command
        sleep.side_effect = lambda _: clock.__setitem__(0, 60)

        with pytest.raises(RuntimeError, match="timed out"):
            stop()

        assert run.call_count == 2
        sleep.assert_called_once()


def test_apptainer_command_log_formatter_redacts_env_values() -> None:
    formatted = _format_apptainer_command_for_log(
        [
            "apptainer",
            "exec",
            "--env",
            "TOKEN=top-secret",
            "--env=OTHER=also-secret",
            "instance://test",
            "printenv",
        ]
    )

    assert "TOKEN=<redacted>" in formatted
    assert "--env=OTHER=<redacted>" in formatted
    assert "top-secret" not in formatted
    assert "also-secret" not in formatted


def test_subprocess_output_text_marks_missing_stderr() -> None:
    assert _subprocess_output_text(None) == "(no stderr)"


def test_worker_start_keeps_launcher_environment() -> None:
    start = endpoint_method(_ContainerActor, "start_instance")
    with patch("subprocess.run") as run:
        start(None, "image.sif", "test-instance", [], {}, [])

    run.assert_called_once()
    assert "env" not in run.call_args.kwargs


def test_worker_applies_container_boot_timeout() -> None:
    start = endpoint_method(_ContainerActor, "start_instance")
    with patch("subprocess.run") as run:
        start(None, "image.sif", "test-instance", [], {}, [], 123)

    assert run.call_args.kwargs["timeout"] == 123


def test_worker_reports_container_boot_timeout_without_leaking_env(
    caplog: pytest.LogCaptureFixture,
) -> None:
    start = endpoint_method(_ContainerActor, "start_instance")
    secret = "top-secret-token"
    timeout_error = subprocess.TimeoutExpired(
        ["apptainer", "instance", "start", "--env", f"TOKEN={secret}"],
        2,
    )
    with (
        patch(
            "subprocess.run",
            side_effect=[timeout_error, subprocess.CompletedProcess([], 0)],
        ) as run,
        caplog.at_level("INFO", logger="agentbox.actors.container"),
    ):
        with pytest.raises(RuntimeError, match="instance start timed out"):
            start(
                None,
                "image.sif",
                "test-instance",
                [],
                {"TOKEN": secret},
                [],
                2,
                "/worker/apptainer.conf",
            )

    assert run.call_args_list[1].args[0] == [
        "apptainer",
        "--config",
        "/worker/apptainer.conf",
        "instance",
        "stop",
        "test-instance",
    ]
    assert secret not in caplog.text


class TestWorkerStartErrors:
    """start_instance must surface the apptainer error rather than hiding stderr."""

    @patch("subprocess.run")
    def test_start_reraises_with_stderr(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=255,
            cmd=["apptainer"],
            output="some stdout",
            stderr="FATAL: image not found",
        )
        start = endpoint_method(_ContainerActor, "start_instance")
        with pytest.raises(RuntimeError) as excinfo:
            start(None, "missing.sif", "inst", [], {}, [])
        msg = str(excinfo.value)
        assert "FATAL: image not found" in msg  # stderr is surfaced
        assert "255" in msg

    @patch("subprocess.run", side_effect=FileNotFoundError())
    def test_start_missing_binary_raises_apptainer_not_found(self, mock_run):
        start = endpoint_method(_ContainerActor, "start_instance")
        with pytest.raises(ApptainerNotFoundError):
            start(None, "x.sif", "inst", [], {}, [])


def test_worker_staging_dir_lifecycle() -> None:
    actor = _ContainerActor()
    create = endpoint_method(_ContainerActor, "create_staging_dir")
    cleanup = endpoint_method(_ContainerActor, "cleanup_staging_dir")

    path = create(actor, "agentbox-test-")

    assert os.path.isdir(path)
    assert os.path.basename(path).startswith("agentbox-test-")
    cleanup(actor, path)
    assert not os.path.exists(path)
    cleanup(actor, path)  # already removed: still a no-op


class TestBindMount:
    def test_same_path(self):
        assert BindMount("/data").to_arg() == "/data"

    def test_src_dst(self):
        assert BindMount("/data", "/mnt").to_arg() == "/data:/mnt"

    def test_options_default_dst_to_src(self):
        assert BindMount("/data", options="ro").to_arg() == "/data:/data:ro"

    def test_container_normalizes_bind_mounts(self):
        from agentbox.actors.container import Container

        c = Container(
            MagicMock(),
            image="test.sif",
            bind_mounts=["/a:/b", BindMount("/data", "/mnt", "ro")],
        )

        assert c._bind_mounts == ["/a:/b", "/data:/mnt:ro"]


class TestBuildExecPrefix:
    """Tests for the shared _build_exec_prefix function."""

    def test_basic_prefix(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst")
        assert prefix[0] == "apptainer"
        assert prefix[1] == "exec"
        assert "--cleanenv" in prefix
        assert prefix[-1] == "instance://my-inst"

    def test_prefix_with_env(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst", env={"FOO": "bar"})
        env_indices = [i for i, arg in enumerate(prefix) if arg == "--env"]
        env_values = [prefix[i + 1] for i in env_indices]
        assert "FOO=bar" in env_values

    def test_prefix_with_cwd(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst", cwd="/workspace")
        assert "--pwd" in prefix
        pwd_idx = prefix.index("--pwd")
        assert prefix[pwd_idx + 1] == "/workspace"

    def test_prefix_with_bind_mounts(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst", bind_mounts=["/data:/data"])
        bind_indices = [i for i, arg in enumerate(prefix) if arg == "--bind"]
        bind_values = [prefix[i + 1] for i in bind_indices]
        assert "/data:/data" in bind_values

    def test_prefix_with_apptainer_args(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst", apptainer_args=["--nv"])
        assert "--nv" in prefix

    def test_prefix_no_default_path(self):
        # The SIF's own %environment owns PATH — _build_exec_prefix must not
        # inject a default PATH, which would shadow conda/Docker SIF PATHs.
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst")
        env_indices = [i for i, arg in enumerate(prefix) if arg == "--env"]
        env_keys = [prefix[i + 1].split("=", 1)[0] for i in env_indices]
        assert "PATH" not in env_keys

    def test_prefix_env_merge(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix("my-inst", instance_env={"A": "1"}, env={"B": "2"})
        env_indices = [i for i, arg in enumerate(prefix) if arg == "--env"]
        env_values = [prefix[i + 1] for i in env_indices]
        assert "A=1" in env_values
        assert "B=2" in env_values

    def test_prefix_env_override(self):
        from agentbox.actors.container import _build_exec_prefix

        prefix = _build_exec_prefix(
            "my-inst", instance_env={"A": "1"}, env={"A": "override"}
        )
        env_indices = [i for i, arg in enumerate(prefix) if arg == "--env"]
        env_values = [prefix[i + 1] for i in env_indices]
        assert "A=override" in env_values
        assert "A=1" not in env_values


class TestContainerInit:
    def test_init_defaults_contract(self):
        """Default construction yields a plain isolated instance, not yet started."""
        from agentbox.actors.container import Container

        machine = MagicMock()
        c = Container(machine, image="test.sif")

        name = c.instance_name
        assert name.startswith("agentbox-ctr-")
        isolation = ["--no-home", "--containall"]
        assert c._start_cmd() == [
            "apptainer",
            "instance",
            "start",
            "--cleanenv",
            *isolation,
            "test.sif",
            name,
        ]
        assert c._host_argv(["true"], None, None)[0] == [
            "apptainer",
            "exec",
            "--cleanenv",
            *isolation,
            f"instance://{name}",
            "true",
        ]
        # Nothing is spawned or staged until start().
        machine.proc_mesh.spawn.assert_not_called()
        with pytest.raises(RuntimeError, match="not started"):
            _ = c.staging_dir

    def test_custom_options(self):
        from agentbox.actors.container import Container

        machine = MagicMock()
        c = Container(
            machine,
            image="my.sif",
            instance_name="custom-inst",
            options=ApptainerOptions(nv=True),
            apptainer_config_path="/worker/apptainer.conf",
            env={"FOO": "bar"},
            bind_mounts=["/data:/data"],
            cwd="/workspace",
        )

        def flag_values(cmd: list[str], flag: str) -> list[str]:
            return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == flag]

        start = c._start_cmd()
        assert start[-2:] == ["my.sif", "custom-inst"]
        assert flag_values(start, "--env") == ["FOO=bar"]
        assert flag_values(start, "--bind") == ["/data:/data"]
        assert "--nv" in start

        exec_cmd, _ = c._host_argv(["true"], None, None)
        assert exec_cmd[-2:] == ["instance://custom-inst", "true"]
        assert flag_values(exec_cmd, "--env") == ["FOO=bar"]
        assert flag_values(exec_cmd, "--bind") == ["/data:/data"]
        assert flag_values(exec_cmd, "--pwd") == ["/workspace"]
        assert "--nv" in exec_cmd
        for cmd in (start, exec_cmd):
            assert flag_values(cmd, "--config") == ["/worker/apptainer.conf"]

    def test_custom_config_is_in_formatted_start_command(self):
        from agentbox.actors.container import Container

        container = Container(
            MagicMock(),
            image="test.sif",
            instance_name="test-inst",
            apptainer_config_path="/worker/apptainer.conf",
        )

        assert container._start_cmd()[:5] == [
            "apptainer",
            "--config",
            "/worker/apptainer.conf",
            "instance",
            "start",
        ]


class TestContainerStagingDir:
    def test_staging_dir_before_start_raises(self):
        from agentbox.actors.container import Container

        machine = MagicMock()
        c = Container(machine, image="test.sif")
        with pytest.raises(RuntimeError, match="not started"):
            _ = c.staging_dir


class TestContainerStart:
    def test_start_creates_staging_and_starts_instance(self):
        from agentbox.actors.container import Container

        container = Container(
            MagicMock(),
            image="test.sif",
            instance_name="test-inst",
            apptainer_config_path="/worker/apptainer.conf",
        )
        container._apptainer_start_args.extend(
            ["--fakeroot", "--overlay", "/runtime/overlay:rw"]
        )
        actor = MagicMock()
        actor.create_staging_dir.choose.return_value.get.return_value = "/tmp/staging"
        actor.start_instance.choose.return_value.get.return_value = None
        container._actor_mesh = actor

        container.start()

        assert container._running is True
        assert container.staging_dir == "/tmp/staging"
        actor.start_instance.choose.assert_called_once_with(
            "test.sif",
            "test-inst",
            [
                "--no-home",
                "--containall",
                "--fakeroot",
                "--overlay",
                "/runtime/overlay:rw",
            ],
            {},
            ["/tmp/staging:/tmp/staging"],
            None,
            "/worker/apptainer.conf",
        )

    async def test_astart_creates_staging_and_starts_instance(self):
        from agentbox.actors.container import Container

        container = Container(
            MagicMock(),
            image="test.sif",
            instance_name="test-inst",
            apptainer_config_path="/worker/apptainer.conf",
            start_timeout=30.0,
        )
        actor = MagicMock()
        actor.create_staging_dir.choose = AsyncMock(return_value="/tmp/staging")
        actor.start_instance.choose = AsyncMock(return_value=None)
        container._actor_mesh = actor

        await container.astart()

        assert container._running is True
        actor.start_instance.choose.assert_awaited_once_with(
            "test.sif",
            "test-inst",
            ["--no-home", "--containall"],
            {},
            ["/tmp/staging:/tmp/staging"],
            30.0,
            "/worker/apptainer.conf",
        )

    def test_start_binds_writable_dir_for_unbound_cwd(self):
        from agentbox.actors.container import Container

        container = Container(
            MagicMock(), image="test.sif", instance_name="test-inst", cwd="/work"
        )
        actor = MagicMock()
        actor.create_staging_dir.choose.return_value.get.side_effect = [
            "/tmp/staging",
            "/tmp/host",
            "/tmp/cwd",
        ]
        container._actor_mesh = actor

        container.start()

        # The host directory for file copies is never bound into the container.
        binds = actor.start_instance.choose.call_args.args[4]
        assert binds == ["/tmp/staging:/tmp/staging", "/tmp/cwd:/work"]
        assert container._cwd_dir == "/tmp/cwd"
        assert container._host_dir == "/tmp/host"
        start_cmd = container._start_cmd()
        assert start_cmd[start_cmd.index("/tmp/cwd:/work") - 1] == "--bind"

    def test_start_reuses_existing_cwd_bind(self):
        from agentbox.actors.container import Container

        container = Container(
            MagicMock(),
            image="test.sif",
            instance_name="test-inst",
            cwd="/work",
            bind_mounts=["/host/work:/work"],
        )
        actor = MagicMock()
        actor.create_staging_dir.choose.return_value.get.return_value = "/tmp/staging"
        container._actor_mesh = actor

        container.start()

        assert [c.args for c in actor.create_staging_dir.choose.call_args_list] == [
            ("agentbox-staging-",),
            ("agentbox-host-",),
        ]
        assert container._cwd_dir is None


class TestContainerExecPrefix:
    def test_exec_prefix_extra_env(self):
        from agentbox.actors.container import Container

        machine = MagicMock()
        c = Container(
            machine, image="test.sif", instance_name="test-inst", env={"A": "1"}
        )

        prefix = c._exec_prefix(env={"B": "2"})
        env_indices = [i for i, arg in enumerate(prefix) if arg == "--env"]
        env_values = [prefix[i + 1] for i in env_indices]
        assert "A=1" in env_values
        assert "B=2" in env_values

    def test_exec_prefix_uses_custom_config(self):
        from agentbox.actors.container import Container

        container = Container(
            MagicMock(),
            image="test.sif",
            apptainer_config_path="/worker/apptainer.conf",
        )

        assert container._exec_prefix()[:4] == [
            "apptainer",
            "--config",
            "/worker/apptainer.conf",
            "exec",
        ]


def test_worker_is_running_uses_custom_config() -> None:
    endpoint = endpoint_method(_ContainerActor, "is_running")
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, '{"instances": [{"instance": "inst"}]}'
        ),
    ) as run:
        running = endpoint(_ContainerActor(), "inst", "/worker/apptainer.conf")

    assert running is True
    assert run.call_args.args[0] == [
        "apptainer",
        "--config",
        "/worker/apptainer.conf",
        "instance",
        "list",
        "--json",
    ]


def test_container_is_running_forwards_custom_config() -> None:
    from agentbox.actors.container import Container

    container = Container(
        MagicMock(),
        image="test.sif",
        instance_name="inst",
        apptainer_config_path="/worker/apptainer.conf",
    )
    actor = MagicMock()
    actor.is_running.choose.return_value.get.return_value = True
    container._actor_mesh = actor

    assert container.is_running() is True
    actor.is_running.choose.assert_called_once_with("inst", "/worker/apptainer.conf")


class TestContainerRunEnv:
    """Per-call env on ``run`` must be injected into the container as
    ``--env``, not handed to the apptainer launcher process — where ``--cleanenv`` would scrub it and a
    PATH-less dict would make the ``apptainer`` binary itself unresolvable.
    """

    @patch("agentbox.remote.subprocess._run")
    def test_run_env_injected_as_container_env_not_launcher(self, mock_run, caplog):
        from agentbox.actors.container import Container

        caplog.set_level(logging.DEBUG, logger="agentbox.actors.container")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        machine = MagicMock()
        c = Container(machine, image="test.sif", instance_name="ti", env={"A": "1"})

        c.run("echo hi", env={"B": "2"})

        assert mock_run.call_count == 1
        cmd = mock_run.call_args.args[0]
        env_vals = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--env"]
        assert "A=1" in env_vals  # instance env
        assert "B=2" in env_vals  # per-call env, both inside the container
        # The launcher env is left intact so PATH (and thus `apptainer`) resolves.
        assert "env" not in mock_run.call_args.kwargs
        assert mock_run.call_args.kwargs["cwd"] is None
        assert "Executing Apptainer command:" in caplog.text
        assert "/bin/sh -c 'echo hi'" in caplog.text
        assert "A=1" not in caplog.text
        assert "B=2" not in caplog.text
        assert "A=<redacted>" in caplog.text
        assert "B=<redacted>" in caplog.text

    @patch("agentbox.remote.subprocess._run")
    def test_run_command_is_not_logged_at_info(self, mock_run, caplog):
        from agentbox.actors.container import Container

        caplog.set_level(logging.INFO, logger="agentbox.actors.container")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        container = Container(
            MagicMock(), image="test.sif", instance_name="ti", env={"A": "1"}
        )

        container.run("echo hi", env={"B": "2"})

        assert "Executing Apptainer command:" not in caplog.text


class TestBindDst:
    """Destination extraction used by Container.start() to decide whether the
    requested cwd is already bound (regression for the substring false-positive
    that skipped the writable cwd bind when an unrelated dst shared a prefix).
    """

    @pytest.mark.parametrize(
        "bind", ["/work", "/host/data:/work", "/host/data:/work:ro"]
    )
    def test_extracts_destination(self, bind):
        from agentbox.actors.container import _bind_dst

        assert _bind_dst(bind) == "/work"

    def test_prefix_collision_is_not_a_match(self):
        from agentbox.actors.container import _bind_dst

        # The whole point of the fix: a different destination that merely shares
        # a prefix with the requested cwd must NOT be treated as that cwd.
        cwd = "/work"
        binds = ["/host/data:/workspace", "/h:/workfoo", "/work:/elsewhere"]
        assert not any(_bind_dst(b) == cwd for b in binds)


class TestContainerStopCleanup:
    """Shutdown preserves resources until stopping succeeds, including on retry."""

    @pytest.fixture(params=["stop", "astop"])
    def lifecycle(self, request, tmp_path):
        from agentbox.actors.container import Container

        c = Container(
            MagicMock(),
            image="t.sif",
            instance_name="inst",
            apptainer_config_path="/worker/apptainer.conf",
        )
        actor = MagicMock()
        c._actor_mesh = actor
        c._running = True
        c._staging_dir = str(tmp_path / "staging")
        c._cwd_dir = str(tmp_path / "cwd")
        c._host_dir = str(tmp_path / "host")
        if request.param == "astop":
            actor.stop_instance.choose = AsyncMock()
            actor.cleanup_staging_dir.choose = AsyncMock()
            stop_rpc = actor.stop_instance.choose
            cleanup_rpc = actor.cleanup_staging_dir.choose
        else:
            stop_rpc = actor.stop_instance.choose.return_value.get
            cleanup_rpc = actor.cleanup_staging_dir.choose.return_value.get

        async def stop():
            if request.param == "astop":
                await c.astop()
            else:
                c.stop()

        yield c, actor, stop, stop_rpc, cleanup_rpc

    @pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
    async def test_failed_stop_preserves_resources_for_retry(
        self, lifecycle, error_type
    ):
        c, actor, stop, stop_rpc, _ = lifecycle
        staging, cwd = c._staging_dir, c._cwd_dir
        stop_rpc.side_effect = error_type("stop interrupted")

        with pytest.raises(error_type, match="stop interrupted"):
            await stop()

        actor.cleanup_staging_dir.choose.assert_not_called()
        assert c._running is True
        assert c._staging_dir == staging
        assert c._cwd_dir == cwd
        assert c._host_dir is not None

        stop_rpc.side_effect = None
        await stop()

        assert stop_rpc.call_count == 2
        assert c._running is False
        assert c._staging_dir is None
        assert c._cwd_dir is None
        assert c._host_dir is None

    async def test_successful_stop_is_ordered_and_idempotent(self, lifecycle):
        c, actor, stop, stop_rpc, cleanup_rpc = lifecycle
        staging, cwd, host = c._staging_dir, c._cwd_dir, c._host_dir

        def cleanup(*args):
            assert stop_rpc.call_count == 1
            assert c._running is False

        cleanup_rpc.side_effect = cleanup
        await stop()
        await stop()

        actor.stop_instance.choose.assert_called_once_with(
            "inst", "/worker/apptainer.conf"
        )
        assert [
            call.args[0] for call in actor.cleanup_staging_dir.choose.call_args_list
        ] == [staging, cwd, host]
        assert c._staging_dir is None
        assert c._cwd_dir is None
        assert c._host_dir is None

    @pytest.mark.parametrize("failed_directory", ["staging", "cwd"])
    async def test_cleanup_rpc_failure_preserves_only_failed_directory(
        self, lifecycle, failed_directory
    ):
        c, actor, stop, _, cleanup_rpc = lifecycle
        staging, cwd, host = c._staging_dir, c._cwd_dir, c._host_dir
        error = RuntimeError("cleanup failed")
        cleanup_rpc.side_effect = (
            [error, None, None]
            if failed_directory == "staging"
            else [None, error, None]
        )

        with pytest.raises(RuntimeError, match="cleanup failed"):
            await stop()

        assert [
            call.args[0] for call in actor.cleanup_staging_dir.choose.call_args_list
        ] == [staging, cwd, host]
        assert c._running is False
        assert c._staging_dir == (staging if failed_directory == "staging" else None)
        assert c._cwd_dir == (cwd if failed_directory == "cwd" else None)

        cleanup_rpc.side_effect = None
        await stop()

        actor.stop_instance.choose.assert_called_once_with(
            "inst", "/worker/apptainer.conf"
        )
        assert actor.cleanup_staging_dir.choose.call_count == 4
        actor.cleanup_staging_dir.choose.assert_called_with(
            staging if failed_directory == "staging" else cwd
        )
        assert c._staging_dir is None
        assert c._cwd_dir is None


def test_userns_is_passed_to_start_and_exec() -> None:
    options = ApptainerOptions(userns=True)

    assert "--userns" in options.to_start_args()
    assert "--userns" in options.to_exec_args()


def test_secure_isolation_is_enabled_for_start_and_exec_by_default() -> None:
    options = ApptainerOptions()

    assert "--containall" in options.to_start_args()
    assert "--containall" in options.to_exec_args()
    assert "--no-home" in options.to_start_args()
    assert "--no-home" in options.to_exec_args()


def test_disabling_containall_warns_and_omits_flag() -> None:
    with pytest.warns(UserWarning, match="shares the worker's PID and IPC namespaces"):
        options = ApptainerOptions(containall=False)

    assert "--containall" not in options.to_start_args()
    assert "--containall" not in options.to_exec_args()


def test_fakeroot_is_passed_only_to_instance_start() -> None:
    options = ApptainerOptions(fakeroot=True)

    assert "--fakeroot" in options.to_start_args()
    assert "--fakeroot" not in options.to_exec_args()


def test_network_namespace_is_configured_only_at_instance_start() -> None:
    options = ApptainerOptions(network="none")

    assert options.to_start_args()[-3:] == ["--net", "--network", "none"]
    assert "--net" not in options.to_exec_args()
    assert "--network" not in options.to_exec_args()
