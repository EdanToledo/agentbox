# API map

This page maps concepts to their implementations. The public API is exported
from [`src/agentbox/__init__.py`](../src/agentbox/__init__.py); source
docstrings and [tests](../tests) are the reference for signatures, defaults,
exceptions, and edge cases.

| Area | Purpose | Source | Tests and examples |
| --- | --- | --- | --- |
| Manager | Create pools, acquire machines, scale, shut down | [`actors/manager.py`](../src/agentbox/actors/manager.py) | [`test_manager_*`](../tests) |
| Execution API | `run`, `arun`, `spawn`, `upload`, `download` shared by `Machine` and `Container` | [`actors/target.py`](../src/agentbox/actors/target.py) | [`test_machine.py`](../tests/test_machine.py), [`machine_demo.py`](../examples/machine_demo.py) |
| Machine | Leased worker host | [`actors/machine.py`](../src/agentbox/actors/machine.py) | [`test_machine.py`](../tests/test_machine.py) |
| Container | Apptainer lifecycle, isolation options, bind mounts | [`actors/container.py`](../src/agentbox/actors/container.py) | [`test_external_container.py`](../tests/test_external_container.py), [`container_demo.py`](../examples/container_demo.py) |
| Resource managers | Local and [Slurm](https://slurm.schedmd.com/overview.html) backends behind one pool interface | [`actors/resource_managers`](../src/agentbox/actors/resource_managers) | [`test_resource_manager.py`](../tests/test_resource_manager.py), [`test_slurm_array.py`](../tests/test_slurm_array.py) |
| Remote primitives | Process, PTY, streaming, and file-transfer actors | [`remote`](../src/agentbox/remote) | [`test_pty.py`](../tests/test_pty.py), [`tty_terminal.py`](../examples/tty_terminal.py) |
| File transfer | Verified streaming copies over Ethernet or RDMA | [`remote/file_transfer.py`](../src/agentbox/remote/file_transfer.py), [`remote/chunk_io.py`](../src/agentbox/remote/chunk_io.py), [`remote/rdma.py`](../src/agentbox/remote/rdma.py) | [`test_file_transfer.py`](../tests/test_file_transfer.py), [`test_rdma_transfer.py`](../tests/test_rdma_transfer.py) |

## API shape

Applications acquire a `Machine` from `AgentBoxManager`, optionally wrap it in
a `Container`, and call the execution API on either. The manager owns worker
lifetime; context managers own leases and containers.

- `run(cmd, *, cwd, env, input, timeout, check, text)` always captures output
  and returns a `CompletedProcess`; it raises `TimeoutExpired`, and
  `CalledProcessError` when `check=True`.
- `spawn(cmd, *, pty=False, ...)` returns a pipe-backed `Popen`, or a
  `PtyProcess` with `pty=True`.
- `env` adds variables to the target's environment; `cwd` is a path on the
  target.

Use synchronous methods from synchronous code and the `a`-prefixed methods
from an event loop. Modules and names beginning with `_` are implementation
details.

## Extending AgentBox

Integrations with other systems, such as evaluation harnesses or agent
frameworks, live in their own packages that depend on
`agentbox`. The core does not import them or carry options for them.
That said, it is not unusual to subtree `agentbox` for customizations inside
larger projects.

An integration typically:

1. Owns an `AgentBoxManager` (local or Slurm) for its process and decides how
   many workers to start and when to shut them down.
2. Leases a `Machine` per unit of work with `acquire_machine()` or
   `aacquire_machine()`, using the lease as a context manager.
3. Optionally wraps the machine in a `Container` with `ApptainerOptions` and
   `BindMount`s for isolation.
4. Maps its own operations onto the shared execution API (`run`/`arun`,
   `spawn`, `upload`/`download`), which `Machine` and `Container` both
   implement through `ExecutionTarget`.

Import only the top-level exports from
[`agentbox/__init__.py`](../src/agentbox/__init__.py); underscored
modules and names can change without notice. Code written against
`ExecutionTarget` works on both hosts and containers; see
[`tests/typecheck/target.py`](../tests/typecheck/target.py) for that contract.

See [Core concepts](concepts.md) for the architecture and
[Examples](examples.md) for runnable entry points.
