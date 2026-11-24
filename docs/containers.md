# Containers

A `Container` adds [Apptainer](https://apptainer.org/) isolation to
an acquired `Machine` without changing the execution model: it implements the
same [`ExecutionTarget`](../src/agentbox/actors/target.py) API (`run`,
`spawn`, `upload`, `download`) and executes every command inside the instance
through `apptainer exec`. Monarch and AgentBox stay on the worker host, so
images need no AgentBox dependencies.
Typically, users should build a SIF customized for their task.

```text
Machine (worker host, Monarch)
  └── Container (Apptainer instance)
        ├── image: SIF path or OCI reference (docker://...)
        ├── explicit env, bind mounts, cwd
        ├── ApptainerOptions: isolation, GPUs, overlays, network
        └── private staging directory for file transfer
```

## Isolation defaults

[`ApptainerOptions`](../src/agentbox/actors/container.py) defaults to
`--containall` (private `/tmp`, IPC, and PID namespaces, no automatic host
binds) and `--no-home`. Both `instance start` and every `exec` use
`--cleanenv`, so the container sees only the image's `%environment`, then the
`Container(env=...)` values, then per-call `env`. `PATH` is not injected; the
image's own `PATH` applies.

Opt-in options cover GPU passthrough (`nv`, `nvccli` for clusters that use
nvidia-container-cli, `rocm`), writable layers
(`writable_tmpfs`, `overlay`, `writable`), `fakeroot`, and `network="none"`
for a loopback-only network. With fakeroot and writable storage a workload can
install packages without host root access.

Bind mounts are `"src:dst"` strings or `BindMount(src, dst, options="ro")`
objects; `src` must exist on the worker. If `cwd` is not already a bind
target, a fresh writable worker directory is bound there, because image
filesystems are read-only.

The exact command construction and validation live in
[`container.py`](../src/agentbox/actors/container.py) and are covered by
[`tests/test_external_container.py`](../tests/test_external_container.py).

## Files

`upload` and `download` move files and directories through a private,
per-container staging directory on the worker that is bound at the same path
inside the instance. The container can change anything in that directory at
any moment, so the host reads and writes copies only in a second worker
directory the container cannot see, and moves each copy across with a rename,
which never follows a symlink the container planted. A directory crosses as one
tar archive that the image's `tar` packs or unpacks, so the host never walks a
tree the container can change; directory copies need `tar` in the image, and
on upload that `tar` decides how entries already in the destination are
treated.

## Services

The container shares the worker's network namespace unless
`network="none"`, so a server started with `container.spawn(...)` is reachable
at the worker's address without port mapping. The application is responsible
for readiness checks, authentication, exposure, and shutdown (stop the
process, or let `Container.stop()` tear down the instance).

## Lifecycle

Use `with Container(...)` / `async with Container(...)`, or call
`start`/`astart` and `stop`/`astop` explicitly. `start_timeout` bounds image
pulls and instance startup. Containers must be stopped before their
`Machine` lease is released.

Runnable entry points:
[`container_demo.py`](../examples/container_demo.py) (commands, interactive
shell, files, overlays) and
[`tty_terminal.py`](../examples/tty_terminal.py) (`--image` for an
interactive terminal inside a container).
