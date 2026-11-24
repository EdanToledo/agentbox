# Installation

AgentBox requires the Python version declared in
[`pyproject.toml`](../pyproject.toml) (3.12+) and installs
[`torchmonarch`](https://github.com/meta-pytorch/monarch) as its runtime.
Apptainer is needed only for containers, and only on the workers that run
them.

## Local development

```bash
python -m pip install -e ".[dev]"
python -c "import agentbox"
pytest
```

All dependencies come from PyPI, so any environment manager that provides
Python 3.12 works. With conda (or mamba/micromamba):

```bash
conda create -n agentbox python=3.12 pip
conda activate agentbox
python -m pip install -e ".[dev]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Use the same environment (or a binary-compatible `torchmonarch` build)
everywhere a distributed run imports the package: [Slurm](https://slurm.schedmd.com/overview.html) workers run the
coordinator's Python interpreter by default (`create_slurm(python_exe=...)`
overrides it), so the environment must live on storage the compute nodes can
see. Mismatched Monarch builds fail before application code starts.

## Apptainer

Install `apptainer` from your distribution or the
[Apptainer releases](https://github.com/apptainer/apptainer/releases) on every
host that runs containers. No root daemon is needed. Verify with
`apptainer version`, then run
[`examples/container_demo.py`](../examples/container_demo.py).

## macOS

Local workers run natively. Apptainer is Linux-only, so containers run in a
[Lima](https://lima-vm.io/) VM:

```bash
brew install lima
# AgentBox runs `apptainer`; Lima installs only `apptainer.lima`.
ln -s "$(brew --prefix)/bin/apptainer.lima" "$(brew --prefix)/bin/apptainer"
```

The wrapper runs Apptainer in the Lima VM named `apptainer` (override with
`LIMA_INSTANCE`). That VM must have Apptainer installed and mount
`~/.agentbox` writable: the package stages files there (see
`ensure_short_tmpdir` in [`env.py`](../src/agentbox/env.py)), and Lima
mounts your home read-only by default. If you don't have one, create it:

```bash
mkdir -p ~/.agentbox
limactl start --name=apptainer \
  --set '.mounts += [{"location": "~/.agentbox", "writable": true}]' \
  template:apptainer
apptainer exec docker://alpine echo ok
```

## Development checks

[`CONTRIBUTING.md`](../CONTRIBUTING.md) and
[the CI workflow](../.github/workflows/ci.yml) list the formatting, linting,
type-checking, and test commands.
