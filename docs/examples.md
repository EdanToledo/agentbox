# Examples

The scripts under [`examples`](../examples) are executable documentation. Each
has a module docstring with usage and success criteria; run it with `--help`
for options.

| Example | Needs | Focus |
| --- | --- | --- |
| [`machine_demo.py`](../examples/machine_demo.py) | nothing | `run`/`arun`, `spawn` (pipes and PTY), `upload`/`download` on a local worker |
| [`tty_terminal.py`](../examples/tty_terminal.py) | nothing (Apptainer with `--image`) | Interactive or one-shot remote terminal over a PTY |
| [`container_demo.py`](../examples/container_demo.py) | Apptainer | Container commands, interactive shell, files, overlays, GPUs |

## Agent harnesses

[`examples/harnesses`](../examples/harnesses) runs complete agent CLIs inside a
container. Each script installs its CLI with the vendor's installer and, with
`--prompt`, runs one headless task. The Claude Code and Codex examples keep the
real provider key on the worker host in a LiteLLM proxy
([`host_llm_bridge.py`](../examples/harnesses/host_llm_bridge.py)); the
container only receives a per-run bridge credential. The Muse Code example
passes `META_API_KEY` to the `muse exec` process inside the container.

| Example | Needs | Harness |
| --- | --- | --- |
| [`claude_container.py`](../examples/harnesses/claude_container.py) | Apptainer, `ANTHROPIC_API_KEY` for `--prompt` | Anthropic's Claude Code |
| [`codex_container.py`](../examples/harnesses/codex_container.py) | Apptainer, `OPENAI_API_KEY` for `--prompt` | OpenAI's Codex CLI |
| [`muse_container.py`](../examples/harnesses/muse_container.py) | Apptainer, `META_API_KEY` for `--prompt` | Meta's [Muse Code](https://dev.meta.ai/) |

[`examples/ci`](../examples/ci) holds the container smoke tests that CI runs
against a cached SIF; pass `--local` to run them against
`docker://python:3.12-slim`.
