# Contributing to AgentBox

We want to make contributing to this project as easy and transparent as
possible.

## Our development process

AgentBox is developed in the open on GitHub; `main` is the source of truth.
Changes land through reviewed pull requests that pass CI.

## Pull requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Run the validation commands below.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Use check-mode commands for pre-merge validation:

```bash
ufmt check .
ruff check .
bash scripts/typecheck.sh
pytest
```

Apply automatic formatting and safe Ruff fixes separately:

```bash
ufmt format .
ruff check --fix .
```

Container smoke tests need Apptainer and run in CI; locally use
`python examples/ci/ci_basic_smoketest.py --local`.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License

By contributing to AgentBox, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
