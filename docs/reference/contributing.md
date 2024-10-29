# Contributing

## Development

This package is in active development and welcomes contributions!

Feel free to bring up any questions/comments on [the Zulip chat](https://egraphs.zulipchat.com/) or open an issue.

All feedback is welcome and encouraged, it's great to hear about anything that works well or could be improved.

### Getting Started

To get started locally developing with this project, fork it and clone it to your local machine.

Using [the Github CLI](https://github.com/cli/cli#installation) this would be:

```bash
brew install gh
gh repo fork egraphs-good/egglog-python --clone
cd egglog-python
```

Then [install Rust](https://www.rust-lang.org/tools/install) and get a Python environment set up with a compatible version. Using [uv](https://docs.astral.sh/uv/getting-started/installation/) this would be:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the package in editable mode with the development dependencies:

```bash
uv sync --all-extras
```

Anytime you change the rust code, you can run `uv sync --reinstall-package egglog --all-extras` to force recompiling the rust code.

If you would like to download a new version of the visualizer source, run `make clean; make`. This will download
the most recent released version from the github actions artifact in the [egraph-visualizer](https://github.com/egraphs-good/egraph-visualizer) repo. It is checked in because it's a pain to get cargo to include only one git ignored file while ignoring the rest of the files that were ignored.

### Running Tests

To run the tests, you can use the `pytest` command:

```bash
uv run pytest
```

All code must pass ruff linters and formaters. This will be checked automatically by the pre-commit if you run `pre-commit install`.

To run it manually, you can use:

```bash
uv run pre-commit run --all-files ruff
```

If you make changes to the rust bindings, you can check that the stub files accurately reflect the rust code by running:

```bash
make stubtest
```

All code must all pass MyPy type checking. To run that locally use:

```bash
make mypy
```

Finally, to build the docs locally and test that they work, you can run:

```bash
make docs
```

## Making changes

All changes that impact users should be documented in the `docs/changelog.md` file. Please also add tests for any new features
or bug fixes.

When you are ready to submit your changes, please open a pull request. The CI will run the tests and check the code style.

## Documentation

We use the [Diátaxis framework](https://diataxis.fr/) to organize our documentation. The "explanation" section has
been renamed to "Blog" since most of the content there is more like a blog post than a reference manual. It uses
the [ABlog](https://ablog.readthedocs.io/en/stable/index.html#how-it-works) extension.

## Governance

The governance is currently informal, with Saul Shanabrook as the lead maintainer. If the project grows and there
are more contributors, we will formalize the governance structure in a way to allow it to be multi-stakeholder and
to spread out the power and responsibility.
