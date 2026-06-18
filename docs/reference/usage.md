# Usage

## Installation

`egglog` supports Python 3.11 and newer. The examples below create an isolated
environment first so that new installs do not depend on packages already present
on your machine.

With `pip`:

```shell
python3.13 -m venv .venv  # or any supported Python 3.11+
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install egglog
python -c "from egglog import EGraph; EGraph(); print('egglog ok')"
```

With `uv` in a project:

```shell
uv init egglog-demo
cd egglog-demo
uv add egglog
uv run python -c "from egglog import EGraph; EGraph(); print('egglog ok')"
```

With `uv` in a standalone virtual environment:

```shell
uv venv --python 3.13 .venv  # or any supported Python 3.11+
uv pip install --python .venv/bin/python egglog
.venv/bin/python -c "from egglog import EGraph; EGraph(); print('egglog ok')"
```

If you already have an active environment, the install command is simply:

```shell
python -m pip install egglog
```

To run the array demos, install the optional array dependencies:

```shell
python -m pip install "egglog[array]"
uv add "egglog[array]"
```

From a source checkout for development, use the repo's uv workflow:

```shell
uv sync --all-extras
uv run python -c "from egglog import EGraph; EGraph(); print('egglog ok')"
```

It follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) in terms of what Python versions are supported.

## Parallelism and threads

The underlying Rust library uses Rayon for parallelism. You can control the worker thread count via the environment variable `RAYON_NUM_THREADS`. If this variable is not set or is invalid, the Python bindings default to using a single thread (`1`).

```shell
export RAYON_NUM_THREADS=4  # use 4 threads
```

(community)=

## Community

There is [a Zulip stream](https://egraphs.zulipchat.com/#narrow/stream/375765-egglog) for the `egglog` project
which you are welcome to open a thread on.

There are also [Github issues](https://github.com/egraphs-good/egglog-python/issues) and [discussions](https://github.com/egraphs-good/egglog-python/discussions)
which you can use to ask questions.

## Stability

This project is in active development and has not been used in a production setting yet.

The API is subject to change, but efforts will be made to preserve backwards compatibility at least with the
high level API.

However, since it is a wrapper around the Rust library [`egglog`](https://github.com/egraphs-good/egglog), any breaking
changes to that package that would affect the high level API would require a major version bump.
