# Usage

## Installation

`egglog` supports CPython 3.12, 3.13, and 3.14, with beta support for
free-threaded CPython 3.14t. Python version support follows
[SPEC 0](https://scientific-python.org/specs/spec-0000/). The examples below
create an isolated environment first so that new installs do not depend on
packages already present on your machine.

With `pip`:

```shell
python3.14 -m venv .venv  # or any supported Python 3.12-3.14
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
uv venv --python 3.14 .venv  # or any supported Python 3.12-3.14
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

[Numba 0.63 and later](https://numba.readthedocs.io/en/0.64.0/reference/deprecation.html#deprecation-of-macos-x86-64-intel-platform-support)
no longer publish Intel macOS binaries. This affects `egglog[array]`, not the
base `egglog` installation.

From a source checkout for development, use the repo's uv workflow:

```shell
uv sync --all-extras
uv run python -c "from egglog import EGraph; EGraph(); print('egglog ok')"
```

## Parallelism and threads

Configure Rust worker threads per e-graph with `num_threads`. The default of `1`
keeps execution serial; `0` uses the machine's available parallelism. You can
change the setting later with `set_num_threads` and inspect it with
`num_threads`.

```python
from egglog import EGraph

egraph = EGraph(num_threads=4)
egraph.set_num_threads(0)
assert egraph.num_threads() >= 1
```

For concurrent use from Python threads, see
[thread safety](python-integration.md#thread-safety).

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
