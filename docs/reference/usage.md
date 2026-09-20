# Usage

## Installation

`egglog` supports GIL-enabled CPython 3.12, 3.13, and 3.14, and free-threaded
CPython 3.14t. Importing `egglog` does not re-enable the GIL on a free-threaded
build. The examples below create an isolated environment first so that new
installs do not depend on packages already present on your machine.

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

[Numba 0.63 and later](https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-macos-x86-64-intel-platform-support)
no longer publish binaries for Intel (`x86_64`) macOS. That upstream limitation
affects the optional `array` extra on Intel Macs, not the base `egglog`
installation.

From a source checkout for development, use the repo's uv workflow:

```shell
uv sync --all-extras
uv run python -c "from egglog import EGraph; EGraph(); print('egglog ok')"
```

The rolling Python support window follows
[SPEC 0](https://scientific-python.org/specs/spec-0000/), rather than retaining
security-only Python branches indefinitely.

## Parallelism and threads

Configure worker threads per e-graph with `num_threads`. The default of `1`
keeps execution serial; `0` uses the machine's available parallelism. You can
change the setting later with `set_num_threads` and inspect it with
`num_threads`.

```python
from egglog import EGraph

egraph = EGraph(num_threads=4)
egraph.set_num_threads(0)
assert egraph.num_threads() >= 1
```

The high-level Python API is not thread-safe, including on free-threaded
CPython. Use it from one thread, or protect **all** high-level egglog operations
with one caller-owned `threading.Lock`, even when using separate `EGraph`
instances. This includes expression construction, converter and rule
registration, and the entire duration of experimental ambient-egraph contexts.
Declarations, conversion tables, and expression caches are shared across
e-graphs.

Complete module initialization before starting workers, and return local DSL
objects from a setup helper before handing them to another thread. Lazy
annotation resolution must not inspect another thread's
[still-executing defining frame](https://docs.python.org/3.14/howto/free-threading-python.html#frame-objects).

This limitation does not prevent the Rust engine from using `num_threads`.
Python callbacks running on engine workers must not concurrently use the
high-level API, and callers must synchronize any shared mutable callback state.
Independent [low-level binding instances](bindings.md#thread-safety) can run
concurrently without a shared caller lock.

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
