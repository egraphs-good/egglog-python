# Usage

## Installation

You can install this package with `pip`:

```shell
pip install egglog
```

To be able to run the array demos:

```shell
pip install egglog[array]
```

To see interactive widgets:

```shell
pip install anywidget
```

It follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) in terms of what Python versions are supported.

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
