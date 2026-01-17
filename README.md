# egglog Python

[![Documentation Status](https://readthedocs.org/projects/egglog-python/badge/?version=latest)](https://egglog-python.readthedocs.io/latest/?badge=latest) [![Test](https://github.com/egraphs-good/egglog-python/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/egraphs-good/egglog-python/actions/workflows/CI.yml) [![PyPi Package](https://img.shields.io/pypi/v/egglog.svg)](https://pypi.org/project/egglog/) [![License](https://img.shields.io/pypi/l/egglog.svg)](https://pypi.org/project/egglog/) [![Python Versions](https://img.shields.io/pypi/pyversions/egglog.svg)](https://pypi.org/project/egglog/) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/egraphs-good/egglog-python)

**egglog** is a Python package that provides bindings to the Rust library [`egglog`](https://github.com/egraphs-good/egglog/), allowing you to use **e-graphs** in Python for optimization, symbolic computation, and program analysis.

## What are e-graphs?

E-graphs (equality graphs) are data structures that efficiently represent equivalence classes of expressions. They enable powerful program optimizations through **equality saturation** - a technique that finds optimal expressions by exploring many equivalent representations simultaneously.

The underlying [`egglog`](https://github.com/egraphs-good/egglog) Rust library combines:
- **Datalog**: Efficient incremental reasoning and queries
- **Equality Saturation**: Term rewriting and optimization
- **E-graphs**: Compact representation of equivalent expressions

See the paper ["Better Together: Unifying Datalog and Equality Saturation"](https://arxiv.org/abs/2304.04332) for details.

## Installation

```shell
pip install egglog
```

Requires Python 3.11+ and works on Linux, macOS, and Windows.

## Quick Example

Here's how to use egglog to prove that `2 * (x + 3)` is equivalent to `6 + 2 * x` through algebraic rewriting:

```{code-cell} python
from __future__ import annotations
from egglog import *


class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __add__(self, other: Num) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...


egraph = EGraph()

# Create two expressions
expr1 = egraph.let("expr1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("expr2", Num(6) + Num(2) * Num.var("x"))


# Define rewrite rules for algebraic simplification
@egraph.register
def _num_rules(a: Num, b: Num, c: Num, i: i64, j: i64):
    yield rewrite(a + b).to(b + a)                    # Commutative
    yield rewrite(a * (b + c)).to((a * b) + (a * c))  # Distributive
    yield rewrite(Num(i) + Num(j)).to(Num(i + j))     # Constant folding
    yield rewrite(Num(i) * Num(j)).to(Num(i * j))


# Run equality saturation
egraph.saturate()

# Prove the expressions are equivalent
egraph.check(expr1 == expr2)

# Extract the simplified form
egraph.extract(expr1)
```

## Features

- **Pythonic API**: Natural Python syntax with type hints and decorators
- **High Performance**: Powered by Rust and the battle-tested egglog library
- **Type Safe**: Full type annotations and mypy support
- **Datalog Integration**: Combine e-graphs with relational queries
- **Rich Ecosystem**: Jupyter integration, visualization tools, examples
- **Well Documented**: Comprehensive tutorials, guides, and API reference

## Use Cases

egglog-python is useful for:
- **Compiler optimizations**: Optimize IR or DSL programs
- **Symbolic mathematics**: Simplify and manipulate mathematical expressions
- **Program synthesis**: Generate optimal programs from specifications
- **Query optimization**: Optimize database queries or data transformations
- **Theorem proving**: Automated reasoning with equality

## Documentation

**[📚 Full Documentation](https://egglog-python.readthedocs.io/)** - Tutorials, guides, and API reference

Key sections:
- [Tutorials](https://egglog-python.readthedocs.io/latest/tutorials/) - Step-by-step guides
- [How-to Guides](https://egglog-python.readthedocs.io/latest/how-to-guides/) - Task-oriented recipes
- [API Reference](https://egglog-python.readthedocs.io/latest/reference/) - Complete API documentation
- [Examples](https://egglog-python.readthedocs.io/latest/tutorials/examples/) - Real-world usage examples

## Contributing

Contributions are welcome! Whether you want to:
- Report a bug or request a feature
- Improve documentation
- Add examples
- Contribute code

See **[docs/reference/contributing.md](docs/reference/contributing.md)** for:
- Development setup and environment
- Running tests, linters, and type checkers
- Code standards and architecture overview
- Common patterns and troubleshooting

## Community

- **[💬 Zulip Chat](https://egraphs.zulipchat.com/)** - Join the e-graphs community
- **[🐛 Issues](https://github.com/egraphs-good/egglog-python/issues)** - Report bugs or request features
- **[📖 Changelog](https://egglog-python.readthedocs.io/latest/reference/changelog.html)** - See what's new

## Citation

If you use **egglog-python** in academic work, please cite:

```bibtex
@misc{Shanabrook2023EgglogPython,
  title         = {Egglog Python: A Pythonic Library for E-graphs},
  author        = {Saul Shanabrook},
  year          = {2023},
  eprint        = {2305.04311},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PL},
  doi           = {10.48550/arXiv.2305.04311},
  url           = {https://arxiv.org/abs/2305.04311},
  note          = {Presented at EGRAPHS@PLDI 2023}
}
```

## License

MIT License - see LICENSE file for details.
