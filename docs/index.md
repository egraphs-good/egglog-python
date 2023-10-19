---
file_format: mystnb
---

# [`egglog`](https://github.com/egraphs-good/egglog/) Python

`egglog` is a Python package that provides bindings to the Rust library of the same name,
allowing you to use e-graphs in Python for optimization, symbolic computation, and analysis.

```shell
pip install egglog
```

_This follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) in terms of what Python versions are supported_

```{code-cell} python
from __future__ import annotations
from egglog import *

egraph = EGraph()

@egraph.class_
class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num:  ...

    def __add__(self, other: Num) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...

expr1 = egraph.let("expr1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("expr2", Num(6) + Num(2) * Num.var("x"))

@egraph.register
def _num_rule(a: Num, b: Num, c: Num, i: i64, j: i64):
    yield rewrite(a + b).to(b + a)
    yield rewrite(a * (b + c)).to((a * b) + (a * c))
    yield rewrite(Num(i) + Num(j)).to(Num(i + j))
    yield rewrite(Num(i) * Num(j)).to(Num(i * j))

egraph.saturate()
```

```{code-cell} python
egraph.check(eq(expr1).to(expr2))
egraph.extract(expr1)
```

## Status of this project

This package is in development and is not ready for production use. The upstream `egglog` package itself
is also subject to changes and is less stable than [`egg`](https://github.com/egraphs-good/egg).

~~If you are looking for a more stable e-graphs library in Python, you can try [`snake-egg`](https://github.com/egraphs-good/snake-egg), which wraps `egg`.~~ At this point, this library is more actively developed than the `snake-egg` Python bindings.

`egglog` is a rewrite of the `egg` library to use [relational e-matching](https://arxiv.org/abs/2108.02290) and to add datalog features.
See the ["Better Together: Unifying Datalog and Equality Saturation"](https://arxiv.org/abs/2304.04332) paper for more details

> We present egglog, a fixpoint reasoning system that unifies Datalog and equality saturation (EqSat). Like Datalog, it supports efficient incremental execution, cooperating analyses, and lattice-based reasoning. Like EqSat, it supports term rewriting, efficient congruence closure, and extraction of optimized terms.

## How documentation is organized

We use the [Diátaxis framework](https://diataxis.fr/) to organize our documentation.

```{toctree}
:maxdepth: 2
auto_examples/index
tutorials
how-to-guides
explanation
reference
```
