---
file_format: mystnb
---

# `egglog` Python

`egglog` is a Python package that provides bindings to [the Rust library of the same name](https://github.com/egraphs-good/egglog/),
allowing you to use e-graphs in Python for optimization, symbolic computation, and analysis.

It wraps the Rust library [`egglog`](https://github.com/egraphs-good/egglog) which
is a rewrite of the `egg` library to use [relational e-matching](https://arxiv.org/abs/2108.02290) and to add datalog features.
See the ["Better Together: Unifying Datalog and Equality Saturation"](https://arxiv.org/abs/2304.04332) paper for more details

> We present egglog, a fixpoint reasoning system that unifies Datalog and equality saturation (EqSat). Like Datalog, it supports efficient incremental execution, cooperating analyses, and lattice-based reasoning. Like EqSat, it supports term rewriting, efficient congruence closure, and extraction of optimized terms.

```shell
pip install egglog
```

```{code-cell} python
from __future__ import annotations
from egglog import *


class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num:  ...

    def __add__(self, other: Num) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...

egraph = EGraph()

expr1 = egraph.let("expr1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("expr2", Num(6) + Num(2) * Num.var("x"))

@egraph.register
def _num_rule(a: Num, b: Num, c: Num, i: i64, j: i64):
    yield rewrite(a + b).to(b + a)
    yield rewrite(a * (b + c)).to((a * b) + (a * c))
    yield rewrite(Num(i) + Num(j)).to(Num(i + j))
    yield rewrite(Num(i) * Num(j)).to(Num(i * j))

egraph.saturate()
egraph.check(expr1 == expr2)
egraph.extract(expr1)
```

```{toctree}
:maxdepth: 2
tutorials
how-to-guides
explanation
reference
```
