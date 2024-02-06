---
file_format: mystnb
---

# High Level: `egglog`

The top level module contains the high level API for using e-graphs in Python.

The high level API is not documented yet, because adding supporting for our custom objects requires [a custom AutoDoc extension](https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html#autodoc-ext-tutorial).

The high level API builds on the low level API and is designed to:

1. Statically type check as much as possible with MyPy
2. Be concise to write
3. Feel "pythonic"

Here is the same example using the high level API:

```{code-cell} python
from __future__ import annotations

from egglog import *

egraph = EGraph()

@egraph.class_
class Math(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    @classmethod
    def var(cls, v: StringLike) -> Math:
        ...

    def __add__(self, other: Math) -> Math:
        ...

    def __mul__(self, other: Math) -> Math:
        ...

# expr1 = 2 * (x + 3)
expr1 = egraph.let("expr1", Math(2) * (Math.var("x") + Math(3)))

# expr2 = 6 + 2 * x
expr2 = egraph.let("expr2", Math(6) + Math(2) * Math.var("x"))

a, b, c = vars_("a b c", Math)
x, y = vars_("x y", i64)

egraph.register(
    rewrite(a + b).to(b + a),
    rewrite(a * (b + c)).to((a * b) + (a * c)),
    rewrite(Math(x) + Math(y)).to(Math(x + y)),
    rewrite(Math(x) * Math(y)).to(Math(x * y)),
)

egraph.run(10)

egraph.check(eq(expr1).to(expr2))
```

```{eval-rst}
.. automodule:: egglog
   :members:
   :imported-members:
   :private-members:
   :show-inheritance:
```
