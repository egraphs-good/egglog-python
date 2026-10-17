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

## Multisets

`MultiSet[T]` retains repeated elements. Use symbolic elements in its constructor
and an explicit type for an empty multiset:

```python
values = MultiSet(i64(2), i64(1), i64(2))
empty = MultiSet[i64]()
assert values.value == [i64(2), i64(1), i64(2)]
```

`.value` returns a new list of owned expression children; iteration, `len` and
membership also inspect occurrences without running an engine. Symbolic children
stay symbolic. A computed operation needs extraction before `.value` is
available. Native multiset iteration order is not an ordering guarantee.
`MultiSetLike[T, HostT]` accepts a multiset or a tuple, including the empty tuple;
conversion uses the requested element type. Lists and sets are not multiset
conversion inputs.

The generated ordinary API includes `single`, `pick`, `pick_max`, `insert`,
`remove`, `length`, `count`, `contains`, `not_contains`, `reset_counts`, nested
`sum_multisets`, and `+`, `-`, `&`. Four free views preserve native input order:
`multiset_remove_swapped`, `multiset_subtract_swapped`,
`multiset_contains_swapped`, and `multiset_not_contains_swapped`. In particular,
`multiset_subtract_swapped(x, xs)` computes **xs minus x**. Missing multiplicity,
negative counts and picking an empty multiset retain their native failures.

`union_values()` is action-only and requires nominal equality-sort elements;
primitive or container elements do not qualify. Its write restriction is checked
at runtime, not expressed by the generic Python type bound. The seven
higher-order operations (map, filter, filter-not, reduce, flat-map, fill-index
and clear-index) are not available in this generated API yet.

```{eval-rst}
.. automodule:: egglog
   :members:
   :imported-members:
   :private-members:
   :show-inheritance:
```
