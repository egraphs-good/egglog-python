---
file_format: mystnb
---

# PyData NYC '23

E-graphs in Python with **`egglog`**

_Saul Shanabrook_

+++ {"slideshow": {"slide_type": "slide"}}

## Aims:

- Faithful bindings for `egglog` rust library.
- "Pythonic" interface, using standard type definitions.
- Usable as base for optimizing/translating expressions for data science libraries in Python

+++ {"slideshow": {"slide_type": "slide"}}

## What is an e-graph?

> E-graphs are these super wonderful data structures for managing equality and equivalence information. They are traditionally **used inside of constraint solvers and automated theorem provers** to implement congruence closure, an efficient algorithm for equational reasoning---but they can also be used to implement **rewrite systems**.
>
> [Talia Ringer - "Proof Automation" course](https://dependenttyp.es/classes/readings/17-egraphs.html)

+++ {"slideshow": {"slide_type": "slide"}}

1. Define types and functions/operators
2. Define rewrite rules
3. Add expressions to graph
4. Run rewrite rules on expressions until saturated (addtional applications have no effect)
5. Extract out lowest cost expression

## Example

```{code-cell} python
---
slideshow: {"slide_type": "slide"}
---
from __future__ import annotations
from egglog import *

egraph = EGraph()

@egraph.class_
class NDArray(Expr):
    def __init__(self, i: i64Like) -> None: ...
    def __add__(self, other: NDArray) -> NDArray: ...
    def __mul__(self, other: NDArray) -> NDArray: ...

@egraph.function(cost=2)
def arange(i: i64Like) -> NDArray: ...

# Register rewrite rule that asserts for all values x of type NDArray
# x + x = x * 2
x = var("x", NDArray)
egraph.register(
    rewrite(
        x + x
    ).to(
        x * NDArray(2)
    )
)

```

```{code-cell} python
---
slideshow: {"slide_type": "slide"}
---
res = arange(10) + arange(10)
egraph.register(res)
egraph.saturate()
```

```{code-cell} python
egraph.extract(res)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Example with Scikit-learn

Optimize Scikit-learn function with Numba by building an e-graph that implements the Array API.

```{code-cell} python
from sklearn import config_context
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def run_lda(x, y):
    with config_context(array_api_dispatch=True):
        lda = LinearDiscriminantAnalysis()
        return lda.fit(x, y).transform(x)


X_np, y_np = make_classification(random_state=0, n_samples=1000000)
run_lda(X_np, y_np)
```

+++ {"slideshow": {"slide_type": "slide"}}

```{code-cell} python
from egglog.exp.array_api import *
from egglog.exp.array_api_jit import jit

@jit
def optimized_fn(X, y):
    # Add metadata about input shapes and dtypes, so that abstract array
    # can pass scikit-learn runtime checks
    assume_dtype(X, X_np.dtype)
    assume_shape(X, X_np.shape)
    assume_isfinite(X)

    assume_dtype(y, y_np.dtype)
    assume_shape(y, y_np.shape)
    assume_value_one_of(y, (0, 1))

    return run_lda(X, y)
```

Here is an example of a rewrite rule we used to generate Numba compatible code:

```python
rewrite(
    std(x, axis)
).to(
    sqrt(mean(square(x - mean(x, axis, keepdims=TRUE)), axis))
)
```

+++ {"slideshow": {"slide_type": "slide"}}

```{code-cell} python
# See generated code
import inspect
print(inspect.getsource(optimized_fn))
```

+++ {"slideshow": {"slide_type": "slide"}}

```{code-cell} python
import numba
import numpy as np

numba_fn = numba.njit(fastmath=True)(optimized_fn)
assert np.allclose(run_lda(X_np, y_np), numba_fn(X_np, y_np))
```

+++ {"slideshow": {"slide_type": "slide"}}

```{code-cell} python
---
tags: [remove-input]
---
import timeit
import pandas as pd

stmts = {
    "original": "run_lda(X_np, y_np)",
    "numba": "numba_fn(X_np, y_np)",
}
df = pd.DataFrame.from_dict(
    {
        name: timeit.repeat(stmt, globals=globals(), number=1, repeat=10)
        for name, stmt in stmts.items()
    }
)
```

+++ {"slideshow": {"slide_type": "slide"}}

# ~30% speedup

_on my machine, not a scientific benchmark_

```{code-cell} python
import seaborn as sns

df_melt = pd.melt(df, var_name="function", value_name="time")
_ = sns.catplot(data=df_melt, x="function", y="time", kind="swarm")
```

- Composable

## Conclusions

- `egglog` is a Python interface to e-graphs, which respects the underlying semantics but provides a Python interface.
- Flexible enough to represent Array API and translate this back to Python source
- If you have a Python library which optimizes/translates expressions, try it out!
- Goal: support the ecosystem in collaborating better between libraries, to encourage experimentation and innovation, connection to academic computer science community.

`pip install egglog`
