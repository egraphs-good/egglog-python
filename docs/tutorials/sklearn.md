---
file_format: mystnb
---

# Optimizing Scikit-Learn with Array API and Numba

In this tutorial, we will walk through increasing the performance of your scikit-learn code using `egglog` and [`numba`](https://numba.readthedocs.io/en/stable/user/5minguide.html).

One of the goals of `egglog` is to be used by other scientific computing libraries to create flexible APIs,
which conform to existing user expectations but allow a greater flexability in how they perform execution.

To work towards that we goal, we have built an prototype of a [Array API standard](https://data-apis.org/array-api/2022.12/index.html) conformant API
that can be used with [Scikit-Learn's experimental Array API support](https://scikit-learn.org/stable/modules/array_api.html),
to optimize it using Numba.

## Normal execution

We can create a test data set and use `LDA` to create a classification. Then we can run it on the dataset, to
return the estimated classification for out test data:

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

## Building our inputs

Now, we can try executing it with `egglog` instead. In this mode, we aren't actually passing in any particular
NDArray, but instead just using variables to represent the X and Y values.

These are defined in the `egglog.exp.array_api` module, as typed values:

```python
@array_api_module.class_
class NDArray(Expr):
    @array_api_module.method(cost=200)
    @classmethod
    def var(cls, name: StringLike) -> NDArray: ...

    @property
    def shape(self) -> TupleInt: ...

    ...

@array_api_module.function(mutates_first_arg=True)
def assume_shape(x: NDArray, shape: TupleInt) -> None: ...
```

We can use these functon to provides some metadata about the arguments as well:

```{code-cell} python
from copy import copy

from egglog.exp.array_api import NDArray, assume_dtype, assume_shape, assume_isfinite, assume_value_one_of

X_arr = NDArray.var("X")
X_orig = copy(X_arr)

assume_dtype(X_arr, X_np.dtype)
assume_shape(X_arr, X_np.shape)
assume_isfinite(X_arr)

y_arr = NDArray.var("y")
y_orig = copy(y_arr)

assume_dtype(y_arr, y_np.dtype)
assume_shape(y_arr, y_np.shape)
assume_value_one_of(y_arr, (0, 1))
```

While most of the execution can be deferred, every time sklearn triggers Python control flow (`if`, `for`, etc), we
need to execute eagerly and be able to give a definate value. For example, scikit-learn checks to makes sure that the
number of samples we pass in is greater than the number of unique classes:

```python
class LinearDiscriminantAnalysis(...):
    ...
    def fit(self, X, y):
        ...
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )
        ...
```

Without the assumptions above, we wouldn't know if the conditional is true or false. So we provide just enough information
for sklearn to finish executing and give us a result.

## Getting a result

```{code-cell} python
from egglog import EGraph
from egglog.exp.array_api import array_api_module

with EGraph([array_api_module]):
    X_r2 = run_lda(X_arr, y_arr)
X_r2
```

We have to instantiante an `EGraph` so that when we do reach eager evaluation, we will have something to run it on.

We now have extracted out a program which is semantically equivalent to the original call! One thing you might notice
is that the expression has more types than customary NumPy code. Every object is lifted into a strongly typed `egglog`
class. This is so that when we run optimizations, we know the types of all the objects. It still is compatible with
normal Python objects, but they are [converted](#type-promotion) when they are passed as argument.

## Optimizing our result

Now that we have the an expression, we can run our rewrite rules to "optimize" it, extracting out the lowest cost
(smallest) expression afterword:

```{code-cell} python
egraph = EGraph([array_api_module])
egraph.register(X_r2)
egraph.run(10000)
X_r2_optimized = egraph.extract(X_r2)
X_r2_optimized
```

We see that for example expressions that referenced the shape of our input arrays have been resolved to their
values.

We can also take a look at the e-graph itself, even though it's quite large, where we can see that equivalent
expressions show up in the same group, or "e-class":

```{code-cell} python
egraph.display(n_inline_leaves=1, split_primitive_outputs=True)
```

## Translating for Numba

We are getting closer to a form we could translate back to Numba, but we have to make a few changes. Numba doesn't
support the `axis` keyword for `mean` or `std`, but it does support it for `sum`, so we have to translate all forms
from one to the other, with a rule like this (defined in `array_api_numba`):

```python
rewrite(std(x, axis)).to(sqrt(mean(square(abs(x - mean(x, axis, keepdims=TRUE))), axis)))
```

We can run those additional rewrites now to get a new extracted version

```{code-cell} python
from egglog.exp.array_api_numba import array_api_numba_module
egraph = EGraph([array_api_numba_module])
egraph.register(X_r2)
egraph.run(10000)
X_r2_optimized = egraph.extract(X_r2)
X_r2_optimized
```

## Compiling back to Python source

Now we finally have a version that we could run with Numba! However, this isn't in NumPy code. What Numba needs
is a function that uses `numpy`, not our typed dialect.

So we use another module that provides a translation of all our methods into Python strings. The rules in it look like this:

```python
# the sqrt of an array should use the `np.sqrt` function and be assigned to its own variable, so it can be reused
rewrite(ndarray_program(sqrt(x))).to((Program("np.sqrt(") + ndarray_program(x) + ")").assign())

# To compile a setitem call, we first compile the source, assign it to a variable, then add an assignment statement
mod_x = copy(x)
mod_x[idx] = y
assigned_x = ndarray_program(x).assign()
yield rewrite(ndarray_program(mod_x)).to(
    assigned_x.statement(assigned_x + "[" + index_key_program(idx) + "] = " + ndarray_program(y))
)
```

We pull in all those rewrite rules from the `array_api_program_gen` module, and run them to get back a real Python function:

```{code-cell} python
from egglog.exp.array_api_program_gen import ndarray_function_two, array_api_module_string

egraph = EGraph([array_api_module_string])
fn_program = ndarray_function_two(X_r2_optimized, X_orig, y_orig)
egraph.register(fn_program)
egraph.run(10000)
fn = egraph.load_object(egraph.extract(fn_program.py_object))
```

We can verify that the function gives the same result:

```{code-cell} python
import numpy as np
assert np.allclose(run_lda(X_np, y_np), fn(X_np, y_np))
```

We can also take a look at its generated source:

```{code-cell} python
import inspect
print(inspect.getsource(fn))
```

Although it isn't the prettiest, we can see that it has only emitted each expression once, for common subexpression
elimination, and preserves the "imperative" aspects of setitem.

## Compiling to Numba

Now we finally have a function we can run with numba, and dump the resulting LLVM:

```{code-cell} python
import numba
import os

os.environ['NUMBA_DUMP_OPTIMIZED'] = 'TRUE'

fn_numba = numba.njit(fn)

assert np.allclose(run_lda(X_np, y_np), fn_numba(X_np, y_np))
```

## Evalauting performance

Let's see if it actually made anything quicker! Let's run a number of trials for the original function, our
extracted version, and the optimized extracted version:

```{code-cell} python
import timeit
import pandas as pd

stmts = {
    "original": "run_lda(X_np, y_np)",
    "extracted": "fn(X_np, y_np)",
    "extracted-numba": "fn_numba(X_np, y_np)",
}
df = pd.DataFrame.from_dict(
    {name: timeit.repeat(stmt, globals=globals(), number=1, repeat=10) for name, stmt in stmts.items()}
)

df
```

```{code-cell} python
import seaborn as sns

df_melt = pd.melt(df, var_name="function", value_name="time")
sns.catplot(data=df_melt, x="function", y="time", kind="swarm")
```

We see that the numba version is in fact faster, and the other two are about the same. It isn't significantly faster through,
so we might want to run a profiler on the original function to see where most of the time is spent:

```{code-cell} python
%load_ext line_profiler
%lprun -f fn fn(X_np, y_np)
```

We see that most of the time is spent in the SVD funciton, which [wouldn't be improved much by numba](https://github.com/numba/numba/issues/2423)
since it is will call out to LAPACK, just like NumPy. The only savings would come from the other parts of the progarm,
which can be inlined into

## Conclusion

To recap, in this tutorial we:

1. Tried using a normal scikit-learn LDA function on some test data.
2. Built up an abstract array and called it with that instead
3. Optimized it and translated it to work with Numba
4. Compiled it to a standalone Python funciton, which was optimized with Numba
5. Verified that this improved our performance with this test data.

The implementation of the Array API provided here is experimental, and not complete, but at least serves to show it is
possible to build an API like that with `egglog`.
