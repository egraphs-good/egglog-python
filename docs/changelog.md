# Changelog

_This project uses semantic versioning_

## UNRELEASED

## 10.0.2 (2025-06-22)

- Fix using `f64Like` when not importing star (also properly includes removal of `Callable` special case from previous release).
- Fix Python 3.10 compatibility

## 10.0.1 (2025-04-06)

- Fix bug on resolving types if not all imported to your module [#286](https://github.com/egraphs-good/egglog-python/pull/286)
  - Also stops special casing including `Callable` as a global. So if you previously included this in a `TYPE_CHECKING` block so it wasn't
    available at runtime you will have to move this to a runtime import if used in a type alias.

## 10.0.0 (2025-03-28)

- Change builtins to not evaluate values in egraph and changes facts to compare structural equality instead of using an egraph when converting to a boolean, removing magic context (`EGraph.current` and `Schedule.current`) that was added in release 9.0.0.
- Fix bug that improperly upcasted values for ==

## 9.0.1 (2025-03-20)

- Add missing i64.log2 method to the bindings

## 9.0.0 (2025-03-20)

### Evaluating Primitives

Previously, if you had an egglog primitive object like an `i64`, you would have to call
`egraph.eval(i)` to get back an `int`. Now you can just call `int(i)`. This will implicitly create an e-graph and use it to extract the int value of the expression. This also means you can use this to evaluate compound expressions, like `int(i64(1) + 10)`.

This is also supported for container types, like vecs and sets. You can also use the `.eval()` method on any primitive to get the Python object.

For example:

````python
>>> from egglog import *
>>> Vec(i64(1), i64(2))[0]
Vec(1, 2)[0]
>>> int(Vec(i64(1), i64(2))[0])
1
>>> list(Vec(i64(1), i64(2)))
[i64(1), i64(2)]
>>> Rational(1, 2).eval()
Fraction(1, 2)
>>>


You can also manually set the e-graph to use, instead of it having to create a new one, with the `egraph.set_current` context manager:

```python
>>> egraph = EGraph()
>>> x = egraph.let("x", i64(1))
>>> x + 2
x + 2
>>> (x + 2).eval()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/saul/p/egg-smol-python/python/egglog/builtins.py", line 134, in eval
    value = _extract_lit(self)
            ^^^^^^^^^^^^^^^^^^
  File "/Users/saul/p/egg-smol-python/python/egglog/builtins.py", line 1031, in _extract_lit
    report = (EGraph.current or EGraph())._run_extract(cast("RuntimeExpr", e), 0)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/saul/p/egg-smol-python/python/egglog/egraph.py", line 1047, in _run_extract
    self._egraph.run_program(
EggSmolError: At 0:0 of
Unbound symbol %x
When running commands:
(extract (+ %x 2) 0)

Extracting: (+ %x 2)
>>> with egraph.set_current():
...     print((x + 2).eval())
...
3
````

There is a tradeoff here for more ease of use at the expense of some added implicit behavior using global state.

### Equal and Not Equal operators

Previously, if you wanted to create an equality fact, you would have to do `eq(x).to(y)`. And similarly, if you wanted to create a not equal expression, you would have to do `ne(x).to(y)`. I had set it up this way so that you could implement `__eq__` and `__ne__` on your custom data types to provide other functions (for monkeytyping purposes) and still be able to use the equality operators from egglog.

However, ergonmically this was a bit painful, so with this release, the `==` and `!=` methods are now supported on all egglog expressions. If you override them for your types, you can still use the functions:

```python
>>> i64(1) == i64(2)
eq(i64(1)).to(i64(2))
>>> i64(1) != i64(2)
ne(i64(1)).to(i64(2))
>>> class A(Expr):
...     def __init__(self) -> None: ...
...     def __eq__(self, other: "A") -> "A": ...
...
>>> A() == A()
A() == A()
>>> eq(A()).to(A())
eq(A()).to(A())
```

### Evaluating Facts

Similar to the above with primitives, you can now evaluate facts and see if they are true. This will implicitly create and run them on the e-graph. For example:

```python
>>> i64(10) == i64(9) + 1
eq(i64(10)).to(i64(9) + 1)
>>> bool(i64(10) == i64(9) + 1)
True
```

Again, you can set the current e-graph with the context manager to use that instead:

```python
>>> egraph = EGraph()
>>> s = egraph.let("s", MultiSet(i64(1), 2))
>>> with egraph.set_current():
...     assert s.contains(1)
```

### Experimental Array API Support

This release also continues to experiment with a proof of concept array API implementation
that allows deferred optimization and analysis. It is still very much a work in progress, but open to collaboration and feedback. The goal is to see egglog might be possible to be used as an end to end array compiler.

The changes in this release allow more functional programming constructrs to be used to create arrays, and then allow their properties to be optimized. For example, we can create
a linalg function (in [an example inspired by Siu](https://gist.github.com/sklam/5e5737137d48d6e5b816d14a90076f1d)):

```python
from egglog.exp.array_api import *

@function(ruleset=array_api_ruleset, subsume=True)
def linalg_norm(X: NDArrayLike, axis: TupleIntLike) -> NDArray:
    X = cast(NDArray, X)
    return NDArray(
        X.shape.deselect(axis),
        X.dtype,
        lambda k: ndindex(X.shape.select(axis))
        .foldl_value(lambda carry, i: carry + ((x := X.index(i + k)).conj() * x).real(), init=0.0)
        .sqrt(),
    )
```

Then we are able to check the shape of the output based on the input:

```python
>>> X = constant("X", NDArray)
>>> assume_shape(X, (3, 2, 3, 4))
>>> res = linalg_norm(X, (0, 1))
>>> assert res.shape.eval() == (Int(3), Int(4))
```

As well as see the symbolic form of it's output:

```python
>>> i = constant("i", Int)
>>> j = constant("j", Int)
>>> idxed = res.index((i, j))
>>> EGraph().simplify(idxed, array_api_schedule)
(
    (
        (
            (
                (
                    (X.index(TupleInt.from_vec(Vec[Int](Int(0), Int(0), i, j))).conj() * X.index(TupleInt.from_vec(Vec[Int](Int(0), Int(0), i, j)))).real()
                    + (X.index(TupleInt.from_vec(Vec[Int](Int(0), Int(1), i, j))).conj() * X.index(TupleInt.from_vec(Vec[Int](Int(0), Int(1), i, j)))).real()
                )
                + (X.index(TupleInt.from_vec(Vec[Int](Int(1), Int(0), i, j))).conj() * X.index(TupleInt.from_vec(Vec[Int](Int(1), Int(0), i, j)))).real()
            )
            + (X.index(TupleInt.from_vec(Vec[Int](Int(1), Int(1), i, j))).conj() * X.index(TupleInt.from_vec(Vec[Int](Int(1), Int(1), i, j)))).real()
        )
        + (X.index(TupleInt.from_vec(Vec[Int](Int(2), Int(0), i, j))).conj() * X.index(TupleInt.from_vec(Vec[Int](Int(2), Int(0), i, j)))).real()
    )
    + (X.index(TupleInt.from_vec(Vec[Int](Int(2), Int(1), i, j))).conj() * X.index(TupleInt.from_vec(Vec[Int](Int(2), Int(1), i, j)))).real()
).sqrt()
```

### All changes

- Fix pretty printing of lambda functions
- Add support for subsuming rewrite generated by default function and method definitions
- Add better error message when using @function in class (thanks @shinawy)
- Add error method if `@method` decorator is in wrong place
- Subsumes lambda functions after replacing
- Add working loopnest test and rewrite array api suport to be more general
- Improve tracebacks on failing conversions.
- Use `add_note` for exception to add more context, instead of raising a new exception, to make it easier to debug.
- Add conversions from generic types to be supported at runtime and typing level (so can go from `(1, 2, 3)` to `TupleInt`)
- Open files with webbrowser instead of internal graphviz util for better support
- Add support for not visualizing when using `.saturate()` method [#254](https://github.com/egraphs-good/egglog-python/pull/254)
- Upgrade [egglog](https://github.com/egraphs-good/egglog/compare/b0d b06832264c9b22694bd3de2bdacd55bbe9e32...saulshanabrook:egg-smol:889ca7635368d7e382e16a93b2883aba82f1078f) [#258](https://github.com/egraphs-good/egglog-python/pull/258)
  - This includes a few big changes to the underlying bindings, which I won't go over in full detail here. See the [pyi diff](https://github.com/egraphs-good/egglog-python/pull/258/files#diff-f34a5dd5d6568cd258ed9f786e5abce03df5ee95d356ea9e1b1b39e3505e5d62) for all public changes.
  - Creates seperate parent classes for `BuiltinExpr` vs `Expr` (aka eqsort aka user defined expressions). This is to
    allow us statically to differentiate between the two, to be more precise about what behavior is allowed. For example,
    `union` can only take `Expr` and not `BuiltinExpr`.
  - Removes deprecated support for modules and building functions off of the e-egraph.
  - Updates function constructor to remove `default` and `on_merge`. You also can't set a `cost` when you use a `merge`
    function or return a primitive.
  - `eq` now only takes two args, instead of being able to compare any number of values.
- Removes `eval` method from `EGraph` and moves primitive evaluation to methods on each builtin and support `int(...)` type conversions on primitives. [#265](https://github.com/egraphs-good/egglog-python/pull/265)
- Change how to set global EGraph context with `with egraph.set_current()` and `EGraph.current` and add support for setting global schedule as well with `with schedule.set_current()` and `Schedule.current`. [#265](https://github.com/egraphs-good/egglog-python/pull/265)
- Adds support for using `==` and `!=` directly on values instead of `eq` and `ne` functions. [#265](https://github.com/egraphs-good/egglog-python/pull/265)
- Add multiset, bigint, and bigrat builtins

## 8.0.1 (2024-10-24)

- Upgrade dependencies including [egglog](https://github.com/egraphs-good/egglog/compare/saulshanabrook:egg-smol:a555b2f5e82c684442775cc1a5da94b71930113c...b0db06832264c9b22694bd3de2bdacd55bbe9e32)
- Fix bug with non glob star import
- Fix bug extracting functions

## 8.0.0 (2024-10-17)

- Adds ability to use anonymous functions where callables are needed. These are automatically transformed to egglog
  functions with default rewrites.
- Upgrade [egglog](https://github.com/egraphs-good/egglog/compare/fb4a9f114f9bb93154d6eff0dbab079b5cb4ebb6...saulshanabrook:egg-smol:a555b2f5e82c684442775cc1a5da94b71930113c)
  - Adds source annotations to expressions for tracebacks
  - Adds ability to inline other functions besides primitives in serialized output
- Adds `remove` and `set` methods to `Vec`
- Upgrades to use the new egraph-visualizer so we can have interactive visualizations

## 7.2.0 (2024-05-23)

- Adds ability to use function bodies as default rewrites ([#167](https://github.com/egraphs-good/egglog-python/pull/167))
- Fixed bug with creating empty maps and adding to maps ([#168](https://github.com/egraphs-good/egglog-python/pull/168))

## 7.1.0 (2024-05-03)

## New Feaatures

- Upgrade [egglog](https://github.com/egraphs-good/egglog/compare/0113af1d6476b75d4319591cc3d675f96a71cdc5...fb4a9f114f9bb93154d6eff0dbab079b5cb4ebb6) ([#143](https://github.com/egraphs-good/egglog-python/pull/143))
  - Adds `bindings.UnstableCombinedRulset` to commands
  - Adds `UnstableFn` sort
- Adds support for first class functions as values using Python's built in `Callable` syntax and `partial`.
- Adds way to combine ruleset with `r1 | r2` syntax or the experimental `unstable_combine_rulesets(*rs, name=None)` function.

## Minor improvements

- Fixes a bug where you could not write binary dunder methods (like `__add__`) that didn't have symetric arguments
- Use function name as ruleset name by default when creating ruleset from function
- Adds ability to refer to methods and property off of classes instead of only off of instances (i.e. `Math.__add__(x, y)`)

## 7.0.0 (2024-04-27)

- Defers adding rules in functions until they are used, so that you can use types that are not present yet.
- Removes ability to set custom default ruleset for egraph. Either just use the empty default ruleset or explicitly set it for every run
- Automatically mark Python builtin operators as preserved if they must return a real Python value
- Properly pretty print all items (rewrites, actions, exprs, etc) so that expressions are de-duplicated and state is handled correctly.
- Add automatic releases from github manual action

## 6.1.0 (2024-03-06)

- Upgrade [egglog](https://github.com/egraphs-good/egglog/compare/4cc011f6b48029dd72104a38a2ca0c7657846e0b...0113af1d6476b75d4319591cc3d675f96a71cdc5)
  - Adds subsume action
- Makes all objects besides EGraphs "sendable" aka threadsafe ([#129](https://github.com/egraphs-good/egglog-python/pull/129))

## 6.0.1 (2024-02-28)

- Upgrade dependencies, including [egglog](https://github.com/egraphs-good/egglog/compare/ceed816e9369570ffed9feeba157b19471dda70d...4cc011f6b48029dd72104a38a2ca0c7657846e0b)
- Fix bug where saturate wasn't properly getting translated.

## 6.0.0 (2024-02-06)

### Remove modules / Auto register functions/classes

You can now create classes and functions without an EGraph! They will be automatically registered on any EGraph if they
are used in any of the rules or commands. This means the methods to add functions/classes on an EGraph are deprecated and
moved to the top level module:

- `egraph.class_` -> Removed, simply subclass from `egglog.Expr`
- `egraph.method` -> `egglog.method`
- `egraph.function` -> `egglog.function`
- `egraph.relation` -> `egglog.relation`
- `egraph.ruleset` -> `egglog.Ruleset`
- `egraph.Module` -> Removed

The goal of this change is to remove the complexity of `Module`s and remove the need to think about what functions/classes
need to be registered for each `EGraph`.

In turn, if you want to collect a set of rules, you can do that with a ruleset. Whenever you now run a ruleset or schedule,
the ruleset will be automatically registered on the EGraph.

For backwards compatability, the existing methods and functions are preserved, to make this easier to adopt. They will
all now raise deprication warnings.

### Allow future type references in classes

Classes can now reference types that have not been defined yet, as long as they are defined before the class is used in a
rule or expression. For example:

```python
class A(Expr):
    def __init__(self, b: B) -> None: ...

class B(Expr):
    ...
```

### Top level commands

We can now simplify and check expressions without explicity making an EGraph:

```python
check(<fact>, [<schedule>], *[<actions>])
# is equivalent to
e = EGraph()
e.register(*<actions>)
e.run(<schedule>)
e.check(<fact>)

simplify(<expr>, [<schedule>])
# is equivalent to
EGraph().simplify(<expr>, [<schedule>])
```

## 5.0.0 (2024-01-16)

- Move egglog `!=` function to be called with `ne(x).to(y)` instead of `x != y` so that user defined expressions
  can

## 4.0.1 (2023-11-27)

- Fix keyword args for `__init__` methods (#96)[https://github.com/metadsl/egglog-python/pull/96].

## 4.0.0 (2023-11-24)

- Fix `as_egglog_string` proprety.
- Move `EGraph.eval_fn` to `py_eval_fn` since it doesn't need the `EGraph` anymore.

## 3.1.0 (2023-11-21)

- Update graphs to include more compact Python names of functions (#79)[https://github.com/metadsl/egglog-python/pull/79].
- Add `as_egglog_string` property to get egglog source from e-graph (#82)[https://github.com/metadsl/egglog-python/pull/82].
- Add `include_cost` flag to `egraph.extract` to return the integer cost as well as an expression (#86)[https://github.com/metadsl/egglog-python/pull/86].
- Automatically try converting arguments to `eq`, `rewrite`, `set_`, and `union` to the correct type (#84)[https://github.com/metadsl/egglog-python/pull/84].
- Update RTD name to new project name of `egglog-python` from `egg-smol-python` (#18)[https://github.com/egraphs-good/egglog-python/pull/18].
- Move project to egraphs-good org!

## 3.0.0 (2023-11-19)

Add support for outputing the serialization e-graph from the low level bindings. Note that this is not yet exposed a the high level yet.

This removes the existing to graphviz function on the EGraph low level binding and moves it to a method on the serialized EGraph.

See (#78)[https://github.com/egraphs-good/egglog-python/pull/78] for more details.

## 2.0.0 (2023-11-17)

## Simplify accessing primitives

Previously, there was no public way of turning an egglog primitive, i.e. `i64(10)`, into a Python primitive, i.e. `int(10)`. Now there is a `egraph.eval(...)` method which will evaluate a primitive expression and return a Python object.

We also change the `PyObject` primitive to behave similarly. Instead of calling `egraph.load_object(pyobj)` you can now call `egraph.eval(pyobj)` to get the underlying Python object. Also, to unify it with the other objects, you can create a `PyObject` by using the constructor instead of `egraph.save_object(pyobj)`.

## Bug fixes

- Properly expose `birewrite` at top level (#72)[https://github.com/egraphs-good/egglog-python/pull/72].
- Fix generation of graphviz interactive SVGs in docs.

## Enhancements

- Added PyData lighting talk and Portland state talk to [explanations](./explanation).
- Add experimental `jit` decorator to wrap all ndarray/numba functionality together.
- Switch to Ruff for linting

## 1.0.1 (2023-10-26)

- Adds youtube video to [presentation slides](./explanation/2023_07_presentation).

## 1.0.0 (2023-10-26)

### Breaking Changes

- Test on Python 3.9 - 3.11, stop testing on 3.8 to follow Scientific Python versioning policy
- Bump [egglog dep](https://github.com/egraphs-good/egglog/compare/45d05e727cceaab13413b4e51a60ee3be9fbf403...ceed816e9369570ffed9feeba157b19471dda70d)
  - Adds `Bool` builtin
  - Rename `PrintTable` command to `PrintFunction`
  - Change extract command back to taking an expression instead of a fact
  - Adds `numer` and `denom` functions to `Rational` sort.
  - Adds `terms_encoding` boolean flag for creating an EGraph
  - Allow print size command to be called with no args to print all sizes
  - Add `rebuild` method for sets, maps, and vecs.

### New Features

- Add ability to print egglog string of module with `.as_egglog_string`
- Add ability to visualize changes in egraph as it runs with `.saturate()`
- Add ability to make functions and module unextractable as well as increase the cost of a whole module.
- Convert reflected methods based on both types
- Allow specifying custom costs for conversions
- In `py_exec` make a temporary file with source for tracebacks
- Add an experimental Array API implementation with a scikit-learn test

## 0.7.0 (2023-10-04)

- Bump [egglog dep](https://github.com/egraphs-good/egglog/compare/4d67f262a6f27aa5cfb62a2cfc7df968959105df...45d05e727cceaab13413b4e51a60ee3be9fbf403)

### New Features

- Adds ability for custom user defined types in a union for proper static typing with conversions [#49](https://github.com/egraphs-good/egglog-python/pull/49)
- Adds `py_eval` function to `EGraph` as a helper to eval Python code. [#49](https://github.com/egraphs-good/egglog-python/pull/49)
- Adds on hover behavior for edges in graphviz SVG output to make them easier to trace [#49](https://github.com/egraphs-good/egglog-python/pull/49)
- Adds `egglog.exp.program_gen` module that will compile expressions into Python statements/functions [#49](https://github.com/egraphs-good/egglog-python/pull/49)
- Adds `py_exec` primitive function for executing Python code [#49](https://github.com/egraphs-good/egglog-python/pull/49)

### Bug fixes

- Clean up example in tutorial with demand based expression generation [#49](https://github.com/egraphs-good/egglog-python/pull/49)

## 0.6.0 (2023-09-20)

- Bump [egglog dep](https://github.com/egraphs-good/egglog/compare/c83fc750878755eb610a314da90f9273b3bfe25d...4d67f262a6f27aa5cfb62a2cfc7df968959105df)

### Breaking Changes

- Switches `RunReport` to include more granular timings

### New Features

- Add ability to pass `seminaive` flag to Egraph to replicate `--naive` CLI flag [#48](https://github.com/egraphs-good/egglog-python/pull/48)
- Add ability to inline leaves $n$ times instead of just once for visualization [#48](https://github.com/egraphs-good/egglog-python/pull/48)
- Add `Relation` and `PrintOverallStatistics` low level commands [#46](https://github.com/egraphs-good/egglog-python/pull/46)
- Adds `count-matches` and `replace` string commands [#46](https://github.com/egraphs-good/egglog-python/pull/46)

### Uncategorized

- Added initial supported for Python objects [#31](https://github.com/egraphs-good/egglog-python/pull/31)

  - Renamed `BaseExpr` to `Expr` for succinctness
  - Add [slides for zoom presentation with Open Teams](explanation/2023_07_presentation)
  - Started adding tutorial for using with array API and sklearn], using this to drive
    the support for more Python integration
  - Added a PyObject sort with the `save_object` and `load_object` egraphs methods and the `exec`
  - Added more general mechanism to upcast Python arguments into egglog expressions, by registering `converter`s
  - Added support for default arguments (this required refactoring declerations so that pretty printing can lookup expressions)
  - Added support for properties
  - Added support for passing args as keywords
  - Add support for pure Python methods, using the `preserve` kwarg to implement functions like `__bool__` on expressions.
  - Fix `__str__` method when pretty printing breaks.
  - Added to/from i64 to i64 methods.
  - Upgraded `egg-smol` dependency ([changes](https://github.com/saulshanabrook/egg-smol/compare/353c4387640019bd2066991ee0488dc6d5c54168...2ac80cb1162c61baef295d8e6d00351bfe84883f))

- Add support for functions which mutates their args, like `__setitem__` [#35](https://github.com/egraphs-good/egglog-python/pull/35)
- Makes conversions transitive [#38](https://github.com/egraphs-good/egglog-python/pull/38)
- Add support for reflective operators [#39](https://github.com/egraphs-good/egglog-python/pull/39)
  - Make reflective operators map directly to non-reflective [#40](https://github.com/egraphs-good/egglog-python/pull/40)
- Includes latest egglog changes [#42](https://github.com/egraphs-good/egglog-python/pull/42)
  - Switches to termdag introduced in [egglog #176](https://github.com/egraphs-good/egglog/pull/176)
  - Removes custom fork of egglog now that visualizations are in core
  - Adds int and float to string functions
  - Switches `define` to `let`
- Tidy up notebook appearence [#43](https://github.com/egraphs-good/egglog-python/pull/43)
  - Display expressions as code in Jupyter notebook
  - Display all expressions when graphing
- Start adding to string support [#45](https://github.com/egraphs-good/egglog-python/pull/45)
  - Fix adding rules for sorts defined in other modules
  - Split out array API into multiple module
  - tidy up docs homepage

## 0.5.1 (2023-07-18)

- Added support for negation on `f64` sort
- Upgraded `egg-smol` dependency ([changes](https://github.com/saulshanabrook/egg-smol/compare/4b7ec0a640b430bc86ec1d9f79e38a06e62c0cb7...353c4387640019bd2066991ee0488dc6d5c54168))

## 0.5.0 (2023-05-03)

- Renamed `config()` to `run()` to better match `egglog` command
- Fixed `relation` type signature
- Added default limit of 1 to `run()` to match `egglog` command and moved to second arg
- Upgraded `egglog` dependency ([changes](https://github.com/egraphs-good/egglog/compare/30feaaab88452ec4b6c5f7a199345298bac2dd0f...39b199d9bfce9cc47d0c54977279c5b04231e717))
  - Added `Set` sort and removed set method from `Map`
  - Added `Vec` sort
  - Added support for variable args for builtin functions, to use in creation of `Vec` and `Set` sorts.
  - Added suport for joining `String`s
- Switch generated egg names to use `.` as seperate (i.e. `Math.__add__`) instead of `_` (i.e. `Math___add__`)
- Adds support for modules to define functions/sorts/rules without executing them, for reuse in other modules
  - Moved simplifying and running rulesets to the `run` and `simplify` methods on the `EGraph` from those methods on the `Ruleset` since we can now create `Rulset`s for modules which don't have an EGraph attached and can't be run
- Fixed extracting classmethods which required generic args to cls
- Added support for alternative way of creating variables using functions
- Add NDarray example
- Render EGraphs with `graphviz` in the notebook (used in progress egglog PR).
  - Add images to doc examples
- Add `%%egglog` magic to the notebook

## 0.4.0 (2023-05-03)

- Change name to `egglog` from `egg-smol`, to mirror [upstream change](https://github.com/egraphs-good/egglog/commit/9484242f025f6c2adb6f29d75a45dd77b0eaad57). Note that all previous versions are published under the `egg-smol` PyPi package while this and later are under `egglog`.

## 0.3.1 (2023-05-02)

- Fix bug calling methods on paramterized types (e.g. `Map[i64, i64].empty().insert(i64(0), i64(1))`)
- Fix bug for Unit type (egg name is `Unit` not `unit`)
- Use `@class_` decorator to force subclassing `Expr`
- Workaround extracting definitions until [upstream is fixed](https://github.com/egraphs-good/egglog/pull/140)
- Rename `Map.map_remove` to `Map.remove`.
- Add lambda calculus example

## 0.3.0 (2023-04-26)

- [Upgrade `egglog` from `08a6e8fecdb77e6ba72a1b1d9ff4aff33229912c` to `6f2633a5fa379487fb389b80fc1225866f8b8c1a`.](https://github.com/egraphs-good/egglog-python/pull/14)

## 0.2.0 (2023-03-27)

This release adds support for a high level API for e-graphs.

There is an examples of the high level API in the [tutorials](tutorials/getting-started).
