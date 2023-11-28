# Changelog

_This project uses semantic versioning_

## Unreleased

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
