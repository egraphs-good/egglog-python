# Changelog

_This project uses semantic versioning. Before 1.0.0, this means that every breaking changes will result in a minor version bump. After 1.0.0, this means that every breaking change will result in a major version bump. We will move to 1.0.0, once we have any repository that depends on this project._

## Unreleased

- Added initial supported for Python objects [#31](https://github.com/metadsl/egglog-python/pull/31)

  - Renamed `BaseExpr` to `Expr` for succinctness
  - Add [slides for zoom presentation with Open Teams](explanation/2023_07_presentation)
  - Started adding [tutorial for using with array API and sklearn](tutorials/array-api), using this to drive
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

- Add support for functions which mutates their args, like `__setitem__` [#35](https://github.com/metadsl/egglog-python/pull/35)
- Makes conversions transitive [#38](https://github.com/metadsl/egglog-python/pull/38)
- Add support for reflective operators [#39](https://github.com/metadsl/egglog-python/pull/39)
  - Make reflective operators map directly to non-reflective [#40](https://github.com/metadsl/egglog-python/pull/40)

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

- [Upgrade `egglog` from `08a6e8fecdb77e6ba72a1b1d9ff4aff33229912c` to `6f2633a5fa379487fb389b80fc1225866f8b8c1a`.](https://github.com/metadsl/egglog-python/pull/14)

## 0.2.0 (2023-03-27)

This release adds support for a high level API for e-graphs.

There is an examples of the high level API in the [tutorials](tutorials/getting-started).
