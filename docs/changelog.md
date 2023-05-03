# Changelog

## Unreleased

## 0.4.0 (2023-05-03)

- Change name to `egglog` from `egg-smol`, to mirror [upstream change](https://github.com/egraphs-good/egglog/commit/9484242f025f6c2adb6f29d75a45dd77b0eaad57). Note that all previous versions are published under the `egg-smol` PyPi package while this and later are under `egglog`.

## 0.3.1 (2023-05-02)

- Fix bug calling methods on paramterized types (e.g. `Map[i64, i64].empty().insert(i64(0), i64(1))`)
- Fix bug for Unit type (egg name is `Unit` not `unit`)
- Use `@class_` decorator to force subclassing `BaseExpr`
- Workaround extracting definitions until [upstream is fixed](https://github.com/egraphs-good/egglog/pull/140)
- Rename `Map.map_remove` to `Map.remove`.
- Add lambda calculus example

## 0.3.0 (2023-04-26)

- [Upgrade `egglog` from `08a6e8fecdb77e6ba72a1b1d9ff4aff33229912c` to `6f2633a5fa379487fb389b80fc1225866f8b8c1a`.](https://github.com/metadsl/egglog-python/pull/14)

## 0.2.0 (2023-03-27)

This release adds support for a high level API for e-graphs.

There is an examples of the high level API in the [tutorials](tutorials/getting-started).
