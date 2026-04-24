# `param_eq` Pipeline Design Notes

This file records the accepted baseline invariants that are easy to lose when
editing [`pipeline.py`](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq/pipeline.py).

## Baseline Constant Analysis

- The retained baseline uses `const_value(num) -> f64` as a partial lookup.
- Undefined `const_value(...)` means the expression has no known constant value.
- The merge function is `collapse_floats_with_tol(old, new, tol)`.
- Constant folding rules match literal `Num(...)` structure directly and subsume
  the folded composite expression.
- Rewrite rules may assume the analysis pass runs between outer pipeline
  iterations, but not between every match inside one rewrite batch.

## Container Work

The container path remains an experimental variant and is not part of the
baseline cleanup gate. Keep container-specific tools, tests, and comparison
artifacts out of baseline-focused changes unless they are needed for shared
artifact schema support.

## Artifacts

The retained package now uses three layers:

- vendored raw paper sources in
  [`artifacts/original`](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq/artifacts/original)
- cleaned source joins from
  [`original_results.py`](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq/original_results.py)
- minimal generated result CSVs from
  [`live_results.py`](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq/live_results.py)
  and
  [`egglog_results.py`](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq/egglog_results.py)

Consumers should join in memory through those modules rather than rebuilding a
denormalized paper artifact.
