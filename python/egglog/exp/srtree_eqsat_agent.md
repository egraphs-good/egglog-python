# Srtree-EqSat Replication Handoff

This file is a continuation note for the Egglog replication work in
[srtree_eqsat.py](/Users/saul/p/egg-smol-python/python/egglog/exp/srtree_eqsat.py).

## Goal

Recreate the `srtree-eqsat` simplification pipeline from the de Franca and Kronberger 2023 paper inside Egglog, compare it against the Haskell source implementation, and then test a multiset-based replacement for the binary A/C-heavy parts of the rewrite system.

The current status is:
- the baseline Egglog replication is working on two representative `example_hl` rows
- the offline Haskell comparison path is working for those same rows
- the multiset hypothesis is partially implemented and instrumented
- the current multiset design does not yet remove the need for safety guards

## Important Paths

- Egglog engine:
  [srtree_eqsat.py](/Users/saul/p/egg-smol-python/python/egglog/exp/srtree_eqsat.py)
- Doc-script notebook:
  [2026_03_srtree_eqsat_replication.py](/Users/saul/p/egg-smol-python/docs/explanation/2026_03_srtree_eqsat_replication.py)
- Offline Haskell helper:
  [haskell_compare.hs](/Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/haskell_compare.hs)
- Small compare runner:
  [run_compare.py](/Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/run_compare.py)
- Experiment README:
  [README.md](/Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/README.md)
- Regression tests:
  [test_srtree_eqsat.py](/Users/saul/p/egg-smol-python/python/tests/test_srtree_eqsat.py)
- Source repo used for replication:
  `/Users/saul/p/srtree-eqsat`

## What Was Implemented

### 1. Python-facing DSL

The Egglog DSL uses Python operator overloads instead of named binary constructors:
- `+`
- `-`
- `*`
- `/`
- `**`
- unary `-`
- `abs(...)`

Named helpers are kept only for operators without Python syntax:
- `exp`
- `log`
- `sqrt`
- `cbrt`
- `sqr`
- `cube`

This is implemented in `Expr` in [srtree_eqsat.py](/Users/saul/p/egg-smol-python/python/egglog/exp/srtree_eqsat.py).

### 2. `hl` Parsing

`parse_hl_expr(source)` uses restricted Python `eval` with:
- `__builtins__ = {}`
- `alpha`, `beta`, `theta` bound to Egglog vars
- `sqr`, `cube`, `cbrt`, `exp`, `log`, `sqrt`, `abs` bound to the Egglog wrappers

This works for the source repo's `test/example_hl` syntax.

### 3. Baseline Pipeline

The baseline tries to match the Haskell pipeline in `srtree-eqsat/src/Data/SRTree/EqSat.hs`.

Implemented rule families:
- `rewritesBasic`
- `constReduction`
- `constFusion`
- `rewritesFun`

Implemented schedule shape:
- `rewriteConst = constReduction` plus constant-analysis support
- `rewriteAll = rewritesBasic + constReduction + constFusion + rewritesFun` plus constant-analysis support
- const pass once
- then full pass, extract, rebuild, and repeat up to two times

Important: the baseline uses backoff because the Haskell source does.
- const pass: backoff `(100, 10)`
- full pass: backoff `(2500, 30)`

The baseline runner is:
- `run_baseline_pipeline(expr, node_cutoff, iteration_limit, ...)`

### 4. Cost Model

The Egglog extraction cost matches the Haskell source:
- `Const = 5`
- `Param = 5`
- `Var = 1`
- unary op adds `1`
- binary op adds `1`

### 5. Metrics

The module reports:
- runtime
- total function size via `sum(size for _, size in egraph.all_function_sizes())`
- serialized node count
- serialized e-class count
- parameter counts before/after
- reduction ratio
- numeric Jacobian rank
- Jacobian-rank gap
- stop reason
- sampled max absolute numeric error

Parameter counting uses paper-style `floatConstsToParam` semantics:
- non-integer float literals become parameters
- integers stay constants
- parameters are relabeled left-to-right

### 6. Generated Python Evaluation

The module compiles extracted expressions to Python via Egglog `Program` generation where possible:
- `expr_program`
- `program_gen_ruleset`

If the extracted term still contains multiset wrappers, it falls back to a manual Python renderer.

This is used for:
- numeric equivalence checks
- Jacobian-rank estimation

### 7. Cutoff Driver

Each stage is run one scheduled step at a time through `egraph.run(...)`, not `egraph.saturate(...)`.

The stage driver checks after every step:
- `run_report.updated`
- total function size
- serialized node/e-class counts

Reported stop reasons:
- `saturated`
- `cutoff_hit`
- `budget_hit`

## Chosen Examples

The current pass intentionally stays on two `test/example_hl` rows.

### Row 1

Small sanity case.

Source:

```python
sqr(-9.29438919215253 + 2.93547417364396 * theta)
```

### Row 50

Function-heavy representative case.

Source:

```python
(exp(0.743694003014863 * alpha) * (-0.0121179632900701 * theta + 0.00904122619609017 * alpha) * (-3.05659895630567 * theta + 8.63005732191704) + -0.557193153898209 * alpha - log(0.782997897866162 * theta) + sqr(exp(-0.144728813168975 * theta)) * (-1.54770141702422 + -3.31046821812388 * theta) + 6.34043434659957 * beta * 0.643712432648199) * -0.0413897531650583 + 0.530747148732844
```

The row selection is encoded in `core_examples()`.

## Haskell Comparison: What Works and What Does Not

### What works

The offline helper now uses the source repo's exported `simplifyEqSat` directly.

That gives stable:
- runtime
- parameter counts before/after
- input/output tree node counts
- final simplified expression

### What does not work

The source repo's public API does not expose:
- intermediate e-graph sizes
- final memo size
- final e-class count
- internal stop reason

I tried copying the old source rule driver to recover those details, but forcing the row 50 `rewriteAll` e-graph internals hit:

```text
IntMap.!: key 200 is not an element of the map
```

That failure occurred while introspecting the copied old Haskell e-graph state, not while calling the source repo's exported `simplifyEqSat`.

Because of that, the current comparison data in `HASKELL_REFERENCE_ROWS` intentionally leaves:
- `memo_size = -1`
- `eclass_count = -1`

and the notebook explicitly says those values are unavailable from the exported API path.

## Embedded Haskell Reference Rows

These live in `HASKELL_REFERENCE_ROWS` in [srtree_eqsat.py](/Users/saul/p/egg-smol-python/python/egglog/exp/srtree_eqsat.py).

### Row 1

- runtime: `0.001351s`
- parameters: `2 -> 2`
- tree nodes: `7 -> 7`
- simplified:

```python
(-9.29438919215253 + (2.93547417364396 * x[:, 2])) ** 2.0
```

### Row 50

- runtime: `0.547641s`
- parameters: `14 -> 12`
- tree nodes: `60 -> 46`
- simplified:

```python
(-4.13897531650583e-2 * ((np.exp((x[:, 2] * -0.144728813168975)) ** 2.0) * (-1.54770141702422 - (x[:, 2] * 3.31046821812388)) + ((x[:, 1] * 4.081416417295803) + (((((np.exp((0.743694003014863 * x[:, 0])) * ((x[:, 0] * 9.04122619609017e-3) - (x[:, 2] * 1.21179632900701e-2))) * (8.63005732191704 - (x[:, 2] * 3.05659895630567))) - (x[:, 0] * 0.557193153898209)) - np.log(x[:, 2])) + -12.578528004457953))))
```

## Baseline Egglog Results

These were verified live from the module.

### Row 1 baseline

- stop: `saturated`
- total size: `25`
- e-graph nodes: `30`
- e-classes: `19`
- cost: `23`
- parameters: `2 -> 2`
- reduction ratio: `0.0`
- Jacobian rank gap: `0`
- numeric max error: `0.0`

### Row 50 baseline

- stop: `saturated`
- total size: `178`
- e-graph nodes: `197`
- e-classes: `98`
- cost: `127`
- parameters: `14 -> 13`
- reduction ratio: `0.07142857142857142`
- Jacobian rank gap: `0`
- numeric max error: `1.1102230246251565e-16`

Important comparison:
- row 1 matches the Haskell parameter result
- row 50 is close but not exact: Egglog gets `14 -> 13`, Haskell gets `14 -> 12`

## Likely Causes of the Row 50 Baseline Gap

The most likely reasons the baseline still differs from Haskell on row 50 are:

1. Weaker nonlinear constant analysis in the Egglog replication.
   The Haskell `Analysis (Maybe Double)` path can fold constants through the full `evalFun` route. The Egglog replication has constant analysis for the algebraic core, but not a full equivalent for nonlinear builtin `f64` function evaluation inside the e-graph.

2. Extraction tie-break differences.
   Even with the same apparent cost model, Egglog and the old Haskell implementation can still choose different representatives.

3. Hidden Haskell intermediate state.
   Because the public Haskell API does not expose the intermediate e-graph and my copied introspection path crashed on row 50, there is less visibility into which exact merge or extraction path caused the final difference.

## Multiset Pipeline Design

The multiset experiment introduces:
- `sum_(MultiSet[Expr])`
- `product_(MultiSet[Expr])`

Current multiset pipeline:

1. lower binary additive/multiplicative islands into multisets
2. simplify in multiset form
3. reify multisets back to binary form and run cleanup rules

This uses fresh e-graphs between stages.

The current multiset runner is:
- `run_multiset_pipeline(expr, saturate_without_limits=True|False, node_cutoff=..., iteration_limit=...)`

Important current behavior:
- the multiset stages do not use backoff
- the bounded runs use explicit cutoffs/budgets
- the doc currently shows the bounded runs, because the unrestricted representative case is not practical yet

## Multiset Results

### Row 1 multiset

Bounded run:
- final stop: `budget_hit`
- final total size: `19`
- final nodes: `27`
- final e-classes: `19`
- cost: `23`
- parameters: `2 -> 2`
- numeric max error: `0.0`

So the multiset path shrinks the final bounded e-graph footprint on row 1, but it does not improve the simplification result.

### Row 50 multiset

Bounded run:
- final stop: `budget_hit`
- final total size: `120`
- final nodes: `180`
- final e-classes: `120`
- cost: `134`
- parameters: `14 -> 14`
- numeric max error: `0.0`

So the multiset path reduces final e-graph footprint relative to the bounded baseline output, but it loses the row 50 parameter reduction entirely.

## Where the Multiset Blow-Up Is Coming From

The dominant growth is not gone. It moved.

### Row 50 multiset stage details

#### `multiset_lower`

- stop: `budget_hit`
- total size: `477`
- nodes: `649`
- e-classes: `365`

Top matches:
- `product flattening = 196`
- `sum flattening = 49`
- `const union = 15`

This is the main blow-up point.

#### `multiset_simplify`

- stop: `saturated`
- total size: `78`
- nodes: `117`
- e-classes: `99`

This stage is not the main problem.

#### `multiset_reify_cleanup`

- stop: `budget_hit`
- total size: `120`
- nodes: `180`
- e-classes: `120`

Top matches:
- `product reify = 28`
- `const union = 15`
- `const set = 15`

This is the second main problem.

## Main Conclusion So Far

The current multiset design does **not** yet let the pipeline safely run to saturation without limits on the representative case.

What happened:
- binary A/C blow-up was one motivation
- but the present multiset lowering/reify rules create their own churn
- especially around repeated multiplicative flattening and reification

So the honest current answer is:
- the baseline replication is good enough to be useful
- the multiset hypothesis is still open
- the current multiset rule design is not yet the right one

## Commands

### Run the Egglog compare helper

```bash
cd /Users/saul/p/egg-smol-python
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/run_compare.py
```

### Run the doc-script notebook

```bash
cd /Users/saul/p/egg-smol-python
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/docs/explanation/2026_03_srtree_eqsat_replication.py
```

### Run the offline Haskell helper

```bash
cd /Users/saul/p/srtree-eqsat
stack exec -- runghc /Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/haskell_compare.hs 1 50
```

### Run the regression tests

```bash
cd /Users/saul/p/egg-smol-python
uv run --project /Users/saul/p/egg-smol-python pytest /Users/saul/p/egg-smol-python/python/tests/test_srtree_eqsat.py
```

## Verified State

These all passed before writing this note:
- `python3 -m py_compile` on the new module, doc-script, runner, and test
- `uv run ... run_compare.py`
- `uv run ... 2026_03_srtree_eqsat_replication.py`
- `uv run ... pytest python/tests/test_srtree_eqsat.py`

The regression test file currently pins:
- row parsing
- row 50 baseline parameter reduction `14 -> 13`
- row 50 multiset lowering hotspot behavior
- no direct `egraph.saturate(...)` calls

## Constraints / Design Decisions Worth Preserving

1. Do not use `egraph.saturate(...)`.
   The current implementation consistently uses `egraph.run(...)`, and the stage driver applies the schedule one step at a time.

2. Keep the Python DSL operator-based.
   This makes source examples much easier to read and keeps the parser environment simple.

3. Keep the notebook self-contained.
   The doc-script should continue to run without the source repo being present, with Haskell numbers embedded as data.

4. Be explicit about unavailable Haskell internals.
   The exported Haskell API does not provide them. Do not fabricate them.

## Suggested Next Steps

The best next step is **not** “turn up the iteration limit.” The useful next steps are:

1. Redesign multiset lowering so it normalizes additive/multiplicative islands once instead of repeatedly flattening equivalent container shapes.

2. Reduce reify churn.
   Right now reification turns containers back into binary trees by repeated `pick/remove`, which causes another round of combinatorial structure growth.

3. Consider whether reification should be delayed or made deterministic.
   A canonical fold order might cut the `product reify` explosion.

4. Improve nonlinear constant analysis if exact baseline parity matters.
   This is the most likely lever for closing the row 50 `14 -> 13` vs `14 -> 12` gap.

5. Only expand to more `example_hl` rows after the multiset lowering path is cheaper.
   The two-row pass already shows the current failure mode clearly.

## If You Need to Continue the Haskell Side

If the next agent wants deeper source-side introspection:
- start from the current stable helper that calls exported `simplifyEqSat`
- treat any copied old `hegg` graph introspection code with suspicion
- row 50 specifically triggered an old-graph internal crash when I tried to inspect intermediate e-graph sizes
- if deeper stats are required, the likely fix is to patch the source repo or the pinned old `hegg` snapshot directly, not to keep layering more bookkeeping into the copied helper

## Short Summary

- Baseline Egglog replication is working and useful.
- Row 1 matches Haskell on parameter count.
- Row 50 is close but still off by one parameter.
- The multiset version shrinks bounded graph size but does not yet improve simplification quality.
- The multiset blow-up is currently dominated by multiplicative flattening and reification.
- The right continuation is to redesign the multiset normalization/reify path, not to widen the current budgets.
