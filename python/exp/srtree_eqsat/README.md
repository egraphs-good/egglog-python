# Srtree-EqSat Baseline Replication

This folder contains the experiment artifacts and regeneration scripts for the
baseline replication of the `srtree-eqsat` simplification pipeline in Egglog.

The baseline is the current focus. The multiset experiment lives separately in
[python/egglog/exp/srtree_eqsat_multiset.py](../egglog/exp/srtree_eqsat_multiset.py)
so the core replication stays easy to inspect.

## What This Replicates

The source system is the Haskell `srtree-eqsat` project from de Franca and
Kronberger (2023). The Egglog translation keeps the same broad structure:

- constant analysis over the expression language
- `rewritesBasic`
- `constReduction`
- `constFusion`
- `rewritesFun`
- the same backoff schedule shape as the Haskell implementation

The goal here is not to recreate the paper's full benchmark suite. It is to
check whether Egglog behaves similarly to the source Haskell EqSat
implementation on the `test/example_hl` corpus shipped with `srtree-eqsat`.

## Repository Layout

- Baseline engine:
  [python/egglog/exp/srtree_eqsat.py](../egglog/exp/srtree_eqsat.py)
- Experimental multiset engine:
  [python/egglog/exp/srtree_eqsat_multiset.py](../egglog/exp/srtree_eqsat_multiset.py)
- Full-corpus driver:
  [compare_all_rows.py](compare_all_rows.py)
- Small selected-row runner:
  [run_compare.py](run_compare.py)
- One-row Haskell driver:
  [haskell_compare.hs](haskell_compare.hs)
- Full-corpus artifact:
  [corpus_baseline_rows.csv](corpus_baseline_rows.csv)
- Compact corpus summary:
  [corpus_baseline_summary.md](corpus_baseline_summary.md)
- Shareable notebook source:
  [docs/explanation/2026_03_srtree_eqsat_replication.py](../../../docs/explanation/2026_03_srtree_eqsat_replication.py)
- Executed notebook:
  [docs/explanation/2026_03_srtree_eqsat_replication.ipynb](../../../docs/explanation/2026_03_srtree_eqsat_replication.ipynb)

## Baseline Design

### DSL

The Egglog-facing sort is `Num`, not `Expr`.

- built-in arithmetic uses Python syntax: `+`, `-`, `*`, `/`, `**`, unary `-`
- named helpers remain for functions: `exp`, `log`, `sqrt`, `cbrt`, `sqr`, `cube`
- `test/example_hl` rows are parsed with restricted `eval(...)`

### Evaluation

The baseline no longer uses generated Python code for evaluation. Instead it
walks the extracted Egglog term directly with `get_callable_args(...)`.

Important details:

- `render_num(num)` is display-only
- `eval_num(num, env, params=None)` is the structural evaluator
- `cbrt` is evaluated as a real cube root, not `x ** (1/3)`
- invalid-domain evaluations return `None`

### Numeric Validation

The numeric checks use deterministic rejection sampling over the original
expression and keep only points where the expression is real and finite.

Rejected cases include:

- `log(x)` with `x <= 0`
- `sqrt(x)` with `x < 0`
- division by zero
- negative-base non-integer powers
- `nan`, `inf`, or complex results

The same sampling policy is used for:

- numeric max-error checks
- Jacobian-rank estimation
- `optimal_params`

### Metrics

Each corpus row records:

- Egglog status
- Haskell status
- Egglog runtime
- Haskell runtime
- parameter counts before and after
- estimated optimal parameter count
- gap to the estimated optimum
- Egglog total function size
- Egglog serialized node count
- Egglog serialized e-class count
- Haskell output node count
- input and output expressions

Parameter counting follows the paper-style `floatConstsToParam` idea:

- non-integer float literals become parameters
- integers remain constants
- parameters are relabeled left-to-right

`optimal_params` is currently estimated as the numeric Jacobian rank of the
original expression. That estimate is useful, but it is not behaving like a
strict lower bound on every row.

## Haskell Comparison Path

The source repo does not expose the README CLI as a buildable executable, so
the comparison uses a generated one-row Haskell driver that calls the exported
`simplifyEqSat` library function.

What the exported path gives:

- runtime
- parameter counts before and after
- tree node counts before and after
- final simplified expression

What it does not give:

- internal memo size
- final e-class count
- internal stop reason

That is why the corpus artifact does not try to report Haskell e-graph internals.

## Current Corpus Results

These are the current checked-in results from
[corpus_baseline_rows.csv](corpus_baseline_rows.csv).

- total rows: `657`
- comparable Haskell rows: `651`
- Haskell failures: `6`
- Egglog non-saturated rows: `2`
- Egglog domain-limited numeric rows: `2`
- parameter mismatches on comparable rows: `60`

Timing totals:

- Egglog total: `15.276714s`
- Egglog mean: `0.023252s`
- Egglog median: `0.023343s`
- Egglog max: `0.130203s`
- Haskell total: `82.596276s`
- Haskell mean: `0.126876s`
- Haskell median: `0.012404s`
- Haskell max: `0.644085s`

Interpretation:

- Egglog is faster overall on the full sweep
- Haskell is still faster on many easy rows, but has a much heavier tail
- most rows saturate under the current Egglog cutoff and iteration budget
- the remaining gap is mostly about simplification quality on a subset of rows,
  not widespread nontermination

## Regeneration

Assumptions:

- run commands from the `egg-smol-python` repository root
- keep a sibling checkout of `srtree-eqsat` at `../srtree-eqsat`

Rebuild the local Python package:

```bash
uv sync --all-extras --reinstall-package egglog
```

Generate the full corpus artifact:

```bash
uv run --project . python python/exp/srtree_eqsat/compare_all_rows.py
```

Run the selected-row comparison:

```bash
uv run --project . python python/exp/srtree_eqsat/run_compare.py
```

Execute the notebook source as a plain script:

```bash
uv run --project . python docs/explanation/2026_03_srtree_eqsat_replication.py
```

Regenerate and execute the notebook:

```bash
uv run --project . jupytext --to ipynb docs/explanation/2026_03_srtree_eqsat_replication.py
uv run --project . jupyter nbconvert --to notebook --execute --inplace docs/explanation/2026_03_srtree_eqsat_replication.ipynb
```

Validate the code:

```bash
uv run --project . ruff check \
  python/egglog/exp/srtree_eqsat.py \
  python/egglog/exp/srtree_eqsat_multiset.py \
  python/exp/srtree_eqsat/compare_all_rows.py \
  docs/explanation/2026_03_srtree_eqsat_replication.py \
  python/tests/test_srtree_eqsat.py \
  python/exp/srtree_eqsat/run_compare.py
```

```bash
uv run --project . mypy \
  python/egglog/exp/srtree_eqsat.py \
  python/egglog/exp/srtree_eqsat_multiset.py \
  python/exp/srtree_eqsat/compare_all_rows.py
```

```bash
uv run --project . pytest python/tests/test_srtree_eqsat.py
```

## Remaining Limitations

- some rows still miss parameter reductions that Haskell finds
- a few rows go the other direction, where Egglog simplifies more aggressively
  than Haskell
- the numeric-rank estimate is not a perfect proxy for minimal parameter count
- two Egglog rows still stop before saturation under the current budget
- six Haskell rows still fail in the one-row driver path

## Next Steps

1. Cluster the `60` mismatch rows by rewrite pattern.
2. Inspect the two Egglog non-saturated rows to see whether the blocker is
   scheduler behavior, a rewrite hotspot, or size growth.
3. Decide whether the `optimal_params` estimator needs a more stable numeric
   procedure before using it as a headline metric.
4. Only after the baseline is tighter, resume work on the multiset module.
