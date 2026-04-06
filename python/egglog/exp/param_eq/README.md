# Param-Eq Replication

This directory is the canonical retained `param_eq` package for the symbolic
regression simplification replication.

It keeps one baseline:

- archived Haskell paper rows as the published reference
- live Haskell reruns as the primary behavioral baseline
- Egglog as the retained replication implementation

The notebook and checked-in artifacts are meant to answer one practical
question:

- do archived Haskell, live Haskell, and Egglog support the same qualitative
  conclusions on the retained paper metrics?

## What This Package Contains

### Source files

- [pipeline.py](pipeline.py)
  - the retained Egglog translation of the simplification pipeline
- [run_haskell_corpus.py](run_haskell_corpus.py)
  - reruns the local Haskell implementation across the retained corpus rows
- [run_egglog_corpus.py](run_egglog_corpus.py)
  - runs the retained Egglog baseline across the same rows
- [normalize_archives.py](normalize_archives.py)
  - converts the archived Haskell paper outputs into the checked-in CSV
    artifacts used by the notebook
- [generate_haskell_golden.py](generate_haskell_golden.py)
  - regenerates the reduced Haskell-backed golden cases used in tests
- [replication.py](replication.py)
  - jupytext notebook source for the thesis/check notebook
- [paths.py](paths.py)
  - shared local path helpers for this package
- [__main__.py](__main__.py)
  - one-off CLI entrypoint for simplifying a single expression

### Tests

- [test_pipeline.py](test_pipeline.py)
  - replication-local behavioral tests for the retained baseline
- [test_replication_notebook.py](test_replication_notebook.py)
  - smoke test that the notebook source runs and writes an executed notebook

### Checked-in artifacts

- [artifacts/haskell_paper_rows.csv](artifacts/haskell_paper_rows.csv)
  - archived paper-era Haskell results for the retained rows
- [artifacts/haskell_live_rows.csv](artifacts/haskell_live_rows.csv)
  - current local Haskell rerun on the same retained rows
- [artifacts/egglog_paper_rows.csv](artifacts/egglog_paper_rows.csv)
  - current Egglog results on the same retained rows
- [artifacts/pagie_runtime_scatter.csv](artifacts/pagie_runtime_scatter.csv)
  - archived Haskell Figure 9 benchmark sweep, normalized from the original
    Criterion output
- [artifacts/pagie_runtime_compare.csv](artifacts/pagie_runtime_compare.csv)
  - apples-to-apples Pagie runtime sweep used by the notebook for archived
    Haskell, live Haskell, and Egglog
- [haskell_golden.json](haskell_golden.json)
  - reduced Haskell-backed golden cases used by
    [test_pipeline.py](test_pipeline.py)
- [replication.ipynb](replication.ipynb)
  - executed notebook artifact generated from [replication.py](replication.py)

## Problem And Data Overview

This replication is about simplifying formulas that were already produced by
symbolic-regression systems. It is not about training or rerunning those
regressors.

The retained benchmark families are:

- `Pagie`
- `Kotanchek`

The formulas in those datasets were originally produced by several older
symbolic-regression systems:

- `Bingo`
- `EPLEX`
- `GOMEA`
- `Operon`
- `SBP`
- `SRjl`

In the notebook displays, those names are normalized to the paper-facing
labels:

- `GOMEA -> GP-GOMEA`
- `SRjl -> PySR`

Useful columns in the row artifacts:

- `original_expr`
  - archived benchmark formula for the row
- `sympy_expr`
  - archived Sympy-normalized variant of that formula
- `orig_params`
  - parameter count before simplification
- `simpl_params`
  - parameter count after simplification
- `n_rank`
  - the paper's rank target for the row
- `dataset`, `algorithm`, `algo_row`
  - stable identifiers for matching the same row across artifacts

## Artifact Roles

The three main row artifacts serve different purposes:

- [artifacts/haskell_paper_rows.csv](artifacts/haskell_paper_rows.csv)
  - the published paper reference
- [artifacts/haskell_live_rows.csv](artifacts/haskell_live_rows.csv)
  - the current local Haskell behavior
- [artifacts/egglog_paper_rows.csv](artifacts/egglog_paper_rows.csv)
  - the current Egglog behavior

The notebook compares all three side by side. In practice:

- archived Haskell is the published reference
- live Haskell is the primary comparison target for current behavior
- Egglog is the retained replication being evaluated

## Where To Find The Formulas

For the exact formulas used by the notebook and corpus runners, start with the
checked-in row artifacts:

- [artifacts/haskell_paper_rows.csv](artifacts/haskell_paper_rows.csv)
- [artifacts/haskell_live_rows.csv](artifacts/haskell_live_rows.csv)
- [artifacts/egglog_paper_rows.csv](artifacts/egglog_paper_rows.csv)

For the raw Haskell-side inputs and expression dumps, look in the sibling
`param-eq-haskell` checkout:

- `../param-eq-haskell/results/exprs/`
- `../param-eq-haskell/results/exprs_simpl/`
- `../param-eq-haskell/results/dataset/Pagie.csv`
- `../param-eq-haskell/results/dataset/Kotanchek.csv`

## Workflow

The local [Makefile](Makefile) is the canonical entrypoint for this package:

```bash
cd python/egglog/exp/param_eq
make all
```

Useful targets:

- `make archived-artifacts`
  - regenerate the normalized archived paper artifacts
- `make live-haskell`
  - regenerate [artifacts/haskell_live_rows.csv](artifacts/haskell_live_rows.csv)
- `make golden`
  - regenerate [haskell_golden.json](haskell_golden.json)
- `make artifacts`
  - regenerate all checked-in CSV artifacts
- `make notebook`
  - execute [replication.py](replication.py) and refresh
    [replication.ipynb](replication.ipynb)
- `make test`
  - run the replication-local pytest targets

One-off CLI use:

```bash
uv run python -m egglog.exp.param_eq --expr='...'
```

After local engine changes, rebuild the package with:

```bash
uv sync --reinstall-package egglog --all-extras
```

## Data Root

By default the live Haskell checkout is expected at the sibling path
`../param-eq-haskell` relative to the `egg-smol-python` repository root.

Override it with:

```bash
export EGGLOG_PARAM_EQ_DATA_DIR=/path/to/param-eq-haskell
```

The notebook and tests do not shell out to Haskell during normal execution.
They use the checked-in artifacts in [artifacts/](artifacts/) and
[haskell_golden.json](haskell_golden.json).

## Notebook Scope

The notebook in [replication.ipynb](replication.ipynb) is comparison-first.

Its job is to show, using the same plots and the same table layouts, that:

- archived Haskell, live Haskell, and Egglog can be compared directly from one
  shared data pipeline
- the three result sets support similar qualitative conclusions on the retained
  paper metrics
- the remaining archive drift is modest but should still be reported
  explicitly, rather than silently folded into Egglog-vs-Haskell comparisons
