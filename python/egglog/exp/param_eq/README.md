# Param-Eq Paper Replication

This directory is the canonical retained baseline for reproducing the older
`param-eq-haskell` pipeline in Egglog before any multiset changes.

The retained source-of-truth files are:

- Haskell implementation:
  - `../param-eq-haskell/src/FixTree.hs`
  - `../param-eq-haskell/src/Main.hs`
  - `../param-eq-haskell/results/`
- Egglog translation:
  - [pipeline.py](pipeline.py)
- Corpus-level plots and observed results:
  - [replication.py](replication.py)
  - [replication.ipynb](replication.ipynb)
- Current hypothesis ledger and rejected fixes:
  - [replication_status.md](replication_status.md)

The main division of responsibility is:

- [pipeline.py](pipeline.py) says what the translation is
- [replication.py](replication.py) says what happened when we ran it on the
  archived paper rows
- [replication_status.md](replication_status.md) says what still differs, what
  we tried, and what hypothesis remains live

## Retained files

- [pipeline.py](pipeline.py): canonical Egglog translation of the paper-era
  simplification pipeline
- [generate_haskell_golden.py](generate_haskell_golden.py): regenerates
  [haskell_golden.json](haskell_golden.json) from live Haskell runs
- [normalize_archives.py](normalize_archives.py): normalizes the archived
  Haskell paper outputs into checked-in CSV artifacts
- [run_egglog_corpus.py](run_egglog_corpus.py): runs the Egglog baseline across
  the archived paper rows
- [inspect_case.py](inspect_case.py): focused case inspector for schedule and
  extraction debugging
- [replication.py](replication.py): jupytext notebook source
- [replication.ipynb](replication.ipynb): executed notebook artifact generated
  from the jupytext source
- [replication_status.md](replication_status.md): current parity state, rejected
  fixes, and next hypotheses
- [artifacts/](artifacts/): checked-in normalized paper artifacts used by the
  notebook
- [haskell_golden.json](haskell_golden.json): Haskell-backed golden cases used
  by [test_pipeline.py](test_pipeline.py)

## Workflow

The local [Makefile](Makefile) in this directory is the canonical entrypoint:

```bash
cd python/egglog/exp/param_eq
make all
```

The supported targets are:

- `make golden`: regenerate [haskell_golden.json](haskell_golden.json) from live
  Haskell runs
- `make artifacts`: regenerate the normalized paper CSV artifacts
- `make notebook`: run [replication.py](replication.py) and refresh
  [replication.ipynb](replication.ipynb)
- `make test`: run the replication-local pytest targets

## Data root

By default the live Haskell/source data root is expected at the sibling path
`../param-eq-haskell` relative to the `egg-smol-python` checkout.

Override it with:

```bash
export EGGLOG_PARAM_EQ_DATA_DIR=/path/to/param-eq-haskell
```

That path is used by:

- [generate_haskell_golden.py](generate_haskell_golden.py)
- [normalize_archives.py](normalize_archives.py)
- [replication_status.md](replication_status.md) when it cites live Haskell
  source locations

The checked-in notebook and tests do not shell out to Haskell during normal
execution. They only read the checked-in artifacts in [artifacts/](artifacts/)
and [haskell_golden.json](haskell_golden.json).

## What To Read First

If you want the current observed results, start with
[replication.py](replication.py) or [replication.ipynb](replication.ipynb).
That notebook contains the current corpus-level comparisons, plots, and
conclusions about whether Egglog is meeting the archived Haskell baseline.

If you want the current remaining gap, start with
[replication_status.md](replication_status.md). It intentionally does not repeat
the notebook plots or corpus-level numbers. Instead it records:

- the accepted schedule approximation
- the remaining mismatch frontier
- rejected fixes that should not be repeated casually
- the current hypothesis for why exact parity is still missing

## Why the schedule looks the way it does

[pipeline.py](pipeline.py) intentionally cites the exact `FixTree.hs` functions
it mirrors:

- `evalConstant`
- `joinA`
- `modifyA`
- `rewritesBasic`
- `rewritesFun`
- `rewriteTree`
- `simplifyE`

The accepted approximation today is the bounded four-round schedule documented
in [replication_status.md](replication_status.md). The quantitative evidence for
keeping that approximation, and the corpus-level consequences of it, are in the
comparison notebook:

- [replication.py](replication.py), especially:
  - `## 4. Haskell vs Egglog Comparison`
  - `## 5. Current Limitations and Likely A/C Effects`

## Scope

This branch is the pre-multiset baseline.

Future refinement should start from this directory, the notebook, and
[replication_status.md](replication_status.md) before trying multiset/container
changes. The earlier `srtree` and other SR experiment families were
intentionally removed so the retained baseline stays auditable.
