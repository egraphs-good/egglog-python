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

## Problem And Data Overview

This replication is about a symbolic-regression simplification pipeline, not
about training regressors from scratch.

The raw inputs here are candidate formulas that were already produced by older
symbolic-regression systems. The `param-eq-haskell` code then simplifies those
formulas with equality saturation and compares how many free numeric parameters
remain afterward.

The main terms used throughout this directory are:

- `Pagie` and `Kotanchek`
  - the two benchmark families carried forward from the archived Haskell
    results
  - each row is one symbolic-regression formula from one of those benchmark
    suites
- `Bingo`, `EPLEX`, `FEAT`, `GOMEA`, `Operon`, `SBP`, and `SRjl`
  - the symbolic-regression systems that produced the candidate formulas in the
    archived data
  - in the normalized notebook artifacts, `GOMEA` is presented as `GP-GOMEA`
    and `SRjl` as `PySR` to make the labels easier to read
- `original_expr`
  - the original formula as archived from the benchmark pipeline
- `sympy_expr`
  - the corresponding formula after the archived Sympy normalization step
- `orig_params` / `simpl_params`
  - the number of numeric parameters before and after simplification
- `n_rank`
  - the archived paper's simplification-rank statistic for that row

So when the notebook compares Egglog to Haskell, it is asking:

- given the same archived benchmark formula,
- does Egglog simplify it to roughly the same parameter count and overall
  complexity as the Haskell baseline?

The most important practical point is that the algorithm names are only labels
for where a formula came from. The replication target here is the Haskell
simplifier's behavior on those formulas, not the training behavior of Bingo,
EPLEX, Operon, or the other upstream systems.

## Retained files

- [pipeline.py](pipeline.py): canonical Egglog translation of the paper-era
  simplification pipeline
- [generate_haskell_golden.py](generate_haskell_golden.py): regenerates
  [haskell_golden.json](haskell_golden.json) from live Haskell runs
- [normalize_archives.py](normalize_archives.py): normalizes the archived
  Haskell paper outputs into checked-in CSV artifacts
- [run_haskell_corpus.py](run_haskell_corpus.py): runs the current local
  Haskell `simplifyE` pipeline across the retained paper rows and writes a
  live full-corpus comparison artifact
- [run_egglog_corpus.py](run_egglog_corpus.py): runs the Egglog baseline across
  the archived paper rows, and can compare multiple pipeline modes
- [inspect_case.py](inspect_case.py): focused case inspector for schedule and
  extraction debugging
- [trace_tables.py](trace_tables.py),
  [trace_egglog.py](trace_egglog.py),
  [trace_haskell.py](trace_haskell.py), and
  [compare_trace.py](compare_trace.py): local full-state trace/comparison tools
  for stepwise Haskell-vs-Egglog debugging
- [replication.py](replication.py): jupytext notebook source
- [replication.ipynb](replication.ipynb): executed notebook artifact generated
  from the jupytext source
- [replication_status.md](replication_status.md): current parity state, rejected
  fixes, and next hypotheses
- [artifacts/](artifacts/): checked-in normalized paper artifacts used by the
  notebook
  - `haskell_paper_rows.csv`: archived published-paper baseline rows
  - `haskell_live_rows.csv`: current local Haskell results on the same rows
  - `egglog_paper_rows.csv`: current Egglog results on the same rows
  - `ablation_summary.csv`: scheduler-ablation acceptance summary against the
    live Haskell baseline
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
- `make archived-artifacts`: regenerate the normalized archived-paper CSVs
- `make live-haskell`: regenerate the live full-corpus Haskell artifact
- `make artifacts`: regenerate all CSV artifacts, including archived Haskell,
  live Haskell, and Egglog rows
- `make notebook`: run [replication.py](replication.py) and refresh
  [replication.ipynb](replication.ipynb)
- `make test`: run the replication-local pytest targets

For one-off schedule probes, use the module CLI directly:

```bash
uv run python -m egglog.exp.param_eq --mode egglog-haskell-literal --expr='...'
```

For stepwise full-state comparisons, run:

```bash
uv run python -m egglog.exp.param_eq.compare_trace --case-id pagie_operon_15
```

That writes ignored trace artifacts under `python/egglog/exp/param_eq/trace/`,
including one JSON snapshot per step for each system plus a short comparison
summary.

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

## Archived Vs Live Haskell Results

The current notebook is intentionally anchored to the archived paper-style CSVs
in [artifacts/](artifacts/) for the published-figure recreation section, but
the main Egglog-vs-Haskell comparison now uses the live full-corpus Haskell
artifact. That keeps the notebook honest about current behavior while still
making the published-paper drift visible.

That also means there are now two distinct comparison targets:

- archived-paper artifacts
  - what the notebook uses to recreate the published paper figures
- current local Haskell runs
  - what [run_haskell_corpus.py](run_haskell_corpus.py) and
    [generate_haskell_golden.py](generate_haskell_golden.py) use for the live
    comparison baseline

As we found with rows like `kotanchek:EPLEX#4`, those two can drift apart. The
right way to handle that is not to overwrite the archived CSVs. It is to keep
both artifacts side-by-side and compare Egglog primarily to the live one, while
still reporting drift from the published archived results.

That would let us separate three questions cleanly:

- does Egglog match the archived paper artifact?
- does Egglog match the current local Haskell implementation?
- where is the remaining gap actually archive drift rather than an Egglog bug?

## Where To Find The Formulas

For the exact formulas used by the notebook and corpus runner, start with the
checked-in CSV artifacts:

- [artifacts/haskell_paper_rows.csv](artifacts/haskell_paper_rows.csv)
  - `original_expr`: the archived original benchmark formula for that row
  - `sympy_expr`: the archived Sympy-normalized formula for that row
- [artifacts/haskell_live_rows.csv](artifacts/haskell_live_rows.csv)
  - the current local Haskell results for the same retained rows
  - includes live `simpl_params`, `simpl_params_sympy`, and rendered outputs
- [artifacts/egglog_paper_rows.csv](artifacts/egglog_paper_rows.csv)
  - repeats `original_expr` and `sympy_expr`
  - adds the current Egglog outputs and counts for the same rows

The stable row identifiers for cross-referencing are:

- `dataset`
- `algorithm`
- `algo_row`

If you want the raw Haskell-side expression files those rows came from, look
under the live data checkout at:

- `../param-eq-haskell/results/exprs/`
- `../param-eq-haskell/results/exprs_simpl/`
- `../param-eq-haskell/results/dataset/Pagie.csv`
- `../param-eq-haskell/results/dataset/Kotanchek.csv`

Examples:

- `../param-eq-haskell/results/exprs/Operon_exprs_pagie`
- `../param-eq-haskell/results/exprs/EPLEX_exprs_kotanchek`

Those files are the best place to inspect the original benchmark formulas in
the same per-algorithm layout the Haskell pipeline used.

## What To Read First

If you want the current observed results, start with
[replication.py](replication.py) or [replication.ipynb](replication.ipynb).
That notebook contains the current corpus-level comparisons, plots, and
conclusions about whether Egglog is meeting the archived Haskell baseline.

If you want the current remaining gap, start with
[replication_status.md](replication_status.md). It intentionally does not repeat
the notebook plots or corpus-level numbers. Instead it records:

- the accepted baseline
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

The retained default in [pipeline.py](pipeline.py) is now the more literal
Haskell-shaped inner loop:

- outer pass cap `2`
- inner iteration cap `30`
- backoff settings `2500`, `30`
- one reused bound scheduler per `rewriteTree`-like pass
- full Haskell rewrite set `rewritesBasic <> rewritesFun`
- one saturated analysis round after each rewrite round
- stop based on whole-pass e-graph size stability, which is the closest current
  Egglog analogue to Haskell's rebuild-size check

There is still an explicit `egglog-haskell-literal` mode in
[pipeline.py](pipeline.py), but it now describes the retained baseline rather
than a separate exploratory schedule.

## Egglog Changes Needed So Far

Getting close to the Haskell baseline required a few specific Egglog-side
changes. The high-level Python schedule alone was not enough.

### 1. Fresh-rematch scheduling, not only backlog replay

Haskell's inner loop rematches against the rebuilt graph each iteration. The
old Egglog backlog scheduler preserved skipped residual matches instead, which
is a different contract.

To express the Haskell behavior cleanly, the engine split scheduler behavior
into two kinds:

- backlog scheduler:
  - preserves the older Egglog residual-match behavior
- fresh scheduler:
  - rematches the rebuilt graph each iteration, which is closer to
    Haskell / hegg

That landed in the engine layer as:

- [/Users/saul/p/egg-smol/src/scheduler.rs](/Users/saul/p/egg-smol/src/scheduler.rs)
  - `FreshScheduler`
  - `EGraph::add_fresh_scheduler(...)`
- [/Users/saul/p/egglog-experimental/src/scheduling.rs](/Users/saul/p/egglog-experimental/src/scheduling.rs)
  - `back-off`
  - `back-off-egg`
- [/Users/saul/p/egg-smol-python/python/egglog/egraph.py](/Users/saul/p/egg-smol-python/python/egglog/egraph.py)
  - `back_off(..., egg_like=True)`

For `param_eq`, the retained baseline now uses the fresh-rematch path because
that is the closer analogue of the Haskell inner loop.

### 2. One persistent scheduler across one `rewriteTree` pass

Haskell does not recreate the backoff scheduler state every inner iteration.
One `BackoffScheduler 2500 30` lives across a whole `rewriteTree` call.

To reproduce that in Python, we needed a low-level bound-scheduler handle so
the same scheduler instance could be reused across multiple `run(...)` calls on
one `EGraph`.

That required binding-layer support in:

- [/Users/saul/p/egg-smol-python/src/scheduler_handle.rs](/Users/saul/p/egg-smol-python/src/scheduler_handle.rs)
  - `SchedulerHandle`
- [/Users/saul/p/egg-smol-python/src/egraph.rs](/Users/saul/p/egg-smol-python/src/egraph.rs)
  - `add_backoff_scheduler(...)`
  - `run_ruleset_with_scheduler(...)`
- [/Users/saul/p/egg-smol-python/python/egglog/egraph.py](/Users/saul/p/egg-smol-python/python/egglog/egraph.py)
  - `_add_backoff_scheduler(...)`
  - `_run_ruleset_with_scheduler(...)`

Without that handle, changing the Python schedule shape alone was not enough to
get close to Haskell.

### 3. Stop on graph stability, not on whether the scheduler still has work

Haskell's inner loop stops when applying rewrites and rebuilding no longer
changes the graph. In this Python translation, rebuild-like propagation is
approximated by a saturated analysis round after each rewrite round.

So the retained baseline now stops based on whole-pass e-graph size stability
after:

- one rewrite round under the bound scheduler
- one saturated analysis round

That is why [pipeline.py](pipeline.py) uses `_serialized_counts(...)` around
the loop instead of stopping on the scheduler's own `can_stop` result.

### 4. Haskell-style backoff accounting inside the custom scheduler

The next important difference was inside the backoff scheduler itself.

Haskell's backoff effectively counts substitution width:

- total matched substitutions, not only raw match count

Egglog's first Python-side backoff version only counted the number of matched
tuples. That let wide factorization and additive rules run longer than they do
in Haskell.

The retained `param_eq` baseline now uses the Haskell-style threshold by
passing `haskell_backoff=True` through the binding layer. That logic lives in:

- [/Users/saul/p/egg-smol-python/src/scheduler_handle.rs](/Users/saul/p/egg-smol-python/src/scheduler_handle.rs)
  - `BackOffState::choose_or_ban(...)`

In practice, that was necessary to suppress the earlier `pagie_operon_15`
factorization/A-C blowup and make the literal schedule practical.

### 5. Keep the Haskell rewrite set intact

We also learned that the baseline should not be made "fast" by silently
dropping Haskell rules like `add_comm`.

The retained baseline now keeps the Haskell rewrite set intact in
[pipeline.py](pipeline.py):

- `rewritesBasic <> rewritesFun`
- including `add_comm`

At this point, the main remaining gaps are no longer explained by a truncated
ruleset or a clearly weaker high-level schedule. The remaining work is in the
smaller mismatch tail.

The current evidence for keeping the retained baseline while continuing to probe
the remaining mismatch frontier is split across:

- [replication.py](replication.py), especially:
  - `## 5. Current Local Haskell vs Egglog Comparison`
  - `## 6. Scheduler Ablation`
  - `## 7. Current Limitations`
- [replication_status.md](replication_status.md), especially:
  - `## Accepted Baseline`
  - `## Mismatch Frontier`
  - `## Current Live Hypothesis`

Important current update:

- after regenerating a full-corpus live Haskell artifact and rerunning the
  direct Haskell-shaped Egglog baseline, all paper rows still saturate on both
  original and Sympy inputs
- exact parameter-count matches against the **current local Haskell** artifact
  are now:
  - `318 / 341` on original inputs
  - `336 / 341` on Sympy inputs
- by contrast, the archived published rows drift away from the current local
  Haskell results on:
  - `21` original rows
  - `215` Sympy rows
- the live Haskell artifact currently includes `2` archived fallback rows
  because those current local reruns still overflow the Haskell stack budget:
  - `pagie SRjl#18`
  - `kotanchek Bingo#21`
- the remaining live Egglog-vs-Haskell gaps are now small:
  - largest original-input gap: `2` parameters
  - largest Sympy-input gap: `1` parameter
- scheduler-ablation result:
  - `no-haskell-backoff` is a hard fail
    - it times out on `pagie Operon#15` original where the retained baseline
      still saturates in about `11.7s`
  - `no-bound-scheduler` keeps the same medians and nearly the same exact-match
    counts, but it loses full saturation on:
    - `pagie EPLEX#25`
    - `pagie SRjl#1`
  - `no-fresh-rematch` also keeps the same medians and exact-match counts, but
    it loses full saturation on:
    - `pagie SRjl#20`
    - `kotanchek Operon#21`
  - `no-graph-size-stop` matched the targeted row-level probes we used for the
    ablation screen, but it made `pagie Operon#15` much slower
    (`~57.3s` vs `~11.7s` baseline) and was not promoted as a simpler retained
    stack

So the main remaining frontier has changed:

- it is no longer the old timeout tail, the old `pagie_operon_15`
  schedule-boundary miss, or the broad Sympy mismatch that showed up only when
  comparing against the archived published rows
- it is now:
  - which of the remaining small row-level gaps are true live Haskell
    mismatches
  - which rows only differ because the published archived results drift from
    the current local Haskell implementation
  - and whether the two Haskell-overflow fallback rows can be replaced with
    true live results later
- the retained scheduler stack still looks necessary as a whole if we want to
  keep the current notebook-level conclusions without reintroducing timeouts
  or losing all-row saturation

The detailed evidence for that lives in [replication_status.md](replication_status.md).

## Scope

This branch is the pre-multiset baseline.

Future refinement should start from this directory, the notebook, and
[replication_status.md](replication_status.md) before trying multiset/container
changes. The earlier `srtree` and other SR experiment families were
intentionally removed so the retained baseline stays auditable.
