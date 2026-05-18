# Param-Eq Replication

This package keeps the retained `param_eq` replication in three layers:

- vendored raw archive sources in [artifacts/original](artifacts/original)
- live Haskell results in `artifacts/live_results.csv`
- Egglog results in `artifacts/egglog_results.csv`

The source layer is the only checked-in paper archive source of truth. Live
Haskell and Egglog CSVs store only generated metrics keyed back to those source
rows.

## Main Modules

- [original_results.py](original_results.py)
  - raw vendored-file schemas and the cleaned retained-paper source DataFrame
- [live_results.py](live_results.py)
  - minimal live-Haskell result schema, writer, and joined loader
- [egglog_results.py](egglog_results.py)
  - minimal Egglog result schema, writer, and joined loader
- [normalize_archives.py](normalize_archives.py)
  - refreshes the vendored raw archive subset from a local `param-eq-haskell`
    checkout
- [run_runtime_compare.py](run_runtime_compare.py)
  - builds the Pagie runtime comparison CSV from archived runtimes, live
    Haskell, and Egglog
- [replication.py](replication.py)
  - jupytext notebook source for the comparison notebook
- [summarize_corpus_comparison.py](summarize_corpus_comparison.py)
  - Rich CLI for comparing two generated result files

## Source Data

The vendored raw archive subset lives under
[artifacts/original](artifacts/original):

- `results/{dataset}_table_counts.csv`
- `results/{dataset}_results`
- `results/exprs/*`
- `results/exprs_simpl/*`
- `runtimes`

`original_results.py` joins those files into one retained-paper DataFrame with:

- stable row identity:
  - `dataset`, `raw_index`, `algorithm_raw`, `algo_row`, `input_kind`
- paper counts:
  - `before_nodes`, `before_params`, `after_nodes`, `after_params`, `n_params`, `n_rank`
- raw expressions:
  - `orig_expr`, `simpl_expr`
- parsed Python-normalized expressions:
  - `orig_parsed_expr`, `simpl_parsed_expr`
- parsed parameter counts:
  - `orig_parsed_n_params`, `simpl_parsed_n_params`
- rank gaps:
  - `before_rank_difference`, `after_rank_difference`
  - `before_parsed_rank_difference`, `after_parsed_rank_difference`

## Generated Result Files

`live_results.csv` stores only:

- row key columns
- `status`, `runtime_ms`
- `before_nodes`, `before_params`
- `after_nodes`, `after_params`
- `rendered`

`egglog_results.csv` stores only:

- row key columns
- `variant`
- `status`, `runtime_ms`
- `before_nodes`, `before_params`
- `after_nodes`, `after_params`
- `egraph_total_size`, `passes`, `extracted_cost`
- `rendered`

Consumers should load the joined DataFrames from `live_results.py` and
`egglog_results.py`, not read those CSVs directly.

## Workflow

From this directory:

```bash
make archived-artifacts
make live-haskell
make artifacts
make notebook
make test
```

The steps are:

1. vendor the raw archive subset from `../param-eq-haskell`
2. run the current local Haskell simplifier on the retained rows
3. run the Egglog baseline on the same retained rows
4. build the Pagie runtime comparison
5. execute the notebook

Override the Haskell checkout path with:

```bash
export EGGLOG_PARAM_EQ_DATA_DIR=/path/to/param-eq-haskell
```

Override the artifact root used by tests or notebook execution with:

```bash
export EGGLOG_PARAM_EQ_ARTIFACT_DIR=/path/to/artifacts
```

## CLI

Single-expression simplification:

```bash
uv run python -m egglog.exp.param_eq --expr='x0 + 1'
```

Compare two generated result files:

```bash
uv run python -m egglog.exp.param_eq.summarize_corpus_comparison \
  --old python/egglog/exp/param_eq/artifacts/live_results.csv \
  --new python/egglog/exp/param_eq/artifacts/egglog_results.csv \
  --new-variant baseline
```
