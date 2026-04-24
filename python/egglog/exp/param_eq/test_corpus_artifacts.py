"""Schema and join checks for the layered `param_eq` result model."""

from __future__ import annotations

import contextlib
import io
import sys

import pandas as pd

from egglog.exp.param_eq import summarize_corpus_comparison
from egglog.exp.param_eq.egglog_results import (
    EGGLOG_RESULTS_SCHEMA,
    load_egglog_results,
    load_egglog_results_raw,
    write_egglog_results,
)
from egglog.exp.param_eq.live_results import (
    LIVE_RESULTS_SCHEMA,
    load_live_results,
    load_live_results_raw,
    write_live_results,
)
from egglog.exp.param_eq.normalize_archives import vendor_raw_sources
from egglog.exp.param_eq.original_results import (
    ARCHIVED_RUNTIMES_SCHEMA,
    DATASET_RESULTS_SCHEMA,
    EXPR_LINES_SCHEMA,
    ORIGINAL_RESULTS_SCHEMA,
    RAW_ALGORITHMS,
    TABLE_COUNTS_SCHEMA,
    load_archived_runtimes,
    load_dataset_results,
    load_expr_lines,
    load_original_results,
    load_table_counts,
)
from egglog.exp.param_eq.run_runtime_compare import (
    RUNTIME_COMPARE_SCHEMA,
    load_runtime_compare,
    write_runtime_compare,
)


def _write_original_fixture(root) -> None:
    (root / "results" / "exprs").mkdir(parents=True)
    (root / "results" / "exprs_simpl").mkdir(parents=True)

    (root / "results" / "pagie_table_counts.csv").write_text(
        ",orig_nodes,orig_params,simpl_nodes,simpl_params,orig_nodes_sympy,orig_params_sympy,simpl_nodes_sympy,simpl_params_sympy,algorithm,n_params,n_rank\n"
        "0,10,3,8,2,9,4,7,3,Bingo,3,2\n"
        "1,9,3,8,2,9,3,8,2,FEAT,3,\n",
        encoding="utf-8",
    )
    (root / "results" / "kotanchek_table_counts.csv").write_text(
        ",orig_nodes,orig_params,simpl_nodes,simpl_params,orig_nodes_sympy,orig_params_sympy,simpl_nodes_sympy,simpl_params_sympy,algorithm,n_params,n_rank\n"
        "0,11,4,9,3,10,4,8,3,Bingo,4,\n",
        encoding="utf-8",
    )

    (root / "results" / "pagie_results").write_text(
        "algorithm,expr,expr_sympy\nBingo,Log(x0),x0 + 1\n",
        encoding="utf-8",
    )
    (root / "results" / "kotanchek_results").write_text(
        "algorithm,expr,expr_sympy\nBingo,Exp(x1),x1 + 2\n",
        encoding="utf-8",
    )

    for dataset in ("pagie", "kotanchek"):
        for algorithm in RAW_ALGORITHMS:
            expr = "Log(x0)\n" if dataset == "pagie" and algorithm == "Bingo" else "x0 + 1\n"
            simpl = "square(x0)\n" if dataset == "pagie" and algorithm == "Bingo" else "x0\n"
            (root / "results" / "exprs" / f"{algorithm}_exprs_{dataset}").write_text(expr, encoding="utf-8")
            (root / "results" / "exprs_simpl" / f"{algorithm}_exprs_{dataset}").write_text(
                simpl,
                encoding="utf-8",
            )

    (root / "runtimes").write_text(
        "benchmarking Egg/45\ntime                 212.2 ms   (194.7 ms .. 240.8 ms)\n",
        encoding="utf-8",
    )


def test_original_results_raw_loaders_validate_shapes(tmp_path) -> None:
    root = tmp_path / "original"
    _write_original_fixture(root)

    assert list(load_table_counts("pagie", root).columns) == list(TABLE_COUNTS_SCHEMA.columns)
    assert list(load_dataset_results("pagie", root).columns) == list(DATASET_RESULTS_SCHEMA.columns)
    assert list(load_expr_lines("pagie", "Bingo", root=root).columns) == list(EXPR_LINES_SCHEMA.columns)
    assert list(load_archived_runtimes(root).columns) == list(ARCHIVED_RUNTIMES_SCHEMA.columns)


def test_original_results_builds_retained_long_frame(tmp_path) -> None:
    root = tmp_path / "original"
    _write_original_fixture(root)

    frame = load_original_results(root)

    assert list(frame.columns) == list(ORIGINAL_RESULTS_SCHEMA.columns)
    assert len(frame) == 4
    assert frame[["dataset", "raw_index", "algorithm_raw", "algo_row", "input_kind"]].drop_duplicates().shape[0] == len(
        frame
    )
    assert set(frame["algorithm_raw"]) == {"Bingo"}
    pagie_original = frame[(frame["dataset"] == "pagie") & (frame["input_kind"] == "original")].iloc[0]
    assert pagie_original["orig_parsed_expr"] == "log(x0)"
    assert pagie_original["simpl_parsed_expr"] == "x0 ** 2.0"
    pagie_sympy = frame[(frame["dataset"] == "pagie") & (frame["input_kind"] == "sympy")].iloc[0]
    assert pd.isna(pagie_sympy["simpl_expr"])
    assert pd.isna(pagie_sympy["simpl_parsed_expr"])
    kotanchek = frame[frame["dataset"] == "kotanchek"].iloc[0]
    assert pd.isna(kotanchek["n_rank"])
    assert pd.isna(kotanchek["before_rank_difference"])


def test_vendor_raw_sources_copies_minimal_subset(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_original_fixture(source)

    vendor_raw_sources(source, output)

    assert (output / "results" / "pagie_table_counts.csv").exists()
    assert (output / "results" / "exprs" / "Bingo_exprs_pagie").exists()
    assert (output / "results" / "exprs_simpl" / "Bingo_exprs_pagie").exists()
    assert (output / "runtimes").exists()


def test_live_results_round_trip_and_join(tmp_path) -> None:
    root = tmp_path / "original"
    _write_original_fixture(root)
    path = tmp_path / "live_results.csv"
    frame = pd.DataFrame.from_records([
        {
            "dataset": "pagie",
            "raw_index": 0,
            "algorithm_raw": "Bingo",
            "algo_row": 1,
            "input_kind": "original",
            "status": "saturated",
            "runtime_ms": 1.5,
            "before_nodes": 10.0,
            "before_params": 3.0,
            "after_nodes": 1.0,
            "after_params": 1.0,
            "peak_rss_mb": 123.0,
            "rendered": "x0",
        },
        {
            "dataset": "pagie",
            "raw_index": 0,
            "algorithm_raw": "Bingo",
            "algo_row": 1,
            "input_kind": "sympy",
            "status": "missing_timeout",
            "runtime_ms": None,
            "before_nodes": None,
            "before_params": None,
            "after_nodes": None,
            "after_params": None,
            "peak_rss_mb": None,
            "rendered": None,
        },
    ])

    write_live_results(frame, path)
    loaded = load_live_results_raw(path)
    joined = load_live_results(path, source_root=root)

    assert list(loaded.columns) == list(LIVE_RESULTS_SCHEMA.columns)
    assert joined.loc[0, "source_orig_parsed_expr"] == "log(x0)"
    assert joined.loc[0, "simpl_parsed_expr"] == "x0"
    assert joined.loc[0, "after_parsed_rank_difference"] == -2.0
    assert joined.loc[0, "peak_rss_mb"] == 123.0
    assert pd.isna(joined.loc[1, "simpl_parsed_expr"])


def test_egglog_results_round_trip_and_join(tmp_path) -> None:
    root = tmp_path / "original"
    _write_original_fixture(root)
    path = tmp_path / "egglog_results.csv"
    frame = pd.DataFrame.from_records([
        {
            "dataset": "pagie",
            "raw_index": 0,
            "algorithm_raw": "Bingo",
            "algo_row": 1,
            "input_kind": "original",
            "variant": "baseline",
            "status": "saturated",
            "runtime_ms": 1.0,
            "before_nodes": 10.0,
            "before_params": 3.0,
            "after_nodes": 1.0,
            "after_params": 1.0,
            "egraph_total_size": 20.0,
            "passes": 2.0,
            "extracted_cost": 1.0,
            "peak_rss_mb": 456.0,
            "rendered": "x0",
        },
        {
            "dataset": "pagie",
            "raw_index": 0,
            "algorithm_raw": "Bingo",
            "algo_row": 1,
            "input_kind": "sympy",
            "variant": "baseline",
            "status": "timeout",
            "runtime_ms": None,
            "before_nodes": None,
            "before_params": None,
            "after_nodes": None,
            "after_params": None,
            "egraph_total_size": None,
            "passes": None,
            "extracted_cost": None,
            "peak_rss_mb": None,
            "rendered": None,
        },
    ])

    write_egglog_results(frame, path)
    loaded = load_egglog_results_raw(path)
    joined = load_egglog_results(path, variant="baseline", source_root=root)

    assert list(loaded.columns) == list(EGGLOG_RESULTS_SCHEMA.columns)
    assert joined.loc[0, "source_orig_parsed_expr"] == "log(x0)"
    assert joined.loc[0, "simpl_parsed_expr"] == "x0"
    assert joined.loc[0, "egraph_total_size"] == 20.0
    assert joined.loc[0, "peak_rss_mb"] == 456.0
    assert pd.isna(joined.loc[1, "simpl_parsed_expr"])


def test_runtime_compare_round_trip(tmp_path) -> None:
    path = tmp_path / "runtime_compare.csv"
    frame = pd.DataFrame.from_records([
        {
            "implementation": "Archived Haskell",
            "algorithm_raw": None,
            "algorithm": None,
            "algo_row": None,
            "node_count": 45.0,
            "after_nodes": None,
            "runtime_ms": 212.2,
            "peak_rss_mb": None,
            "status": "archived_benchmark",
        },
        {
            "implementation": "Egglog",
            "algorithm_raw": "Bingo",
            "algorithm": "Bingo",
            "algo_row": 1.0,
            "node_count": 3.0,
            "after_nodes": 2.0,
            "runtime_ms": 1.0,
            "peak_rss_mb": 111.0,
            "status": "saturated",
        },
    ])

    write_runtime_compare(frame, path)
    loaded = load_runtime_compare(path)

    assert list(loaded.columns) == list(RUNTIME_COMPARE_SCHEMA.columns)


def test_summarize_corpus_comparison_reads_layered_results(tmp_path, monkeypatch) -> None:
    root = tmp_path / "original"
    _write_original_fixture(root)
    live_path = tmp_path / "live_results.csv"
    egglog_path = tmp_path / "egglog_results.csv"

    write_live_results(
        pd.DataFrame.from_records([
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "status": "saturated",
                "runtime_ms": 1.5,
                "before_nodes": 10.0,
                "before_params": 3.0,
                "after_nodes": 1.0,
                "after_params": 1.0,
                "peak_rss_mb": 10.0,
                "rendered": "x0",
            }
        ]),
        live_path,
    )
    write_egglog_results(
        pd.DataFrame.from_records([
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "variant": "baseline",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 10.0,
                "before_params": 3.0,
                "after_nodes": 1.0,
                "after_params": 1.0,
                "egraph_total_size": 20.0,
                "passes": 2.0,
                "extracted_cost": 1.0,
                "peak_rss_mb": 20.0,
                "rendered": "x0",
            }
        ]),
        egglog_path,
    )

    buffer = io.StringIO()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_corpus_comparison",
            "--old",
            str(live_path),
            "--new",
            str(egglog_path),
            "--new-variant",
            "baseline",
            "--limit",
            "1",
        ],
    )
    with contextlib.redirect_stdout(buffer):
        summarize_corpus_comparison.main()

    output = buffer.getvalue()
    assert "param_eq Corpus Comparison" in output
    assert "Median e-graph nodes" not in output
    assert "Total e-graph size better / worse / same" in output


def test_summarize_corpus_comparison_formats_nan_as_na() -> None:
    assert summarize_corpus_comparison._fmt_number(float("nan")) == "[dim]na[/dim]"
