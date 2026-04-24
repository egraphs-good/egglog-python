"""Smoke test for running the retained param_eq notebook source in-process."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import pandas as pd

from egglog.exp.param_eq.egglog_results import write_egglog_results
from egglog.exp.param_eq.live_results import write_live_results
from egglog.exp.param_eq.run_runtime_compare import write_runtime_compare


def _write_original_fixture(root) -> None:
    (root / "results" / "exprs").mkdir(parents=True)
    (root / "results" / "exprs_simpl").mkdir(parents=True)

    (root / "results" / "pagie_table_counts.csv").write_text(
        ",orig_nodes,orig_params,simpl_nodes,simpl_params,orig_nodes_sympy,orig_params_sympy,simpl_nodes_sympy,simpl_params_sympy,algorithm,n_params,n_rank\n"
        "0,10,3,8,2,9,4,7,3,Bingo,3,2\n",
        encoding="utf-8",
    )
    (root / "results" / "kotanchek_table_counts.csv").write_text(
        ",orig_nodes,orig_params,simpl_nodes,simpl_params,orig_nodes_sympy,orig_params_sympy,simpl_nodes_sympy,simpl_params_sympy,algorithm,n_params,n_rank\n"
        "0,11,4,9,3,10,4,8,3,Bingo,4,3\n",
        encoding="utf-8",
    )
    (root / "results" / "pagie_results").write_text(
        "algorithm,expr,expr_sympy\n"
        "Bingo,x0 + 1,x0 + 1\n",
        encoding="utf-8",
    )
    (root / "results" / "kotanchek_results").write_text(
        "algorithm,expr,expr_sympy\n"
        "Bingo,x1 + 2,x1 + 2\n",
        encoding="utf-8",
    )
    for dataset, expr in (("pagie", "x0 + 1\n"), ("kotanchek", "x1 + 2\n")):
        for algorithm in ("Bingo", "EPLEX", "FEAT", "GOMEA", "Operon", "SBP", "SRjl"):
            (root / "results" / "exprs" / f"{algorithm}_exprs_{dataset}").write_text(expr, encoding="utf-8")
            (root / "results" / "exprs_simpl" / f"{algorithm}_exprs_{dataset}").write_text(
                "x0\n" if dataset == "pagie" else "x1\n",
                encoding="utf-8",
            )
    (root / "runtimes").write_text(
        "benchmarking Egg/45\n"
        "time                 212.2 ms   (194.7 ms .. 240.8 ms)\n",
        encoding="utf-8",
    )


def _live_frame() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 3.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "peak_rss_mb": 10.0,
                "rendered": "x0",
            },
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "sympy",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 3.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "peak_rss_mb": 10.0,
                "rendered": "x0",
            },
            {
                "dataset": "kotanchek",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 4.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "peak_rss_mb": 10.0,
                "rendered": "x1",
            },
            {
                "dataset": "kotanchek",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "sympy",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 4.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "peak_rss_mb": 10.0,
                "rendered": "x1",
            },
        ]
    )


def _egglog_frame() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "variant": "baseline",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 3.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "egraph_total_size": 10.0,
                "passes": 2.0,
                "extracted_cost": 7.0,
                "peak_rss_mb": 20.0,
                "rendered": "x0",
            },
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "sympy",
                "variant": "baseline",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 3.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "egraph_total_size": 10.0,
                "passes": 2.0,
                "extracted_cost": 7.0,
                "peak_rss_mb": 20.0,
                "rendered": "x0",
            },
            {
                "dataset": "kotanchek",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "variant": "baseline",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 4.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "egraph_total_size": 10.0,
                "passes": 2.0,
                "extracted_cost": 7.0,
                "peak_rss_mb": 20.0,
                "rendered": "x1",
            },
            {
                "dataset": "kotanchek",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algo_row": 1,
                "input_kind": "sympy",
                "variant": "baseline",
                "status": "saturated",
                "runtime_ms": 1.0,
                "before_nodes": 4.0,
                "before_params": 1.0,
                "after_nodes": 2.0,
                "after_params": 1.0,
                "egraph_total_size": 10.0,
                "passes": 2.0,
                "extracted_cost": 7.0,
                "peak_rss_mb": 20.0,
                "rendered": "x1",
            },
        ]
    )


def test_replication_notebook_runs_in_process(tmp_path, monkeypatch) -> None:
    notebook_source = Path(__file__).with_name("replication.py")
    notebook_output = notebook_source.with_suffix(".ipynb")
    artifact_dir = tmp_path / "artifacts"

    _write_original_fixture(artifact_dir / "original")
    write_live_results(_live_frame(), artifact_dir / "live_results.csv")
    write_egglog_results(_egglog_frame(), artifact_dir / "egglog_results.csv")
    write_runtime_compare(
        pd.DataFrame.from_records(
            [
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
                    "implementation": "Live Haskell",
                    "algorithm_raw": "Bingo",
                    "algorithm": "Bingo",
                    "algo_row": 1.0,
                    "node_count": 3.0,
                    "after_nodes": 2.0,
                    "runtime_ms": 1.0,
                    "peak_rss_mb": 11.0,
                    "status": "saturated",
                },
                {
                    "implementation": "Egglog",
                    "algorithm_raw": "Bingo",
                    "algorithm": "Bingo",
                    "algo_row": 1.0,
                    "node_count": 3.0,
                    "after_nodes": 2.0,
                    "runtime_ms": 1.0,
                    "peak_rss_mb": 12.0,
                    "status": "saturated",
                },
            ]
        ),
        artifact_dir / "runtime_compare.csv",
    )

    monkeypatch.setenv("EGGLOG_PARAM_EQ_ARTIFACT_DIR", os.fspath(artifact_dir))
    runpy.run_path(str(notebook_source), run_name="__main__")

    payload = json.loads(notebook_output.read_text())
    assert payload["cells"]
    assert any(cell.get("outputs") for cell in payload["cells"] if cell.get("cell_type") == "code")
