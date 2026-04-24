from __future__ import annotations

import pandas as pd

from egglog.exp.param_eq import analyze_baseline_results
from egglog.exp.param_eq import egglog_results
from egglog.exp.param_eq.egglog_results import run_egglog_results_frame
from egglog.exp.param_eq.live_results import run_live_results_frame
from egglog.exp.param_eq.resource_guard import cap_workers_for_memory


def test_cap_workers_for_memory_reduces_parallelism() -> None:
    eight_gib = 8 * 1024 * 1024 * 1024
    assert cap_workers_for_memory(8, memory_limit_mb=2048, total_memory_bytes_value=eight_gib) == 3


def test_run_live_results_frame_keeps_peak_rss(monkeypatch) -> None:
    row = {
        "dataset": "pagie",
        "raw_index": 0,
        "algorithm_raw": "Bingo",
        "algo_row": 1,
        "input_kind": "original",
    }

    monkeypatch.setattr("egglog.exp.param_eq.live_results._load_rows", lambda: [row])
    monkeypatch.setattr(
        "egglog.exp.param_eq.live_results._run_haskell_row",
        lambda row, **_: (
            {
                **row,
                "status": "missing_memory_limit",
                "runtime_ms": None,
                "before_nodes": None,
                "before_params": None,
                "after_nodes": None,
                "after_params": None,
                "peak_rss_mb": 512.0,
                "rendered": None,
            },
            "peak_rss_mb=512.0",
        ),
    )

    frame = run_live_results_frame(workers=4, memory_limit_mb=2048)

    assert frame.loc[0, "status"] == "missing_memory_limit"
    assert frame.loc[0, "peak_rss_mb"] == 512.0


def test_run_egglog_results_frame_keeps_peak_rss(monkeypatch) -> None:
    row = {
        "dataset": "pagie",
        "raw_index": 0,
        "algorithm_raw": "Bingo",
        "algo_row": 1,
        "input_kind": "original",
        "algorithm": "Bingo",
        "orig_parsed_expr": "x0",
    }

    monkeypatch.setattr("egglog.exp.param_eq.egglog_results._load_rows", lambda **_: [row])
    monkeypatch.setattr(
        "egglog.exp.param_eq.egglog_results._run_source_row",
        lambda row, variant, **_: {
            "dataset": row["dataset"],
            "raw_index": row["raw_index"],
            "algorithm_raw": row["algorithm_raw"],
            "algo_row": row["algo_row"],
            "input_kind": row["input_kind"],
            "variant": variant,
            "status": "memory_limit",
            "runtime_ms": None,
            "before_nodes": None,
            "before_params": None,
            "after_nodes": None,
            "after_params": None,
            "egraph_total_size": None,
            "passes": None,
            "extracted_cost": None,
            "peak_rss_mb": 768.0,
            "rendered": None,
        },
    )

    frame = run_egglog_results_frame(variant="baseline", workers=6, memory_limit_mb=2048)

    assert frame.loc[0, "status"] == "memory_limit"
    assert frame.loc[0, "peak_rss_mb"] == 768.0


def test_load_rows_limit_per_dataset_preserves_canonical_row_order(monkeypatch) -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "dataset": "kotanchek",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "orig_parsed_expr": "k0o",
            },
            {
                "dataset": "kotanchek",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 1,
                "input_kind": "sympy",
                "orig_parsed_expr": "k0s",
            },
            {
                "dataset": "kotanchek",
                "raw_index": 1,
                "algorithm_raw": "EPLEX",
                "algorithm": "EPLEX",
                "algo_row": 1,
                "input_kind": "original",
                "orig_parsed_expr": "k1o",
            },
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "orig_parsed_expr": "p0o",
            },
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 1,
                "input_kind": "sympy",
                "orig_parsed_expr": "p0s",
            },
            {
                "dataset": "pagie",
                "raw_index": 1,
                "algorithm_raw": "EPLEX",
                "algorithm": "EPLEX",
                "algo_row": 1,
                "input_kind": "original",
                "orig_parsed_expr": "p1o",
            },
        ]
    )

    monkeypatch.setattr(egglog_results, "load_original_results", lambda: frame)

    limited = egglog_results._load_rows(limit_per_dataset=2)
    assert [(row["dataset"], row["orig_parsed_expr"]) for row in limited] == [
        ("kotanchek", "k0o"),
        ("kotanchek", "k0s"),
        ("pagie", "p0o"),
        ("pagie", "p0s"),
    ]

    pagie_only = egglog_results._load_rows(dataset="pagie", limit_per_dataset=1)
    assert [(row["dataset"], row["orig_parsed_expr"]) for row in pagie_only] == [("pagie", "p0o")]

    global_limit_only = egglog_results._load_rows(limit=3)
    assert [(row["dataset"], row["orig_parsed_expr"]) for row in global_limit_only] == [
        ("kotanchek", "k0o"),
        ("kotanchek", "k0s"),
        ("kotanchek", "k1o"),
    ]


def test_parse_args_accepts_limit_per_dataset() -> None:
    args = egglog_results._parse_args(["--variant", "container", "--limit-per-dataset", "10", "--output", "sample.csv"])

    assert args.variant == "container"
    assert args.limit_per_dataset == 10


def test_write_egglog_rank_misses_outputs_only_rank_misses(monkeypatch, tmp_path) -> None:
    egglog = pd.DataFrame.from_records(
        [
            {
                "dataset": "pagie",
                "raw_index": 0,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 1,
                "input_kind": "original",
                "source_orig_parsed_expr": "x0 + 1.0",
                "source_orig_parsed_n_params": 0.0,
                "source_before_nodes": 3.0,
                "n_rank": 2.0,
                "status": "saturated",
                "runtime_ms": 5.0,
                "peak_rss_mb": 20.0,
                "before_params": 4.0,
                "after_params": 3.0,
                "after_rank_difference": 1.0,
                "simpl_parsed_n_params": 0.0,
                "after_parsed_rank_difference": -2.0,
                "passes": 2.0,
                "egraph_total_size": 100.0,
                "extracted_cost": 7.0,
                "rendered": "x0 + 1.0",
            },
            {
                "dataset": "pagie",
                "raw_index": 1,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 2,
                "input_kind": "sympy",
                "source_orig_parsed_expr": "x0 + 2.0",
                "source_orig_parsed_n_params": 0.0,
                "source_before_nodes": 4.0,
                "n_rank": 2.0,
                "status": "saturated",
                "runtime_ms": 6.0,
                "peak_rss_mb": 21.0,
                "before_params": 5.0,
                "after_params": 4.0,
                "after_rank_difference": 2.0,
                "simpl_parsed_n_params": 4.0,
                "after_parsed_rank_difference": 2.0,
                "passes": 3.0,
                "egraph_total_size": 110.0,
                "extracted_cost": 8.0,
                "rendered": "x0 + 2.0",
            },
            {
                "dataset": "pagie",
                "raw_index": 2,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 3,
                "input_kind": "original",
                "source_orig_parsed_expr": "x0 + x1 + 1.0",
                "source_orig_parsed_n_params": 0.0,
                "source_before_nodes": 5.0,
                "n_rank": 1.0,
                "status": "saturated",
                "runtime_ms": 7.0,
                "peak_rss_mb": 22.0,
                "before_params": 6.0,
                "after_params": 3.0,
                "after_rank_difference": 2.0,
                "simpl_parsed_n_params": 3.0,
                "after_parsed_rank_difference": 2.0,
                "passes": 4.0,
                "egraph_total_size": 120.0,
                "extracted_cost": 9.0,
                "rendered": "x0 + x1 + 1.0",
            },
            {
                "dataset": "pagie",
                "raw_index": 3,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 4,
                "input_kind": "original",
                "source_orig_parsed_expr": "x0",
                "source_orig_parsed_n_params": 0.0,
                "source_before_nodes": 1.0,
                "n_rank": 3.0,
                "status": "saturated",
                "runtime_ms": 8.0,
                "peak_rss_mb": 23.0,
                "before_params": 3.0,
                "after_params": 3.0,
                "after_rank_difference": 0.0,
                "simpl_parsed_n_params": 3.0,
                "after_parsed_rank_difference": 0.0,
                "passes": 1.0,
                "egraph_total_size": 90.0,
                "extracted_cost": 6.0,
                "rendered": "x0",
            },
            {
                "dataset": "pagie",
                "raw_index": 4,
                "algorithm_raw": "Bingo",
                "algorithm": "Bingo",
                "algo_row": 5,
                "input_kind": "original",
                "source_orig_parsed_expr": "x0 + x1",
                "source_orig_parsed_n_params": 0.0,
                "source_before_nodes": 2.0,
                "n_rank": 1.0,
                "status": "timeout",
                "runtime_ms": None,
                "peak_rss_mb": 24.0,
                "before_params": 2.0,
                "after_params": None,
                "after_rank_difference": None,
                "simpl_parsed_n_params": None,
                "after_parsed_rank_difference": None,
                "passes": None,
                "egraph_total_size": None,
                "extracted_cost": None,
                "rendered": None,
            },
        ]
    )

    monkeypatch.setattr(analyze_baseline_results, "load_egglog_results", lambda variant="baseline": egglog)
    output_path = tmp_path / "egglog_rank_misses.csv"

    misses = analyze_baseline_results.write_egglog_rank_misses(output_path)
    written = pd.read_csv(output_path)

    assert list(misses["raw_index"]) == [0, 1, 2]
    assert list(written["raw_index"]) == [0, 1, 2]
    assert set(written["rank_gap"]) == {1.0, 2.0}
    assert "x0" not in set(written["source_orig_parsed_expr"])
    assert "x0 + x1" not in set(written["source_orig_parsed_expr"])
