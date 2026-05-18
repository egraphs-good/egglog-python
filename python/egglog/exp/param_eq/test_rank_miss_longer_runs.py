from __future__ import annotations

import io

import pandas as pd
from rich.console import Console

from egglog.exp.param_eq import analyze_rank_miss_longer_runs as longer_runs


def _config(mode: longer_runs.ProbeMode) -> longer_runs.ProbeConfig:
    return longer_runs.ProbeConfig(
        mode=mode,
        max_passes=7,
        inner_limit=90,
        match_limit=1234,
        ban_length=17,
        timeout_sec=1.0,
        memory_limit_mb=2048,
        sample_interval_sec=0.01,
    )


def _rank_miss_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {
            "dataset": "pagie",
            "raw_index": 0,
            "algorithm_raw": "Bingo",
            "algo_row": 1,
            "input_kind": "original",
            "algorithm": "Bingo",
            "source_orig_parsed_expr": "x0 + 1.0",
            "source_orig_parsed_n_params": 2.0,
            "source_before_nodes": 3.0,
            "n_rank": 2.0,
            "after_params": 3.0,
            "rank_gap": 1.0,
            "rendered": "x0 + 1.0",
        },
        {
            "dataset": "pagie",
            "raw_index": 1,
            "algorithm_raw": "Bingo",
            "algo_row": 2,
            "input_kind": "sympy",
            "algorithm": "Bingo",
            "source_orig_parsed_expr": "x0 + x1 + 2.0",
            "source_orig_parsed_n_params": 3.0,
            "source_before_nodes": 5.0,
            "n_rank": 2.0,
            "after_params": 4.0,
            "rank_gap": 2.0,
            "rendered": "x0 + x1 + 2.0",
        },
        {
            "dataset": "kotanchek",
            "raw_index": 2,
            "algorithm_raw": "SBP",
            "algo_row": 1,
            "input_kind": "original",
            "algorithm": "SBP",
            "source_orig_parsed_expr": "x0 + 3.0",
            "source_orig_parsed_n_params": 2.0,
            "source_before_nodes": 4.0,
            "n_rank": 1.0,
            "after_params": 2.0,
            "rank_gap": 1.0,
            "rendered": "x0 + 3.0",
        },
    ])


def test_configure_pipeline_sets_long_backoff_scheduler() -> None:
    from egglog.exp.param_eq import pipeline

    original = (
        pipeline.MAX_PASSES,
        pipeline.HASKELL_INNER_ITERATION_LIMIT,
        pipeline.BACKOFF_MATCH_LIMIT,
        pipeline.BACKOFF_BAN_LENGTH,
        pipeline.rewrite_scheduler,
    )
    try:
        scheduler_kind = longer_runs.configure_pipeline_for_probe(_config("long_backoff"))
        assert scheduler_kind == "fresh_rematch_backoff"
        assert pipeline.MAX_PASSES == 7
        assert pipeline.HASKELL_INNER_ITERATION_LIMIT == 90
        assert pipeline.BACKOFF_MATCH_LIMIT == 1234
        assert pipeline.BACKOFF_BAN_LENGTH == 17
        assert pipeline.rewrite_scheduler is not None
        assert pipeline.rewrite_scheduler.scheduler.match_limit == 1234
        assert pipeline.rewrite_scheduler.scheduler.ban_length == 17
        assert pipeline.rewrite_scheduler.scheduler.fresh_rematch is True
    finally:
        (
            pipeline.MAX_PASSES,
            pipeline.HASKELL_INNER_ITERATION_LIMIT,
            pipeline.BACKOFF_MATCH_LIMIT,
            pipeline.BACKOFF_BAN_LENGTH,
            pipeline.rewrite_scheduler,
        ) = original


def test_configure_pipeline_sets_no_backoff_scheduler() -> None:
    from egglog.exp.param_eq import pipeline

    original = (
        pipeline.MAX_PASSES,
        pipeline.HASKELL_INNER_ITERATION_LIMIT,
        pipeline.BACKOFF_MATCH_LIMIT,
        pipeline.BACKOFF_BAN_LENGTH,
        pipeline.rewrite_scheduler,
    )
    try:
        scheduler_kind = longer_runs.configure_pipeline_for_probe(_config("no_backoff"))
        assert scheduler_kind == "none"
        assert pipeline.MAX_PASSES == 7
        assert pipeline.HASKELL_INNER_ITERATION_LIMIT == 90
        assert pipeline.rewrite_scheduler is None
    finally:
        (
            pipeline.MAX_PASSES,
            pipeline.HASKELL_INNER_ITERATION_LIMIT,
            pipeline.BACKOFF_MATCH_LIMIT,
            pipeline.BACKOFF_BAN_LENGTH,
            pipeline.rewrite_scheduler,
        ) = original


def test_filter_rank_miss_rows_applies_cli_filters() -> None:
    filtered = longer_runs.filter_rank_miss_rows(
        _rank_miss_rows(),
        dataset="pagie",
        algorithm="Bingo",
        input_kind="sympy",
        offset=0,
        limit=1,
    )

    assert len(filtered) == 1
    assert filtered.loc[0, "raw_index"] == 1


def test_run_probe_frame_computes_deltas_and_statuses() -> None:
    rows = _rank_miss_rows().iloc[:2]

    def fake_runner(row: dict[str, object], config: longer_runs.ProbeConfig) -> dict[str, object]:
        output = longer_runs._base_output_row(row, config)
        if row["raw_index"] == 0:
            output.update({
                "probe_status": "ok",
                "probe_runtime_ms": 10.0,
                "probe_peak_rss_mb": 20.0,
                "probe_passes": 2,
                "probe_total_size": 30,
                "probe_after_params": 2.0,
                "probe_rank_gap": 0.0,
                "probe_after_nodes": 7,
                "probe_rendered": "x0",
                "param_delta": 1.0,
                "rank_gap_delta": 1.0,
                "improved": True,
                "reached_rank": True,
            })
        else:
            output.update({
                "probe_status": "memory_limit",
                "probe_peak_rss_mb": 2049.0,
                "probe_error": "memory_limit",
            })
        return output

    result = longer_runs.run_probe_frame(
        rows,
        [_config("no_backoff")],
        workers=2,
        console=Console(file=io.StringIO()),
        runner=fake_runner,
    )

    assert list(result["raw_index"]) == [0, 1]
    assert result.loc[0, "param_delta"] == 1.0
    assert result.loc[0, "rank_gap_delta"] == 1.0
    assert bool(result.loc[0, "improved"]) is True
    assert bool(result.loc[0, "reached_rank"]) is True
    assert result.loc[1, "probe_status"] == "memory_limit"
    assert pd.isna(result.loc[1, "probe_after_params"])


def test_write_probe_results_round_trips(tmp_path) -> None:
    rows = _rank_miss_rows().iloc[:1]

    def fake_runner(row: dict[str, object], config: longer_runs.ProbeConfig) -> dict[str, object]:
        output = longer_runs._base_output_row(row, config)
        output.update({
            "probe_status": "ok",
            "probe_runtime_ms": 1.0,
            "probe_peak_rss_mb": 2.0,
            "probe_after_params": 3.0,
            "probe_rank_gap": 1.0,
            "probe_rendered": row["rendered"],
            "param_delta": 0.0,
            "rank_gap_delta": 0.0,
            "improved": False,
            "reached_rank": False,
        })
        return output

    result = longer_runs.run_probe_frame(
        rows,
        [_config("long_backoff")],
        workers=1,
        console=Console(file=io.StringIO()),
        runner=fake_runner,
    )
    output_path = tmp_path / "probe.csv"
    longer_runs.write_probe_results(result, output_path)
    written = pd.read_csv(output_path)

    assert list(written.columns) == longer_runs.OUTPUT_COLUMNS
    assert written.loc[0, "probe_mode"] == "long_backoff"
