from __future__ import annotations

from pathlib import Path

import pytest

from egglog.exp.srtree_eqsat import (
    HASKELL_REFERENCE_ROWS,
    core_examples,
    parse_hl_expr,
    run_baseline_pipeline,
    run_multiset_pipeline,
)


def test_core_examples_parse() -> None:
    for example in core_examples():
        parsed = parse_hl_expr(example.source)
        assert parsed == example.expr


def test_row_50_baseline_reduces_one_parameter() -> None:
    row_50 = next(example for example in core_examples() if example.row == 50)
    report = run_baseline_pipeline(
        row_50.expr,
        node_cutoff=50_000,
        iteration_limit=12,
        input_names=row_50.input_names,
        sample_points=row_50.sample_points,
    )

    assert report.stop_reason == "saturated"
    assert report.metric_report.before_parameter_count == 14
    assert report.metric_report.after_parameter_count == 13
    assert report.metric_report.jacobian_rank_gap == 0
    assert report.numeric_max_abs_error == pytest.approx(0.0)


def test_row_50_multiset_blowup_is_in_lowering() -> None:
    row_50 = next(example for example in core_examples() if example.row == 50)
    report = run_multiset_pipeline(
        row_50.expr,
        saturate_without_limits=False,
        node_cutoff=50_000,
        iteration_limit=2,
        input_names=row_50.input_names,
        sample_points=row_50.sample_points,
    )

    lower_stage = report.stages[0]
    hottest_rule, hottest_count = max(lower_stage.matches_per_rule.items(), key=lambda item: item[1])
    assert lower_stage.name == "multiset_lower"
    assert lower_stage.stop_reason == "budget_hit"
    assert "srtree_eqsat_multiset_lower" in hottest_rule
    assert hottest_count > 100


def test_haskell_reference_rows_are_embedded() -> None:
    assert HASKELL_REFERENCE_ROWS[1].after_parameter_count == 2
    assert HASKELL_REFERENCE_ROWS[50].after_parameter_count == 12


def test_no_direct_egraph_saturate_calls() -> None:
    source = Path("/Users/saul/p/egg-smol-python/python/egglog/exp/srtree_eqsat.py").read_text()
    assert "egraph.saturate(" not in source
