from __future__ import annotations

from pathlib import Path

import pytest

from egglog.deconstruct import get_callable_args
from egglog.exp import srtree_eqsat
from egglog.exp.srtree_eqsat import HASKELL_REFERENCE_ROWS, core_examples, parse_hl_expr, run_baseline_pipeline


def test_core_examples_parse() -> None:
    for example in core_examples():
        parsed = parse_hl_expr(example.source)
        assert parsed == example.expr


def test_core_examples_cover_selected_batch() -> None:
    assert [example.row for example in core_examples()] == [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]


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
    assert report.metric_report.optimal_parameter_count is not None
    assert report.numeric_max_abs_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (srtree_eqsat.exp(srtree_eqsat.Num(1.0)), pytest.approx(2.718281828459045)),
        (srtree_eqsat.log(srtree_eqsat.Num(1.0)), pytest.approx(0.0)),
        (srtree_eqsat.sqrt(srtree_eqsat.Num(4.0)), pytest.approx(2.0)),
        (abs(srtree_eqsat.Num(-3.0)), pytest.approx(3.0)),
    ],
)
def test_unary_const_analysis_extracts_constants(expr: srtree_eqsat.Num, expected: pytest.ApproxScalar) -> None:
    report = run_baseline_pipeline(expr, node_cutoff=1_000, iteration_limit=4)
    assert report.stop_reason == "saturated"
    args = get_callable_args(report.extracted, srtree_eqsat.Num)
    assert args is not None
    assert float(args[0].value) == expected


def test_selected_batch_runs_without_cutoff() -> None:
    for example in core_examples():
        report = run_baseline_pipeline(
            example.expr,
            node_cutoff=50_000,
            iteration_limit=12,
            input_names=example.input_names,
            sample_points=example.sample_points,
        )
        assert report.stop_reason == "saturated"
        assert report.numeric_max_abs_error == pytest.approx(0.0)


def test_domain_safe_sampler_avoids_complex_values() -> None:
    expr = srtree_eqsat.cbrt(-(srtree_eqsat.Num.var("alpha")))
    selection = srtree_eqsat._choose_domain_safe_sample_points(expr, ("alpha",), seed=1, count=8)
    assert selection.points
    for point in selection.points:
        value = srtree_eqsat.eval_num(expr, {"alpha": point[0]})
        assert value is not None


def test_haskell_reference_rows_are_embedded() -> None:
    assert HASKELL_REFERENCE_ROWS[1].after_parameter_count == 2
    assert HASKELL_REFERENCE_ROWS[50].after_parameter_count == 12


def test_no_direct_egraph_saturate_calls() -> None:
    source = Path("/Users/saul/p/egg-smol-python/python/egglog/exp/srtree_eqsat.py").read_text()
    assert "egraph.saturate(" not in source
