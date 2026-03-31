from __future__ import annotations

from egglog.exp.param_eq_hegg import count_nodes, count_params, parse_expression, render_num, run_paper_pipeline


def test_parse_expression_handles_leading_negative_literal() -> None:
    expr = parse_expression("-0.000465+0.164361*(exp((x0*(2.011000-x0))))")

    assert render_num(expr) == "(-0.000465 + (0.164361 * exp((x0 * (2.011 - x0)))))"
    assert count_params(expr) == 3
    assert count_nodes(expr) == 10


def test_integer_power_exponents_do_not_count_as_parameters() -> None:
    expr = parse_expression("plog(x0) + ((x0)^(-1))")

    assert count_params(expr) == 0


def test_run_paper_pipeline_saturates_simple_expression() -> None:
    expr = parse_expression("exp(log(abs(x0)))")
    report = run_paper_pipeline(expr, mode="egglog-baseline")

    assert report.status == "saturated"
    assert report.after_nodes <= report.before_nodes
    assert report.after_params == 0
