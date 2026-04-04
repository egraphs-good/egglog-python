"""Regression tests for the retained param-eq replication pipeline."""

from __future__ import annotations

import json

import pytest

from egglog import EGraph, eq
from egglog.exp.param_eq import pipeline as param_eq_hegg
from egglog.exp.param_eq.paths import GOLDEN_PATH
from egglog.exp.param_eq.pipeline import count_nodes, count_params, parse_expression, render_num, run_paper_pipeline

GOLDEN_FIXTURE = json.loads(GOLDEN_PATH.read_text())
GOLDEN_CASES = GOLDEN_FIXTURE["cases"]
COMPARABLE_GOLDEN_CASES = [case for case in GOLDEN_CASES if not case["expected_mismatch"]]
GOLDEN_ANALYSIS_CASES = [case for case in COMPARABLE_GOLDEN_CASES if case["compare_root_analysis"]]
GOLDEN_REWRITE_CASES = [case for case in COMPARABLE_GOLDEN_CASES if case["compare_rewrite_tree"]]
GOLDEN_SIMPLIFY_CASES = [case for case in COMPARABLE_GOLDEN_CASES if case["compare_simplify_e"]]
KNOWN_MISMATCH_CASE_IDS = {case["case_id"] for case in GOLDEN_CASES if case["expected_mismatch"]}


def _expected_analysis(case: dict[str, object]) -> param_eq_hegg.OptionalF64:
    analysis = case["analysis_after"]
    assert isinstance(analysis, dict)
    match analysis["kind"]:
        case "none":
            return param_eq_hegg.OptionalF64.none
        case "some":
            value = analysis["value"]
            assert isinstance(value, int | float)
            return param_eq_hegg.OptionalF64.some(float(value))
    msg = f"Unexpected analysis fixture entry: {analysis!r}"
    raise ValueError(msg)


def test_parse_expression_handles_leading_negative_literal() -> None:
    expr = parse_expression("-0.000465+0.164361*(exp((x0*(2.011000-x0))))")

    assert render_num(expr) == "(-0.000465 + (0.164361 * exp((x0 * (2.011 - x0)))))"
    assert count_params(expr) == 3
    assert count_nodes(expr) == 10


def test_integer_power_exponents_do_not_count_as_parameters() -> None:
    expr = parse_expression("plog(x0) + ((x0)^(-1))")

    assert count_params(expr) == 0


def test_nonconstant_power_exponents_still_count_parameters() -> None:
    expr = parse_expression("(2.0 ** x0) + (x1 ** (x0 + 3.0))")

    assert count_params(expr) == 2


def test_run_paper_pipeline_saturates_simple_expression() -> None:
    expr = parse_expression("exp(log(abs(x0)))")
    report = run_paper_pipeline(expr, mode="egglog-baseline")

    assert report.status == "saturated"
    assert report.after_nodes <= report.before_nodes
    assert report.after_params == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x0 - x0", param_eq_hegg.OptionalF64.none),
        ("2 - 2", param_eq_hegg.OptionalF64.some(0.0)),
        ("x0 / x0", param_eq_hegg.OptionalF64.none),
        ("2 / 2", param_eq_hegg.OptionalF64.some(1.0)),
    ],
)
def test_analysis_matches_haskell_canaries(source: str, expected: param_eq_hegg.OptionalF64) -> None:
    expr = parse_expression(source)
    egraph = EGraph(expr)

    egraph.run(param_eq_hegg.analysis_schedule.saturate())
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(expected))


@pytest.mark.parametrize(
    ("source", "expected_analysis", "expected_render"),
    [
        ("(-2) ** 2", param_eq_hegg.OptionalF64.some(4.0), "4.0"),
        ("(-2) ** 3", param_eq_hegg.OptionalF64.some(-8.0), "-8.0"),
        ("(-2) ** x0", param_eq_hegg.OptionalF64.none, "(-2.0 ** x0)"),
    ],
)
def test_negative_power_cases_match_haskell_expectations(
    source: str,
    expected_analysis: param_eq_hegg.OptionalF64,
    expected_render: str,
) -> None:
    expr = parse_expression(source)
    egraph = EGraph()
    egraph.register(expr)

    egraph.run(param_eq_hegg.total_ruleset)
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(expected_analysis))
    assert run_paper_pipeline(expr, mode="egglog-baseline").rendered == expected_render


def test_merge_panics_on_different_constant_values() -> None:
    egraph = EGraph(
        param_eq_hegg.join_const_value(param_eq_hegg.OptionalF64.some(1.0), param_eq_hegg.OptionalF64.some(2.0))
    )

    with pytest.raises(Exception, match="Merged different constant values"):
        egraph.run(param_eq_hegg.analysis_schedule.saturate())


def test_merge_accepts_close_constant_values() -> None:
    expr = param_eq_hegg.join_const_value(
        param_eq_hegg.OptionalF64.some(1.0),
        param_eq_hegg.OptionalF64.some(1.0 + (param_eq_hegg.CONST_MERGE_TOLERANCE / 2.0)),
    )
    egraph = EGraph(expr)

    egraph.run(param_eq_hegg.analysis_schedule.saturate())
    egraph.check(eq(expr).to(param_eq_hegg.OptionalF64.some(1.0)))


def test_merge_accepts_positive_and_negative_zero() -> None:
    expr = param_eq_hegg.join_const_value(param_eq_hegg.OptionalF64.some(0.0), param_eq_hegg.OptionalF64.some(-0.0))
    egraph = EGraph(expr)

    egraph.run(param_eq_hegg.analysis_schedule.saturate())
    egraph.check(eq(expr).to(param_eq_hegg.OptionalF64.some(0.0)))


def test_constant_prune_deletes_composite_representatives() -> None:
    expr = parse_expression("2 - 2")
    egraph = EGraph(expr)

    egraph.run(param_eq_hegg.analysis_schedule.saturate())

    payload = json.loads(egraph._serialize().to_json())
    ops = {node["op"] for node in payload["nodes"].values()}
    assert "· - ·" not in ops


def test_zero_division_class_stays_mixed_when_analysis_is_none() -> None:
    expr = parse_expression("0 / x0")
    egraph = EGraph()
    egraph.register(expr)

    egraph.run(param_eq_hegg.total_ruleset)
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(param_eq_hegg.OptionalF64.none))

    payload = json.loads(egraph._serialize().to_json())
    ops = {node["op"] for node in payload["nodes"].values()}
    assert "· / ·" in ops


@pytest.mark.parametrize(
    "source",
    [
        "((-0.00978823600529464 * (x0 * x0)) + ((0.0012052881065756 * x1) * ((-8.2380609512329102) * x1)))",
        "(((51.6682472229003906 * x0) * ((-0.0001894439337775) * x0)) + (-0.009929236885765901 * (x1 * x1)))",
    ],
)
def test_reduced_schedule_case_normalizes_if_one_side_is_already_quadratic(source: str) -> None:
    report = run_paper_pipeline(parse_expression(source), mode="egglog-baseline")

    assert report.rendered == "((-0.00978823600529464 * (x0 * x0)) + (-0.009929236885765901 * (x1 * x1)))"


def test_reduced_schedule_case_matches_haskell_canary() -> None:
    case = next(case for case in GOLDEN_CASES if case["case_id"] == "x0_sq_plus_x1_sq")
    report = run_paper_pipeline(parse_expression(case["source"]), mode="egglog-baseline")

    assert report.rendered == case["simplify_e_expr_python"]


def test_sbp_zero_times_quadratic_matches_haskell_canary() -> None:
    case = next(case for case in GOLDEN_CASES if case["case_id"] == "sbp_zero_times_quadratic")
    report = run_paper_pipeline(parse_expression(case["source"]), mode="egglog-baseline")

    assert report.rendered == case["simplify_e_expr_python"]


def test_disabling_add_comm_blocks_left_biased_factorization_path() -> None:
    source = parse_expression("(1 / ((x0 * x0) + 1)) + (4 * (x1 * x1))")
    expected = parse_expression("4 * ((x1 * x1) + (0.25 / ((x0 * x0) + 1)))")

    def bounded_schedule(basic_rules):
        rewrite = param_eq_hegg.run(basic_rules | param_eq_hegg.fun_rules, scheduler=param_eq_hegg.scheduler)
        round_sat = param_eq_hegg.analysis_schedule.saturate() + rewrite
        return param_eq_hegg.scheduler.scope(
            round_sat + round_sat + round_sat + round_sat + param_eq_hegg.analysis_schedule.saturate()
        )

    no_add_comm = EGraph(source)
    no_add_comm.run(bounded_schedule(param_eq_hegg.baseline_basic_rules))
    with pytest.raises(Exception):
        no_add_comm.check(eq(source).to(expected))

    with_add_comm = EGraph(source)
    with_add_comm.run(bounded_schedule(param_eq_hegg.basic_rules))
    with_add_comm.check(eq(source).to(expected))


def test_pagie_operon_15_matches_haskell_semantics_on_samples() -> None:
    case = next(case for case in GOLDEN_CASES if case["case_id"] == "pagie_operon_15")
    observed = run_paper_pipeline(parse_expression(case["source"]), mode="egglog-baseline").extracted
    expected = parse_expression(case["simplify_e_expr_python"])
    samples = (
        {"x0": -1.75, "x1": -1.25},
        {"x0": -0.5, "x1": 0.75},
        {"x0": 0.25, "x1": -1.5},
        {"x0": 0.5, "x1": 1.25},
        {"x0": 1.5, "x1": -0.25},
        {"x0": 1.75, "x1": 1.5},
    )

    for env in samples:
        observed_value = param_eq_hegg._eval_num(observed, env)
        expected_value = param_eq_hegg._eval_num(expected, env)
        assert observed_value is not None
        assert expected_value is not None
        assert abs(observed_value - expected_value) < 1e-9


@pytest.mark.parametrize("case", GOLDEN_ANALYSIS_CASES, ids=lambda case: case["case_id"])
def test_golden_root_analysis_matches_haskell(case: dict[str, object]) -> None:
    expr = parse_expression(str(case["source"]))
    egraph = EGraph()
    egraph.register(expr)

    egraph.run(param_eq_hegg.total_ruleset)
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(_expected_analysis(case)))


@pytest.mark.parametrize("case", GOLDEN_REWRITE_CASES, ids=lambda case: case["case_id"])
def test_golden_rewrite_tree_extraction_matches_haskell(case: dict[str, object]) -> None:
    expr = parse_expression(str(case["source"]))
    extracted, *_ = param_eq_hegg._run_single_pass(expr)

    assert render_num(extracted) == case["rewrite_tree_expr_python"]


@pytest.mark.parametrize("case", GOLDEN_SIMPLIFY_CASES, ids=lambda case: case["case_id"])
def test_golden_simplify_e_matches_haskell(case: dict[str, object]) -> None:
    expr = parse_expression(str(case["source"]))
    report = run_paper_pipeline(expr, mode="egglog-baseline")

    assert report.rendered == case["simplify_e_expr_python"]
    if case["compare_param_count"]:
        assert report.after_params == case["simplify_e_param_count"]


def test_fixture_tracks_known_mismatches() -> None:
    assert {
        "sub_add_left_assoc",
        "pagie_operon_15",
    }.issubset(KNOWN_MISMATCH_CASE_IDS)
