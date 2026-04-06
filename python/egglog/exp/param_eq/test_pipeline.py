"""Regression tests for the retained param-eq replication pipeline."""

from __future__ import annotations

import json

import pytest

from egglog import EGraph, back_off, eq
from egglog.exp.param_eq import pipeline as param_eq_hegg
from egglog.exp.param_eq.paths import GOLDEN_PATH
from egglog.exp.param_eq.pipeline import count_nodes, count_params, parse_expression, render_num, run_paper_pipeline

GOLDEN_FIXTURE = json.loads(GOLDEN_PATH.read_text())
GOLDEN_CASES = GOLDEN_FIXTURE["cases"]
COMPARABLE_GOLDEN_CASES = [case for case in GOLDEN_CASES if not case["expected_mismatch"]]
GOLDEN_ANALYSIS_CASES = [case for case in COMPARABLE_GOLDEN_CASES if case["compare_root_analysis"]]
GOLDEN_REWRITE_CASES = [
    case for case in COMPARABLE_GOLDEN_CASES if case["compare_rewrite_tree"] and case["category"] != "corpus"
]
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
    report = run_paper_pipeline(expr)

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
    egraph, _ = param_eq_hegg._run_single_pass_egraph(expr)
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(expected_analysis))
    assert run_paper_pipeline(expr).rendered == expected_render


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


def test_zero_division_can_extract_zero_even_when_analysis_is_none() -> None:
    expr = parse_expression("0 / x0")
    egraph, _ = param_eq_hegg._run_single_pass_egraph(expr)
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(param_eq_hegg.OptionalF64.none))
    assert render_num(egraph.extract(expr)) == "0.0"


@pytest.mark.parametrize(
    "source",
    [
        "((-0.00978823600529464 * (x0 * x0)) + ((0.0012052881065756 * x1) * ((-8.2380609512329102) * x1)))",
        "(((51.6682472229003906 * x0) * ((-0.0001894439337775) * x0)) + (-0.009929236885765901 * (x1 * x1)))",
    ],
)
def test_reduced_schedule_case_normalizes_if_one_side_is_already_quadratic(source: str) -> None:
    report = run_paper_pipeline(parse_expression(source))

    assert report.rendered == "((-0.00978823600529464 * (x0 * x0)) + (-0.009929236885765901 * (x1 * x1)))"


def test_reduced_schedule_case_matches_haskell_canary() -> None:
    case = next(case for case in GOLDEN_CASES if case["case_id"] == "x0_sq_plus_x1_sq")
    report = run_paper_pipeline(parse_expression(case["source"]))

    assert report.rendered == case["simplify_e_expr_python"]

def test_add_comm_is_required_for_left_biased_factorization_path() -> None:
    source = parse_expression("(1 / ((x0 * x0) + 1)) + (4 * (x1 * x1))")
    expected = parse_expression("4 * ((x1 * x1) + (0.25 / ((x0 * x0) + 1)))")

    def bounded_schedule(basic_rules):
        scheduler = back_off(
            match_limit=param_eq_hegg.BACKOFF_MATCH_LIMIT,
            ban_length=param_eq_hegg.BACKOFF_BAN_LENGTH,
            fresh_rematch=True,
        ).persistent()
        round_sat = param_eq_hegg.run(basic_rules | param_eq_hegg.fun_rules, scheduler=scheduler) + (
            param_eq_hegg.analysis_schedule.saturate()
        )
        return scheduler.scope(round_sat + round_sat + round_sat + round_sat + param_eq_hegg.analysis_schedule.saturate())

    no_add_comm_basic_rules = (
        param_eq_hegg.basic_mul_comm_rules
        | param_eq_hegg.basic_add_assoc_rules
        | param_eq_hegg.basic_mul_assoc_rules
        | param_eq_hegg.basic_mul_div_rules
        | param_eq_hegg.basic_product_regroup_rules
        | param_eq_hegg.basic_other_rules
    )

    no_add_comm = EGraph(source)
    no_add_comm.run(bounded_schedule(no_add_comm_basic_rules))
    with pytest.raises(Exception):
        no_add_comm.check(eq(source).to(expected))

    with_add_comm = EGraph(source)
    with_add_comm.run(bounded_schedule(param_eq_hegg.basic_rules))
    with_add_comm.check(eq(source).to(expected))


def test_reduced_pagie_second_pass_toy_reaches_haskell_form() -> None:
    source = parse_expression("(-2.2516087483e-06) + (-0.009788252341175882 * ((x0 * x0) + 1))")
    haskell_expected = "(-0.009788252341175882 * (1.0002300317431365 + (x0 * x0)))"

    baseline = run_paper_pipeline(source)

    assert baseline.rendered == haskell_expected

    egraph, _ = param_eq_hegg._run_single_pass_egraph(source)
    assert render_num(egraph.extract(source)) == haskell_expected


def test_pagie_operon_15_matches_haskell_semantics_on_samples() -> None:
    case = next(case for case in GOLDEN_CASES if case["case_id"] == "pagie_operon_15")
    observed = run_paper_pipeline(parse_expression(case["source"])).extracted
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
    egraph, _ = param_eq_hegg._run_single_pass_egraph(expr)
    egraph.check(eq(param_eq_hegg.const_value(expr)).to(_expected_analysis(case)))


@pytest.mark.parametrize("case", GOLDEN_REWRITE_CASES, ids=lambda case: case["case_id"])
def test_golden_rewrite_tree_reaches_haskell_form(case: dict[str, object]) -> None:
    expr = parse_expression(str(case["source"]))
    expected = parse_expression(str(case["rewrite_tree_expr_python"]))
    egraph, _ = param_eq_hegg._run_single_pass_egraph(expr)

    egraph.check(eq(expr).to(expected))


@pytest.mark.parametrize("case", GOLDEN_SIMPLIFY_CASES, ids=lambda case: case["case_id"])
def test_golden_simplify_e_matches_haskell(case: dict[str, object]) -> None:
    expr = parse_expression(str(case["source"]))
    report = run_paper_pipeline(expr)
    exact_form_case = case["category"] not in {"analysis", "guards", "corpus"}

    if case["compare_simplify_e_render"] and exact_form_case:
        assert report.rendered == case["simplify_e_expr_python"]
    if case["compare_param_count"] and case["category"] != "analysis":
        assert report.after_params == case["simplify_e_param_count"]


def test_fixture_tracks_known_mismatches() -> None:
    assert {
        "sub_add_left_assoc",
    }.issubset(KNOWN_MISMATCH_CASE_IDS)
    assert "pagie_operon_15" not in KNOWN_MISMATCH_CASE_IDS
