from __future__ import annotations

import json
import math

import pytest

from egglog import EGraph, back_off, eq, rewrite, ruleset, run, var
from egglog.exp.param_eq import (
    DEMO_CASES,
    DemoCase,
    Num,
    binary_to_containers,
    parse_expression,
    pipeline,
    run_paper_pipeline,
    run_paper_pipeline_container,
)
from egglog.exp.param_eq.__main__ import main
from egglog.exp.param_eq.pipeline import container_schedule, containers_analysis_schedule

from .evaluation import evaluate


@pytest.mark.param_eq_smoke
@pytest.mark.parametrize("variant", ["binary", "container"])
@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda case: case.name)
def test_public_end_to_end_cases(case: DemoCase, variant: str) -> None:
    report = (
        run_paper_pipeline(parse_expression(case.source))
        if variant == "binary"
        else run_paper_pipeline_container(binary_to_containers(parse_expression(case.source)))
    )
    expected_status = "iteration_limit" if (case.name, variant) == ("repeated_monomial", "binary") else "saturated"
    assert report.status == expected_status
    assert 1 <= report.passes <= 2
    assert report.extracted_params <= report.before_params
    assert report.extracted_params < report.before_params or report.extracted_nodes < report.before_nodes
    parse_expression(report.extracted)
    assert case.sample_points
    for x0, x1 in case.sample_points:
        expected = evaluate(case.source, x0=x0, x1=x1)
        actual = evaluate(report.extracted, x0=x0, x1=x1)
        assert math.isfinite(expected)
        assert math.isfinite(actual)
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.param_eq_smoke
def test_log_log_is_not_rewritten_to_its_argument() -> None:
    report = run_paper_pipeline(parse_expression("log(log(x0))"))
    assert math.isclose(
        evaluate(report.extracted, x0=math.e**2, x1=0.0),
        evaluate("log(log(x0))", x0=math.e**2, x1=0.0),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


@pytest.mark.param_eq_smoke
def test_container_log_product_rule_equates_expanded_log() -> None:
    source = binary_to_containers(parse_expression("log(2.0 * x0)"))
    expected = binary_to_containers(parse_expression("log(2.0) + log(x0)"))
    egraph = EGraph(source, save_egglog_string=False)

    egraph.run(containers_analysis_schedule)
    egraph.run(container_schedule)

    # Introduce and normalize the comparison only after the product rule has run,
    # so it cannot pre-seed the constructors that the rule's RHS must create.
    expected = egraph.let("expected", expected)
    egraph.run(containers_analysis_schedule)
    egraph.check(eq(source).to(expected))


@pytest.mark.param_eq_smoke
def test_container_log_rule_preserves_even_power_domain() -> None:
    source_text = "log(x0 ** 2.0)"
    report = run_paper_pipeline_container(binary_to_containers(parse_expression(source_text)))

    assert math.isclose(evaluate(source_text, x0=-1.0, x1=0.0), 0.0, abs_tol=1e-9)
    assert math.isclose(evaluate(report.extracted, x0=-1.0, x1=0.0), 0.0, abs_tol=1e-9)


@pytest.mark.param_eq_smoke
@pytest.mark.parametrize("source", ["exp(1000.5)", "(-1.5) ** 0.25"])
def test_nonfinite_constant_results_are_not_folded(source: str) -> None:
    report = run_paper_pipeline(parse_expression(source))

    assert parse_expression(report.extracted) == parse_expression(source)


@pytest.mark.param_eq_smoke
@pytest.mark.parametrize("source", ["(-1.5) ** 0.25", "1e308 * 1e308", "(1e308 * x0) * 1e308"])
def test_container_pipeline_rejects_nonfinite_coefficient_normalization(source: str) -> None:
    with pytest.raises(
        ValueError, match="container pipeline requires every coefficient normalization to remain finite"
    ):
        run_paper_pipeline_container(binary_to_containers(parse_expression(source)))


@pytest.mark.param_eq_smoke
def test_iteration_limited_pass_is_not_reported_as_saturated(monkeypatch: pytest.MonkeyPatch) -> None:
    x = var("iteration_limit_x", Num)
    limited_rules = ruleset(rewrite(x + Num(0.0)).to(x), name="iteration-limit-rules")
    empty_analysis = ruleset(name="iteration-limit-analysis").saturate()
    limited_schedule = run(
        limited_rules,
        scheduler=back_off(match_limit=0, ban_length=100).persistent(),
    )
    monkeypatch.setattr(pipeline, "HASKELL_INNER_ITERATION_LIMIT", 1)

    report = run_paper_pipeline(
        parse_expression("x0 + 0.0"),
        schedule=limited_schedule,
        analysis_schedule=empty_analysis,
    )

    assert report.status == "iteration_limit"


@pytest.mark.param_eq_smoke
def test_cli_emits_complete_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--variant", "binary", "--expr", DEMO_CASES[2].source]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["variant"] == "binary"
    assert payload["status"] == "saturated"
    assert payload["extracted"]
    assert payload["extracted_params"] <= payload["before_params"]
