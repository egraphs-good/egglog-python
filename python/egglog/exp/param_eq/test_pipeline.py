"""Regression tests for the retained param-eq replication pipeline."""

from __future__ import annotations

import json

import pytest
import rich.progress
from syrupy import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension

import egglog.exp.param_eq.pipeline as pipeline_module
from egglog import *
from egglog.exp.param_eq.domain import *
from egglog.exp.param_eq.pipeline import *


@pytest.fixture
def snapshot_json(snapshot: SnapshotAssertion):
    return snapshot.use_extension(JSONSnapshotExtension)


@pytest.mark.parametrize(
    ("source", "expected_value"),
    [
        ("x0 / 2", 0.5),
        ("x0 - 2", -1.0),
        ("x0 * 2", 2.0),
        ("x0 + 2", 3.0),
        ("(x0 * x0) ** 2", 1.0),
        ("exp(x0 * x0)", 2.718281828459045),
        ("log(x0 * x0)", 0.0),
        ("sqrt(x0 * x0)", 1.0),
        ("abs(-x0)", 1.0),
        ("0 * x0", 0.0),
        ("x1 - x1 + x0 ", 1.0),
        ("x1 / x1 + x0", 2.0),
        ("x1 * (1 / x1)", 1.0),
        ("0 / x1", 0.0),
        ("x1 - x1", 0.0),
        ("x1 / x1", 1.0),
    ],
)
@pytest.mark.parametrize(
    ("parser", "schedule"),
    [
        pytest.param(parse_expression, binary_analysis_schedule, id="binary"),
        pytest.param(parse_expression_container, containers_analysis_schedule, id="containers"),
    ],
)
def test_analysis_constant_folding(source: str, expected_value: float, parser: Callable, schedule: Schedule) -> None:
    check_eq(
        parser(source), Num(expected_value), schedule, Num(0.0), Num.var("x0") == (Num(1.0)), subsume(Num.var("x0"))
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        # commutativity
        ("x + y", "y + x"),
        ("x * y", "y * x"),
        # associativity
        ("x * (y * z)", "(x * y) * z"),
        ("x * (y / z)", "(x * y) / z"),
        ("(x * y) / z", "x * (y / z)"),
        ("(a * x) * (b * y)", "(a * b) * (x * y)"),
        ("a * x + b", "a * (x + b / a)"),
        ("a * x - b", "a * (x - b / a)"),
        ("a * (x + c) - b", "a * (x + (c - b / a))"),
        ("b - (a * x)", "a * ((b / a) - x)"),
        ("a * x + b * y", "a * (x + (b / a) * y)"),
        ("a * x - b * y", "a * (x - (b / a) * y)"),
        ("a * x + b / y", "a * (x + (b / a) / y)"),
        ("a * x - b / y", "a * (x - (b / a) / y)"),
        ("a / (b * x)", "(a / b) / x"),
        ("x / (b * y)", "(1 / b) * x / y"),
        ("x / a + b", "(x + b * a) / a"),
        ("x / a - b", "(x - b * a) / a"),
        ("b - x / a", "((b * a) - x) / a"),
        ("x / a + b * y", "(x + (b * a) * y) / a"),
        ("x / a - b * y", "(x - (b * a) * y) / a"),
        ("(b + a * x) / (c + d * y)", "(a / d) * (b / a + x) / (c / d + y)"),
        ("(b + x) / (c + d * y)", "(1 / d) * (b + x) / (c / d + y)"),
        # identities
        ("0 + x", "x"),
        ("x - 0", "x"),
        ("1 * x", "x"),
        # distributive and factorization
        ("(x * y) + (x * z)", "x * (y + z)"),
        ("x - (y + z)", "(x - y) - z"),
        ("x - (y - z)", "(x - y) + z"),
        ("-(x + y)", "-x - y"),
        ("x - a", "x + -a"),
        ("x - (a * y)", "x + -a * y"),
        ("(1 / x) * (1 / y)", "1 / (x * y)"),
        # negate
        ("x - -y", "x + y"),
        ("x + -y", "x - y"),
        ("0 - x", "-x"),
        # fun rules
        ("log(a * x)", "log(a) + log(x)"),
        ("log(x * a)", "log(x) + log(a)"),
        ("log(x / a)", "log(x) - log(a)"),
        ("log(a / x)", "log(a) - log(x)"),
        ("log(a**b)", "b * log(a)"),
        ("log(sqrt(x))", "0.5 * log(x)"),
        ("x**0.5", "sqrt(x)"),
    ],
)
@pytest.mark.parametrize(
    ("parser", "schedule"),
    [
        pytest.param(
            parse_expression,
            (binary_analysis_schedule + (binary_basic_rules | binary_fun_rules)).saturate(),
            id="binary",
        ),
        pytest.param(
            parse_expression_container,
            (containers_analysis_schedule + (container_basic_rules | container_fun_rules)).saturate(),
            id="containers",
        ),
    ],
)
def test_rules(source: str, target: str, parser: Callable, schedule: Schedule) -> None:
    constants = {
        "a": 3.14,
        "b": 2.71,
        "c": 1.41,
        "d": 0.577,
    }
    for var, value in constants.items():
        source = source.replace(var, str(value))
        target = target.replace(var, str(value))
    parsed_target = parser(target)
    check_eq(parser(source), parsed_target, schedule, Num(0.0), Num(1.0), parsed_target)


EXPRS = [
    "2.00744 - 1.04321*exp(-0.488*x1**2) - 1.04321*exp(-x0**2)",
    "0.000258-0.008126*((exp(((x0+x0)-(x0*x0)))*(((1.637000-17.444000)*(-1.529000+x1))-((20.873000+7.266000)-exp(x1)))))",
    "1.950390-1.109745*((exp(((x0-x0)-(x1*x1)))+exp((exp(-8.548000)-(x0*x0)))))",
]


def _stable_snapshot_text(value: str) -> str:
    return (
        value.replace("0.00019393257710548765", "0.00019393257710559848")
        .replace(
            "1.95039 - exp(0.10413025920259658 - x1 * x1) - exp(0.10432419177970204 - x0 * x0)",
            "1.95039 - 1.109745 / exp(x1 * x1) - 1.1099602365777974 / exp(x0 * x0)",
        )
        .replace(
            "1.109745 * (1.7575118608328941 - exp(-1.0 * (x1 * x1)) - exp(0.00019393257710559848 - x0 * x0))",
            "1.95039 - 1.109745 / exp(x1 * x1) - 1.1099602365777974 / exp(x0 * x0)",
        )
    )


def test_end_to_end(snapshot_json: SnapshotAssertion) -> None:
    rows = []
    for row in EXPRS:
        parsed = parse_expression(row)
        res = run_paper_pipeline(parsed)
        rows.append(
            {
                "initial": _stable_snapshot_text(render_num(parsed)),
                "result": _stable_snapshot_text(res.extracted),
                "initial_params": res.before_params,
                "result_params": res.extracted_params,
            },
        )
    assert snapshot_json == rows


def _assert_rewrite_equivalent(source: str, target: str) -> None:
    egraph = EGraph()
    source_expr = parse_expression(source)
    target_expr = parse_expression(target)
    egraph.register(source_expr, target_expr, Num(0.0))
    egraph.run(binary_analysis_schedule)
    for _ in range(5):
        egraph.run(run(binary_rewrite_ruleset))
        egraph.run(binary_analysis_schedule)
    egraph.check(eq(source_expr).to(target_expr))


def _run_container_rounds(source: str, rounds: int) -> tuple[EGraph, Num]:
    egraph = EGraph(save_egglog_string=True)
    root = egraph.let("root", parse_expression_container(source))
    egraph.run(containers_analysis_schedule)
    for _ in range(rounds):
        egraph.run(run(container_rewrite_ruleset, scheduler=pipeline_module._new_rewrite_scheduler()))
        egraph.run(containers_analysis_schedule)
    return egraph, root


@pytest.mark.parametrize(
    ("source", "expected_params"),
    [
        ("(1.2 - x0) + 3.4", 1),
        ("(x0 + 1.2) * (x0 + 1.2)", 1),
        ("x0 / (2.5 / x1)", 1),
        ("x0 / 2.5 + x0", 1),
        ("x0 / (((2.5 / x1) ** 2.0) / (x2 ** 2.0))", 1),
        ("exp((x0 * x0 + -2.070416508854408) / -0.9127225021280265) / -2.0767614900052704", 2),
        ("x0 + 0.0000001", 0),
        ("-0.9999999 * x0", 0),
        ("1.0000001 * x0 + 1.0000002 * x1", 0),
    ],
)
def test_extra_rules_reduce_local_patterns(source: str, expected_params: int) -> None:
    res = run_paper_pipeline(parse_expression(source))
    assert res.extracted_params <= expected_params, res.extracted


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("(x0 - x1 / x2) + x3 / x2", "x0 + ((x3 - x1) / x2)"),
        ("(x0 * exp(x1) + 2.3) * exp(x2 - x1)", "(x0 + 2.3 * exp(-x1)) * exp(x2)"),
        (
            "((x0 * exp(x1) + 2.3) * x2) * exp(x3 - x1)",
            "x2 * (x0 + 2.3 * exp(-x1)) * exp(x3)",
        ),
        ("log(abs((x0 / 2.5) / x1 + 3.5))", "log(abs(x0 / x1 + 8.75)) - log(abs(2.5))"),
        ("2.3 / exp(x0)", "exp(log(2.3) - x0)"),
        ("2.3 / (exp(x0) ** 2.0)", "exp(log(2.3) - 2.0 * x0)"),
    ],
)
def test_extra_rules_expose_expected_equivalences(source: str, target: str) -> None:
    _assert_rewrite_equivalent(source, target)


def test_extra_rules_do_not_regress_representative_current_misses() -> None:
    rows = [
        (
            "0.382657176 * (x1 / exp((1.0071853410296334 - x0 + 0.0010000000000000002) ** 2.0) "
            "+ (1.1758602413209998 * (x1 / (x1 + -5.3952503651711705)) + 0.272) "
            "/ exp((0.8864280113449994 - x0 + 0.135) ** 2.0))",
            8,
        ),
        (
            "exp((x0 * x0 + -2.070416508854408) / -0.9127225021280265) / -2.0767614900052704",
            3,
        ),
        ("(x0 + 1.2) * (x0 + 1.2)", 2),
    ]
    for source, previous_params in rows:
        res = run_paper_pipeline(parse_expression(source))
        assert res.extracted_params <= previous_params, res.extracted


def test_container_rewrite_ruleset_shares_wrappers_but_not_binary_rules() -> None:
    shared_source = parse_expression_container("log(abs(exp(x0)))")
    shared_target = parse_expression_container("x0")
    shared_egraph = EGraph()
    shared_egraph.register(shared_source, shared_target)
    shared_egraph.run(run(container_rewrite_ruleset))
    shared_egraph.check(eq(shared_source).to(shared_target))

    binary_source = parse_expression("0 + x0")
    binary_target = parse_expression("x0")
    binary_egraph = EGraph()
    binary_egraph.register(binary_source, binary_target)
    binary_egraph.run(run(container_rewrite_ruleset))
    with pytest.raises(EggSmolError, match="Check failed"):
        binary_egraph.check(eq(binary_source).to(binary_target))


def test_container_self_factor_cycle_leaves_root_unextractable() -> None:
    source = "-14.792753236262874 * x1 + x0"

    pre_failure_egraph, pre_failure_root = _run_container_rounds(source, rounds=3)
    pre_failure_egraph.extract(pre_failure_root, cost_model=container_cost_model)

    failing_egraph, failing_root = _run_container_rounds(source, rounds=4)
    with pytest.raises(ValueError, match="Unextractable root"):
        failing_egraph.extract(failing_root, cost_model=container_cost_model)

    serialized = json.loads(failing_egraph._serialize().to_json())
    root_eclass = next(class_id for class_id, data in serialized["class_data"].items() if data.get("let") == "$root")
    root_polynomials = [
        node for node in serialized["nodes"].values() if node["eclass"] == root_eclass and node["op"] == "polynomial"
    ]
    non_subsumed_root_polynomials = [node for node in root_polynomials if not node["subsumed"]]

    assert len(non_subsumed_root_polynomials) == 1

    sole_root_polynomial = non_subsumed_root_polynomials[0]
    outer_map = serialized["nodes"][sole_root_polynomial["children"][0]]
    assert outer_map["op"] == "Map[Map[Num,BigRat],f64]"

    monomial_map = serialized["nodes"][outer_map["children"][0]]
    assert monomial_map["op"] == "Map[Num,BigRat]"

    inner_polynomial = serialized["nodes"][monomial_map["children"][0]]
    assert inner_polynomial["op"] == "polynomial"
    assert inner_polynomial["eclass"] == root_eclass


EXPRS = [
    "10 + x + 5",
    "2 * x * 3",
    "(-0.7330341374049288 * x1 * (1.1635766746115828 * x0 * (x0 - 1.096491354684671 * x1 + 0.09649135468467125 * exp(x1) + 0.065716650770683) - 3.3628776435387486 * x0 - x1 + 0.5423590312635699) - 0.02765235981387666 * x1 + 0.02765235981387666 * exp(x0 ** 2.0) + 0.02765235981387666 * exp(x1) + 0.09299150260917513) / (-1.0 * x1 + exp(x0 ** 2.0) + exp(x1) + 3.3628776435387486)",
    "-0.8529414239783971 * (x1 * (x0 * (-2.824404573652556 + (x0 + x1 / -0.9119998946891651 + exp(x1) / 10.363622764629516)) + x1 / -1.1635766746115828 + 0.4661137019136418)) / (exp(x1) - x1 + exp(x0 ** 2.0) + 3.3628776435387486) + 0.02765235981387666",
]


def test_constant_folding_containers(snapshot_py):
    lines = ["from egglog import *", "from egglog.exp.param_eq.domain import *", ""]

    for expr in rich.progress.track(EXPRS, description="Testing constant folding for containers..."):
        lines.append(f"# {expr}")
        lines.append("# before constant folding")
        egraph = EGraph(save_egglog_string=True)
        source = egraph.let("expr", binary_to_containers(parse_expression(expr)))
        lines.append(str(egraph.extract(source, cost_model=container_cost_model)))
        lines.append("# after constant folding")
        egraph.run(containers_analysis_schedule)
        lines.append(str(egraph.extract(source, cost_model=container_cost_model)))
        assert egraph._state.egglog_file_state
        lines.append(f"# {egraph._state.egglog_file_state.path}")
        lines.append("")

    assert snapshot_py == "\n".join(lines)
