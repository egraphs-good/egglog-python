from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import pytest

from egglog import EGraph, get_callable_args, get_callable_fn
from egglog.exp.param_eq import (
    DEMO_CASES,
    ContainerPolynomial,
    DemoCase,
    Num,
    ParamCost,
    binary_to_containers,
    container_cost_model,
    containers_to_binary,
    param_cost_model,
    parse_expression,
    polynomial,
    render_num,
)

from .evaluation import evaluate


@pytest.mark.parametrize("source", [case.source for case in DEMO_CASES])
def test_parse_render_parse_round_trip(source: str) -> None:
    parsed = parse_expression(source)
    rendered = render_num(parsed)
    assert parse_expression(rendered) == parsed


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda case: case.name)
def test_container_round_trip_preserves_values(case: DemoCase) -> None:
    container = EGraph().extract(binary_to_containers(parse_expression(case.source)))
    decoded = render_num(containers_to_binary(container))
    for x0, x1 in case.sample_points:
        assert math.isclose(
            evaluate(case.source, x0=x0, x1=x1),
            evaluate(decoded, x0=x0, x1=x1),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


@pytest.mark.parametrize("source", [*(case.source for case in DEMO_CASES), "3.5 * x0 ** 0.5 / x1 + 2.0"])
def test_container_cost_matches_decoded_binary_cost(source: str) -> None:
    container = binary_to_containers(parse_expression(source))
    extracted_container, container_cost = EGraph().extract(
        container, include_cost=True, cost_model=container_cost_model
    )
    decoded = containers_to_binary(extracted_container)
    _, decoded_cost = EGraph().extract(decoded, include_cost=True, cost_model=param_cost_model)

    assert container_cost == decoded_cost


def test_lowering_distributes_scalar_products_into_a_polynomial() -> None:
    lowered = binary_to_containers(parse_expression(DEMO_CASES[0].source))
    poly_args = get_callable_args(lowered, polynomial)
    assert poly_args is not None
    (poly_expr,) = poly_args
    poly = EGraph().extract(cast("ContainerPolynomial", poly_expr))

    assert len(poly.value) == 2
    assert sorted(coef.value for coef in poly.value.values()) == pytest.approx([2.3 / 7.9, 2.3 / 7.9])
    assert all(all(get_callable_fn(term) != polynomial for term in monomial.value) for monomial in poly.value)


def test_parser_rejects_unsupported_calls() -> None:
    with pytest.raises(ValueError, match="Unsupported function call"):
        parse_expression("sin(x0)")


def test_parser_rejects_nonliteral_power() -> None:
    with pytest.raises(ValueError, match="Power exponent must be a numeric literal"):
        parse_expression("x0 ** x1")


@pytest.mark.parametrize("source", ["log()", "log(x0, x1)"])
def test_parser_rejects_wrong_function_arity(source: str) -> None:
    with pytest.raises(ValueError, match="expects exactly one argument"):
        parse_expression(source)


def test_parser_rejects_function_keywords() -> None:
    with pytest.raises(ValueError, match="does not accept keyword arguments"):
        parse_expression("log(num=x0)")


def test_parser_rejects_boolean_literals() -> None:
    with pytest.raises(ValueError, match="Unsupported constant"):
        parse_expression("True")


@pytest.mark.parametrize("source", ["1e309", "-1e309"])
def test_parser_rejects_nonfinite_numeric_literals(source: str) -> None:
    with pytest.raises(ValueError, match="Numeric literals must be finite"):
        parse_expression(source)


def test_renderer_rejects_nonfinite_results() -> None:
    with pytest.raises(ValueError, match="Cannot render a non-finite"):
        render_num(Num(float("inf")))


def test_float_exponents_keep_the_exact_python_ratio() -> None:
    converted = binary_to_containers(parse_expression("x0 ** 0.1"))
    poly_args = get_callable_args(converted, polynomial)
    assert poly_args is not None
    (poly_expr,) = poly_args
    poly = cast("ContainerPolynomial", poly_expr)
    (mono,) = poly.value
    (exponent,) = mono.value.values()
    assert EGraph().extract(exponent).value == Fraction(*(0.1).as_integer_ratio())


def test_param_cost_is_lexicographic() -> None:
    assert ParamCost(floats=1, ops_and_ints=100) < ParamCost(floats=2, ops_and_ints=0)
    assert ParamCost(1, 2) + ParamCost(3, 4) == ParamCost(4, 6)
