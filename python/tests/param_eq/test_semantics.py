from __future__ import annotations

import math

import pytest

from .evaluation import evaluate


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2.0*x0 + 3.0*x1", 8.0),
        ("log(exp(x0))", 1.0),
        ("sqrt(x0**2)", 1.0),
    ],
)
def test_independent_evaluator(source: str, expected: float) -> None:
    assert math.isclose(evaluate(source, x0=1.0, x1=2.0), expected)


def test_independent_evaluator_rejects_arbitrary_python() -> None:
    with pytest.raises(ValueError, match="Unsupported numeric-test syntax"):
        evaluate("__import__('os')", x0=1.0, x1=2.0)


def test_independent_evaluator_rejects_boolean_literals() -> None:
    with pytest.raises(ValueError, match="Unsupported numeric-test syntax"):
        evaluate("True", x0=1.0, x1=2.0)
