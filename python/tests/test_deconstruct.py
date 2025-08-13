# mypy: disable-error-code="empty-body"
from __future__ import annotations

from functools import partial
from typing import ClassVar

import pytest

from egglog import *


class X(Expr):
    v: ClassVar[X]

    def __init__(self) -> None: ...

    @property
    def p(self) -> X: ...

    @classmethod
    def c(cls) -> X: ...

    def m(self, a: X) -> X: ...


@function
def f(x: X) -> X: ...


@function
def y(x: X, i: i64) -> X: ...


c = constant("c", X)

v = var("v", X)
l = EGraph().let("l", X())


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        (i64(42), 42),
        (i64(42) + i64(1), None),
        (f64(3.14), 3.14),
        (Bool(True), True),
        (PyObject("test"), "test"),
        (UnstableFn(f), f),
        (UnstableFn(f, X()), partial(f, X())),
    ],
)
def test_get_literal_value(expr, value):
    res = get_literal_value(expr)
    if isinstance(res, partial) and isinstance(value, partial):
        assert res.func == value.func
        assert res.args == value.args
        assert res.keywords == value.keywords
    else:
        assert res == value


def test_get_let_name():
    assert get_let_name(l) == "l"
    assert get_let_name(X()) is None


def test_get_var_name():
    assert get_var_name(v) == "v"
    assert get_var_name(X()) is None


@pytest.mark.parametrize(
    ("expr", "fn", "args"),
    [
        pytest.param(f(X()), f, (X(),), id="function call"),
        pytest.param(X().p, X.p, (X(),), id="property"),
        pytest.param(X.c(), X.c, (), id="classmethod"),
        pytest.param(X(), X, (), id="init"),
        pytest.param(X().m(X()), X.m, (X(), X()), id="method call"),
        pytest.param(Vec(i64(1)), Vec, (i64(1),), id="generic class"),
        pytest.param(Vec[i64](), Vec[i64], (), id="generic parameter init"),
        pytest.param(Vec[i64].empty(), Vec[i64].empty, (), id="generic parameter classmethod"),
    ],
)
def test_callable(expr, fn, args):
    assert get_callable_fn(expr) == fn
    assert get_callable_args(expr) == args
    assert get_callable_args(expr, fn) == args


def test_callable_generic_applied():
    assert get_callable_args(Vec(i64(1)), Vec[i64]) == (i64(1),)


def test_callable_generic_applied_method():
    assert get_callable_args(Vec[i64].empty(), Vec[i64].empty) == ()
