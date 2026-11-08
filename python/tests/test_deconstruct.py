# mypy: disable-error-code="empty-body"
from __future__ import annotations

from functools import partial
from typing import ClassVar

import pytest

from egglog import *
from egglog.egraph import to_runtime_expr


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
    from egglog._program import Builder, pack  # noqa: PLC0415
    from egglog.runtime import RuntimeExpr  # noqa: PLC0415

    assert get_let_name(l) == "l"
    assert get_let_name(X()) is None
    assert get_let_name(c) is None
    assert get_let_name(v) is None
    for name in ("", "x", "$x", "a-b", "a_b", "é"):
        assert get_let_name(EGraph().let(name, X())) == name
    with pytest.raises(TypeError, match="Expected Expression"):
        get_let_name(object())

    packed = pack([expr_parts(l).ref])
    node = packed.program.nodes[packed.indices[0]]
    assert node.kind is not None
    assert node.kind.field == "binding"
    assert node.kind.value == "l"
    assert not (
        *packed.program.constructors,
        *packed.program.functions,
        *packed.program.relations,
        *packed.program.user_defined_primitives,
    )
    owner = Builder.from_program(packed.program).publish()
    detached = RuntimeExpr(owner.ref("nodes", packed.indices[0]))
    assert get_let_name(detached) == "l"
    assert get_callable_fn(detached) is None
    assert get_callable_args(detached) is None


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
        pytest.param(Vec(i64(1)), None, None, id="generic class"),
        pytest.param(Vec[i64](), None, None, id="generic parameter init"),
        pytest.param(
            Map[i64, X].from_pairs(Vec(Pair(i64(1), X()))),
            Map[i64, X].from_pairs,
            (Vec(Pair(i64(1), X())),),
            id="generic parameter classmethod",
        ),
    ],
)
def test_callable(expr, fn, args):
    assert get_callable_fn(expr) == fn
    assert get_callable_args(expr) == args
    assert get_callable_args(expr, fn) == args


def test_callable_generic_applied():
    entries = Vec(Pair(i64(1), X()))
    expression = Map.from_pairs(entries)
    assert get_callable_args(expression, Map[i64, X].from_pairs) == (to_runtime_expr(entries),)
    assert get_callable_args(Vec(i64(1)), Vec[i64]) is None


def test_callable_generic_applied_method():
    vector = Vec[i64].empty()
    assert get_callable_args(vector.push(i64(1)), Vec[i64].push) == (to_runtime_expr(vector), to_runtime_expr(i64(1)))
    assert get_callable_args(vector, Vec[i64].empty) is None
