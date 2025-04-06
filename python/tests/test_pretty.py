# mypy: disable-error-code="empty-body"
from __future__ import annotations

from collections.abc import Callable
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import pytest

from egglog import *

if TYPE_CHECKING:
    from egglog.runtime import RuntimeExpr


class A(Expr):
    V: ClassVar[A]

    def __init__(self) -> None: ...
    @classmethod
    def cls_method(cls) -> A: ...
    def method(self) -> A: ...
    def __neg__(self) -> A: ...
    def __add__(self, other: A) -> A: ...
    def __getitem__(self, key: A) -> A: ...
    def __call__(self) -> A: ...
    def __delitem__(self, key: A) -> None: ...
    def __setitem__(self, key: A, value: A) -> None: ...
    @property
    def prop(self) -> A: ...


@function
def f(x: A) -> A: ...


@function
def g() -> A: ...


@function
def h() -> A: ...


@function
def p() -> i64: ...


@function
def has_default(x: A = A()) -> A: ...


@function
def higher_order(x: Callable[[A], A]) -> A: ...


@function
def binary(x: A, y: A) -> A: ...


del_a = A()
del del_a[g()]

del_del_a = copy(del_a)
del del_del_a[h()]

del_del_a_two = copy(del_a)
del del_del_a_two[A()]

setitem_a = A()
setitem_a[g()] = h()

b = constant("b", A)


@function
def my_very_long_function_name() -> A: ...


long_line = my_very_long_function_name() + my_very_long_function_name() + my_very_long_function_name()

r = ruleset(name="r")

PARAMS = [
    # expression function calls
    pytest.param(A(), "A()", id="init"),
    pytest.param(f(A()), "f(A())", id="call"),
    pytest.param(A.cls_method(), "A.cls_method()", id="class method"),
    pytest.param(A().method(), "A().method()", id="instance method"),
    pytest.param(-A(), "-A()", id="unary operator"),
    pytest.param(A() + g(), "A() + g()", id="binary operator"),
    pytest.param(A()[g()], "A()[g()]", id="getitem"),
    pytest.param(A()(), "A()()", id="call"),
    pytest.param(del_a, "_A_1 = A()\ndel _A_1[g()]\n_A_1", id="delitem"),
    pytest.param(setitem_a, "_A_1 = A()\n_A_1[g()] = h()\n_A_1", id="setitem"),
    pytest.param(del_a + del_a, "_A_1 = A()\ndel _A_1[g()]\n_A_1 + _A_1", id="existing de-duplicate"),
    pytest.param(del_del_a, "_A_1 = A()\ndel _A_1[g()]\ndel _A_1[h()]\n_A_1", id="re-use variable"),
    pytest.param(
        del_del_a + del_del_a_two,
        """_A_1 = A()
del _A_1[g()]
_A_2 = copy(_A_1)
del _A_2[h()]
_A_3 = copy(_A_1)
del _A_3[A()]
_A_2 + _A_3""",
        id="copy name",
    ),
    pytest.param(b, "b", id="constant"),
    pytest.param(A.V, "A.V", id="class variable"),
    pytest.param(A().prop, "A().prop", id="property"),
    pytest.param(ne(A()).to(g()), "ne(A()).to(g())", id="ne"),
    pytest.param(has_default(A()), "has_default()", id="has default"),
    pytest.param(
        rewrite(long_line).to(long_line),
        "_A_1 = (my_very_long_function_name() + my_very_long_function_name()) + my_very_long_function_name()\nrewrite(_A_1).to(_A_1)",
        id="wrap long line",
    ),
    # primitives
    pytest.param(Unit(), "Unit()", id="unit"),
    pytest.param(Bool(True), "Bool(True)", id="bool"),
    pytest.param(i64(42), "i64(42)", id="i64"),
    pytest.param(f64(42.1), "f64(42.1)", id="f64"),
    pytest.param(String("hello"), 'String("hello")', id="string"),
    pytest.param(PyObject("hi"), 'PyObject("hi")', id="pyobject"),
    pytest.param(var("x", A), "x", id="variable"),
    # commands
    pytest.param(rewrite(g()).to(h(), A()), "rewrite(g()).to(h(), A())", id="rewrite"),
    pytest.param(rule(g()).then(h()), "rule(g()).then(h())", id="rule"),
    # Actions
    pytest.param(expr_action(A()), "A()", id="action"),
    pytest.param(set_(p()).to(i64(1)), "set_(p()).to(i64(1))", id="set"),
    pytest.param(union(g()).with_(h()), "union(g()).with_(h())", id="union"),
    pytest.param(let("x", A()), 'let("x", A())', id="let"),
    pytest.param(expr_action(A()), "A()", id="expr action"),
    pytest.param(delete(p()), "delete(p())", id="delete"),
    pytest.param(panic("oh no"), 'panic("oh no")', id="panic"),
    # Fact
    pytest.param(expr_fact(A()), "A()", id="expr fact"),
    pytest.param(eq(g()).to(h()), "eq(g()).to(h())", id="eq"),
    # Ruleset
    pytest.param(ruleset(rewrite(g()).to(h())), "ruleset(rewrite(g()).to(h()))", id="ruleset"),
    # Schedules
    pytest.param(r, 'ruleset(name="r")', id="ruleset with name"),
    pytest.param(r.saturate(), 'ruleset(name="r").saturate()', id="saturate"),
    pytest.param(r * 10, 'ruleset(name="r") * 10', id="repeat"),
    pytest.param(r + r, 'ruleset(name="r") + ruleset(name="r")', id="sequence"),
    pytest.param(seq(r, r, r), 'seq(ruleset(name="r"), ruleset(name="r"), ruleset(name="r"))', id="seq"),
    pytest.param(run(r, h()), 'run(ruleset(name="r"), h())', id="run"),
    # Functions
    pytest.param(f, "f", id="function"),
    pytest.param(A().method, "A().method", id="method"),
    # lambda
    pytest.param(higher_order(lambda x: A() + x), "higher_order(lambda x: A() + x)", id="lambda"),
    # partial
    pytest.param(higher_order(partial(binary, A())), "higher_order(partial(binary, A()))", id="partial"),
    pytest.param(higher_order(lambda x: f(b)), "higher_order(lambda x: f(b))", id="partial-lambda"),
]


@pytest.mark.parametrize(("x", "s"), PARAMS)
def test_str(x: RuntimeExpr, s: str) -> None:
    assert str(x) == s
