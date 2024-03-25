# mypy: disable-error-code="empty-body"
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from egglog import *

if TYPE_CHECKING:
    from egglog.runtime import RuntimeExpr


class A(Expr):
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


@function
def f(x: A) -> A: ...


@function
def g() -> A: ...


@function
def h() -> A: ...


del_a = A()
del del_a[g()]

setitem_a = A()
setitem_a[g()] = h()

PARAMS = [
    pytest.param(A(), "A()", id="init"),
    pytest.param(f(A()), "f(A())", id="call"),
    pytest.param(A.cls_method(), "A.cls_method()", id="class method"),
    pytest.param(A().method(), "A().method()", id="instance method"),
    pytest.param(-A(), "-A()", id="unary operator"),
    pytest.param(A() + g(), "A() + g()", id="binary operator"),
    pytest.param(A()[g()], "A()[g()]", id="getitem"),
    pytest.param(A()(), "A()()", id="call"),
    pytest.param(del_a, "_A = A()\n del _A[g()]\n_A", id="delitem"),
    pytest.param(setitem_a, "_A = A()\n _A[g()] = h()\n_A", id="setitem"),
]


@pytest.mark.parametrize(("x", "s"), PARAMS)
def test_str(x: RuntimeExpr, s: str) -> None:
    assert str(x) == s
