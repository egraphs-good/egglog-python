from __future__ import annotations

import doctest
from typing import ClassVar

import pytest

from egglog import Expr, Map, Unit, function, get_callable_args, get_callable_fn, get_literal_value, i64
from egglog.exp import array_api
from egglog.type_constraint_solver import TypeConstraintError


def test_type_str():
    assert str(i64) == "i64"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


def test_function_call():
    @function
    def one() -> i64:
        return i64(1)

    value = one()
    assert isinstance(value, i64)
    assert get_callable_fn(value) == one
    assert get_callable_args(value) == ()
    assert str(value) == "one()"


def test_classmethod_call():
    with pytest.raises(TypeConstraintError):
        Map.empty()
    selected: type[Map[i64, Unit]] = Map[i64, Unit]
    value = selected.empty()
    assert isinstance(value, selected)
    assert get_callable_fn(value) == selected.empty
    assert get_callable_args(value) == ()


def test_expr_special():
    one = i64(1)
    result = one + one
    assert isinstance(result, i64)
    assert str(result) == "i64(1) + 1"
    arguments = get_callable_args(result)
    assert arguments is not None
    assert tuple(get_literal_value(value) for value in arguments) == (1, 1)


def test_class_variable():
    class Number(Expr):
        one: ClassVar[Number]

    value = Number.one
    assert isinstance(value, Number)
    assert str(value) == "Number.one"
    assert get_callable_args(value) == ()


def test_runtime_class_attr_lookup_is_stable():
    assert array_api.TupleInt.__getitem__ is array_api.TupleInt.__getitem__
    assert array_api.TupleInt.__dict__["__getitem__"] is array_api.TupleInt.__dict__["__getitem__"]


def test_doctest_finder_collects_runtime_function_docstrings():
    names = {test.name for test in doctest.DocTestFinder().find(array_api)}
    assert {
        "egglog.exp.array_api.TupleInt.__getitem__",
        "egglog.exp.array_api.TupleInt.if_",
        "egglog.exp.array_api.TupleTupleInt.product",
        "egglog.exp.array_api.Value.diff",
        "egglog.exp.array_api.RecursiveValue.__getitem__",
    } <= names
