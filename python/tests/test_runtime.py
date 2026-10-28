from __future__ import annotations

import doctest
import inspect
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import ClassVar

import pytest

from egglog import (
    Expr,
    Map,
    Pair,
    Unit,
    Vec,
    expr_parts,
    function,
    get_callable_args,
    get_callable_fn,
    get_literal_value,
    i64,
)
from egglog.egraph import to_runtime_expr
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
    assert value.value == {}
    assert get_callable_fn(value) is None
    assert get_callable_args(value) is None
    entries = Vec[Pair[i64, Unit]].empty()
    operation = selected.from_pairs(entries)
    assert isinstance(operation, selected)
    assert get_callable_fn(operation) == selected.from_pairs
    assert get_callable_args(operation) == (to_runtime_expr(entries),)


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


@pytest.mark.parametrize("first_inspection", ["dict", "doctest"])
def test_runtime_class_inspection_prepares_callables_once(first_inspection):
    events: list[str] = []

    class Lazy(Expr):
        TOKEN: ClassVar[Lazy]

        def __init__(self) -> None: ...

        @classmethod
        def make(cls) -> Lazy:
            """
            Construct a value.

            >>> 1
            1
            """
            events.append("make")
            return cls()

        def method(self) -> Lazy:
            events.append("method")
            return self

        @property
        def prop(self) -> i64:
            events.append("prop")
            return i64(1)

    assert events == []
    if first_inspection == "dict":
        namespace = vars(Lazy)
        tests = doctest.DocTestFinder().find(Lazy, name="Lazy")
    else:
        tests = doctest.DocTestFinder().find(Lazy, name="Lazy")
        namespace = vars(Lazy)
    assert "Lazy.make" in {test.name for test in tests}
    assert events == ["make", "method", "prop"]
    assert "TOKEN" not in namespace
    assert namespace["make"] is Lazy.make
    assert namespace["method"] is Lazy.method
    assert namespace["prop"] is Lazy.prop
    assert inspect.ismethoddescriptor(namespace["make"])
    assert vars(Lazy)["make"] is namespace["make"]
    assert get_callable_fn(Lazy.make()) == namespace["make"]
    assert get_callable_args(Lazy.make()) == ()
    value = Lazy()
    for expression, member in [(value.method(), namespace["method"]), (value.prop, namespace["prop"])]:
        assert get_callable_fn(expression) == member
        arguments = get_callable_args(expression)
        assert arguments is not None
        assert len(arguments) == 1
        assert arguments[0].__egg_ref__ == expr_parts(value).ref
    assert isinstance(Lazy.TOKEN, Lazy)
    assert events == ["make", "method", "prop"]


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_class_dict_waits_for_preparation(fail):
    entered, release, requested, returned = Event(), Event(), Event(), Event()
    cached: list[object] = []
    failure = ValueError("inspection preparation marker")

    class Lazy(Expr):
        def __init__(self) -> None: ...

        def earlier(self) -> Lazy:
            return self

        def body(self) -> Lazy:
            cached.append(Lazy.earlier)
            entered.set()
            assert release.wait(5)
            if fail:
                raise failure
            return self

    def inspect_class():
        requested.set()
        namespace = vars(Lazy)
        returned.set()
        return namespace

    with ThreadPoolExecutor(max_workers=2) as executor:
        preparation = executor.submit(vars, Lazy)
        try:
            assert entered.wait(5)
            inspection = executor.submit(inspect_class)
            assert requested.wait(5)
            assert not returned.wait(0.05)
        finally:
            release.set()
        if fail:
            with pytest.raises(ValueError, match="inspection preparation marker") as raised:
                preparation.result(timeout=5)
            assert raised.value is failure
        else:
            preparation.result(timeout=5)
        namespace = inspection.result(timeout=5)
    assert Lazy.earlier is cached[0]
    assert len(cached) == 1
    if fail:
        # A failed attempt retains its completed prefix, not a new namespace.
        assert "earlier" not in namespace
    else:
        assert namespace["earlier"] is cached[0]
