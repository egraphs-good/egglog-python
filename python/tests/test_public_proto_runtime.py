"""Normal Python syntax over the generated-record runtime, without an engine."""

from copy import copy

import pytest

from egglog import EGraph, ExprValueError, Fact, _program, bindings, eq, expr_parts, f64, i64
from egglog.deconstruct import get_callable_args, get_callable_fn, get_literal_value
from egglog.runtime import RuntimeExpr


def test_scalar_local_value_and_presentation() -> None:
    assert i64(3).value == 3
    assert f64(1.25).value == 1.25
    assert str(i64(1) + 2) == "i64(1) + 2"
    assert repr(f64(1.0) + 2.0) == "f64(1.0) + 2.0"
    assert get_literal_value(i64(1) + 2) is None
    with pytest.raises(ExprValueError, match="Cannot get Python value"):
        (i64(1) + 2).value
    with pytest.raises(TypeError, match="Expected Expression"):
        get_literal_value(1)


def test_scalar_call_children_are_retained_records() -> None:
    leaf = i64(1)
    expression = leaf + 2
    assert isinstance(expression, RuntimeExpr)
    assert copy(expression).__egg_ref__ == expression.__egg_ref__
    children = get_callable_args(expression)
    assert children is not None
    assert children[0].__egg_ref__ == leaf.__egg_ref__
    assert tuple(child.value for child in children) == (1, 2)
    assert get_callable_fn(expression) == i64.__add__
    assert tuple(child.__egg_ref__ for child in get_callable_args(expression, i64.__add__)) == tuple(
        child.__egg_ref__ for child in children
    )
    assert str(i64) == "i64"


def test_authoring_never_creates_native_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Authoring accessed the native engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    assert (i64(2) + 3).__egg_ref__.read().kind.value.func == "egglog.core.i64.add"


def test_structural_inspection_and_equality_fact() -> None:
    assert expr_parts(i64(1) + 2) == expr_parts(i64(1) + 2)
    assert hash(expr_parts(i64(1) + 2)) == hash(expr_parts(i64(1) + 2))
    assert bool(eq(i64(1) + 2).to(i64(1) + 2))
    assert not bool(eq(i64(1)).to(i64(2)))
    assert str(eq(i64(1)).to(i64(2))) == "eq(i64(1)).to(i64(2))"
    assert expr_parts(f64(0.0)) == expr_parts(f64(-0.0))


def test_normal_scalar_bytes_boundary() -> None:
    graph = EGraph()
    expression = i64(1) + 2
    assert graph.extract(expression).value == 3
    assert graph.extract(f64(1.0) + 2.0).value == 3.0
    graph.check(eq(expression).to(i64(3)))


def test_scalar_equality_dispatch_respects_operand_sorts() -> None:
    assert (i64(1) == f64(1.0)) is False
    assert (f64(1.0) == i64(1)) is False
    equal = i64(1) == i64(1)
    different = f64(1.0) == f64(2.0)
    assert isinstance(equal, Fact)
    assert isinstance(different, Fact)
    assert bool(equal)
    assert not bool(different)


def test_automatic_threads_reports_actual_count() -> None:
    # Existing public behavior; unresolved until byte-level configuration exists.
    graph = EGraph(num_threads=0)
    assert graph.num_threads() >= 1


def test_ten_thousand_ordinary_calls_are_incremental(monkeypatch: pytest.MonkeyPatch) -> None:
    copied = 0
    visited = 0
    original_copy = _program._copy_record
    original_parts = _program._structural_parts

    def counted_copy(*args, **kwargs):
        nonlocal copied
        copied += 1
        return original_copy(*args, **kwargs)

    def counted_parts(*args, **kwargs):
        nonlocal visited
        visited += 1
        return original_parts(*args, **kwargs)

    monkeypatch.setattr(_program, "_copy_record", counted_copy)
    monkeypatch.setattr(_program, "_structural_parts", counted_parts)
    expression = i64(0)
    for _ in range(10_000):
        expression = expression + 1
        hash(expression)
    # Fixed per-call limits, not wall-clock timing: retained prefixes are not
    # rescanned during authoring/hash, independent of chain depth.
    assert copied < 50 * 10_000
    assert visited < 20 * 10_000
    packed = _program.pack([expression.__egg_ref__])
    assert len(packed.program.nodes) == 20_001
