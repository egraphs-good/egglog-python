# mypy: disable-error-code="empty-body"

# tests translated from unstable-fn.egg


from __future__ import annotations

from functools import partial
from typing import ClassVar, TypeAlias

from egglog import *


class Math(Expr):
    def __init__(self, n: i64Like) -> None: ...

    @classmethod
    def var(cls, s: StringLike) -> Math: ...

    def __add__(self, other: MathLike) -> Math: ...

    def __mul__(self, other: MathLike) -> Math: ...


MathLike: TypeAlias = Math | i64Like | StringLike

converter(String, Math, Math.var)
converter(i64, Math, Math)


@ruleset
def math_ruleset(x: i64, y: i64):
    yield rewrite(Math(x) * Math(y)).to(Math(x * y))


MathFn: TypeAlias = UnstableFn[Math, Math]


class MathList(Expr):
    NIL: ClassVar[MathList]

    def __init__(self, m: MathLike, l: MathListLike) -> None: ...

    def map(self, f: MathFn) -> MathList: ...

    def __mul__(self, x: MathLike) -> MathList: ...


MathListLike: TypeAlias = MathList | None

converter(type(None), MathList, lambda _: MathList.NIL)


@function
def square(x: MathLike) -> Math: ...


@ruleset
def square_ruleset(x: Math):
    yield rewrite(square(x)).to(x * x)


def test_call_fn():
    check_eq(UnstableFn(square)(3), square(3))


def test_string_fn():
    assert str(UnstableFn(square)) == "UnstableFn(square)"


def test_string_fn_partial():
    assert str(UnstableFn(Math.__mul__, Math(2))) == "UnstableFn(Math.__mul__, Math(2))"


@ruleset
def map_ruleset(f: MathFn, x: Math, xs: MathList):
    yield rewrite(MathList.NIL.map(f)).to(MathList.NIL)
    yield rewrite(MathList(x, xs).map(f)).to(MathList(f(x), xs.map(f)))


x = MathList(1, MathList(2, MathList(3, None)))


def test_map():
    check_eq(
        x.map(UnstableFn(square)),
        MathList(1, MathList(4, MathList(9, None))),
        (math_ruleset | square_ruleset | map_ruleset).saturate(),
    )


@ruleset
def list_multiple_ruleset(x: Math, xs: MathList):
    yield rewrite(xs * x).to(xs.map(UnstableFn(Math.__mul__, x)))


def test_partial_application():
    check_eq(
        x * 2,
        MathList(2, MathList(4, MathList(6, None))),
        (math_ruleset | list_multiple_ruleset | map_ruleset).saturate(),
    )


@function
def composed_math(f: MathFn, g: MathFn, x: Math) -> Math: ...


@ruleset
def composed_math_ruleset(f: MathFn, g: MathFn, x: Math):
    yield rewrite(composed_math(f, g, x)).to(f(g(x)))


def test_composed():
    square_of_double = UnstableFn(composed_math, UnstableFn(square), UnstableFn(Math.__mul__, Math(2)))  # type: ignore[arg-type]
    check_eq(
        x.map(square_of_double),
        MathList(4, MathList(16, MathList(36, None))),
        (math_ruleset | square_ruleset | map_ruleset | composed_math_ruleset).saturate(),
    )


i64Fun: TypeAlias = UnstableFn[i64, i64]  # noqa: N816, PYI042


@function
def composed_i64_math(f: MathFn, g: i64Fun, i: i64Like) -> Math: ...


@ruleset
def composed_i64_math_ruleset(f: MathFn, g: i64Fun, i: i64):
    yield rewrite(composed_i64_math(f, g, i)).to(f(Math(g(i))))


def test_composed_i64_math():
    check_eq(
        composed_i64_math(UnstableFn(square), UnstableFn(i64.__mul__, i64(2)), 4),
        Math(64),
        (math_ruleset | square_ruleset | composed_i64_math_ruleset).saturate(),
    )


def test_extract():
    f = UnstableFn(i64.__mul__, i64(2))
    res = EGraph().extract(f)
    assert expr_parts(res) == expr_parts(f)


def test_pass_in_function():
    assert expr_parts(x.map(square)) == expr_parts(x.map(UnstableFn(square)))


def test_pass_in_partial():
    assert expr_parts(x.map(partial(Math.__mul__, Math(2)))) == expr_parts(x.map(UnstableFn(Math.__mul__, Math(2))))
