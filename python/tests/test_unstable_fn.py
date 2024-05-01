# mypy: disable-error-code="empty-body"

# tests translated from unstable-fn.egg


from __future__ import annotations

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


MathFn = UnstableFn[Math, Math]


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
    check(eq(UnstableFn(square)(3)).to(square(3)))


def test_string_fn():
    assert str(UnstableFn(square)) == "UnstableFn(square)"


def test_string_fn_partial():
    assert str(UnstableFn(Math.__mul__, 2)) == "UnstableFn(Math.__mul__, 2)"


@ruleset
def map_ruleset(f: MathFn, x: Math, xs: MathList):
    yield rewrite(MathList.NIL.map(f)).to(MathList.NIL)
    yield rewrite(MathList(x, xs).map(f)).to(MathList(f(x), xs.map(f)))


x = MathList(1, MathList(2, MathList(3, None)))


def test_map():
    squared_r = x.map(UnstableFn(square))
    check(
        eq(squared_r).to(MathList(1, MathList(4, MathList(9, None)))),
        (math_ruleset + square_ruleset + map_ruleset).saturate(),
        squared_r,
    )


@ruleset
def list_multiple_ruleset(x: Math, xs: MathList):
    yield rewrite(xs * x).to(xs.map(UnstableFn(Math.__mul__, x)))


def test_partial_application():
    doubled_x = x * 2
    check(
        eq(doubled_x).to(MathList(2, MathList(4, MathList(6, None)))),
        (math_ruleset + list_multiple_ruleset + map_ruleset).saturate(),
        doubled_x,
    )


@function
def composed_math(f: MathFn, g: MathFn, x: Math) -> Math: ...


@ruleset
def composed_math_ruleset(f: MathFn, g: MathFn, x: Math):
    yield rewrite(composed_math(f, g, x)).to(f(g(x)))


def test_composed():
    square_of_double = UnstableFn(composed_math, UnstableFn(square), UnstableFn(Math.__mul__, 2))
    squared_doubled_x = x.map(square_of_double)
    check(
        eq(squared_doubled_x).to(MathList(4, MathList(16, MathList(36, None)))),
        (math_ruleset + square_ruleset + map_ruleset + composed_math_ruleset).saturate(),
        squared_doubled_x,
    )


i64Fun: TypeAlias = UnstableFn[i64, i64]  # noqa: N816, PYI042


def composed_i64_math(f: MathFn, g: i64Fun, i: i64Like) -> Math: ...


@ruleset
def composed_i64_math_ruleset(f: MathFn, g: i64Fun, i: i64):
    yield rewrite(composed_i64_math(f, g, i)).to(f(g(i)))


def test_composed_i64_math():
    res = composed_i64_math(UnstableFn(square), UnstableFn(i64.__mul__, 2), 4)
    check(
        eq(res).to(square(64)),
        (math_ruleset + square_ruleset + composed_i64_math_ruleset).saturate(),
        res,
    )


def test_extract():
    f = UnstableFn(i64.__mul__, 2)
    res = EGraph().extract(f)
    assert expr_parts(res) == expr_parts(f)
