# mypy: disable-error-code="empty-body"

# tests translated from unstable-fn.egg


from __future__ import annotations

from collections.abc import Callable
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
    assert str(UnstableFn(square)) == "square"


def test_string_fn_partial():
    assert str(UnstableFn(Math.__mul__, Math(2))) == "partial(Math.__mul__, Math(2))"


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
    assert expr_parts(x.map(square)) == expr_parts(x.map(UnstableFn(square)))  # type: ignore[arg-type]


def test_pass_in_partial():
    assert expr_parts(x.map(partial(Math.__mul__, Math(2)))) == expr_parts(x.map(UnstableFn(Math.__mul__, Math(2))))  # type: ignore[arg-type]


class A(Expr):
    def __init__(self) -> None: ...


class B(Expr): ...


class C(Expr):
    def __init__(self) -> None: ...


def test_callable_accepted_as_type():
    from egglog.runtime import RuntimeFunction

    @function
    def func(f: UnstableFn[C, A, B]) -> C: ...

    assert isinstance(func, RuntimeFunction)
    original = func.__egg_decls__, func.__egg_ref__

    @function  # type: ignore[no-redef]
    def func(f: Callable[[A, B], C]) -> C: ...

    assert isinstance(func, RuntimeFunction)
    converted = func.__egg_decls__, func.__egg_ref__

    assert converted == original


@function
def a_to_b(x: A) -> B: ...


@function
def b_to_c(b: B) -> C: ...


class TestNormalFns:
    """
    Verify that you can pass in normal function, that have not been annotated with @function
    and they will work with UnstableFn
    """

    def test_basic(self):
        """
        Verify that if we pass in a function it will work even if it isn't wrapped in @function
        """

        def f(x: A) -> C:
            return b_to_c(a_to_b(x))

        @function
        def call(f: Callable[[A], C], x: A) -> C:
            return f(x)

        x = call(f, A())

        assert check_eq(x, b_to_c(a_to_b(A())), run() * 10)

    def test_nonlocals_lifted(self):
        """
        Verify that if a function body refers to nonlocals, they are lifted into functions args which are partially applied
        """

        # convert to nullary function once this is fixed
        # https://github.com/egraphs-good/egglog/issues/382
        @function
        def call(f: Callable[[A], C]) -> C:
            return f(A())

        @function
        def f(x: A) -> C:
            # Verify that inner variable is lifted to partial application in a rewrite like this
            def inner(_: A) -> C:
                return b_to_c(a_to_b(x))

            return call(inner)

        assert check_eq(f(A()), b_to_c(a_to_b(A())), run() * 10)

    def test_rewrite_ruleset_used(self):
        """
        Verify if I use a function in a rule defined in a ruleset function, it will use that context for the ruleset name.
        """

        @function
        def transform_a(a: A) -> A: ...

        @function
        def my_transform_a(a: A) -> A: ...

        r = ruleset()

        @function(ruleset=r)
        def apply_f(f: Callable[[A], A], x: A) -> A:
            return f(x)

        @r.register
        def _rewrite(a: A):
            yield rewrite(transform_a(a)).to(apply_f(lambda x: my_transform_a(x), a))

        assert check_eq(transform_a(A()), my_transform_a(A()), r * 10)

    def test_rewrite_ruleset_used_constructor(self):
        """
        Verify if I use a function in a rule defined in a ruleset function, it will use that context for the ruleset name.
        """

        @function
        def transform_a(a: A) -> A: ...

        @function
        def my_transform_a(a: A) -> A: ...

        apply_ruleset = ruleset()

        @function(ruleset=apply_ruleset)
        def apply_f(f: Callable[[A], A], x: A) -> A:
            return f(x)

        @ruleset
        def my_ruleset(a: A):
            yield rewrite(transform_a(a)).to(apply_f(lambda x: my_transform_a(x), a))

        assert check_eq(transform_a(A()), my_transform_a(A()), (my_ruleset | apply_ruleset) * 10)

    def test_default_rewrite_used(self):
        """
        Verify that if I use a function in a default definition, it will use the the ruleset of the
        function that is being defined.
        """
        r = ruleset()

        @function
        def my_transform_a(a: A) -> A: ...

        @function(ruleset=r)
        def apply_f(f: Callable[[A], A], x: A) -> A:
            return f(x)

        @function(ruleset=r)
        def transform_a(a: A) -> A:
            return apply_f(lambda x: my_transform_a(x), a)

        assert check_eq(transform_a(A()), my_transform_a(A()), r * 10)

    def test_multiple_lambdas(self):
        """
        Verify that multiple lambdas can be added and the name won't conflict
        """

        @function
        def apply_f(f: Callable[[A], A], x: A) -> A:
            return f(x)

        alt_a = constant("alt_a", A)

        egraph = EGraph()
        x = egraph.let("x", apply_f(lambda x: A(), A()))
        y = egraph.let("y", apply_f(lambda x: alt_a, A()))
        egraph.run(10)
        egraph.check(eq(x).to(A()))
        egraph.check(eq(y).to(alt_a))

    def test_name_is_body(self):
        @function
        def higher_order(f: Callable[[A], A]) -> A: ...

        @function
        def transform_a(a: A) -> A: ...

        v = higher_order(lambda a: transform_a(a))
        assert str(v) == "higher_order(lambda a: transform_a(a))"

    def test_multiple_same(self):
        """
        Test that multiple lambdas with the same body.
        """

        @function
        def apply_f(f: Callable[[A], A], x: A) -> A:
            return f(x)

        egraph = EGraph()
        x = egraph.let("x", apply_f(lambda x: A(), A()))
        y = egraph.let("y", apply_f(lambda x: A(), A()))
        egraph.run(10)
        egraph.check(eq(x).to(A()))
        egraph.check(eq(y).to(A()))

    def test_multiple_same_different_type(self):
        """
        Test that multiple lambdas with the same body but different types work.
        """

        @function
        def apply_A(f: Callable[[A], A], x: A) -> A:
            return f(x)

        @function
        def apply_C(f: Callable[[C], C], x: C) -> C:
            return f(x)

        egraph = EGraph()
        x = egraph.let("x", apply_A(lambda x: x, A()))
        y = egraph.let("y", apply_C(lambda x: x, C()))
        egraph.run(10)
        egraph.check(eq(x).to(A()))
        egraph.check(eq(y).to(C()))
