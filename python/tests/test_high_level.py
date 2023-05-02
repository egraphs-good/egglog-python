from __future__ import annotations

import importlib
import pathlib
from typing import ClassVar

import pytest
from egg_smol import *

EXAMPLE_FILES = list((pathlib.Path(__file__).parent / "../egg_smol/examples").glob("*.py"))


# Test all files in the `examples` directory by importing them in this parametrized test
@pytest.mark.parametrize("name", [f.stem for f in EXAMPLE_FILES if f.stem != "__init__"])
def test_example(name):
    importlib.import_module(f"egg_smol.examples.{name}")


class TestExprStr:
    def test_unwrap_lit(self):
        assert str(i64(1) + 1) == "i64(1) + 1"
        assert str(i64(1).max(2)) == "i64(1).max(2)"


def test_eqsat_basic():
    egraph = EGraph()

    @egraph.class_
    class Math(BaseExpr):
        def __init__(self, value: i64Like) -> None:
            ...

        @classmethod
        def var(cls, v: StringLike) -> Math:  # type: ignore[empty-body]
            ...

        def __add__(self, other: Math) -> Math:  # type: ignore[empty-body]
            ...

        def __mul__(self, other: Math) -> Math:  # type: ignore[empty-body]
            ...

    # expr1 = 2 * (x + 3)
    expr1 = egraph.define("expr1", Math(2) * (Math.var("x") + Math(3)))

    # expr2 = 6 + 2 * x
    expr2 = egraph.define("expr2", Math(6) + Math(2) * Math.var("x"))

    a, b, c = vars_("a b c", Math)
    x, y = vars_("x y", i64)

    egraph.register(
        rewrite(a + b).to(b + a),
        rewrite(a * (b + c)).to((a * b) + (a * c)),
        rewrite(Math(x) + Math(y)).to(Math(x + y)),
        rewrite(Math(x) * Math(y)).to(Math(x * y)),
    )

    egraph.run(10)

    egraph.check(eq(expr1).to(expr2))


def test_fib():
    egraph = EGraph()

    @egraph.function
    def fib(x: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    f0, f1, x = vars_("f0 f1 x", i64)
    egraph.register(
        set_(fib(0)).to(i64(1)),
        set_(fib(1)).to(i64(1)),
        rule(
            eq(f0).to(fib(x)),
            eq(f1).to(fib(x + 1)),
        ).then(set_(fib(x + 2)).to(f0 + f1)),
    )
    egraph.run(7)
    egraph.check(eq(fib(i64(7))).to(i64(21)))


@pytest.mark.xfail
def test_fib_demand():
    egraph = EGraph()

    @egraph.class_
    class Num(BaseExpr):
        def __init__(self, i: i64Like) -> None:
            ...

        def __add__(self, other: Num) -> Num:  # type: ignore[empty-body]
            ...

    @egraph.function
    def fib(x: i64Like) -> Num:  # type: ignore[empty-body]
        ...

    a, b, x = vars_("a b x", i64)
    f = var("f", Num)
    egraph.register(
        rewrite(Num(a) + Num(b)).to(Num(a + b)),
        rule(eq(f).to(fib(x)), x > 1).then(set_(fib(x)).to(fib(x - 1) + fib(x - 2))),
        set_(fib(0)).to(Num(0)),
        set_(fib(1)).to(Num(1)),
    )
    f7 = egraph.define("f7", fib(7))
    egraph.run(14)
    egraph.check(eq(f7).to(Num(13)))
    res = egraph.extract(f7)
    assert expr_parts(res) == expr_parts(Num(13))


def test_push_pop():
    egraph = EGraph()

    @egraph.function(merge=lambda old, new: old.max(new))
    def foo() -> i64:  # type: ignore[empty-body]
        ...

    egraph.register(set_(foo()).to(i64(1)))
    egraph.check(eq(foo()).to(i64(1)))

    with egraph:
        egraph.register(set_(foo()).to(i64(2)))
        egraph.check(eq(foo()).to(i64(2)))

    egraph.check(eq(foo()).to(i64(1)))


def test_constants():
    egraph = EGraph()

    one = egraph.constant("one", i64)
    egraph.register(set_(one).to(i64(1)))
    egraph.check(eq(one).to(i64(1)))


def test_class_vars():
    egraph = EGraph()

    @egraph.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[i64]

    egraph.register(set_(Numeric.ONE).to(i64(1)))
    egraph.check(eq(Numeric.ONE).to(i64(1)))


def test_simplify_constant():
    egraph = EGraph()

    @egraph.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[Numeric]

        def __init__(self, v: i64) -> None:
            pass

    assert expr_parts(egraph.simplify(Numeric.ONE, 10)) == expr_parts(Numeric.ONE)

    egraph.register(union(Numeric.ONE).with_(Numeric(i64(1))))

    assert expr_parts(egraph.simplify(Numeric.ONE, 10)) == expr_parts(Numeric(i64(1)))


def test_extract_constant_twice():
    # Sometimes extrcting a constant twice will give an error
    egraph = EGraph()

    @egraph.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[Numeric]

    egraph.extract(Numeric.ONE)
    egraph.extract(Numeric.ONE)
