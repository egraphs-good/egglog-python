from __future__ import annotations

from egg_smol import *


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
    f7 = egraph.define("f7", fib(7), cost=2)
    egraph.run(14)
    egraph.check(eq(f7).to(Num(13)))
    res = egraph.extract(f7)
    assert expr_parts(res) == expr_parts(Num(13))


def test_resolution():
    egraph = EGraph()

    @egraph.class_
    class Bool(BaseExpr):
        @egraph.method(egg_fn="True")
        @classmethod
        def true(cls) -> Bool:  # type: ignore[empty-body]
            ...

        @egraph.method(egg_fn="False")
        @classmethod
        def false(cls) -> Bool:  # type: ignore[empty-body]
            ...

        @egraph.method(egg_fn="or")
        def __or__(self, other: Bool) -> Bool:  # type: ignore[empty-body]
            ...

        @egraph.method(egg_fn="neg")
        def __invert__(self) -> Bool:  # type: ignore[empty-body]
            ...

    T = Bool.true()
    F = Bool.false()

    p, a, b, c, as_, bs = vars_("p a b c as bs", Bool)
    egraph.register(
        # clauses are assumed in the normal form (or a (or b (or c False)))
        set_(~F).to(T),
        set_(~T).to(F),
        # "Solving" negation equations
        rule(eq(~p).to(T)).then(union(p).with_(F)),
        rule(eq(~p).to(F)).then(union(p).with_(T)),
        # canonicalize associtivity. "append" for clauses terminate with false
        rewrite((a | b) | c).to(a | (b | c)),
        # commutativity
        rewrite(a | (b | c)).to(b | (a | c)),
        # absoprtion
        rewrite(a | (a | b)).to(a | b),
        rewrite(a | (~a | b)).to(T),
        # Simplification
        rewrite(F | a).to(a),
        rewrite(a | F).to(a),
        rewrite(T | a).to(T),
        rewrite(a | T).to(T),
        # unit propagation
        # This is kind of interesting actually.
        # Looks a bit like equation solving
        rule(eq(T).to(p | F)).then(union(p).with_(T)),
        # resolution
        # This counts on commutativity to bubble everything possible up to the front of the clause.
        rule(
            eq(T).to(a | as_),
            eq(T).to(~a | bs),
        ).then(
            set_(as_ | bs).to(T),
        ),
    )

    # Example predicate
    @egraph.function
    def pred(x: i64Like) -> Bool:  # type: ignore[empty-body]
        ...

    p0 = egraph.define("p0", pred(0))
    p1 = egraph.define("p1", pred(1))
    p2 = egraph.define("p2", pred(2))
    egraph.register(
        set_(p1 | (~p2 | F)).to(T),
        set_(p2 | (~p0 | F)).to(T),
        set_(p0 | (~p1 | F)).to(T),
        union(p1).with_(F),
        set_(~p0 | (~p1 | (p2 | F))).to(T),
    )
    egraph.run(10)
    egraph.check(T != F)
    egraph.check(eq(p0).to(F))
    egraph.check(eq(p2).to(F))


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
