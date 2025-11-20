# mypy: disable-error-code="empty-body"
"""
Multiset example based off of egglog version
============================================
"""

from __future__ import annotations

from egglog import *


class Math(Expr):
    def __init__(self, x: i64Like) -> None: ...
    def __add__(self, other: MathLike) -> Math: ...
    def __radd__(self, other: MathLike) -> Math: ...
    def __mul__(self, other: MathLike) -> Math: ...
    def __rmul__(self, other: MathLike) -> Math: ...


MathLike = Math | i64Like
converter(i64, Math, Math)


@function
def sum(xs: MultiSetLike[Math, MathLike]) -> Math: ...


@function
def product(xs: MultiSetLike[Math, MathLike]) -> Math: ...


@function
def square(x: Math) -> Math: ...


x = constant("x", Math)
expr1 = 2 * (x + 3)
expr2 = 6 + 2 * x


@ruleset
def math_ruleset(a: Math, b: Math, c: Math, i: i64, j: i64, xs: MultiSet[Math], ys: MultiSet[Math], zs: MultiSet[Math]):
    yield rewrite(a + b).to(sum(MultiSet(a, b)))
    yield rewrite(a * b).to(product(MultiSet(a, b)))
    # 0 or 1 elements sums/products also can be extracted back to numbers
    yield rule(a == sum(xs), xs.length() == i64(1)).then(a == xs.pick())
    yield rule(a == product(xs), xs.length() == i64(1)).then(a == xs.pick())
    yield rewrite(sum(MultiSet[Math]())).to(Math(0))
    yield rewrite(product(MultiSet[Math]())).to(Math(1))
    # distributive rule (a * (b + c) = a*b + a*c)
    yield rule(
        b == product(ys),
        a == sum(xs),
        ys.contains(a),
        ys.length() > 1,
        zs == ys.remove(a),
    ).then(
        b == sum(xs.map(lambda x: product(zs.insert(x)))),
    )
    # constants
    yield rule(
        a == sum(xs),
        b == Math(i),
        xs.contains(b),
        ys == xs.remove(b),
        c == Math(j),
        ys.contains(c),
    ).then(
        a == sum(ys.remove(c).insert(Math(i + j))),
    )
    yield rule(
        a == product(xs),
        b == Math(i),
        xs.contains(b),
        ys == xs.remove(b),
        c == Math(j),
        ys.contains(c),
    ).then(
        a == product(ys.remove(c).insert(Math(i * j))),
    )


egraph = EGraph()
egraph.register(expr1, expr2)
egraph.run(math_ruleset.saturate())
egraph.check(expr1 == expr2)
