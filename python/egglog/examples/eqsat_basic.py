# mypy: disable-error-code="empty-body"
"""
Basic equality saturation example.
==================================
"""
from __future__ import annotations

from egglog import *

egraph = EGraph()


class Num(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    @classmethod
    def var(cls, name: StringLike) -> Num:
        ...

    def __add__(self, other: Num) -> Num:
        ...

    def __mul__(self, other: Num) -> Num:
        ...


expr1 = Num(2) * (Num.var("x") + Num(3))
expr2 = Num(6) + Num(2) * Num.var("x")

a, b, c = vars_("a b c", Num)
i, j = vars_("i j", i64)

check(
    # Check that these expressions are equal
    eq(expr1).to(expr2),
    # After running these rules, up to ten times
    ruleset(
        rewrite(a + b).to(b + a),
        rewrite(a * (b + c)).to((a * b) + (a * c)),
        rewrite(Num(i) + Num(j)).to(Num(i + j)),
        rewrite(Num(i) * Num(j)).to(Num(i * j)),
    )
    * 10,
    # On these two initial expressions
    expr1,
    expr2,
)
