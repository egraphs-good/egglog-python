# mypy: disable-error-code="empty-body"
"""
Higher Order Functions
======================
"""

from __future__ import annotations

from collections.abc import Callable

from egglog import *


class Math(Expr):
    def __init__(self, i: i64Like) -> None: ...
    def __add__(self, other: Math) -> Math: ...


class MathList(Expr):
    def __init__(self) -> None: ...
    def append(self, i: Math) -> MathList: ...
    def map(self, f: Callable[[Math], Math]) -> MathList: ...


@ruleset
def math_ruleset(i: i64, j: i64, xs: MathList, x: Math, f: Callable[[Math], Math]):
    yield rewrite(Math(i) + Math(j)).to(Math(i + j))
    yield rewrite(xs.append(x).map(f)).to(xs.map(f).append(f(x)))
    yield rewrite(MathList().map(f)).to(MathList())


@function(ruleset=math_ruleset)
def incr_list(xs: MathList) -> MathList:
    return xs.map(lambda x: x + Math(1))


egraph = EGraph()
y = egraph.let("y", incr_list(MathList().append(Math(1)).append(Math(2))))
egraph.run(math_ruleset.saturate())
egraph.check(eq(y).to(MathList().append(Math(2)).append(Math(3))))

egraph
