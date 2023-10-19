"""
Boolean data type example and test
==================================
"""

from __future__ import annotations

from egglog import *

T = Bool(True)
F = Bool(False)
egraph = EGraph()
egraph.check(eq(T & T).to(T))
egraph.check(eq(T & F).to(F))
egraph.check(eq(T | F).to(T))
egraph.check((T | F) != F)

egraph.check(eq(i64(1).bool_lt(2)).to(T))
egraph.check(eq(i64(2).bool_lt(1)).to(F))
egraph.check(eq(i64(1).bool_lt(1)).to(F))

egraph.check(eq(i64(1).bool_le(2)).to(T))
egraph.check(eq(i64(2).bool_le(1)).to(F))
egraph.check(eq(i64(1).bool_le(1)).to(T))

R = egraph.relation("R", i64)


@egraph.function
def f(i: i64Like) -> Bool:  # type: ignore[empty-body]
    ...


i = var("i", i64)
egraph.register(
    rule(R(i)).then(set_(f(i)).to(T)),
    R(i64(0)),
)

egraph.run(3)
egraph.check(eq(f(0)).to(T))
