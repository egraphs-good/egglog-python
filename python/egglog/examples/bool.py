# mypy: disable-error-code="empty-body"
"""
Boolean data type example and test
==================================
"""

from __future__ import annotations

from egglog import *

T = Bool(True)
F = Bool(False)
check(eq(T & T).to(T))
check(eq(T & F).to(F))
check(eq(T | F).to(T))
check(ne(T | F).to(F))

check(eq(i64(1).bool_lt(2)).to(T))
check(eq(i64(2).bool_lt(1)).to(F))
check(eq(i64(1).bool_lt(1)).to(F))

check(eq(i64(1).bool_le(2)).to(T))
check(eq(i64(2).bool_le(1)).to(F))
check(eq(i64(1).bool_le(1)).to(T))

R = relation("R", i64)


@function
def f(i: i64Like) -> Bool: ...


i = var("i", i64)
check(
    eq(f(0)).to(T),
    ruleset(rule(R(i)).then(set_(f(i)).to(T))) * 3,
    R(i64(0)),
)
