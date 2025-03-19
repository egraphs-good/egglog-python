# mypy: disable-error-code="empty-body"
"""
Multiset example based off of egglog version
============================================
"""

from __future__ import annotations

from collections import Counter

from egglog import *


class Math(Expr):
    def __init__(self, x: i64Like) -> None: ...


@function
def square(x: Math) -> Math: ...


@ruleset
def math_ruleset(i: i64):
    yield rewrite(square(Math(i))).to(Math(i * i))


egraph = EGraph()

xs = MultiSet(Math(1), Math(2), Math(3))
egraph.register(xs)

with egraph.set_current():
    assert xs == MultiSet(Math(1), Math(3), Math(2))
    assert xs != MultiSet(Math(1), Math(1), Math(2), Math(3))

    assert Counter(xs) == Counter({Math(1): 1, Math(2): 1, Math(3): 1})

    inserted = MultiSet(Math(1), Math(2), Math(3), Math(4))
    egraph.register(inserted)
    assert xs.insert(Math(4)) == inserted

    assert xs.contains(Math(1))
    assert xs.not_contains(Math(4))
    assert Math(1) in xs
    assert Math(4) not in xs

    assert xs.remove(Math(1)) == MultiSet(Math(2), Math(3))

    assert xs.length() == i64(3)
    assert len(xs) == 3

    assert MultiSet(Math(1), Math(1)).length() == i64(2)

    assert MultiSet(Math(1)).pick() == Math(1)

    mapped = xs.map(square)
    egraph.register(mapped)
    egraph.run(math_ruleset)
    assert mapped == MultiSet(Math(1), Math(4), Math(9))

    assert xs + xs == MultiSet(Math(1), Math(2), Math(3), Math(1), Math(2), Math(3))
