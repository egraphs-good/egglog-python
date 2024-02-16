"""
Schedule demo
=============
"""

from __future__ import annotations

from egglog import *

left = relation("left", i64)
right = relation("right", i64)


x, y = vars_("x y", i64)
step_left = ruleset(
    rule(
        left(x),
        right(x),
    ).then(left(x + 1)),
)
step_right = ruleset(
    rule(
        left(x),
        right(y),
        eq(x).to(y + 1),
    ).then(right(x)),
)

egraph = EGraph()
egraph.register(left(i64(0)), right(i64(0)))
egraph.run((step_right.saturate() + step_left.saturate()) * 10)
egraph.check(left(i64(10)), right(i64(9)))
egraph.check_fail(left(i64(11)), right(i64(10)))
egraph
