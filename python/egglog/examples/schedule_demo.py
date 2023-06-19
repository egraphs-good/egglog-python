"""
Schedule demo
=============
"""
from __future__ import annotations

from egglog import *

egraph = EGraph()

left = egraph.relation("left", i64)
right = egraph.relation("right", i64)

egraph.register(left(i64(0)), right(i64(0)))

x, y = vars_("x y", i64)

step_left = egraph.ruleset("step-left")
egraph.register(rule(left(x), right(x), ruleset=step_left).then(left(x + 1)))

step_right = egraph.ruleset("step-right")
egraph.register(rule(left(x), right(y), eq(x).to(y + 1), ruleset=step_right).then(right(x)))

egraph.run(
    seq(
        run(step_right).saturate(),
        run(step_left).saturate(),
    )
    * 10
)
egraph.check(left(i64(10)), right(i64(9)))
egraph.check_fail(left(i64(11)), right(i64(10)))
egraph
