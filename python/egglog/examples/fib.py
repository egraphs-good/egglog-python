# mypy: disable-error-code="empty-body"
"""
Fibonacci numbers example
=========================
"""

from __future__ import annotations

from egglog import *


@function
def fib(x: i64Like) -> i64: ...


f0, f1, x = vars_("f0 f1 x", i64)
check(
    eq(fib(i64(7))).to(i64(21)),
    ruleset(
        rule(
            eq(f0).to(fib(x)),
            eq(f1).to(fib(x + 1)),
        ).then(set_(fib(x + 2)).to(f0 + f1)),
    )
    * 7,
    set_(fib(0)).to(i64(1)),
    set_(fib(1)).to(i64(1)),
)
