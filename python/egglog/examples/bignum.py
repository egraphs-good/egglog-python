# mypy: disable-error-code="empty-body"
"""
BigNum/BigRat Example
=====================
"""

from __future__ import annotations

from egglog import *

x = BigInt(-1234)
y = BigInt.from_string("2")
z = BigRat(x, y)

egraph = EGraph()

assert egraph.extract(z.numer.to_string()).eval() == "-617"


@function
def bignums(x: BigInt, y: BigInt) -> BigRat: ...


egraph.register(set_(bignums(x, y)).to(z))

c = var("c", BigRat)
a, b = vars_("a b", BigInt)
egraph.check(
    bignums(a, b) == c,
    c.numer == a >> 1,
    c.denom == b >> 1,
)
