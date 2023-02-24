from __future__ import annotations
from egg_smol import *


egraph = EGraph()


@egraph.register
class Math(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    @classmethod
    def var(cls, v: str) -> Math:  # type: ignore[empty-body]
        ...

    def __add__(self, other: Math) -> Math:  # type: ignore[empty-body]
        ...

    def __mul__(self, other: Math) -> Math:  # type: ignore[empty-body]
        ...


# expr1 = 2 * (x + 3)
expr1 = Math(2) * (Math.var("x") + Math(3))

# expr2 = 6 + 2 * x
expr2 = Math(6) + Math(2) * Math.var("x")

a, b, c = var[Math].a, var[Math].b, var[Math].c

x, y = var[i64].i, var[i64].j

egraph.register(
    rewrite(a + b).to(b + a),
    rewrite(a * (b + c)).to((a * b) + (a * c)),
    rewrite(Math(x) + Math(y)).to(Math(x + y)),
    rewrite(Math(x) * Math(y)).to(Math(x * y)),
)

egraph.run(10)

egraph.check(eq(expr1).to(expr2))
