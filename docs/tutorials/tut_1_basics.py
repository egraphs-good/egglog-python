# # 01 - Basics of Equality Saturation
#
# _[This tutorial is translated from egglog.](https://egraphs-good.github.io/egglog-tutorial/01-basics.html)_
#
# In this tutorial, we will build an optimizer for a subset of linear algebra using egglog.
# We will start by optimizing simple integer arithmetic expressions.
# Our initial DSL supports constants, variables, addition, and multiplication.

# +
# mypy: disable-error-code="empty-body"
from __future__ import annotations
from typing import TypeAlias
from egglog import *


class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __add__(self, other: ExprLike) -> Num: ...
    def __mul__(self, other: ExprLike) -> Num: ...

    # Support inverse operations for convenience
    # they will be translated to non-reversed ones
    def __radd__(self, other: ExprLike) -> Num: ...
    def __rmul__(self, other: ExprLike) -> Num: ...


ExprLike: TypeAlias = Num | StringLike | i64Like
converter(i64, Num, Num)
converter(str, Num, Num.var)
# -

# Now, let's define some simple expressions.

x = Num.var("x")
expr1 = 2 * (x * 3)
expr2 = 6 * x

# You should see an e-graph with two expressions.

egraph = EGraph()
egraph.register(expr1, expr2)
egraph

# We can print the values of the expressions as well to see their fully expanded forms.

String("Hello, world!")

i64(42)

expr1

expr2

# We can use the `check` commands to check properties of our e-graph.

x, y = vars_("x y", Num)
assert egraph.check_bool(expr1 == x * y)

# This checks if `expr1` is equivalent to some expression `x * y`, where `x` and `y` are
# variables that can be mapped to any `Num` expression in the e-graph.
#
# Checks can fail. For example the following check fails because `expr1` is not equivalent to
# `x + y` for any `x` and `y` in the e-graph.

assert not egraph.check_bool(expr1 == x + y)

# Let us define some rewrite rules over our small DSL.

egraph.register(rewrite(x + y).to(y + x))

# This rule asserts that addition is commutative. More concretely, this rules says, if the e-graph
# contains expressions of the form `x + y`, then the e-graph should also contain the
# expression `y + x`, and they should be equivalent.
#
# Similarly, we can define the associativity rule for addition.

z = var("z", Num)
egraph.register(rewrite(x + (y + z)).to((x + y) + z))

# This rule says, if the e-graph contains expressions of the form `x + (y + z)`, then the e-graph should also contain
# the expression `(x + y) + z`, and they should be equivalent.

# There are two subtleties to rules:
#
# 1. Defining a rule is different from running it. The following check would fail at this point
#    because the commutativity rule has not been run (we've inserted `x + 3` but not yet derived `3 + x`).

assert not egraph.check_bool((x + 3) == (3 + x))

# 2. Rules are not instantiated for every possible term; they are only instantiated for terms that are
#    in the e-graph. For instance, even if we ran the commutativity rule above, the following check would
#    still fail because the e-graph does not contain either of the terms `Num(-2) + Num(2)` or `Num(2) + Num(-2)`.

assert not egraph.check_bool(Num(-2) + 2 == Num(2) + -2)

# Let's also define commutativity and associativity for multiplication.

egraph.register(
    rewrite(x * y).to(y * x),
    rewrite(x * (y * z)).to((x * y) * z),
)

# `egglog` also defines a set of built-in functions over primitive types, such as `+` and `*`,
# and supports operator overloading, so the same operator can be used with different types.

egraph.extract(i64(1) + 2)

egraph.extract(String("1") + "2")

egraph.extract(f64(1.0) + 2.0)

# With primitives, we can define rewrite rules that talk about the semantics of operators.
# The following rules show constant folding over addition and multiplication.

a, b = vars_("a b", i64)
egraph.register(
    rewrite(Num(a) + Num(b)).to(Num(a + b)),
    rewrite(Num(a) * Num(b)).to(Num(a * b)),
)

# While we have defined several rules, the e-graph has not changed since we inserted the two
# expressions. To run rules we have defined so far, we can use `run`.

egraph.run(10)

# This tells `egglog` to run our rules for 10 iterations. More precisely, egglog runs the
# following pseudo code:
#
# ```
# for i in range(10):
#     for r in rules:
#         ms = r.find_matches(egraph)
#         for m in ms:
#             egraph = egraph.apply_rule(r, m)
#     egraph = rebuild(egraph)
# ```
#
# In other words, `egglog` computes all the matches for one iteration before making any
# updates to the e-graph. This is in contrast to an evaluation model where rules are immediately
# applied and the matches are obtained on demand over a changing e-graph.
#
# We can now look at the e-graph and see that that `2 * (x + 3)` and `6  + (2 * x)` are now in the same E-class.

egraph

# We can also check this fact explicitly

egraph.check(expr1 == expr2)
