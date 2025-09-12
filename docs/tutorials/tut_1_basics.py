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
from collections.abc import Iterable
from egglog import *


class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __add__(self, other: NumLike) -> Num: ...
    def __mul__(self, other: NumLike) -> Num: ...

    # Support inverse operations for convenience
    # they will be translated to non-reversed ones
    def __radd__(self, other: NumLike) -> Num: ...
    def __rmul__(self, other: NumLike) -> Num: ...


NumLike: TypeAlias = Num | StringLike | i64Like
# -


# The signature here takes `NumLike` not `Num` so that you can write `Num(1) + 2` instead of
# `Num(1) + Num(2)`. This is helpful for ease of use and also for compatibility when you are trying to
# create expressions that act like Python objects which perform upcasting.
#
# To support this, you must define conversions between primitive types and your expression types.
# When a value is passed into a function, it will find the type it should be converted to and
# transitively apply the conversions you have defined:

converter(i64, Num, Num)
converter(String, Num, Num.var)

# Now, let's define some simple expressions.

egraph = EGraph()
x = Num.var("x")
expr1 = egraph.let("expr1", 2 * (x * 3))
expr2 = egraph.let("expr2", 6 * x)

# You should see an e-graph with two expressions.

egraph

# We can `.extract` the values of the expressions as well to see their fully expanded forms.

egraph.extract(String("Hello, world!"))

egraph.extract(i64(42))

egraph.extract(expr1)

egraph.extract(expr2)

# We can use the `check` commands to check properties of our e-graph.

x, y = vars_("x y", Num)
egraph.check(expr1 == x * y)

# This checks if `expr1` is equivalent to some expression `x * y`, where `x` and `y` are
# variables that can be mapped to any `Num` expression in the e-graph.
#
# Checks can fail. For example the following check fails because `expr1` is not equivalent to
# `x + y` for any `x` and `y` in the e-graph.

egraph.check_fail(expr1 == x + y)

# Let us define some rewrite rules over our small DSL.


@egraph.register
def _add_comm(x: Num, y: Num):
    yield rewrite(x + y).to(y + x)


# This could also been written like:
#
# ```python
# x, y = vars_("x y", Num)
# egraph.register(rewrite(x + y).to(y + x))
# ```
#
# In this tutorial we will use the function form to define rewrites and rules, because then then we only
# have to write the variable names once as arguments and they are not leaked to the outer scope.


# This rule asserts that addition is commutative. More concretely, this rules says, if the e-graph
# contains expressions of the form `x + y`, then the e-graph should also contain the
# expression `y + x`, and they should be equivalent.
#
# Similarly, we can define the associativity rule for addition.


@egraph.register
def _add_assoc(x: Num, y: Num, z: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x + (y + z)).to((x + y) + z)


# This rule says, if the e-graph contains expressions of the form `x + (y + z)`, then the e-graph should also contain
# the expression `(x + y) + z`, and they should be equivalent.

# There are two subtleties to rules:
#
# 1. Defining a rule is different from running it. The following check would fail at this point
#    because the commutativity rule has not been run (we've inserted `x + 3` but not yet derived `3 + x`).

egraph.check_fail((x + 3) == (3 + x))

# 2. Rules are not instantiated for every possible term; they are only instantiated for terms that are
#    in the e-graph. For instance, even if we ran the commutativity rule above, the following check would
#    still fail because the e-graph does not contain either of the terms `Num(-2) + Num(2)` or `Num(2) + Num(-2)`.

egraph.check_fail(Num(-2) + 2 == Num(2) + -2)

# Let's also define commutativity and associativity for multiplication.


@egraph.register
def _mul(x: Num, y: Num, z: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x * y).to(y * x)
    yield rewrite(x * (y * z)).to((x * y) * z)


# `egglog` also defines a set of built-in functions over primitive types, such as `+` and `*`,
# and supports operator overloading, so the same operator can be used with different types.

egraph.extract(i64(1) + 2)

egraph.extract(String("1") + "2")

egraph.extract(f64(1.0) + 2.0)

# With primitives, we can define rewrite rules that talk about the semantics of operators.
# The following rules show constant folding over addition and multiplication.


@egraph.register
def _const_fold(a: i64, b: i64) -> Iterable[RewriteOrRule]:
    yield rewrite(Num(a) + Num(b)).to(Num(a + b))
    yield rewrite(Num(a) * Num(b)).to(Num(a * b))


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
