# # 03 - E-class Analysis
#
# _[This tutorial is translated from egglog.](https://egraphs-good.github.io/egglog-tutorial/03-analysis.html)_
#
# Datalog is a relational language for deductive reasoning. In the last lesson, we write our first
# equality saturation program in egglog, but you can also write rules for deductive reasoning a la Datalog.
# In this lesson, we will write several classic Datalog programs in egglog. One of the benifits
# of egglog being a language for program optimization is that it can talk about terms natively,
# so in egglog we get Datalog with terms for free.
#
# In this lesson, we learn how to combine the power of equality saturation and Datalog.
# We will show how we can define program analyses using Datalog-style deductive reasoning,
# how EqSat-style rewrite rules can make the program analyses more accurate, and how
# accurate program analyses can enable more powerful rewrites.
#
# Our first example will continue with the `path` example in [lesson 2](./tut_2_datalog).
# In this case, there is a path from `e1` to `e2` if `e1` is less than or equal to `e2`.

# +
# mypy: disable-error-code="empty-body"
from __future__ import annotations
from collections.abc import Iterable
from typing import TypeAlias
from egglog import *


class Num(Expr):
    # in this example we use big 🐀 to represent numbers
    # you can find a list of primitive types in the standard library in [`builtins.py`](https://github.com/egraphs-good/egglog-python/blob/main/python/egglog/builtins.py)
    def __init__(self, value: BigRatLike) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: NumLike) -> Num: ...
    def __radd__(self, other: NumLike) -> Num: ...
    def __mul__(self, other: NumLike) -> Num: ...
    def __rmul__(self, other: NumLike) -> Num: ...
    def __truediv__(self, other: NumLike) -> Num: ...
    def __le__(self, other: NumLike) -> Unit: ...

    @property
    def non_zero(self) -> Unit: ...


NumLike: TypeAlias = Num | StringLike | BigRatLike
converter(BigRat, Num, Num)
converter(String, Num, Num.var)
# -

# Let's define some BigRat constants that will be useful later.

zero = BigRat(0, 1)
one = BigRat(1, 1)
two = BigRat(2, 1)


# We define a less-than-or-equal-to relation between two expressions.
# `a.__le__(b)` means that `a <= b` for all possible values of variables.

# We define rules to deduce the `le` relation.

egraph = EGraph()


@egraph.register
def _(
    e1: Num, e2: Num, e3: Num, n1: BigRat, n2: BigRat, x: String, e1a: Num, e1b: Num, e2a: Num, e2b: Num
) -> Iterable[RewriteOrRule]:
    #  We start with transitivity of `<=`:
    yield rule(e1 <= e2, e2 <= e3).then(e1 <= e3)
    # Base case for for `Num`:
    yield rule(e1 == Num(n1), e2 == Num(n2), n1 <= n2).then(e1 <= e2)
    # Base case for `Var`:`
    yield rule(e1 == Num.var(x)).then(e1 <= e1)  # noqa: PLR0124
    # Recursive case for `Add`:
    yield rule(
        e1 == (e1a + e1b),
        e2 == (e2a + e2b),
        e1a <= e2a,
        e1b <= e2b,
    ).then(e1 <= e2)


# Note that we have not defined any rules for multiplication. This would require a more complex
# analysis on the positivity of the expressions.
#
# On the other hand, these rules by themselves are pretty weak. For example, they cannot deduce `x + 1 <= 2 + x`.
# But EqSat-style axiomatic rules make these rules more powerful:


@egraph.register
def _(x: Num, y: Num, z: Num, a: BigRat, b: BigRat) -> Iterable[RewriteOrRule]:
    yield birewrite(x + (y + z)).to((x + y) + z)
    yield birewrite(x * (y * z)).to((x * y) * z)
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)
    yield rewrite(x * (y + z)).to((x * y) + (x * z))
    yield rewrite(x + zero).to(x)
    yield rewrite(x * one).to(x)
    yield rewrite(Num(a) + Num(b)).to(Num(a + b))
    yield rewrite(Num(a) * Num(b)).to(Num(a * b))


# To check our rules

expr1 = egraph.let("expr1", Num.var("y") + (Num(two) + "x"))
expr2 = egraph.let("expr2", Num.var("x") + Num.var("y") + Num(one) + Num(two))
egraph.check_fail(expr1 <= expr2)
egraph.run(run().saturate())
egraph.check(expr1 <= expr2)
egraph

# A useful special case of the <= analysis is if an expression is upper bounded
# or lower bounded by certain numbers, i.e., interval analysis:


# +
@function(merge=lambda old, new: old.min(new))
def upper_bound(e: Num) -> BigRat: ...


@function(merge=lambda old, new: old.max(new))
def lower_bound(e: Num) -> BigRat: ...


# -

# In the above functions, unlike `<=`, we define upper bound and lower bound as functions from
# expressions to a unique number.
# This is because we are always interested in the tightest upper bound
# and lower bounds, so


@egraph.register
def _(e: Num, n: BigRat) -> Iterable[RewriteOrRule]:
    yield rule(e <= Num(n)).then(set_(upper_bound(e)).to(n))
    yield rule(Num(n) <= e).then(set_(lower_bound(e)).to(n))


# We can define more specific rules for obtaining the upper and lower bounds of an expression
# based on the upper and lower bounds of its children.


@egraph.register
def _(e: Num, e1: Num, e2: Num, u1: BigRat, u2: BigRat, l1: BigRat, l2: BigRat) -> Iterable[RewriteOrRule]:
    yield rule(
        e == (e1 + e2),
        upper_bound(e1) == u1,
        upper_bound(e2) == u2,
    ).then(set_(upper_bound(e)).to(u1 + u2))
    yield rule(
        e == (e1 + e2),
        lower_bound(e1) == l1,
        lower_bound(e2) == l2,
    ).then(set_(lower_bound(e)).to(l1 + l2))
    # ... and the giant rule for multiplication:
    yield rule(
        e == (e1 * e2),
        l1 == lower_bound(e1),
        l2 == lower_bound(e2),
        u1 == upper_bound(e1),
        u2 == upper_bound(e2),
    ).then(
        set_(lower_bound(e)).to((l1 * l2).min((l1 * u2).min((u1 * l2).min(u1 * u2)))),
        set_(upper_bound(e)).to((l1 * l2).max((l1 * u2).max((u1 * l2).max(u1 * u2)))),
    )
    # Similarly,
    yield rule(e == e1 * e1).then(set_(lower_bound(e)).to(zero))


# The interval analysis is not only useful for numerical tools like [Herbie](https://herbie.uwplse.org/),
# but it can also guard certain optimization rules, making EqSat-based rewriting more powerful!
#
# For example, we are interested in non-zero expressions


@egraph.register
def _(e: Num, e2: Num) -> Iterable[RewriteOrRule]:
    yield rule(lower_bound(e) > zero).then(e.non_zero)
    yield rule(upper_bound(e) < zero).then(e.non_zero)
    yield rewrite(e / e).to(Num(one), e.non_zero)
    yield rewrite(e * (e2 / e)).to(e2, e.non_zero)


# This non-zero analysis lets us optimize expressions that contain division safely.
# 2 * (x / (1 + 2 / 2)) is equivalent to x

expr3 = egraph.let("expr3", Num(two) * (Num.var("x") / (Num(one) + (Num(two) / Num(two)))))
expr4 = egraph.let("expr4", Num.var("x"))
egraph.check_fail(expr3 == expr4)
egraph.run(run().saturate())
egraph.check(expr3 == expr4)

# (x + 1)^2 + 2

expr5 = egraph.let("expr5", (Num.var("x") + Num(one)) * (Num.var("x") + Num(one)) + Num(two))
expr6 = egraph.let("expr6", expr5 / expr5)
egraph.run(run().saturate())
egraph.check(expr6 == Num(one))

# ## Debugging tips!

# `function_size` is used to return the size of a table and `all_function_sizes` for to return the size of every table.
# This is useful for debugging performance, by seeing how the table sizes evolve as the iteration count increases.

egraph.function_size(Num.__le__)

egraph.all_function_sizes()

# `function_values` extracts every instance of a constructor, function, or relation in the e-graph.
# It takes the maximum number of instances to extract as a second argument, so as not to spend time
# printing millions of rows. `function_values` is particularly useful when debugging small e-graphs.

list(egraph.function_values(Num.__le__, 15))

# `extract_multiple` can also be used to extract that many different "variants" of the
# first argument. This is useful when trying to figure out why one e-class is failing to be unioned with another.

egraph.extract_multiple(expr3, 3)
