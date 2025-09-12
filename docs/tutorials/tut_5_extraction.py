# # 05 - Extraction and Cost
#
# _[This tutorial is translated from egglog.](https://egraphs-good.github.io/egglog-tutorial/05-cost-model-and-extraction.html)_
#
#  In this lesson, we will learn how to use `run-schedule` to improve the performance of egglog.
#  We start by using the same language as the previous lesson.


#  In the previous sessions, we have seen examples of defining and analyzing syntactic terms in egglog.
#  After running the rewrite rules, the e-graph may contain a myriad of terms.
#  We often want to pick out one or a handful of terms for further processing.
#  Extraction is the process of picking out individual terms out of the many terms represented by an e-graph.
#  We have seen `extract` command in the previous sessions, which allows us to extract the optimal term from the e-graph.
#
#  Optimality needs to be defined with regard to some cost model.
#  A cost model is a function that assigns a cost to each term in the e-graph.
#  By default, `extract` uses AST size as its cost model and picks the term with the smallest cost.
#
#  In this session, we will show several ways of customizing the cost model in egglog.
#  Let's first see a simple example of setting costs with the `cost` argument.


#  Here we have the same `Num`` language but annotated with `cost` keywords.

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

    @method(cost=2)
    def __add__(self, other: NumLike) -> Num: ...
    @method(cost=10)
    def __mul__(self, other: NumLike) -> Num: ...

    # These will be translated to non-reversed ones
    def __radd__(self, other: NumLike) -> Num: ...
    def __rmul__(self, other: NumLike) -> Num: ...


NumLike: TypeAlias = Num | StringLike | i64Like
converter(i64, Num, Num)
converter(String, Num, Num.var)
# -


#  The default cost of a function is 1.
#  Intuitively, the additional `cost` attributes mark the multiplication operation as more expensive than addition.
#
#  Let's look at how cost is computed for a concrete term in the default tree cost model.

egraph = EGraph()
expr = egraph.let("expr", Num.var("x") * 2 + 1)

#  This term has a total cost of 18 because:
#
# ```python
# (
#     (
#         Num.var("x")  # cost = 1  (from Num.var) + 1  (from "x") = 2
#         *  # cost = 10 (from *) + 2  (from left operand) + 2 (from right operand) = 14
#         Num(2)  # cost = 1  (from Num) + 1  (from 2)   = 2
#     )
#     +  # cost = 2  (from +) + 14 (from left operand) + 2 (from right operand) = 18
#     Num(1)  # cost = 1  (from Num) + 1  (from 1)   = 2
# )
# ```
#
#
#  We can use the `extract` command to extract the lowest cost variant of the term.
#  For now it gives the only version that we just defined. We can also pass `include_cost=True` to see the cost of the extracted term.


egraph.extract(expr, include_cost=True)

#  Let's introduces more variants with rewrites


@egraph.register
def _(x: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x * 2).to(x + x)


egraph.run(1)
egraph.extract(expr, include_cost=True)


#  It now extracts the lower cost variant that correspondes to `x + x + 1`, which is equivalent to the original term.
#  If there are multiple variants of the same lowest cost, `extract` break ties arbitrarily.


#  ## Setting custom cost for e-nodes

#  The `cost` keyword sets an uniform additional cost to each appearance of corresponding constructor.
#  However, this is not expressive enough to cover the case where additional cost of an operation is not a fixed constant.
#  We can use the `set_cost` feature provided by `egglog-experimental` to get more fine-grained control of individual e-node's cost.

#  To show how this feature works, we define a toy language of matrices. This feature is automatically enabled for
#  constructors where it used on.


class Matrix(Expr):
    def __init__(self, rows: i64Like, cols: i64Like) -> None: ...
    def __matmul__(self, other: Matrix) -> Matrix: ...

    #  We also define two analyses for the number of rows and columns
    @property
    def row(self) -> i64: ...
    @property
    def col(self) -> i64: ...


@egraph.register
def _(x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64) -> Iterable[RewriteOrRule]:
    yield rule(x == Matrix(r, c)).then(set_(x.row).to(r), set_(x.col).to(c))
    yield rule(
        x == (y @ z),
        r == y.row,
        y.col == z.row,
        c == z.col,
    ).then(set_(x.row).to(r), set_(x.col).to(c))

    #  Now we define the cost of matrix multiplication as a product of the dimensions
    yield rule(
        y @ z,
        r == y.row,
        m == y.col,
        c == z.col,
    ).then(set_cost(y @ z, r * m * c))

    yield birewrite(x @ (y @ z)).to((x @ y) @ z)


#  Let's optimize matrix multiplication with this cost model

Mexpr = egraph.let("Mexpr", (Matrix(64, 8) @ Matrix(8, 256)) @ Matrix(256, 2))
egraph.run(5)

#  Thanks to our cost model, egglog is able to extract the equivalent program with lowest cost using the dimension information we provided:

egraph.extract(Mexpr)

egraph
