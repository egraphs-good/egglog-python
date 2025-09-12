# # 02 - Datalog
#
# _[This tutorial is translated from egglog.](https://egraphs-good.github.io/egglog-tutorial/02-datalog.html)_
#
# Datalog is a relational language for deductive reasoning. In the last lesson, we write our first
# equality saturation program in egglog, but you can also write rules for deductive reasoning a la Datalog.
# In this lesson, we will write several classic Datalog programs in egglog. One of the benifits
# of egglog being a language for program optimization is that it can talk about terms natively,
# so in egglog we get Datalog with terms for free.


# mypy: disable-error-code="empty-body"
from __future__ import annotations
from collections.abc import Iterable
from typing import TypeAlias
from egglog import *

# Let's first define relations edge and path.
# We use `edge(a, b)` to mean the tuple (a, b) is in the `edge` relation.
# `edge(a, b)` means there are directed edges from `a` to `b`,
# and we will use it to compute the `path` relation,
# where `path(a, b)` means there is a (directed) path from `a` to `b`.

edge = relation("edge", i64, i64)
path = relation("path", i64, i64)

# We can insert edges into our relation by asserting facts:

egraph = EGraph()
egraph.register(
    edge(i64(1), i64(2)),
    edge(i64(2), i64(3)),
    edge(i64(3), i64(4)),
)

# Fact definitions are similar to definitions using `egraph.let` definitions in the last lesson,
# in that facts are immediately added to relations.
#
# Now let's tell egglog how to derive the `path` relation.
#
# First, if an edge from a to b exists, then it is already a proof
# that there exists a path from a to b.


@egraph.register
def _(a: i64, b: i64) -> Iterable[RewriteOrRule]:
    yield rule(edge(a, b)).then(path(a, b))


# A rule has the form `rule(atom1, atom2 ..).then(action1 action2 ..)`.
#
# For the rule we have just defined, the only atom is `path(a, b)`, which asks egglog to search
# for possible `a` and `b`s such that `path(a, b)` is a fact in the database.
#
# We call the first part the "query" of a rule, and the second part the "body" of a rule.
# In Datalog terminology, confusingly, the first part is called the "body" of the rule
# while the second part is called the "head" of the rule. This is because Datalog rules
# are usually written as `head :- body`. To avoid confusion, we will refrain from using
# Datalog terminology.
#
# The rule above defines the base case of the path relation. The inductive case reads as follows:
# if we know there is a path from `a` to `b`, and there is an edge from `b` to `c`, then
# there is also a path from `a` to `c`.
# This can be expressed as egglog rule below:


@egraph.register
def _(a: i64, b: i64, c: i64) -> Iterable[RewriteOrRule]:
    yield rule(path(a, b), edge(b, c)).then(path(a, c))


# Again, defining a rule does not mean running it in egglog, which may be a surprise to those familiar with Datalog.
# The user still needs to run the program.
# For instance, the following check would fail at this point.

egraph.check_fail(path(i64(1), i64(4)))

# But it passes after we run our rules for 10 iterations.

egraph.run(10)
egraph.check(path(i64(1), i64(4)))

# For many deductive rules, we do not know the number of iterations
# needed to reach a fixpoint. The egglog language provides the `saturate` scheduling primitive to run the rules until fixpoint.

egraph.run(run().saturate())
egraph

# We will cover more details about schedules later in the tutorial.


# Our last example determines whether there is a path from one node to another,
# but we don't know the details about the path.
# Let's slightly extend our program to obtain the length of the shortest path between any two nodes.


# +
@function
def edge_len(from_: i64Like, to: i64Like) -> i64: ...


@function(merge=lambda old, new: old.min(new))
def path_len(from_: i64Like, to: i64Like) -> i64: ...


# -

# Here, we use a new decorator called `function` to define a table that respects the functional dependency.
# A relation is just a function with output domain `Unit`.
# By defining `edge_len` and `path_len` with `function`, we can associate a length to each path.
#
# What happens it the same tuple of a function is mapped to two values?
# In the case of relation, this is easy: `Unit` only has one value, so the two values must be identical.
# But in general, that would be a violation of functional dependency, the property that `a = b` implies `f(a) = f(b)`.
# Egglog allows us to specify how to reconcile two values that are mapped from the same tuple using _merge expressions_.
#
# For instance, for `path`, the merge expression is `old.min(new)`. The merge function is passed two special values
# `old` and `new` that denotes the current output of the tuple and the output of the new, to-be-inserted value.
# The merge expression for `path` says that, when there are two paths from `a` to `b` with lengths `old` and `new`,
# we keep the shorter one, i.e., `old.min(new)`.
#
# For `edge_len`, we can define the merge expression the same as `path_len`, which means that we only keep the shortest edge
# if there are multiple edges. But we can also assert that `edge_len` does not have a merge expression,
# which is the default if none is provided.
# This means we don't expect there will be multiple edges between two nodes. More generally, it is the user's
# responsibility to ensure no tuples with conflicting output values exist. If a conflict happens, egglog will
# raise an error.

# Now let's insert the same edges as before, but we will assign a length to each edge. This is done using the `set` action,
# which takes a tuple and an output value:

egraph = EGraph()
egraph.register(
    set_(edge_len(1, 2)).to(i64(10)),
    set_(edge_len(2, 3)).to(i64(10)),
    set_(edge_len(1, 3)).to(i64(30)),
)

# Let us define the reflexive rule and transitive rule for the `path` function.
# In this rule, we use the `set` action to set the output value of the `path` function.
# On the query side, we use `a == b` (or `eq(a).to(b)` if equality is overloaded)  to bind the output value of a function.


@egraph.register
def _(a: i64, b: i64, c: i64, ab: i64, bc: i64) -> Iterable[RewriteOrRule]:
    yield rule(edge_len(a, b) == ab).then(set_(path_len(a, b)).to(ab))
    yield rule(path_len(a, b) == ab, edge_len(b, c) == bc).then(set_(path_len(a, c)).to(ab + bc))


# Let's run our rules and check we get the desired shortest path

egraph.run(run().saturate())
egraph.check(path_len(1, 3) == 20)
egraph


# Now let us combine the knowledge we have learned in lessons 1 and 2 to write a program that combines
# both equality saturation and Datalog.
#
# We reuse our path example, but this time the nodes are terms constructed using the `Node` constructor,
#
# We start by defining a new, union-able sort (created by sub-classing expression) with a new constructor


# +
class Node(Expr):
    def __init__(self, x: i64Like) -> None: ...
    def edge(self, other: NodeLike) -> Unit: ...
    def path(self, other: NodeLike) -> Unit: ...


NodeLike: TypeAlias = Node | i64Like

converter(i64, Node, Node)
# -

# Note: We could have equivalently written
#
# ```python
# class Node(Sort): ...
#
# @function
# def mk(x: i64Like) -> Node: ...
# edge_node = relation("edge_node", Node, Node)
# path_node = relation("path_node", Node, Node)
# ```
#
# All methods of classes are syntactic sugar for creating functions. Note that properties, classmethods, and classvars
# are also all supported ways of defining functions.


# +
egraph = EGraph()


@egraph.register
def _(x: Node, y: Node, z: Node) -> Iterable[RewriteOrRule]:
    yield rule(x.edge(y)).then(x.path(y))
    yield rule(x.path(y), y.edge(z)).then(x.path(z))


egraph.register(
    Node(1).edge(2),
    Node(2).edge(3),
    Node(3).edge(1),
    Node(5).edge(6),
)
egraph
# -

# Because we defined our nodes using custom expression `sort`, we can "union" two nodes.
# This makes them indistinguishable to rules in `egglog`.

egraph.register(union(Node(3)).with_(Node(5)))
egraph

# `union` is a new function here, but it is our old friend: `rewrite`s are implemented as rules whose
# actions are `union`s. For instance, `rewrite(a + b).to(b + a)` is lowered to the following
# rule:
#
# ```python
# rule(a + b == e).then(union(e).with_(b + a)))
# ```

# +
egraph.run(run().saturate())
egraph.check(
    Node(3).edge(6),
    Node(1).path(6),
)
egraph
# -

# We can also give a new meaning to equivalence by adding the following rule.


@egraph.register
def _(x: Node, y: Node) -> Iterable[RewriteOrRule]:
    yield rule(x.path(y), y.path(x)).then(union(x).with_(y))


# This rule says that if there is a path from `x` to `y` and from `y` to `x`, then
# `x` and `y` are equivalent.
# This rule allows us to tell if `a` and `b` are in the same connected component by checking
# `egraph.check(Node(a) == Node(b))`.

egraph.run(run().saturate())
egraph.check(
    Node(1) == Node(2),
    Node(1) == Node(3),
    Node(2) == Node(3),
)
egraph
