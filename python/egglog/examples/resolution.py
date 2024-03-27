"""
Resolution theorem proving.
===========================
"""

from __future__ import annotations

from typing import ClassVar

from egglog import *

egraph = EGraph()


class Boolean(Expr):
    FALSE: ClassVar[Boolean]

    def __or__(self, other: Boolean) -> Boolean:  # type: ignore[empty-body]
        ...

    def __invert__(self) -> Boolean:  # type: ignore[empty-body]
        ...


# Show off two ways of creating constants, either as top level values or as classvars
T = constant("T", Boolean)
F = Boolean.FALSE

p, a, b, c, as_, bs = vars_("p a b c as bs", Boolean)
egraph.register(
    # clauses are assumed in the normal form (or a (or b (or c False)))
    union(~F).with_(T),
    union(~T).with_(F),
    # "Solving" negation equations
    rule(eq(~p).to(T)).then(union(p).with_(F)),
    rule(eq(~p).to(F)).then(union(p).with_(T)),
    # canonicalize associtivity. "append" for clauses terminate with false
    rewrite((a | b) | c).to(a | (b | c)),
    # commutativity
    rewrite(a | (b | c)).to(b | (a | c)),
    # absoprtion
    rewrite(a | (a | b)).to(a | b),
    rewrite(a | (~a | b)).to(T),
    # Simplification
    rewrite(F | a).to(a),
    rewrite(a | F).to(a),
    rewrite(T | a).to(T),
    rewrite(a | T).to(T),
    # unit propagation
    # This is kind of interesting actually.
    # Looks a bit like equation solving
    rule(eq(T).to(p | F)).then(union(p).with_(T)),
    # resolution
    # This counts on commutativity to bubble everything possible up to the front of the clause.
    rule(
        eq(T).to(a | as_),
        eq(T).to(~a | bs),
    ).then(
        union(as_ | bs).with_(T),
    ),
)


# Example predicate
@function
def pred(x: i64Like) -> Boolean:  # type: ignore[empty-body]
    ...


p0 = egraph.let("p0", pred(0))
p1 = egraph.let("p1", pred(1))
p2 = egraph.let("p2", pred(2))
egraph.register(
    union(p1 | (~p2 | F)).with_(T),
    union(p2 | (~p0 | F)).with_(T),
    union(p0 | (~p1 | F)).with_(T),
    union(p1).with_(F),
    union(~p0 | (~p1 | (p2 | F))).with_(T),
)
egraph.run(10)
egraph.check(ne(T).to(F))
egraph.check(eq(p0).to(F))
egraph.check(eq(p2).to(F))
egraph
