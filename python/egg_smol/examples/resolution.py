"""
Resolution theorem proving.
===========================
"""
from __future__ import annotations

from typing import ClassVar

from egg_smol import *

egraph = EGraph()


@egraph.class_
class Bool(BaseExpr):
    FALSE: ClassVar[Bool]

    def __or__(self, other: Bool) -> Bool:  # type: ignore[empty-body]
        ...

    def __invert__(self) -> Bool:  # type: ignore[empty-body]
        ...


# Show off two ways of creating constants, either as top level values or as classvars
T = egraph.constant("T", Bool)
F = Bool.FALSE

p, a, b, c, as_, bs = vars_("p a b c as bs", Bool)
egraph.register(
    # clauses are assumed in the normal form (or a (or b (or c False)))
    set_(~F).to(T),
    set_(~T).to(F),
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
        set_(as_ | bs).to(T),
    ),
)


# Example predicate
@egraph.function
def pred(x: i64Like) -> Bool:  # type: ignore[empty-body]
    ...


p0 = egraph.define("p0", pred(0))
p1 = egraph.define("p1", pred(1))
p2 = egraph.define("p2", pred(2))
egraph.register(
    set_(p1 | (~p2 | F)).to(T),
    set_(p2 | (~p0 | F)).to(T),
    set_(p0 | (~p1 | F)).to(T),
    union(p1).with_(F),
    set_(~p0 | (~p1 | (p2 | F))).to(T),
)
egraph.run(10)
egraph.check(T != F)
egraph.check(eq(p0).to(F))
egraph.check(eq(p2).to(F))
