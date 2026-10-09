from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from egglog import Expr, Pair, Vec, f64, i64, i64Like, rewrite, rule, ruleset, var, vars_


class Node(Expr):
    COUNT: ClassVar[i64]

    def __init__(self) -> None: ...
    def score(self, value: i64Like = 1) -> i64: ...

    @property
    def weight(self) -> f64: ...

    @classmethod
    def make(cls, value: i64Like = 1) -> Node: ...

    def pairs(self, values: Vec[Pair[i64, f64]]) -> Vec[Pair[i64, f64]]: ...


node = Node()
node.score(f64(1))  # error: arg-type
node.score(1, 2)  # error: call-arg
node.score(unknown=1)  # error: call-arg
Node.score(i64(1), 1)  # error: arg-type
Node.make(f64(1))  # error: arg-type
Node.make(1, 2)  # error: call-arg
Node.COUNT()  # error: operator
node.weight()  # error: operator
wrong_count: f64 = Node.COUNT  # error: assignment
wrong_weight: i64 = node.weight  # error: assignment
wrong_result: i64 = Node.make()  # error: assignment
swapped = Vec(Pair(f64(1), i64(2)))
node.pairs(swapped)  # error: arg-type
wrong_variable: i64 = var("node", Node)  # error: assignment
wrong_variables: Iterable[i64] = vars_("left right", Node)  # error: arg-type
wrong_rule: i64 = rule().then()  # error: assignment
rewrite(var("node", Node)).to(i64(1))  # error: arg-type
rule("not a fact")  # error: arg-type
rule().then("not an action")  # error: arg-type
ruleset().append(node)  # error: arg-type
rewrite(node, ruleset="not a ruleset")  # error: arg-type
rule(node, ruleset=i64(1))  # error: arg-type
