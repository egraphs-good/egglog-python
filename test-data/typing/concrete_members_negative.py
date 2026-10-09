from __future__ import annotations

from typing import ClassVar

from egglog import Expr, Pair, Vec, f64, i64, i64Like


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
