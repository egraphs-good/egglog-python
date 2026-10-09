"""Ordinary user class annotations retain exact public member types."""

from __future__ import annotations

from typing import ClassVar, assert_type

from egglog import Expr, Pair, Vec, f64, i64, i64Like, method


class Node(Expr):
    ROOT: ClassVar[Node]
    COUNT: ClassVar[i64]

    def __init__(self) -> None: ...

    @method(egg_fn="exact member")
    def score(self, value: i64Like = 1) -> i64: ...

    @property
    def weight(self) -> f64: ...

    @classmethod
    def make(cls, value: i64Like = 1) -> Node: ...

    def pairs(self, values: Vec[Pair[i64, f64]]) -> Vec[Pair[i64, f64]]: ...


node = Node()
assert_type(Node.ROOT, Node)
assert_type(Node.COUNT, i64)
assert_type(node.score(), i64)
assert_type(node.score(value=i64(1) + 2), i64)
assert_type(Node.score(node, 2), i64)
assert_type(node.weight, f64)
assert_type(Node.make(), Node)
assert_type(Node.make(value=3), Node)
assert_type(node.pairs(Vec(Pair(i64(1), f64(2)))), Vec[Pair[i64, f64]])
