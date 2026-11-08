from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import ClassVar

from egglog import (
    EGraph,
    Expr,
    Maybe,
    Pair,
    UnstableFn,
    Vec,
    birewrite,
    convert,
    f64,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    run,
    seq,
    union,
    var,
    vars_,
)


class Node(Expr):
    COUNT: ClassVar[i64]

    def __init__(self) -> None: ...
    def score(self, value: i64Like = 1) -> i64: ...

    @property
    def weight(self) -> f64: ...

    @classmethod
    def make(cls, value: i64Like = 1) -> Node: ...

    def pairs(self, values: Vec[Pair[i64, f64]]) -> Vec[Pair[i64, f64]]: ...

    def strict(self, value: i64) -> i64: ...


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


class EagerDefaults(Expr):
    ROOT: ClassVar[Node] = Node()
    COUNT: ClassVar[i64] = i64(1)
    MISSING: ClassVar[Maybe[i64]] = Maybe[i64].none()


wrong_eager_root: i64 = EagerDefaults.ROOT  # error: assignment
wrong_eager_count: f64 = EagerDefaults.COUNT  # error: assignment
wrong_eager_missing: i64 = EagerDefaults.MISSING  # error: assignment
EagerDefaults.COUNT()  # error: operator


class EagerMembers(Expr):
    def __init__(self) -> None: ...

    def first(self, value: i64Like = 1) -> i64:
        return convert(value, i64) + 2

    @property
    def own(self) -> EagerMembers:
        return self

    @classmethod
    def make(cls) -> EagerMembers:
        return cls()


eager = EagerMembers()
eager.first(f64(1))  # error: arg-type
eager.first(unknown=1)  # error: call-arg
EagerMembers.first(node, 1)  # error: arg-type
eager.own()  # error: operator
EagerMembers.make(1)  # error: call-arg
wrong_eager_member: f64 = eager.first()  # error: assignment
wrong_eager_owner: Node = eager.own  # error: assignment

swapped = Vec(Pair(f64(1), i64(2)))
node.pairs(swapped)  # error: arg-type
wrong_variable: i64 = var("node", Node)  # error: assignment
wrong_variables: Iterable[i64] = vars_("left right", Node)  # error: arg-type
wrong_rule: i64 = rule().then()  # error: assignment
rewrite(var("node", Node)).to(i64(1))  # error: arg-type
birewrite(var("node", Node)).to(i64(1))  # error: arg-type
rule("not a fact")  # error: arg-type
rule().then("not an action")  # error: arg-type
ruleset().append(node)  # error: arg-type
rewrite(node, ruleset="not a ruleset")  # error: arg-type
birewrite(node, ruleset="not a ruleset")  # error: arg-type
rule(node, ruleset=i64(1))  # error: arg-type
union(node).with_(i64(1))  # error: arg-type
union(i64(1))  # error: type-var
wrong_union: i64 = union(node).with_(node)  # error: assignment


@function
def exact(value: i64) -> i64: ...


named = UnstableFn(exact)
named(f64(1))  # error: arg-type
named(1)  # error: arg-type
named()  # error: call-arg
named(i64(1), i64(2))  # error: call-arg
UnstableFn[i64, i64](Node.strict, i64(1))  # error: arg-type
UnstableFn[f64, i64](exact)  # error: arg-type
UnstableFn[i64, i64](i64(1))  # error: call-overload
wrong_callback: Callable[[i64], f64] = named.value  # error: assignment
wrong_wrapper: UnstableFn[f64, i64] = named  # error: assignment
UnstableFn(exact, i64(1))(i64(2))  # error: call-arg


class Defaults(Expr, ruleset=ruleset()):
    FIRST: ClassVar[Node] = Node()

    def __init__(self) -> None: ...

    def later(self, value: i64Like) -> Node: ...

    def earlier(self, value: i64Like = 1) -> Node:
        return self.later(value)

    @classmethod
    def choose(cls, thunk: Callable[[], Node]) -> Node: ...


Defaults().earlier(f64(1))  # error: arg-type
Defaults.earlier(node, 1)  # error: arg-type
wrong_default: i64 = Defaults.FIRST  # error: assignment
wrong_body_result: i64 = Defaults().earlier()  # error: assignment
Defaults.choose(i64)  # error: arg-type

run() * "three"  # error: operator
seq(run(), i64(1))  # error: arg-type
run(i64(1))  # error: arg-type
bad_report: i64 = EGraph().run(run())  # error: assignment
bad_updated: str = EGraph().run(run()).updated  # error: assignment
bad_can_stop: bytes = EGraph().run(run()).can_stop  # error: assignment
bad_format: int = EGraph().stats().format  # error: assignment
bad_content: str = EGraph().stats().content  # error: assignment


class ObjectHolder(Expr):
    def __init__(self, value: object = None) -> None: ...

    def opaque(self, value: object) -> object: ...


ObjectHolder(None, None)  # error: call-arg
wrong_object_result: i64 = ObjectHolder().opaque(None)  # error: assignment
