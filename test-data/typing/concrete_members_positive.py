"""Ordinary user class annotations retain exact public member types."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import ClassVar, assert_type

from egglog import (
    Action,
    EGraph,
    Expr,
    Pair,
    RewriteOrRule,
    Ruleset,
    RunReport,
    Schedule,
    UnstableFn,
    Vec,
    eq,
    f64,
    function,
    i64,
    i64Like,
    method,
    rewrite,
    rule,
    ruleset,
    run,
    seq,
    union,
    var,
    vars_,
)
from egglog.run_report import RuleKey


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

    def strict(self, value: i64) -> i64: ...

    def combine(self, left: i64, right: f64) -> i64: ...


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


@function
def exact(value: i64) -> i64: ...


named = UnstableFn(exact)
assert_type(named, UnstableFn[i64, i64])
assert_type(UnstableFn[i64, i64](exact), UnstableFn[i64, i64])
assert_type(named(i64(1)), i64)
assert_type(named.value, Callable[[i64], i64])
assert_type(named.eval(), Callable[[i64], i64])
assert_type(UnstableFn(node.strict), UnstableFn[i64, i64])
assert_type(UnstableFn(Node.strict, node), UnstableFn[i64, i64])
assert_type(UnstableFn(Node.combine, node, i64(1)), UnstableFn[i64, f64])
nullary = UnstableFn(exact, i64(1))
assert_type(nullary, UnstableFn[i64])
assert_type(nullary(), i64)
assert_type(nullary.value, Callable[[], i64])
callback: Callable[[i64], i64] = named
assert_type(callback(i64(1)), i64)

variable = var("node", Node)
assert_type(variable, Node)
assert_type(vars_("left right", Node), Iterable[Node])
assert_type(var("number", i64), i64)
assert_type(rewrite(variable).to(node), RewriteOrRule)
assert_type(rule(eq(variable).to(node)).then(node), RewriteOrRule)
assert_type(union(variable).with_(node), Action)
assert_type(rule(variable).then(union(variable).with_(node)), RewriteOrRule)
destination = ruleset(name="typed_rules")
assert_type(destination, Ruleset)
assert_type(rewrite(variable, ruleset=destination).to(node), RewriteOrRule)
assert_type(rule(variable, ruleset=destination).then(node), RewriteOrRule)
destination.append(rewrite(variable).to(node))
assert_type(rewrite(variable).to(node) == rewrite(variable).to(node), bool)


class Defaults(Expr, ruleset=destination):
    FIRST: ClassVar[Node] = Node()

    def __init__(self) -> None: ...

    def later(self, value: i64Like) -> Node: ...

    def earlier(self, value: i64Like = 1) -> Node:
        return self.later(value)

    @property
    def own(self) -> Defaults:
        return self

    @classmethod
    def named(cls) -> Defaults:
        return cls()

    @classmethod
    def choose(cls, thunk: Callable[[], Node]) -> Node: ...


defaults = Defaults()
assert_type(Defaults.FIRST, Node)
assert_type(defaults.earlier(), Node)
assert_type(Defaults.earlier(defaults, 1), Node)
assert_type(defaults.own, Defaults)
assert_type(Defaults.named(), Defaults)
assert_type(Defaults.choose(Node), Node)

schedule = (destination + run()).saturate() * 3
assert_type(schedule, Schedule)
assert_type(seq(destination, run()), Schedule)
graph = EGraph()
report = graph.run(schedule)
assert_type(report, RunReport)
assert_type(graph.stats(), RunReport)
assert_type(next(iter(report.num_matches_per_rule)), RuleKey)
assert_type(next(iter(report.num_matches_per_rule.values())), int)
