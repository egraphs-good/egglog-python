from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from . import bindings
from .declarations import *
from .registry import *

__all__ = ["EGraph"]

EXPR = TypeVar("EXPR", bound=BaseExpr)


@dataclass
class EGraph(Registry):
    _egraph: bindings.EGraph = field(default_factory=bindings.EGraph)

    def run(self, iterations: int) -> None:
        """
        Run the egraph for a given number of iterations.
        """
        self._egraph.run_rules(iterations)

    def check(self, fact: Fact) -> None:
        """
        Check if a fact is true in the egraph.
        """
        return self._egraph.check_fact(self._to_egg_fact(fact))

    def extract(self, expr: EXPR) -> EXPR:
        """
        Extract the lowest cost expression from the egraph.
        """
        egg_expr = self._to_egg_expr(expr)
        cost, new_egg_expr, variants = self._egraph.extract_expr(egg_expr)
        return self._from_egg_expr(new_egg_expr)

    def define(self, name: str, expr: EXPR) -> EXPR:
        """
        Define a new expression in the egraph.
        """
        self._egraph.define_expr(name, self._to_egg_expr(expr))

    def _on_register_function(self, decl: FunctionDecl) -> None:
        self._egraph.declare_function(decl)

    def _on_register_sort(self, ref: TypeRef) -> None:
        self._egraph.declare_sort(name, presort_and_args)

    def _on_register_rewrite(self, rewrite: RewriteDecl) -> None:
        self._egraph.add_rewrite(rewrite)

    def _on_register_rule(self, rule: RuleDecl) -> None:
        self._egraph.add_rule(rule)

    def _to_egg_expr(self, expr: EXPR) -> bindings.Expr:
        return bindings.Expr.from_python(expr)
