from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, TypeVar, cast

from . import bindings
from .builtins import BUILTINS, BaseExpr
from .declarations import *
from .registry import *
from .registry import _expr_to_decl, _expr_to_type, _fact_to_decl, decl_to_expr
from .runtime import *

__all__ = ["EGraph"]

EXPR = TypeVar("EXPR", bound=BaseExpr)


@dataclass
class EGraph(Registry):
    _egraph: bindings.EGraph = field(default_factory=bindings.EGraph)

    def __post_init__(self) -> None:
        # Copy the builtin declarations
        self._decls = deepcopy(BUILTINS._decls)

    def run(self, iterations: int) -> None:
        """
        Run the egraph for a given number of iterations.
        """
        self._egraph.run_rules(iterations)

    def check(self, fact: Fact) -> None:
        """
        Check if a fact is true in the egraph.
        """
        fact_decl = _fact_to_decl(fact)
        fact_egg = fact_decl_to_egg(self._decls, fact_decl)
        return self._egraph.check_fact(fact_egg)

    def extract(self, expr: EXPR) -> EXPR:
        """
        Extract the lowest cost expression from the egraph.
        """
        egg_expr = _expr_to_decl(expr).to_egg(self._decls)
        _cost, new_egg_expr, _variants = self._egraph.extract_expr(egg_expr)
        tp, new_expr_decl = tp_and_expr_decl_from_egg(self._decls, new_egg_expr)
        if tp != _expr_to_type(expr):
            raise RuntimeError(f"Type mismatch: {tp} != {_expr_to_type(expr)}")
        return decl_to_expr(new_expr_decl, expr)

    def define(self, name: str, expr: EXPR, cost: Optional[int] = None) -> EXPR:
        """
        Define a new expression in the egraph and return a reference to it.
        """
        expr_decl = _expr_to_decl(expr)
        tp = _expr_to_type(expr)
        self._egraph.define(name, expr_decl.to_egg(self._decls), cost)
        self._decls.constants[name] = ConstantDecl(tp, expr_decl, cost)
        ref = ConstantRef(name)
        self._register_callable_ref(name, ref)
        return cast(EXPR, RuntimeExpr(self._decls, tp, CallDecl(ref)))

    def _on_register_function(self, ref: CallableRef, decl: FunctionDecl) -> None:
        # Don't need to registry constants, since they are already registered
        if isinstance(ref, ConstantRef):
            raise RuntimeError("Constants should not be registered as functions")
        egg_decl = decl.to_egg(self._decls, self._egraph, ref)
        self._egraph.declare_function(egg_decl)

    def _on_register_sort(self, name: str) -> None:
        self._egraph.declare_sort(name, None)

    def _on_register_rewrite(self, rewrite: RewriteDecl) -> None:
        self._egraph.add_rewrite(rewrite.to_egg(self._decls))

    def _on_register_rule(self, rule: RuleDecl) -> None:
        self._egraph.add_rule(rule.to_egg(self._decls))

    def _on_register_action(self, decl: ActionDecl) -> None:
        self._egraph.eval_actions(action_decl_to_egg(self._decls, decl))
