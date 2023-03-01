from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, cast

from . import bindings
from .declarations import *
from .registry import *
from .runtime import *

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
        return self._from_egg_expr(new_egg_expr, expr)

    def define(self, name: str, expr: EXPR) -> EXPR:
        """
        Define a new expression in the egraph and return a reference to it.
        """
        self._egraph.define(name, self._to_egg_expr(expr))
        return self._create_var(name, expr)

    def _on_register_function(self, ref: CallableRef, decl: FunctionDecl) -> None:
        self._egraph.declare_function(decl)

    def _on_register_sort(self, name: str) -> None:
        self._egraph.declare_sort(name, None)

    def _on_register_rewrite(self, rewrite: RewriteDecl) -> None:
        self._egraph.add_rewrite(rewrite)

    def _on_register_rule(self, rule: RuleDecl) -> None:
        self._egraph.add_rule(self._rule_decl_to_egg(rule))

    def _rule_decl_to_egg(self, rule: RuleDecl) -> bindings.Rule:
        return bindings.Rule(
            list(map(self._action_decl_to_egg, rule.head)),
            list(map(self._fact_decl_to_egg, rule.body)),
        )

    def _action_decl_to_egg(self, action: ActionDecl) -> bindings.Action:
        if isinstance(action, LetDecl):
            return bindings.Let(
                action.name, self._expr_decl_to_egg_expr(action.value)
            )
        elif isinstance(action, SetDecl):
            args = list(map(self._expr_decl_to_egg_expr, action.call.args))
            rhs = self._expr_decl_to_egg_expr(action.rhs)
            lhs = self._ca action.call.callable
            return bindings.Set(
                action.name, self._expr_decl_to_egg_expr(action.value)
            )


    def _to_egg_function_decl(
        self, ref: CallableRef, decl: FunctionDecl
    ) -> bindings.FunctionDecl:
        egg_name = self._declarations.callable_ref_to_egg_fn[ref]
        return_tp = decl.return_type
        if isinstance(return_tp, ClassTypeVar):
            raise ValueError(
                "Type variables are not supported currently for function types."
            )

        return_sort = self._get_egg_sort(return_tp)
        arg_sorts = [self._get_egg_sort(a) for a in decl.arg_types]
        schema = bindings.Schema(arg_sorts, return_sort)
        default = self._expr_decl_to_egg_expr(decl.default) if decl.default else None
        merge = self._expr_decl_to_egg_expr(decl.merge) if decl.merge else None
        return bindings.FunctionDecl(
            egg_name, schema=schema, default=default, cost=decl.cost, merge=merge
        )

    def _get_egg_sort(self, tp: TypeOrVarRef) -> str:
        if isinstance(tp, ClassTypeVar):
            raise ValueError("egg-smol does not support class type variables yet")
        ref_to_sort = self._declarations.type_ref_to_egg_sort
        if tp.args and tp not in ref_to_sort:
            # Register a new sort if this type has type arguments and is not already registered
            new_sort_name = self._declarations.register_new_sort(tp)
            presort = tp.name
            arg_sorts: list[bindings._Expr] = [
                bindings.Var(self._get_egg_sort(a)) for a in tp.args
            ]
            self._egraph.declare_sort(new_sort_name, (presort, arg_sorts))
            return new_sort_name
        return ref_to_sort[tp]

    def _to_egg_fact(self, fact: Fact) -> bindings._Fact:
        if isinstance(fact, Eq):
            return bindings.Eq(list(map(self._to_egg_expr, fact.exprs)))
        elif isinstance(fact, RuntimeExpr):
            return bindings.Fact(self._runtime_expr_to_egg_expr(fact))
        raise NotImplementedError(f"Unknown fact type: {fact}")

    def _create_var(self, name: str, expr: EXPR) -> EXPR:
        assert isinstance(expr, RuntimeExpr)
        return RuntimeExpr(expr.decls, expr.tp, VarDecl(name))

    def _to_egg_expr(self, expr: BaseExpr) -> bindings._Expr:
        assert isinstance(expr, RuntimeExpr)
        return self._runtime_expr_to_egg_expr(expr)

    def _from_egg_expr(self, expr: bindings._Expr, original_expr: EXPR) -> EXPR:
        assert isinstance(original_expr, RuntimeExpr)
        return cast(EXPR, self._egg_expr_to_runtime_expr(expr, original_expr.tp))

    def _runtime_expr_to_egg_expr(self, expr: RuntimeExpr) -> bindings._Expr:
        assert expr.decls == self._declarations
        return self._expr_decl_to_egg_expr(expr.expr)

    def _egg_expr_to_runtime_expr(
        self, expr: bindings._Expr, tp: TypeRef
    ) -> RuntimeExpr:
        return RuntimeExpr(self._declarations, tp, self._egg_expr_to_expr_decl(expr))

    def _expr_decl_to_egg_expr(self, expr_decl: ExprDecl) -> bindings._Expr:
        if isinstance(expr_decl, VarDecl):
            return bindings.Var(expr_decl.name)
        elif isinstance(expr_decl, LitDecl):
            if isinstance(expr_decl.value, int):
                return bindings.Lit(bindings.Int(expr_decl.value))
            elif isinstance(expr_decl.value, str):
                return bindings.Lit(bindings.String(expr_decl.value))
            elif expr_decl.value is None:
                return bindings.Lit(bindings.Unit())
            raise NotImplementedError(
                f"Unsupported literal type: {type(expr_decl.value)}"
            )
        elif isinstance(expr_decl, CallDecl):
            self._call_decl_to_egg_expr(expr_decl)
        raise NotImplementedError(f"Unsupported expression type: {type(expr_decl)}")

    def _call_decl_to_egg_expr(self, expr_decl: CallDecl) -> bindings._Expr:
        egg_fn = self._declarations.callable_ref_to_egg_fn[expr_decl.callable]
        return bindings.Call(
            egg_fn,
            [self._expr_decl_to_egg_expr(arg) for arg in expr_decl.args],
        )
    def _egg_expr_to_expr_decl(self, expr: bindings._Expr) -> ExprDecl:
        if isinstance(expr, bindings.Var):
            return VarDecl(expr.name)
        elif isinstance(expr, bindings.Lit):
            lit: LitType
            if isinstance(expr.value, (bindings.Int, bindings.String)):
                lit = expr.value.value
            elif isinstance(expr.value, bindings.Unit):
                lit = None
            else:
                raise NotImplementedError(
                    f"Unsupported literal type: {type(expr.value)}"
                )
            return LitDecl(lit)
        elif isinstance(expr, bindings.Call):
            callable_ref = self._declarations.egg_fn_to_callable_ref[expr.name]
            return CallDecl(
                callable_ref,
                tuple(map(self._egg_expr_to_expr_decl, expr.args)),
            )
        raise NotImplementedError(f"Unsupported expression type: {type(expr)}")
