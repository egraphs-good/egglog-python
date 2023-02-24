from __future__ import annotations
from typing import Union
from dataclasses import dataclass, field

from . import high_level, bindings


@dataclass(frozen=True)
class FnRef:
    name: str


@dataclass(frozen=True)
class MethodRef:
    kind_name: str
    name: str


@dataclass(frozen=True)
class ClassMethodRef:
    kind_name: str
    name: str


CallableRef = Union[FnRef, MethodRef, ClassMethodRef]


@dataclass
class EggMapper:
    """
    Maps from Python expression to egg expresions and vice versa.
    """

    namespace: high_level.Namespace = field(default_factory=high_level.Namespace)
    # Bidirectional mapping from egg function names to python callable references.
    fn_name_to_ref: dict[str, CallableRef] = field(default_factory=dict)
    ref_to_fn_name: dict[CallableRef, str] = field(default_factory=dict)

    def to_egg(self, expr: high_level.Expr_) -> bindings._Expr:
        """
        Convert a python expression to an egg expression.
        """
        if isinstance(expr, high_level.Lit):
            value = expr.value
            l: bindings._Literal
            if value is None:
                l = bindings.Unit()
            elif isinstance(value, int):
                l = bindings.Int(value)
            elif isinstance(value, str):
                l = bindings.String(value)
            else:
                raise ValueError("Unknown literal", value)
            return bindings.Lit(l)
        if isinstance(expr, high_level.Var):
            return bindings.Var(expr.name)
        if isinstance(expr, high_level.Call):
            ref, additional_args = self.callable_to_ref_and_additonal_args(expr.fn)
            name = self.ref_to_fn_name[ref]
            args = [self.to_egg(arg) for arg in additional_args + expr.args]
            return bindings.Call(name, args)

    def from_egg(self, expr: bindings._Expr) -> high_level.Expr_:
        """
        Convert an egg expression to an untyped python expression.
        """
        if isinstance(expr, bindings.Lit):
            val = expr.value
            return high_level.Lit(None if isinstance(val, bindings.Unit) else val.value)
        if isinstance(expr, bindings.Var):
            return high_level.Var(expr.name)
        if isinstance(expr, bindings.Call):
            ref = self.fn_name_to_ref[expr.name]
            callable = self.ref_to_callable(ref)
            args = tuple(self.from_egg(arg) for arg in expr.args)
            return high_level.Call(callable, args)
        raise ValueError("Unknown expr", expr)

    def ref_to_callable(self, ref: CallableRef) -> high_level.Callable_:
        ns = self.namespace
        if isinstance(ref, FnRef):
            return ns.get_function(ref.name)
        if isinstance(ref, MethodRef):
            res = getattr(ns.get_kind(ref.kind_name), ref.name)
            if not isinstance(res, high_level.BoundMethod):
                raise ValueError("Not a method", ref)
        if isinstance(ref, ClassMethodRef):
            res = getattr(ns.get_kind(ref.kind_name), ref.name)
            if not isinstance(res, high_level.BoundClassMethod):
                raise ValueError("Not a class method", ref)
            return res
        raise ValueError("Unknown ref", ref)

    def callable_to_ref_and_additonal_args(
        self, callable: high_level.Callable_
    ) -> tuple[CallableRef, tuple[high_level.Expr_, ...]]:
        if isinstance(callable, high_level.Function):
            return FnRef(callable.name), ()
        if isinstance(callable, high_level.BoundMethod):
            return MethodRef(callable.self.type.kind.name, callable.name), (
                callable.self.value,
            )
        if isinstance(callable, high_level.BoundClassMethod):
            return ClassMethodRef(callable.kind.name, callable.name), ()
        raise ValueError("Unknown callable", callable)
