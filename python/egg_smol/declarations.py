"""
Data only descriptions of the components of an egraph and the expressions.

Status: Done
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union, cast

from . import bindings

__all__ = [
    "Declarations",
    "JustTypeRef",
    "ClassTypeVarRef",
    "TypeRefWithVars",
    "TypeOrVarRef",
    "FunctionRef",
    "MethodRef",
    "ClassMethodRef",
    "CallableRef",
    "FunctionDecl",
    "VarDecl",
    "LitType",
    "LitDecl",
    "CallDecl",
    "ExprDecl",
    "expr_decl_from_egg",
    "ClassDecl",
    "RewriteDecl",
    "EqDecl",
    "FactDecl",
    "fact_decl_to_egg",
    "RuleDecl",
    "LetDecl",
    "SetDecl",
    "DeleteDecl",
    "UnionDecl",
    "PanicDecl",
    "ActionDecl",
    "action_decl_to_egg",
]


@dataclass
class Declarations:
    functions: dict[str, FunctionDecl] = field(default_factory=dict)
    classes: dict[str, ClassDecl] = field(default_factory=dict)

    # Bidirectional mapping between egg function names and python callable references.
    egg_fn_to_callable_ref: dict[str, CallableRef] = field(default_factory=dict)
    callable_ref_to_egg_fn: dict[CallableRef, str] = field(default_factory=dict)

    # Bidirectional mapping between egg sort names and python type references.
    egg_sort_to_type_ref: dict[str, JustTypeRef] = field(default_factory=dict)
    type_ref_to_egg_sort: dict[JustTypeRef, str] = field(default_factory=dict)

    rewrites: list[RewriteDecl] = field(default_factory=list)
    rules: list[RuleDecl] = field(default_factory=list)
    actions: list[ActionDecl] = field(default_factory=list)

    # includes: Optional[Declarations] = None

    def integrity_check(self) -> None:
        """
        Checks that:
            1. None of the functions and classes  have the same names
            2. Each mapping is bidrectional
        """
        raise NotImplementedError()


# Have two different types of type refs, one that can include vars recursively and one that cannot.
# We only use the one with vars for classmethods and methods, and the other one for egg references as
# well as runtime values.
@dataclass(frozen=True)
class JustTypeRef:
    name: str
    args: tuple[JustTypeRef, ...] = ()

    def generate_egg_name(self) -> str:
        """
        Generates an egg sort name for this type reference by linearizing the type.
        """
        if not self.args:
            return self.name
        args = "_".join(a.generate_egg_name() for a in self.args)
        return f"{self.name}__{args}"

    def to_egg(self, decls: Declarations, egraph: bindings.EGraph) -> str:
        if self in decls.type_ref_to_egg_sort:
            return decls.type_ref_to_egg_sort[self]
        elif not self.args:
            raise ValueError(f"Type {self.name} is not registered.")
        # If this is a type with arguments and it is not registered, then we need to register i
        new_name = self.generate_egg_name()
        assert new_name not in decls.egg_sort_to_type_ref
        decls.egg_sort_to_type_ref[new_name] = self
        decls.type_ref_to_egg_sort[self] = new_name
        arg_sorts = [
            cast(bindings._Expr, bindings.Var(a.to_egg(decls, egraph)))
            for a in self.args
        ]
        egraph.declare_sort(new_name, (self.name, arg_sorts))
        return new_name

    def to_var(self) -> TypeRefWithVars:
        return TypeRefWithVars(self.name, tuple(a.to_var() for a in self.args))

@dataclass(frozen=True)
class ClassTypeVarRef:
    """
    A class type variable represents one of the types of the class, if it is a generic
    class.
    """

    index: int

    def to_just(self) -> JustTypeRef:
        raise NotImplementedError("egg-smol does not support generic classes yet.")


@dataclass(frozen=True)
class TypeRefWithVars:
    name: str
    args: tuple[TypeOrVarRef, ...] = ()

    def to_just(self) -> JustTypeRef:
        return JustTypeRef(self.name, tuple(a.to_just() for a in self.args))


TypeOrVarRef = Union[ClassTypeVarRef, TypeRefWithVars]


@dataclass(frozen=True)
class FunctionRef:
    name: str

    def to_egg(self, decls: Declarations) -> str:
        return decls.callable_ref_to_egg_fn[self]

    def generate_egg_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class MethodRef:
    class_name: str
    method_name: str

    def to_egg(self, decls: Declarations) -> str:
        return decls.callable_ref_to_egg_fn[self]

    def generate_egg_name(self) -> str:
        return f"{self.class_name}__{self.method_name}"


@dataclass(frozen=True)
class ClassMethodRef:
    class_name: str
    method_name: str

    def to_egg(self, decls: Declarations) -> str:
        return decls.callable_ref_to_egg_fn[self]

    def generate_egg_name(self) -> str:
        return f"{self.class_name}__{self.method_name}"


CallableRef = Union[FunctionRef, MethodRef, ClassMethodRef]


@dataclass(frozen=True)
class FunctionDecl:
    # TODO: Add arg name to arg so can call with keyword arg
    arg_types: tuple[TypeOrVarRef, ...]
    return_type: TypeOrVarRef
    cost: Optional[int] = None
    default: Optional[ExprDecl] = None
    merge: Optional[ExprDecl] = None

    def to_egg(
        self, decls: Declarations, egraph: bindings.EGraph, ref: CallableRef
    ) -> bindings.FunctionDecl:
        return bindings.FunctionDecl(
            decls.callable_ref_to_egg_fn[ref],
            # Remove all vars from the type refs, raising an errory if we find one,
            # since we cannot create egg functions with vars
            bindings.Schema(
                [a.to_just().to_egg(decls, egraph) for a in self.arg_types],
                self.return_type.to_just().to_egg(decls, egraph),
            ),
            self.default.to_egg(decls) if self.default else None,
            self.merge.to_egg(decls) if self.merge else None,
            self.cost,
        )


@dataclass(frozen=True)
class VarDecl:
    name: str

    @classmethod
    def from_egg(cls, var: bindings.Var) -> VarDecl:
        return cls(var.name)

    def to_egg(self, _decls: Declarations) -> bindings.Var:
        return bindings.Var(self.name)


LitType = Union[int, str, None]


@dataclass(frozen=True)
class LitDecl:
    value: LitType

    @classmethod
    def from_egg(cls, lit: bindings.Lit) -> LitDecl:
        if isinstance(lit.value, (bindings.Int, bindings.String)):
            return cls(lit.value.value)
        elif isinstance(lit.value, bindings.Unit):
            return cls(None)
        raise NotImplementedError(f"Unsupported literal type: {type(lit.value)}")

    def to_egg(self, _decls: Declarations) -> bindings.Lit:
        if self.value is None:
            return bindings.Lit(bindings.Unit())
        if isinstance(self.value, int):
            return bindings.Lit(bindings.Int(self.value))
        if isinstance(self.value, str):
            return bindings.Lit(bindings.String(self.value))
        raise NotImplementedError(f"Unsupported literal type: {type(self.value)}")


@dataclass(frozen=True)
class CallDecl:
    callable: CallableRef
    args: tuple[ExprDecl, ...]

    @classmethod
    def from_egg(cls, decls: Declarations, call: bindings.Call) -> CallDecl:
        callable_ref = decls.egg_fn_to_callable_ref[call.name]
        return cls(callable_ref, tuple(expr_decl_from_egg(decls, a) for a in call.args))

    def to_egg(self, decls: Declarations) -> bindings.Call:
        egg_fn = decls.callable_ref_to_egg_fn[self.callable]
        return bindings.Call(egg_fn, tuple(a.to_egg(decls) for a in self.args))


ExprDecl = Union[VarDecl, LitDecl, CallDecl]


def expr_decl_from_egg(decls: Declarations, expr: bindings._Expr) -> ExprDecl:
    if isinstance(expr, bindings.Var):
        return VarDecl.from_egg(expr)
    if isinstance(expr, bindings.Lit):
        return LitDecl.from_egg(expr)
    if isinstance(expr, bindings.Call):
        return CallDecl.from_egg(decls, expr)
    raise NotImplementedError(f"Unsupported expression type: {type(expr)}")


@dataclass
class ClassDecl:
    methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_methods: dict[str, FunctionDecl] = field(default_factory=dict)
    n_type_vars: int = 0

    def all_refs(self, name: str) -> Iterable[MethodRef | ClassMethodRef]:
        for method_name in self.methods:
            yield MethodRef(name, method_name)
        for method_name in self.class_methods:
            yield ClassMethodRef(name, method_name)


@dataclass(frozen=True)
class RewriteDecl:
    lhs: ExprDecl
    rhs: ExprDecl
    conditions: tuple[FactDecl, ...]

    def to_egg(self, decls: Declarations) -> bindings.Rewrite:
        return bindings.Rewrite(
            self.lhs.to_egg(decls),
            self.rhs.to_egg(decls),
            [fact_decl_to_egg(decls, c) for c in self.conditions],
        )


@dataclass(frozen=True)
class EqDecl:
    exprs: tuple[ExprDecl, ...]

    def to_egg(self, decls: Declarations) -> bindings.Eq:
        return bindings.Eq([e.to_egg(decls) for e in self.exprs])


FactDecl = Union[ExprDecl, EqDecl]


def fact_decl_to_egg(decls: Declarations, fact: FactDecl) -> bindings._Fact:
    if isinstance(fact, ExprDecl):
        return bindings.Fact(fact.to_egg(decls))
    if isinstance(fact, EqDecl):
        return fact.to_egg(decls)
    raise NotImplementedError(f"Unsupported fact type: {type(fact)}")


@dataclass(frozen=True)
class RuleDecl:
    head: tuple[ActionDecl, ...]
    body: tuple[FactDecl, ...]

    def to_egg(self, decls: Declarations) -> bindings.Rule:
        return bindings.Rule(
            [action_decl_to_egg(decls, a) for a in self.head],
            [fact_decl_to_egg(decls, f) for f in self.body],
        )


@dataclass(frozen=True)
class LetDecl:
    name: str
    value: ExprDecl

    def to_egg(self, decls: Declarations) -> bindings.Let:
        return bindings.Let(self.name, self.value.to_egg(decls))


@dataclass(frozen=True)
class SetDecl:
    call: CallDecl
    rhs: ExprDecl

    def to_egg(self, decls: Declarations) -> bindings.Set:
        return bindings.Set(
            self.call.callable.to_egg(decls),
            [a.to_egg(decls) for a in self.call.args],
            self.rhs.to_egg(decls),
        )


@dataclass(frozen=True)
class DeleteDecl:
    call: CallDecl

    def to_egg(self, decls: Declarations) -> bindings.Delete:
        return bindings.Delete(
            self.call.callable.to_egg(decls), [a.to_egg(decls) for a in self.call.args]
        )


@dataclass(frozen=True)
class UnionDecl:
    lhs: ExprDecl
    rhs: ExprDecl

    def to_egg(self, decls: Declarations) -> bindings.Union:
        return bindings.Union(self.lhs.to_egg(decls), self.rhs.to_egg(decls))


@dataclass(frozen=True)
class PanicDecl:
    message: str

    def to_egg(self, _decls: Declarations) -> bindings.Panic:
        return bindings.Panic(self.message)


ActionDecl = Union[LetDecl, SetDecl, DeleteDecl, UnionDecl, PanicDecl, ExprDecl]


def action_decl_to_egg(decls: Declarations, action: ActionDecl) -> bindings._Action:
    if isinstance(action, ExprDecl):
        return bindings.Expr_(action.to_egg(decls))
    return action.to_egg(decls)
