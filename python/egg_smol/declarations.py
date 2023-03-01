"""
Data only descriptions of the components of an egraph and the expressions.

Status: Done
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = [
    "Declarations",
    "TypeRef",
    "ClassTypeVar",
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
    "ClassDecl",
    "RewriteDecl",
    "EqDecl",
    "FactDecl",
    "RuleDecl",
    "LetDecl",
    "SetDecl",
    "DeleteDecl",
    "UnionDecl",
    "PanicDecl",
    "ActionDecl",
]


@dataclass
class Declarations:
    functions: dict[str, FunctionDecl] = field(default_factory=dict)
    classes: dict[str, ClassDecl] = field(default_factory=dict)

    # Bidirectional mapping between egg function names and python callable references.
    egg_fn_to_callable_ref: dict[str, CallableRef] = field(default_factory=dict)
    callable_ref_to_egg_fn: dict[CallableRef, str] = field(default_factory=dict)

    # Bidirectional mapping between egg sort names and python type references.
    egg_sort_to_type_ref: dict[str, TypeRef] = field(default_factory=dict)
    type_ref_to_egg_sort: dict[TypeRef, str] = field(default_factory=dict)

    rewrites: list[RewriteDecl] = field(default_factory=list)
    rules: list[RuleDecl] = field(default_factory=list)

    # includes: Optional[Declarations] = None

    def integrity_check(self) -> None:
        """
        Checks that:
            1. None of the functions and classes  have the same names
            2. Each mapping is bidrectional
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class TypeRef:
    name: str
    args: tuple[TypeRef, ...] = ()


@dataclass(frozen=True)
class ClassTypeVar:
    """
    A class type variable represents one of the types of the class, if it is a generic
    class.
    """

    index: int


TypeOrVarRef = Union[TypeRef, ClassTypeVar]


@dataclass(frozen=True)
class FunctionRef:
    name: str


@dataclass(frozen=True)
class MethodRef:
    class_name: str
    method_name: str


@dataclass(frozen=True)
class ClassMethodRef:
    class_name: str
    method_name: str


CallableRef = Union[FunctionRef, MethodRef, ClassMethodRef]


@dataclass(frozen=True)
class FunctionDecl:
    arg_types: tuple[TypeOrVarRef, ...]
    return_type: TypeOrVarRef
    cost: Optional[int] = None
    default: Optional[ExprDecl] = None
    merge: Optional[ExprDecl] = None


@dataclass(frozen=True)
class VarDecl:
    name: str


LitType = Union[int, str, None]


@dataclass(frozen=True)
class LitDecl:
    value: LitType


@dataclass(frozen=True)
class CallDecl:
    callable: CallableRef
    args: tuple[ExprDecl, ...]


ExprDecl = Union[VarDecl, LitDecl, CallDecl]


@dataclass
class ClassDecl:
    methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_methods: dict[str, FunctionDecl] = field(default_factory=dict)
    n_type_vars: int = 0


@dataclass(frozen=True)
class RewriteDecl:
    lhs: ExprDecl
    rhs: ExprDecl
    conditions: tuple[FactDecl, ...]


@dataclass(frozen=True)
class EqDecl:
    exprs: tuple[ExprDecl, ...]


FactDecl = Union[ExprDecl, EqDecl]


@dataclass(frozen=True)
class RuleDecl:
    head: tuple[ActionDecl, ...]
    body: tuple[FactDecl, ...]


@dataclass(frozen=True)
class LetDecl:
    name: str
    value: ExprDecl


@dataclass(frozen=True)
class SetDecl:
    call: CallDecl
    rhs: ExprDecl


@dataclass(frozen=True)
class DeleteDecl:
    call: CallDecl


@dataclass(frozen=True)
class UnionDecl:
    lhs: ExprDecl
    rhs: ExprDecl


@dataclass(frozen=True)
class PanicDecl:
    message: str


ActionDecl = Union[LetDecl, SetDecl, DeleteDecl, UnionDecl, PanicDecl, ExprDecl]
