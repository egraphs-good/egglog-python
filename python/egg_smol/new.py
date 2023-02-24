from __future__ import annotations
from typing import (
    Optional,
    overload,
    TypeVar,
    Callable,
    ParamSpec,
    Any,
    NoReturn,
    Union,
    Generic,
)
from dataclasses import dataclass, field

from . import bindings as egg

__all__ = [
    "Registry",
    "Rewrite",
    "Rule",
    "Expr",
    "Unit",
    "i64",
    "i64Like",
    "BUILTINS",
    "rewrite",
    "eq",
    "let",
    "set",
    "delete",
    "union",
    "panic",
    "var",
]

T = TypeVar("T")
P = ParamSpec("P")
TYPE = TypeVar("TYPE", bound=type)
CALLABLE = TypeVar("CALLABLE", bound=Callable)
EXPR = TypeVar("EXPR", bound="Expr")


@dataclass
class Registry:
    """
    A registry holds all the declerations of classes and functions as well as the mapping
    to and from egg names to python names.
    """

    _functions: dict[str, FunctionDecl] = field(default_factory=dict)
    _classes: dict[str, ClassDecl] = field(default_factory=dict)

    # Bidirectional mapping between egg function names and python callable references.
    _egg_fn_to_callable_ref: dict[str, CallableRef] = field(default_factory=dict)
    _callable_ref_to_egg_fn: dict[CallableRef, str] = field(default_factory=dict)

    # Bidirectional mapping between egg sort names and python type references.
    _egg_sort_to_type_ref: dict[str, TypeRef] = field(default_factory=dict)
    _type_ref_to_egg_sort: dict[TypeRef, str] = field(default_factory=dict)

    _rewrites: list[RewriteDecl] = field(default_factory=list)
    _rules: list[RuleDecl] = field(default_factory=list)

    # Called whenever a sort is declared.
    # Callback should take the sort name and an optional presort and presort args.
    on_declare_sort: Optional[
        Callable[[str, Optional[tuple[str, list[egg._Expr]]]], None]
    ] = None

    # Called whenever a function is declared.
    on_declare_function: Optional[Callable[[egg.FunctionDecl], None]] = None

    # Called whenever a rewrite is declared.
    on_declare_rewrite: Optional[Callable[[egg.Rewrite], None]] = None

    # Called whenever a rule is declared.
    on_declare_rule: Optional[Callable[[egg.Rule], None]] = None

    def integrity_check(self) -> None:
        """
        Checks that:
            1. None of the functions and classes  have the same names
            2. Each mapping is bidrectional
        """
        ...

    @overload
    def __call__(self, *, egg_sort: str) -> Callable[[TYPE], TYPE]:
        """
        Registers a class with some paramaters.
        """
        ...

    @overload
    def __call__(self, cls: TYPE, /) -> TYPE:
        """
        Registers a class.
        """
        ...

    @overload
    def __call__(  # type: ignore # Ignore error about incompatible overloads, with below
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: int = 0,
    ) -> Callable[[CALLABLE], CALLABLE]:
        """
        Registers a function with some paramaters.
        """
        ...

    @overload
    def __call__(
        self,
        *,
        egg_fn: Optional[str] = None,
        default: Optional[T] = None,
        cost: int = 0,
        merge: Optional[Callable[[T, T], T]] = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Registers a function with some paramaters.

        Includes the params that specify the return type in a seperate overload,
        so if you don't use those the first one is matched. Otherwise, it would match on this one,
        and think that "T == Nothing" and then you wouldn't be able to use it.
        """
        ...

    @overload
    def __call__(self, fn: CALLABLE, /) -> CALLABLE:
        """
        Registers a function.
        """
        ...

    @overload
    def __call__(
        self, rewrite_or_rule: Rewrite | Rule, /, *rewrite_or_rules: Rewrite | Rule
    ) -> None:
        """
        Registers any number of rewrites or a rules.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """
        Register a class, type, rewrite or a rule.
        """
        ...

    def to_egg_expr(self, expr: Expr) -> egg._Expr:
        """
        Convert a python expression to an egg expression.
        """
        ...

    def from_egg_expr(self, expr: egg._Expr) -> Expr:
        """
        Convert an egg expression to a python expression.
        """
        ...


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


def rewrite(lhs: EXPR) -> RewriteBuilder[EXPR]:
    return RewriteBuilder(lhs=lhs)


@dataclass
class RewriteBuilder(Generic[EXPR]):
    lhs: EXPR

    def to(self, rhs: EXPR, *conditions: Fact) -> Rewrite:
        return Rewrite(lhs=self.lhs, rhs=rhs, conditions=list(conditions))


@dataclass
class Rewrite:
    lhs: Expr
    rhs: Expr
    conditions: list[Fact]


def eq(expr: EXPR) -> EqBuilder[EXPR]:
    return EqBuilder(expr)


@dataclass
class EqBuilder(Generic[EXPR]):
    expr: Expr

    def to(self, *exprs: EXPR) -> Eq:
        return Eq([self.expr, *exprs])


@dataclass
class Eq:
    exprs: list[Expr]


Fact = Union["Unit", Eq]


def panic(message: str) -> Panic:
    return Panic(message)


def let(name: str, expr: Expr) -> Let:
    return Let(name, expr)


def delete(expr: Expr) -> Delete:
    return Delete(expr)


def union(lhs: EXPR) -> UnionBuilder[EXPR]:
    return UnionBuilder(lhs=lhs)


def set(lhs: EXPR) -> SetBuilder[EXPR]:
    return SetBuilder(lhs=lhs)


@dataclass
class SetBuilder(Generic[EXPR]):
    lhs: Expr

    def to(self, rhs: EXPR) -> Set:
        return Set(lhs=self.lhs, rhs=rhs)


@dataclass
class UnionBuilder(Generic[EXPR]):
    lhs: Expr

    def with_(self, rhs: EXPR) -> Union_:
        return Union_(lhs=self.lhs, rhs=rhs)


class VarBuilder:
    def __getitem__(self, tp: type[EXPR]) -> TypedVarBuilder[EXPR]:
        return TypedVarBuilder(tp)


@dataclass
class TypedVarBuilder(Generic[EXPR]):
    tp: type[EXPR]

    def __getattr__(self, name: str) -> EXPR:
        ...


var = VarBuilder()


@dataclass
class Delete:
    expr: Expr


@dataclass
class Panic:
    message: str


@dataclass
class Union_(Generic[EXPR]):
    lhs: Expr
    rhs: Expr


@dataclass
class Set:
    lhs: Expr
    rhs: Expr


@dataclass
class Let:
    name: str
    value: Expr


Action = Union[Let, Set, Delete, Union_, Panic, "Expr"]


@dataclass
class Rule:
    header: list[Action]
    body: list[Fact]


class Expr:
    """
    Expression base class, which adds suport for != to all expression types.
    """

    def __ne__(self: EXPR, __o: EXPR) -> Unit:  # type: ignore
        ...

    def __eq__(self, other: NoReturn) -> NoReturn:  # type: ignore
        """
        Equality is currently not supported
        """
        raise NotImplementedError()


BUILTINS = Registry()


@BUILTINS(egg_sort="unit")
class Unit(Expr):
    def __init__(self) -> None:
        ...


@BUILTINS(egg_sort="i64")
class i64(Expr):
    def __init__(self, value: int):
        ...

    @BUILTINS(egg_fn="+")
    def __add__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="-")
    def __sub__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="*")
    def __mul__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="/")
    def __truediv__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="%")
    def __mod__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="&")
    def __and__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="|")
    def __or__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="^")
    def __xor__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="<<")
    def __lshift__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn=">>")
    def __rshift__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="not-64")
    def __invert__(self) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="<")
    def __lt__(self, other: i64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @BUILTINS(egg_fn=">")
    def __gt__(self, other: i64Like) -> Unit:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="min")
    def min(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    @BUILTINS(egg_fn="max")
    def max(self, other: i64Like) -> i64:  # type: ignore[empty-body]
        ...


# The types which can be converted into an i64
i64Like = Union[int, i64]
