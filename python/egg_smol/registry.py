from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    overload,
)

from .declarations import *

if TYPE_CHECKING:
    from .builtins import BaseExpr, Unit

__all__ = [
    "Registry",
    "rewrite",
    "eq",
    "panic",
    "let",
    "delete",
    "union",
    "set_",
    "if_",
    "var",
    "vars",
    "Rewrite",
    "Eq",
    "Fact",
    "Delete",
    "Panic",
    "Union_",
    "Set",
    "Let",
    "Action",
    "Rule",
    "BaseExpr",
]

T = TypeVar("T")
P = ParamSpec("P")
TYPE = TypeVar("TYPE", bound=type)
CALLABLE = TypeVar("CALLABLE", bound=Callable)
EXPR = TypeVar("EXPR", bound="BaseExpr")


@dataclass
class Registry:
    """
    A registry holds all the declerations of classes and functions as well as the mapping
    to and from egg names to python names.
    """

    _declarations: Declarations = field(default_factory=Declarations)

    def _on_register_sort(self, name: str) -> None:
        """
        Called whenever a sort is registered.
        """
        pass

    def _on_register_function(self, ref: CallableRef, decl: FunctionDecl) -> None:
        """
        Called whenever a function is registered.
        """
        pass

    def _on_register_rewrite(self, decl: RewriteDecl) -> None:
        """
        Called whenever a rewrite is registered.
        """
        pass

    def _on_register_rule(self, decl: RuleDecl) -> None:
        """
        Called whenever a rule is registered.
        """
        pass

    @overload
    def class_(self, *, egg_sort: str) -> Callable[[TYPE], TYPE]:

        ...

    @overload
    def class_(self, cls: TYPE, /) -> TYPE:
        ...

    def class_(self, *args, **kwargs) -> Any:
        """
        Registers a class.
        """
        ...

    # We seperate the function and method overloads to make it simpler to know if we are modifying a function or method,
    # So that we can add the functions eagerly to the registry and wait on the methods till we process the class.

    # We have to seperate method/function overloads for those that use the T params and those that don't
    # Otherwise, if you say just pass in `cost` then the T param is inferred as `Nothing` and
    # It will break the typing.
    @overload
    def method(  # type: ignore
        self, *, egg_fn: Optional[str] = None, cost: Optional[int] = None
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

    @overload
    def method(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[T] = None,
        merge: Optional[Callable[[T, T], T]] = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        ...

    def method(self, *, egg_fn=None, default=None, cost=None, merge=None) -> Any:
        ...

    @overload
    def function(self, fn: CALLABLE, /) -> CALLABLE:
        ...

    @overload
    def function(
        self, *, egg_fn: Optional[str] = None, cost: int = 0
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

    @overload
    def function(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: int = 0,
        default: Optional[T] = None,
        merge: Optional[Callable[[T, T], T]] = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        ...

    def function(self, *args, **kwargs) -> Any:
        """
        Registers a function.
        """
        ...

    def register(self, *values: Rewrite | Rule | Action) -> None:
        """
        Registers any number of rewrites or rules.
        """
        ...


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


def rewrite(lhs: EXPR) -> _RewriteBuilder[EXPR]:
    return _RewriteBuilder(lhs=lhs)


def eq(expr: EXPR) -> _EqBuilder[EXPR]:
    return _EqBuilder(expr)


def panic(message: str) -> Panic:
    return Panic(message)


def let(name: str, expr: BaseExpr) -> Let:
    return Let(name, expr)


def delete(expr: BaseExpr) -> Delete:
    return Delete(expr)


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    return _UnionBuilder(lhs=lhs)


def set_(lhs: EXPR) -> _SetBuilder[EXPR]:
    return _SetBuilder(lhs=lhs)


def if_(*facts: Fact) -> _RuleBuilder:
    return _RuleBuilder(facts=facts)


def var(name: str, bound: type[EXPR]) -> EXPR:
    ...


def vars(name: str, bound: type[EXPR]) -> Iterable[EXPR]:
    ...


@dataclass
class _RewriteBuilder(Generic[EXPR]):
    lhs: EXPR

    def to(self, rhs: EXPR, *conditions: Fact) -> Rewrite:
        return Rewrite(lhs=self.lhs, rhs=rhs, conditions=list(conditions))

    def __str__(self) -> str:
        return f"rewrite({self.lhs})"


@dataclass
class _EqBuilder(Generic[EXPR]):
    expr: BaseExpr

    def to(self, *exprs: EXPR) -> Eq:
        return Eq([self.expr, *exprs])

    def __str__(self) -> str:
        return f"eq({self.expr})"


@dataclass
class _SetBuilder(Generic[EXPR]):
    lhs: BaseExpr

    def to(self, rhs: EXPR) -> Set:
        return Set(lhs=self.lhs, rhs=rhs)

    def __str__(self) -> str:
        return f"set_({self.lhs})"


@dataclass
class _UnionBuilder(Generic[EXPR]):
    lhs: BaseExpr

    def with_(self, rhs: EXPR) -> Union_:
        return Union_(lhs=self.lhs, rhs=rhs)

    def __str__(self) -> str:
        return f"union({self.lhs})"


@dataclass
class _RuleBuilder:
    facts: tuple[Fact, ...]

    def then(self, *actions: Action) -> Rule:
        return Rule(actions, self.facts)


@dataclass
class Rewrite:
    lhs: BaseExpr
    rhs: BaseExpr
    conditions: list[Fact]

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self.rhs, *self.conditions]))
        return f"rewrite({self.lhs}).to({args_str})"


@dataclass
class Eq:
    exprs: list[BaseExpr]

    def __str__(self) -> str:
        first, *rest = self.exprs
        args_str = ", ".join(map(str, rest))
        return f"eq({first}).to({args_str})"


Fact = Union["Unit", Eq]


@dataclass
class Delete:
    expr: BaseExpr

    def __str__(self) -> str:
        return f"delete({self.expr})"


@dataclass
class Panic:
    message: str

    def __str__(self) -> str:
        return f"panic({self.message})"


@dataclass
class Union_(Generic[EXPR]):
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"union({self.lhs}).with_({self.rhs})"


@dataclass
class Set:
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"set_({self.lhs}).to({self.rhs})"


@dataclass
class Let:
    name: str
    value: BaseExpr

    def __str__(self) -> str:
        return f"let({self.name}, {self.value})"


Action = Union[Let, Set, Delete, Union_, Panic, "Expr"]


@dataclass
class Rule:
    header: tuple[Action, ...]
    body: tuple[Fact, ...]
