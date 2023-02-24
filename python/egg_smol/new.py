from __future__ import annotations
from typing import Optional, overload, TypeVar, Callable, ParamSpec, Any, NoReturn, Self
from dataclasses import dataclass, field

from . import bindings as egg

__all__ = ["Registry", "Rewrite", "Rule"]

T = TypeVar("T")
P = ParamSpec("P")

TYPE = TypeVar("TYPE", bound=type)


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

    @overload
    def __call__(
        self,
        *,
        egg_fn: Optional[str] = None,
        default: Optional[T] = None,
        merge: Optional[Callable[[T, T], T]] = None,
        cost: int = 0,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Registers a function with some paramaters.
        """
        ...

    @overload
    def __call__(self, fn: Callable[P, T], /) -> Callable[P, T]:
        """
        Registers a function.
        """
        ...

    @overload
    def __call__(self, *, egg_sort: str) -> Callable[[TYPE], TYPE]:
        """
        Registers a class with some paramaters.
        """
        ...

    # Class without paramaters
    @overload
    def __call__(self, cls: TYPE, /) -> TYPE:
        """
        Registers a class.
        """
        ...

    @overload
    def __call__(self, rewrite_or_rule: Rewrite | Rule, /) -> None:
        """
        Registers a rewrite or a rule.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """
        Register a class, type, rewrite or a rule.
        """
        ...

    def to_egg_expr(self, expr: Any) -> egg._Expr:
        """
        Convert a python expression to an egg expression.
        """
        ...

    def from_egg_expr(self, expr: egg._Expr) -> Any:
        """
        Convert an egg expression to a python expression.
        """
        ...

    def integrity_check(self) -> None:
        """
        Checks that:
            1. None of the functions and classes  have the same names
            2. Each mapping is bidrectional
        """
        ...


@dataclass
class Rewrite:
    ...


@dataclass
class Rule:
    ...


@dataclass
class Eq:
    ...


EXPR = TypeVar("EXPR", bound="Expr")


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


class Unit(Expr):
    ...
