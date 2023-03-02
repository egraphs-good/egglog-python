from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from types import UnionType
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
    cast,
    get_args,
    overload,
)

from .declarations import *
from .runtime import *

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
    "Fact",
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

    _decls: Declarations = field(default_factory=Declarations)

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

    def _on_register_action(self, decl: ActionDecl) -> None:
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
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    def method(self, *, egg_fn=None, default=None, cost=None, merge=None) -> Any:
        ...

    @overload
    def function(self, fn: CALLABLE, /) -> CALLABLE:
        ...

    @overload
    def function(
        self, *, egg_fn: Optional[str] = None, cost: Optional[int] = None
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

    @overload
    def function(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    def function(self, *args, **kwargs) -> Any:
        """
        Registers a function.
        """
        # If we have any positional args, then we are calling it directly on a function
        if args:
            assert len(args) == 1
            return self._function(args[0])
        # otherwise, we are passing some keyword args, so save those, and then return a partial
        return lambda fn: self._function(fn, **kwargs)

    def _function(
        self,
        fn: Callable[P, EXPR],
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
    ) -> Callable[P, EXPR]:
        """
        Uncurried version of function decorator
        """
        name = fn.__name__
        # TODO: Verify that the signature is correct, that we have all the types we need
        sig = signature(fn, eval_str=True)
        return_type = self._resolve_type_annotation(sig.return_annotation)
        arg_types = tuple(
            self._resolve_type_annotation(t.annotation) for t in sig.parameters.values()
        )
        default_decl = None if default is None else expr_to_decl(default)
        merge_decl = (
            None
            if merge is None
            else expr_to_decl(
                merge(
                    cast(EXPR, self._create_var(return_type, "old")),
                    cast(EXPR, self._create_var(return_type, "new")),
                )
            )
        )
        decl = FunctionDecl(
            return_type=return_type,
            arg_types=arg_types,
            cost=cost,
            default=default_decl,
            merge=merge_decl,
        )

        if name in self._decls.functions:
            raise ValueError(f"Function {name} already registered")
        self._decls.functions[name] = decl
        egg_fn = egg_fn or name
        if egg_fn in self._decls.egg_fn_to_callable_ref:
            raise ValueError(f"Egg function {egg_fn} already registered")
        ref: FunctionRef = FunctionRef(name)
        self._decls.callable_ref_to_egg_fn[ref] = egg_fn or name
        self._decls.egg_fn_to_callable_ref[egg_fn] = ref
        return cast(Callable[P, EXPR], RuntimeFunction(self._decls, name))

    def _resolve_type_annotation(self, tp: object) -> TypeRef:
        if isinstance(tp, RuntimeClass):
            return TypeRef(tp.name)
        if isinstance(tp, RuntimeParamaterizedClass):
            return tp.ref
        if isinstance(tp, TypeVar):
            # TODO: Probably do a lookup in the class to map typevars to indices for the class
            raise TypeError("TypeVars are not supported")
        # If there is a union, it should be of a literal and another type to allow type promotion
        if isinstance(tp, UnionType):
            args = get_args(tp)
            if len(args) != 2:
                raise TypeError("Union types are only supported for type promotion")
            fst, snd = args
            if fst in {int, str}:
                return self._resolve_type_annotation(snd)
            if snd in {int, str}:
                return self._resolve_type_annotation(fst)
            raise TypeError("Union types are only supported for type promotion")
        raise TypeError(f"Unexpected type annotation {tp}")

    def register(self, *values: Rewrite | Rule | Action) -> None:
        """
        Registers any number of rewrites or rules.
        """
        for value in values:
            self._register_single(value)

    def _register_single(self, value: Rewrite | Rule | Action) -> None:
        if isinstance(value, Rewrite):
            decl = value._to_decl()
            self._decls.rewrites.append(decl)
            self._on_register_rewrite(decl)
        elif isinstance(value, Rule):
            decl = value._to_decl()
            self._decls.rules.append(decl)
            self._on_register_rule(decl)
        elif isinstance(value, Action):
            decl = action_to_decl(value)
            self._decls.actions.append(decl)
            self._on_register_action(decl)
        else:
            raise TypeError(f"Unexpected type {type(value)}")

    def _create_var(self, tp: TypeRef, name: str) -> BaseExpr:
        expr = RuntimeExpr(self._decls, tp, VarDecl(name))
        return cast(BaseExpr, expr)


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


def expr_to_decl(expr: BaseExpr) -> ExprDecl:
    assert isinstance(expr, RuntimeExpr)
    return expr.expr


EXPR = TypeVar("EXPR", bound="BaseExpr")


def decl_to_expr(expr: ExprDecl, source_expr: EXPR) -> EXPR:
    assert isinstance(source_expr, RuntimeExpr)
    return RuntimeExpr(source_expr.decls, source_expr.tp, expr)


@dataclass
class Rewrite:
    lhs: BaseExpr
    rhs: BaseExpr
    conditions: list[Fact]

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self.rhs, *self.conditions]))
        return f"rewrite({self.lhs}).to({args_str})"

    def _to_decl(self) -> RewriteDecl:
        return RewriteDecl(
            expr_to_decl(self.lhs),
            expr_to_decl(self.rhs),
            tuple(fact_to_decl(fact) for fact in self.conditions),
        )


@dataclass
class Eq:
    exprs: list[BaseExpr]

    def __str__(self) -> str:
        first, *rest = self.exprs
        args_str = ", ".join(map(str, rest))
        return f"eq({first}).to({args_str})"

    def _to_decl(self) -> EqDecl:
        return EqDecl(tuple(expr_to_decl(expr) for expr in self.exprs))


Fact = Union["Unit", Eq]


def fact_to_decl(fact: Fact) -> FactDecl:
    if isinstance(fact, Eq):
        return fact._to_decl()
    return expr_to_decl(fact)


@dataclass
class Delete:
    expr: BaseExpr

    def __str__(self) -> str:
        return f"delete({self.expr})"

    def _to_decl(self) -> DeleteDecl:
        decl = expr_to_decl(self.expr)
        if not isinstance(decl, CallDecl):
            raise ValueError(f"Can only delete calls not {decl}")
        return DeleteDecl(decl)


@dataclass
class Panic:
    message: str

    def __str__(self) -> str:
        return f"panic({self.message})"

    def _to_decl(self) -> PanicDecl:
        return PanicDecl(self.message)


@dataclass
class Union_(Generic[EXPR]):
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"union({self.lhs}).with_({self.rhs})"

    def _to_decl(self) -> UnionDecl:
        return UnionDecl(expr_to_decl(self.lhs), expr_to_decl(self.rhs))


@dataclass
class Set:
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"set_({self.lhs}).to({self.rhs})"

    def _to_decl(self) -> SetDecl:
        lhs = expr_to_decl(self.lhs)
        if not isinstance(lhs, CallDecl):
            raise ValueError(
                f"Can only create a call with a call for the lhs, got {lhs}"
            )
        return SetDecl(lhs, expr_to_decl(self.rhs))


@dataclass
class Let:
    name: str
    value: BaseExpr

    def __str__(self) -> str:
        return f"let({self.name}, {self.value})"

    def _to_decl(self) -> LetDecl:
        return LetDecl(self.name, expr_to_decl(self.value))


Action = Union[Let, Set, Delete, Union_, Panic, "BaseExpr"]


def action_to_decl(action: Action) -> ActionDecl:
    if isinstance(action, BaseExpr):
        return expr_to_decl(action)
    return action._to_decl()


@dataclass
class Rule:
    header: tuple[Action, ...]
    body: tuple[Fact, ...]

    def _to_decl(self) -> RuleDecl:
        return RuleDecl(
            tuple(action_to_decl(action) for action in self.header),
            tuple(fact_to_decl(fact) for fact in self.body),
        )
