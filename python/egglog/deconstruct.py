"""
Utility functions to deconstruct expressions in Python.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, TypeVar, overload

from typing_extensions import TypeVarTuple, Unpack

from .declarations import *
from .egraph import BaseExpr, Expr
from .runtime import *
from .thunk import *

if TYPE_CHECKING:
    from .builtins import Bool, PyObject, String, UnstableFn, f64, i64


T = TypeVar("T", bound=BaseExpr)
TS = TypeVarTuple("TS", default=Unpack[tuple[BaseExpr, ...]])

__all__ = ["get_callable_args", "get_callable_fn", "get_let_name", "get_literal_value", "get_var_name"]


@overload
def get_literal_value(x: String) -> str | None: ...


@overload
def get_literal_value(x: Bool) -> bool | None: ...


@overload
def get_literal_value(x: i64) -> int | None: ...


@overload
def get_literal_value(x: f64) -> float | None: ...


@overload
def get_literal_value(x: PyObject) -> object: ...


@overload
def get_literal_value(x: UnstableFn[T, Unpack[TS]]) -> Callable[[Unpack[TS]], T] | None: ...


@overload
def get_literal_value(x: Expr) -> None: ...


def get_literal_value(x: object) -> object:
    """
    Returns the literal value of an expression if it is a literal.
    If it is not a literal, returns None.
    """
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    match x.__egg_typed_expr__.expr:
        case LitDecl(v):
            return v
        case PyObjectDecl(obj):
            return obj
        case PartialCallDecl(call):
            fn, args = _deconstruct_call_decl(x.__egg_decls_thunk__, call)
            if not args:
                return fn
            return partial(fn, *args)
    return None


def get_let_name(x: BaseExpr) -> str | None:
    """
    Check if the expression is a `let` expression and return the name of the variable.
    If it is not a `let` expression, return None.
    """
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    match x.__egg_typed_expr__.expr:
        case LetRefDecl(name):
            return name
    return None


def get_var_name(x: BaseExpr) -> str | None:
    """
    Check if the expression is a variable and return its name.
    If it is not a variable, return None.
    """
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    match x.__egg_typed_expr__.expr:
        case UnboundVarDecl(name, _egg_name):
            return name
    return None


def get_callable_fn(x: T) -> Callable[..., T] | T | None:
    """
    Gets the function of an expression, or if it's a constant or classvar, return that.
    """
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    match x.__egg_typed_expr__.expr:
        case CallDecl() as call:
            fn, _ = _deconstruct_call_decl(x.__egg_decls_thunk__, call)
            return fn
    return None


@overload
def get_callable_args(x: T, fn: None = ...) -> tuple[BaseExpr, ...]: ...


@overload
def get_callable_args(x: T, fn: Callable[[Unpack[TS]], T]) -> tuple[Unpack[TS]] | None: ...


def get_callable_args(x: T, fn: Callable[[Unpack[TS]], T] | None = None) -> tuple[Unpack[TS]] | None:
    """
    Gets all the arguments of an expression.
    If a function is provided, it will only return the arguments if the expression is a call
    to that function.

    Note that recursively calling the arguments is the safe way to walk the expression tree.
    """
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    match x.__egg_typed_expr__.expr:
        case CallDecl() as call:
            actual_fn, args = _deconstruct_call_decl(x.__egg_decls_thunk__, call)
            if fn is None:
                return args
            # Compare functions and classes without considering bound type parameters, so that you can pass
            # in a binding like Vec[i64] and match Vec[i64](...) or Vec(...) calls.
            if (
                isinstance(actual_fn, RuntimeFunction)
                and isinstance(fn, RuntimeFunction)
                and actual_fn.__egg_ref__ == fn.__egg_ref__
            ):
                return args
            if (
                isinstance(actual_fn, RuntimeClass)
                and isinstance(fn, RuntimeClass)
                and actual_fn.__egg_tp__.name == fn.__egg_tp__.name
            ):
                return args
    return None


def _deconstruct_call_decl(
    decls_thunk: Callable[[], Declarations], call: CallDecl
) -> tuple[Callable, tuple[object, ...]]:
    """
    Deconstructs a CallDecl into a runtime callable and its arguments.
    """
    args = call.args
    arg_exprs = tuple(RuntimeExpr(decls_thunk, Thunk.value(a)) for a in args)
    # TODO: handle values? Like constants
    if isinstance(call.callable, InitRef):
        return RuntimeClass(
            decls_thunk,
            TypeRefWithVars(call.callable.class_name, tuple(tp.to_var() for tp in (call.bound_tp_params or []))),
        ), arg_exprs
    egg_bound = (
        JustTypeRef(call.callable.class_name, call.bound_tp_params or ())
        if isinstance(call.callable, ClassMethodRef)
        else None
    )

    return RuntimeFunction(decls_thunk, Thunk.value(call.callable), egg_bound), arg_exprs
