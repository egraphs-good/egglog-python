from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType, TypeVar, cast

from .declarations import *
from .pretty import *
from .runtime import *
from .thunk import *

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from .egraph import Expr

__all__ = ["convert", "convert_to_same_type", "converter", "resolve_literal"]
# Mapping from (source type, target type) to and function which takes in the runtimes values of the source and return the target
TypeName = NewType("TypeName", str)
CONVERSIONS: dict[tuple[type | TypeName, TypeName], tuple[int, Callable]] = {}
# Global declerations to store all convertable types so we can query if they have certain methods or not
_CONVERSION_DECLS = Declarations.create()
# Defer a list of declerations to be added to the global declerations, so that we can not trigger them procesing
# until we need them
_TO_PROCESS_DECLS: list[DeclerationsLike] = []


def _retrieve_conversion_decls() -> Declarations:
    _CONVERSION_DECLS.update(*_TO_PROCESS_DECLS)
    _TO_PROCESS_DECLS.clear()
    return _CONVERSION_DECLS


T = TypeVar("T")
V = TypeVar("V", bound="Expr")


class ConvertError(Exception):
    pass


def converter(from_type: type[T], to_type: type[V], fn: Callable[[T], V], cost: int = 1) -> None:
    """
    Register a converter from some type to an egglog type.
    """
    to_type_name = process_tp(to_type)
    if not isinstance(to_type_name, str):
        raise TypeError(f"Expected return type to be a egglog type, got {to_type_name}")
    _register_converter(process_tp(from_type), to_type_name, fn, cost)


def _register_converter(a: type | TypeName, b: TypeName, a_b: Callable, cost: int) -> None:
    """
    Registers a converter from some type to an egglog type, if not already registered.

    Also adds transitive converters, i.e. if registering A->B and there is already B->C, then A->C will be registered.
    Also, if registering A->B and there is already D->A, then D->B will be registered.
    """
    if a == b:
        return
    if (a, b) in CONVERSIONS and CONVERSIONS[(a, b)][0] <= cost:
        return
    CONVERSIONS[(a, b)] = (cost, a_b)
    for (c, d), (other_cost, c_d) in list(CONVERSIONS.items()):
        if b == c:
            _register_converter(a, d, _ComposedConverter(a_b, c_d), cost + other_cost)
        if a == d:
            _register_converter(c, b, _ComposedConverter(c_d, a_b), cost + other_cost)


@dataclass
class _ComposedConverter:
    """
    A converter which is composed of multiple converters.

    _ComposeConverter(a_b, b_c) is equivalent to lambda x: b_c(a_b(x))

    We use the dataclass instead of the lambda to make it easier to debug.
    """

    a_b: Callable
    b_c: Callable

    def __call__(self, x: object) -> object:
        return self.b_c(self.a_b(x))

    def __str__(self) -> str:
        return f"{self.b_c} ∘ {self.a_b}"


def convert(source: object, target: type[V]) -> V:
    """
    Convert a source object to a target type.
    """
    assert isinstance(target, RuntimeClass)
    return cast(V, resolve_literal(target.__egg_tp__, source, target.__egg_decls_thunk__))


def convert_to_same_type(source: object, target: RuntimeExpr) -> RuntimeExpr:
    """
    Convert a source object to the same type as the target.
    """
    tp = target.__egg_typed_expr__.tp
    return resolve_literal(tp.to_var(), source, Thunk.value(target.__egg_decls__))


def process_tp(tp: type | RuntimeClass) -> TypeName | type:
    """
    Process a type before converting it, to add it to the global declerations and resolve to a ref.
    """
    if isinstance(tp, RuntimeClass):
        _TO_PROCESS_DECLS.append(tp)
        egg_tp = tp.__egg_tp__
        if egg_tp.args:
            raise TypeError(f"Cannot register a converter for a generic type, got {tp}")
        return TypeName(egg_tp.name)
    return tp


def min_convertable_tp(a: object, b: object, name: str) -> TypeName:
    """
    Returns the minimum convertable type between a and b, that has a method `name`, raising a ConvertError if no such type exists.
    """
    decls = _retrieve_conversion_decls()
    a_tp = _get_tp(a)
    b_tp = _get_tp(b)
    a_converts_to = {
        to: c for ((from_, to), (c, _)) in CONVERSIONS.items() if from_ == a_tp and decls.has_method(to, name)
    }
    b_converts_to = {
        to: c for ((from_, to), (c, _)) in CONVERSIONS.items() if from_ == b_tp and decls.has_method(to, name)
    }
    if isinstance(a_tp, str):
        a_converts_to[a_tp] = 0
    if isinstance(b_tp, str):
        b_converts_to[b_tp] = 0
    common = set(a_converts_to) & set(b_converts_to)
    if not common:
        raise ConvertError(f"Cannot convert {a_tp} and {b_tp} to a common type")
    return min(common, key=lambda tp: a_converts_to[tp] + b_converts_to[tp])


def identity(x: object) -> object:
    return x


TYPE_ARGS = ContextVar[tuple[RuntimeClass, ...]]("TYPE_ARGS")


def get_type_args() -> tuple[type, ...]:
    """
    Get the type args for the type being converted.
    """
    return cast(tuple[type, ...], TYPE_ARGS.get())


@contextmanager
def with_type_args(args: tuple[JustTypeRef, ...], decls: Callable[[], Declarations]) -> Generator[None, None, None]:
    token = TYPE_ARGS.set(tuple(RuntimeClass(decls, a.to_var()) for a in args))
    try:
        yield
    finally:
        TYPE_ARGS.reset(token)


def resolve_literal(
    tp: TypeOrVarRef, arg: object, decls: Callable[[], Declarations] = _retrieve_conversion_decls
) -> RuntimeExpr:
    arg_type = _get_tp(arg)

    # If we have any type variables, dont bother trying to resolve the literal, just return the arg
    try:
        tp_just = tp.to_just()
    except NotImplementedError:
        # If this is a var, it has to be a runtime expession
        assert isinstance(arg, RuntimeExpr), f"Expected a runtime expression, got {arg}"
        return arg
    tp_name = TypeName(tp_just.name)
    if arg_type == tp_name:
        # If the type is an egg type, it has to be a runtime expr
        assert isinstance(arg, RuntimeExpr)
        return arg
    # Try all parent types as well, if we are converting from a Python type
    for arg_type_instance in arg_type.__mro__ if isinstance(arg_type, type) else [arg_type]:
        try:
            fn = CONVERSIONS[(arg_type_instance, tp_name)][1]
        except KeyError:
            continue
        break
    else:
        raise ConvertError(f"Cannot convert {arg_type} to {tp_name}")
    with with_type_args(tp_just.args, decls):
        return fn(arg)


def _get_tp(x: object) -> TypeName | type:
    if isinstance(x, RuntimeExpr):
        return TypeName(x.__egg_typed_expr__.tp.name)
    tp = type(x)
    # If this value has a custom metaclass, let's use that as our index instead of the type
    if type(tp) is not type:
        return type(tp)
    return tp
