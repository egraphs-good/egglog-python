from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from .declarations import *
from .pretty import *
from .runtime import *

if TYPE_CHECKING:
    from collections.abc import Callable

    from .egraph import Expr

__all__ = ["convert", "converter", "resolve_literal"]
# Mapping from (source type, target type) to and function which takes in the runtimes values of the source and return the target
CONVERSIONS: dict[tuple[type | JustTypeRef, JustTypeRef], tuple[int, Callable]] = {}
# Global declerations to store all convertable types so we can query if they have certain methods or not
CONVERSIONS_DECLS = Declarations()

T = TypeVar("T")
V = TypeVar("V", bound="Expr")


class ConvertError(Exception):
    pass


def converter(from_type: type[T], to_type: type[V], fn: Callable[[T], V], cost: int = 1) -> None:
    """
    Register a converter from some type to an egglog type.
    """
    to_type_name = process_tp(to_type)
    if not isinstance(to_type_name, JustTypeRef):
        raise TypeError(f"Expected return type to be a egglog type, got {to_type_name}")
    _register_converter(process_tp(from_type), to_type_name, fn, cost)


def _register_converter(a: type | JustTypeRef, b: JustTypeRef, a_b: Callable, cost: int) -> None:
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
    return cast(V, resolve_literal(target.__egg_tp__, source))


def convert_to_same_type(source: object, target: RuntimeExpr) -> RuntimeExpr:
    """
    Convert a source object to the same type as the target.
    """
    tp = target.__egg_typed_expr__.tp
    return resolve_literal(tp.to_var(), source)


def process_tp(tp: type | RuntimeClass) -> JustTypeRef | type:
    """
    Process a type before converting it, to add it to the global declerations and resolve to a ref.
    """
    global CONVERSIONS_DECLS
    if isinstance(tp, RuntimeClass):
        CONVERSIONS_DECLS |= tp
        return tp.__egg_tp__.to_just()
    return tp


def min_convertable_tp(a: object, b: object, name: str) -> JustTypeRef:
    """
    Returns the minimum convertable type between a and b, that has a method `name`, raising a ConvertError if no such type exists.
    """
    a_tp = _get_tp(a)
    b_tp = _get_tp(b)
    a_converts_to = {
        to: c
        for ((from_, to), (c, _)) in CONVERSIONS.items()
        if from_ == a_tp and CONVERSIONS_DECLS.has_method(to.name, name)
    }
    b_converts_to = {
        to: c
        for ((from_, to), (c, _)) in CONVERSIONS.items()
        if from_ == b_tp and CONVERSIONS_DECLS.has_method(to.name, name)
    }
    if isinstance(a_tp, JustTypeRef):
        a_converts_to[a_tp] = 0
    if isinstance(b_tp, JustTypeRef):
        b_converts_to[b_tp] = 0
    common = set(a_converts_to) & set(b_converts_to)
    if not common:
        raise ConvertError(f"Cannot convert {a_tp} and {b_tp} to a common type")
    return min(common, key=lambda tp: a_converts_to[tp] + b_converts_to[tp])


def identity(x: object) -> object:
    return x


def resolve_literal(tp: TypeOrVarRef, arg: object) -> RuntimeExpr:
    arg_type = _get_tp(arg)

    # If we have any type variables, dont bother trying to resolve the literal, just return the arg
    try:
        tp_just = tp.to_just()
    except NotImplementedError:
        # If this is a var, it has to be a runtime exprssions
        assert isinstance(arg, RuntimeExpr)
        return arg
    if arg_type == tp_just:
        # If the type is an egg type, it has to be a runtime expr
        assert isinstance(arg, RuntimeExpr)
        return arg
    # Try all parent types as well, if we are converting from a Python type
    for arg_type_instance in arg_type.__mro__ if isinstance(arg_type, type) else [arg_type]:
        try:
            fn = CONVERSIONS[(cast(JustTypeRef | type, arg_type_instance), tp_just)][1]
        except KeyError:
            continue
        break
    else:
        raise ConvertError(f"Cannot convert {arg_type} to {tp_just}")
    return fn(arg)


def _get_tp(x: object) -> JustTypeRef | type:
    if isinstance(x, RuntimeExpr):
        return x.__egg_typed_expr__.tp
    tp = type(x)
    # If this value has a custom metaclass, let's use that as our index instead of the type
    if type(tp) != type:
        return type(tp)
    return tp
