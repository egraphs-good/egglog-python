from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from .declarations import *
from .pretty import *
from .runtime import *
from .thunk import *
from .type_constraint_solver import TypeConstraintError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from .egraph import BaseExpr
    from .type_constraint_solver import TypeConstraintSolver

__all__ = ["ConvertError", "convert", "convert_to_same_type", "converter", "resolve_literal"]
# Mapping from (source type, target type) to and function which takes in the runtimes values of the source and return the target
CONVERSIONS: dict[tuple[type | JustTypeRef, JustTypeRef], tuple[int, Callable]] = {}
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
V = TypeVar("V", bound="BaseExpr")


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
        if _is_type_compatible(b, c):
            _register_converter(
                a, d, _ComposedConverter(a_b, c_d, c.args if isinstance(c, JustTypeRef) else ()), cost + other_cost
            )
        if _is_type_compatible(a, d):
            _register_converter(
                c, b, _ComposedConverter(c_d, a_b, a.args if isinstance(a, JustTypeRef) else ()), cost + other_cost
            )


def _is_type_compatible(source: type | JustTypeRef, target: type | JustTypeRef) -> bool:
    """
    Types must be equal or also support unbound to bound typevar like B -> B[C]
    """
    if source == target:
        return True
    if isinstance(source, JustTypeRef) and isinstance(target, JustTypeRef) and source.args and not target.args:
        return source.name == target.name
        # TODO: Support case where B[T] where T is typevar is mapped to B[C]
    return False


@dataclass
class _ComposedConverter:
    """
    A converter which is composed of multiple converters.

    _ComposeConverter(a_b, b_c) is equivalent to lambda x: b_c(a_b(x))

    We use the dataclass instead of the lambda to make it easier to debug.
    """

    a_b: Callable
    b_c: Callable
    b_args: tuple[JustTypeRef, ...]

    def __call__(self, x: object) -> object:
        # if we have A -> B and B[C] -> D then we should use (C,) as the type args
        # when converting from A -> B
        if self.b_args:
            with with_type_args(self.b_args, _retrieve_conversion_decls):
                first_res = self.a_b(x)
        else:
            first_res = self.a_b(x)
        return self.b_c(first_res)

    def __str__(self) -> str:
        return f"{self.b_c} ∘ {self.a_b}"


def convert(source: object, target: type[V]) -> V:
    """
    Convert a source object to a target type.
    """
    assert isinstance(target, RuntimeClass)
    return cast("V", resolve_literal(target.__egg_tp__, source, target.__egg_decls_thunk__))


def convert_to_same_type(source: object, target: RuntimeExpr) -> RuntimeExpr:
    """
    Convert a source object to the same type as the target.
    """
    tp = target.__egg_typed_expr__.tp
    return resolve_literal(tp.to_var(), source, Thunk.value(target.__egg_decls__))


def process_tp(tp: type | RuntimeClass) -> JustTypeRef | type:
    """
    Process a type before converting it, to add it to the global declerations and resolve to a ref.
    """
    if isinstance(tp, RuntimeClass):
        _TO_PROCESS_DECLS.append(tp)
        egg_tp = tp.__egg_tp__
        return egg_tp.to_just()
    return tp


def min_convertable_tp(a: object, b: object, name: str) -> JustTypeRef:
    """
    Returns the minimum convertable type between a and b, that has a method `name`, raising a ConvertError if no such type exists.
    """
    decls = _retrieve_conversion_decls()
    a_tp = _get_tp(a)
    b_tp = _get_tp(b)
    # Make sure at least one of the types has the method, to avoid issue with == upcasting improperly
    if not (
        (isinstance(a_tp, JustTypeRef) and decls.has_method(a_tp.name, name))
        or (isinstance(b_tp, JustTypeRef) and decls.has_method(b_tp.name, name))
    ):
        raise ConvertError(f"Neither {a_tp} nor {b_tp} has method {name}")
    a_converts_to = {
        to: c for ((from_, to), (c, _)) in CONVERSIONS.items() if from_ == a_tp and decls.has_method(to.name, name)
    }
    b_converts_to = {
        to: c for ((from_, to), (c, _)) in CONVERSIONS.items() if from_ == b_tp and decls.has_method(to.name, name)
    }
    if isinstance(a_tp, JustTypeRef) and decls.has_method(a_tp.name, name):
        a_converts_to[a_tp] = 0
    if isinstance(b_tp, JustTypeRef) and decls.has_method(b_tp.name, name):
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
    return cast("tuple[type, ...]", TYPE_ARGS.get())


@contextmanager
def with_type_args(args: tuple[JustTypeRef, ...], decls: Callable[[], Declarations]) -> Generator[None, None, None]:
    token = TYPE_ARGS.set(tuple(RuntimeClass(decls, a.to_var()) for a in args))
    try:
        yield
    finally:
        TYPE_ARGS.reset(token)


def resolve_literal(
    tp: TypeOrVarRef,
    arg: object,
    decls: Callable[[], Declarations] = _retrieve_conversion_decls,
    tcs: TypeConstraintSolver | None = None,
    cls_name: str | None = None,
) -> RuntimeExpr:
    """
    Try to convert an object to a type, raising a ConvertError if it is not possible.

    If the type has vars in it, they will be tried to be resolved into concrete vars based on the type constraint solver.

    If it cannot be resolved, we assume that the value passed in will resolve it.
    """
    arg_type = _get_tp(arg)

    # If we have any type variables, dont bother trying to resolve the literal, just return the arg
    try:
        tp_just = tp.to_just()
    except NotImplementedError:
        # If this is a generic arg but passed in a non runtime expression, try to resolve the generic
        # args first based on the existing type constraint solver
        if tcs:
            try:
                tp_just = tcs.substitute_typevars(tp, cls_name)
            # If we can't resolve the type var yet, then just assume it is the right value
            except TypeConstraintError:
                assert isinstance(arg, RuntimeExpr), f"Expected a runtime expression, got {arg}"
                tp_just = arg.__egg_typed_expr__.tp
        else:
            # If this is a var, it has to be a runtime expession
            assert isinstance(arg, RuntimeExpr), f"Expected a runtime expression, got {arg}"
            return arg
    if tcs:
        tcs.infer_typevars(tp, tp_just, cls_name)
    if arg_type == tp_just:
        # If the type is an egg type, it has to be a runtime expr
        assert isinstance(arg, RuntimeExpr)
        return arg
    # Try all parent types as well, if we are converting from a Python type
    for arg_type_instance in arg_type.__mro__ if isinstance(arg_type, type) else [arg_type]:
        if (key := (arg_type_instance, tp_just)) in CONVERSIONS:
            fn = CONVERSIONS[key][1]
            break
        # Try broadening if we have a convert to the general type instead of the specific one too, for generics
        if tp_just.args and (key := (arg_type_instance, JustTypeRef(tp_just.name))) in CONVERSIONS:
            fn = CONVERSIONS[key][1]
            break
    # if we didn't find any raise an error
    else:
        raise ConvertError(f"Cannot convert {arg_type} to {tp_just}")
    with with_type_args(tp_just.args, decls):
        return fn(arg)


def _debug_print_converers():
    """
    Prints a mapping of all source types to target types that have a conversion function.
    """
    source_to_targets = defaultdict(list)
    for source, target in CONVERSIONS:
        source_to_targets[source].append(target)


def _get_tp(x: object) -> JustTypeRef | type:
    if isinstance(x, RuntimeExpr):
        return x.__egg_typed_expr__.tp
    tp = type(x)
    # If this value has a custom metaclass, let's use that as our index instead of the type
    if type(tp) is not type:
        return type(tp)
    return tp
