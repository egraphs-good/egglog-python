"""Python conversion policy over canonical protobuf sort references."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, cast

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Ref, StructuralView
from .runtime import RuntimeClass, RuntimeExpr, _class_for_sort

__all__ = ["ConvertError", "convert", "converter", "get_type_args"]

V = TypeVar("V")
CONVERSIONS: dict[tuple[type | RuntimeClass, RuntimeClass], tuple[int, Callable[[Any], RuntimeExpr]]] = {}
TYPE_ARGS = ContextVar[tuple[RuntimeClass, ...]]("TYPE_ARGS", default=())


class ConvertError(Exception):
    pass


def converter(from_type: type | RuntimeClass, to_type: type | RuntimeClass, fn: Callable, cost: int = 1) -> None:
    """Register a Python converter; target semantics remain owned sort records."""
    if not isinstance(to_type, RuntimeClass):
        raise TypeError(f"Expected an egglog return type, got {to_type!r}")
    if from_type == to_type:
        return
    key = from_type, to_type
    if key in CONVERSIONS and CONVERSIONS[key][0] <= cost:
        return
    CONVERSIONS[key] = cost, fn
    # Preserve the existing transitive conversion policy without storing type
    # trees or synthesizing a callable signature.
    for (source, target), (other_cost, other_fn) in tuple(CONVERSIONS.items()):
        if target == from_type:

            def composed(value: Any, first=other_fn, second=fn) -> RuntimeExpr:
                return second(first(value))

            converter(source, to_type, composed, cost + other_cost)
        if to_type == source:

            def composed(value: Any, first=fn, second=other_fn) -> RuntimeExpr:
                return second(first(value))

            converter(from_type, target, composed, cost + other_cost)


def resolve_literal(sort: Ref, value: object) -> RuntimeExpr:
    """Resolve Python values against a canonical concrete expected sort."""
    if not isinstance(sort, Ref) or sort.role != "sorts":
        msg = "Conversions require a canonical sort reference"
        raise TypeError(msg)
    if isinstance(value, RuntimeExpr):
        node: pb.Node = value.__egg_ref__.read()
        actual = value.__egg_ref__.owner.ref("sorts", node.sort_id)
        if StructuralView(actual) == StructuralView(sort):
            return value
        source: type | RuntimeClass = _class_for_sort(actual)
    else:
        source = type(value)
    target = _class_for_sort(sort)
    candidates = source.__mro__ if isinstance(source, type) and not isinstance(source, RuntimeClass) else (source,)
    source_sort: pb.Sort = sort.read()
    assert source_sort.kind is not None
    arguments = (
        tuple(_class_for_sort(sort.owner.ref("sorts", index)) for index in source_sort.kind.value.args)
        if source_sort.kind.field == "family"
        else ()
    )
    for candidate in candidates:
        conversion = CONVERSIONS.get((candidate, target))
        if conversion is not None:
            with with_type_args(arguments):
                result = conversion[1](value)
            if not isinstance(result, RuntimeExpr):
                raise ConvertError(f"Converter to {target} returned {type(result).__name__}, not an expression")
            payload: pb.Node = result.__egg_ref__.read()
            actual = result.__egg_ref__.owner.ref("sorts", payload.sort_id)
            if StructuralView(actual) != StructuralView(sort):
                raise ConvertError(f"Converter to {target} returned the wrong canonical sort")
            return result
    raise ConvertError(f"Cannot convert {value} of type {source} to {target}")


def convert(source: object, target: type[V]) -> V:
    runtime_target = cast("object", target)
    if not isinstance(runtime_target, RuntimeClass) or runtime_target.__egg_sort__ is None:
        msg = "Conversion target must be a concrete egglog type"
        raise TypeError(msg)
    return cast("V", resolve_literal(runtime_target.__egg_sort__, source))


def convert_to_same_type(source: object, target: RuntimeExpr) -> RuntimeExpr:
    node: pb.Node = target.__egg_ref__.read()
    return resolve_literal(target.__egg_ref__.owner.ref("sorts", node.sort_id), source)


@contextmanager
def with_type_args(arguments: tuple[RuntimeClass, ...]) -> Generator[None, None, None]:
    token = TYPE_ARGS.set(arguments)
    try:
        yield
    finally:
        TYPE_ARGS.reset(token)


def get_type_args() -> tuple[RuntimeClass, ...]:
    return TYPE_ARGS.get()
