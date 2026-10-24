"""Python conversion policy over canonical protobuf sort references."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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


def _sort_type_args(sort: Ref) -> tuple[RuntimeClass, ...]:
    record: pb.Sort = sort.read()
    assert record.kind is not None
    indices = (
        record.kind.value.args
        if record.kind.field == "application"
        else [record.kind.value.result, *record.kind.value.params]
        if record.kind.field == "func"
        else ()
    )
    return tuple(_class_for_sort(sort.owner.ref("sorts", index)) for index in indices)


@dataclass
class _ComposedConverter:
    first: Callable[[Any], RuntimeExpr]
    second: Callable[[Any], RuntimeExpr]
    intermediate: RuntimeClass

    def __call__(self, value: Any) -> RuntimeExpr:
        sort = self.intermediate.__egg_sort__
        if sort is None:
            converted = self.first(value)
        else:
            with with_type_args(_sort_type_args(sort)):
                converted = self.first(value)
        return self.second(converted)


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
        if isinstance(from_type, RuntimeClass) and (
            target == from_type
            or (
                target.__egg_sort__ is None
                and from_type.__egg_sort__ is not None
                and target.__egg_definition__ == from_type.__egg_definition__
            )
        ):
            converter(source, to_type, _ComposedConverter(other_fn, fn, from_type), cost + other_cost)
        if isinstance(source, RuntimeClass) and (
            to_type == source
            or (
                to_type.__egg_sort__ is None
                and source.__egg_sort__ is not None
                and to_type.__egg_definition__ == source.__egg_definition__
            )
        ):
            converter(from_type, target, _ComposedConverter(fn, other_fn, source), cost + other_cost)


def _lookup_conversion(
    source: type | RuntimeClass, target: RuntimeClass
) -> tuple[int, Callable[[Any], RuntimeExpr]] | None:
    """Compare exact/family costs, preserving source MRO within each and exact ties."""
    candidates = source.__mro__ if isinstance(source, type) and not isinstance(source, RuntimeClass) else (source,)
    assert target.__egg_sort__ is not None
    target_sort: pb.Sort = target.__egg_sort__.read()
    assert target_sort.kind is not None
    conversion = None
    for candidate in candidates:
        conversion = CONVERSIONS.get((candidate, target))
        if conversion is not None:
            break
    if target_sort.kind.field in {"application", "func"}:
        generic = next(
            (
                registered
                for candidate in candidates
                for (registered_source, registered_target), registered in CONVERSIONS.items()
                if registered_source == candidate
                and registered_target.__egg_sort__ is None
                and registered_target.__egg_definition__ == target.__egg_definition__
            ),
            None,
        )
        if generic is not None and (conversion is None or generic[0] < conversion[0]):
            return generic
    return conversion


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
    conversion = _lookup_conversion(source, target)
    if conversion is None:
        raise ConvertError(f"Cannot convert {value} of type {source} to {target}")
    with with_type_args(_sort_type_args(sort)):
        result = conversion[1](value)
    if not isinstance(result, RuntimeExpr):
        raise ConvertError(f"Converter to {target} returned {type(result).__name__}, not an expression")
    payload: pb.Node = result.__egg_ref__.read()
    actual = result.__egg_ref__.owner.ref("sorts", payload.sort_id)
    if StructuralView(actual) != StructuralView(sort):
        raise ConvertError(f"Converter to {target} returned the wrong canonical sort")
    return result


def _conversion_cost(sort: Ref, value: object) -> int | None:
    """Price a canonical input without invoking a converter or changing context."""
    node = value.__egg_ref__ if isinstance(value, RuntimeExpr) else value if isinstance(value, Ref) else None
    if node is not None:
        payload: pb.Node = node.read()
        actual = node.owner.ref("sorts", payload.sort_id)
        if StructuralView(actual) == StructuralView(sort):
            return 0
        source: type | RuntimeClass = _class_for_sort(actual)
    else:
        source = type(value)
    conversion = _lookup_conversion(source, _class_for_sort(sort))
    return conversion[0] if conversion is not None else None


def _binary_conversion_candidates(
    left: object, right: object
) -> Generator[tuple[RuntimeClass, int, int], int | None, None]:
    """Try the original receiver first; lazily price alternatives and RHS lower bounds."""
    original = left if isinstance(left, RuntimeExpr) else right
    assert isinstance(original, RuntimeExpr)
    reference = original.__egg_ref__
    payload: pb.Node = reference.read()
    original_sort = reference.owner.ref("sorts", payload.sort_id)
    original_class = _class_for_sort(original_sort)
    cost = _conversion_cost(original_sort, left)
    cutoff = None
    if cost is not None:
        cutoff = yield original_class, cost, 0
    # Exact signatures can finish before scanning the conversion graph or
    # preparing an unrelated target's member. Bounds also exclude losing ties.
    other_source: type | RuntimeClass = type(right)
    minimum_other = None
    if isinstance(right, RuntimeExpr):
        reference = right.__egg_ref__
        payload = reference.read()
        other_source = _class_for_sort(reference.owner.ref("sorts", payload.sort_id))
        minimum_other = 0
    sources = (
        other_source.__mro__
        if isinstance(other_source, type) and not isinstance(other_source, RuntimeClass)
        else (other_source,)
    )
    for (source, _), (cost, _) in CONVERSIONS.items():
        if source in sources and (minimum_other is None or cost < minimum_other):
            minimum_other = cost
    if (
        cutoff is not None
        and all(type(cost) is int and cost >= 0 for cost, _ in CONVERSIONS.values())
        and (minimum_other if minimum_other is not None else 0) >= cutoff
    ):
        return
    classes = dict.fromkeys(target for _, target in CONVERSIONS if target.__egg_sort__ is not None)
    for cls in classes:
        if cls == original_class:
            continue
        sort = cls.__egg_sort__
        assert sort is not None
        cost = _conversion_cost(sort, left)
        if cost is not None:
            yield cls, cost, minimum_other if minimum_other is not None else 0


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
