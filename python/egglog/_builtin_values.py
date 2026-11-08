"""Validated builtin value inspection and host conversions, independent of API generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from ._presentation import CallKind, presentation, validate_default, validate_view
from ._program import Builder, Ref, StructuralView
from .conversion import get_type_args, resolve_literal
from .egraph import BaseExpr
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction

if TYPE_CHECKING:
    from .builtins import Map, Maybe, MultiSet, Pair, Set, Vec

_T = TypeVar("_T", bound=BaseExpr)
_L = TypeVar("_L", bound=BaseExpr)
_R = TypeVar("_R", bound=BaseExpr)
_K = TypeVar("_K", bound=BaseExpr)
_V = TypeVar("_V", bound=BaseExpr)


@dataclass
class ExprValueError(AttributeError):
    """A symbolic expression is not in a locally inspectable value form."""

    expr: object
    allowed: str

    def __str__(self) -> str:
        return f"Cannot get Python value of {self.expr}, must be of form {self.allowed}. Try calling `extract` on it to get the underlying value."


@overload
def _pair_value(expr: Pair[_L, _R]) -> tuple[_L, _R]: ...


@overload
def _pair_value(expr: RuntimeExpr) -> tuple[RuntimeExpr, RuntimeExpr]: ...


def _pair_value(expr: object) -> tuple[object, object]:
    """Inspect owned children, never evaluate or manufacture a semantic Call."""
    assert isinstance(expr, RuntimeExpr)
    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    sort = reference.owner.ref("sorts", node.sort_id)
    record: pb.Sort = sort.read()
    if record.kind is None or record.kind.field != "primitive" or record.kind.value.name != "Pair":
        raise ExprValueError(expr, "Pair(left, right)")
    arguments = record.kind.value.args
    if len(arguments) != 2:
        msg = "Pair inspection requires two sort arguments"
        raise ValueError(msg)
    if (
        node.kind is not None
        and node.kind.field == "primitive_value"
        and node.kind.value.value is not None
        and node.kind.value.value.field == "pair"
    ):
        value = node.kind.value.value.value
        if not value.has_field("first") or not value.has_field("second"):
            msg = "Pair value requires both child references"
            raise ValueError(msg)
        children = reference.owner.ref("nodes", value.first), reference.owner.ref("nodes", value.second)
    else:
        raise ExprValueError(expr, "Pair(left, right)")
    for child, argument in zip(children, arguments, strict=True):
        payload: pb.Node = child.read()
        if StructuralView(child.owner.ref("sorts", payload.sort_id)) != StructuralView(
            sort.owner.ref("sorts", argument)
        ):
            msg = "Pair child has the wrong canonical sort"
            raise ValueError(msg)
    return RuntimeExpr(children[0]), RuntimeExpr(children[1])


@overload
def _maybe_value(expr: Maybe[_T]) -> _T | None: ...


@overload
def _maybe_value(expr: RuntimeExpr) -> RuntimeExpr | None: ...


def _maybe_value(expr: object) -> object:
    """Inspect exact optional presence while leaving a present child symbolic."""
    assert isinstance(expr, RuntimeExpr)
    from .runtime import _definition_at  # noqa: PLC0415

    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    sort = reference.owner.ref("sorts", node.sort_id)
    record: pb.Sort = sort.read()
    if (
        record.kind is None
        or record.kind.field != "primitive"
        or record.kind.value.name != "Maybe"
        or len(record.kind.value.args) != 1
    ):
        raise ExprValueError(expr, "Maybe.none() or Maybe.some(value)")
    family: pb.PrimitiveSort = _definition_at(reference, "primitive_sort", "Maybe").read()
    if not isinstance(family, pb.PrimitiveSort) or family.arity != 1:
        msg = "Maybe inspection requires the canonical family"
        raise ValueError(msg)
    child: Ref | None
    if (
        node.kind is not None
        and node.kind.field == "primitive_value"
        and node.kind.value.value is not None
        and node.kind.value.value.field == "maybe"
    ):
        payload = node.kind.value.value.value
        child = reference.owner.ref("nodes", payload.value) if payload.has_field("value") else None
    else:
        raise ExprValueError(expr, "Maybe.none() or Maybe.some(value)")
    if child is None:
        return None
    child_node: pb.Node = child.read()
    if StructuralView(child.owner.ref("sorts", child_node.sort_id)) != StructuralView(
        sort.owner.ref("sorts", record.kind.value.args[0])
    ):
        msg = "Maybe child has the wrong canonical sort"
        raise ValueError(msg)
    return RuntimeExpr(child)


def _tuple_to_pair(value: tuple[object, ...]) -> RuntimeExpr:
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    if len(value) != 2:
        raise ValueError(f"Expected a tuple of length 2 for Pair conversion, got length {len(value)}")
    left, right = get_type_args()
    assert left.__egg_sort__ is not None
    assert right.__egg_sort__ is not None
    result = _BASE_CLASSES["Pair"](
        resolve_literal(left.__egg_sort__, value[0]), resolve_literal(right.__egg_sort__, value[1])
    )
    assert isinstance(result, RuntimeExpr)
    return result


@overload
def _container_items(expr: Vec[_T] | Set[_T] | MultiSet[_T]) -> tuple[_T, ...]: ...


@overload
def _container_items(expr: RuntimeExpr) -> tuple[RuntimeExpr, ...]: ...


def _container_items(expr: object) -> tuple[object, ...]:
    """Validate ordered collection children without evaluating or normalizing them."""
    assert isinstance(expr, RuntimeExpr)
    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    sort = reference.owner.ref("sorts", node.sort_id)
    record: pb.Sort = sort.read()
    if (
        record.kind is None
        or record.kind.field != "primitive"
        or record.kind.value.name not in {"Vec", "Set", "MultiSet"}
    ):
        raise ExprValueError(expr, "a Vec, Set or MultiSet value")
    family = record.kind.value.name
    allowed = "Vec(*args) or Vec.empty()" if family == "Vec" else f"{family}(*args)"
    if len(record.kind.value.args) != 1:
        msg = f"{family} inspection requires one sort argument"
        raise ValueError(msg)
    if (
        node.kind is not None
        and node.kind.field == "primitive_value"
        and node.kind.value.value is not None
        and node.kind.value.value.field == family.lower()
        and isinstance(node.kind.value.value.value, pb.ValueList)
    ):
        children = tuple(reference.owner.ref("nodes", index) for index in node.kind.value.value.value.items)
    else:
        raise ExprValueError(expr, allowed)
    element = sort.owner.ref("sorts", record.kind.value.args[0])
    for child in children:
        payload = child.read()
        if StructuralView(child.owner.ref("sorts", payload.sort_id)) != StructuralView(element):
            msg = f"{family} child has the wrong canonical sort"
            raise ValueError(msg)
    return tuple(RuntimeExpr(child) for child in children)


def _container_member(target: RuntimeClass | RuntimeExpr, name: str, kind: CallKind) -> RuntimeFunction:
    """Select and bind a supplied codec member without authoring a Call."""
    from .runtime import _definition_at  # noqa: PLC0415

    origin = target.__egg_definition__ if isinstance(target, RuntimeClass) else target.__egg_ref__
    assert isinstance(origin, Ref), "A container member requires a native family definition"
    reference = _definition_at(origin, "callable", name)
    views = presentation(reference).views
    view = next((index for index, candidate in enumerate(views) if candidate.kind == kind), None)
    if view is None:
        raise NotImplementedError(f"{name} has no supplied Python {kind.name} view")
    return RuntimeFunction(reference, view, target)


def _collection_to_expr(value: list[object] | tuple[object, ...] | set[object], family: str) -> RuntimeExpr:
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    (element,) = get_type_args()
    assert element.__egg_sort__ is not None
    target = _BASE_CLASSES[family][element]
    assert isinstance(target, RuntimeClass), "A concrete collection codec requires a concrete element sort"
    result = target(*(resolve_literal(element.__egg_sort__, item) for item in value))
    assert isinstance(result, RuntimeExpr)
    return result


def _map_entries(expr: RuntimeExpr) -> tuple[tuple[Ref, Ref], ...]:
    """Inspect immediate ordered Map children without traversing their graphs."""
    try:
        reference = expr.__egg_ref__
        node: pb.Node = reference.read()
        sort = reference.owner.ref("sorts", node.sort_id)
        record: pb.Sort = sort.read()
        if (
            record.kind is None
            or record.kind.field != "primitive"
            or record.kind.value.name != "Map"
            or len(record.kind.value.args) != 2
        ):
            msg = "Map inspection requires the canonical two-argument family"
            raise ExprValueError(expr, msg)
        arguments = [sort.owner.ref("sorts", index) for index in record.kind.value.args]
        if (
            node.kind is None
            or node.kind.field != "primitive_value"
            or node.kind.value.value is None
            or node.kind.value.value.field != "map"
        ):
            raise ExprValueError(expr, "a Map value payload")
        entries = []
        for entry in node.kind.value.value.value.entries:
            if not entry.has_field("key") or not entry.has_field("value"):
                msg = "Map entry requires key and value references"
                raise ExprValueError(expr, msg)
            entries.append((reference.owner.ref("nodes", entry.key), reference.owner.ref("nodes", entry.value)))
        for entry in entries:
            for child, expected in zip(entry, arguments, strict=True):
                payload = child.read()
                if StructuralView(child.owner.ref("sorts", payload.sort_id)) != StructuralView(expected):
                    msg = "Map entry has the wrong canonical sort"
                    raise ExprValueError(expr, msg)
        return tuple(entries)
    except (LookupError, ValueError) as error:
        raise ExprValueError(expr, "a Map value payload") from error


def _dict_to_map(value: dict[object, object]) -> RuntimeExpr:
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    key, item = get_type_args()
    assert key.__egg_sort__ is not None
    assert item.__egg_sort__ is not None
    target = _BASE_CLASSES["Map"][key, item]
    assert isinstance(target, RuntimeClass), "A concrete Map codec requires concrete key and value sorts"
    assert target.__egg_sort__ is not None
    builder = Builder()
    entries = []
    for source_key, source_value in value.items():
        converted_key = resolve_literal(key.__egg_sort__, source_key)
        converted_value = resolve_literal(item.__egg_sort__, source_value)
        entries.append(
            pb.MapEntry(
                key=builder.import_ref(converted_key.__egg_ref__), value=builder.import_ref(converted_value.__egg_ref__)
            )
        )
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(target.__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(value=Oneof[Literal["map"], pb.MapValue]("map", pb.MapValue(entries=entries))),
            ),
        ),
    )
    return RuntimeExpr(builder.publish().ref("nodes", root))


@overload
def _map_value(expr: Map[_K, _V]) -> dict[_K, _V]: ...


@overload
def _map_value(expr: RuntimeExpr) -> dict[RuntimeExpr, RuntimeExpr]: ...


def _map_value(expr: object) -> object:
    """Return typed symbolic pairs after the shared Map child-sort validation."""
    assert isinstance(expr, RuntimeExpr)
    return {RuntimeExpr(key): RuntimeExpr(value) for key, value in _map_entries(expr)}


def _provided_default(native_key: str, view_index: int, parameter_name: str) -> RuntimeExpr:
    """Borrow a validated authored default without evaluating its host payload."""
    from ._catalog import builtin_catalog  # noqa: PLC0415 -- import only when an authored default is borrowed

    reference = builtin_catalog().definitions["callable", native_key]
    view = presentation(reference).views[view_index]
    validate_view(reference, view)
    parameter = next(parameter for parameter in view.params if parameter.name == parameter_name)
    if parameter.default is None:
        raise ValueError(f"Missing authored default for {native_key}.{parameter_name}")
    validate_default(parameter.default)
    return RuntimeExpr(parameter.default)


def _function_to_expr(value: object) -> RuntimeExpr:
    """Specialize a callable codec at its actual structural target sort."""
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    assert isinstance(value, RuntimeFunction | FunctionType | MethodType | partial)
    target = _BASE_CLASSES["UnstableFn"][get_type_args()]
    assert isinstance(target, RuntimeClass), "A function codec requires concrete argument and result sorts"
    result = target(value.func, *value.args) if isinstance(value, partial) else target(value)
    assert isinstance(result, RuntimeExpr)
    return result


def _none_to_maybe(value: None) -> RuntimeExpr:
    """Resolve an absent optional value at the actual concrete element sort."""
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    (element,) = get_type_args()
    target = _BASE_CLASSES["Maybe"][element]
    assert isinstance(target, RuntimeClass), "A Maybe codec requires a concrete element sort"
    result = _container_member(target, "egglog.experimental.maybe.none", CallKind.CLASS_METHOD)()
    assert isinstance(result, RuntimeExpr)
    return result
