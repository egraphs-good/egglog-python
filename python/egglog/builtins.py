"""Catalog-generated builtin wrappers and handwritten Python value protocols."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import FunctionType, GenericAlias, MethodType
from typing import Any, Literal, Protocol, TypeVar, cast

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof
from typing_extensions import deprecated

from ._builtin_protocols import SCALAR_PROTOCOLS
from ._catalog import builtin_catalog
from ._presentation import CallKind, presentation
from ._program import Builder, Ref, StructuralView
from .conversion import converter, get_type_args, resolve_literal
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction, _Intrinsic


@dataclass
class ExprValueError(AttributeError):
    """A symbolic expression is not in a locally inspectable value form."""

    expr: object
    allowed: str

    def __str__(self) -> str:
        return f"Cannot get Python value of {self.expr}, must be of form {self.allowed}. Try calling `extract` on it to get the underlying value."


def _scalar_value(expr: RuntimeExpr) -> object:
    from .deconstruct import get_literal_value  # noqa: PLC0415

    value = get_literal_value(expr)
    if value is not None:
        return value
    node: pb.Node = expr.__egg_ref__.read()
    from .runtime import _class_for_sort  # noqa: PLC0415

    cls = _class_for_sort(expr.__egg_ref__.owner.ref("sorts", node.sort_id))
    raise ExprValueError(expr, str(cls))


@deprecated("use .value")
def _eval(expr: RuntimeExpr) -> object:
    return expr.value


def _integer(expr: RuntimeExpr) -> int:
    return int(cast("Any", expr.value))


def _float(expr: RuntimeExpr) -> float:
    return float(cast("Any", expr.value))


def _pair_value(expr: RuntimeExpr) -> tuple[RuntimeExpr, RuntimeExpr]:
    """Inspect owned children, never evaluate or manufacture a semantic Call."""
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


def _maybe_value(expr: RuntimeExpr) -> RuntimeExpr | None:
    """Inspect exact optional presence while leaving a present child symbolic."""
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


def _container_items(expr: RuntimeExpr) -> tuple[RuntimeExpr, ...]:
    """Validate ordered collection children without evaluating or normalizing them."""
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


def _function_value(expr: RuntimeExpr) -> object:
    from .deconstruct import get_literal_value  # noqa: PLC0415

    value = get_literal_value(expr)
    if value is None:
        raise ExprValueError(expr, "UnstableFn(f, *args)")
    return value


__all__ = ["ExprValueError"]
_catalog = builtin_catalog()
UnstableFn = RuntimeClass(_Intrinsic.APPLY)
UnstableFn.__egg_hooks__.update(value=property(_function_value), eval=_eval)
__all__ += ["UnstableFn"]
converter(RuntimeFunction, UnstableFn, lambda value: UnstableFn[get_type_args()](value))
converter(FunctionType, UnstableFn, lambda value: UnstableFn[get_type_args()](value))
converter(MethodType, UnstableFn, lambda value: UnstableFn[get_type_args()](value))
converter(partial, UnstableFn, lambda value: UnstableFn[get_type_args()](value.func, *value.args))
for _definition in _catalog.definitions.values():
    _declaration: (
        pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive | pb.PrimitiveSort
    ) = _definition.read()
    _kind = _declaration
    if not isinstance(_kind, pb.PrimitiveSort):
        continue
    _binding = presentation(_definition).type_view
    _path = tuple(_binding.path) if _binding is not None else (*__name__.split("."), _kind.name)
    if _path[:-1] != tuple(__name__.split(".")):
        continue
    _name = _path[-1]
    if not _name.isidentifier() or _name in globals():
        raise ValueError(f"Cannot generate Python builtin type {_path!r}")
    _class = RuntimeClass(_definition)
    globals()[_name] = _class
    __all__ += [_name]  # noqa: PLE0604 -- names come from the generated catalog
    _protocol = SCALAR_PROTOCOLS.get(_kind.name)
    if _kind.name == "Pair" and isinstance(_kind, pb.PrimitiveSort):
        _class.__egg_hooks__["value"] = property(_pair_value)
        converter(tuple, _class, _tuple_to_pair)
    if _kind.name == "Maybe" and isinstance(_kind, pb.PrimitiveSort):
        _class.__egg_hooks__["value"] = property(_maybe_value)
        converter(
            type(None),
            _class,
            lambda _value, cls=_class: _container_member(
                cls[get_type_args()[0]], "egglog.experimental.maybe.none", CallKind.CLASS_METHOD
            )(),
        )
    if _kind.name == "Vec" and isinstance(_kind, pb.PrimitiveSort):
        _class.__egg_hooks__.update(
            value=property(_container_items),
            __iter__=lambda expr: iter(expr.value),
            __len__=lambda expr: len(expr.value),
            __contains__=lambda expr, item: item in expr.value,
            eval=_eval,
        )
        converter(list, _class, lambda value: _collection_to_expr(value, "Vec"))
        converter(tuple, _class, lambda value: _collection_to_expr(value, "Vec"))
        _host_element = TypeVar("_host_element")
        globals()[f"{_name}Like"] = (
            GenericAlias(_class, _class.__egg_arguments__)
            | GenericAlias(tuple, (_host_element, ...))
            | GenericAlias(list, _host_element)
        )
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
    if _kind.name == "Set" and isinstance(_kind, pb.PrimitiveSort):
        _class.__egg_hooks__.update(
            value=property(lambda expr: set(_container_items(expr))),
            __iter__=lambda expr: iter(dict.fromkeys(_container_items(expr))),
            __len__=lambda expr: len(expr.value),
            __contains__=lambda expr, item: item in expr.value,
            eval=_eval,
        )
        converter(set, _class, lambda value: _collection_to_expr(value, "Set"))
        _host_set_element = TypeVar("_host_set_element")
        globals()[f"{_name}Like"] = GenericAlias(_class, _class.__egg_arguments__) | GenericAlias(
            set, _host_set_element
        )
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
    if _kind.name == "MultiSet" and isinstance(_kind, pb.PrimitiveSort):
        _class.__egg_hooks__.update(
            value=property(lambda expr: list(_container_items(expr))),
            __iter__=lambda expr: iter(expr.value),
            __len__=lambda expr: len(expr.value),
            __contains__=lambda expr, item: item in expr.value,
            eval=_eval,
        )
        converter(tuple, _class, lambda value: _collection_to_expr(value, "MultiSet"))
        _host_multiset_element = TypeVar("_host_multiset_element")
        globals()[f"{_name}Like"] = GenericAlias(_class, _class.__egg_arguments__) | GenericAlias(
            tuple, (_host_multiset_element, ...)
        )
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
    if _kind.name == "Map" and isinstance(_kind, pb.PrimitiveSort):
        _class.__egg_hooks__.update(
            value=property(lambda expr: {RuntimeExpr(key): RuntimeExpr(value) for key, value in _map_entries(expr)}),
            __iter__=lambda expr: iter(expr.value),
            __len__=lambda expr: len(expr.value),
            __contains__=lambda expr, key: key in expr.value,
            eval=_eval,
        )
        converter(dict, _class, _dict_to_map)
        _host_key, _host_value = TypeVar("_host_key"), TypeVar("_host_value")
        globals()[f"{_name}Like"] = GenericAlias(_class, _class.__egg_arguments__) | GenericAlias(
            dict, (_host_key, _host_value)
        )
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
    if _protocol is not None and _protocol.literal_types:
        _value_protocol = _scalar_value
        if _protocol.literal_arm == "custom":
            from .deconstruct import _pyobject_value

            _value_protocol = _pyobject_value
        _class.__egg_hooks__.update(value=property(_value_protocol), eval=_eval)
        _literal_types = _protocol.literal_types
        _like: Any = _class
        for _literal_type in _literal_types:
            _like = _like | _literal_type
            if _protocol.value_type is float:
                converter(_literal_type, _class, lambda value, target=_class: target(float(value)))
            else:
                converter(_literal_type, _class, _class)
        globals()[f"{_name}Like"] = _like
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
        for _hook, _result in _protocol.hooks:
            _class.__egg_hooks__[_hook] = {int: _integer, float: _float, bool: lambda expr: bool(expr.value)}[_result]
    elif _kind.name == "Unit":
        from .egraph import expr_fact

        _class.__egg_hooks__["__bool__"] = lambda expr: bool(expr_fact(expr))

# Symbolic i64 conversions remain native calls; host literals encode directly.
for _family, _protocol in SCALAR_PROTOCOLS.items():
    if _protocol.conversion_sort is None or ("primitive_sort", _family) not in _catalog.definitions:
        continue
    from .runtime import _BASE_CLASSES

    _class = _BASE_CLASSES[_family]
    _source = _BASE_CLASSES[_protocol.conversion_sort]
    if _protocol.value_type is int:
        converter(_source, _class, _class)
    else:
        converter(_source, _class, lambda value, target=_class: target(value, 1))
    _like = _class | _source
    for _literal_type in (*_protocol.literal_types, *SCALAR_PROTOCOLS[_protocol.conversion_sort].literal_types):
        _like = _like | _literal_type
    globals()[f"{_class.__name__}Like"] = _like

for _path, (_definition, _view) in _catalog.functions.items():
    if _path[:-1] == tuple(__name__.split(".")):
        if _path[-1] in globals():
            raise ValueError(f"Python builtin binding collision: {_path!r}")
        globals()[_path[-1]] = RuntimeFunction(_definition, _view)
        __all__ += [_path[-1]]  # noqa: PLE0604


if ("primitive_sort", "PyObject") in _catalog.definitions:

    class PyObjectFunction(Protocol):
        def __call__(self, *__args: PyObject) -> PyObject: ...  # noqa: F821 -- native-catalog class

    @deprecated("use PyObject(fn) directly")
    def py_eval_fn(fn: Callable) -> PyObjectFunction:
        """Compatibility adapter for stored Python callables; prefer PyObject(fn)."""
        return globals()["PyObject"](fn)

    __all__ += ["py_eval_fn"]  # The Protocol remains module-only, as in the original API.
