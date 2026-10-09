"""Catalog-generated builtin wrappers and handwritten Python value protocols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import GenericAlias
from typing import Any, TypeVar, cast

from egglog_proto.egglog.v1 import egglog_pb as pb
from typing_extensions import deprecated

from ._builtin_protocols import SCALAR_PROTOCOLS
from ._catalog import builtin_catalog
from ._program import Ref, StructuralView, _copy_record
from .conversion import converter, get_type_args, resolve_literal
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction


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


def _exact_value(expr: RuntimeExpr) -> int | Fraction:  # noqa: C901
    """Inspect inert exact values or the legacy constructor value forms locally."""
    from .deconstruct import get_literal_value  # noqa: PLC0415
    from .runtime import _definition_at  # noqa: PLC0415
    from .type_constraint_solver import infer_sort  # noqa: PLC0415

    value = get_literal_value(expr)
    if value is not None:
        return cast("int | Fraction", value)
    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    sort = reference.owner.ref("sorts", node.sort_id)
    family: pb.Sort = sort.read()
    assert family.kind is not None
    allowed = "a literal value or a locally inspectable exact constructor"
    if node.kind is None or node.kind.field != "call" or family.kind.field != "family":
        raise ExprValueError(expr, allowed)
    call = node.kind.value
    if (family.kind.value.name, call.func) not in {
        ("Rational", "egglog.experimental.rational.make"),
        ("BigInt", "egglog.core.bigint.from_string"),
        ("BigRat", "egglog.core.bigrat.make"),
    }:
        raise ExprValueError(expr, allowed)
    declaration = _definition_at(reference, "callable", call.func)
    record: pb.Declaration = declaration.read()
    if record.kind is None or record.kind.field != "host_primitive":
        raise ExprValueError(expr, allowed)
    primitive = record.kind.value
    if primitive.typing is None or primitive.typing.field != "signature":
        raise ExprValueError(expr, allowed)
    signature = primitive.typing.value
    if signature.type_params or signature.varargs or not signature.has_field("output"):
        msg = "Exact constructor inspection requires a closed fixed signature"
        raise ValueError(msg)
    infer_sort(declaration.owner.ref("sorts", signature.output), sort, 0, {})
    if len(call.args) != len(signature.inputs):
        msg = "Exact constructor has invalid canonical argument count"
        raise ValueError(msg)
    children = [reference.owner.ref("nodes", index) for index in call.args]
    for child, parameter in zip(children, signature.inputs, strict=True):
        infer_sort(
            declaration.owner.ref("sorts", parameter.sort), child.owner.ref("sorts", child.read().sort_id), 0, {}
        )
    if family.kind.value.name == "BigInt":
        (text,) = children
        text_value = RuntimeExpr(text).value
        if not isinstance(text_value, str):
            raise ExprValueError(expr, allowed)
        # Inert BigInts print as these existing symbolic Calls. Inspect their
        # canonical decimals exactly at arbitrary magnitude; keep int's legacy
        # handling/errors for every other spelling (not Decimal's exponent or
        # fractional syntax).
        return int(Decimal(text_value)) if re.fullmatch(r"0|-?[1-9][0-9]*", text_value) else int(text_value)
    numerator, denominator = children
    numerator_value, denominator_value = RuntimeExpr(numerator).value, RuntimeExpr(denominator).value
    if not isinstance(numerator_value, int) or not isinstance(denominator_value, int):
        raise ExprValueError(expr, allowed)
    return Fraction(numerator_value, denominator_value)


@deprecated("use .value")
def _eval(expr: RuntimeExpr) -> object:
    return expr.value


def _integer(expr: RuntimeExpr) -> int:
    return int(cast("Any", expr.value))


def _float(expr: RuntimeExpr) -> float:
    return float(cast("Any", expr.value))


def _pair_call_children(reference: Ref, sort: Ref) -> tuple[Ref, Ref]:
    """Validate codec provenance and the actual canonical constructor scheme."""
    from .runtime import _definition_at  # noqa: PLC0415
    from .type_constraint_solver import infer_sort  # noqa: PLC0415

    node: pb.Node = reference.read()
    assert node.kind is not None
    assert node.kind.field == "call"
    declaration = _definition_at(reference, "callable", node.kind.value.func)
    callable_record: pb.Declaration = declaration.read()
    if callable_record.kind is None or callable_record.kind.field != "host_primitive":
        raise ExprValueError(RuntimeExpr(reference), "Pair(left, right)")
    primitive = callable_record.kind.value
    if primitive.typing is None or primitive.typing.field != "signature":
        raise ExprValueError(RuntimeExpr(reference), "Pair(left, right)")
    signature = primitive.typing.value
    if signature.varargs or len(signature.inputs) != 2 or not signature.has_field("output"):
        msg = "Invalid canonical Pair constructor signature"
        raise ValueError(msg)
    if len(node.kind.value.args) != 2:
        msg = "Pair inspection requires two children"
        raise ValueError(msg)
    substitutions: dict[int, Ref] = {}
    infer_sort(declaration.owner.ref("sorts", signature.output), sort, len(signature.type_params), substitutions)
    children = tuple(reference.owner.ref("nodes", index) for index in node.kind.value.args)
    for child, parameter in zip(children, signature.inputs, strict=True):
        payload: pb.Node = child.read()
        infer_sort(
            declaration.owner.ref("sorts", parameter.sort),
            child.owner.ref("sorts", payload.sort_id),
            len(signature.type_params),
            substitutions,
        )
    if len(substitutions) != len(signature.type_params):
        msg = "Pair inspection has unresolved signature parameters"
        raise ValueError(msg)
    return children[0], children[1]


def _pair_value(expr: RuntimeExpr) -> tuple[RuntimeExpr, RuntimeExpr]:
    """Inspect owned children, never evaluate or manufacture a semantic Call."""
    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    sort = reference.owner.ref("sorts", node.sort_id)
    record: pb.Sort = sort.read()
    if record.kind is None or record.kind.field != "family" or record.kind.value.name != "Pair":
        raise ExprValueError(expr, "Pair(left, right)")
    arguments = record.kind.value.args
    if len(arguments) != 2:
        msg = "Pair inspection requires two sort arguments"
        raise ValueError(msg)
    if node.kind is not None and node.kind.field == "call" and node.kind.value.func == "egglog.core.pair.make":
        children = _pair_call_children(reference, sort)
    elif (
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


def _container_items(expr: RuntimeExpr) -> tuple[RuntimeExpr, ...]:  # noqa: C901
    """Validate ordered Vec/Set children without evaluating or normalizing them."""
    from .runtime import _definition_at  # noqa: PLC0415
    from .type_constraint_solver import infer_sort  # noqa: PLC0415

    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    sort = reference.owner.ref("sorts", node.sort_id)
    record: pb.Sort = sort.read()
    if record.kind is None or record.kind.field != "family" or record.kind.value.name not in {"Vec", "Set"}:
        raise ExprValueError(expr, "a Vec or Set value")
    family = record.kind.value.name
    allowed = "Vec(*args) or Vec.empty()" if family == "Vec" else "Set(*args)"
    if len(record.kind.value.args) != 1:
        msg = f"{family} inspection requires one sort argument"
        raise ValueError(msg)
    constructor = f"egglog.core.{family.lower()}.of"
    if (
        node.kind is not None
        and node.kind.field == "call"
        and (
            node.kind.value.func == constructor or (family == "Vec" and node.kind.value.func == "egglog.core.vec.empty")
        )
    ):
        call = node.kind.value
        declaration = _definition_at(reference, "callable", call.func)
        callable_record: pb.Declaration = declaration.read()
        if callable_record.kind is None or callable_record.kind.field != "host_primitive":
            raise ExprValueError(expr, allowed)
        primitive = callable_record.kind.value
        if primitive.typing is None or primitive.typing.field != "signature":
            raise ExprValueError(expr, allowed)
        signature = primitive.typing.value
        variadic = call.func == constructor
        if (
            signature.inputs
            or len(signature.varargs) != int(variadic)
            or not signature.has_field("output")
            or (not variadic and call.args)
        ):
            msg = f"Invalid canonical {family} constructor signature or arguments"
            raise ValueError(msg)
        substitutions: dict[int, Ref] = {}
        infer_sort(declaration.owner.ref("sorts", signature.output), sort, len(signature.type_params), substitutions)
        children = tuple(reference.owner.ref("nodes", index) for index in call.args)
        for child in children:
            payload: pb.Node = child.read()
            infer_sort(
                declaration.owner.ref("sorts", signature.varargs[0].sort),
                child.owner.ref("sorts", payload.sort_id),
                len(signature.type_params),
                substitutions,
            )
        if len(substitutions) != len(signature.type_params):
            msg = f"{family} inspection has unresolved signature parameters"
            raise ValueError(msg)
    elif (
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


def _container_member(target: RuntimeClass | RuntimeExpr, name: str, kind: pb.PythonCallKind) -> RuntimeFunction:
    """Select and bind a supplied codec member without authoring a Call."""
    from .runtime import _definition_at  # noqa: PLC0415

    origin = target.__egg_definition__ if isinstance(target, RuntimeClass) else target.__egg_ref__
    reference = _definition_at(origin, "callable", name)
    declaration: pb.Declaration = reference.read()
    views = (
        declaration.bindings.python.views
        if declaration.bindings is not None and declaration.bindings.python is not None
        else []
    )
    view = next((index for index, candidate in enumerate(views) if candidate.kind == kind), None)
    if view is None:
        raise NotImplementedError(f"{name} has no supplied Python {kind.name} view")
    return RuntimeFunction(reference, view, target)


def _collection_to_expr(value: list[object] | tuple[object, ...] | set[object], family: str) -> RuntimeExpr:
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    (element,) = get_type_args()
    assert element.__egg_sort__ is not None
    target = _BASE_CLASSES[family][element]
    result = (
        target(*(resolve_literal(element.__egg_sort__, item) for item in value))
        if value
        else _container_member(target, f"egglog.core.{family.lower()}.empty", pb.PythonCallKind.CLASS_METHOD)()
    )
    assert isinstance(result, RuntimeExpr)
    return result


def _map_entries(expr: RuntimeExpr) -> tuple[tuple[Ref, Ref], ...]:  # noqa: C901, PLR0912
    """Inspect finite Map records in order, without normalizing or making Calls."""
    from .runtime import _definition_at  # noqa: PLC0415
    from .type_constraint_solver import TypeConstraintError, infer_sort  # noqa: PLC0415

    try:
        reference = expr.__egg_ref__
        # Equality treats Unions as opaque identities; validation must still
        # visit their children. Reuse the canonical field-role visitor without
        # storing another expression model or changing comparison semantics.
        pending = [(reference, False)]
        active: set[Ref] = set()
        complete: set[Ref] = set()
        while pending:
            current, leaving = pending.pop()
            if current in complete:
                continue
            if leaving:
                active.remove(current)
                complete.add(current)
                continue
            if current in active:
                raise ExprValueError(expr, "an acyclic Map value graph")
            active.add(current)
            children: list[Ref] = []

            def visit(role: str, index: int, origin=current.owner, found=children) -> int:
                found.append(origin.ref(role, index))
                return index

            _copy_record(current.read(), relocate=visit)
            pending.append((current, True))
            pending.extend((child, False) for child in children)
        node: pb.Node = reference.read()
        sort = reference.owner.ref("sorts", node.sort_id)
        record: pb.Sort = sort.read()
        if (
            record.kind is None
            or record.kind.field != "family"
            or record.kind.value.name != "Map"
            or len(record.kind.value.args) != 2
        ):
            msg = "Map inspection requires the canonical two-argument family"
            raise ExprValueError(expr, msg)
        arguments = [sort.owner.ref("sorts", index) for index in record.kind.value.args]
        inserted: list[tuple[Ref, Ref]] = []
        entries: list[tuple[Ref, Ref]] = []
        while True:
            node = reference.read()
            if StructuralView(reference.owner.ref("sorts", node.sort_id)) != StructuralView(sort):
                msg = "Map insert base has the wrong sort"
                raise ExprValueError(expr, msg)
            if (
                node.kind is not None
                and node.kind.field == "primitive_value"
                and node.kind.value.value is not None
                and node.kind.value.value.field == "map"
            ):
                for entry in node.kind.value.value.value.entries:
                    if not entry.has_field("key") or not entry.has_field("value"):
                        msg = "Map entry requires key and value references"
                        raise ExprValueError(expr, msg)
                    entries.append((reference.owner.ref("nodes", entry.key), reference.owner.ref("nodes", entry.value)))
                break
            if (
                node.kind is None
                or node.kind.field != "call"
                or node.kind.value.func not in {"egglog.core.map.empty", "egglog.core.map.of", "egglog.core.map.insert"}
            ):
                raise ExprValueError(expr, "Map.empty(), Map.insert() or Map.from_pairs(finite entries)")
            call = node.kind.value
            declaration = _definition_at(reference, "callable", call.func)
            declared: pb.Declaration = declaration.read()
            if (
                declared.kind is None
                or declared.kind.field != "host_primitive"
                or declared.kind.value.typing is None
                or declared.kind.value.typing.field != "signature"
            ):
                msg = "Map inspection requires native callable provenance"
                raise ExprValueError(expr, msg)
            signature = declared.kind.value.typing.value
            if signature.varargs or not signature.has_field("output") or len(call.args) != len(signature.inputs):
                msg = "Invalid canonical Map signature or call arity"
                raise ExprValueError(expr, msg)
            substitutions: dict[int, Ref] = {}
            infer_sort(
                declaration.owner.ref("sorts", signature.output), sort, len(signature.type_params), substitutions
            )
            children = [reference.owner.ref("nodes", index) for index in call.args]
            for child, parameter in zip(children, signature.inputs, strict=True):
                payload: pb.Node = child.read()
                infer_sort(
                    declaration.owner.ref("sorts", parameter.sort),
                    child.owner.ref("sorts", payload.sort_id),
                    len(signature.type_params),
                    substitutions,
                )
            if len(substitutions) != len(signature.type_params):
                msg = "Map inspection has unresolved signature parameters"
                raise ExprValueError(expr, msg)
            if call.func == "egglog.core.map.empty":
                if children:
                    msg = "Map empty must have no children"
                    raise ExprValueError(expr, msg)
                break
            if call.func == "egglog.core.map.of":
                (vector,) = children
                vector_sort: pb.Sort = vector.owner.ref("sorts", vector.read().sort_id).read()
                if (
                    vector_sort.kind is None
                    or vector_sort.kind.field != "family"
                    or vector_sort.kind.value.name != "Vec"
                ):
                    raise ExprValueError(expr, "Map.from_pairs(Vec(*pairs))")
                for pair in _container_items(RuntimeExpr(vector)):
                    left, right = _pair_value(pair)
                    entries.append((left.__egg_ref__, right.__egg_ref__))
                break
            reference, key, value = children
            inserted.append((key, value))
        entries.extend(reversed(inserted))
        for entry in entries:
            for child, expected in zip(entry, arguments, strict=True):
                payload = child.read()
                if StructuralView(child.owner.ref("sorts", payload.sort_id)) != StructuralView(expected):
                    msg = "Map entry has the wrong canonical sort"
                    raise ExprValueError(expr, msg)
        return tuple(entries)
    except (LookupError, ValueError, TypeConstraintError) as error:
        raise ExprValueError(expr, "a finite Map.empty(), Map.insert(), Map.from_pairs(...) or Map value") from error


def _dict_to_map(value: dict[object, object]) -> RuntimeExpr:
    from .runtime import _BASE_CLASSES  # noqa: PLC0415

    key, item = get_type_args()
    assert key.__egg_sort__ is not None
    assert item.__egg_sort__ is not None
    target = _BASE_CLASSES["Map"][key, item]
    result = _container_member(target, "egglog.core.map.empty", pb.PythonCallKind.CLASS_METHOD)()
    assert isinstance(result, RuntimeExpr)
    for source_key, source_value in value.items():
        insert = _container_member(result, "egglog.core.map.insert", pb.PythonCallKind.METHOD)
        metadata: pb.Declaration = insert.__egg_ref__.read()
        assert metadata.bindings is not None
        assert metadata.bindings.python is not None
        assert insert.__egg_view__ is not None
        view = metadata.bindings.python.views[insert.__egg_view__]
        arguments = {
            1: resolve_literal(key.__egg_sort__, source_key),
            2: resolve_literal(item.__egg_sort__, source_value),
        }
        result = insert(**{parameter.name: arguments[parameter.core_input] for parameter in view.params})
        assert isinstance(result, RuntimeExpr)
    return result


__all__ = ["ExprValueError"]
_catalog = builtin_catalog()
for _definition in _catalog.definitions.values():
    _declaration: pb.Declaration = _definition.read()
    _kind = _declaration.kind
    if _kind is None or not isinstance(_kind.value, pb.HostSortFamily | pb.EqSort):
        continue
    _binding = _kind.value.bindings
    _path = (
        tuple(_binding.python.path)
        if _binding is not None and _binding.python is not None
        else (*__name__.split("."), _kind.value.name)
    )
    if _path[:-1] != tuple(__name__.split(".")):
        continue
    _name = _path[-1]
    if not _name.isidentifier() or _name in globals():
        raise ValueError(f"Cannot generate Python builtin type {_path!r}")
    _class = RuntimeClass(_definition)
    globals()[_name] = _class
    __all__ += [_name]  # noqa: PLE0604 -- names come from the generated catalog
    _protocol = SCALAR_PROTOCOLS.get(_kind.value.name)
    if _kind.value.name == "Pair" and isinstance(_kind.value, pb.HostSortFamily):
        _class.__egg_hooks__["value"] = property(_pair_value)
        converter(tuple, _class, _tuple_to_pair)
    if _kind.value.name == "Vec" and isinstance(_kind.value, pb.HostSortFamily):
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
    if _kind.value.name == "Set" and isinstance(_kind.value, pb.HostSortFamily):
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
    if _kind.value.name == "Map" and isinstance(_kind.value, pb.HostSortFamily):
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
    if _protocol is not None and (_protocol.literal_types or not _protocol.literal_constructor):
        _class.__egg_hooks__.update(
            value=property(_scalar_value if _protocol.literal_constructor else _exact_value), eval=_eval
        )
        _literal_types = _protocol.literal_types
        _like: Any = _class
        for _literal_type in _literal_types:
            _like = _like | _literal_type
            if not _protocol.literal_constructor:
                converter(
                    _literal_type, _class, lambda value, target=_class: target(value.numerator, value.denominator)
                )
            elif _protocol.value_type is float:
                converter(_literal_type, _class, lambda value, target=_class: target(float(value)))
            else:
                converter(_literal_type, _class, _class)
        globals()[f"{_name}Like"] = _like
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
        for _hook, _result in _protocol.hooks:
            _class.__egg_hooks__[_hook] = {int: _integer, float: _float, bool: lambda expr: bool(expr.value)}[_result]
    elif _kind.value.name == "Unit":
        from .egraph import expr_fact

        _class.__egg_hooks__["__bool__"] = lambda expr: bool(expr_fact(expr))

# Expression conversion policy is separate from formed-value decoding. In
# particular, Python ints/Fractions still pass through bounded i64 Calls.
for _family, _protocol in SCALAR_PROTOCOLS.items():
    if _protocol.conversion_sort is None or ("host_sort_family", _family) not in _catalog.definitions:
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
