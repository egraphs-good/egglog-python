"""Inspect owned expression records locally, without evaluating expressions."""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

from egglog_proto.egglog.v1 import egglog_pb as pb

from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction, _class_for_sort, _definition_at

__all__ = [
    "get_callable_args",
    "get_callable_fn",
    "get_constant_name",
    "get_let_name",
    "get_literal_value",
    "get_var_name",
]


def get_literal_value(x: object) -> object:
    """Return a local literal payload, or None for symbolic expressions."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    node: pb.Node = x.__egg_ref__.read()
    if node.kind is None or node.kind.field != "primitive_value":
        return None
    value = node.kind.value.value
    if value is None:
        msg = "Primitive value has no payload"
        raise ValueError(msg)
    match value.field:
        case "i64" | "string" | "bool":
            return value.value
        case "f64_bits":
            return struct.unpack("!d", struct.pack("!Q", value.value))[0]
        case "unit":
            return None
        case _:
            # Containers have their own .value protocols, retaining symbolic
            # children. They are not scalar literals.
            return None


def get_constant_name(x: object) -> object:
    """Inspect a retained constant name (constant presentation is not migrated)."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    msg = "Constant presentation is not migrated"
    raise NotImplementedError(msg)


def get_let_name(x: object) -> str | None:
    """Inspect a retained local name (the local-let policy is not migrated)."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    msg = "Retained local-let presentation is not migrated"
    raise NotImplementedError(msg)


def get_var_name(x: object) -> str | None:
    """Return a variable node's canonical name, or None for other node kinds."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    node: pb.Node = x.__egg_ref__.read()
    return node.kind.value if node.kind is not None and node.kind.field == "var" else None


def _deconstruct_call(x: RuntimeExpr) -> tuple[RuntimeClass | RuntimeFunction, tuple[RuntimeExpr, ...]] | None:
    """Select a declaration's Python view and restore its argument ordering."""
    node: pb.Node = x.__egg_ref__.read()
    if node.kind is None or node.kind.field != "call":
        return None
    call = node.kind.value
    declaration = _definition_at(x.__egg_ref__, "callable", call.func)
    record: pb.Declaration = declaration.read()
    arguments = tuple(RuntimeExpr(x.__egg_ref__.owner.ref("nodes", index)) for index in call.args)
    if record.bindings is None or record.bindings.python is None or not record.bindings.python.views:
        return RuntimeFunction(declaration), arguments
    view = record.bindings.python.views[0]
    ordered: list[RuntimeExpr] = []
    if view.has_field("receiver"):
        ordered.append(arguments[view.receiver])
    kind = record.kind.value if record.kind is not None else None
    fixed = (
        len(kind.typing.value.inputs)
        if isinstance(kind, pb.HostPrimitive) and kind.typing is not None and kind.typing.field == "signature"
        else len(arguments)
    )
    for parameter in view.params:
        ordered.extend(
            arguments[parameter.core_input :] if parameter.core_input == fixed else (arguments[parameter.core_input],)
        )
    if view.kind == pb.PythonCallKind.INITIALIZER:
        return _class_for_sort(x.__egg_ref__.owner.ref("sorts", node.sort_id)), tuple(ordered)
    owner = None
    if view.owner is not None and view.owner.kind is not None and view.owner.kind.field == "sort":
        if view.has_field("receiver"):
            receiver = arguments[view.receiver].__egg_ref__
            owner = _class_for_sort(receiver.owner.ref("sorts", receiver.read().sort_id))
        else:
            owner = _class_for_sort(declaration.owner.ref("sorts", view.owner.kind.value))
    return RuntimeFunction(declaration, 0, owner), tuple(ordered)


def get_callable_fn(x: object) -> Callable[..., Any] | None:
    """Get the generated callable responsible for a call expression."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    result = _deconstruct_call(x)
    return None if result is None else result[0]


def get_callable_args(x: object, fn: object = None) -> tuple[RuntimeExpr, ...] | None:
    """Return symbolic children in Python argument order, without evaluation."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    result = _deconstruct_call(x)
    if result is None:
        return None
    actual, arguments = result
    if fn is None:
        return arguments
    if isinstance(actual, RuntimeClass) and isinstance(fn, RuntimeClass):
        return arguments if actual.__egg_definition__ == fn.__egg_definition__ else None
    if isinstance(actual, RuntimeFunction) and isinstance(fn, RuntimeFunction):
        return arguments if actual.__egg_ref__ == fn.__egg_ref__ else None
    return None
