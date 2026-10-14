"""Inspect owned expression records locally, without evaluating expressions."""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from functools import partial
from math import gcd
from typing import Any

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._builtin_protocols import SCALAR_PROTOCOLS
from .declarations import Ident
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction, _class_for_sort, _definition_at

__all__ = [
    "get_callable_args",
    "get_callable_fn",
    "get_constant_name",
    "get_let_name",
    "get_literal_value",
    "get_var_name",
]


def _exact_literal_value(sort: pb.Sort, primitive: pb.PrimitiveValue) -> int | Fraction:
    """Validate canonical exact payload/sort agreement before host decoding."""
    value = primitive.value
    if (
        value is None
        or sort.kind is None
        or sort.kind.field != "family"
        or sort.kind.value.args
        or (protocol := SCALAR_PROTOCOLS.get(sort.kind.value.name)) is None
        or protocol.literal_arm != value.field
    ):
        msg = "Exact value payload has the wrong canonical sort"
        raise ValueError(msg)
    if value.field == "big_int":
        if not re.fullmatch(r"0|-?[1-9][0-9]*", value.value):
            msg = "BigInt value is not canonical decimal"
            raise ValueError(msg)
        # Decimal's exact string construction/int conversion does not inherit
        # int(str)'s process-wide decimal digit cap or apply context rounding.
        return int(Decimal(value.value))
    if value.field == "big_rat":
        if not re.fullmatch(r"0|-?[1-9][0-9]*", value.value.numerator) or not re.fullmatch(
            r"[1-9][0-9]*", value.value.denominator
        ):
            msg = "BigRat value is not canonical decimal"
            raise ValueError(msg)
        numerator, denominator = int(Decimal(value.value.numerator)), int(Decimal(value.value.denominator))
    elif value.field == "rational":
        numerator, denominator = value.value.numerator, value.value.denominator
        if not (-(2**63) <= numerator < 2**63 and 0 < denominator < 2**63):
            msg = "Rational value requires an i64 numerator and positive i64 denominator"
            raise ValueError(msg)
    else:
        msg = "Exact value payload has the wrong canonical sort"
        raise ValueError(msg)
    if denominator <= 0 or gcd(numerator, denominator) != 1:
        msg = "Rational value must be reduced with a positive denominator"
        raise ValueError(msg)
    return Fraction(numerator, denominator)


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
    sort: pb.Sort = x.__egg_ref__.owner.ref("sorts", node.sort_id).read()
    exact_sort = (
        sort.kind is not None
        and sort.kind.field == "family"
        and sort.kind.value.name in {"BigInt", "BigRat", "Rational"}
    )
    if exact_sort or value.field in {"big_int", "big_rat", "rational"}:
        return _exact_literal_value(sort, node.kind.value)
    match value.field:
        case "i64" | "string" | "bool":
            return value.value
        case "f64_bits":
            return struct.unpack("!d", struct.pack("!Q", value.value))[0]
        case "partial_call":
            from .runtime import _partial_function, create_callable  # noqa: PLC0415

            reference = x.__egg_ref__
            declaration = _definition_at(reference, "callable", value.value.func)
            record: pb.Declaration = declaration.read()
            view_index = (
                0
                if record.bindings is not None and record.bindings.python is not None and record.bindings.python.views
                else None
            )
            captures = tuple(RuntimeExpr(reference.owner.ref("nodes", index)) for index in value.value.args)
            _, children, owner_sort = _partial_function(
                RuntimeFunction(declaration, view_index),
                captures,
                reference.owner.ref("sorts", node.sort_id),
                convert_captures=False,
            )
            function = create_callable(declaration, owner_sort=owner_sort)
            if not isinstance(function, RuntimeFunction | RuntimeClass):
                msg = "Function value has no callable Python presentation"
                raise TypeError(msg)
            return partial(function, *children) if children else function
        case _:
            # Containers have their own .value protocols, retaining symbolic
            # children. They are not scalar literals.
            return None


def get_constant_name(x: object) -> Ident | None:
    """Return the supplied constant presentation name without evaluating it."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    node: pb.Node = x.__egg_ref__.read()
    if node.kind is None or node.kind.field != "call":
        return None
    declaration: pb.Declaration = _definition_at(x.__egg_ref__, "callable", node.kind.value.func).read()
    if declaration.bindings is None or declaration.bindings.python is None:
        return None
    views = declaration.bindings.python.views
    if views and views[0].kind == pb.PythonCallKind.CONSTANT:
        if node.kind.value.args:
            msg = "A constant call must be nullary"
            raise ValueError(msg)
        return Ident(views[0].path[-1], ".".join(views[0].path[:-1]) or None)
    return None


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


def _deconstruct_call(x: RuntimeExpr) -> tuple[RuntimeClass | RuntimeFunction, tuple[RuntimeExpr, ...]] | None:  # noqa: C901, PLR0912
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
    if any(
        view.has_field("tail_repetitions") and view.kind not in {pb.PythonCallKind.FUNCTION, pb.PythonCallKind.METHOD}
        for view in record.bindings.python.views
    ):
        msg = "Tail specialization supports only function and method views"
        raise ValueError(msg)
    kind = record.kind.value if record.kind is not None else None
    signature = (
        kind.typing.value
        if isinstance(kind, pb.HostPrimitive) and kind.typing is not None and kind.typing.field == "signature"
        else None
    )
    fixed = len(signature.inputs) if signature is not None else len(arguments)
    tail_width = len(signature.varargs) if signature is not None else 0
    application = isinstance(kind, pb.HostPrimitive) and kind.typing is not None and kind.typing.field == "application"
    if application:
        from .type_constraint_solver import infer_sort  # noqa: PLC0415

        if not arguments:
            msg = "Function application has no function input"
            raise ValueError(msg)
        function = arguments[0].__egg_ref__
        function_sort = function.owner.ref("sorts", function.read().sort_id)
        function_type: pb.Sort = function_sort.read()
        if (
            function_type.kind is None
            or function_type.kind.field != "func"
            or len(arguments) != len(function_type.kind.value.params) + 1
        ):
            msg = "Function application has invalid argument sorts or arity"
            raise ValueError(msg)
        infer_sort(
            function_sort.owner.ref("sorts", function_type.kind.value.result),
            x.__egg_ref__.owner.ref("sorts", node.sort_id),
            0,
            {},
        )
        for argument, expected in zip(arguments[1:], function_type.kind.value.params, strict=True):
            child = argument.__egg_ref__
            infer_sort(
                function_sort.owner.ref("sorts", expected), child.owner.ref("sorts", child.read().sort_id), 0, {}
            )
        fixed, tail_width = 1, 1
    if (
        len(arguments) < fixed
        or (len(arguments) - fixed) % (tail_width or 1)
        or (not tail_width and len(arguments) != fixed)
    ):
        msg = "Call has invalid canonical argument count"
        raise ValueError(msg)
    selected = next(
        (
            (index, view)
            for index, view in enumerate(record.bindings.python.views)
            if not view.has_field("tail_repetitions")
            or (tail_width and len(arguments) == fixed + tail_width * view.tail_repetitions)
        ),
        None,
    )
    if selected is None:
        return RuntimeFunction(declaration), arguments
    view_index, view = selected
    ordered: list[RuntimeExpr] = []
    if view.has_field("receiver"):
        ordered.append(arguments[view.receiver])
    for parameter in view.params:
        ordered.extend(
            arguments[parameter.core_input :]
            if tail_width and not view.has_field("tail_repetitions") and parameter.core_input == fixed
            else (arguments[parameter.core_input],)
        )
    if view.kind == pb.PythonCallKind.INITIALIZER:
        return _class_for_sort(x.__egg_ref__.owner.ref("sorts", node.sort_id)), tuple(ordered)
    owner = None
    if view.owner is not None and view.owner.kind is not None:
        if view.has_field("receiver"):
            receiver = arguments[view.receiver].__egg_ref__
            owner = _class_for_sort(receiver.owner.ref("sorts", receiver.read().sort_id))
        elif view.owner.kind.field == "sort":
            owner = _class_for_sort(declaration.owner.ref("sorts", view.owner.kind.value))
    return RuntimeFunction(declaration, view_index, owner), tuple(ordered)


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
        return arguments if actual.__egg_ref__ == fn.__egg_ref__ or actual._identity() == fn._identity() else None
    return None
