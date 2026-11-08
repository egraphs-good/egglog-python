"""Inspect owned expression records locally, without evaluating expressions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, TypeVar, Unpack, cast, overload

from egglog_proto.egglog.v1 import egglog_pb as pb
from typing_extensions import TypeVarTuple

from . import bindings
from ._builtin_protocols import SCALAR_PROTOCOLS
from ._presentation import CallKind, presentation
from ._program import Ref, StructuralView
from .declarations import Ident
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction, _class_for_sort, _definition_at
from .type_constraint_solver import infer_call_bindings, infer_sort, substitute_sort

if TYPE_CHECKING:
    from fractions import Fraction

    from .builtins import BigInt, BigRat, Bool, PyObject, Rational, String, UnstableFn, f64, i64
    from .egraph import BaseExpr, Expr

T = TypeVar("T", bound="BaseExpr")
TS = TypeVarTuple("TS", default=Unpack[tuple["BaseExpr", ...]])

__all__ = [
    "get_callable_args",
    "get_callable_fn",
    "get_constant_name",
    "get_let_name",
    "get_literal_value",
    "get_var_name",
]


def _pyobject_value(expr: RuntimeExpr) -> object:
    """Materialize a trusted PyObject literal after validating its owned shape."""
    import cloudpickle  # noqa: PLC0415 -- explicit host materialization only

    from ._catalog import builtin_catalog  # noqa: PLC0415
    from .builtins import ExprValueError  # noqa: PLC0415

    reference = expr.__egg_ref__
    node: pb.Node = reference.read()
    if node.kind is None or node.kind.field != "primitive_value":
        raise ExprValueError(expr, "a PyObject literal")
    value = node.kind.value.value
    sort: pb.Sort = reference.owner.ref("sorts", node.sort_id).read()
    if (
        sort.kind is None
        or sort.kind.field != "primitive"
        or sort.kind.value.name != "PyObject"
        or sort.kind.value.args
        or value is None
        or value.field != "custom"
        or value.value.args
        or StructuralView(_definition_at(reference, "primitive_sort", "PyObject"))
        != StructuralView(builtin_catalog().definitions["primitive_sort", "PyObject"])
    ):
        msg = "PyObject value requires its canonical nullary family and childless Custom payload"
        raise ValueError(msg)
    return cloudpickle.loads(value.value.payload)


@overload
def get_literal_value(x: String) -> str | None: ...


@overload
def get_literal_value(x: Bool) -> bool | None: ...


@overload
def get_literal_value(x: i64 | BigInt) -> int | None: ...


@overload
def get_literal_value(x: f64) -> float | None: ...


@overload
def get_literal_value(x: BigRat | Rational) -> Fraction | None: ...


@overload
def get_literal_value(x: PyObject) -> object: ...


@overload
def get_literal_value(x: UnstableFn[T, *TS]) -> Callable[[Unpack[TS]], T] | None: ...


@overload
def get_literal_value(x: Expr) -> None: ...


@overload
def get_literal_value(x: object) -> object: ...


def get_literal_value(x: object) -> object:  # noqa: C901, PLR0912 -- payload variants retain their host protocols
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
    if value.field == "custom" or (
        sort.kind is not None and sort.kind.field == "primitive" and sort.kind.value.name == "PyObject"
    ):
        return _pyobject_value(x)
    if value.field in (protocol.literal_arm for protocol in SCALAR_PROTOCOLS.values()):
        if sort.kind is None or sort.kind.field != "primitive" or sort.kind.value.args:
            msg = "Scalar value requires a nullary primitive sort"
            raise ValueError(msg)
        return bindings._decode_scalar(sort.kind.value.name, node.kind.value.to_binary())
    match value.field:
        case "lambda":
            if sort.kind is None or sort.kind.field != "func":
                msg = "Lambda value requires a Func sort"
                raise TypeError(msg)
            return x
        case "partial_call":
            from .runtime import _partial_function, create_callable  # noqa: PLC0415

            reference = x.__egg_ref__
            function_sort = reference.owner.ref("sorts", node.sort_id)
            declaration = _definition_at(reference, "callable", value.value.func)
            record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = (
                declaration.read()
            )
            captures = tuple(RuntimeExpr(reference.owner.ref("nodes", index)) for index in value.value.args)
            if sort.kind is None or sort.kind.field != "func":
                msg = "Named function value requires a Func sort"
                raise TypeError(msg)
            kind = record
            signature = kind.signature if isinstance(kind, pb.Primitive) and kind.signature is not None else None
            fixed = (
                len(signature.inputs)
                if signature is not None
                else len(kind.inputs)
                if isinstance(kind, pb.Function | pb.Constructor | pb.UserDefinedPrimitive)
                else 1
            )
            tail_width = len(signature.varargs) if signature is not None else int(isinstance(kind, pb.Primitive))
            total = len(captures) + len(sort.kind.value.params)
            views = presentation(declaration).views
            view_index = next(
                (
                    index
                    for index, candidate in enumerate(views)
                    if (
                        not (candidate.tail_repetitions is not None)
                        and (not (candidate.receiver is not None) or candidate.receiver < total)
                    )
                    or (
                        (candidate.tail_repetitions is not None)
                        and tail_width
                        and total == fixed + tail_width * candidate.tail_repetitions
                    )
                ),
                None,
            )
            _, children, owner_sort = _partial_function(
                RuntimeFunction(declaration, view_index),
                captures,
                function_sort,
                convert_captures=False,
            )
            type_bindings = ()
            if signature is not None:
                _, type_bindings = infer_call_bindings(
                    declaration,
                    [
                        *(child.__egg_ref__.owner.ref("sorts", child.__egg_ref__.read().sort_id) for child in children),
                        *(function_sort.owner.ref("sorts", index) for index in sort.kind.value.params),
                    ],
                    function_sort.owner.ref("sorts", sort.kind.value.result),
                )
                # An existing concrete owner already carries these bindings.
                # Retain the rest of the concrete Func constraints explicitly.
                owner_bindings: dict[int, Ref] = {}
                if view_index is not None and owner_sort is not None:
                    view = views[view_index]
                    if view.owner is not None:
                        infer_sort(
                            view.owner,
                            owner_sort,
                            signature.type_param_count,
                            owner_bindings,
                        )
                type_bindings = tuple((index, bound) for index, bound in type_bindings if index not in owner_bindings)
            function = create_callable(
                declaration, owner_sort=owner_sort, view_index=view_index, type_bindings=type_bindings
            )
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
    declaration = _definition_at(x.__egg_ref__, "callable", node.kind.value.func)
    views = presentation(declaration).views
    if views and views[0].kind == CallKind.CONSTANT:
        if node.kind.value.args:
            msg = "A constant call must be nullary"
            raise ValueError(msg)
        return Ident(views[0].path[-1], ".".join(views[0].path[:-1]) or None)
    return None


def get_let_name(x: object) -> str | None:
    """Return a global binding's exact name, or None for other expressions."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    node: pb.Node = x.__egg_ref__.read()
    return node.kind.value if node.kind is not None and node.kind.field == "binding" else None


def get_var_name(x: object) -> str | None:
    """Return a variable node's canonical name, or None for other node kinds."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    node: pb.Node = x.__egg_ref__.read()
    return node.kind.value if node.kind is not None and node.kind.field == "var" else None


def _deconstruct_call(x: RuntimeExpr) -> tuple[RuntimeClass | RuntimeFunction, tuple[RuntimeExpr, ...]] | None:  # noqa: C901, PLR0912
    """Select a declaration's Python view and restore its argument ordering."""
    node: pb.Node = x.__egg_ref__.read()
    if node.kind is not None and node.kind.field == "apply":
        from .runtime import _Intrinsic  # noqa: PLC0415

        application = node.kind.value
        if not application.has_field("function"):
            msg = "Function application has no function input"
            raise ValueError(msg)
        receiver = x.__egg_ref__.owner.ref("nodes", application.function)
        function_sort = receiver.owner.ref("sorts", receiver.read().sort_id)
        function_type: pb.Sort = function_sort.read()
        if (
            function_type.kind is None
            or function_type.kind.field != "func"
            or len(application.args) != len(function_type.kind.value.params)
        ):
            msg = "Function application has invalid argument sorts or arity"
            raise ValueError(msg)
        infer_sort(
            function_sort.owner.ref("sorts", function_type.kind.value.result),
            x.__egg_ref__.owner.ref("sorts", node.sort_id),
            0,
            {},
        )
        arguments = tuple(RuntimeExpr(x.__egg_ref__.owner.ref("nodes", index)) for index in application.args)
        for argument, expected in zip(arguments, function_type.kind.value.params, strict=True):
            child = argument.__egg_ref__
            infer_sort(
                function_sort.owner.ref("sorts", expected), child.owner.ref("sorts", child.read().sort_id), 0, {}
            )
        return RuntimeFunction(_Intrinsic.APPLY, bound=_class_for_sort(function_sort)), (
            RuntimeExpr(receiver),
            *arguments,
        )
    if node.kind is None or node.kind.field != "call":
        return None
    call = node.kind.value
    declaration = _definition_at(x.__egg_ref__, "callable", call.func)
    record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = declaration.read()
    arguments = tuple(RuntimeExpr(x.__egg_ref__.owner.ref("nodes", index)) for index in call.args)
    kind = record
    signature = kind.signature if isinstance(kind, pb.Primitive) and kind.signature is not None else None
    bindings: tuple[tuple[int, Ref], ...] = ()
    type_bindings = ()
    if signature is not None or isinstance(kind, pb.Function | pb.Constructor | pb.UserDefinedPrimitive):
        determined, bindings = infer_call_bindings(
            declaration,
            [arg.__egg_ref__.owner.ref("sorts", arg.__egg_ref__.read().sort_id) for arg in arguments],
            x.__egg_ref__.owner.ref("sorts", node.sort_id),
        )
        type_bindings = tuple((index, sort) for index, sort in bindings if index not in determined)
    if not presentation(declaration).views:
        return RuntimeFunction(declaration, type_bindings=type_bindings), arguments
    if any(
        (view.tail_repetitions is not None) and view.kind not in {CallKind.FUNCTION, CallKind.METHOD}
        for view in presentation(declaration).views
    ):
        msg = "Tail specialization supports only function and method views"
        raise ValueError(msg)
    fixed = len(signature.inputs) if signature is not None else len(arguments)
    tail_width = len(signature.varargs) if signature is not None else 0
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
            for index, view in enumerate(presentation(declaration).views)
            if (
                not (view.tail_repetitions is not None)
                and (not (view.receiver is not None) or view.receiver < len(arguments))
            )
            or (
                (view.tail_repetitions is not None)
                and tail_width
                and len(arguments) == fixed + tail_width * view.tail_repetitions
            )
        ),
        None,
    )
    if selected is None:
        return RuntimeFunction(declaration, type_bindings=type_bindings), arguments
    view_index, view = selected
    tail_slot = fixed + ((view.receiver is not None) and view.receiver == fixed)
    ordered: list[RuntimeExpr] = []
    if view.receiver is not None:
        ordered.append(arguments[view.receiver])
    for parameter in view.params:
        ordered.extend(
            arguments[parameter.core_input :]
            if tail_width and not (view.tail_repetitions is not None) and parameter.core_input == tail_slot
            else (arguments[parameter.core_input],)
        )
    if view.kind == CallKind.INITIALIZER:
        return _class_for_sort(x.__egg_ref__.owner.ref("sorts", node.sort_id)), tuple(ordered)
    owner = None
    if view.owner is not None:
        owner_sort = None
        if view.receiver is not None:
            receiver = arguments[view.receiver].__egg_ref__
            owner_sort = receiver.owner.ref("sorts", receiver.read().sort_id)
        else:
            owner_sort = substitute_sort(view.owner, dict(bindings))
        pattern = view.owner
        size = signature.type_param_count if signature is not None else 0
        infer_sort(pattern, owner_sort, size, dict(bindings))
        # A concrete owner already carries its own specialization.
        # Keep only result bindings absent from that owner.
        owner_bindings: dict[int, Ref] = {}
        infer_sort(pattern, owner_sort, size, owner_bindings)
        type_bindings = tuple((index, sort) for index, sort in type_bindings if index not in owner_bindings)
        owner = _class_for_sort(owner_sort)
    return RuntimeFunction(declaration, view_index, owner, type_bindings=type_bindings), tuple(ordered)


def get_callable_fn(x: T) -> Callable[..., T | None] | None:
    """Get the generated callable responsible for a call expression."""
    if not isinstance(x, RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    result = _deconstruct_call(x)
    return None if result is None else cast("Callable[..., T | None]", result[0])


@overload
def get_callable_args(x: BaseExpr, fn: None = ...) -> tuple[BaseExpr, ...] | None: ...


@overload
def get_callable_args(x: BaseExpr, fn: Callable[[Unpack[TS]], BaseExpr | None]) -> tuple[*TS] | None: ...


def get_callable_args(x: object, fn: object = None) -> tuple[object, ...] | None:
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
        if actual.__egg_intrinsic__ is not None or fn.__egg_intrinsic__ is not None:
            return arguments if actual.__egg_intrinsic__ == fn.__egg_intrinsic__ else None
        return arguments if actual.__egg_ref__ == fn.__egg_ref__ or actual._identity() == fn._identity() else None
    return None
