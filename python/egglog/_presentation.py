"""Python surface choices over owned semantic references, never wire records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from egglog_proto.egglog.v1 import egglog_pb as pb

if TYPE_CHECKING:
    from ._program import Owner, Ref


class CallKind(Enum):
    FUNCTION = auto()
    INITIALIZER = auto()
    METHOD = auto()
    CLASS_METHOD = auto()
    PROPERTY = auto()
    CLASS_VARIABLE = auto()
    CONSTANT = auto()


@dataclass(frozen=True, slots=True)
class TypePresentation:
    path: tuple[str, ...]
    type_params: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Parameter:
    core_input: int
    name: str
    default: Ref | None = None

    def __post_init__(self) -> None:
        if type(self.core_input) is not int or not 0 <= self.core_input < 2**32:
            msg = "Python parameter requires an unsigned core input index"
            raise TypeError(msg)
        if not isinstance(self.name, str) or not self.name:
            msg = "Python parameter requires a nonempty name"
            raise ValueError(msg)
        if self.default is not None:
            validate_default(self.default)


@dataclass(frozen=True, slots=True)
class CallPresentation:
    kind: CallKind
    path: tuple[str, ...] = ()
    owner: Ref | None = None
    receiver: int | None = None
    params: tuple[Parameter, ...] = ()
    mutates: int | None = None
    tail_repetitions: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.kind), CallKind):
            msg = "Python callable requires a local presentation form"
            raise TypeError(msg)
        if self.owner is not None and self.owner.role != "sorts":
            msg = "Python owner must reference a semantic sort"
            raise TypeError(msg)
        for index in (self.receiver, self.mutates, self.tail_repetitions):
            if index is not None and (type(index) is not int or not 0 <= index < 2**32):
                msg = "Python slot and tail indices must be unsigned integers"
                raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class Presentation:
    type_view: TypePresentation | None = None
    views: tuple[CallPresentation, ...] = ()


def validate_default(root: Ref) -> None:
    """Validate closed syntax in each lambda use-site binder without evaluation."""
    from ._program import StructuralView, _structural_parts  # noqa: PLC0415
    from .type_constraint_solver import infer_sort  # noqa: PLC0415

    if root.role != "nodes":
        msg = "Python defaults must reference semantic expressions"
        raise TypeError(msg)
    node: pb.Node = root.read()
    sort = root.owner.ref("sorts", node.sort_id)
    infer_sort(sort, sort, 0, {})
    pending: list[tuple[Ref, tuple[Ref, ...]]] = [(root, ())]
    seen: set[tuple[Ref, tuple[Ref, ...]]] = set()
    while pending:
        reference, binder = pending.pop()
        if (reference, binder) in seen:
            continue
        seen.add((reference, binder))
        record: pb.Node = reference.read()
        if record.kind is None:
            msg = "Python default expression has no kind"
            raise ValueError(msg)
        if record.kind.field == "var":
            name = record.kind.value
            index = int(name[1:]) if name.startswith("_") and name[1:].isascii() and name[1:].isdigit() else -1
            if not 0 <= index < len(binder) or name != f"_{index}":
                msg = "Python default contains a free variable"
                raise ValueError(msg)
            if StructuralView(reference.owner.ref("sorts", record.sort_id)) != StructuralView(binder[index]):
                msg = "Python default lambda variable has the wrong sort"
                raise ValueError(msg)
        elif (
            record.kind.field == "primitive_value"
            and record.kind.value.value is not None
            and record.kind.value.value.field == "lambda"
        ):
            value = record.kind.value.value.value
            function: pb.Sort = reference.owner.ref("sorts", record.sort_id).read()
            if function.kind is None or function.kind.field != "func":
                msg = "Python default lambda requires a Function sort"
                raise ValueError(msg)
            captures = [reference.owner.ref("nodes", index) for index in value.captures]
            body_binder = tuple(capture.owner.ref("sorts", capture.read().sort_id) for capture in captures) + tuple(
                reference.owner.ref("sorts", index) for index in function.kind.value.params
            )
            pending.extend((capture, binder) for capture in captures)
            body = reference.owner.ref("nodes", value.body)
            infer_sort(
                reference.owner.ref("sorts", function.kind.value.result),
                body.owner.ref("sorts", body.read().sort_id),
                0,
                {},
            )
            pending.append((body, body_binder))
        else:
            _, children = _structural_parts(reference, identity=False)
            pending.extend((child, binder) for child, _ in children if child.role == "nodes")


def presentation(reference: Ref) -> Presentation:
    """Resolve canonical local ownership and derive a conservative type name."""
    reference = reference.owner.ref(reference.role, reference.index)
    if reference.role not in {"declarations", "primitive_sorts", "primitives"}:
        msg = "Python presentation requires a semantic declaration reference"
        raise TypeError(msg)
    from ._program import _definition_key  # noqa: PLC0415

    record: pb.Declaration | pb.PrimitiveSort | pb.Primitive = reference.read()
    key = _definition_key(record)
    assert key is not None
    supplied = reference.owner._presentations.get(key)
    if supplied is not None:
        return supplied
    if isinstance(record, pb.Declaration):
        assert record.kind is not None
        kind = record.kind.value
    else:
        kind = record
    if isinstance(kind, pb.PrimitiveSort | pb.EqSort):
        return Presentation(type_view=TypePresentation(("egglog", "builtins", kind.name)))
    return Presentation()


def validate_view(reference: Ref, view: CallPresentation) -> None:  # noqa: C901, PLR0912
    """Admit a local layout against its actual semantic signature before use."""
    from ._program import StructuralView  # noqa: PLC0415
    from .type_constraint_solver import TypeConstraintError, infer_sort  # noqa: PLC0415

    record: pb.Declaration | pb.Primitive = reference.read()
    if isinstance(record, pb.Declaration):
        if record.kind is None:
            msg = "Python callable declaration has no kind"
            raise ValueError(msg)
        kind = record.kind.value
    else:
        kind = record
    if isinstance(kind, pb.Primitive):
        signature = kind.signature
        if signature is None:
            msg = "Python callable primitive has no signature"
            raise ValueError(msg)
        if not signature.has_field("output"):
            msg = "Python callable signature has no output"
            raise ValueError(msg)
        inputs, tail, output, binder = signature.inputs, signature.varargs, signature.output, signature.type_param_count
    elif isinstance(kind, pb.Constructor | pb.Function | pb.UserDefinedPrimitive):
        inputs, tail, output, binder = kind.inputs, [], kind.output, 0
    elif isinstance(kind, pb.Relation):
        inputs, tail, output, binder = kind.inputs, [], None, 0
    else:
        msg = "Python call layout requires a callable signature"
        raise TypeError(msg)
    specialized = view.tail_repetitions is not None
    consumes_tail = not specialized and view.receiver == len(inputs)
    if consumes_tail and (len(tail) != 1 or view.kind != CallKind.METHOD):
        msg = "Tail receiver requires a homogeneous variadic method"
        raise ValueError(msg)
    if specialized and (not tail or view.kind not in {CallKind.METHOD, CallKind.FUNCTION}):
        msg = "Tail specialization requires a variadic function or method"
        raise ValueError(msg)
    tail_slot = len(inputs) + consumes_tail
    count = (
        len(inputs) + len(tail) * view.tail_repetitions if view.tail_repetitions is not None else tail_slot + bool(tail)
    )
    slots = ([view.receiver] if view.receiver is not None else []) + [parameter.core_input for parameter in view.params]
    if count > 2**32 - 1 or len(slots) != count or sorted(slots) != list(range(count)):
        msg = "Python view must map every core input exactly once"
        raise ValueError(msg)
    patterns = list(inputs)
    if specialized:
        assert view.tail_repetitions is not None
        patterns.extend(tail * view.tail_repetitions)
    elif consumes_tail:
        patterns.extend(tail)
    if view.owner is not None:
        pending = [view.owner]
        seen = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            sort: pb.Sort = current.read()
            if sort.kind is None or (sort.kind.field == "var" and sort.kind.value >= binder):
                msg = "Python owner contains an out-of-scope signature variable"
                raise ValueError(msg)
            children = (
                sort.kind.value.args
                if sort.kind.field == "primitive"
                else [*sort.kind.value.params, sort.kind.value.result]
                if sort.kind.field == "func"
                else ()
            )
            pending.extend(current.owner.ref("sorts", index) for index in children)
        if view.receiver is not None and StructuralView(view.owner) != StructuralView(
            reference.owner.ref("sorts", patterns[view.receiver].sort)
        ):
            msg = "Python receiver input must match its owner"
            raise ValueError(msg)
        if view.kind == CallKind.INITIALIZER and (
            output is None or StructuralView(view.owner) != StructuralView(reference.owner.ref("sorts", output))
        ):
            msg = "Python initializer output differs from owner"
            raise ValueError(msg)
    names = set()
    has_default = False
    for position, parameter in enumerate(view.params):
        is_tail = bool(tail) and not specialized and parameter.core_input == tail_slot
        if not parameter.name.isidentifier() or parameter.name in names:
            msg = "Python parameters require distinct identifiers"
            raise ValueError(msg)
        names.add(parameter.name)
        if is_tail and (position != len(view.params) - 1 or parameter.default is not None):
            msg = "A uniform tail must be final and have no default"
            raise ValueError(msg)
        if parameter.default is not None:
            if specialized and parameter.core_input >= len(inputs):
                msg = "Specialized tail inputs cannot have defaults"
                raise ValueError(msg)
            try:
                infer_sort(
                    reference.owner.ref("sorts", patterns[parameter.core_input].sort),
                    parameter.default.owner.ref("sorts", parameter.default.read().sort_id),
                    binder,
                    {},
                )
            except TypeConstraintError as error:
                msg = "Default root has an invalid sort"
                raise ValueError(msg) from error
            has_default = True
        elif has_default and not is_tail:
            msg = "Required parameter follows a default"
            raise ValueError(msg)
    if view.mutates is not None and (
        view.kind == CallKind.INITIALIZER
        or view.mutates >= len(inputs)
        or output is None
        or view.mutates not in slots
        or StructuralView(reference.owner.ref("sorts", inputs[view.mutates].sort))
        != StructuralView(reference.owner.ref("sorts", output))
    ):
        msg = "Invalid mutated input"
        raise ValueError(msg)


def mapping_views(
    catalog: pb.BuiltinCatalog, templates: Owner, data: Mapping[str, object]
) -> dict[tuple[str, str], Presentation]:
    """Resolve stable mapping keys and independently owned owner/default roots."""
    type_rows = data["types"]
    callable_rows = data["callables"]
    assert isinstance(type_rows, dict)
    assert isinstance(callable_rows, dict)
    result: dict[tuple[str, str], Presentation] = {}
    admitted = set()
    declarations: list[pb.PrimitiveSort | pb.Primitive] = [*catalog.primitive_sorts, *catalog.primitives]
    for declaration in declarations:
        key = declaration.name
        nominal = isinstance(declaration, pb.PrimitiveSort)
        identity = ("primitive_sort" if nominal else "callable", key)
        rows = type_rows if nominal else callable_rows
        if key not in rows:
            continue
        admitted.add((nominal, key))
        if nominal:
            row = rows[key]
            result[identity] = Presentation(
                type_view=TypePresentation(tuple(row["path"]), tuple(row.get("typeParams", ())))
            )
            continue
        for row in rows[key]:
            for parameter in row.get("params", ()):
                if "default" in parameter and (
                    type(parameter["default"]) is not int
                    or not 0 <= parameter["default"] < len(templates._slots["nodes"])
                ):
                    msg = "Default root has an invalid index"
                    raise ValueError(msg)
        views = []
        for row in rows[key]:
            kind = CallKind[row["form"]]
            receiver = row.get("receiver", 0 if kind in {CallKind.METHOD, CallKind.PROPERTY} else None)
            path = (
                (key.rsplit(".", 1)[-1],) if kind in {CallKind.METHOD, CallKind.CLASS_METHOD, CallKind.PROPERTY} else ()
            )
            views.append(
                CallPresentation(
                    kind=kind,
                    path=tuple(row.get("path", path)),
                    owner=templates.ref("sorts", row["owner"]) if "owner" in row else None,
                    receiver=receiver,
                    params=tuple(
                        Parameter(
                            core_input=parameter.get(
                                "coreInput", position + (receiver is not None and position >= receiver)
                            ),
                            name=parameter["name"],
                            default=templates.ref("nodes", parameter["default"]) if "default" in parameter else None,
                        )
                        for position, parameter in enumerate(row.get("params", ()))
                    ),
                    mutates=row.get("mutates"),
                    tail_repetitions=row.get("tailRepetitions"),
                )
            )
        result[identity] = Presentation(views=tuple(views))
    missing = {(True, key) for key in type_rows} | {(False, key) for key in callable_rows}
    if missing - admitted:
        raise ValueError(f"Python mapping names absent native definitions: {sorted(missing - admitted)}")
    return result


def builtin_owner(catalog: pb.BuiltinCatalog, *, mapping: Mapping[str, object] | None = None) -> Owner:
    """Join local surface choices to actual native definitions and owned templates."""
    from ._builtin_presentation import DATA  # noqa: PLC0415
    from ._program import Builder  # noqa: PLC0415

    mapping = DATA if mapping is None else mapping

    def local_views(owner: Owner) -> dict[tuple[str, str], Presentation]:
        fragment = Builder.from_program(pb.Program.from_json(json.dumps(mapping["fragment"])), environment=(owner,))
        return mapping_views(catalog, fragment.publish(), mapping)

    return Builder.from_builtins(catalog).publish(presentation_factory=local_views)
