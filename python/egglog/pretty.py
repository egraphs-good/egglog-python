"""
Pretty printing for declarations.
"""

from __future__ import annotations

import ast
import struct
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, TypeAlias

import black
from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Ref, StructuralView, _structural_parts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = [
    "BINARY_METHODS",
    "UNARY_METHODS",
    "pretty_ref",
]
BLACK_MODE = black.Mode(line_length=180)
_REF_BLACK_MODE = black.Mode(line_length=BLACK_MODE.line_length, string_normalization=False)

# Special methods which we might want to use as functions
# Mapping to the operator they represent for pretty printing them
# https://docs.python.org/3/reference/datamodel.html
BINARY_METHODS = {
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
    # Numeric
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__matmul__": "@",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    # TODO: Support divmod, with tuple return value
    # "__divmod__": "divmod",
    # TODO: Three arg power
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
}


UNARY_METHODS = {
    "__pos__": "+",
    "__neg__": "-",
    "__invert__": "~",
}


def _name_expr(tp_name: str, expr: str, *, copy_identifier: bool, statements: list[str], counts: dict[str, int]) -> str:
    """Name a computed expression, copying an identifier only when it is shared."""
    if expr.isidentifier() and not copy_identifier:
        return expr
    counts[tp_name] = counts.get(tp_name, 0) + 1
    name = f"_{tp_name}_{counts[tp_name]}"
    statements.append(f"{name} = copy({expr})" if expr.isidentifier() else f"{name} = {expr}")
    return name


_RefFrame: TypeAlias = str | Ref | tuple[Ref, bool, bool] | tuple[Ref, pb.Action | pb.Command]


@dataclass
class _SharedFrame:
    """Name a repeated call only after its children have been rendered."""

    reference: Ref
    output_start: int


@dataclass
class _MutationFrame:
    """Resume a call after naming its input, then capture its statement output."""

    reference: Ref
    target: Ref
    target_piece: tuple[Ref, bool, bool]
    pieces: list[_RefFrame]
    type_name: str
    output_start: int
    name: str | None = None


def _format_pyobject(reference: Ref, bare: bool) -> str:
    """Trusted public-Python display, distinct from opaque record inspection."""
    from .deconstruct import _pyobject_value  # noqa: PLC0415
    from .runtime import RuntimeExpr, _class_for_sort  # noqa: PLC0415

    value = repr(_pyobject_value(RuntimeExpr(reference)))
    if not is_valid_python_expr(value):
        value = f"eval({value!r})"
    if not bare:
        node: pb.Node = reference.read()
        value = f"{_class_for_sort(reference.owner.ref('sorts', node.sort_id))}({value})"
    # Preserve the public host-literal printer's quote/invalid-repr policy.
    value = ast.unparse(ast.parse(value))
    try:
        return black.format_str(value, mode=BLACK_MODE).strip()
    except black.parsing.InvalidInput:
        return value


def pretty_ref(  # noqa: C901, PLR0912
    root: Ref,
    *,
    custom_literal: Callable[[Ref, bool], str] | None = None,
    variables: Mapping[str, str] | None = None,
) -> str:
    """Print records inertly unless the caller supplies a trusted Custom formatter."""
    from .runtime import _definition_at  # noqa: PLC0415

    # Frames stream output rather than recursively copying each child's string.
    # The Boolean asks scalar arguments to omit their inferred wrapper.
    frames: list[_RefFrame | _MutationFrame | _SharedFrame] = [(root, False, False)]
    output: list[str] = []
    statements: list[str] = []
    expression_names: dict[Ref, str] = {}
    counts: dict[str, int] = {}
    parents: Counter[Ref] = Counter()
    pending, seen = [root], set()
    while pending:
        reference = pending.pop()
        if reference in seen:
            continue
        seen.add(reference)
        for child, _ in _structural_parts(reference, identity=False)[1]:
            if child.role not in {"sorts", "declarations", "files"}:
                parents[child] += 1
                pending.append(child)
    active: set[Ref] = set()
    # Lambda bodies and captures are embedded expressions, not statement suites.
    format_output = variables is None
    while frames:
        frame = frames.pop()
        if isinstance(frame, _SharedFrame):
            if frame.reference not in expression_names:
                rendered = "".join(output[frame.output_start :])
                if parents[frame.reference] * (len(rendered) - 10) > 110:
                    shared_node: pb.Node = frame.reference.read()
                    sort_record: pb.Sort = frame.reference.owner.ref("sorts", shared_node.sort_id).read()
                    type_name = "expr"
                    if sort_record.kind is not None:
                        if sort_record.kind.field == "eq":
                            type_name = sort_record.kind.value.rsplit(".", 1)[-1]
                        elif sort_record.kind.field == "family":
                            type_name = sort_record.kind.value.name
                    name = _name_expr(
                        type_name if type_name.isidentifier() else "expr",
                        rendered,
                        copy_identifier=False,
                        statements=statements,
                        counts=counts,
                    )
                    expression_names[frame.reference] = name
                    del output[frame.output_start :]
                    output.append(name)
            continue
        if isinstance(frame, _MutationFrame):
            rendered = "".join(output[frame.output_start :])
            del output[frame.output_start :]
            if frame.name is None:
                frame.name = _name_expr(
                    frame.type_name,
                    rendered,
                    copy_identifier=parents[frame.target] > 1,
                    statements=statements,
                    counts=counts,
                )
                expression_names[frame.reference] = frame.name
                frame.output_start = len(output)
                frames.append(frame)
                frames.extend(
                    reversed([frame.name if piece is frame.target_piece else piece for piece in frame.pieces])
                )
            else:
                statements.append(rendered)
                output.append(frame.name)
            continue
        if isinstance(frame, str):
            output.append(frame)
            continue
        if isinstance(frame, Ref):
            active.remove(frame)
            continue
        if len(frame) == 2:
            # Inline Rule.head actions borrow the rule owner, not its arena
            # identity: re-entering that Ref would falsely signal a cycle.
            reference, record = frame
            bare = parentheses = False
        else:
            reference, bare, parentheses = frame
            if reference in expression_names:
                output.append(expression_names[reference])
                continue
            if reference in active:
                msg = "Cyclic expression presentation is not migrated"
                raise ValueError(msg)
            active.add(reference)
            if len(active) > 100:
                format_output = False
            frames.append(reference)
            record = reference.read()
            if (
                isinstance(record, pb.Node)
                and record.kind is not None
                and record.kind.field == "call"
                and parents[reference] > 1
                and variables is None
                and root.role == "nodes"
            ):
                frames.append(_SharedFrame(reference, len(output)))
        owner = reference.owner
        if isinstance(record, pb.Command) and record.kind is not None and record.kind.field == "action":
            record = record.kind.value
        pieces: list[_RefFrame]
        if isinstance(record, pb.Command):
            if record.kind is None:
                msg_0 = "Cannot print a command without a kind"
                raise ValueError(msg_0)
            if record.kind.field == "run":
                run = record.kind.value
                groups = ", ".join(repr(name) for name in run.groups)
                if run.groups == [""]:
                    groups = ""
                if run.scheduler:
                    groups += (", " if groups else "") + f"scheduler={run.scheduler!r}"
                pieces = [f"run({groups})"]
            elif record.kind.field == "add_rules":
                addition = record.kind.value
                pieces = ["ruleset("]
                for index, rule_index in enumerate(addition.rules):
                    if index:
                        pieces.append(", ")
                    pieces.append((owner.ref("rules", rule_index), False, False))
                if addition.group:
                    pieces.append((", " if addition.rules else "") + f"name={addition.group!r}")
                pieces.append(")")
            elif record.kind.field == "bind_scheduler":
                binding = record.kind.value
                scheduler = binding.scheduler
                if scheduler is None or scheduler.kind is None:
                    msg_0 = "Cannot print BindScheduler without a configuration"
                    raise ValueError(msg_0)
                if scheduler.kind.field != "back_off":
                    raise NotImplementedError(f"Scheduler presentation is not migrated: {scheduler.kind.field}")
                scheduler_options: pb.BackOff = scheduler.kind.value
                settings = [
                    f"{name}={getattr(scheduler_options, name)}"
                    for name in ("match_limit", "ban_length", "node_limit")
                    if scheduler_options.has_field(name)
                ]
                pieces = [f"bind_scheduler({binding.name!r}, back_off({', '.join(settings)}))"]
            elif record.kind.field in {"repeat", "saturate"}:
                control = record.kind.value
                if control.until:
                    if isinstance(control, pb.Repeat):
                        pieces = [f"repeat(times={control.times}, body=seq("]
                    else:
                        limit = (
                            f"max_iterations={control.max_iterations}, " if control.has_field("max_iterations") else ""
                        )
                        pieces = [f"saturate({limit}body=seq("]
                    for index, child in enumerate(control.body):
                        if index:
                            pieces.append(", ")
                        pieces.append((reference, child))
                    pieces.append("), until=[")
                    for index, fact in enumerate(control.until):
                        if index:
                            pieces.append(", ")
                        pieces.append((owner.ref("nodes", fact), False, False))
                    pieces.append("])")
                elif isinstance(control, pb.Repeat) and control.times == 1:
                    pieces = ["seq("]
                    for index, child in enumerate(control.body):
                        if index:
                            pieces.append(", ")
                        pieces.append((reference, child))
                    pieces.append(")")
                else:
                    pieces = ["("]
                    if len(control.body) == 1:
                        pieces.append((reference, control.body[0]))
                    else:
                        pieces.append("seq(")
                        for index, child in enumerate(control.body):
                            if index:
                                pieces.append(", ")
                            pieces.append((reference, child))
                        pieces.append(")")
                    pieces.append(")")
                    if isinstance(control, pb.Saturate) and not control.has_field("max_iterations"):
                        pieces.append(".saturate()")
                    else:
                        count = control.times if isinstance(control, pb.Repeat) else control.max_iterations
                        pieces.append(f" * {count}")
            elif record.kind.field == "stats":
                pieces = ["stats()"]
            else:
                msg_0 = f"Command presentation is not migrated: {record.kind.field}"
                raise NotImplementedError(msg_0)
        elif isinstance(record, pb.Sort):
            if record.kind is None:
                msg = "Cannot print a sort without a kind"
                raise ValueError(msg)
            if record.kind.field == "func":
                pieces = ["UnstableFn["]
                for index, argument in enumerate([record.kind.value.result, *record.kind.value.params]):
                    if index:
                        pieces.append(", ")
                    pieces.append((owner.ref("sorts", argument), False, False))
                pieces.append("]")
                frames.extend(reversed(pieces))
                continue
            if record.kind.field not in {"eq", "family"}:
                raise NotImplementedError(f"Sort presentation is not migrated: {record.kind.field}")
            name = record.kind.value if record.kind.field == "eq" else record.kind.value.name
            namespace = "eq_sort" if record.kind.field == "eq" else "host_sort_family"
            sort_declaration: pb.Declaration = _definition_at(reference, namespace, name).read()
            assert sort_declaration.kind is not None
            assert isinstance(sort_declaration.kind.value, pb.EqSort | pb.HostSortFamily)
            bindings = sort_declaration.kind.value.bindings
            path = bindings.python.path if bindings is not None and bindings.python is not None else [name]
            pieces = [path[-1]]
            if record.kind.field == "family" and record.kind.value.args:
                pieces.append("[")
                for index, argument in enumerate(record.kind.value.args):
                    if index:
                        pieces.append(", ")
                    pieces.append((owner.ref("sorts", argument), False, False))
                pieces.append("]")
        elif isinstance(record, pb.Node):
            if record.kind is None:
                msg = "Cannot print a node without a kind"
                raise ValueError(msg)
            sort = owner.ref("sorts", record.sort_id)
            match record.kind.field:
                case "var":
                    name = record.kind.value
                    pieces = [variables.get(name, name) if variables is not None else name]
                case "primitive_value":
                    primitive = record.kind.value.value
                    if primitive is None:
                        msg = "Cannot print a primitive without a value"
                        raise ValueError(msg)
                    if primitive.field == "f64_bits":
                        value = repr(struct.unpack("!d", struct.pack("!Q", primitive.value))[0])
                    elif primitive.field in {"i64", "string", "bool"}:
                        value = repr(primitive.value)
                    elif primitive.field == "unit":
                        value = ""
                    elif primitive.field == "custom" and custom_literal is not None:
                        frames.append(custom_literal(reference, bare))
                        continue
                    elif primitive.field in {"big_int", "big_rat", "rational"}:
                        from .deconstruct import get_literal_value  # noqa: PLC0415
                        from .runtime import RuntimeExpr  # noqa: PLC0415

                        exact = get_literal_value(RuntimeExpr(reference))
                        if primitive.field == "rational":
                            assert isinstance(exact, Fraction)
                            arguments = [str(exact.numerator), str(exact.denominator)]
                        else:
                            string_constructor = _definition_at(reference, "callable", "egglog.core.bigint.from_string")
                            integer_metadata: pb.Declaration = string_constructor.read()
                            integer_views = (
                                integer_metadata.bindings.python.views
                                if integer_metadata.bindings and integer_metadata.bindings.python
                                else []
                            )
                            class_view = next(
                                (v for v in integer_views if v.kind == pb.PythonCallKind.CLASS_METHOD), None
                            )
                            if (
                                class_view is None
                                or class_view.owner is None
                                or class_view.owner.kind is None
                                or class_view.owner.kind.field != "sort"
                            ):
                                msg = "BigInt value presentation requires a supplied from-string classmethod"
                                raise ValueError(msg)
                            integer_sort = string_constructor.owner.ref("sorts", class_view.owner.kind.value)
                            constructor = f"{pretty_ref(integer_sort)}.{class_view.path[-1]}"
                            if primitive.field == "big_int":
                                frames.append(f"{constructor}({primitive.value!r})")
                                continue
                            assert isinstance(exact, Fraction)
                            # The validated wire already owns canonical decimal
                            # strings; str(large_int) would impose a host limit.
                            arguments = [
                                f"{constructor}({part!r})"
                                for part in (primitive.value.numerator, primitive.value.denominator)
                            ]
                        ratio_constructor = _definition_at(
                            reference,
                            "callable",
                            "egglog.experimental.rational.make"
                            if primitive.field == "rational"
                            else "egglog.core.bigrat.make",
                        )
                        ratio_metadata: pb.Declaration = ratio_constructor.read()
                        ratio_views = (
                            ratio_metadata.bindings.python.views
                            if ratio_metadata.bindings and ratio_metadata.bindings.python
                            else []
                        )
                        init_view = next((v for v in ratio_views if v.kind == pb.PythonCallKind.INITIALIZER), None)
                        if init_view is None or sorted(
                            p.core_input for p in init_view.params if p.has_field("core_input")
                        ) != [0, 1]:
                            msg = "Ratio value presentation requires a supplied two-input initializer"
                            raise ValueError(msg)
                        pieces = [
                            (sort, False, False),
                            "(",
                            ", ".join(arguments[p.core_input] for p in init_view.params),
                            ")",
                        ]
                        frames.extend(reversed(pieces))
                        continue
                    elif primitive.field == "maybe":
                        from .builtins import _container_member, _maybe_value  # noqa: PLC0415
                        from .runtime import RuntimeExpr, _class_for_sort  # noqa: PLC0415

                        child = _maybe_value(RuntimeExpr(reference))
                        constructor = _container_member(
                            _class_for_sort(sort),
                            "egglog.experimental.maybe.none" if child is None else "egglog.experimental.maybe.some",
                            pb.PythonCallKind.CLASS_METHOD,
                        )
                        pieces = [str(constructor), "("]
                        if child is not None:
                            pieces.append((child.__egg_ref__, False, False))
                        pieces.append(")")
                        frames.extend(reversed(pieces))
                        continue
                    elif primitive.field == "pair":
                        from .builtins import _pair_value  # noqa: PLC0415
                        from .runtime import RuntimeExpr  # noqa: PLC0415

                        left, right = _pair_value(RuntimeExpr(reference))
                        pieces = [
                            (sort, False, False),
                            "(",
                            (left.__egg_ref__, False, False),
                            ", ",
                            (right.__egg_ref__, False, False),
                            ")",
                        ]
                        frames.extend(reversed(pieces))
                        continue
                    elif primitive.field == "partial_call":
                        from functools import partial  # noqa: PLC0415

                        from .deconstruct import get_literal_value  # noqa: PLC0415
                        from .runtime import RuntimeExpr, RuntimeFunction  # noqa: PLC0415

                        value = get_literal_value(RuntimeExpr(reference))
                        function = value.func if isinstance(value, partial) else value
                        if isinstance(function, RuntimeFunction):
                            declaration = function.__egg_ref__
                            declared: pb.Declaration = declaration.read()
                            computed = declared.kind.value if declared.kind is not None else None
                            if (
                                isinstance(computed, pb.Primitive)
                                and computed.name.startswith("__egglog_python_lambda_")
                                and declared.bindings is not None
                                and declared.bindings.python is not None
                                and len(declared.bindings.python.views) == 1
                            ):
                                view = declared.bindings.python.views[0]
                                names = [parameter.name for parameter in view.params]
                                captures = value.args if isinstance(value, partial) else ()
                                bindings = {
                                    f"_{index}": "("
                                    + pretty_ref(
                                        child.__egg_ref__,
                                        custom_literal=custom_literal,
                                        variables=variables if variables is not None else {},
                                    )
                                    + ")"
                                    for index, child in enumerate(captures)
                                }
                                bindings.update({
                                    f"_{index}": name for index, name in enumerate(names) if index >= len(captures)
                                })
                                body = pretty_ref(
                                    declaration.owner.ref("nodes", computed.body),
                                    custom_literal=custom_literal,
                                    variables=bindings,
                                )
                                parameters = ", ".join(names[len(captures) :])
                                text = f"lambda {parameters}: {body}" if parameters else f"lambda: {body}"
                                pieces = [f"({text})" if parentheses else text]
                                frames.extend(reversed(pieces))
                                continue
                        if isinstance(value, partial):
                            pieces = ["partial(", str(value.func)]
                            for child in value.args:
                                assert isinstance(child, RuntimeExpr)
                                pieces.extend([", ", (child.__egg_ref__, False, False)])
                            pieces.append(")")
                        else:
                            pieces = [str(value)]
                        frames.extend(reversed(pieces))
                        continue
                    elif primitive.field in {"vec", "set", "multiset"}:
                        from .builtins import _container_items, _container_member  # noqa: PLC0415
                        from .runtime import RuntimeExpr, _class_for_sort  # noqa: PLC0415

                        children = _container_items(RuntimeExpr(reference))
                        if not children and primitive.field == "vec":
                            frames.append(
                                str(
                                    _container_member(
                                        _class_for_sort(sort), "egglog.core.vec.empty", pb.PythonCallKind.CLASS_METHOD
                                    )
                                )
                                + "()"
                            )
                            continue
                        pieces = [(sort, False, False), "("]
                        for index, child in enumerate(children):
                            if index:
                                pieces.append(", ")
                            pieces.append((child.__egg_ref__, False, False))
                        pieces.append(")")
                        frames.extend(reversed(pieces))
                        continue
                    elif primitive.field == "map":
                        from .builtins import _container_member, _map_entries  # noqa: PLC0415
                        from .runtime import RuntimeExpr, _class_for_sort  # noqa: PLC0415

                        entries = _map_entries(RuntimeExpr(reference))
                        target = _class_for_sort(sort)
                        empty = _container_member(target, "egglog.core.map.empty", pb.PythonCallKind.CLASS_METHOD)
                        pieces = [str(empty), "()"]
                        if entries:
                            insert = _container_member(target, "egglog.core.map.insert", pb.PythonCallKind.METHOD)
                            metadata: pb.Declaration = insert.__egg_ref__.read()
                            assert metadata.bindings is not None
                            assert metadata.bindings.python is not None
                            assert insert.__egg_view__ is not None
                            insert_view = metadata.bindings.python.views[insert.__egg_view__]
                            for key, value_ref in entries:
                                arguments = {1: key, 2: value_ref}
                                pieces.extend([".", insert_view.path[-1], "("])
                                for index, parameter in enumerate(insert_view.params):
                                    if index:
                                        pieces.append(", ")
                                    pieces.append((arguments[parameter.core_input], False, False))
                                pieces.append(")")
                        frames.extend(reversed(pieces))
                        continue
                    else:
                        raise NotImplementedError(f"Value presentation is not migrated: {primitive.field}")
                    pieces = [value] if bare else [(sort, False, False), "(", value, ")"]
                case "call":
                    call = record.kind.value
                    declaration = _definition_at(reference, "callable", call.func)
                    declared = declaration.read()
                    views = (
                        declared.bindings.python.views
                        if declared.bindings is not None
                        and declared.bindings.python is not None
                        and declared.bindings.python.views
                        else []
                    )
                    args = [owner.ref("nodes", index) for index in call.args]
                    if any(
                        candidate.has_field("tail_repetitions")
                        and candidate.kind not in {pb.PythonCallKind.FUNCTION, pb.PythonCallKind.METHOD}
                        for candidate in views
                    ):
                        msg = "Tail specialization supports only function and method views"
                        raise ValueError(msg)
                    assert declared.kind is not None
                    kind = declared.kind.value
                    if (
                        isinstance(kind, pb.HostPrimitive)
                        and kind.typing is not None
                        and kind.typing.field == "application"
                    ):
                        from .type_constraint_solver import infer_sort  # noqa: PLC0415

                        if not args:
                            msg = "Function application has no function input"
                            raise ValueError(msg)
                        function_sort = args[0].owner.ref("sorts", args[0].read().sort_id)
                        function_type: pb.Sort = function_sort.read()
                        if (
                            function_type.kind is None
                            or function_type.kind.field != "func"
                            or len(args) != len(function_type.kind.value.params) + 1
                        ):
                            msg = "Function application has invalid argument sorts or arity"
                            raise ValueError(msg)
                        infer_sort(function_sort.owner.ref("sorts", function_type.kind.value.result), sort, 0, {})
                        pieces = [(args[0], False, False), "("]
                        for index, (argument, expected) in enumerate(
                            zip(args[1:], function_type.kind.value.params, strict=True)
                        ):
                            infer_sort(
                                function_sort.owner.ref("sorts", expected),
                                argument.owner.ref("sorts", argument.read().sort_id),
                                0,
                                {},
                            )
                            if index:
                                pieces.append(", ")
                            pieces.append((argument, True, False))
                        pieces.append(")")
                        frames.extend(reversed(pieces))
                        continue
                    signature = (
                        kind.typing.value
                        if isinstance(kind, pb.HostPrimitive)
                        and kind.typing is not None
                        and kind.typing.field == "signature"
                        else None
                    )
                    fixed = len(signature.inputs) if signature is not None else len(args)
                    tail_width = len(signature.varargs) if signature is not None else 0
                    if (
                        len(args) < fixed
                        or (len(args) - fixed) % (tail_width or 1)
                        or (not tail_width and len(args) != fixed)
                    ):
                        msg = "Call has invalid canonical argument count"
                        raise ValueError(msg)
                    if call.func == "egglog.core.ne" and len(args) == 2:
                        pieces = ["ne(", (args[0], False, False), ").to(", (args[1], False, False), ")"]
                        frames.extend(reversed(pieces))
                        continue
                    view = next(
                        (
                            candidate
                            for candidate in views
                            if (
                                not candidate.has_field("tail_repetitions")
                                and (not candidate.has_field("receiver") or candidate.receiver < len(args))
                            )
                            or (
                                candidate.has_field("tail_repetitions")
                                and tail_width
                                and len(args) == fixed + tail_width * candidate.tail_repetitions
                            )
                        ),
                        None,
                    )
                    if view is None:
                        pieces = [call.func, "("]
                        selected = [(argument, False, False) for argument in args]
                    else:
                        selected = []
                        tail_slot = fixed + (view.has_field("receiver") and view.receiver == fixed)
                        defaults: list[Ref | None] = []
                        for parameter in view.params:
                            for argument in (
                                args[parameter.core_input :]
                                if tail_width
                                and not view.has_field("tail_repetitions")
                                and parameter.core_input == tail_slot
                                else [args[parameter.core_input]]
                            ):
                                selected.append((argument, True, False))
                                defaults.append(
                                    declaration.owner.ref("nodes", parameter.default_expr)
                                    if parameter.has_field("default_expr")
                                    else None
                                )
                        while (
                            selected
                            and defaults[-1] is not None
                            and not (
                                view.has_field("mutates") and view.params[len(selected) - 1].core_input == view.mutates
                            )
                            and StructuralView(selected[-1][0]) == StructuralView(defaults[-1])
                        ):
                            selected.pop()
                            defaults.pop()
                        name = view.path[-1] if view.path else "__init__"
                        receiver_piece = (args[view.receiver], False, True) if view.has_field("receiver") else None
                        if (
                            view.kind == pb.PythonCallKind.METHOD
                            and name in UNARY_METHODS
                            and not selected
                            and not view.has_field("mutates")
                        ):
                            pieces = [UNARY_METHODS[name], (args[view.receiver], False, True)]
                            if parentheses:
                                pieces = ["(", *pieces, ")"]
                            frames.extend(reversed(pieces))
                            continue
                        if (
                            view.kind == pb.PythonCallKind.METHOD
                            and name in BINARY_METHODS
                            and len(selected) == 1
                            and not view.has_field("mutates")
                        ):
                            pieces = [
                                (args[view.receiver], False, True),
                                f" {BINARY_METHODS[name]} ",
                                (selected[0][0], True, True),
                            ]
                            if parentheses:
                                pieces = ["(", *pieces, ")"]
                            frames.extend(reversed(pieces))
                            continue
                        if (
                            view.kind == pb.PythonCallKind.METHOD
                            and name == "__getitem__"
                            and len(selected) == 1
                            and not view.has_field("mutates")
                        ):
                            pieces = [(args[view.receiver], False, True), "[", (selected[0][0], True, False), "]"]
                            frames.extend(reversed(pieces))
                            continue
                        close_call = True
                        if (
                            view.kind == pb.PythonCallKind.METHOD
                            and name in {"__setitem__", "__delitem__"}
                            and view.has_field("mutates")
                            and view.mutates == view.receiver
                        ):
                            assert receiver_piece is not None
                            pieces = [receiver_piece, "[", selected[0], "]"]
                            if name == "__delitem__":
                                pieces.insert(0, "del ")
                            if name == "__setitem__":
                                pieces.extend([" = ", selected[1]])
                            selected = []
                            close_call = False
                        elif view.kind == pb.PythonCallKind.INITIALIZER:
                            pieces = [(sort, False, False), "("]
                        elif view.kind in {pb.PythonCallKind.METHOD, pb.PythonCallKind.PROPERTY}:
                            assert receiver_piece is not None
                            pieces = [receiver_piece, ".", name]
                            if view.kind == pb.PythonCallKind.PROPERTY:
                                frames.extend(reversed(pieces))
                                continue
                            pieces.append("(")
                        elif view.kind in {pb.PythonCallKind.CLASS_METHOD, pb.PythonCallKind.CLASS_VARIABLE}:
                            assert view.owner is not None
                            assert view.owner.kind is not None
                            if view.owner.kind.field != "sort":
                                msg = "Function-type class member presentation is not migrated"
                                raise NotImplementedError(msg)
                            owner_sort = declaration.owner.ref("sorts", view.owner.kind.value)
                            if (
                                isinstance(kind, pb.HostPrimitive)
                                and kind.typing is not None
                                and kind.typing.field == "signature"
                            ):
                                from .type_constraint_solver import infer_sort, substitute_sort  # noqa: PLC0415

                                signature = kind.typing.value
                                substitutions: dict[int, Ref] = {}
                                infer_sort(
                                    declaration.owner.ref("sorts", signature.output),
                                    sort,
                                    len(signature.type_params),
                                    substitutions,
                                )
                                patterns = list(signature.inputs)
                                if signature.varargs:
                                    tail_count = len(args) - len(patterns)
                                    if tail_count < 0 or tail_count % len(signature.varargs):
                                        msg = "Class member has invalid variadic argument count"
                                        raise ValueError(msg)
                                    patterns.extend(signature.varargs * (tail_count // len(signature.varargs)))
                                for parameter, argument in zip(patterns, args, strict=True):
                                    payload: pb.Node = argument.read()
                                    infer_sort(
                                        declaration.owner.ref("sorts", parameter.sort),
                                        argument.owner.ref("sorts", payload.sort_id),
                                        len(signature.type_params),
                                        substitutions,
                                    )
                                owner_sort = substitute_sort(owner_sort, substitutions)
                            pieces = [(owner_sort, False, False), ".", name]
                            if view.kind == pb.PythonCallKind.CLASS_VARIABLE:
                                frames.extend(reversed(pieces))
                                continue
                            pieces.append("(")
                        elif view.kind == pb.PythonCallKind.FUNCTION:
                            pieces = [name, "("]
                        elif view.kind == pb.PythonCallKind.CONSTANT:
                            if args or view.params or view.owner is not None or view.has_field("receiver"):
                                msg = "A constant view must be ownerless and nullary"
                                raise ValueError(msg)
                            frames.append(name)
                            continue
                        else:
                            raise NotImplementedError(f"Python view presentation is not migrated: {view.kind}")
                    for index, argument_frame in enumerate(selected):
                        if index:
                            pieces.append(", ")
                        pieces.append(argument_frame)
                    if view is None or close_call:
                        pieces.append(")")
                    if view is not None and view.has_field("mutates"):
                        if view.has_field("receiver") and view.mutates == view.receiver:
                            assert receiver_piece is not None
                            target_piece = receiver_piece
                        else:
                            target_piece = selected[
                                next(
                                    i for i, parameter in enumerate(view.params) if parameter.core_input == view.mutates
                                )
                            ]
                        target = args[view.mutates]
                        frames.extend([
                            _MutationFrame(
                                reference, target, target_piece, pieces, pretty_ref(sort).partition("[")[0], len(output)
                            ),
                            (target, False, False),
                        ])
                        continue
                case "union":
                    pieces = ["eq("]
                    for index, argument in enumerate(record.kind.value.members):
                        if index:
                            pieces.append(").to(" if index == 1 else ", ")
                        pieces.append((owner.ref("nodes", argument), False, False))
                    pieces.append(")")
                case _:
                    raise NotImplementedError(f"Node presentation is not migrated: {record.kind.field}")
        elif isinstance(record, pb.Rule):
            if record.kind is None or record.eval_mode == pb.RuleEvalMode.UNSPECIFIED:
                msg = "Cannot print a rule without kind and evaluation mode"
                raise ValueError(msg)
            if record.include_subsumed:
                msg = "include_subsumed rule presentation is not migrated"
                raise NotImplementedError(msg)
            if record.kind.field in {"rewrite", "birewrite"}:
                if record.has_field("name") or record.no_decomp or record.eval_mode != pb.RuleEvalMode.SEMINAIVE:
                    msg = "Rewrite option presentation is not migrated"
                    raise NotImplementedError(msg)
                rewrite = record.kind.value
                assert isinstance(rewrite, pb.Rewrite | pb.BiRewrite)
                pieces = [f"{record.kind.field}(", (owner.ref("nodes", rewrite.lhs), False, False)]
                if isinstance(rewrite, pb.Rewrite) and rewrite.subsume:
                    pieces.append(", subsume=True")
                pieces.extend([").to(", (owner.ref("nodes", rewrite.rhs), False, False)])
                for condition in rewrite.conditions:
                    pieces.extend([", ", (owner.ref("nodes", condition), False, False)])
                pieces.append(")")
            elif record.kind.field == "rule":
                rule = record.kind.value
                pieces = ["rule("]
                for index, fact in enumerate(rule.query):
                    if index:
                        pieces.append(", ")
                    pieces.append((owner.ref("nodes", fact), False, False))
                options = []
                if record.has_field("name"):
                    options.append(f"name={record.name!r}")
                if record.eval_mode != pb.RuleEvalMode.SEMINAIVE:
                    options.append(f"eval_mode={record.eval_mode.name.lower().replace('_', '-')!r}")
                if record.no_decomp:
                    options.append("no_decomp=True")
                if options:
                    pieces.append((", " if rule.query else "") + ", ".join(options))
                pieces.append(").then(")
                for index, action in enumerate(rule.head):
                    if index:
                        pieces.append(", ")
                    pieces.append((reference, action))
                pieces.append(")")
            else:
                raise NotImplementedError(f"Rule presentation is not migrated: {record.kind.field}")
        elif isinstance(record, pb.Action):
            action = record
            if action.kind is None:
                msg = "Cannot print an action without a kind"
                raise ValueError(msg)
            if action.kind.field == "term":
                term = owner.ref("nodes", action.kind.value)
                node: pb.Node = term.read()
                if node.kind is not None and node.kind.field == "union":
                    if len(node.kind.value.members) != 2:
                        msg = "Only binary Union action presentation is migrated"
                        raise NotImplementedError(msg)
                    lhs, rhs = node.kind.value.members
                    pieces = [
                        "union(",
                        (term.owner.ref("nodes", lhs), False, False),
                        ").with_(",
                        (term.owner.ref("nodes", rhs), False, False),
                        ")",
                    ]
                else:
                    pieces = [(term, False, False)]
            elif action.kind.field == "set":
                assignment = action.kind.value
                if assignment.target is None or not assignment.has_field("value"):
                    msg = "Cannot print a Set without target and value"
                    raise ValueError(msg)
                target = assignment.target
                declaration = _definition_at(reference, "callable", target.func)
                declared = declaration.read()
                view = (
                    declared.bindings.python.views[0]
                    if declared.bindings is not None
                    and declared.bindings.python is not None
                    and declared.bindings.python.views
                    else None
                )
                name = view.path[-1] if view is not None else target.func
                pieces = ["set_(", name]
                if view is None or view.kind == pb.PythonCallKind.FUNCTION:
                    pieces.append("(")
                    indices = target.args if view is None else [target.args[p.core_input] for p in view.params]
                    if view is not None:
                        while indices:
                            parameter = view.params[len(indices) - 1]
                            if not parameter.has_field("default_expr") or StructuralView(
                                owner.ref("nodes", indices[-1])
                            ) != StructuralView(declaration.owner.ref("nodes", parameter.default_expr)):
                                break
                            indices.pop()
                    for index, argument in enumerate(indices):
                        if index:
                            pieces.append(", ")
                        pieces.append((owner.ref("nodes", argument), view is not None, False))
                    pieces.append(")")
                elif view.kind != pb.PythonCallKind.CONSTANT:
                    msg = "Set presentation for symbolic members is not migrated"
                    raise NotImplementedError(msg)
                elif target.args:
                    msg = "A constant Set target must be nullary"
                    raise ValueError(msg)
                pieces.extend([").to(", (owner.ref("nodes", assignment.value), False, False), ")"])
            elif action.kind.field in {"delete", "subsume"}:
                from protobuf import Oneof  # noqa: PLC0415

                from ._catalog import builtin_catalog  # noqa: PLC0415
                from ._program import Builder, _copy_record  # noqa: PLC0415

                target = _copy_record(action.kind.value)
                declaration = _definition_at(reference, "callable", target.func)
                declared = declaration.read()
                if declared.kind is None or not isinstance(
                    declared.kind.value, pb.Constructor | pb.Function | pb.Relation
                ):
                    msg = "Delete and Subsume require a table call"
                    raise ValueError(msg)
                builder = Builder()
                builder.import_ref(declaration)
                table = declared.kind.value
                if isinstance(table, pb.Relation):
                    builder.import_ref(builtin_catalog().definitions["host_sort_family", "Unit"])
                    result_sort = builder.add(
                        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Unit")))
                    )
                else:
                    result_sort = builder.import_ref(declaration.owner.ref("sorts", table.output))
                target.args = [builder.import_ref(owner.ref("nodes", index)) for index in target.args]
                call_node = builder.add(
                    "nodes",
                    pb.Node(
                        sort_id=result_sort,
                        kind=Oneof[Literal["call"], pb.Call]("call", target),
                    ),
                )
                pieces = [
                    f"{action.kind.field}(",
                    (builder.publish().ref("nodes", call_node), False, False),
                    ")",
                ]
            elif action.kind.field == "panic":
                pieces = [black.format_str(f"panic({action.kind.value!r})", mode=BLACK_MODE).strip()]
            else:
                raise NotImplementedError(f"Action presentation is not migrated: {action.kind.field}")
        else:
            raise NotImplementedError(f"Record presentation is not migrated: {reference.role}")
        frames.extend(reversed(pieces))
    program = "\n".join([*statements, "".join(output)])
    # Formatting is recursive; deep records retain the iterative raw display.
    if format_output and len(program) > BLACK_MODE.line_length:
        try:
            return black.format_str(program, mode=_REF_BLACK_MODE).strip()
        except (black.parsing.InvalidInput, RecursionError):
            pass
    return program


def is_valid_python_expr(s: str) -> bool:
    try:
        ast.parse(s, mode="eval")
    except SyntaxError:
        return False
    return True
