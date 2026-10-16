"""
Pretty printing for declarations.
"""

from __future__ import annotations

import ast
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, TypeAlias, assert_never

import black
import cloudpickle
from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Ref, StructuralView
from .declarations import *

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = [
    "BINARY_METHODS",
    "UNARY_METHODS",
    "pretty_callable_ref",
    "pretty_decl",
    "pretty_ref",
]
MAX_LINE_LENGTH = 110
LINE_DIFFERENCE = 10
BLACK_MODE = black.Mode(line_length=180)

# Use this special character in place of the args, so that if the args are inlined
# in the viz, they will replace it
ARG_STR = "·"

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

NAMED_UNARY_METHODS = {
    "__abs__": "abs",
    "__round__": "round",
    "__trunc__": "trunc",
    "__floor__": "floor",
    "__ceil__": "ceil",
}


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


def pretty_ref(root: Ref, *, custom_literal: Callable[[Ref, bool], str] | None = None) -> str:  # noqa: C901, PLR0912
    """Print records inertly unless the caller supplies a trusted Custom formatter."""
    from .runtime import _definition_at  # noqa: PLC0415

    # Frames stream output rather than recursively copying each child's string.
    # The Boolean asks scalar arguments to omit their inferred wrapper.
    frames: list[str | Ref | tuple[Ref, bool, bool] | tuple[Ref, pb.Action | pb.Command]] = [(root, False, False)]
    output: list[str] = []
    active: set[Ref] = set()
    while frames:
        frame = frames.pop()
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
            if reference in active:
                msg = "Cyclic expression presentation is not migrated"
                raise ValueError(msg)
            active.add(reference)
            frames.append(reference)
            record = reference.read()
        owner = reference.owner
        if isinstance(record, pb.Command) and record.kind is not None and record.kind.field == "action":
            record = record.kind.value
        pieces: list[str | tuple[Ref, bool, bool] | tuple[Ref, pb.Action | pb.Command]]
        if isinstance(record, pb.Command):
            if record.kind is None:
                msg_0 = "Cannot print a command without a kind"
                raise ValueError(msg_0)
            if record.kind.field == "run":
                selected = record.kind.value.ruleset
                if selected is None or selected.kind is None:
                    msg_0 = "Cannot print a Run without a ruleset"
                    raise ValueError(msg_0)
                if selected.kind.field == "index":
                    pieces = [(owner.ref("rulesets", selected.kind.value), False, False)]
                else:
                    pieces = [f"run({selected.kind.value})" if selected.kind.value else "run()"]
            elif record.kind.field in {"repeat", "saturate"}:
                control = record.kind.value
                if isinstance(control, pb.Repeat) and control.times == 1:
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
                    pieces = [record.kind.value]
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
                        from .runtime import RuntimeExpr  # noqa: PLC0415

                        value = get_literal_value(RuntimeExpr(reference))
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
                    elif primitive.field in {"vec", "set"}:
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
                    declared: pb.Declaration = declaration.read()
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
                        function_sort = owner.ref("sorts", args[0].read().sort_id)
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
                                owner.ref("sorts", argument.read().sort_id),
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
                            and StructuralView(selected[-1][0]) == StructuralView(defaults[-1])
                        ):
                            selected.pop()
                            defaults.pop()
                        name = view.path[-1] if view.path else "__init__"
                        if view.kind == pb.PythonCallKind.METHOD and name in UNARY_METHODS and not selected:
                            pieces = [UNARY_METHODS[name], (args[view.receiver], False, True)]
                            if parentheses:
                                pieces = ["(", *pieces, ")"]
                            frames.extend(reversed(pieces))
                            continue
                        if view.kind == pb.PythonCallKind.METHOD and name in BINARY_METHODS and len(selected) == 1:
                            pieces = [
                                (args[view.receiver], False, True),
                                f" {BINARY_METHODS[name]} ",
                                (selected[0][0], True, True),
                            ]
                            if parentheses:
                                pieces = ["(", *pieces, ")"]
                            frames.extend(reversed(pieces))
                            continue
                        if view.kind == pb.PythonCallKind.METHOD and name == "__getitem__" and len(selected) == 1:
                            pieces = [(args[view.receiver], False, True), "[", (selected[0][0], True, False), "]"]
                            frames.extend(reversed(pieces))
                            continue
                        if view.kind == pb.PythonCallKind.INITIALIZER:
                            pieces = [(sort, False, False), "("]
                        elif view.kind in {pb.PythonCallKind.METHOD, pb.PythonCallKind.PROPERTY}:
                            pieces = [(args[view.receiver], False, True), ".", name]
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
                    pieces.append(")")
                case "union":
                    pieces = ["eq("]
                    for index, argument in enumerate(record.kind.value.members):
                        if index:
                            pieces.append(").to(" if index == 1 else ", ")
                        pieces.append((owner.ref("nodes", argument), False, False))
                    pieces.append(")")
                case _:
                    raise NotImplementedError(f"Node presentation is not migrated: {record.kind.field}")
        elif isinstance(record, pb.RuleDecl):
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
        elif isinstance(record, pb.Ruleset):
            if record.kind is None:
                msg = "Cannot print a ruleset without a kind"
                raise ValueError(msg)
            if record.kind.field == "rules":
                pieces = ["ruleset("]
                for index, rule_index in enumerate(record.kind.value.rules):
                    if index:
                        pieces.append(", ")
                    pieces.append((owner.ref("rules", rule_index), False, False))
            else:
                pieces = ["unstable_combine_rulesets("]
                for index, child in enumerate(record.kind.value.rulesets):
                    if index:
                        pieces.append(", ")
                    if child.kind is None:
                        msg = "Cannot print a ruleset reference without a kind"
                        raise ValueError(msg)
                    if child.kind.field == "index":
                        pieces.append((owner.ref("rulesets", child.kind.value), False, False))
                    else:
                        pieces.append(child.kind.value)
            if record.has_field("name"):
                if len(pieces) > 1:
                    pieces.append(", ")
                pieces.append(f"name={record.name!r}")
            pieces.append(")")
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
            else:
                raise NotImplementedError(f"Action presentation is not migrated: {action.kind.field}")
        else:
            raise NotImplementedError(f"Record presentation is not migrated: {reference.role}")
        frames.extend(reversed(pieces))
    return "".join(output)


AllDecls: TypeAlias = (
    RulesetDecl
    | CombinedRulesetDecl
    | CommandDecl
    | ActionDecl
    | FactDecl
    | ExprDecl
    | ScheduleDecl
    | BackOffDecl
    | EGraphDecl
    | TypedExprDecl
)


def pretty_decl(
    decls: Declarations, decl: AllDecls, *, wrapping_fn: str | None = None, ruleset_ident: Ident | None = None
) -> str:
    """
    Pretty print a decleration.

    This will use re-format the result and put the expression on the last line, preceeded by the statements.
    """
    traverse = TraverseContext(decls)
    traverse(decl, toplevel=True)
    pretty = traverse.pretty()
    expr = pretty(decl, ruleset_ident=ruleset_ident)
    if wrapping_fn:
        expr = f"{wrapping_fn}({expr})"
    program = "\n".join([*pretty.statements, expr])
    # First unparse AST to get consistent formatting, then use black to format it nicely
    try:
        ast_tree = ast.parse(program, mode="exec")
    except SyntaxError:
        return program
    program = ast.unparse(ast_tree)
    try:
        # TODO: Try replacing with ruff for speed
        # https://github.com/amyreese/ruff-api
        return black.format_str(program, mode=BLACK_MODE).strip()
    except black.parsing.InvalidInput:
        return program


def pretty_callable_ref(
    decls: Declarations,
    ref: CallableRef,
    first_arg: TypedExprDecl | None = None,
    bound_tp_params: tuple[JustTypeRef, ...] | None = None,
    include_all_args: bool = False,
) -> str:
    """
    Pretty print a callable reference, using a dummy value for
    the args if the function is not in the form `f(x, ...)`.

    To be used in the visualization.
    """
    if ref == FunctionRef(Ident.builtin("!=")):
        return "!="

    # Pass in three dummy args, which are the max used for any operation that
    # is not a generic function call
    signature = decls.get_callable_decl(ref).signature

    def concrete_type(tp: TypeOrVarRef) -> JustTypeRef:
        match tp:
            case TypeVarRef():
                return JustTypeRef(Ident.builtin("Unit"))
            case TypeRefWithVars(ident, type_args):
                return JustTypeRef(ident, tuple(concrete_type(arg) for arg in type_args))

    if isinstance(signature, FunctionSignature):
        args = [TypedExprDecl(concrete_type(tp), UnboundVarDecl(ARG_STR)) for tp in signature.arg_types]
    else:
        args = [TypedExprDecl(JustTypeRef(Ident.builtin("Unit")), UnboundVarDecl(ARG_STR)) for _ in range(3)]
    while len(args) < 3:
        args.append(TypedExprDecl(JustTypeRef(Ident.builtin("Unit")), UnboundVarDecl(ARG_STR)))
    if first_arg is not None:
        args[0] = first_arg
    context = PrettyContext(decls, defaultdict(lambda: 0))
    res = context._call_inner(ref, args, bound_tp_params=bound_tp_params, parens=False)
    # Either returns a function or a function with args. If args are provided, they would just be called,
    # on the function, so return them, because they are dummies
    if isinstance(res, tuple):
        # If we want to include all args as ARG_STR, then we need to figure out how many to use
        # used for set_cost so that `cost(E(...))` will show up as a call
        if include_all_args:
            assert isinstance(signature, FunctionSignature)
            correct_args = args[: len(signature.arg_types)]
            return f"{res[0]}({', '.join(context(a, parens=False, unwrap_lit=res[2]) for a in correct_args)})"
        return res[0]
    return res


# TODO: Add a different pretty callable ref that doesnt fill in wholes but instead returns the function
# so that things like Math.__add__ will be represented properly


@dataclass
class TraverseContext:
    """
    State for traversing expressions (or declerations that contain expressions), so we can know how many parents each
    expression has.
    """

    decls: Declarations

    # All expressions we have seen (incremented the parent counts of all children)
    _seen: set[AllDecls] = field(default_factory=set)
    # The number of parents for each expressions
    parents: Counter[AllDecls] = field(default_factory=Counter)

    def pretty(self) -> PrettyContext:
        """
        Create a pretty context from the state of this traverse context.
        """
        return PrettyContext(self.decls, self.parents)

    def __call__(self, decl: AllDecls, toplevel: bool = False) -> None:  # noqa: C901, PLR0912
        if not toplevel:
            self.parents[decl] += 1
        if decl in self._seen:
            return
        match decl:
            case RewriteDecl(_, lhs, rhs, conditions) | BiRewriteDecl(_, lhs, rhs, conditions):
                self(lhs)
                self(rhs)
                for cond in conditions:
                    self(cond)
            case RuleDecl(head, body, _, _, _):
                for action in head:
                    self(action)
                for fact in body:
                    self(fact)
            case SetDecl(_, lhs, rhs) | UnionDecl(_, lhs, rhs) | EqDecl(_, lhs, rhs):
                self(lhs)
                self(rhs)
            case LetDecl(_, d) | ExprActionDecl(d) | ExprFactDecl(d):
                self(d)
            case ChangeDecl(_, d, _) | SaturateDecl(d) | RepeatDecl(d, _) | ActionCommandDecl(d):
                self(d)
            case PanicDecl(_) | UnboundVarDecl(_) | LetRefDecl(_) | LitDecl(_) | PyObjectDecl(_):
                pass
            case SequenceDecl(decls) | RulesetDecl(decls):
                for de in decls:
                    if isinstance(de, DefaultRewriteDecl):
                        continue
                    self(de)
            case CallDecl(ref, exprs, _) | GetCostDecl(ref, exprs):
                match ref:
                    case UnnamedFunctionRef(_, res):
                        self(res)
                    case _:
                        for e in exprs:
                            self(e)
            case RunDecl(_, until, scheduler):
                if until:
                    for f in until:
                        self(f)
                if scheduler:
                    self(scheduler)
            case PartialCallDecl(c):
                self(c)
            case CombinedRulesetDecl(_):
                pass
            case DefaultRewriteDecl():
                pass
            case SetCostDecl(_, e, c):
                self(e)
                self(c)
            case BackOffDecl() | ValueDecl():
                pass
            case LetSchedulerDecl(scheduler, schedule):
                self(scheduler)
                self(schedule)
            case DummyDecl():
                pass
            case EGraphDecl() as eg:
                for a in eg.to_actions:
                    self(a)
            case TypedExprDecl():
                self(decl.expr)
            case _:
                assert_never(decl)

        self._seen.add(decl)


@dataclass
class PrettyContext:
    """

    We need to build up a list of all the expressions we are pretty printing, so that we can see who has parents and who is mutated
    and create temp variables for them.

    """

    decls: Declarations
    parents: Mapping[AllDecls, int]

    # All the expressions we have saved as names
    names: dict[AllDecls, str] = field(default_factory=dict)
    # A list of statements assigning variables or calling destructive ops
    statements: list[str] = field(default_factory=list)
    # Mapping of type to the number of times we have generated a name for that type, used to generate unique names
    _gen_name_types: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def __call__(
        self, decl: AllDecls, *, unwrap_lit: bool = False, parens: bool = False, ruleset_ident: Ident | None = None
    ) -> str:
        if decl in self.names:
            return self.names[decl]
        expr, tp_name = self.uncached(decl, unwrap_lit=unwrap_lit, parens=parens, ruleset_ident=ruleset_ident)
        # We use a heuristic to decide whether to name this sub-expression as a variable.
        # The rough goal is to reduce the number of newlines, given our line length of ~180
        # We determine it's worth making a new line for this expression if the total characters
        # it would take up is > than some constant (~ line length).
        line_diff: int = len(expr) - LINE_DIFFERENCE
        n_parents = self.parents[decl]
        if n_parents > 1 and (
            n_parents * line_diff > MAX_LINE_LENGTH
            # Schedulers with multiple parents need to be the same object, b/c are created with hidden UUIDs
            or tp_name == "scheduler"
        ):
            self.names[decl] = expr_name = self._name_expr(tp_name, expr, copy_identifier=False)
            return expr_name
        return expr

    def uncached(  # noqa: C901, PLR0911, PLR0912
        self, decl: AllDecls, *, unwrap_lit: bool, parens: bool, ruleset_ident: Ident | None
    ) -> tuple[str, str]:
        """
        Returns a tuple of a string value of the declaration and the "type" to use when create a memoized cached version
        for deduplication.
        """
        match decl:
            case LitDecl(value):
                match value:
                    case None:
                        return "Unit()", "Unit"
                    case bool(b):
                        return str(b) if unwrap_lit else f"Bool({b})", "Bool"
                    case int(i):
                        return str(i) if unwrap_lit else f"i64({i})", "i64"
                    case float(f):
                        return str(f) if unwrap_lit else f"f64({f})", "f64"
                    case str(s):
                        return repr(s) if unwrap_lit else f"String({s!r})", "String"
                assert_never(value)
            case UnboundVarDecl(name) | LetRefDecl(name):
                return name, name
            case CallDecl(_, _, _):
                return self._call(decl, parens)
            case PartialCallDecl(CallDecl(ref, typed_args, _)):
                return self._pretty_partial(ref, list(typed_args), parens), "fn"
            case PyObjectDecl(pickled):
                value = cloudpickle.loads(pickled)
                value_str = repr(value)
                if not is_valid_python_expr(value_str):
                    # If this isn't a valid python expr, represent as string
                    value_str = f"eval({value_str!r})"
                return value_str if unwrap_lit else f"PyObject({value_str})", "PyObject"
            case ActionCommandDecl(action):
                return self(action), "action"
            case RewriteDecl(_, lhs, rhs, conditions) | BiRewriteDecl(_, lhs, rhs, conditions):
                args = ", ".join(map(self, (rhs, *conditions)))
                fn = "rewrite" if isinstance(decl, RewriteDecl) else "birewrite"
                return f"{fn}({self(lhs)}).to({args})", "rewrite"
            case RuleDecl(head, body, name, eval_mode, no_decomp):
                args = list(map(self, body))
                if name:
                    args.append(f"name={name!r}")
                if eval_mode != "seminaive":
                    args.append(f"eval_mode={eval_mode!r}")
                if no_decomp:
                    args.append("no_decomp=True")
                r = ", ".join(map(self, head))
                return f"rule({', '.join(args)}).then({r})", "rule"
            case SetDecl(_, lhs, rhs):
                return f"set_({self(lhs)}).to({self(rhs)})", "action"
            case UnionDecl(_, lhs, rhs):
                return f"union({self(lhs)}).with_({self(rhs)})", "action"
            case LetDecl(name, expr):
                return f"let({name!r}, {self(expr.expr)})", "action"
            case ExprActionDecl(expr):
                return self(expr.expr), "action"
            case ExprFactDecl(expr):
                return self(expr.expr), "fact"
            case ChangeDecl(_, expr, change):
                return f"{change}({self(expr)})", "action"
            case PanicDecl(s):
                return f"panic({s!r})", "action"
            case SetCostDecl(_, expr, cost):
                return f"set_cost({self(expr)}, {self(cost, unwrap_lit=True)})", "action"
            case EqDecl(_, left, right):
                return f"eq({self(left)}).to({self(right)})", "fact"
            case RulesetDecl(rules):
                if ruleset_ident:
                    return f"ruleset(name={ruleset_ident.name!r})", f"ruleset_{ruleset_ident.name}"
                args = ", ".join(self(r) for r in rules if not isinstance(r, DefaultRewriteDecl))
                return f"ruleset({args})", "ruleset"
            case CombinedRulesetDecl(rulesets):
                list_args = [r.name for r in rulesets]
                if ruleset_ident:
                    list_args.append(f"name={ruleset_ident.name!r})")
                return (f"unstable_combine_rulesets({', '.join(list_args)})", "combined_ruleset")
            case SaturateDecl(schedule):
                return f"{self(schedule, parens=True)}.saturate()", "schedule"
            case RepeatDecl(schedule, times):
                return f"{self(schedule, parens=True)} * {times}", "schedule"
            case SequenceDecl(schedules):
                if len(schedules) == 2:
                    return f"{self(schedules[0], parens=True)} + {self(schedules[1], parens=True)}", "schedule"
                args = ", ".join(map(self, schedules))
                return f"seq({args})", "schedule"
            case LetSchedulerDecl(scheduler, schedule):
                return f"{self(scheduler, parens=True)}.scope({self(schedule, parens=True)})", "schedule"
            case RunDecl(ruleset_ident, until, scheduler):
                if ruleset_ident.name == "" and ruleset_ident not in self.decls._rulesets:
                    ruleset_str = None
                else:
                    ruleset = self.decls._rulesets[ruleset_ident]
                    ruleset_str = self(ruleset, ruleset_ident=ruleset_ident)
                if not until and not scheduler:
                    return "run()" if ruleset_str is None else ruleset_str, "schedule"
                arg_lst = ["None" if ruleset_str is None else ruleset_str, *map(self, until or [])]
                if scheduler:
                    arg_lst.append(f"scheduler={self(scheduler)}")
                return f"run({', '.join(arg_lst)})", "schedule"
            case DefaultRewriteDecl():
                msg = "implicit rewrites should not be pretty printed"
                raise TypeError(msg)
            case BackOffDecl(_, match_limit, ban_length, persistent):
                list_args = []
                if match_limit is not None:
                    list_args.append(f"match_limit={match_limit}")
                if ban_length is not None:
                    list_args.append(f"ban_length={ban_length}")
                rendered = f"back_off({', '.join(list_args)})"
                if persistent:
                    rendered += ".persistent()"
                return rendered, "scheduler"
            case ValueDecl(value):
                return str(value), "value"
            case DummyDecl():
                return "__InternalDummyValueShouldNotBeSeenOpenAnIssue()", "dummy"
            case GetCostDecl(ref, args):
                return f"get_cost({self(CallDecl(ref, args))})", "get_cost"
            case EGraphDecl() as eg:
                return f"EGraph({', '.join(map(self, eg.to_actions))}).freeze()", "egraph"
            case TypedExprDecl(tp, expr):
                return (
                    self(expr, unwrap_lit=unwrap_lit, ruleset_ident=ruleset_ident, parens=parens),
                    tp.ident.name,
                )
        assert_never(decl)

    def _call(
        self,
        decl: CallDecl,
        parens: bool,
    ) -> tuple[str, str]:
        """
        Pretty print the call. Also returns if it was saved as a name.

        :param parens: If true, wrap the call in parens if it is a binary method call.
        """
        args = list(decl.args)
        ref = decl.callable
        # Special case !=
        if decl.callable == FunctionRef(Ident.builtin("!=")):
            l, r = self(args[0]), self(args[1])
            return f"ne({l}).to({r})", "Unit"
        signature = self.decls.get_callable_decl(ref).signature

        # Determine how many of the last arguments are defaults, by iterating from the end and comparing the arg with the default
        n_defaults = 0
        # Dont try counting defaults for function application
        if isinstance(signature, FunctionSignature):
            for arg, default in zip(
                reversed(args), reversed(signature.arg_defaults), strict=not signature.var_arg_type
            ):
                if arg.expr != default:
                    break
                n_defaults += 1
        if n_defaults:
            args = args[:-n_defaults]

        # If this is a function application, the type is the first type arg of the function object
        if signature == "fn-app":
            tp_name = decl.args[0].tp.args[0].ident.name
        else:
            assert isinstance(signature, FunctionSignature)
            tp_name = signature.semantic_return_type.ident.name
        if isinstance(signature, FunctionSignature) and signature.mutates:
            first_arg = args[0]
            expr_str = self(first_arg)
            # copy an identifier expression iff it has multiple parents (b/c then we can't mutate it directly)
            has_multiple_parents = self.parents[first_arg] > 1
            self.names[decl] = expr_name = self._name_expr(tp_name, expr_str, copy_identifier=has_multiple_parents)
            # Set the first arg to be the name of the mutated arg and return the name
            args[0] = TypedExprDecl(args[0].tp, LetRefDecl(expr_name))
        else:
            expr_name = None
        res = self._call_inner(ref, args, decl.bound_tp_params, parens)
        expr = (
            (f"{res[0]}({', '.join(self(a, parens=False, unwrap_lit=res[2]) for a in res[1])})")
            if isinstance(res, tuple)
            else res
        )
        # If we have a name, then we mutated
        if expr_name:
            self.statements.append(expr)
            return expr_name, tp_name
        return expr, tp_name

    def _call_inner(  # noqa: C901, PLR0911, PLR0912
        self,
        ref: CallableRef,
        args: list[TypedExprDecl],
        bound_tp_params: tuple[JustTypeRef, ...] | None,
        parens: bool,
    ) -> tuple[str, list[TypedExprDecl], bool] | str:
        """
        Pretty print the call, returning either the full function call or a tuple of the function and the args.
        """
        match ref:
            case FunctionRef(Ident(name)):
                return name, args, True
            case ClassMethodRef(class_name, method_name):
                tp_ref = JustTypeRef(class_name, bound_tp_params or ())
                # Generic calls retain wrappers so their class parameters remain inferable from the arguments.
                unwrap_lit = not self.decls.get_class_decl(class_name).type_vars
                return f"{tp_ref}.{method_name}", args, unwrap_lit
            case MethodRef(class_name, method_name):
                slf, *args = args
                non_str_slf = slf
                slf = self(slf, parens=True)
                cls_has_generic_params = self.decls.get_class_decl(class_name).type_vars
                # only unwrap literals if the class doesn't have generic params, otherwise we might lose type information that is needed for the method call
                unwrap_lit = not cls_has_generic_params
                match method_name:
                    case _ if method_name in UNARY_METHODS:
                        expr = f"{UNARY_METHODS[method_name]}{slf}"
                        return f"({expr})" if parens else expr
                    case _ if method_name in BINARY_METHODS:
                        expr = (
                            f"{slf} {BINARY_METHODS[method_name]} {self(args[0], parens=True, unwrap_lit=unwrap_lit)}"
                        )
                        return f"({expr})" if parens else expr
                    case "__getitem__":
                        return f"{slf}[{self(args[0], unwrap_lit=unwrap_lit)}]"
                    case "__call__":
                        return slf, args, unwrap_lit
                    case "__delitem__":
                        return f"del {slf}[{self(args[0], unwrap_lit=unwrap_lit)}]"
                    case "__setitem__":
                        return f"{slf}[{self(args[0], unwrap_lit=unwrap_lit)}] = {self(args[1], unwrap_lit=unwrap_lit)}"
                    case _ if method_name in NAMED_UNARY_METHODS:
                        return NAMED_UNARY_METHODS[method_name], [non_str_slf, *args], unwrap_lit
                    case "__getattr__" if isinstance(args[0].expr, LitDecl) and isinstance(args[0].expr.value, str):
                        return f"{slf}.{args[0].expr.value}"
                    case _:
                        return f"{slf}.{method_name}", args, unwrap_lit
            case ConstantRef(Ident(name)):
                return name
            case ClassVariableRef(Ident(class_name), variable_name):
                return f"{class_name}.{variable_name}"
            case PropertyRef(_class_name, property_name):
                return f"{self(args[0], parens=True)}.{property_name}"
            case InitRef(class_name):
                tp_ref = JustTypeRef(class_name, bound_tp_params or ())
                unwrap_lit = not self.decls.get_class_decl(class_name).type_vars
                return str(tp_ref), args, unwrap_lit
            case UnnamedFunctionRef():
                expr = self._pretty_function_body(ref, [])
                return f"({expr})", args, True
        assert_never(ref)

    def _generate_name(self, typ: str) -> str:
        self._gen_name_types[typ] += 1
        return f"_{typ}_{self._gen_name_types[typ]}"

    def _name_expr(self, tp_name: str, expr_str: str, copy_identifier: bool) -> str:
        # tp_name =
        # If the thing we are naming is already a variable, we don't need to name it
        if expr_str.isidentifier():
            if copy_identifier:
                name = self._generate_name(tp_name)
                self.statements.append(f"{name} = copy({expr_str})")
            else:
                name = expr_str
        else:
            name = self._generate_name(tp_name)
            self.statements.append(f"{name} = {expr_str}")
        return name

    def _pretty_partial(self, ref: CallableRef, args: list[TypedExprDecl], parens: bool) -> str:
        """
        Returns a partial function call as a string.
        """
        match ref:
            case FunctionRef(Ident(name)):
                fn = name
            case UnnamedFunctionRef():
                res = self._pretty_function_body(ref, args)
                return f"({res})" if parens else res
            case (
                ClassMethodRef(Ident(class_name), method_name)
                | MethodRef(Ident(class_name), method_name)
                | PropertyRef(Ident(class_name), method_name)
            ):
                fn = f"{class_name}.{method_name}"
            case InitRef(Ident(class_name)):
                fn = class_name
            case ConstantRef(_):
                msg = "Constants should not be callable"
                raise NotImplementedError(msg)
            case ClassVariableRef(_, _):
                msg = "Class variables should not be callable"
                raise NotADirectoryError(msg)
            case _:
                assert_never(ref)
        if not args:
            return fn
        arg_strs = (
            fn,
            *(self(a, parens=False, unwrap_lit=True) for a in args),
        )
        return f"partial({', '.join(arg_strs)})"

    def _pretty_function_body(self, fn: UnnamedFunctionRef, args: list[TypedExprDecl]) -> str:
        """
        Pretty print the body of a function, partially applying some arguments.
        """
        var_args = fn.args
        replacements = dict(zip(var_args, args, strict=False))
        var_args = var_args[len(args) :]
        res = replace_typed_expr(fn.res, replacements)
        arg_names = fn.args[len(args) :]
        prefix = "lambda"
        if arg_names:
            prefix += f" {', '.join(self(a) for a in arg_names)}"
        return f"{prefix}: {self(res)}"


def is_valid_python_expr(s: str) -> bool:
    try:
        ast.parse(s, mode="eval")
    except SyntaxError:
        return False
    return True
