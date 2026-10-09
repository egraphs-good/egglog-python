"""
Package an actual native catalog export without importing the Python frontend.

First run egglog-experimental's export_builtin_catalog example from the pinned
native checkout. This command consumes that binary snapshot; it never discovers
signatures through an EGraph during import or authoring. --check is a byte-for-
byte reproducibility gate, so changed native records cannot be silently hidden.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import runpy
from pathlib import Path
from typing import TYPE_CHECKING, cast

from egglog_proto.egglog.v1 import egglog_pb as pb

if TYPE_CHECKING:
    from egglog._builtin_protocols import ScalarProtocol


_PACKAGE = Path(__file__).resolve().parents[1] / "python/egglog"
_POLICY = runpy.run_path(str(_PACKAGE / "_builtin_protocols.py"))
_SCALARS = cast("dict[str, ScalarProtocol]", _POLICY["SCALAR_PROTOCOLS"])
_REFLECTED = cast("set[str]", _POLICY["NUMERIC_BINARY_METHODS"])


def render_stubs(program: pb.Program) -> str:  # noqa: C901, PLR0912
    """Derive the admitted public surface; never execute package/native code."""
    module = ("egglog", "builtins")
    families: dict[str, pb.HostSortFamily] = {}
    names: dict[str, str] = {}
    exported = {"ExprValueError"}
    reserved_names = {
        "AttributeError",
        "BaseExpr",
        "BuiltinExpr",
        "Generic",
        "Iterator",
        "Never",
        "TypeAlias",
        "TypeVar",
        "dataclass",
        "deprecated",
        "__all__",
        "object",
        "list",
        "dict",
        "set",
        "classmethod",
        "str",
        "tuple",
    }
    for protocol in _SCALARS.values():
        reserved_names.update(t.__name__ for t in (protocol.value_type, *protocol.literal_types))
        reserved_names.update(result.__name__ for _, result in protocol.hooks)

    def identifier(name: str) -> str:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Invalid Python identifier: {name!r}")
        return name

    for declaration in program.declarations:
        kind = declaration.kind
        if kind is None or kind.field != "host_sort_family":
            continue
        family = kind.value
        binding = family.bindings.python if family.bindings is not None else None
        path = tuple(binding.path) if binding is not None else (*module, family.name)
        if path[:-1] != module:
            raise ValueError(f"Cannot emit a foreign Python type path in builtins.pyi: {path}")
        name = identifier(path[-1])
        if name in reserved_names:
            raise ValueError(f"Python static name collision with a host/import name: {name}")
        if name in exported or family.name in families:
            raise ValueError(f"Python binding collision: {name}")
        if binding is not None and binding.type_params and len(binding.type_params) != family.arity:
            msg = "Family type parameter arity mismatch"
            raise ValueError(msg)
        families[family.name] = family
        names[family.name] = name
        exported.add(name)
        protocol = _SCALARS.get(family.name)
        if protocol is not None:
            if family.arity:
                msg = "Scalar codec requires a nullary family"
                raise ValueError(msg)
            if protocol.literal_types or protocol.conversion_sort:
                if name + "Like" in exported:
                    raise ValueError(f"Python binding collision: {name}Like")
                exported.add(name + "Like")

    variables: dict[str, list[str]] = {}
    used_names = exported | reserved_names
    for family_name, family in families.items():
        binding = family.bindings.python if family.bindings is not None else None
        labels = (
            list(binding.type_params)
            if binding is not None and binding.type_params
            else [f"T{i}" for i in range(family.arity)]
        )
        variables[family_name] = [f"_{names[family_name]}_{identifier(label)}_{i}" for i, label in enumerate(labels)]
        for variable in variables[family_name]:
            if variable in used_names:
                raise ValueError(f"Python static name collision with a generated TypeVar: {variable}")
            used_names.add(variable)

    def sort_type(
        index: int,
        *,
        input_type: bool = False,
        parameters: tuple[str, ...] = (),
        active: frozenset[int] = frozenset(),
    ) -> str:
        if index < 0 or index >= len(program.sorts) or index in active:
            msg = "Invalid or cyclic signature sort"
            raise ValueError(msg)
        kind = program.sorts[index].kind
        if kind is not None and kind.field == "var":
            if kind.value >= len(parameters):
                msg = "Unbound static signature parameter"
                raise ValueError(msg)
            return parameters[kind.value]
        if kind is None or kind.field != "family" or kind.value.name not in families:
            msg = "Static callable generation requires closed catalog family sorts"
            raise ValueError(msg)
        family = kind.value
        if len(family.args) != families[family.name].arity:
            msg = "Signature family arity mismatch"
            raise ValueError(msg)
        name = names[family.name]
        if family.args:
            return (
                name
                + "["
                + ", ".join(sort_type(i, parameters=parameters, active=active | {index}) for i in family.args)
                + "]"
            )
        protocol = _SCALARS.get(family.name)
        return (
            name + "Like"
            if input_type and protocol is not None and (protocol.literal_types or protocol.conversion_sort)
            else name
        )

    methods: dict[str, dict[str, str]] = {name: {} for name in families}
    functions: dict[str, str] = {}
    for declaration in program.declarations:
        if declaration.bindings is None or declaration.bindings.python is None:
            # These native definitions have no admitted Python callable surface
            # yet. Conservative free-wrapper generation is a separate runtime gap.
            continue
        if declaration.kind is None or declaration.kind.field != "host_primitive":
            msg = "Unsupported builtin static declaration"
            raise ValueError(msg)
        primitive = declaration.kind.value
        if primitive.typing is None or primitive.typing.field != "signature":
            msg = "Function application static generation is not supported"
            raise ValueError(msg)
        signature = primitive.typing.value
        if len(signature.varargs) > 1 or not signature.has_field("output"):
            msg = "Static callable generation requires a fixed concrete signature"
            raise ValueError(msg)
        for view in declaration.bindings.python.views:
            free = view.kind == pb.PythonCallKind.FUNCTION
            if (
                view.kind
                not in {
                    pb.PythonCallKind.FUNCTION,
                    pb.PythonCallKind.METHOD,
                    pb.PythonCallKind.INITIALIZER,
                    pb.PythonCallKind.PROPERTY,
                    pb.PythonCallKind.CLASS_METHOD,
                }
                or view.has_field("mutates")
                or len(view.path)
                != (len(module) + 1 if free else 0 if view.kind == pb.PythonCallKind.INITIALIZER else 1)
                or view.has_field("receiver") != (view.kind in {pb.PythonCallKind.METHOD, pb.PythonCallKind.PROPERTY})
                or (view.kind == pb.PythonCallKind.PROPERTY and view.params)
            ):
                msg = "Unsupported builtin Python static view"
                raise ValueError(msg)
            parameters: tuple[str, ...] = ()
            family_name = ""
            owner = ""
            if free:
                if view.owner is not None or tuple(view.path[:-1]) != module or signature.type_params:
                    msg = "Static free functions require an ownerless closed builtin view"
                    raise ValueError(msg)
            else:
                if view.owner is None or view.owner.kind is None or view.owner.kind.field != "sort":
                    msg = "Unsupported builtin Python static view"
                    raise ValueError(msg)
                if view.owner.kind.value >= len(program.sorts):
                    msg = "Invalid static owner sort"
                    raise ValueError(msg)
                owner_kind = program.sorts[view.owner.kind.value].kind
                if owner_kind is None or owner_kind.field != "family" or owner_kind.value.name not in families:
                    msg = "Static owner must be a catalog family"
                    raise ValueError(msg)
                family_name = owner_kind.value.name
                if signature.type_params:
                    if (
                        len(signature.type_params) != families[family_name].arity
                        or len(owner_kind.value.args) != len(signature.type_params)
                        or any(
                            index >= len(program.sorts)
                            or (argument := program.sorts[index].kind) is None
                            or argument.field != "var"
                            or argument.value != position
                            for position, index in enumerate(owner_kind.value.args)
                        )
                    ):
                        msg = "Generic static owner requires ordered supported family parameters"
                        raise ValueError(msg)
                    parameters = tuple(variables[family_name])
                elif owner_kind.value.args:
                    msg = "Generic static owner requires ordered supported family parameters"
                    raise ValueError(msg)
                owner = sort_type(view.owner.kind.value, parameters=parameters)
            output = sort_type(signature.output, parameters=parameters)
            receiver_slots = [view.receiver] if view.has_field("receiver") else []
            slots = [*receiver_slots, *(p.core_input for p in view.params if p.has_field("core_input"))]
            specialized = view.has_field("tail_repetitions")
            slot_count = len(signature.inputs) + bool(signature.varargs)
            if specialized:
                if view.kind not in {pb.PythonCallKind.FUNCTION, pb.PythonCallKind.METHOD}:
                    msg = "Tail specialization supports only function and method views"
                    raise ValueError(msg)
                if not signature.varargs:
                    msg = "Tail specialization requires a variadic host signature"
                    raise ValueError(msg)
                slot_count = len(signature.inputs) + len(signature.varargs) * view.tail_repetitions
            if (
                slot_count > 2**32 - 1
                or len(slots) != slot_count
                or len(slots) != len(receiver_slots) + len(view.params)
                or sorted(slots) != list(range(slot_count))
            ):
                msg = "Python view must cover each input exactly once"
                raise ValueError(msg)
            effective_inputs = list(signature.inputs)
            if specialized:
                effective_inputs.extend(signature.varargs * view.tail_repetitions)
            if view.has_field("receiver") and (
                view.receiver >= len(effective_inputs)
                or sort_type(effective_inputs[view.receiver].sort, parameters=parameters) != owner
            ):
                msg = "Python receiver must select a fixed input matching owner"
                raise ValueError(msg)
            if view.kind == pb.PythonCallKind.INITIALIZER and output != owner:
                msg = "Python initializer output differs from owner"
                raise ValueError(msg)
            name = "__init__" if view.kind == pb.PythonCallKind.INITIALIZER else identifier(view.path[-1])
            rendered_parameters = []
            bound_name = "" if free else "cls" if view.kind == pb.PythonCallKind.CLASS_METHOD else "self"
            parameter_names = {bound_name} if bound_name else set()
            has_default = False
            for parameter in view.params:
                parameter_name = identifier(parameter.name)
                if parameter_name in parameter_names:
                    msg = "Duplicate Python parameter name"
                    raise ValueError(msg)
                parameter_names.add(parameter_name)
                tail = bool(signature.varargs) and not specialized and parameter.core_input == len(signature.inputs)
                if tail and (parameter is not view.params[-1] or parameter.has_field("default_expr")):
                    msg = "A uniform tail must be the final parameter without a default"
                    raise ValueError(msg)
                expected_sort = signature.varargs[0].sort if tail else effective_inputs[parameter.core_input].sort
                annotation = sort_type(expected_sort, input_type=True, parameters=parameters)
                default = ""
                if parameter.has_field("default_expr"):
                    if (
                        signature.type_params
                        or (specialized and parameter.core_input >= len(signature.inputs))
                        or parameter.default_expr >= len(program.nodes)
                        or sort_type(program.nodes[parameter.default_expr].sort_id) != sort_type(expected_sort)
                    ):
                        msg = "Default root has an invalid sort or index"
                        raise ValueError(msg)
                    default = " = ..."
                    has_default = True
                elif has_default and not tail:
                    msg = "Required parameter follows a default"
                    raise ValueError(msg)
                rendered_parameters.append(f"{'*' if tail else ''}{parameter_name}: {annotation}{default}")
            prefix = "    @property\n" if view.kind == pb.PythonCallKind.PROPERTY else ""
            if view.kind == pb.PythonCallKind.CLASS_METHOD:
                prefix = "    @classmethod\n"
            result_type = "None" if view.kind == pb.PythonCallKind.INITIALIZER else output
            if free:
                if name in used_names:
                    raise ValueError(f"Python static name collision: {name}")
                exported.add(name)
                used_names.add(name)
                functions[name] = f"def {name}({', '.join(rendered_parameters)}) -> {result_type}: ...\n"
                continue
            generated = {
                name: prefix
                + f"    def {name}({bound_name}{', ' if rendered_parameters else ''}{', '.join(rendered_parameters)}) -> {result_type}: ..."
            }
            if name in _REFLECTED and family_name in _SCALARS:
                if (
                    (signature.varargs and not specialized)
                    or len(rendered_parameters) != 1
                    or output != owner
                    or any(sort_type(argument.sort) != owner for argument in effective_inputs)
                ):
                    msg = "Reflected static generation requires homogeneous scalar input/output"
                    raise ValueError(msg)
                if view.params[0].name != "other" or view.params[0].has_field("default_expr"):
                    msg = "Numeric forwarding requires one required 'other' parameter"
                    raise ValueError(msg)
                reverse = "__r" + name[2:]
                # Both RuntimeExpr numeric forwarders own this required argument.
                annotation = sort_type(effective_inputs[view.params[0].core_input].sort, input_type=True)
                generated[reverse] = f"    def {reverse}(self, other: {annotation}) -> {output}: ..."
            for method_name, text in generated.items():
                if method_name in methods[family_name]:
                    raise ValueError(f"Python member collision: {owner}.{method_name}")
                methods[family_name][method_name] = text

    vec_protocol = "Vec" in methods and "__init__" in methods["Vec"]
    if vec_protocol:
        alias = names["Vec"] + "Like"
        host_variable = f"_{names['Vec']}_HostT"
        if alias in used_names or host_variable in used_names:
            msg = "Python static name collision with a sequence protocol"
            raise ValueError(msg)
        exported.add(alias)
        used_names.update((alias, host_variable))
    map_protocol = "Map" in families
    set_protocol = "Set" in methods and "__init__" in methods["Set"]
    if set_protocol:
        set_alias = names["Set"] + "Like"
        set_host = f"_{names['Set']}_HostT"
        if set_alias in used_names or set_host in used_names:
            msg = "Python static name collision with a set protocol"
            raise ValueError(msg)
        exported.add(set_alias)
        used_names.update((set_alias, set_host))
    if map_protocol:
        if families["Map"].arity != 2:
            msg = "Map codec requires two family arguments"
            raise ValueError(msg)
        map_alias = names["Map"] + "Like"
        map_hosts = (f"_{names['Map']}_HostK", f"_{names['Map']}_HostV")
        if any(name in used_names for name in (map_alias, *map_hosts)):
            msg = "Python static name collision with a mapping protocol"
            raise ValueError(msg)
        exported.add(map_alias)
        used_names.update((map_alias, *map_hosts))
    lines = [
        "# Generated from the native catalog and Python host protocol policy. Do not edit.",
        "# ruff: noqa: N801, N816, PYI042 -- Preserve canonical Python presentation names.",
        "",
        *(["from collections.abc import Iterator"] if vec_protocol or map_protocol or set_protocol else []),
        "from dataclasses import dataclass",
        *(["from fractions import Fraction"] if any(name in families for name in ("Rational", "BigRat")) else []),
        "from typing import Generic, Never, TypeAlias, TypeVar",
        "",
        "from typing_extensions import deprecated",
        "",
        "from .egraph import BaseExpr, BuiltinExpr",
        "",
        "__all__ = " + json.dumps(sorted(exported)),
        "",
        "@dataclass",
        "class ExprValueError(AttributeError):",
        "    expr: object",
        "    allowed: str",
        "",
    ]
    for family_name in families:
        name = names[family_name]
        family_variables = variables[family_name]
        lines.extend(f'{variable} = TypeVar("{variable}", bound=BaseExpr)' for variable in family_variables)
        if family_variables:
            lines.append("")
        generic = f", Generic[{', '.join(family_variables)}]" if family_variables else ""
        lines.append(f"class {name}(BuiltinExpr{generic}):")
        protocol = _SCALARS.get(family_name)
        reserved = set()
        if protocol is None:
            # No initializer view: forbid construction without making the class
            # token abstract (convert accepts an ordinary type[V]).
            if "__init__" not in methods[family_name]:
                lines.extend([
                    "    # No Python initializer is supplied; this remains a usable type token.",
                    "    def __init__(self, _unavailable: Never, /) -> None: ...",
                ])
                reserved.add("__init__")
            if family_name == "Pair" and "__init__" in methods[family_name]:
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    f"    def value(self) -> tuple[{', '.join(family_variables)}]: ...",
                ])
                reserved.update({"value", "__match_args__"})
            elif (family_name == "Vec" and vec_protocol) or (family_name == "Set" and set_protocol):
                (element,) = family_variables
                value_type = f"tuple[{element}, ...]" if family_name == "Vec" else f"set[{element}]"
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    f"    def value(self) -> {value_type}: ...",
                    f"    def __iter__(self) -> Iterator[{element}]: ...",
                    "    def __len__(self) -> int: ...",
                    f"    def __contains__(self, item: {element}) -> bool: ...",
                    '    @deprecated("use .value")',
                    f"    def eval(self) -> {value_type}: ...",
                ])
                reserved.update({"value", "__match_args__", "__iter__", "__len__", "__contains__", "eval"})
            elif family_name == "Map":
                key, value = family_variables
                value_type = f"dict[{key}, {value}]"
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    f"    def value(self) -> {value_type}: ...",
                    f"    def __iter__(self) -> Iterator[{key}]: ...",
                    "    def __len__(self) -> int: ...",
                    f"    def __contains__(self, key: {key}) -> bool: ...",
                    '    @deprecated("use .value")',
                    f"    def eval(self) -> {value_type}: ...",
                ])
                reserved.update({"value", "__match_args__", "__iter__", "__len__", "__contains__", "eval"})
        else:
            value_type = protocol.value_type.__name__
            if protocol.literal_types or not protocol.literal_constructor:
                if protocol.literal_constructor:
                    lines.append(f"    def __init__(self, value: {value_type}, /) -> None: ...")
                    reserved.add("__init__")
                elif "__init__" not in methods[family_name]:
                    lines.append("    def __init__(self, _unavailable: Never, /) -> None: ...")
                    reserved.add("__init__")
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    f"    def value(self) -> {value_type}: ...",
                    '    @deprecated("use .value")',
                    f"    def eval(self) -> {value_type}: ...",
                ])
                reserved.update({"value", "eval", "__match_args__"})
            else:
                lines.append("    def __init__(self) -> None: ...")
                reserved.add("__init__")
            for hook, result in protocol.hooks:
                lines.append(f"    def {hook}(self) -> {result.__name__}: ...")
                reserved.add(hook)
        if reserved & methods[family_name].keys():
            raise ValueError(f"Python member collides with host protocol: {name}")
        lines.extend(methods[family_name].values())
        lines.append("")
        if family_name == "Vec" and vec_protocol:
            lines.extend([
                f'{host_variable} = TypeVar("{host_variable}")',
                f"{alias}: TypeAlias = {name}[{family_variables[0]}] | tuple[{host_variable}, ...] | list[{host_variable}]",
                "",
            ])
        if family_name == "Map" and map_protocol:
            lines.extend(f'{host} = TypeVar("{host}")' for host in map_hosts)
            lines.extend([
                f"{map_alias}: TypeAlias = {name}[{', '.join(family_variables)}] | dict[{', '.join(map_hosts)}]",
                "",
            ])
        if family_name == "Set" and set_protocol:
            lines.extend([
                f'{set_host} = TypeVar("{set_host}")',
                f"{set_alias}: TypeAlias = {name}[{family_variables[0]}] | set[{set_host}]",
                "",
            ])
        if protocol is not None and (protocol.literal_types or protocol.conversion_sort):
            converted = []
            if protocol.conversion_sort:
                if protocol.conversion_sort not in names:
                    msg = "Host conversion requires its catalog source family"
                    raise ValueError(msg)
                converted.append(names[protocol.conversion_sort] + "Like")
            lines.append(
                f"{name}Like: TypeAlias = "
                + " | ".join([name, *(t.__name__ for t in protocol.literal_types), *converted])
            )
            lines.append("")
    lines.extend(functions.values())
    result = "\n".join(lines)
    ast.parse(result, filename="builtins.pyi")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", type=Path, default=_PACKAGE / "_builtin_catalog.pb")
    parser.add_argument("--stubs-output", type=Path, default=_PACKAGE / "builtins.pyi")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = args.export.read_bytes()
    program = pb.Program.from_binary(encoded)
    if program.ir_version != 1 or program.commands or program.rules or program.rulesets:
        parser.error("Expected a version-1 declaration-only native catalog")
    names: set[tuple[str, str]] = set()
    for declaration in program.declarations:
        kind = declaration.kind
        if kind is None or kind.field not in {"host_sort_family", "host_primitive"}:
            parser.error("Native catalog may contain only host family/primitive declarations")
        namespace = "sort" if kind.field == "host_sort_family" else "callable"
        key = namespace, kind.value.name
        if not key[1] or key in names:
            parser.error(f"Duplicate or empty native definition: {key}")
        names.add(key)
    # Validate/render every artifact before replacing either output.
    try:
        stub = render_stubs(program)
    except ValueError as error:
        parser.error(str(error))
    if args.check:
        if args.output.read_bytes() != encoded:
            parser.error("Packaged catalog differs from the native export; regenerate it")
        if args.stubs_output.read_text() != stub:
            parser.error("Packaged builtin stubs differ from the native catalog/protocols; regenerate them")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
        args.stubs_output.parent.mkdir(parents=True, exist_ok=True)
        args.stubs_output.write_text(stub)
    print(f"{len(names)} native definitions; SHA-256 {hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
