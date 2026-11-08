"""
Package an actual native catalog export without importing the Python frontend.

Use --extension to load the compiled transport directly and request its strict
GetBuiltins catalog, including the actual Python-object provider. An existing
binary export is also accepted. The high-level egglog package is never imported.
--check is a byte-for-byte reproducibility gate, so changed native records cannot
be silently hidden.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import keyword
import runpy
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

from egglog_proto.egglog.v1 import egglog_pb as pb

if TYPE_CHECKING:
    from egglog._builtin_protocols import ScalarProtocol
    from egglog._presentation import CallKind as CallKindType
    from egglog._presentation import CallPresentation, Presentation, TypePresentation
    from egglog._program import Builder as SemanticBuilder
    from egglog._program import Owner, Ref


_PACKAGE = Path(__file__).resolve().parents[1] / "python/egglog"
_POLICY = runpy.run_path(str(_PACKAGE / "_builtin_protocols.py"))
_CODEGEN_PACKAGE = "_egglog_presentation_codegen"
if _CODEGEN_PACKAGE not in sys.modules:
    _namespace = ModuleType(_CODEGEN_PACKAGE)
    _namespace.__path__ = [str(_PACKAGE)]
    sys.modules[_CODEGEN_PACKAGE] = _namespace
_RECORDS = vars(importlib.import_module(f"{_CODEGEN_PACKAGE}._presentation"))
_GRAPH = vars(importlib.import_module(f"{_CODEGEN_PACKAGE}._program"))
_DATA = runpy.run_path(str(_PACKAGE / "_builtin_presentation.py"))["DATA"]
CallKind = cast("type[CallKindType]", _RECORDS["CallKind"])
Builder = cast("type[SemanticBuilder]", _GRAPH["Builder"])
_mapping_views = cast(
    "Callable[[pb.BuiltinCatalog, Owner, Mapping[str, object]], dict[tuple[str, str], Presentation]]",
    _RECORDS["mapping_views"],
)
_validate_view = cast("Callable[[Ref, CallPresentation], None]", _RECORDS["validate_view"])
_SCALARS = cast("dict[str, ScalarProtocol]", _POLICY["SCALAR_PROTOCOLS"])
_REFLECTED = cast("set[str]", _POLICY["NUMERIC_BINARY_METHODS"])


def render_builtins(program: pb.BuiltinCatalog, *, mapping: Mapping[str, object] | None = None) -> str:  # noqa: C901, PLR0912
    """Generate ordinary Python declarations from canonical native records."""
    # Reject dangling sort addresses before owned references are traversed.
    for sort in program.sorts:
        kind = sort.kind
        if kind is None:
            msg = "Invalid or cyclic signature sort"
            raise ValueError(msg)
        children = (
            kind.value.args
            if kind.field == "primitive"
            else [*kind.value.params, kind.value.result]
            if kind.field == "func"
            else ()
        )
        if any(type(index) is not int or not 0 <= index < len(program.sorts) for index in children):
            msg = "Invalid or cyclic signature sort"
            raise ValueError(msg)
    semantic_owner = Builder.from_builtins(program).publish()
    mapping = _DATA if mapping is None else mapping
    templates = Builder.from_program(
        pb.Program.from_json(json.dumps(mapping["fragment"])), environment=(semantic_owner,)
    ).publish()
    surface = _mapping_views(program, templates, mapping)
    for key, supplied in surface.items():
        reference = semantic_owner.ref(*semantic_owner._builtin_indices[key])
        for view in supplied.views:
            _validate_view(reference, view)
    family_views: dict[str, TypePresentation | None] = {}
    private_names = {
        "_collection_to_expr",
        "_container_items",
        "_dict_to_map",
        "_function_to_expr",
        "_map_value",
        "_maybe_value",
        "_none_to_maybe",
        "_pair_value",
        "_provided_default",
        "_tuple_to_pair",
    }
    module = ("egglog", "builtins")
    families: dict[str, pb.PrimitiveSort] = {}
    names: dict[str, str] = {}
    exported = {"ExprValueError"}
    reserved_names = {
        "AttributeError",
        "FunctionType",
        "MethodType",
        "partial",
        "expr_fact",
        "BaseExpr",
        "BuiltinExpr",
        "CallKind",
        "Generic",
        "Iterator",
        "Never",
        "TypeAlias",
        "TypeVar",
        "dataclass",
        "deprecated",
        "__all__",
        "object",
        "pb",
        "Ref",
        "RuntimeClass",
        "RuntimeExpr",
        "RuntimeFunction",
        "list",
        "dict",
        "set",
        "classmethod",
        "method",
        "function",
        "convert",
        "converter",
        "cast",
        "str",
        "tuple",
        *private_names,
    }
    for protocol in _SCALARS.values():
        reserved_names.update(t.__name__ for t in (protocol.value_type, *protocol.literal_types))
        reserved_names.update(result.__name__ for _, result in protocol.hooks)

    pyobject_protocol = any(declaration.name == "PyObject" for declaration in program.primitive_sorts)
    if pyobject_protocol:
        reserved_names.update({"Callable", "Protocol", "PyObjectFunction", "py_eval_fn"})
        exported.add("py_eval_fn")

    # Structural function application is a frontend protocol, not a builtin descriptor.
    function_protocol = True
    if function_protocol:
        # Python presentation policy for the structural owner, not a host family.
        exported.add("UnstableFn")
        reserved_names.update({
            "Callable",
            "TypeVarTuple",
            "Unpack",
            "overload",
            "_UnstableFn_T",
            "_UnstableFn_TS",
            "_UnstableFn_T1",
            "_UnstableFn_T2",
        })

    def identifier(name: str) -> str:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Invalid Python identifier: {name!r}")
        return name

    type_order = mapping.get("typeOrder", _DATA["typeOrder"])
    assert isinstance(type_order, list)
    order = {name: index for index, name in enumerate(type_order)}
    for family in sorted(program.primitive_sorts, key=lambda family: (order.get(family.name, len(order)), family.name)):
        binding = surface.get(("primitive_sort", family.name))
        binding = binding.type_view if binding is not None else None
        path = tuple(binding.path) if binding is not None else (*module, family.name)
        if path[:-1] != module:
            raise ValueError(f"Cannot emit a foreign Python type path in builtins.py: {path}")
        name = identifier(path[-1])
        if name in reserved_names:
            raise ValueError(f"Python static name collision with a host/import name: {name}")
        if name in exported or family.name in families:
            raise ValueError(f"Python binding collision: {name}")
        if binding is not None and binding.type_params and len(binding.type_params) != family.arity:
            msg = "Family type parameter arity mismatch"
            raise ValueError(msg)
        family_views[family.name] = binding
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
        binding = family_views[family_name]
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
        index: int | Ref,
        *,
        input_type: bool = False,
        parameters: tuple[str, ...] = (),
        used_parameters: set[int] | None = None,
        active: frozenset[Ref] = frozenset(),
    ) -> str:
        used_parameters = set() if used_parameters is None else used_parameters
        reference = semantic_owner.ref("sorts", index) if isinstance(index, int) else index
        if reference in active:
            msg = "Invalid or cyclic signature sort"
            raise ValueError(msg)
        kind = reference.read().kind
        if kind is not None and kind.field == "var":
            if kind.value < 0 or kind.value >= len(parameters):
                msg = "Unbound static signature parameter"
                raise ValueError(msg)
            used_parameters.add(kind.value)
            return parameters[kind.value]
        if kind is not None and kind.field == "func":
            function = kind.value
            if not function_protocol:
                msg = "Function sort requires an admitted structural function view"
                raise ValueError(msg)
            result = sort_type(
                reference.owner.ref("sorts", function.result),
                parameters=parameters,
                used_parameters=used_parameters,
                active=active | {reference},
            )
            args = [
                sort_type(
                    reference.owner.ref("sorts", i),
                    parameters=parameters,
                    used_parameters=used_parameters,
                    active=active | {reference},
                )
                for i in function.params
            ]
            if input_type:
                return f"Callable[[{', '.join(args)}], {result}]"
            return f"UnstableFn[{', '.join([result, *args])}]"
        if kind is None or kind.field != "primitive" or kind.value.name not in families:
            msg = "Static callable generation requires catalog family or function sorts"
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
                + ", ".join(
                    sort_type(
                        reference.owner.ref("sorts", i),
                        parameters=parameters,
                        used_parameters=used_parameters,
                        active=active | {reference},
                    )
                    for i in family.args
                )
                + "]"
            )
        protocol = _SCALARS.get(family.name)
        return (
            name + "Like"
            if input_type and protocol is not None and (protocol.literal_types or protocol.conversion_sort)
            else name
        )

    methods: dict[str, dict[str, str]] = {name: {} for name in families}
    method_variables: dict[str, list[str]] = {name: [] for name in families}
    functions: dict[str, str] = {}
    member_defaults: dict[Ref, tuple[str, str]] = {}
    free_defaults: dict[Ref, tuple[str, str]] = {}
    for primitive in sorted(program.primitives, key=lambda declaration: declaration.name):
        key = "callable", primitive.name
        if key not in surface or not surface[key].views:
            # These native definitions have no admitted Python callable surface
            # yet. Conservative free-wrapper generation is a separate runtime gap.
            continue
        signature = primitive.signature
        if signature is None:
            msg = "Builtin primitive requires its semantic signature"
            raise ValueError(msg)
        if len(signature.varargs) > 1 or not signature.has_field("output"):
            msg = "Static callable generation requires a fixed concrete signature"
            raise ValueError(msg)
        # Bound scalar counts by the actual referenced sort closure before
        # creating parameter names; a small malformed packet cannot request
        # range-sized frontend allocations.
        pending_sorts = [signature.output, *signature.inputs, *signature.varargs]
        seen_sorts = set()
        referenced_parameters = set()
        while pending_sorts:
            index = pending_sorts.pop()
            if index in seen_sorts:
                continue
            seen_sorts.add(index)
            kind = program.sorts[index].kind
            assert kind is not None
            if kind.field == "var":
                referenced_parameters.add(kind.value)
            elif kind.field == "primitive":
                pending_sorts.extend(kind.value.args)
            elif kind.field == "func":
                pending_sorts.extend([kind.value.result, *kind.value.params])
        if len(referenced_parameters) != signature.type_param_count or any(
            index >= signature.type_param_count for index in referenced_parameters
        ):
            msg = "Static signature requires all generic parameters in its sort closure"
            raise ValueError(msg)
        for view_index, view in enumerate(surface[key].views):
            free = view.kind == CallKind.FUNCTION
            if (
                view.kind
                not in {
                    CallKind.FUNCTION,
                    CallKind.METHOD,
                    CallKind.INITIALIZER,
                    CallKind.PROPERTY,
                    CallKind.CLASS_METHOD,
                }
                or (view.mutates is not None)
                or len(view.path) != (len(module) + 1 if free else 0 if view.kind == CallKind.INITIALIZER else 1)
                or (view.receiver is not None) != (view.kind in {CallKind.METHOD, CallKind.PROPERTY})
                or (view.kind == CallKind.PROPERTY and view.params)
            ):
                msg = "Unsupported builtin Python static view"
                raise ValueError(msg)
            parameters: tuple[str, ...] = ()
            used_parameters: set[int] = set()
            family_name = ""
            owner = ""
            name = "__init__" if view.kind == CallKind.INITIALIZER else identifier(view.path[-1])
            if free:
                if (view.owner is not None) or tuple(view.path[:-1]) != module:
                    msg = "Static free functions require an ownerless builtin view"
                    raise ValueError(msg)
                if signature.type_param_count and (signature.varargs or (view.tail_repetitions is not None)):
                    msg = "Generic static free functions require fixed signatures"
                    raise ValueError(msg)
                function_name = identifier(view.path[-1])
                parameters = tuple(f"_{function_name}_T{index}" for index in range(signature.type_param_count))
                for variable in parameters:
                    if variable in used_names:
                        raise ValueError(f"Python static name collision with a generated TypeVar: {variable}")
                    used_names.add(variable)
            else:
                if not (view.owner is not None):
                    msg = "Unsupported builtin Python static view"
                    raise ValueError(msg)
                owner_kind = view.owner.read().kind
                if owner_kind is None or owner_kind.field != "primitive" or owner_kind.value.name not in families:
                    msg = "Static owner must be a catalog family"
                    raise ValueError(msg)
                family_name = owner_kind.value.name
                owner_parameters: dict[int, str] = {}
                if len(owner_kind.value.args) != families[family_name].arity:
                    msg = "Generic static owner requires distinct in-range family parameters"
                    raise ValueError(msg)
                for position, index in enumerate(owner_kind.value.args):
                    if (
                        index < 0
                        or (argument := view.owner.owner.ref("sorts", index).read().kind) is None
                        or argument.field != "var"
                        or argument.value < 0
                        or argument.value >= signature.type_param_count
                        or argument.value in owner_parameters
                    ):
                        msg = "Generic static owner requires distinct in-range family parameters"
                        raise ValueError(msg)
                    owner_parameters[argument.value] = variables[family_name][position]
                parameters = tuple(
                    owner_parameters.get(index, f"_{names[family_name]}_{name}_T{index}")
                    for index in range(signature.type_param_count)
                )
                for index, variable in enumerate(parameters):
                    if index not in owner_parameters:
                        if variable in used_names:
                            raise ValueError(f"Python static name collision with a generated TypeVar: {variable}")
                        used_names.add(variable)
                        method_variables[family_name].append(variable)
                owner = sort_type(view.owner, parameters=parameters, used_parameters=used_parameters)
            output = sort_type(signature.output, parameters=parameters, used_parameters=used_parameters)
            receiver_slots = [view.receiver] if (view.receiver is not None) else []
            slots = [*receiver_slots, *(p.core_input for p in view.params)]
            specialized = view.tail_repetitions is not None
            consumes_tail = not specialized and (view.receiver is not None) and view.receiver == len(signature.inputs)
            if consumes_tail and (len(signature.varargs) != 1 or view.kind != CallKind.METHOD):
                msg = "Tail receiver requires a homogeneous variadic method"
                raise ValueError(msg)
            tail_slot = len(signature.inputs) + consumes_tail
            slot_count = tail_slot + bool(signature.varargs)
            if specialized:
                if view.kind not in {CallKind.FUNCTION, CallKind.METHOD}:
                    msg = "Tail specialization supports only function and method views"
                    raise ValueError(msg)
                if not signature.varargs:
                    msg = "Tail specialization requires a variadic host signature"
                    raise ValueError(msg)
                assert view.tail_repetitions is not None
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
                assert view.tail_repetitions is not None
                effective_inputs.extend(signature.varargs * view.tail_repetitions)
            elif consumes_tail:
                effective_inputs.extend(signature.varargs)
            if (view.receiver is not None) and (
                view.receiver >= len(effective_inputs)
                or sort_type(effective_inputs[view.receiver], parameters=parameters) != owner
            ):
                msg = "Python receiver must select a fixed input matching owner"
                raise ValueError(msg)
            if view.kind == CallKind.INITIALIZER and output != owner:
                msg = "Python initializer output differs from owner"
                raise ValueError(msg)
            rendered_parameters = []
            bound_name = "" if free else "cls" if view.kind == CallKind.CLASS_METHOD else "self"
            parameter_names = {bound_name} if bound_name else set()
            has_default = False
            for parameter in view.params:
                parameter_name = identifier(parameter.name)
                if parameter_name in parameter_names:
                    msg = "Duplicate Python parameter name"
                    raise ValueError(msg)
                parameter_names.add(parameter_name)
                tail = bool(signature.varargs) and not specialized and parameter.core_input == tail_slot
                if tail and (parameter is not view.params[-1] or (parameter.default is not None)):
                    msg = "A uniform tail must be the final parameter without a default"
                    raise ValueError(msg)
                expected_sort = signature.varargs[0] if tail else effective_inputs[parameter.core_input]
                annotation = sort_type(
                    expected_sort, input_type=True, parameters=parameters, used_parameters=used_parameters
                )
                # Uniform Vec tails accept the existing element-wise list/tuple
                # codec while retaining the same symbolic element type.
                expected_kind = program.sorts[expected_sort].kind
                if (
                    tail
                    and expected_kind is not None
                    and expected_kind.field == "primitive"
                    and expected_kind.value.name == "Vec"
                ):
                    element = sort_type(expected_kind.value.args[0], parameters=parameters)
                    annotation = f"{names['Vec']}Like[{element}, {element}]"
                default = ""
                if parameter.default is not None:
                    if (
                        signature.type_param_count
                        or (specialized and parameter.core_input >= len(signature.inputs))
                        or sort_type(parameter.default.owner.ref("sorts", parameter.default.read().sort_id))
                        != sort_type(expected_sort)
                    ):
                        msg = "Default root has an invalid sort or index"
                        raise ValueError(msg)
                    defaults = free_defaults if free else member_defaults
                    if parameter.default not in defaults:
                        default_name = f"_default_{'free' if free else 'member'}_{len(defaults)}"
                        node = parameter.default.read()
                        expression = None
                        if free and node.kind is not None and node.kind.field == "call" and not node.kind.value.args:
                            candidates = surface.get(("callable", node.kind.value.func))
                            for candidate in candidates.views if candidates is not None else ():
                                if candidate.kind == CallKind.CLASS_METHOD and candidate.owner is not None:
                                    target = sort_type(candidate.owner)
                                    expression = f"{target}.{candidate.path[0]}()"
                                    break
                        if expression is None:
                            expression = (
                                f"cast({annotation!r}, _provided_default({primitive.name!r}, "
                                f"{view_index}, {parameter.name!r}))"
                            )
                        defaults[parameter.default] = default_name, expression
                    default = f" = {defaults[parameter.default][0]}"
                    has_default = True
                elif has_default and not tail:
                    msg = "Required parameter follows a default"
                    raise ValueError(msg)
                rendered_parameters.append(f"{'*' if tail else ''}{parameter_name}: {annotation}{default}")
            reverse = ", reverse_args=True" if view.receiver is not None and view.receiver > 0 else ""
            # Reciprocal rich-comparison decorators cause a MyPy inference
            # cycle. BuiltinExpr already resolves these ordinary headers to the
            # exact admitted native family/member view, with the same checks.
            prefix = (
                ""
                if name in {"__lt__", "__le__", "__gt__", "__ge__"}
                else (f"    @method(egg_fn={primitive.name!r}{reverse})\n")
            )
            if view.kind == CallKind.PROPERTY:
                prefix = "    @property\n" + prefix
            if view.kind == CallKind.CLASS_METHOD:
                prefix += "    @classmethod\n"
            result_type = "None" if view.kind == CallKind.INITIALIZER else output
            if used_parameters != set(range(len(parameters))):
                msg = "Unused static signature parameter" + (" in static owner view" if not free else "")
                raise ValueError(msg)
            if free:
                if name in used_names:
                    raise ValueError(f"Python static name collision: {name}")
                exported.add(name)
                used_names.add(name)
                functions[name] = (
                    "".join(f'{variable} = TypeVar("{variable}", bound=BaseExpr)\n' for variable in parameters)
                    + ("\n" if parameters else "")
                    + f"@function(egg_fn={primitive.name!r}, builtin=True)\n"
                    + f"def {name}({', '.join(rendered_parameters)}) -> {result_type}: ...\n"
                )
                continue
            generated = {
                name: prefix
                + f"    def {name}({bound_name}{', ' if rendered_parameters else ''}{', '.join(rendered_parameters)}) -> {result_type}: ..."
            }
            if name in _REFLECTED and family_name in _SCALARS:
                if (
                    (signature.varargs and not specialized)
                    or len(rendered_parameters) != 1
                    or view.params[0].name != "other"
                    or (view.params[0].default is not None)
                ):
                    msg = "Numeric forwarding requires one required 'other' parameter"
                    raise ValueError(msg)
                # Forward calls use the declared signature. Reflected calls
                # first convert the left operand to self's sort before swapping.
                if (
                    name not in {"__lt__", "__le__", "__gt__", "__ge__"}
                    and output == owner
                    and all(sort_type(argument) == owner for argument in effective_inputs)
                ):
                    reverse = "__r" + name[2:]
                    annotation = sort_type(effective_inputs[view.params[0].core_input], input_type=True)
                    generated[reverse] = (
                        "    @method(preserve=True)\n"
                        f"    def {reverse}(self, other: {annotation}) -> {output}:\n"
                        f"        return convert(other, {owner}).{name}(self)"
                    )
            for method_name, text in generated.items():
                if method_name in methods[family_name]:
                    raise ValueError(f"Python member collision: {owner}.{method_name}")
                methods[family_name][method_name] = text

    for family_name, family_method_variables in method_variables.items():
        for variable in family_method_variables:
            if variable in methods[family_name]:
                raise ValueError(f"Python member collision with a generated TypeVar: {names[family_name]}.{variable}")

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
    multiset_protocol = "MultiSet" in methods and "__init__" in methods["MultiSet"]
    if multiset_protocol:
        multiset_alias = names["MultiSet"] + "Like"
        multiset_host = f"_{names['MultiSet']}_HostT"
        if multiset_alias in used_names or multiset_host in used_names:
            msg = "Python static name collision with a multiset protocol"
            raise ValueError(msg)
        exported.add(multiset_alias)
        used_names.update((multiset_alias, multiset_host))
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
    collections_imports = (["Callable"] if function_protocol or pyobject_protocol else []) + (
        ["Iterator"] if vec_protocol or map_protocol or set_protocol or multiset_protocol else []
    )
    lines = [
        "# Generated from the native catalog and Python host protocols. Do not edit.",
        '# mypy: disable-error-code="empty-body"',
        "# ruff: noqa: N801, N816, PYI042 -- Preserve canonical Python presentation names.",
        "",
        "from __future__ import annotations",
        "",
        *(["from collections.abc import " + ", ".join(collections_imports)] if collections_imports else []),
        *(["from fractions import Fraction"] if any(name in families for name in ("Rational", "BigRat")) else []),
        "from functools import partial",
        "from types import FunctionType, MethodType",
        "from typing import Generic, Never"
        + (", Protocol" if pyobject_protocol else "")
        + ", TypeAlias, TypeVar"
        + (", TypeVarTuple, Unpack" if function_protocol else "")
        + (
            ", cast"
            if member_defaults or any("cast(" in expression for _, expression in free_defaults.values())
            else ""
        )
        + (", overload" if function_protocol or any(p.conversion_sort for p in _SCALARS.values()) else ""),
        "",
        "from typing_extensions import deprecated",
        "",
        "from ._builtin_values import (",
        "    ExprValueError,",
        "    _collection_to_expr,",
        "    _container_items,",
        "    _dict_to_map,",
        "    _function_to_expr,",
        "    _map_value,",
        "    _maybe_value,",
        "    _none_to_maybe,",
        "    _pair_value,",
        *(
            ["    _provided_default,"]
            if member_defaults or any("_provided_default(" in expression for _, expression in free_defaults.values())
            else []
        ),
        "    _tuple_to_pair,",
        ")",
        "from .conversion import convert, converter",
        "from .egraph import BaseExpr, BuiltinExpr, expr_fact, function, method",
        "from .runtime import RuntimeExpr, RuntimeFunction",
        "",
        "__all__ = " + json.dumps(sorted(exported)),
        "",
    ]
    lines.extend(f"{name} = {expression}" for name, expression in member_defaults.values())
    if member_defaults:
        lines.append("")
    if function_protocol:
        lines.extend([
            '_UnstableFn_T = TypeVar("_UnstableFn_T", bound=BaseExpr)',
            '_UnstableFn_TS = TypeVarTuple("_UnstableFn_TS")',
            '_UnstableFn_T1 = TypeVar("_UnstableFn_T1")',
            '_UnstableFn_T2 = TypeVar("_UnstableFn_T2")',
            "",
            'class UnstableFn(BuiltinExpr, Generic[_UnstableFn_T, *_UnstableFn_TS], egg_sort="UnstableFn"):',
            '    __match_args__ = ("value",)',
            "    @overload",
            "    def __init__(self, f: Callable[[Unpack[_UnstableFn_TS]], _UnstableFn_T]) -> None: ...",
            "    @overload",
            "    def __init__(self, f: Callable[[_UnstableFn_T1, Unpack[_UnstableFn_TS]], _UnstableFn_T], _a: _UnstableFn_T1, /) -> None: ...",
            "    @overload",
            "    def __init__(self, f: Callable[[_UnstableFn_T1, _UnstableFn_T2, Unpack[_UnstableFn_TS]], _UnstableFn_T], _a: _UnstableFn_T1, _b: _UnstableFn_T2, /) -> None: ...",
            "    def __init__(self, f: Callable[..., BaseExpr], *partial: object) -> None: ...",
            "    def __call__(self, *args: *_UnstableFn_TS) -> _UnstableFn_T: ...",
            "    @property",
            "    @method(preserve=True)",
            "    def value(self) -> Callable[[Unpack[_UnstableFn_TS]], _UnstableFn_T]:",
            "        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle",
            "        value = get_literal_value(self)",
            "        if value is None:",
            '            raise ExprValueError(self, "UnstableFn(f, *args)")',
            "        return value",
            "    @method(preserve=True)",
            '    @deprecated("use .value")',
            "    def eval(self) -> Callable[[Unpack[_UnstableFn_TS]], _UnstableFn_T]:",
            "        return self.value",
            "",
        ])
    # A widening Like alias includes its symbolic conversion source. Emit that
    # source class first as well, so reciprocal operator inference sees its
    # already-checked methods rather than a deferred decorated class.
    class_order: list[str] = []
    pending = dict(families)
    while pending:
        ready = [
            family_name
            for family_name in pending
            if (protocol := _SCALARS.get(family_name)) is None or protocol.conversion_sort not in pending
        ]
        if not ready:
            msg = "Cyclic host conversion aliases"
            raise ValueError(msg)
        for family_name in ready:
            class_order.append(family_name)
            del pending[family_name]
    aliases: dict[str, list[str]] = {}
    for family_name in class_order:
        name = names[family_name]
        family_variables = variables[family_name]
        lines.extend(
            f'{variable} = TypeVar("{variable}", bound=BaseExpr)'
            for variable in [*family_variables, *method_variables[family_name]]
        )
        if family_variables or method_variables[family_name]:
            lines.append("")
        generic = f", Generic[{', '.join(family_variables)}]" if family_variables else ""
        lines.append(f"class {name}(BuiltinExpr{generic}, egg_sort={family_name!r}):")
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
                    "    @method(preserve=True)",
                    f"    def value(self) -> tuple[{', '.join(family_variables)}]:",
                    "        return _pair_value(self)",
                ])
                reserved.update({"value", "__match_args__"})
            elif family_name == "Maybe":
                if len(family_variables) != 1:
                    msg = "Maybe codec requires one family argument"
                    raise ValueError(msg)
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    "    @method(preserve=True)",
                    f"    def value(self) -> {family_variables[0]} | None:",
                    "        return _maybe_value(self)",
                ])
                reserved.update({"value", "__match_args__"})
            elif (
                (family_name == "Vec" and vec_protocol)
                or (family_name == "Set" and set_protocol)
                or (family_name == "MultiSet" and multiset_protocol)
            ):
                (element,) = family_variables
                value_type = (
                    f"tuple[{element}, ...]"
                    if family_name == "Vec"
                    else f"{'set' if family_name == 'Set' else 'list'}[{element}]"
                )
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    "    @method(preserve=True)",
                    f"    def value(self) -> {value_type}:",
                    "        return "
                    + (
                        "_container_items(self)"
                        if family_name == "Vec"
                        else f"{'set' if family_name == 'Set' else 'list'}(_container_items(self))"
                    ),
                    "    @method(preserve=True)",
                    f"    def __iter__(self) -> Iterator[{element}]:",
                    "        return iter(dict.fromkeys(_container_items(self)))"
                    if family_name == "Set"
                    else "        return iter(self.value)",
                    "    @method(preserve=True)",
                    "    def __len__(self) -> int:",
                    "        return len(self.value)",
                    "    @method(preserve=True)",
                    f"    def __contains__(self, item: {element}) -> bool:",
                    "        return item in self.value",
                    "    @method(preserve=True)",
                    '    @deprecated("use .value")',
                    f"    def eval(self) -> {value_type}:",
                    "        return self.value",
                ])
                reserved.update({"value", "__match_args__", "__iter__", "__len__", "__contains__", "eval"})
            elif family_name == "Map":
                key, value = family_variables
                value_type = f"dict[{key}, {value}]"
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    "    @method(preserve=True)",
                    f"    def value(self) -> {value_type}:",
                    "        return _map_value(self)",
                    "    @method(preserve=True)",
                    f"    def __iter__(self) -> Iterator[{key}]:",
                    "        return iter(self.value)",
                    "    @method(preserve=True)",
                    "    def __len__(self) -> int:",
                    "        return len(self.value)",
                    "    @method(preserve=True)",
                    f"    def __contains__(self, key: {key}) -> bool:",
                    "        return key in self.value",
                    "    @method(preserve=True)",
                    '    @deprecated("use .value")',
                    f"    def eval(self) -> {value_type}:",
                    "        return self.value",
                ])
                reserved.update({"value", "__match_args__", "__iter__", "__len__", "__contains__", "eval"})
        else:
            value_type = protocol.value_type.__name__
            if "__init__" in methods[family_name]:
                # RuntimeClass owns scalar codec/native dispatch; this is a real
                # Python typing header, not a second semantic initializer.
                initializer = methods[family_name]["__init__"].splitlines()[-1]
                initializer = initializer.replace(") -> None:", ", /) -> None:")
                if protocol.value_type.__name__ == "Fraction":
                    header = ast.parse("class _Header:\n" + initializer).body[0]
                    assert isinstance(header, ast.ClassDef)
                    declaration = header.body[0]
                    assert isinstance(declaration, ast.FunctionDef)
                    arguments = declaration.args.posonlyargs[1:]
                    if len(arguments) != 2 or any(argument.annotation is None for argument in arguments):
                        msg = "Fraction codec requires its two native initializer inputs"
                        raise ValueError(msg)
                    first, second = arguments
                    first_type = ast.unparse(first.annotation)
                    second_type = ast.unparse(second.annotation)
                    initializer = (
                        "    @overload\n"
                        "    def __init__(self, value: int | Fraction, /) -> None: ...\n"
                        "    @overload\n" + initializer + "\n"
                        f"    def __init__(self, {first.arg}: {first_type} | Fraction, "
                        f"{second.arg}: {second_type} | None = None, /) -> None: ..."
                    )
                methods[family_name]["__init__"] = initializer
            if protocol.literal_types:
                if protocol.conversion_sort is None:
                    lines.append(f"    def __init__(self, value: {value_type}, /) -> None: ...")
                    reserved.add("__init__")
                elif "__init__" not in methods[family_name]:
                    lines.append("    def __init__(self, _unavailable: Never, /) -> None: ...")
                    reserved.add("__init__")
                lines.extend([
                    '    __match_args__ = ("value",)',
                    "    @property",
                    "    @method(preserve=True)",
                    f"    def value(self) -> {value_type}:",
                ])
                if family_name == "PyObject":
                    lines.extend([
                        "        from .deconstruct import _pyobject_value  # noqa: PLC0415 -- materialize only on explicit inspection",
                        "        if not isinstance(self, RuntimeExpr):",
                        '            msg = "Expected a runtime expression"',
                        "            raise TypeError(msg)",
                        "        return _pyobject_value(self)",
                    ])
                else:
                    lines.extend([
                        "        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle",
                        "        value = get_literal_value(self)",
                        "        if value is None:",
                        f"            raise ExprValueError(self, {name!r})",
                        "        return value",
                    ])
                lines.extend([
                    "    @method(preserve=True)",
                    '    @deprecated("use .value")',
                    f"    def eval(self) -> {value_type}:",
                    "        return self.value",
                ])
                reserved.update({"value", "eval", "__match_args__"})
            else:
                lines.append("    def __init__(self) -> None: ...")
                reserved.add("__init__")
            for hook, result in protocol.hooks:
                lines.extend([
                    "    @method(preserve=True)",
                    f"    def {hook}(self) -> {result.__name__}:",
                    "        return bool(expr_fact(self))"
                    if family_name == "Unit"
                    else f"        return {result.__name__}(self.value)",
                ])
                reserved.add(hook)
        if reserved & methods[family_name].keys():
            raise ValueError(f"Python member collides with host protocol: {name}")
        lines.extend(methods[family_name].values())
        lines.append("")
        alias_lines = aliases[family_name] = []
        if family_name == "Vec" and vec_protocol:
            alias_lines.extend([
                f'{host_variable} = TypeVar("{host_variable}")',
                f"{alias}: TypeAlias = {name}[{family_variables[0]}] | tuple[{host_variable}, ...] | list[{host_variable}]",
                "",
            ])
        if family_name == "Map" and map_protocol:
            alias_lines.extend(f'{host} = TypeVar("{host}")' for host in map_hosts)
            alias_lines.extend([
                f"{map_alias}: TypeAlias = {name}[{', '.join(family_variables)}] | dict[{', '.join(map_hosts)}]",
                "",
            ])
        if family_name == "Set" and set_protocol:
            alias_lines.extend([
                f'{set_host} = TypeVar("{set_host}")',
                f"{set_alias}: TypeAlias = {name}[{family_variables[0]}] | set[{set_host}]",
                "",
            ])
        if family_name == "MultiSet" and multiset_protocol:
            alias_lines.extend([
                f'{multiset_host} = TypeVar("{multiset_host}")',
                f"{multiset_alias}: TypeAlias = {name}[{family_variables[0]}] | tuple[{multiset_host}, ...]",
                "",
            ])
        if protocol is not None and (protocol.literal_types or protocol.conversion_sort):
            converted = []
            if protocol.conversion_sort:
                if protocol.conversion_sort not in names:
                    msg = "Host conversion requires its catalog source family"
                    raise ValueError(msg)
                converted.append(names[protocol.conversion_sort] + "Like")
            alias_lines.append(
                f"{name}Like: TypeAlias = "
                + " | ".join([
                    name,
                    *(
                        t.__name__
                        for t in protocol.literal_types
                        if not protocol.conversion_sort or t not in _SCALARS[protocol.conversion_sort].literal_types
                    ),
                    *converted,
                ])
            )
            alias_lines.append("")
    # Method annotations are postponed; bind runtime aliases after all classes.
    for family_name in class_order:
        lines.extend(aliases[family_name])
    # Explicit registrations are generated from the same host-codec policy as
    # the declaration bodies. No runtime scan creates or exposes API members.
    lines.extend([
        "converter(RuntimeFunction, UnstableFn, _function_to_expr)",
        "converter(FunctionType, UnstableFn, _function_to_expr)",
        "converter(MethodType, UnstableFn, _function_to_expr)",
        "converter(partial, UnstableFn, _function_to_expr)",
    ])
    if "Pair" in names:
        lines.append(f"converter(tuple, {names['Pair']}, _tuple_to_pair)")
    if "Maybe" in names:
        lines.append(f"converter(type(None), {names['Maybe']}, _none_to_maybe)")
    for family_name, host_types in (("Vec", ("list", "tuple")), ("Set", ("set",)), ("MultiSet", ("tuple",))):
        if family_name in names:
            lines.extend(
                f"converter({host}, {names[family_name]}, lambda value: _collection_to_expr(value, {family_name!r}))"
                for host in host_types
            )
    if "Map" in names:
        lines.append(f"converter(dict, {names['Map']}, _dict_to_map)")
    for family_name, protocol in _SCALARS.items():
        if family_name not in names:
            continue
        name = names[family_name]
        for literal in protocol.literal_types:
            implementation = f"lambda value: {name}(float(value))" if protocol.value_type is float else name
            lines.append(f"converter({literal.__name__}, {name}, {implementation})")
        if protocol.conversion_sort is not None:
            implementation = name if protocol.value_type is int else f"lambda value: {name}(value, 1)"
            lines.append(f"converter({names[protocol.conversion_sort]}, {name}, {implementation})")
    lines.append("")
    lines.extend(f"{name} = {expression}" for name, expression in free_defaults.values())
    lines.append("")
    lines.extend(functions.values())
    if pyobject_protocol:
        name = names["PyObject"]
        lines.extend([
            "",
            "class PyObjectFunction(Protocol):",
            f"    def __call__(self, *__args: {name}) -> {name}: ...",
            "",
            '@deprecated("use PyObject(fn) directly")',
            "def py_eval_fn(fn: Callable) -> PyObjectFunction:",
            f"    return {name}(fn)",
        ])
    result = "\n".join(lines) + "\n"
    ast.parse(result, filename="builtins.py")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("export", type=Path, nargs="?")
    source.add_argument("--extension", type=Path, help="compiled bindings extension (no frontend import)")
    parser.add_argument("--output", type=Path, default=_PACKAGE / "_builtin_catalog.pb")
    parser.add_argument("--python-output", type=Path, default=_PACKAGE / "builtins.py")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.extension is None:
        encoded = args.export.read_bytes()
    else:
        loader = importlib.machinery.ExtensionFileLoader("bindings", str(args.extension.resolve()))
        spec = importlib.util.spec_from_file_location("bindings", args.extension, loader=loader)
        if spec is None:
            parser.error("Cannot load the compiled bindings extension")
        extension = importlib.util.module_from_spec(spec)
        loader.exec_module(extension)
        encoded = extension._ProtoEngine().get_builtins(pb.GetBuiltinsRequest().to_binary())
    program = pb.BuiltinCatalog.from_binary(encoded)
    names: set[tuple[str, str]] = set()
    declarations: list[pb.PrimitiveSort | pb.Primitive] = [*program.primitive_sorts, *program.primitives]
    for declaration in declarations:
        namespace = "sort" if isinstance(declaration, pb.PrimitiveSort) else "callable"
        key = namespace, declaration.name
        if not key[1] or key in names:
            parser.error(f"Duplicate or empty native definition: {key}")
        names.add(key)
    # Validate/render every artifact before replacing either output.
    try:
        source = render_builtins(program)
    except ValueError as error:
        parser.error(str(error))
    obsolete_stub = args.python_output.with_suffix(".pyi")
    if args.check:
        if obsolete_stub.exists():
            parser.error("Obsolete generated builtin stub shadows the Python declarations; regenerate them")
        if args.output.read_bytes() != encoded:
            parser.error("Packaged catalog differs from the native export; regenerate it")
        if args.python_output.read_text() != source:
            parser.error("Packaged builtin declarations differ from the native catalog/protocols; regenerate them")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
        args.python_output.parent.mkdir(parents=True, exist_ok=True)
        args.python_output.write_text(source)
        obsolete_stub.unlink(missing_ok=True)
    print(f"{len(names)} native definitions; SHA-256 {hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
