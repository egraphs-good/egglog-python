"""Python runtime wrappers; semantic state is owned generated protobuf records."""

from __future__ import annotations

import struct
from collections.abc import Callable
from contextlib import suppress
from inspect import Parameter, Signature
from threading import RLock
from types import MethodType
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin
from weakref import WeakKeyDictionary, WeakValueDictionary

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from ._builtin_protocols import NUMERIC_BINARY_METHODS, SCALAR_PROTOCOLS
from ._catalog import Catalog, builtin_catalog
from ._program import Builder, Owner, Ref, StructuralView, _definition_key, clone_nodes
from .type_constraint_solver import TypeConstraintError, infer_sort, substitute_sort

__all__ = [
    "ALWAYS_MUTATES_SELF",
    "ALWAYS_PRESERVED",
    "DUMMY_VALUE",
    "LIT_IDENTS",
    "NUMERIC_BINARY_METHODS",
    "RuntimeClass",
    "RuntimeExpr",
    "RuntimeFunction",
    "create_callable",
    "define_expr_method",
    "resolve_callable",
    "resolve_type_annotation",
]

# special methods that return none and mutate self
ALWAYS_MUTATES_SELF = {
    "__setitem__",
    "__delitem__",
    # "__setattr__",
}

# special methods which must return real python values instead of lazy expressions
ALWAYS_PRESERVED = {
    "__bytes__",
    # "__format__",
    "__hash__",
    "__bool__",
    "__len__",
    "__length_hint__",
    "__iter__",
    "__reversed__",
    "__contains__",
    "__index__",
    "__buffer__",
    "__complex__",
    "__int__",
    "__float__",
    "__str__",
    "__repr__",
}

# Methods that need to be defined on the runtime type that holds `Expr` objects, so that they can be used as methods.

TYPE_DEFINED_METHODS = {
    "__call__",
    "__getitem__",
    "__pos__",
    "__neg__",
    "__invert__",
    "__round__",
    "__abs__",
    "__trunc__",
    "__floor__",
    "__ceil__",
    *ALWAYS_PRESERVED,
    *ALWAYS_MUTATES_SELF,
} - {"__str__", "__repr__"}  # these are defined on RuntimeFunction

# Methods that need to be defined as descriptors on the class so that they can be looked up on the class itself.
CLASS_DESCRIPTOR_METHODS = {
    "__eq__",
    "__ne__",
    "__str__",
    "__repr__",
    "__getattr__",
    *NUMERIC_BINARY_METHODS,
    *TYPE_DEFINED_METHODS,
}

_OMITTED = object()


def _author_call(  # noqa: C901, PLR0912
    declaration: Ref,
    view_index: int | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    node_of: Callable[[object], Ref | None],
    convert: Callable[[Ref, object], Ref],
    receiver: object = _OMITTED,
    owner: Ref | None = None,
) -> tuple[Ref, object | None]:
    """
    Bind a Python call directly to its canonical declaration and node records.

    The callbacks are Python conversion policy, not signature providers. All
    parameter ordering, defaults, generic patterns and mutation selection come
    from the supplied immutable records. The return's second item identifies
    the supplied wrapper to replace for a mutating presentation.
    """
    if declaration.role != "declarations":
        msg = "Call authoring requires a declaration reference"
        raise TypeError(msg)
    record: pb.Declaration = declaration.read()
    if record.kind is None:
        msg = "Callable declaration has no kind"
        raise ValueError(msg)
    kind = record.kind.value
    binder_size = 0
    varargs: list[pb.Arg] = []
    match kind:
        case pb.HostPrimitive():
            if kind.typing is not None and kind.typing.field == "application":
                if view_index is None or record.bindings is None or record.bindings.python is None:
                    msg = "Function application requires its canonical Python view"
                    raise ValueError(msg)
                # Check the admitted structural owner, including mutation/tail
                # exclusions, through the same presentation index as imports.
                Catalog.from_refs((declaration,))
                application_view = record.bindings.python.views[view_index]
                if (
                    application_view.owner is None
                    or application_view.owner.kind is None
                    or application_view.owner.kind.field != "function"
                ):
                    msg = "Function application requires its structural function owner"
                    raise ValueError(msg)
                if receiver is _OMITTED:
                    if not args:
                        msg = "Missing receiver for unbound method"
                        raise TypeError(msg)
                    receiver, *remaining = args
                    args = tuple(remaining)
                first = receiver if isinstance(receiver, Ref) else node_of(receiver)
                if first is None:
                    msg = "Function application requires a function expression"
                    raise TypeError(msg)
                function_node: pb.Node = first.read()
                function_sort = first.owner.ref("sorts", function_node.sort_id)
                function_type: pb.Sort = function_sort.read()
                if function_type.kind is None or function_type.kind.field != "func":
                    msg = "Function application requires a Func sort"
                    raise TypeError(msg)
                infer_sort(function_sort, function_sort, 0, {})
                parameter = application_view.params[0]
                bound = Signature([Parameter(parameter.name, Parameter.VAR_POSITIONAL)]).bind(*args, **kwargs)
                supplied = bound.arguments.get(parameter.name, ())
                assert isinstance(supplied, tuple)
                if len(supplied) != len(function_type.kind.value.params):
                    raise TypeError(f"Function application expects {len(function_type.kind.value.params)} arguments")
                converted = [first]
                for index, value in zip(function_type.kind.value.params, supplied, strict=True):
                    expected = function_sort.owner.ref("sorts", index)
                    child = value if isinstance(value, Ref) else node_of(value)
                    if child is None or StructuralView(
                        child.owner.ref("sorts", child.read().sort_id)
                    ) != StructuralView(expected):
                        child = convert(expected, value)
                    infer_sort(expected, child.owner.ref("sorts", child.read().sort_id), 0, {})
                    converted.append(child)
                builder = Builder()
                builder.import_ref(declaration)
                node_index = builder.add(
                    "nodes",
                    pb.Node(
                        sort_id=builder.import_ref(function_sort.owner.ref("sorts", function_type.kind.value.result)),
                        kind=Oneof[Literal["call"], pb.Call](
                            "call", pb.Call(func=kind.name, args=[builder.import_ref(node) for node in converted])
                        ),
                    ),
                )
                return builder.publish().ref("nodes", node_index), None
            if kind.typing is None or kind.typing.field != "signature":
                msg = "Host primitive has no typing form"
                raise ValueError(msg)
            signature = kind.typing.value
            if not signature.has_field("output"):
                msg = "Host signature has no output"
                raise ValueError(msg)
            inputs, output = signature.inputs, signature.output
            binder_size = len(signature.type_params)
            varargs = signature.varargs
        case pb.Constructor() | pb.Function() | pb.Primitive():
            inputs, output = kind.inputs, kind.output
        case _:
            msg = f"Call authoring for {record.kind.field} is not migrated yet"
            raise NotImplementedError(msg)
    if view_index is None:
        # Conservative derived free-function binding: no synthetic metadata is
        # persisted or installed. The generator checks its exported path.
        view = pb.PythonCallable(
            kind=pb.PythonCallKind.FUNCTION,
            path=[kind.name],
            params=[pb.PythonParameter(core_input=i, name=arg.name or f"arg{i}") for i, arg in enumerate(inputs)],
        )
        if varargs:
            view.params.append(pb.PythonParameter(core_input=len(inputs), name=varargs[0].name or "args"))
    else:
        if record.bindings is None or record.bindings.python is None:
            msg = "Requested Python view is absent"
            raise ValueError(msg)
        view = record.bindings.python.views[view_index]
    specialized = view.has_field("tail_repetitions")
    slot_count = len(inputs) + bool(varargs)
    if specialized:
        if view.kind not in {pb.PythonCallKind.FUNCTION, pb.PythonCallKind.METHOD}:
            msg = "Tail specialization supports only function and method views"
            raise ValueError(msg)
        if not varargs:
            msg = "Tail specialization requires a variadic host signature"
            raise ValueError(msg)
        if any(not parameter.has_field("core_input") for parameter in view.params):
            msg = "Python view parameter is missing its core input"
            raise ValueError(msg)
        slot_count = len(inputs) + len(varargs) * view.tail_repetitions
        if slot_count > 2**32 - 1 or slot_count != len(view.params) + view.has_field("receiver"):
            msg = "Tail specialization must map every effective input exactly once"
            raise ValueError(msg)
        if any(p.core_input >= len(inputs) and p.has_field("default_expr") for p in view.params) or (
            view.has_field("mutates") and view.mutates >= len(inputs)
        ):
            msg = "Specialized tail inputs cannot have defaults or mutation"
            raise ValueError(msg)
    substitutions: dict[int, Ref] = {}
    if owner is not None and view.owner is not None and view.owner.kind is not None and view.owner.kind.field == "sort":
        infer_sort(declaration.owner.ref("sorts", view.owner.kind.value), owner, binder_size, substitutions)
    core_values: dict[int, object] = {}
    if view.has_field("receiver"):
        if receiver is _OMITTED:
            if not args:
                msg = "Missing receiver for unbound method"
                raise TypeError(msg)
            receiver, *remaining = args
            args = tuple(remaining)
        core_values[view.receiver] = receiver
    elif receiver is not _OMITTED:
        msg = "This Python view does not accept a receiver"
        raise TypeError(msg)
    parameters = [
        Parameter(
            parameter.name,
            Parameter.VAR_POSITIONAL
            if varargs and not specialized and parameter.core_input == len(inputs)
            else Parameter.POSITIONAL_OR_KEYWORD,
            default=_OMITTED if parameter.has_field("default_expr") else Parameter.empty,
        )
        for parameter in view.params
    ]
    bound = Signature(parameters).bind(*args, **kwargs)
    bound.apply_defaults()
    omitted = [parameter for parameter in view.params if bound.arguments[parameter.name] is _OMITTED]
    defaults = iter(clone_nodes(declaration.owner.ref("nodes", parameter.default_expr) for parameter in omitted))
    for parameter in view.params:
        if parameter.core_input in core_values:
            msg = "Python view maps one core input more than once"
            raise ValueError(msg)
        value = bound.arguments[parameter.name]
        core_values[parameter.core_input] = next(defaults) if value is _OMITTED else value
    expected_slots = set(range(slot_count))
    if core_values.keys() != expected_slots:
        msg = "Python view does not map every core input exactly once"
        raise ValueError(msg)
    ordered_values = [core_values[index] for index in range(slot_count if specialized else len(inputs))]
    patterns = [declaration.owner.ref("sorts", arg.sort) for arg in inputs]
    if varargs:
        tail = tuple(ordered_values[len(inputs) :]) if specialized else core_values[len(inputs)]
        assert isinstance(tail, tuple)
        if len(tail) % len(varargs):
            raise TypeError(f"Variadic arguments must form complete groups of {len(varargs)}")
        if not specialized:
            ordered_values.extend(tail)
        patterns.extend(
            declaration.owner.ref("sorts", arg.sort) for _ in range(len(tail) // len(varargs)) for arg in varargs
        )
    if (
        view.has_field("receiver")
        and view.owner is not None
        and view.owner.kind is not None
        and view.owner.kind.field == "sort"
        and StructuralView(patterns[view.receiver])
        != StructuralView(declaration.owner.ref("sorts", view.owner.kind.value))
    ):
        msg = "Python receiver input must match its owner"
        raise ValueError(msg)
    existing = [value if isinstance(value, Ref) else node_of(value) for value in ordered_values]
    for pattern, node in zip(patterns, existing, strict=True):
        if node is not None:
            payload: pb.Node = node.read()
            # A registered Python conversion may still satisfy this input.
            with suppress(TypeConstraintError):
                infer_sort(pattern, node.owner.ref("sorts", payload.sort_id), binder_size, substitutions)
    converted = []
    for pattern, value, original_node in zip(patterns, ordered_values, existing, strict=True):
        node = original_node
        expected = substitute_sort(pattern, substitutions)
        if node is not None:
            payload = node.read()
            try:
                infer_sort(expected, node.owner.ref("sorts", payload.sort_id), 0, {})
            except TypeConstraintError:
                node = None
        if node is None:
            node = convert(expected, value)
            payload = node.read()
            infer_sort(expected, node.owner.ref("sorts", payload.sort_id), 0, {})
        converted.append(node)
    result_sort = substitute_sort(declaration.owner.ref("sorts", output), substitutions)
    if len(substitutions) != binder_size:
        msg = "Not every generic parameter was determined by the call"
        raise TypeConstraintError(msg)
    builder = Builder()
    builder.import_ref(declaration)
    node_index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(result_sort),
            kind=Oneof[Literal["call"], pb.Call](
                "call", pb.Call(func=kind.name, args=[builder.import_ref(node) for node in converted])
            ),
        ),
    )
    return builder.publish().ref("nodes", node_index), core_values[view.mutates] if view.has_field("mutates") else None


# These are Python codecs/protocol hooks, not callable signatures. Sort and
# callable definitions still come from the native-generated catalog.
LIT_IDENTS = frozenset(name for name, protocol in SCALAR_PROTOCOLS.items() if protocol.literal_constructor)
DUMMY_VALUE = object()
_CLASS_CACHE: WeakKeyDictionary[Owner, WeakValueDictionary[int, RuntimeClass]] = WeakKeyDictionary()
_BASE_CLASSES: dict[str, RuntimeClass] = {}


class _ClassContext:
    """Owner-retained Python lifecycle/hooks; all declaration semantics remain refs."""

    def __init__(
        self, prepare: Callable[[RuntimeClass], Catalog], hooks: dict[str, Any], match_args: tuple[str, ...]
    ) -> None:
        self.prepare: Callable[[RuntimeClass], Catalog] | None = prepare
        self.hooks = hooks
        self.match_args = match_args
        self.runtime_class: RuntimeClass
        self.catalog: Catalog | None = None
        self.resolving = False
        self.error: Exception | None = None
        self.lock = RLock()

    def resolve(self) -> None:
        """Complete one class attempt, retaining its immutable published prefix."""
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.resolving:
                if self.catalog is not None:
                    return
                msg = "Recursively resolving class declarations"
                raise RuntimeError(msg)
            if self.prepare is None:
                return
            self.resolving = True
            try:
                catalog = self.prepare(self.runtime_class)
            except Exception as error:
                if self.catalog is None:
                    self.error = error
                raise
            else:
                self.catalog = catalog
            finally:
                self.prepare = None
                self.resolving = False


def _definition_at(origin: Ref, namespace: str, name: str) -> Ref:
    """Resolve owned definitions before consulting the ambient native catalog."""
    matches: set[Ref] = set()
    for index in range(len(origin.owner._slots["declarations"])):
        candidate = origin.owner.ref("declarations", index)
        if _definition_key(candidate.read()) == (namespace, name):
            matches.add(candidate)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous local definition: {namespace} {name!r}")
    if matches:
        return next(iter(matches))
    try:
        return builtin_catalog().definitions[namespace, name]
    except KeyError as exc:
        raise LookupError(f"No canonical definition for {namespace} {name!r}") from exc


def _class_for_sort(sort: Ref) -> RuntimeClass:
    """Recover a Python wrapper from its canonical concrete sort record."""
    sort = sort.owner.ref("sorts", sort.index)
    cache = _CLASS_CACHE.setdefault(sort.owner, WeakValueDictionary())
    if (existing := cache.get(sort.index)) is not None:
        return existing
    record: pb.Sort = sort.read()
    if record.kind is None:
        msg = "Expression sort is missing its kind"
        raise ValueError(msg)
    match record.kind.field:
        case "family":
            name = record.kind.value.name
            namespace = "host_sort_family"
        case "eq":
            name = record.kind.value
            namespace = "eq_sort"
        case "func":
            infer_sort(sort, sort, 0, {})
            definition, _ = builtin_catalog().members[("function", ""), "__call__"]
            result = RuntimeClass(definition, sort=sort)
            cache[sort.index] = result
            return result
        case _:
            msg = "Function/pattern runtime class projection is not migrated yet"
            raise NotImplementedError(msg)
    definition = _definition_at(sort, namespace, name)
    for context in definition.owner._keepalive:
        if isinstance(context, _ClassContext):
            return context.runtime_class
    result = RuntimeClass(definition, sort=sort)
    cache[sort.index] = result
    return result


class RuntimeClassDescriptor:
    """Expose generated special methods on a runtime class, not its metaclass."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, owner: RuntimeClass | None = None) -> object:
        if owner is None:
            raise AttributeError(self.name)
        return RuntimeClass.__getattr__(owner, self.name)


class RuntimeClass(type):
    """A Python type presentation over an owned sort/family declaration."""

    def __new__(cls, definition: Ref, *, sort: Ref | None = None, arguments: tuple[object, ...] = ()) -> RuntimeClass:  # noqa: C901, PLR0912
        declaration: pb.Declaration = definition.read()
        kind = declaration.kind
        structural_function = (
            kind is not None
            and isinstance(kind.value, pb.HostPrimitive)
            and kind.value.typing is not None
            and kind.value.typing.field == "application"
        )
        if structural_function:
            catalog = Catalog.from_refs((definition,))
            if (("function", ""), "__call__") not in catalog.members:
                msg = "Function class requires a canonical Python application view"
                raise ValueError(msg)
            binding = None
            path = ("egglog", "builtins", "UnstableFn")
        elif kind is not None and isinstance(kind.value, pb.HostSortFamily | pb.EqSort):
            binding = kind.value.bindings
            path = (
                tuple(binding.python.path)
                if binding is not None and binding.python is not None
                else ("egglog", "builtins", kind.value.name)
            )
        else:
            msg = "RuntimeClass requires a canonical sort declaration"
            raise TypeError(msg)
        assert kind is not None
        if not path:
            msg = "An empty type path cannot be generated as a Python class"
            raise ValueError(msg)
        namespace = {
            "__module__": ".".join(path[:-1]),
            "__doc__": declaration.doc if declaration.has_field("doc") else None,
            **{name: RuntimeClassDescriptor(name) for name in CLASS_DESCRIPTOR_METHODS},
        }
        result = type.__new__(cls, path[-1], (), namespace)
        result.__egg_definition__ = definition
        result.__egg_sort__ = sort
        result.__egg_arguments__ = arguments
        result.__egg_attr_cache__ = {}
        result.__egg_hooks__ = {}
        result.__egg_pending__ = None
        result.__egg_context__ = next(
            (context for context in definition.owner._keepalive if isinstance(context, _ClassContext)), None
        )
        if result.__egg_context__ is not None:
            result.__egg_context__.runtime_class = result
            result.__egg_hooks__ = result.__egg_context__.hooks
            result.__egg_pending__ = result.__egg_context__.resolve
        if structural_function:
            if sort is None and not arguments:
                _BASE_CLASSES["UnstableFn"] = result
            elif "UnstableFn" in _BASE_CLASSES:
                result.__egg_hooks__ = _BASE_CLASSES["UnstableFn"].__egg_hooks__
        elif sort is None and not arguments:
            arity = kind.value.arity if isinstance(kind.value, pb.HostSortFamily) else 0
            if arity:
                labels = binding.python.type_params if binding is not None and binding.python is not None else []
                result.__egg_arguments__ = tuple(
                    TypeVar(labels[index] if labels else f"T{index}") for index in range(arity)
                )
            else:
                builder = Builder()
                builder.import_ref(definition)
                sort_kind = (
                    Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=kind.value.name))
                    if isinstance(kind.value, pb.HostSortFamily)
                    else Oneof[Literal["eq"], str]("eq", kind.value.name)
                )
                index = builder.add("sorts", pb.Sort(kind=sort_kind))
                result.__egg_sort__ = builder.publish().ref("sorts", index)
            if isinstance(kind.value, pb.HostSortFamily):
                _BASE_CLASSES[kind.value.name] = result
        if sort is not None and isinstance(kind.value, pb.HostSortFamily) and kind.value.name in _BASE_CLASSES:
            result.__egg_hooks__ = _BASE_CLASSES[kind.value.name].__egg_hooks__
            result.__egg_pending__ = _BASE_CLASSES[kind.value.name].__egg_pending__
        if result.__egg_sort__ is not None:
            actual = result.__egg_sort__
            _CLASS_CACHE.setdefault(actual.owner, WeakValueDictionary())[actual.index] = result
        return result

    def __init__(cls, definition: Ref, *, sort: Ref | None = None, arguments: tuple[object, ...] = ()) -> None:
        # type.__init__ expects (name, bases, namespace), not semantic records.
        pass

    __egg_definition__: Ref
    __egg_sort__: Ref | None
    __egg_arguments__: tuple[object, ...]
    __egg_attr_cache__: dict[str, object]
    __egg_hooks__: dict[str, Any]
    __egg_pending__: Callable[[], None] | None
    __egg_context__: _ClassContext | None

    def __instancecheck__(cls, instance: object) -> bool:
        if not isinstance(instance, RuntimeExpr):
            return False
        node: pb.Node = instance.__egg_ref__.read()
        actual = _class_for_sort(instance.__egg_ref__.owner.ref("sorts", node.sort_id))
        return cls.__egg_definition__ == actual.__egg_definition__

    def __call__(cls, *args: object, **kwargs: object) -> RuntimeExpr | None:  # noqa: C901, PLR0912
        declaration: pb.Declaration = cls.__egg_definition__.read()
        assert declaration.kind is not None
        name = declaration.kind.value.name
        if isinstance(declaration.kind.value, pb.HostPrimitive):
            if kwargs or not args:
                msg = "UnstableFn requires a callable and positional prefix arguments"
                raise TypeError(msg)
            function, *captures = args
            if isinstance(function, MethodType) and isinstance(function.__self__, RuntimeExpr):
                for member_name, member in RuntimeExpr.__dict__.items():
                    if member is function.__func__:
                        function = function.__self__.__getattr__(member_name)
                        break
            if isinstance(function, RuntimeClass):
                function = RuntimeClass.__getattr__(function, "__init__")
            if not isinstance(function, RuntimeFunction):
                msg = "UnstableFn requires a named RuntimeFunction"
                raise TypeError(msg)
            sort, children, _ = _partial_function(function, tuple(captures), cls.__egg_sort__)
            record: pb.Declaration = function.__egg_ref__.read()
            assert record.kind is not None
            builder = Builder()
            builder.import_ref(function.__egg_ref__)
            node = builder.add(
                "nodes",
                pb.Node(
                    sort_id=builder.import_ref(sort),
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value",
                        pb.PrimitiveValue(
                            value=Oneof[Literal["partial_call"], pb.Call](
                                "partial_call",
                                pb.Call(
                                    func=record.kind.value.name,
                                    args=[builder.import_ref(child.__egg_ref__) for child in children],
                                ),
                            )
                        ),
                    ),
                ),
            )
            return RuntimeExpr(builder.publish().ref("nodes", node))
        if isinstance(declaration.kind.value, pb.HostSortFamily) and name in LIT_IDENTS:
            if kwargs or len(args) != (0 if name == "Unit" else 1):
                raise TypeError(f"{cls} expects {'zero' if name == 'Unit' else 'one'} positional literal argument")
            if args and not isinstance(args[0], SCALAR_PROTOCOLS[name].literal_types):
                raise TypeError(f"{cls} expects a {SCALAR_PROTOCOLS[name].value_type.__name__} literal")
            if cls.__egg_sort__ is None:
                msg = "Literal constructor has an unresolved sort"
                raise TypeError(msg)
            primitive = pb.PrimitiveValue()
            match SCALAR_PROTOCOLS[name].literal_arm:
                case "i64":
                    primitive.value = Oneof[Literal["i64"], int]("i64", cast("int", args[0]))
                case "f64_bits":
                    bits = struct.unpack("!Q", struct.pack("!d", args[0]))[0]
                    primitive.value = Oneof[Literal["f64_bits"], int]("f64_bits", bits)
                case "string":
                    primitive.value = Oneof[Literal["string"], str]("string", cast("str", args[0]))
                case "bool":
                    primitive.value = Oneof[Literal["bool"], bool]("bool", cast("bool", args[0]))
                case "unit":
                    primitive.value = Oneof[Literal["unit"], pb.Unit]("unit", pb.Unit())
            builder = Builder()
            builder.import_ref(cls.__egg_definition__)
            index = builder.add(
                "nodes",
                pb.Node(
                    sort_id=builder.import_ref(cls.__egg_sort__),
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue]("primitive_value", primitive),
                ),
            )
            return RuntimeExpr(builder.publish().ref("nodes", index))
        if cls.__egg_pending__ is not None:
            cls.__egg_pending__()
        return cast("RuntimeFunction", RuntimeClass.__getattr__(cls, "__init__"))(*args, **kwargs)

    def __getitem__(cls, arguments: object) -> RuntimeClass:
        values = arguments if isinstance(arguments, tuple) else (arguments,)
        declaration: pb.Declaration = cls.__egg_definition__.read()
        assert declaration.kind is not None
        kind = declaration.kind.value
        structural_function = (
            isinstance(kind, pb.HostPrimitive) and kind.typing is not None and kind.typing.field == "application"
        )
        if (structural_function and not values) or (
            not structural_function and (not isinstance(kind, pb.HostSortFamily) or len(values) != kind.arity)
        ):
            raise TypeError(f"Wrong number of type arguments for {cls}")
        if any(isinstance(value, TypeVar) for value in values):
            return RuntimeClass(cls.__egg_definition__, arguments=values)
        if not all(isinstance(value, RuntimeClass) and value.__egg_sort__ is not None for value in values):
            msg = "Type arguments must resolve to canonical sorts"
            raise TypeError(msg)
        builder = Builder()
        builder.import_ref(cls.__egg_definition__)
        indices = [builder.import_ref(cast("Ref", cast("RuntimeClass", value).__egg_sort__)) for value in values]
        index = builder.add(
            "sorts",
            pb.Sort(
                kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=indices[1:], result=indices[0]))
            )
            if structural_function
            else pb.Sort(
                kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=kind.name, args=indices))
            ),
        )
        return _class_for_sort(builder.publish().ref("sorts", index))

    def __getattr__(cls, name: str) -> object:  # noqa: C901, PLR0911
        if name.startswith("__egg_"):
            raise AttributeError(name)
        if name == "__args__":
            if cls.__egg_sort__ is None:
                return cls.__egg_arguments__
            record: pb.Sort = cls.__egg_sort__.read()
            assert record.kind is not None
            if record.kind.field == "func":
                return tuple(
                    _class_for_sort(cls.__egg_sort__.owner.ref("sorts", index))
                    for index in [record.kind.value.result, *record.kind.value.params]
                )
            return (
                tuple(_class_for_sort(cls.__egg_sort__.owner.ref("sorts", index)) for index in record.kind.value.args)
                if record.kind.field == "family"
                else ()
            )
        if name == "__origin__":
            if cls.__egg_arguments__:
                declaration: pb.Declaration = cls.__egg_definition__.read()
                assert declaration.kind is not None
                return _BASE_CLASSES[
                    "UnstableFn"
                    if isinstance(declaration.kind.value, pb.HostPrimitive)
                    else declaration.kind.value.name
                ]
            # typing probes a concrete class's origin while resolving another
            # signature; absence must not recursively compile either class.
            raise AttributeError(name)
        if name in cls.__egg_hooks__:
            return cls.__egg_hooks__[name]
        # Even a cached prefix member must wait for another thread's active
        # attempt. Same-thread reentry uses the already-published declarations.
        if cls.__egg_pending__ is not None:
            cls.__egg_pending__()
        if name in cls.__egg_attr_cache__:
            return cls.__egg_attr_cache__[name]
        declaration = cls.__egg_definition__.read()
        assert declaration.kind is not None
        definition_key = _definition_key(declaration)
        assert definition_key is not None
        key = (("function", "") if isinstance(declaration.kind.value, pb.HostPrimitive) else definition_key), name
        catalog = builtin_catalog()
        if cls.__egg_context__ is not None:
            catalog = cls.__egg_context__.catalog
            assert catalog is not None
        elif cls.__egg_definition__ != catalog.definitions.get(definition_key):
            raise AttributeError(f"{cls} has no canonical Python member {name!r}")
        try:
            reference, view = catalog.members[key]
        except KeyError as exc:
            raise AttributeError(f"{cls} has no generated Python member {name!r}") from exc
        function = RuntimeFunction(reference, view, cls)
        payload: pb.Declaration = reference.read()
        assert payload.bindings is not None
        assert payload.bindings.python is not None
        result = (
            function() if payload.bindings.python.views[view].kind == pb.PythonCallKind.CLASS_VARIABLE else function
        )
        cls.__egg_attr_cache__[name] = result
        return result

    def __str__(cls) -> str:
        if cls.__egg_sort__ is not None:
            from .pretty import pretty_ref  # noqa: PLC0415

            return pretty_ref(cls.__egg_sort__)
        return cls.__name__

    def __repr__(cls) -> str:
        return str(cls)

    def __hash__(cls) -> int:
        return (
            hash(StructuralView(cls.__egg_sort__))
            if cls.__egg_sort__ is not None
            else hash((cls.__egg_definition__, cls.__egg_arguments__))
        )

    def __eq__(cls, other: object) -> bool:
        if not isinstance(other, RuntimeClass):
            return NotImplemented
        if cls.__egg_sort__ is not None and other.__egg_sort__ is not None:
            return StructuralView(cls.__egg_sort__) == StructuralView(other.__egg_sort__)
        return cls.__egg_definition__ == other.__egg_definition__ and cls.__egg_arguments__ == other.__egg_arguments__

    def __or__(cls, other: object) -> Any:
        return Union[cls, other]  # noqa: UP007 -- runtime union of language class values

    @property
    def __parameters__(cls) -> tuple[object, ...]:
        return cls.__egg_arguments__

    @property
    def __match_args__(cls) -> tuple[str, ...]:
        if cls.__egg_context__ is not None:
            return cls.__egg_context__.match_args
        return ("value",) if "value" in cls.__egg_hooks__ else ()


class RuntimeFunctionMeta(type):
    def __getattribute__(cls, name: str) -> object:
        # Instance module metadata comes from its record, not this Python type.
        if name == "__module__":
            return __name__
        return super().__getattribute__(name)


class RuntimeFunction(metaclass=RuntimeFunctionMeta):
    """A selected language view over one owned canonical callable declaration."""

    def __init__(
        self,
        declaration: Ref | Callable[[], Ref],
        view_index: int | None = None,
        bound: RuntimeClass | RuntimeExpr | None = None,
    ) -> None:
        self.__egg_declaration__ = declaration if isinstance(declaration, Ref) else None
        self.__egg_prepare__ = declaration if callable(declaration) else None
        self.__egg_error__: Exception | None = None
        self.__egg_resolving__ = False
        self.__egg_lock__ = RLock()
        self.__egg_view__ = view_index
        self.__egg_bound__ = bound

    @property
    def __egg_ref__(self) -> Ref:
        """Materialize raw Python source once, publishing only a complete record."""
        with self.__egg_lock__:
            if self.__egg_declaration__ is not None:
                return self.__egg_declaration__
            if self.__egg_error__ is not None:
                raise self.__egg_error__.with_traceback(None)
            if self.__egg_resolving__:
                msg = "Recursively resolving function declarations"
                raise ValueError(msg)
            assert self.__egg_prepare__ is not None
            self.__egg_resolving__ = True
            try:
                reference = self.__egg_prepare__()
                if not isinstance(reference, Ref) or reference.role != "declarations":
                    msg = "Function source must publish a canonical declaration"
                    raise TypeError(msg)  # noqa: TRY301 -- invalid preparation is a terminal cached error
                self.__egg_declaration__ = reference.owner.ref(reference.role, reference.index)
            except Exception as error:
                self.__egg_error__ = error
                raise
            finally:
                self.__egg_prepare__ = None
                self.__egg_resolving__ = False
            return self.__egg_declaration__

    @property
    def __module__(self) -> str | None:  # type: ignore[override]
        record: pb.Declaration = self.__egg_ref__.read()
        if self.__egg_view__ is None or record.bindings is None or record.bindings.python is None:
            return None
        view = record.bindings.python.views[self.__egg_view__]
        if view.owner is not None and view.owner.kind is not None and view.owner.kind.field == "sort":
            return _class_for_sort(self.__egg_ref__.owner.ref("sorts", view.owner.kind.value)).__module__
        return ".".join(view.path[:-1]) or None

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        record: pb.Declaration = self.__egg_ref__.read()
        return record.doc if record.has_field("doc") else None

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        from .conversion import resolve_literal  # noqa: PLC0415

        bound = self.__egg_bound__
        owner = bound.__egg_sort__ if isinstance(bound, RuntimeClass) else None
        receiver = bound if isinstance(bound, RuntimeExpr) else _OMITTED
        root, mutated = _author_call(
            self.__egg_ref__,
            self.__egg_view__,
            args,
            kwargs,
            node_of=lambda value: value.__egg_ref__ if isinstance(value, RuntimeExpr) else None,
            convert=lambda sort, value: resolve_literal(sort, value).__egg_ref__,
            receiver=receiver,
            owner=owner,
        )
        if mutated is not None:
            if isinstance(mutated, RuntimeExpr):
                mutated.__egg_ref__ = root
            return None
        return RuntimeExpr(root)

    def __str__(self) -> str:
        record: pb.Declaration = self.__egg_ref__.read()
        assert record.kind is not None
        if self.__egg_view__ is None:
            return record.kind.value.name
        assert record.bindings is not None
        assert record.bindings.python is not None
        view = record.bindings.python.views[self.__egg_view__]
        if view.kind == pb.PythonCallKind.FUNCTION:
            return view.path[-1]
        name = "__init__" if view.kind == pb.PythonCallKind.INITIALIZER else view.path[0]
        return f"{self.__egg_bound__}.{name}"

    def __repr__(self) -> str:
        return str(self)

    def _identity(self) -> tuple[object, ...]:
        """Compare callable records or named user functions, without binding."""
        reference = self.__egg_ref__
        record: pb.Declaration = reference.read()
        if (
            record.kind is not None
            and record.kind.field in {"function", "constructor"}
            and self.__egg_view__ is not None
            and record.bindings is not None
            and record.bindings.python is not None
        ):
            view = record.bindings.python.views[self.__egg_view__]
            if view.kind == pb.PythonCallKind.FUNCTION:
                return "function", tuple(view.path)
        if record.kind is not None and record.kind.field == "host_primitive":
            return StructuralView(reference), self.__egg_view__
        return reference, self.__egg_view__

    def __hash__(self) -> int:
        return hash((*self._identity(), self.__egg_bound__))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeFunction):
            return NotImplemented
        return self._identity() == other._identity() and bool(self.__egg_bound__ == other.__egg_bound__)


def _partial_function(  # noqa: C901, PLR0912
    function: RuntimeFunction, captures: tuple[object, ...], expected: Ref | None, *, convert_captures: bool = True
) -> tuple[Ref, tuple[RuntimeExpr, ...], Ref | None]:
    """
    Type a named prefix against its complete call and preserve its Python order.

    Construction and inert decoding share this boundary: selected presentations
    must preserve every core position, and mutation is rejected before conversion.
    All inference operates on the actual canonical records, never a dummy call.
    Decoded captures are validated as stored values; only authoring may convert.
    """
    from .conversion import resolve_literal  # noqa: PLC0415

    declaration = function.__egg_ref__
    record: pb.Declaration = declaration.read()
    if record.kind is None:
        msg = "Function value callable has no kind"
        raise ValueError(msg)
    kind = record.kind.value
    view = None
    if function.__egg_view__ is not None:
        if record.bindings is None or record.bindings.python is None:
            msg = "Function value Python view is absent"
            raise ValueError(msg)
        view = record.bindings.python.views[function.__egg_view__]
        if view.has_field("mutates"):
            msg = "Mutable partial functions not supported"
            raise AssertionError(msg)
    bound = function.__egg_bound__
    if isinstance(bound, RuntimeExpr):
        captures = (bound, *captures)
    expected_params: list[Ref] | None = None
    expected_result = None
    if expected is not None:
        expected_type: pb.Sort = expected.read()
        if expected_type.kind is None or expected_type.kind.field != "func":
            msg = "Function value requires a Func sort"
            raise TypeError(msg)
        infer_sort(expected, expected, 0, {})
        expected_params = [expected.owner.ref("sorts", index) for index in expected_type.kind.value.params]
        expected_result = expected.owner.ref("sorts", expected_type.kind.value.result)
    substitutions: dict[int, Ref] = {}
    binder_size = 0
    tail: list[Ref] = []
    application = False
    if isinstance(kind, pb.HostPrimitive):
        if kind.typing is None:
            msg = "Function value callable has no typing form"
            raise ValueError(msg)
        if kind.typing.field == "signature":
            signature = kind.typing.value
            if not signature.has_field("output"):
                msg = "Function value signature has no output"
                raise ValueError(msg)
            binder_size = len(signature.type_params)
            patterns = [declaration.owner.ref("sorts", argument.sort) for argument in signature.inputs]
            tail = [declaration.owner.ref("sorts", argument.sort) for argument in signature.varargs]
            output = declaration.owner.ref("sorts", signature.output)
        else:
            application = True
            if captures and isinstance(captures[0], RuntimeExpr):
                first = captures[0].__egg_ref__
                function_sort = first.owner.ref("sorts", first.read().sort_id)
            elif expected_params:
                function_sort = expected_params[0]
            else:
                msg = "Partial application needs its concrete function input sort"
                raise TypeConstraintError(msg)
            actual: pb.Sort = function_sort.read()
            if actual.kind is None or actual.kind.field != "func":
                msg = "Partial application input must have a Func sort"
                raise TypeConstraintError(msg)
            infer_sort(function_sort, function_sort, 0, {})
            patterns = [function_sort]
            tail = [function_sort.owner.ref("sorts", index) for index in actual.kind.value.params]
            output = function_sort.owner.ref("sorts", actual.kind.value.result)
    elif isinstance(kind, pb.Constructor | pb.Function | pb.Primitive):
        patterns = [declaration.owner.ref("sorts", argument.sort) for argument in kind.inputs]
        output = declaration.owner.ref("sorts", kind.output)
    else:
        msg = "Function value requires a named value callable"
        raise TypeError(msg)
    fixed = len(patterns)
    specialized = view is not None and view.has_field("tail_repetitions")
    if application:
        patterns.extend(tail)
    elif tail:
        total = (
            len(captures) + len(expected_params)
            if expected_params is not None
            else fixed + len(tail) * view.tail_repetitions
            if specialized and view is not None
            else max(fixed, len(captures))
        )
        if total < fixed or (total - fixed) % len(tail):
            msg = "Function prefix must determine complete variadic argument groups"
            raise TypeError(msg)
        patterns.extend(tail * ((total - fixed) // len(tail)))
    if len(captures) > len(patterns) or (
        expected_params is not None and len(captures) + len(expected_params) != len(patterns)
    ):
        msg = "Function prefix and remaining parameters have the wrong arity"
        raise TypeError(msg)
    if view is not None:
        order = [view.receiver] if view.has_field("receiver") else []
        for parameter in view.params:
            order.extend(
                range(fixed, len(patterns))
                if (tail or application) and not specialized and parameter.core_input == fixed
                else (parameter.core_input,)
            )
        if order != list(range(len(patterns))):
            msg = "Reordered Python function-value presentations are not supported"
            raise NotImplementedError(msg)
        if specialized and (not tail or len(patterns) != fixed + len(tail) * view.tail_repetitions):
            msg = "Function value does not match the selected tail specialization"
            raise TypeError(msg)
        if (
            view.owner is not None
            and view.owner.kind is not None
            and view.owner.kind.field == "sort"
            and isinstance(bound, RuntimeClass)
            and bound.__egg_sort__ is not None
        ):
            infer_sort(
                declaration.owner.ref("sorts", view.owner.kind.value), bound.__egg_sort__, binder_size, substitutions
            )
    for pattern, value in zip(patterns, captures, strict=False):
        if isinstance(value, RuntimeExpr):
            node = value.__egg_ref__
            try:
                infer_sort(pattern, node.owner.ref("sorts", node.read().sort_id), binder_size, substitutions)
            except TypeConstraintError:
                if not convert_captures:
                    raise
        elif not convert_captures:
            msg = "Decoded function captures must be canonical expressions"
            raise TypeError(msg)
    if expected_params is not None:
        for pattern, actual_sort in zip(patterns[len(captures) :], expected_params, strict=True):
            infer_sort(pattern, actual_sort, binder_size, substitutions)
        assert expected_result is not None
        infer_sort(output, expected_result, binder_size, substitutions)
    children = tuple(
        resolve_literal(substitute_sort(pattern, substitutions), value)
        if convert_captures
        else cast("RuntimeExpr", value)
        for pattern, value in zip(patterns, captures, strict=False)
    )
    result = substitute_sort(output, substitutions)
    remaining = [substitute_sort(pattern, substitutions) for pattern in patterns[len(captures) :]]
    if len(substitutions) != binder_size:
        msg = "Not every generic parameter was determined by the function value"
        raise TypeConstraintError(msg)
    owner = None
    if view is not None and view.owner is not None and view.owner.kind is not None:
        owner = (
            substitute_sort(declaration.owner.ref("sorts", view.owner.kind.value), substitutions)
            if view.owner.kind.field == "sort"
            else patterns[0]
        )
    builder = Builder()
    builder.import_ref(declaration)
    index = builder.add(
        "sorts",
        pb.Sort(
            kind=Oneof[Literal["func"], pb.FuncSort](
                "func",
                pb.FuncSort(
                    params=[builder.import_ref(parameter) for parameter in remaining], result=builder.import_ref(result)
                ),
            )
        ),
    )
    function_sort = builder.publish().ref("sorts", index)
    infer_sort(function_sort, function_sort, 0, {})
    return function_sort, children, owner


class RuntimeExpr:
    """The sole public symbolic wrapper. Its node Ref is its semantic state."""

    __slots__ = ("__egg_ref__",)

    def __init__(self, reference: Ref) -> None:
        if not isinstance(reference, Ref) or reference.role != "nodes":
            msg = "RuntimeExpr requires an owned protobuf node reference"
            raise TypeError(msg)
        self.__egg_ref__ = reference.owner.ref("nodes", reference.index)

    def __getattr__(self, name: str) -> object:
        if name.startswith("__egg_"):
            raise AttributeError(name)
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if name in cls.__egg_hooks__:
            return cls.__egg_hooks__[name].__get__(self, cls)
        function = RuntimeClass.__getattr__(cls, name)
        if not isinstance(function, RuntimeFunction):
            return function
        result = RuntimeFunction(function.__egg_ref__, function.__egg_view__, self)
        declaration: pb.Declaration = function.__egg_ref__.read()
        if function.__egg_view__ is not None:
            assert declaration.bindings is not None
            assert declaration.bindings.python is not None
            if declaration.bindings.python.views[function.__egg_view__].kind == pb.PythonCallKind.PROPERTY:
                return result()
        return result

    def __str__(self) -> str:
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if "__str__" in cls.__egg_hooks__:
            return cls.__egg_hooks__["__str__"].__get__(self, cls)()
        from .pretty import pretty_ref  # noqa: PLC0415

        return pretty_ref(self.__egg_ref__)

    def __repr__(self) -> str:
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if "__repr__" in cls.__egg_hooks__:
            return cls.__egg_hooks__["__repr__"].__get__(self, cls)()
        return str(self)

    def __hash__(self) -> int:
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if "__hash__" in cls.__egg_hooks__:
            if cls.__egg_hooks__["__hash__"] is None:
                raise TypeError(f"unhashable type: {cls.__name__!r}")
            return cls.__egg_hooks__["__hash__"].__get__(self, cls)()
        return hash(StructuralView(self.__egg_ref__))

    def __eq__(self, other: object) -> Any:
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if "__eq__" in cls.__egg_hooks__:
            return cls.__egg_hooks__["__eq__"].__get__(self, cls)(other)
        if not isinstance(other, RuntimeExpr):
            return NotImplemented
        left: pb.Node = self.__egg_ref__.read()
        right: pb.Node = other.__egg_ref__.read()
        if StructuralView(self.__egg_ref__.owner.ref("sorts", left.sort_id)) != StructuralView(
            other.__egg_ref__.owner.ref("sorts", right.sort_id)
        ):
            return NotImplemented
        from .egraph import eq  # noqa: PLC0415

        return eq(cast("Any", self)).to(other)

    def __ne__(self, other: object) -> Any:
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if "__ne__" in cls.__egg_hooks__:
            return cls.__egg_hooks__["__ne__"].__get__(self, cls)(other)
        from .egraph import ne  # noqa: PLC0415

        return ne(cast("Any", self)).to(other)

    def __copy__(self) -> RuntimeExpr:
        return RuntimeExpr(self.__egg_ref__)

    def __replace_expr__(self, other: RuntimeExpr) -> None:
        self.__egg_ref__ = other.__egg_ref__


def define_expr_method(name: str, *, reverse: bool = False) -> None:
    """Install a Python special-method entrypoint using the generated member view."""

    def invoke(self: RuntimeExpr, *args: object, **kwargs: object) -> object:
        return cast("Callable", self.__getattr__(name))(*args, **kwargs)

    def reflected(self: RuntimeExpr, other: object) -> object:
        from .conversion import convert_to_same_type  # noqa: PLC0415

        return cast("Callable", convert_to_same_type(other, self).__getattr__(name))(self)

    setattr(RuntimeExpr, f"__r{name[2:]}" if reverse else name, reflected if reverse else invoke)


for _name in TYPE_DEFINED_METHODS - {"__hash__"}:
    define_expr_method(_name)

for _name in NUMERIC_BINARY_METHODS:
    define_expr_method(_name)
    define_expr_method(_name, reverse=True)


def resolve_type_annotation(tp: object) -> Ref | TypeVar | RuntimeClass:
    """Keep unresolved Python generics as language state; concrete types are refs."""
    if isinstance(tp, TypeVar):
        return tp
    if get_origin(tp) is Union:
        return resolve_type_annotation(get_args(tp)[0])
    if tp is Callable or get_origin(tp) is Callable:
        arguments = get_args(tp)
        if len(arguments) != 2 or not isinstance(arguments[0], list):
            msg = "Callable annotations require a fixed parameter list"
            raise NotImplementedError(msg)
        sorts = [resolve_type_annotation(argument) for argument in [*arguments[0], arguments[1]]]
        if any(not isinstance(sort, Ref) or sort.role != "sorts" for sort in sorts):
            msg = "Callable annotations require concrete parameter and result sorts"
            raise NotImplementedError(msg)
        builder = Builder()
        indices = [builder.import_ref(cast("Ref", sort)) for sort in sorts]
        index = builder.add(
            "sorts",
            pb.Sort(
                kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=indices[:-1], result=indices[-1]))
            ),
        )
        return builder.publish().ref("sorts", index)
    if isinstance(origin := get_origin(tp), RuntimeClass):
        return resolve_type_annotation(origin[get_args(tp)])
    if isinstance(tp, RuntimeClass):
        return tp.__egg_sort__ if tp.__egg_sort__ is not None else tp
    raise TypeError(f"Unexpected type annotation {tp!r}")


def resolve_callable(value: object) -> Ref:
    if isinstance(value, RuntimeFunction):
        return value.__egg_ref__
    if isinstance(value, RuntimeClass):
        member = RuntimeClass.__getattr__(value, "__init__")
        if isinstance(member, RuntimeFunction):
            return member.__egg_ref__
    if isinstance(value, RuntimeExpr):
        node: pb.Node = value.__egg_ref__.read()
        if node.kind is not None and node.kind.field == "call":
            return _definition_at(value.__egg_ref__, "callable", node.kind.value.func)
    raise TypeError(f"Cannot resolve a canonical callable for {value!r}")


def create_callable(reference: Ref, *, owner_sort: Ref | None = None) -> RuntimeClass | RuntimeFunction | RuntimeExpr:
    """Recover the first canonical Python callable view without evaluating it."""
    record: pb.Declaration = reference.read()
    if record.bindings is None or record.bindings.python is None or not record.bindings.python.views:
        return RuntimeFunction(reference)
    view = record.bindings.python.views[0]
    owner = None
    if view.owner is not None:
        if owner_sort is not None:
            owner = _class_for_sort(owner_sort)
        elif view.owner.kind is None or view.owner.kind.field != "sort":
            msg = "A table callable's Python owner must be a concrete sort"
            raise ValueError(msg)
        else:
            owner = _class_for_sort(reference.owner.ref("sorts", view.owner.kind.value))
    if view.kind == pb.PythonCallKind.INITIALIZER:
        if owner is None:
            msg = "An initializer view requires a Python owner"
            raise ValueError(msg)
        return owner
    if view.kind not in {pb.PythonCallKind.CONSTANT, pb.PythonCallKind.CLASS_VARIABLE}:
        return RuntimeFunction(reference, 0, owner)
    if (
        record.kind is None
        or not isinstance(record.kind.value, pb.Function | pb.Constructor | pb.Primitive)
        or record.kind.value.inputs
    ):
        msg = "A constant view requires a nullary concrete value callable"
        raise ValueError(msg)
    signature = record.kind.value
    builder = Builder()
    builder.import_ref(reference)
    node = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(reference.owner.ref("sorts", signature.output)),
            kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func=signature.name)),
        ),
    )
    return RuntimeExpr(builder.publish().ref("nodes", node))
