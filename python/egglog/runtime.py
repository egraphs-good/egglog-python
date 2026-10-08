"""Python runtime wrappers; semantic state is owned generated protobuf records."""

from __future__ import annotations

import itertools
import struct
from collections.abc import Callable
from contextlib import suppress
from inspect import Parameter, Signature
from threading import RLock
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin
from weakref import WeakKeyDictionary, WeakValueDictionary

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

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

NUMERIC_BINARY_METHODS = {
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
}


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
    varargs = None
    match kind:
        case pb.HostPrimitive():
            if kind.typing is None or kind.typing.field != "signature":
                msg = "Structural function application authoring is not migrated yet"
                raise NotImplementedError(msg)
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
        if varargs is not None:
            view.params.append(pb.PythonParameter(core_input=len(inputs), name=varargs.name or "args"))
    else:
        if record.bindings is None or record.bindings.python is None:
            msg = "Requested Python view is absent"
            raise ValueError(msg)
        view = record.bindings.python.views[view_index]
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
            if varargs is not None and parameter.core_input == len(inputs)
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
    expected_slots = set(range(len(inputs) + (varargs is not None)))
    if core_values.keys() != expected_slots:
        msg = "Python view does not map every core input exactly once"
        raise ValueError(msg)
    ordered_values = [core_values[index] for index in range(len(inputs))]
    patterns = [declaration.owner.ref("sorts", arg.sort) for arg in inputs]
    if varargs is not None:
        tail = core_values[len(inputs)]
        assert isinstance(tail, tuple)
        ordered_values.extend(tail)
        patterns.extend([declaration.owner.ref("sorts", varargs.sort)] * len(tail))
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
_LITERAL_ARMS = {"i64": "i64", "f64": "f64_bits", "String": "string", "bool": "bool", "Unit": "unit"}
LIT_IDENTS = frozenset(_LITERAL_ARMS)
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
        """Publish the whole member scope once, never a partially usable signature."""
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.resolving:
                msg = "Recursively resolving class declarations"
                raise RuntimeError(msg)
            if self.prepare is None:
                return
            self.resolving = True
            try:
                catalog = self.prepare(self.runtime_class)
            except Exception as error:
                self.error = error
                raise
            else:
                self.catalog = catalog
                self.runtime_class.__egg_attr_cache__.clear()
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
        case "eq":
            name = record.kind.value
        case _:
            msg = "Function/pattern runtime class projection is not migrated yet"
            raise NotImplementedError(msg)
    definition = _definition_at(sort, "sort", name)
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

    def __new__(cls, definition: Ref, *, sort: Ref | None = None, arguments: tuple[object, ...] = ()) -> RuntimeClass:
        declaration: pb.Declaration = definition.read()
        kind = declaration.kind
        if kind is None or not isinstance(kind.value, pb.HostSortFamily | pb.EqSort):
            msg = "RuntimeClass requires a canonical sort declaration"
            raise TypeError(msg)
        binding = kind.value.bindings
        path = (
            tuple(binding.python.path)
            if binding is not None and binding.python is not None
            else ("egglog", "builtins", kind.value.name)
        )
        if not path:
            msg = "An empty type path cannot be generated as a Python class"
            raise ValueError(msg)
        namespace = {
            "__module__": ".".join(path[:-1]),
            "__doc__": declaration.doc or None,
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
        if sort is None and not arguments:
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

    def __call__(cls, *args: object, **kwargs: object) -> RuntimeExpr | None:
        declaration: pb.Declaration = cls.__egg_definition__.read()
        assert declaration.kind is not None
        name = declaration.kind.value.name
        if isinstance(declaration.kind.value, pb.HostSortFamily) and name in _LITERAL_ARMS:
            if kwargs or len(args) != (0 if name == "Unit" else 1):
                raise TypeError(f"{cls} expects {'zero' if name == 'Unit' else 'one'} positional literal argument")
            if cls.__egg_sort__ is None:
                msg = "Literal constructor has an unresolved sort"
                raise TypeError(msg)
            primitive = pb.PrimitiveValue()
            match _LITERAL_ARMS[name]:
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
        if not isinstance(kind, pb.HostSortFamily) or len(values) != kind.arity:
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
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=kind.name, args=indices))),
        )
        return _class_for_sort(builder.publish().ref("sorts", index))

    def __getattr__(cls, name: str) -> object:
        if name.startswith("__egg_"):
            raise AttributeError(name)
        if name == "__origin__":
            if cls.__egg_arguments__:
                declaration: pb.Declaration = cls.__egg_definition__.read()
                assert declaration.kind is not None
                return _BASE_CLASSES[declaration.kind.value.name]
            # typing probes a concrete class's origin while resolving another
            # signature; absence must not recursively compile either class.
            raise AttributeError(name)
        if name in cls.__egg_hooks__:
            return cls.__egg_hooks__[name]
        if name in cls.__egg_attr_cache__:
            return cls.__egg_attr_cache__[name]
        if cls.__egg_pending__ is not None:
            cls.__egg_pending__()
        declaration = cls.__egg_definition__.read()
        assert declaration.kind is not None
        key = declaration.kind.value.name, name
        catalog = builtin_catalog()
        if cls.__egg_context__ is not None:
            catalog = cls.__egg_context__.catalog
            assert catalog is not None
        elif cls.__egg_definition__ != catalog.definitions.get(("sort", key[0])):
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


class RuntimeFunction:
    """A selected language view over one owned canonical callable declaration."""

    def __init__(
        self, declaration: Ref, view_index: int | None = None, bound: RuntimeClass | RuntimeExpr | None = None
    ) -> None:
        self.__egg_ref__ = declaration
        self.__egg_view__ = view_index
        self.__egg_bound__ = bound

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
            return ".".join(view.path)
        name = "__init__" if view.kind == pb.PythonCallKind.INITIALIZER else view.path[0]
        return f"{self.__egg_bound__}.{name}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.__egg_ref__, self.__egg_view__, self.__egg_bound__))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeFunction):
            return NotImplemented
        return (
            self.__egg_ref__ == other.__egg_ref__
            and self.__egg_view__ == other.__egg_view__
            and bool(self.__egg_bound__ == other.__egg_bound__)
        )


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


def define_expr_method(name: str) -> None:
    """Install a Python special-method entrypoint using the generated member view."""

    def invoke(self: RuntimeExpr, *args: object, **kwargs: object) -> object:
        return cast("Callable", self.__getattr__(name))(*args, **kwargs)

    setattr(RuntimeExpr, name, invoke)


for _name in TYPE_DEFINED_METHODS - {"__hash__"}:
    define_expr_method(_name)

for _name, _reverse in itertools.product(NUMERIC_BINARY_METHODS, (False, True)):

    def binary(self: RuntimeExpr, other: object, name: str = _name, reverse: bool = _reverse) -> object:
        if reverse:
            from .conversion import convert_to_same_type  # noqa: PLC0415

            return cast("Callable", convert_to_same_type(other, self).__getattr__(name))(self)
        return cast("Callable", self.__getattr__(name))(other)

    setattr(RuntimeExpr, f"__r{_name[2:]}" if _reverse else _name, binary)


def resolve_type_annotation(tp: object) -> Ref | TypeVar | RuntimeClass:
    """Keep unresolved Python generics as language state; concrete types are refs."""
    if isinstance(tp, TypeVar):
        return tp
    if get_origin(tp) is Union:
        return resolve_type_annotation(get_args(tp)[0])
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


def create_callable(reference: Ref) -> RuntimeFunction:
    record: pb.Declaration = reference.read()
    view = (
        0
        if record.bindings is not None and record.bindings.python is not None and record.bindings.python.views
        else None
    )
    return RuntimeFunction(reference, view)
