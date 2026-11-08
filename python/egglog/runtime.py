"""Python runtime wrappers; semantic state is owned generated protobuf records."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from enum import Enum
from functools import partial
from inspect import Parameter, Signature
from threading import RLock
from types import FunctionType, GenericAlias, MethodType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast, get_args, get_origin
from weakref import WeakKeyDictionary, WeakValueDictionary

import cloudpickle
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from . import bindings
from ._builtin_protocols import NUMERIC_BINARY_METHODS, SCALAR_PROTOCOLS
from ._catalog import Catalog
from ._presentation import CallKind, CallPresentation, Presentation, presentation
from ._presentation import Parameter as PresentationParameter
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


class _Intrinsic(Enum):
    APPLY = "apply"


def _author_call(  # noqa: C901, PLR0912
    declaration: Ref | _Intrinsic,
    view_index: int | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    node_of: Callable[[object], Ref | None],
    convert: Callable[[Ref, object], Ref],
    receiver: object = _OMITTED,
    owner: Ref | None = None,
    type_bindings: tuple[tuple[int, Ref], ...] = (),
    plan_only: bool = False,
) -> tuple[Ref, object | None] | list[tuple[Ref, object]]:
    """
    Bind a Python call directly to its canonical declaration and node records.

    The callbacks are Python conversion policy, not signature providers. All
    parameter ordering, defaults, generic patterns and mutation selection come
    from the supplied immutable records. The return's second item identifies
    the supplied wrapper to replace for a mutating presentation.
    Planning returns only expected input sorts and supplied values, before
    callbacks/converters; an empty plan cannot infer all signature binders.
    """
    if declaration is _Intrinsic.APPLY:
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
        bound = Signature([Parameter("args", Parameter.VAR_POSITIONAL)]).bind(*args, **kwargs)
        supplied = bound.arguments.get("args", ())
        assert isinstance(supplied, tuple)
        if len(supplied) != len(function_type.kind.value.params):
            raise TypeError(f"Function application expects {len(function_type.kind.value.params)} arguments")
        converted = [first]
        for index, value in zip(function_type.kind.value.params, supplied, strict=True):
            expected = function_sort.owner.ref("sorts", index)
            child = value if isinstance(value, Ref) else node_of(value)
            if child is None or StructuralView(child.owner.ref("sorts", child.read().sort_id)) != StructuralView(
                expected
            ):
                child = convert(expected, value)
            infer_sort(expected, child.owner.ref("sorts", child.read().sort_id), 0, {})
            converted.append(child)
        builder = Builder()
        node_index = builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(function_sort.owner.ref("sorts", function_type.kind.value.result)),
                kind=Oneof[Literal["apply"], pb.Apply](
                    "apply",
                    pb.Apply(
                        function=builder.import_ref(converted[0]),
                        args=[builder.import_ref(node) for node in converted[1:]],
                    ),
                ),
            ),
        )
        return builder.publish().ref("nodes", node_index), None
    assert isinstance(declaration, Ref)
    if declaration.role not in {"constructors", "functions", "relations", "user_defined_primitives", "primitives"}:
        msg = "Call authoring requires a declaration reference"
        raise TypeError(msg)
    record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = declaration.read()
    kind = record
    binder_size = 0
    varargs: list[int] = []
    output: int | None
    match kind:
        case pb.Primitive():
            if kind.signature is None:
                msg = "Host primitive has no signature"
                raise ValueError(msg)
            signature = kind.signature
            if not signature.has_field("output"):
                msg = "Host signature has no output"
                raise ValueError(msg)
            inputs, output = signature.inputs, signature.output
            binder_size = signature.type_param_count
            varargs = signature.varargs
        case pb.Constructor() | pb.Function() | pb.UserDefinedPrimitive():
            inputs, output = kind.inputs, kind.output
        case pb.Relation():
            inputs, output = kind.inputs, None
    if view_index is None:
        # Conservative derived free-function binding: no synthetic metadata is
        # persisted or installed. The generator checks its exported path.
        view = CallPresentation(
            kind=CallKind.FUNCTION,
            path=(kind.name,),
            params=tuple(PresentationParameter(core_input=i, name=f"arg{i}") for i in range(len(inputs)))
            + ((PresentationParameter(core_input=len(inputs), name="args"),) if varargs else ()),
        )
    else:
        if not presentation(declaration).views:
            msg = "Requested Python view is absent"
            raise ValueError(msg)
        view = presentation(declaration).views[view_index]
    if (view.mutates is not None) and (
        view.kind == CallKind.INITIALIZER
        or view.mutates >= len(inputs)
        or output is None
        or StructuralView(declaration.owner.ref("sorts", inputs[view.mutates]))
        != StructuralView(declaration.owner.ref("sorts", output))
        or not (
            ((view.receiver is not None) and view.mutates == view.receiver)
            or any(parameter.core_input == view.mutates for parameter in view.params)
        )
    ):
        msg = "Invalid mutated input"
        raise ValueError(msg)
    specialized = view.tail_repetitions is not None
    consumes_tail = not specialized and (view.receiver is not None) and view.receiver == len(inputs)
    if consumes_tail and (len(varargs) != 1 or view.kind != CallKind.METHOD):
        msg = "Tail receiver requires a homogeneous variadic method"
        raise ValueError(msg)
    tail_slot = len(inputs) + consumes_tail
    slot_count = tail_slot + bool(varargs)
    if specialized:
        if view.kind not in {CallKind.FUNCTION, CallKind.METHOD}:
            msg = "Tail specialization supports only function and method views"
            raise ValueError(msg)
        if not varargs:
            msg = "Tail specialization requires a variadic host signature"
            raise ValueError(msg)
        if any(type(parameter.core_input) is not int for parameter in view.params):
            msg = "Python view parameter is missing its core input"
            raise ValueError(msg)
        assert view.tail_repetitions is not None
        slot_count = len(inputs) + len(varargs) * view.tail_repetitions
        if slot_count > 2**32 - 1 or slot_count != len(view.params) + (view.receiver is not None):
            msg = "Tail specialization must map every effective input exactly once"
            raise ValueError(msg)
        if any(p.core_input >= len(inputs) and (p.default is not None) for p in view.params) or (
            (view.mutates is not None) and view.mutates >= len(inputs)
        ):
            msg = "Specialized tail inputs cannot have defaults or mutation"
            raise ValueError(msg)
    substitutions: dict[int, Ref] = dict(type_bindings)
    if owner is not None and (view.owner is not None):
        infer_sort(view.owner, owner, binder_size, substitutions)
    core_values: dict[int, object] = {}
    if view.receiver is not None:
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
            if varargs and not specialized and parameter.core_input == tail_slot
            else Parameter.POSITIONAL_OR_KEYWORD,
            default=_OMITTED if (parameter.default is not None) else Parameter.empty,
        )
        for parameter in view.params
    ]
    bound = Signature(parameters).bind(*args, **kwargs)
    bound.apply_defaults()
    omitted = [parameter for parameter in view.params if bound.arguments[parameter.name] is _OMITTED]
    defaults = iter(clone_nodes(parameter.default for parameter in omitted if parameter.default is not None))
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
    if view.mutates is not None:
        target = node_of(core_values[view.mutates])
        if target is None:
            msg = "Mutating arguments require a supplied expression wrapper"
            raise TypeError(msg)
        target_node: pb.Node = target.read()
        try:
            infer_sort(
                declaration.owner.ref("sorts", inputs[view.mutates]),
                target.owner.ref("sorts", target_node.sort_id),
                binder_size,
                substitutions,
            )
        except TypeConstraintError as error:
            from .conversion import ConvertError  # noqa: PLC0415

            msg = "Cannot convert a mutated argument: its wrapper must already have the input sort"
            raise ConvertError(msg) from error
    ordered_values = [core_values[index] for index in range(slot_count if specialized else tail_slot)]
    patterns = [declaration.owner.ref("sorts", arg) for arg in inputs]
    if varargs:
        tail = tuple(ordered_values[len(inputs) :]) if specialized else core_values[tail_slot]
        assert isinstance(tail, tuple)
        if len(tail) % len(varargs):
            raise TypeError(f"Variadic arguments must form complete groups of {len(varargs)}")
        if not specialized:
            ordered_values.extend(tail)
        patterns.extend(
            declaration.owner.ref("sorts", arg)
            for _ in range(len(tail) // len(varargs) + consumes_tail)
            for arg in varargs
        )
    if (
        (view.receiver is not None)
        and (view.owner is not None)
        and StructuralView(patterns[view.receiver]) != StructuralView(view.owner)
    ):
        msg = "Python receiver input must match its owner"
        raise ValueError(msg)
    existing = []
    for pattern, value in zip(patterns, ordered_values, strict=True):
        node = value if isinstance(value, Ref) else node_of(value)
        callback = value
        prefix: tuple[object, ...] = ()
        if isinstance(callback, partial) and not callback.keywords:
            prefix = callback.args
            callback = callback.func
        # A closed named callback/prefix can determine a method-local result binder.
        # Generic callbacks still use their concrete expected conversion below.
        if (
            not plan_only
            and node is None
            and pattern.read().kind.field == "func"
            and isinstance(callback, RuntimeFunction)
        ):
            with suppress(TypeConstraintError):
                expression = _BASE_CLASSES["UnstableFn"](callback, *prefix)
                assert isinstance(expression, RuntimeExpr)
                node = expression.__egg_ref__
        existing.append(node)
    for pattern, node in zip(patterns, existing, strict=True):
        if node is not None:
            payload: pb.Node = node.read()
            # A registered Python conversion may still satisfy this input.
            with suppress(TypeConstraintError):
                infer_sort(pattern, node.owner.ref("sorts", payload.sort_id), binder_size, substitutions)
    if plan_only:
        if len(substitutions) != binder_size:
            return []
        if output is not None:
            substitute_sort(declaration.owner.ref("sorts", output), substitutions)
        return [
            (substitute_sort(pattern, substitutions), value)
            for pattern, value in zip(patterns, ordered_values, strict=True)
        ]
    for position, (pattern, value, node) in enumerate(zip(patterns, ordered_values, existing, strict=True)):
        sort_kind = pattern.read().kind
        if node is None and sort_kind is not None and sort_kind.field == "func":
            result = None
            with suppress(TypeConstraintError):
                result = substitute_sort(pattern.owner.ref("sorts", sort_kind.value.result), substitutions)
            if isinstance(value, FunctionType):
                from .egraph import _compile_python_function  # noqa: PLC0415

                parameters = [
                    substitute_sort(pattern.owner.ref("sorts", index), substitutions)
                    for index in sort_kind.value.params
                ]
                expression = _compile_python_function(value, parameters, result)
            else:
                callback = value
                prefix = ()
                if isinstance(callback, partial) and not callback.keywords:
                    prefix = callback.args
                    callback = callback.func
                if not isinstance(callback, RuntimeFunction):
                    continue
                context_parameters: list[Ref | None] = []
                for index in sort_kind.value.params:
                    parameter = None
                    with suppress(TypeConstraintError):
                        parameter = substitute_sort(pattern.owner.ref("sorts", index), substitutions)
                    context_parameters.append(parameter)
                expression = _function_value(callback, prefix, None, context=(context_parameters, result))
            callback_node = expression.__egg_ref__
            infer_sort(
                pattern, callback_node.owner.ref("sorts", callback_node.read().sort_id), binder_size, substitutions
            )
            existing[position] = callback_node
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
    if output is None:
        result_sort = _BASE_CLASSES["Unit"].__egg_sort__
        assert result_sort is not None
    else:
        result_sort = substitute_sort(declaration.owner.ref("sorts", output), substitutions)
    if len(substitutions) != binder_size:
        msg = "Not every generic parameter was determined by the call"
        raise TypeConstraintError(msg)
    builder = Builder()
    builder.import_ref(declaration)
    arguments = [builder.import_ref(node) for node in converted]
    primitive: pb.PrimitiveValue | None = None
    if isinstance(kind, pb.Primitive) and kind.name in {
        "egglog.core.pair.make",
        "egglog.core.vec.of",
        "egglog.core.vec.empty",
        "egglog.core.set.of",
        "egglog.core.set.empty",
        "egglog.core.multiset.of",
        "egglog.core.multiset.empty",
        "egglog.core.map.empty",
        "egglog.experimental.maybe.none",
        "egglog.experimental.maybe.some",
    }:
        from ._catalog import builtin_catalog  # noqa: PLC0415

        canonical = builtin_catalog().definitions.get(("callable", kind.name))
        if canonical is None or StructuralView(declaration) != StructuralView(canonical):
            msg = "Data construction requires its canonical native declaration"
            raise TypeConstraintError(msg)
        match kind.name:
            case "egglog.core.pair.make":
                primitive = pb.PrimitiveValue(
                    value=Oneof[Literal["pair"], pb.PairValue](
                        "pair", pb.PairValue(first=arguments[0], second=arguments[1])
                    )
                )
            case "egglog.core.vec.of" | "egglog.core.vec.empty":
                primitive = pb.PrimitiveValue(
                    value=Oneof[Literal["vec"], pb.ValueList]("vec", pb.ValueList(items=arguments))
                )
            case "egglog.core.set.of" | "egglog.core.set.empty":
                primitive = pb.PrimitiveValue(
                    value=Oneof[Literal["set"], pb.ValueList]("set", pb.ValueList(items=arguments))
                )
            case "egglog.core.multiset.of" | "egglog.core.multiset.empty":
                primitive = pb.PrimitiveValue(
                    value=Oneof[Literal["multiset"], pb.ValueList]("multiset", pb.ValueList(items=arguments))
                )
            case "egglog.core.map.empty":
                primitive = pb.PrimitiveValue(value=Oneof[Literal["map"], pb.MapValue]("map", pb.MapValue()))
            case "egglog.experimental.maybe.none" | "egglog.experimental.maybe.some":
                primitive = pb.PrimitiveValue(
                    value=Oneof[Literal["maybe"], pb.MaybeValue](
                        "maybe", pb.MaybeValue(value=arguments[0] if arguments else None)
                    )
                )
    node_kind = (
        Oneof[Literal["primitive_value"], pb.PrimitiveValue]("primitive_value", primitive)
        if primitive is not None
        else Oneof[Literal["call"], pb.Call]("call", pb.Call(func=kind.name, args=arguments))
    )
    node_index = builder.add("nodes", pb.Node(sort_id=builder.import_ref(result_sort), kind=node_kind))
    return builder.publish().ref("nodes", node_index), core_values[view.mutates] if (view.mutates is not None) else None


# These are Python codecs/protocol hooks, not callable signatures. Sort and
# callable definitions still come from the native-generated catalog.
LIT_IDENTS = frozenset(SCALAR_PROTOCOLS)
DUMMY_VALUE = object()
_CLASS_CACHE: WeakKeyDictionary[Owner, WeakValueDictionary[int, RuntimeClass]] = WeakKeyDictionary()
_BASE_CLASSES: dict[str, RuntimeClass] = {}


class _ClassContext:
    """Owner-retained Python lifecycle/hooks; all declaration semantics remain refs."""

    def __init__(
        self,
        prepare: Callable[[RuntimeClass], Catalog],
        hooks: dict[str, Any],
        match_args: tuple[str, ...],
        *,
        native_members: frozenset[str] | None = None,
    ) -> None:
        self.prepare: Callable[[RuntimeClass], Catalog] | None = prepare
        self.hooks = hooks
        self.match_args = match_args
        self.native_members = native_members
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
                self._publish_callables(catalog)
            finally:
                # Private eager headers never survive an incomplete class
                # attempt; completed dependencies remain in the catalog/cache.
                for name, value in tuple(self.runtime_class.__egg_attr_cache__.items()):
                    if isinstance(value, RuntimeFunction) and value.__egg_pending_declaration__ is not None:
                        del self.runtime_class.__egg_attr_cache__[name]
                self.prepare = None
                self.resolving = False

    def _publish_callables(self, catalog: Catalog) -> None:
        """Stage canonical callables, preserving Python attributes and class variables."""
        if self.native_members is not None:
            for name in self.native_members:
                member = self.runtime_class.__egg_attr_cache__.get(name)
                if isinstance(member, RuntimeFunction):
                    type.__setattr__(self.runtime_class, name, member)
            return
        assert isinstance(self.runtime_class.__egg_definition__, Ref)
        definition = _definition_key(self.runtime_class.__egg_definition__.read())
        # Bypass the public __dict__ trigger while staging publication.
        namespaces = [
            type.__getattribute__(base, "__dict__")
            for base in (
                *type.__getattribute__(self.runtime_class, "__mro__"),
                *type.__getattribute__(type(self.runtime_class), "__mro__"),
            )
        ]
        members: dict[str, RuntimeFunction] = {}
        for (owner, name), (reference, view_index) in catalog.members.items():
            if owner != definition or name.startswith("__egg_"):
                continue
            current = next((namespace[name] for namespace in namespaces if name in namespace), _OMITTED)
            if current is not _OMITTED and not isinstance(current, RuntimeClassDescriptor):
                continue
            if presentation(reference).views[view_index].kind == CallKind.CLASS_VARIABLE:
                continue
            member = RuntimeClass.__getattr__(self.runtime_class, name)
            assert isinstance(member, RuntimeFunction)
            members[name] = member
        # Publish only complete selected callables, never invoking a
        # class variable or maintaining a second semantic catalog.
        for name, member in members.items():
            type.__setattr__(self.runtime_class, name, member)


def _definition_at(origin: Ref, namespace: str, name: str) -> Ref:
    """Resolve owned definitions before consulting the ambient native catalog."""
    matches = {
        origin.owner.ref(role, index) for role, index in origin.owner.definition_indices.get((namespace, name), ())
    }
    if namespace in {"eq_sort", "binding"}:
        matches.update(
            environment.ref(role, index)
            for environment in origin.owner._environment
            for role, index in environment.definition_indices.get((namespace, name), ())
        )
    if len(matches) > 1:
        first = next(iter(matches))
        if namespace not in {"eq_sort", "binding"} or any(
            StructuralView(first) != StructuralView(other) for other in matches
        ):
            raise ValueError(f"Ambiguous local definition: {namespace} {name!r}")
    if matches:
        return next(
            (reference for reference in matches if (namespace, name) in reference.owner._presentations),
            next(iter(matches)),
        )
    role = "primitives" if namespace == "callable" else "primitive_sorts"
    candidates = (origin.owner.ref(role, index) for index in range(len(origin.owner._slots[role])))
    selected = tuple(
        candidate
        for candidate in candidates
        if _definition_key(
            cast("pb.Primitive | pb.PrimitiveSort", candidate.owner._slots[candidate.role][candidate.index])
        )
        == (namespace, name)
    )
    matches.update(selected)
    for environment in (*origin.owner._environment, origin.owner):
        if (slot := environment._builtin_indices.get((namespace, name))) is not None:
            matches.add(environment.ref(*slot))
    if matches:
        first = selected[0] if selected else next(iter(matches))
        rest = matches - {first}
        if any(StructuralView(first) != StructuralView(other) for other in rest):
            raise ValueError(f"Conflicting builtin environments: {namespace} {name!r}")
        return first
    raise LookupError(f"No owned canonical definition for {namespace} {name!r}")


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
        case "primitive":
            name = record.kind.value.name
            namespace = "primitive_sort"
        case "eq":
            name = record.kind.value
            namespace = "eq_sort"
        case "func":
            infer_sort(sort, sort, 0, {})
            result = RuntimeClass(_Intrinsic.APPLY, sort=sort)
            cache[sort.index] = result
            return result
        case _:
            msg = "Function/pattern runtime class projection is not migrated yet"
            raise NotImplementedError(msg)
    definition = _definition_at(sort, namespace, name)
    for context in definition.owner._keepalive:
        if isinstance(context, _ClassContext):
            if record.kind.field == "eq":
                return context.runtime_class
            break
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


class _RuntimeClassAlias(GenericAlias):
    """Use Python substitution, then materialize a concrete canonical sort."""

    def __getitem__(self, arguments: object) -> RuntimeClass | _RuntimeClassAlias:  # type: ignore[override]
        # Unlike a plain GenericAlias, a fully substituted language alias must
        # produce the same canonical class as direct family specialization.
        substituted = super().__getitem__(arguments)
        return cast("RuntimeClass", substituted.__origin__)[substituted.__args__]


class RuntimeClass(type):
    """A Python type presentation over an owned sort/family declaration."""

    def __new__(  # noqa: C901, PLR0912
        cls,
        definition: Ref | _Intrinsic,
        *,
        sort: Ref | None = None,
        arguments: tuple[object, ...] = (),
        context: _ClassContext | None = None,
    ) -> RuntimeClass:
        structural_function = definition is _Intrinsic.APPLY
        declaration = definition.read() if isinstance(definition, Ref) else None
        kind = declaration
        if structural_function:
            binding = None
            path = ("egglog", "builtins", "UnstableFn")
        elif isinstance(kind, pb.PrimitiveSort | pb.Sort):
            assert isinstance(definition, Ref)
            if isinstance(kind, pb.Sort):
                assert kind.kind is not None
                assert kind.kind.field == "eq"
                name = kind.kind.value
            else:
                name = kind.name
            binding = presentation(definition).type_view
            path = tuple(binding.path) if binding is not None else ("egglog", "builtins", name)
        else:
            msg = "RuntimeClass requires a canonical sort declaration"
            raise TypeError(msg)
        if not path:
            msg = "An empty type path cannot be generated as a Python class"
            raise ValueError(msg)
        namespace = {
            "__module__": ".".join(path[:-1]),
            "__doc__": declaration.doc if declaration is not None and declaration.has_field("doc") else None,
            **{name: RuntimeClassDescriptor(name) for name in CLASS_DESCRIPTOR_METHODS},
        }
        result = type.__new__(cls, path[-1], (), namespace)
        result.__egg_definition__ = definition
        result.__egg_sort__ = sort
        result.__egg_arguments__ = arguments
        result.__egg_attr_cache__ = {}
        result.__egg_hooks__ = {}
        result.__egg_pending__ = None
        result.__egg_context__ = context or (
            next((context for context in definition.owner._keepalive if isinstance(context, _ClassContext)), None)
            if isinstance(definition, Ref)
            else None
        )
        if result.__egg_context__ is not None:
            if sort is None and not arguments:
                result.__egg_context__.runtime_class = result
            result.__egg_hooks__ = result.__egg_context__.hooks
            result.__egg_pending__ = result.__egg_context__.resolve
        if structural_function:
            if sort is None and not arguments:
                _BASE_CLASSES["UnstableFn"] = result
            elif result.__egg_context__ is None and "UnstableFn" in _BASE_CLASSES:
                result.__egg_hooks__ = _BASE_CLASSES["UnstableFn"].__egg_hooks__
                result.__egg_context__ = _BASE_CLASSES["UnstableFn"].__egg_context__
                result.__egg_pending__ = _BASE_CLASSES["UnstableFn"].__egg_pending__
        elif sort is None and not arguments:
            assert isinstance(definition, Ref)
            assert isinstance(kind, pb.PrimitiveSort | pb.Sort)
            arity = kind.arity if isinstance(kind, pb.PrimitiveSort) else 0
            if arity:
                labels = binding.type_params if binding is not None else ()
                result.__egg_arguments__ = tuple(
                    TypeVar(labels[index] if labels else f"T{index}") for index in range(arity)
                )
            elif isinstance(kind, pb.Sort):
                result.__egg_sort__ = definition
            else:
                builder = Builder()
                builder.import_ref(definition)
                index = builder.add(
                    "sorts",
                    pb.Sort(
                        kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef](
                            "primitive", pb.PrimitiveSortRef(name=kind.name)
                        )
                    ),
                )
                result.__egg_sort__ = builder.publish().ref("sorts", index)
            if isinstance(kind, pb.PrimitiveSort):
                _BASE_CLASSES[kind.name] = result
        if (
            result.__egg_context__ is None
            and sort is not None
            and isinstance(kind, pb.PrimitiveSort)
            and (base := _BASE_CLASSES.get(kind.name)) is not None
            and isinstance(definition, Ref)
            and isinstance(base.__egg_definition__, Ref)
            and StructuralView(definition) == StructuralView(base.__egg_definition__)
        ):
            result.__egg_hooks__ = base.__egg_hooks__
            result.__egg_pending__ = base.__egg_pending__
            result.__egg_context__ = base.__egg_context__
        if result.__egg_sort__ is not None:
            actual = result.__egg_sort__
            _CLASS_CACHE.setdefault(actual.owner, WeakValueDictionary())[actual.index] = result
        return result

    if TYPE_CHECKING:

        def __init__(
            cls,
            definition: Ref | _Intrinsic,
            *,
            sort: Ref | None = None,
            arguments: tuple[object, ...] = (),
            context: _ClassContext | None = None,
        ) -> None: ...

    else:

        def __init__(cls, *args: object, **kwargs: object) -> None:
            # __new__ fully initializes canonical records; type() can replay
            # its original (name, bases, namespace) arguments here instead.
            pass

    __egg_definition__: Ref | _Intrinsic
    __egg_sort__: Ref | None
    __egg_arguments__: tuple[object, ...]
    __egg_attr_cache__: dict[str, object]
    __egg_hooks__: dict[str, Any]
    __egg_pending__: Callable[[], None] | None
    __egg_context__: _ClassContext | None

    def __getattribute__(cls, name: str) -> object:
        if name == "__dict__":
            # Raw access avoids recursively entering the publication trigger
            # during preparation, including its permitted same-thread reentry.
            pending: Callable[[], None] | None = type.__getattribute__(cls, "__dict__").get("__egg_pending__")
            if pending is not None:
                pending()
        return type.__getattribute__(cls, name)

    def __instancecheck__(cls, instance: object) -> bool:
        if not isinstance(instance, RuntimeExpr):
            return False
        node: pb.Node = instance.__egg_ref__.read()
        actual = _class_for_sort(instance.__egg_ref__.owner.ref("sorts", node.sort_id))
        return cls == actual if cls.__egg_sort__ is not None else cls.__egg_definition__ == actual.__egg_definition__

    def __call__(cls, *args: object, **kwargs: object) -> RuntimeExpr | None:  # noqa: C901, PLR0912
        if cls.__egg_definition__ is _Intrinsic.APPLY:
            if kwargs or not args:
                msg = "UnstableFn requires a callable and positional prefix arguments"
                raise TypeError(msg)
            function, *captures = args
            if isinstance(function, MethodType) and isinstance(function.__self__, RuntimeExpr):
                for member_name, member in RuntimeExpr.__dict__.items():
                    if member is function.__func__:
                        function = _get_expr_member(function.__self__, member_name)
                        break
            if isinstance(function, RuntimeClass):
                function = RuntimeClass.__getattr__(function, "__init__")
            if isinstance(function, FunctionType):
                from .egraph import _compile_python_function  # noqa: PLC0415

                if captures or cls.__egg_sort__ is None:
                    msg = "Python callbacks require a concrete expected function type and no explicit prefix"
                    raise TypeError(msg)
                expected: pb.Sort = cls.__egg_sort__.read()
                assert expected.kind is not None
                assert expected.kind.field == "func"
                return _compile_python_function(
                    function,
                    [cls.__egg_sort__.owner.ref("sorts", index) for index in expected.kind.value.params],
                    cls.__egg_sort__.owner.ref("sorts", expected.kind.value.result),
                )
            if not isinstance(function, RuntimeFunction):
                msg = "UnstableFn requires a named RuntimeFunction"
                raise TypeError(msg)
            return _function_value(function, tuple(captures), cls.__egg_sort__, environment=())
        assert isinstance(cls.__egg_definition__, Ref)
        declaration: pb.Sort | pb.PrimitiveSort = cls.__egg_definition__.read()
        kind = declaration
        if isinstance(kind, pb.Sort):
            assert kind.kind is not None
            assert kind.kind.field == "eq"
            name = kind.kind.value
        else:
            name = kind.name
        if (
            isinstance(kind, pb.PrimitiveSort)
            and name in SCALAR_PROTOCOLS
            and SCALAR_PROTOCOLS[name].conversion_sort
            and kwargs
        ):
            raise TypeError(f"{cls} expects positional constructor arguments")
        if (
            isinstance(kind, pb.PrimitiveSort)
            and name in LIT_IDENTS
            and not (
                SCALAR_PROTOCOLS[name].conversion_sort
                and any(isinstance(arg, RuntimeExpr) for arg in (*args, *kwargs.values()))
            )
        ):
            protocol = SCALAR_PROTOCOLS[name]
            if kwargs:
                raise TypeError(f"{cls} expects positional literal arguments")
            if protocol.conversion_sort and protocol.value_type is not int:
                if len(args) == 1 and isinstance(args[0], protocol.literal_types):
                    value = args[0]
                elif len(args) == 2 and all(isinstance(arg, int) for arg in args):
                    value = args
                else:
                    raise TypeError(f"{cls} expects an int/Fraction literal or two int literals")
            else:
                if len(args) != (0 if name == "Unit" else 1):
                    raise TypeError(f"{cls} expects {'zero' if name == 'Unit' else 'one'} positional literal argument")
                if args and not isinstance(args[0], protocol.literal_types):
                    raise TypeError(f"{cls} expects a {protocol.value_type.__name__} literal")
                value = args[0] if args else None
            if cls.__egg_sort__ is None:
                msg = "Literal constructor has an unresolved sort"
                raise TypeError(msg)
            if protocol.literal_arm == "custom":
                try:
                    payload = cloudpickle.dumps(value)
                except Exception as error:
                    # Neither user formatting nor add_note may hide the error.
                    with suppress(BaseException):
                        error.add_note(f"Failed to pickle object of type {type(value)}")
                    raise
                primitive = pb.PrimitiveValue(
                    value=Oneof[Literal["custom"], pb.CustomValue]("custom", pb.CustomValue(payload=payload))
                )
            else:
                primitive = pb.PrimitiveValue.from_binary(bindings._encode_scalar(name, value))
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

    def __getitem__(cls, arguments: object) -> RuntimeClass | _RuntimeClassAlias:
        values = arguments if isinstance(arguments, tuple) else (arguments,)
        structural_function = cls.__egg_definition__ is _Intrinsic.APPLY
        declaration = cls.__egg_definition__.read() if isinstance(cls.__egg_definition__, Ref) else None
        kind = declaration
        if (structural_function and not values) or (
            not structural_function and (not isinstance(kind, pb.PrimitiveSort) or len(values) != kind.arity)
        ):
            raise TypeError(f"Wrong number of type arguments for {cls}")
        if any(isinstance(value, TypeVar | _RuntimeClassAlias) for value in values):
            # Python unions intentionally ignore type-parameter descriptors on
            # classes, so unresolved applications must be actual generic aliases.
            return _RuntimeClassAlias(cls, values)
        if not all(isinstance(value, RuntimeClass) and value.__egg_sort__ is not None for value in values):
            msg = "Type arguments must resolve to canonical sorts"
            raise TypeError(msg)
        builder = Builder()
        if isinstance(cls.__egg_definition__, Ref):
            builder.import_ref(cls.__egg_definition__)
        indices = [builder.import_ref(cast("Ref", cast("RuntimeClass", value).__egg_sort__)) for value in values]
        if structural_function:
            sort_record = pb.Sort(
                kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=indices[1:], result=indices[0]))
            )
        else:
            assert isinstance(kind, pb.PrimitiveSort)
            sort_record = pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef](
                    "primitive", pb.PrimitiveSortRef(name=kind.name, args=indices)
                )
            )
        index = builder.add("sorts", sort_record)
        return _class_for_sort(builder.publish().ref("sorts", index))

    def __getattr__(  # noqa: C901, PLR0911, PLR0912
        cls, name: str, _missing_ok: bool = False, _method_only: bool = False
    ) -> object:
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
                if record.kind.field == "primitive"
                else ()
            )
        if name == "__origin__":
            if cls.__egg_definition__ is _Intrinsic.APPLY:
                if cls.__egg_arguments__:
                    return _BASE_CLASSES["UnstableFn"]
                raise AttributeError(name)
            if cls.__egg_arguments__:
                assert isinstance(cls.__egg_definition__, Ref)
                declaration: (
                    pb.Constructor
                    | pb.Function
                    | pb.Relation
                    | pb.UserDefinedPrimitive
                    | pb.Primitive
                    | pb.PrimitiveSort
                ) = cls.__egg_definition__.read()
                kind = declaration
                return _BASE_CLASSES["UnstableFn" if isinstance(kind, pb.Primitive) else kind.name]
            # typing probes a concrete class's origin while resolving another
            # signature; absence must not recursively compile either class.
            raise AttributeError(name)
        if name in cls.__egg_hooks__:
            return cls.__egg_hooks__[name]
        context = cls.__egg_context__
        if context is not None and context.native_members is not None:
            if name not in context.native_members:
                if _missing_ok:
                    return _OMITTED
                raise AttributeError(f"{cls} has no declared Python member {name!r}")
            context.resolve()
            template = context.runtime_class.__egg_attr_cache__.get(name)
            if isinstance(template, RuntimeFunction):
                result = RuntimeFunction(template.__egg_ref__, template.__egg_view__, cls)
                cls.__egg_attr_cache__[name] = result
                return result
        if cls.__egg_definition__ is _Intrinsic.APPLY:
            if name == "__call__":
                if name not in cls.__egg_attr_cache__:
                    cls.__egg_attr_cache__[name] = RuntimeFunction(_Intrinsic.APPLY, bound=cls)
                return cls.__egg_attr_cache__[name]
            if _missing_ok:
                return _OMITTED
            raise AttributeError(f"{cls} has no intrinsic Python member {name!r}")
        assert isinstance(cls.__egg_definition__, Ref)
        # Even a cached prefix member must wait for another thread's active
        # attempt. Same-thread reentry uses the already-published declarations.
        if cls.__egg_pending__ is not None:
            cls.__egg_pending__()
        cached = cls.__egg_attr_cache__.get(name)
        if isinstance(cached, RuntimeFunction) and cached.__egg_pending_declaration__ is not None:
            pending_presentation = cached.__egg_pending_presentation__
            assert pending_presentation is not None
            if _method_only and pending_presentation.views[0].kind != CallKind.METHOD:
                raise AttributeError(f"{cls} member {name!r} is not an instance method")
            # A callable can escape only after its exact final definition exists.
            _ = cached.__egg_ref__
        # A cached class variable is already evaluated; fallback eligibility
        # must instead be decided from its declaration without invoking it.
        if not _method_only and name in cls.__egg_attr_cache__:
            return cls.__egg_attr_cache__[name]
        if (
            _method_only
            and isinstance(cached, RuntimeFunction)
            and cached.__egg_view__ is not None
            and presentation(cached.__egg_ref__).views[cached.__egg_view__].kind == CallKind.METHOD
        ):
            return cached
        declaration = cls.__egg_definition__.read()
        definition_key = _definition_key(declaration)
        assert definition_key is not None
        key = (("function", "") if isinstance(declaration, pb.Primitive) else definition_key), name
        if cls.__egg_context__ is not None:
            catalog = cls.__egg_context__.catalog
            assert catalog is not None
        else:
            owner = cls.__egg_definition__.owner
            catalog = owner._presentation_catalog
            if catalog is None:
                catalog = Catalog(owner)
                owner._presentation_catalog = catalog
            if cls.__egg_definition__ != catalog.definitions.get(definition_key):
                if _missing_ok:
                    return _OMITTED
                raise AttributeError(f"{cls} has no canonical Python member {name!r}")
        try:
            reference, view = catalog.members[key]
        except KeyError as exc:
            if _missing_ok:
                return _OMITTED
            raise AttributeError(f"{cls} has no generated Python member {name!r}") from exc
        function = RuntimeFunction(reference, view, cls)
        if _method_only and presentation(reference).views[view].kind != CallKind.METHOD:
            raise AttributeError(f"{cls} member {name!r} is not an instance method")
        result = function() if presentation(reference).views[view].kind == CallKind.CLASS_VARIABLE else function
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
        declaration: Ref | Callable[[], Ref] | _Intrinsic,
        view_index: int | None = None,
        bound: RuntimeClass | RuntimeExpr | None = None,
        *,
        type_bindings: tuple[tuple[int, Ref], ...] = (),
        pending_declaration: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | None = None,
        pending_presentation: Presentation | None = None,
        pending_values: bool = False,
    ) -> None:
        self.__egg_intrinsic__ = declaration if isinstance(declaration, _Intrinsic) else None
        self.__egg_declaration__ = declaration if isinstance(declaration, Ref) else None
        self.__egg_prepare__ = declaration if callable(declaration) else None
        self.__egg_pending_declaration__ = pending_declaration
        self.__egg_pending_presentation__ = pending_presentation
        # Class staging readiness only; semantics remain in the private record.
        self.__egg_pending_values__ = pending_values
        self.__egg_error__: Exception | None = None
        self.__egg_resolving__ = False
        self.__egg_lock__ = RLock()
        self.__egg_view__ = view_index
        self.__egg_bound__ = bound
        if type_bindings:
            record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = (
                self.__egg_ref__.read()
            )
            kind = record
            if not isinstance(kind, pb.Primitive) or kind.signature is None:
                msg = "Type bindings require a generic callable signature"
                raise TypeConstraintError(msg)
            signature = kind.signature
            indices = [index for index, _ in type_bindings]
            if indices != sorted(set(indices)) or any(
                index < 0 or index >= signature.type_param_count for index in indices
            ):
                msg = "Invalid callable type binding indices"
                raise TypeConstraintError(msg)
            for _index, sort in type_bindings:
                infer_sort(sort, sort, 0, {})
        self.__egg_type_bindings__ = tuple(
            (index, sort.owner.ref("sorts", sort.index)) for index, sort in type_bindings
        )

    def __get__(self, _instance: object, _owner: type | None = None) -> RuntimeFunction:
        # Selection/binding already happened in RuntimeClass/_get_expr_member.
        # Direct namespace lookup must still respect the authoring barrier.
        bound = self.__egg_bound__
        if isinstance(bound, RuntimeClass) and (pending := bound.__egg_pending__) is not None:
            pending()
        return self

    @property
    def __egg_ref__(self) -> Ref:
        """Materialize raw Python source once, publishing only a complete record."""
        with self.__egg_lock__:
            if self.__egg_intrinsic__ is not None:
                msg = "Intrinsic function application has no canonical declaration"
                raise TypeError(msg)
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
                if not isinstance(reference, Ref) or reference.role not in {
                    "constructors",
                    "functions",
                    "relations",
                    "user_defined_primitives",
                    "primitives",
                }:
                    msg = "Function source must publish a canonical callable definition"
                    raise TypeError(msg)  # noqa: TRY301 -- invalid preparation is a terminal cached error
                reference = reference.owner.ref(reference.role, reference.index)
                if self.__egg_pending_declaration__ is not None:
                    assert isinstance(self.__egg_bound__, RuntimeClass)
                    context = self.__egg_bound__.__egg_context__
                    assert context is not None
                    assert context.catalog is not None
                    context.catalog = Catalog.from_refs([*context.catalog.roots, reference])
                    self.__egg_pending_declaration__ = None
                    self.__egg_pending_presentation__ = None
                    self.__egg_pending_values__ = False
                self.__egg_declaration__ = reference
            except Exception as error:
                self.__egg_error__ = error
                raise
            finally:
                self.__egg_prepare__ = None
                self.__egg_resolving__ = False
            return self.__egg_declaration__

    @property
    def __module__(self) -> str | None:  # type: ignore[override]
        if self.__egg_intrinsic__ is not None:
            return "egglog.builtins"
        if self.__egg_view__ is None or not presentation(self.__egg_ref__).views:
            return None
        view = presentation(self.__egg_ref__).views[self.__egg_view__]
        if view.owner is not None:
            return _class_for_sort(view.owner).__module__
        return ".".join(view.path[:-1]) or None

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        if self.__egg_intrinsic__ is not None:
            return None
        record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = (
            self.__egg_ref__.read()
        )
        return record.doc if record.has_field("doc") else None

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        from .conversion import resolve_literal  # noqa: PLC0415

        bound = self.__egg_bound__
        owner = bound.__egg_sort__ if isinstance(bound, RuntimeClass) else None
        receiver = bound if isinstance(bound, RuntimeExpr) else _OMITTED
        result = _author_call(
            self.__egg_intrinsic__ or self.__egg_ref__,
            self.__egg_view__,
            args,
            kwargs,
            node_of=lambda value: value.__egg_ref__ if isinstance(value, RuntimeExpr) else None,
            convert=lambda sort, value: resolve_literal(sort, value).__egg_ref__,
            receiver=receiver,
            owner=owner,
            type_bindings=self.__egg_type_bindings__,
        )
        assert isinstance(result, tuple)
        root, mutated = result
        if mutated is not None:
            if isinstance(mutated, RuntimeExpr):
                mutated.__egg_ref__ = root
            return None
        return RuntimeExpr(root)

    def __str__(self) -> str:
        if self.__egg_intrinsic__ is not None:
            return f"{self.__egg_bound__ if self.__egg_bound__ is not None else 'UnstableFn'}.__call__"
        record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = (
            self.__egg_ref__.read()
        )
        kind = record
        if self.__egg_view__ is None:
            return kind.name
        view = presentation(self.__egg_ref__).views[self.__egg_view__]
        if view.kind == CallKind.FUNCTION:
            return view.path[-1]
        name = "__init__" if view.kind == CallKind.INITIALIZER else view.path[0]
        return f"{self.__egg_bound__}.{name}"

    def __repr__(self) -> str:
        return str(self)

    def _identity(self) -> tuple[object, ...]:
        """Compare callable records or named user functions, without binding."""
        if self.__egg_intrinsic__ is not None:
            return (self.__egg_intrinsic__,)
        reference = self.__egg_ref__
        record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = reference.read()
        if (
            isinstance(record, pb.Function | pb.Constructor | pb.UserDefinedPrimitive)
            and self.__egg_view__ is not None
            and presentation(self.__egg_ref__).views
        ):
            view = presentation(self.__egg_ref__).views[self.__egg_view__]
            if view.kind == CallKind.FUNCTION:
                return "function", tuple(view.path)
        if isinstance(record, pb.Primitive):
            selected = None
            if self.__egg_view__ is not None:
                view = presentation(reference).views[self.__egg_view__]
                selected = (
                    view.kind,
                    view.path,
                    StructuralView(view.owner) if view.owner is not None else None,
                    view.receiver,
                    tuple(
                        (
                            parameter.core_input,
                            parameter.name,
                            StructuralView(parameter.default) if parameter.default is not None else None,
                        )
                        for parameter in view.params
                    ),
                    view.mutates,
                    view.tail_repetitions,
                )
            return StructuralView(reference), selected
        return reference, self.__egg_view__

    def __hash__(self) -> int:
        identity = (*self._identity(), self.__egg_bound__)
        if self.__egg_type_bindings__:
            identity += (tuple((index, StructuralView(sort)) for index, sort in self.__egg_type_bindings__),)
        return hash(identity)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeFunction):
            return NotImplemented
        return (
            self._identity() == other._identity()
            and tuple((index, StructuralView(sort)) for index, sort in self.__egg_type_bindings__)
            == tuple((index, StructuralView(sort)) for index, sort in other.__egg_type_bindings__)
            and bool(self.__egg_bound__ == other.__egg_bound__)
        )


def _function_value(
    function: RuntimeFunction,
    captures: tuple[object, ...],
    expected: Ref | None,
    *,
    context: tuple[list[Ref | None], Ref | None] | None = None,
    environment: tuple[Owner, ...] = (),
) -> RuntimeExpr:
    """Author a closed named prefix or a Lambda binding intrinsic application."""
    sort, children, application_sort = _partial_function(
        function, captures, expected, context=context, environment=environment
    )
    if function.__egg_intrinsic__ is _Intrinsic.APPLY:
        assert application_sort is not None
        application_type: pb.Sort = application_sort.read()
        assert application_type.kind is not None
        assert application_type.kind.field == "func"
        signature = application_type.kind.value
        builder = Builder(environment=environment)
        arguments = [
            builder.add(
                "nodes",
                pb.Node(
                    sort_id=builder.import_ref(parameter),
                    kind=Oneof[Literal["var"], str]("var", f"_{index}"),
                ),
            )
            for index, parameter in enumerate([
                application_sort,
                *(application_sort.owner.ref("sorts", index) for index in signature.params),
            ])
        ]
        body = builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(application_sort.owner.ref("sorts", signature.result)),
                kind=Oneof[Literal["apply"], pb.Apply]("apply", pb.Apply(function=arguments[0], args=arguments[1:])),
            ),
        )
        node = builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(sort),
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(
                        value=Oneof[Literal["lambda"], pb.Lambda](
                            "lambda",
                            pb.Lambda(
                                captures=[builder.import_ref(child.__egg_ref__) for child in children],
                                body=body,
                            ),
                        )
                    ),
                ),
            ),
        )
        return RuntimeExpr(builder.publish().ref("nodes", node))
    record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = (
        function.__egg_ref__.read()
    )
    kind = record
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
                            func=kind.name,
                            args=[builder.import_ref(child.__egg_ref__) for child in children],
                        ),
                    )
                ),
            ),
        ),
    )
    return RuntimeExpr(builder.publish().ref("nodes", node))


def _partial_function(  # noqa: C901, PLR0912
    function: RuntimeFunction,
    captures: tuple[object, ...],
    expected: Ref | None,
    *,
    convert_captures: bool = True,
    context: tuple[list[Ref | None], Ref | None] | None = None,
    environment: tuple[Owner, ...] = (),
) -> tuple[Ref, tuple[RuntimeExpr, ...], Ref | None]:
    """
    Type a named prefix against its complete call and preserve its Python order.

    Construction and inert decoding share this boundary: selected presentations
    must preserve every core position, and mutation is rejected before conversion.
    All inference operates on the actual canonical records, never a dummy call.
    Decoded captures are validated as stored values; only authoring may convert.
    Context contributes only independently closed components, never variables
    from another signature's binder. Missing components remain constraints to solve.
    """
    from .conversion import resolve_literal  # noqa: PLC0415

    declaration = function.__egg_intrinsic__ or function.__egg_ref__
    record = declaration.read() if isinstance(declaration, Ref) else None
    kind = record
    view = None
    if function.__egg_view__ is not None:
        assert isinstance(declaration, Ref)
        if not presentation(declaration).views:
            msg = "Function value Python view is absent"
            raise ValueError(msg)
        view = presentation(declaration).views[function.__egg_view__]
        if view.mutates is not None:
            msg = "Mutable partial functions not supported"
            raise AssertionError(msg)
    bound = function.__egg_bound__
    if isinstance(bound, RuntimeExpr):
        captures = (bound, *captures)
    expected_params: list[Ref | None] | None = None
    expected_result = None
    if expected is not None:
        expected_type: pb.Sort = expected.read()
        if expected_type.kind is None or expected_type.kind.field != "func":
            msg = "Function value requires a Func sort"
            raise TypeError(msg)
        infer_sort(expected, expected, 0, {})
        expected_params = [expected.owner.ref("sorts", index) for index in expected_type.kind.value.params]
        expected_result = expected.owner.ref("sorts", expected_type.kind.value.result)
    elif context is not None:
        expected_params, expected_result = context
        # Only closed components cross into this callable's independent binder.
        for component in [*expected_params, expected_result]:
            if component is not None:
                infer_sort(component, component, 0, {})
    substitutions: dict[int, Ref] = dict(function.__egg_type_bindings__)
    binder_size = 0
    tail: list[Ref] = []
    application = declaration is _Intrinsic.APPLY
    if application:
        if captures and isinstance(captures[0], RuntimeExpr):
            first = captures[0].__egg_ref__
            function_sort = first.owner.ref("sorts", first.read().sort_id)
        elif isinstance(bound, RuntimeClass) and bound.__egg_sort__ is not None:
            function_sort = bound.__egg_sort__
        elif expected_params and expected_params[0] is not None:
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
    elif isinstance(kind, pb.Primitive):
        assert isinstance(declaration, Ref)
        signature = kind.signature
        if signature is None or not signature.has_field("output"):
            msg = "Function value signature has no output"
            raise ValueError(msg)
        binder_size = signature.type_param_count
        patterns = [declaration.owner.ref("sorts", argument) for argument in signature.inputs]
        tail = [declaration.owner.ref("sorts", argument) for argument in signature.varargs]
        output = declaration.owner.ref("sorts", signature.output)
    elif isinstance(kind, pb.Constructor | pb.Function | pb.UserDefinedPrimitive):
        assert isinstance(declaration, Ref)
        patterns = [declaration.owner.ref("sorts", argument) for argument in kind.inputs]
        output = declaration.owner.ref("sorts", kind.output)
    else:
        msg = "Function value requires a named value callable"
        raise TypeError(msg)
    fixed = len(patterns)
    specialized = view is not None and (view.tail_repetitions is not None)
    consumes_tail = view is not None and not specialized and (view.receiver is not None) and view.receiver == fixed
    if view is not None and consumes_tail and (len(tail) != 1 or view.kind != CallKind.METHOD):
        msg = "Tail receiver requires a homogeneous variadic method"
        raise ValueError(msg)
    if application:
        patterns.extend(tail)
    elif tail:
        total = (
            len(captures) + len(expected_params)
            if expected_params is not None
            else fixed + len(tail) * view.tail_repetitions
            if view is not None and view.tail_repetitions is not None
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
        if consumes_tail and len(patterns) <= fixed:
            msg = "Consuming receiver view requires a tail operand"
            raise TypeError(msg)
        order = [view.receiver] if (view.receiver is not None) else []
        for parameter in view.params:
            order.extend(
                range(fixed + consumes_tail, len(patterns))
                if (tail or application) and not specialized and parameter.core_input == fixed + consumes_tail
                else (parameter.core_input,)
            )
        if order != list(range(len(patterns))):
            msg = "Reordered Python function-value presentations are not supported"
            raise NotImplementedError(msg)
        if view.tail_repetitions is not None and (
            not tail or len(patterns) != fixed + len(tail) * view.tail_repetitions
        ):
            msg = "Function value does not match the selected tail specialization"
            raise TypeError(msg)
        if (view.owner is not None) and isinstance(bound, RuntimeClass) and bound.__egg_sort__ is not None:
            infer_sort(view.owner, bound.__egg_sort__, binder_size, substitutions)
    for pattern, value in zip(patterns, captures, strict=False):
        inferred_value = value
        pattern_kind = pattern.read().kind
        if (
            convert_captures
            and pattern_kind is not None
            and pattern_kind.field == "func"
            and isinstance(value, RuntimeFunction)
        ):
            with suppress(TypeConstraintError):
                inferred_value = _BASE_CLASSES["UnstableFn"](value)
        if isinstance(inferred_value, RuntimeExpr):
            node = inferred_value.__egg_ref__
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
            if actual_sort is not None:
                infer_sort(pattern, actual_sort, binder_size, substitutions)
    if expected_result is not None:
        infer_sort(output, expected_result, binder_size, substitutions)
    result = substitute_sort(output, substitutions)
    remaining = [substitute_sort(pattern, substitutions) for pattern in patterns[len(captures) :]]
    if len(substitutions) != binder_size:
        msg = "Not every generic parameter was determined by the function value"
        raise TypeConstraintError(msg)
    owner = patterns[0] if application else None
    if view is not None and view.owner is not None:
        owner = substitute_sort(view.owner, substitutions)
    # Inference never depends on converting a host capture. A speculative
    # prefix must be completely typable before any converter can have effects.
    children = tuple(
        resolve_literal(substitute_sort(pattern, substitutions), value)
        if convert_captures
        else cast("RuntimeExpr", value)
        for pattern, value in zip(patterns, captures, strict=False)
    )
    builder = Builder(environment=environment)
    if isinstance(declaration, Ref):
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


if TYPE_CHECKING:
    from .egraph import BaseExpr as _RuntimeExprBase
else:
    _RuntimeExprBase = object


class RuntimeExpr(_RuntimeExprBase):
    """The sole public symbolic wrapper. Its node Ref is its semantic state."""

    __slots__ = ("__egg_ref__",)

    def __init__(self, reference: Ref) -> None:
        if not isinstance(reference, Ref) or reference.role != "nodes":
            msg = "RuntimeExpr requires an owned protobuf node reference"
            raise TypeError(msg)
        self.__egg_ref__ = reference.owner.ref("nodes", reference.index)

    def __getattr__(self, name: str | RuntimeExpr) -> object:
        if isinstance(name, str):
            if name.startswith("__egg_"):
                raise AttributeError(name)
            result = _get_expr_member(self, name, missing_ok=True)
            if result is not _OMITTED:
                return result
        fallback = _get_expr_member(self, "__getattr__", missing_ok=True, method_only=True)
        if fallback is _OMITTED:
            raise AttributeError(name)
        return cast("Callable", fallback)(name)

    def __str__(self) -> str:
        node: pb.Node = self.__egg_ref__.read()
        cls = _class_for_sort(self.__egg_ref__.owner.ref("sorts", node.sort_id))
        if "__str__" in cls.__egg_hooks__:
            return cls.__egg_hooks__["__str__"].__get__(self, cls)()
        from .pretty import _format_pyobject, pretty_ref  # noqa: PLC0415

        return pretty_ref(self.__egg_ref__, custom_literal=_format_pyobject)

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
        member = _get_expr_member(self, "__eq__", missing_ok=True, method_only=True)
        if member is not _OMITTED:
            return cast("Callable", member)(other)
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
        member = _get_expr_member(self, "__ne__", missing_ok=True, method_only=True)
        if member is not _OMITTED:
            return cast("Callable", member)(other)
        from .egraph import ne  # noqa: PLC0415

        return ne(cast("Any", self)).to(other)

    def __copy__(self) -> RuntimeExpr:
        return RuntimeExpr(self.__egg_ref__)

    def __replace_expr__(self, other: RuntimeExpr) -> None:
        self.__egg_ref__ = other.__egg_ref__


def _get_expr_member(
    expression: RuntimeExpr, name: str, *, missing_ok: bool = False, method_only: bool = False
) -> object:
    """Resolve a genuine member without dynamic fallback or swallowed user errors."""
    node: pb.Node = expression.__egg_ref__.read()
    cls = _class_for_sort(expression.__egg_ref__.owner.ref("sorts", node.sort_id))
    if name in cls.__egg_hooks__:
        hook = cls.__egg_hooks__[name]
        if method_only and not isinstance(hook, FunctionType):
            raise AttributeError(f"{cls} member {name!r} is not an instance method")
        return None if hook is None else hook.__get__(expression, cls)
    function = RuntimeClass.__getattr__(cls, name, missing_ok, method_only)
    if not isinstance(function, RuntimeFunction):
        return function
    result = RuntimeFunction(function.__egg_intrinsic__ or function.__egg_ref__, function.__egg_view__, expression)
    if (
        function.__egg_view__ is not None
        and presentation(function.__egg_ref__).views[function.__egg_view__].kind == CallKind.PROPERTY
    ):
        return result()
    return result


def _numeric_method_cost(function: RuntimeFunction, owner: Ref, left: object, right: object) -> int | None:
    """Price the actual mapped signature without materializing converted operands."""
    from .conversion import _conversion_cost, resolve_literal  # noqa: PLC0415

    if function.__egg_intrinsic__ is not None or function.__egg_view__ is None:
        return None
    if presentation(function.__egg_ref__).views[function.__egg_view__].mutates is not None:
        return None
    plan = _author_call(
        function.__egg_ref__,
        function.__egg_view__,
        (right,),
        {},
        node_of=lambda value: value.__egg_ref__ if isinstance(value, RuntimeExpr) else None,
        convert=lambda sort, value: resolve_literal(sort, value).__egg_ref__,
        receiver=left,
        owner=owner,
        type_bindings=function.__egg_type_bindings__,
        plan_only=True,
    )
    assert isinstance(plan, list)
    costs = [_conversion_cost(sort, value) for sort, value in plan]
    return None if not plan or None in costs else sum(cost for cost in costs if cost is not None)


def _numeric_binary_call(name: str, left: object, right: object) -> object:  # noqa: C901
    """Choose the cheapest applicable exact signature, then run only its conversions."""
    from .conversion import _binary_conversion_candidates, convert_to_same_type, resolve_literal  # noqa: PLC0415

    original = left if isinstance(left, RuntimeExpr) else right
    assert isinstance(original, RuntimeExpr)
    member = _get_expr_member(original, name, missing_ok=True, method_only=True)
    mutating = (
        isinstance(member, RuntimeFunction)
        and member.__egg_view__ is not None
        and presentation(member.__egg_ref__).views[member.__egg_view__].mutates is not None
    )
    if member is not _OMITTED and (not isinstance(member, RuntimeFunction) or mutating):
        if isinstance(left, RuntimeExpr):
            return cast("Callable", member)(right)
        return cast("Callable", _get_expr_member(convert_to_same_type(left, original), name))(right)
    best: tuple[int, RuntimeClass, RuntimeFunction] | None = None
    candidates = _binary_conversion_candidates(left, right)
    while True:
        try:
            cls, left_cost, minimum_other = candidates.send(best[0] if best is not None else None)
        except StopIteration:
            break
        sort = cls.__egg_sort__
        assert sort is not None
        if best is not None and left_cost + minimum_other >= best[0]:
            continue
        function = RuntimeClass.__getattr__(cls, name, True, True)
        if not isinstance(function, RuntimeFunction):
            continue
        cost = _numeric_method_cost(function, sort, left, right)
        if cost is not None and (best is None or cost < best[0]):
            best = cost, cls, function
            if cost == 0:
                break
    if best is not None:
        _, cls, function = best
        assert cls.__egg_sort__ is not None
        receiver = resolve_literal(cls.__egg_sort__, left)
        return RuntimeFunction(
            function.__egg_ref__, function.__egg_view__, receiver, type_bindings=function.__egg_type_bindings__
        )(right)
    if not isinstance(left, RuntimeExpr):
        left = convert_to_same_type(left, original)
    assert isinstance(left, RuntimeExpr)
    return cast("Callable", _get_expr_member(left, name))(right)


def define_expr_method(name: str, *, reverse: bool = False) -> None:
    """Install a Python special-method entrypoint using the generated member view."""

    def invoke(self: RuntimeExpr, *args: object, **kwargs: object) -> object:
        if name in NUMERIC_BINARY_METHODS and len(args) == 1 and not kwargs:
            return _numeric_binary_call(name, self, args[0])
        return cast("Callable", _get_expr_member(self, name))(*args, **kwargs)

    def reflected(self: RuntimeExpr, other: object) -> object:
        return _numeric_binary_call(name, other, self)

    setattr(RuntimeExpr, f"__r{name[2:]}" if reverse else name, reflected if reverse else invoke)


for _name in TYPE_DEFINED_METHODS - {"__hash__"}:
    define_expr_method(_name)

for _name in NUMERIC_BINARY_METHODS:
    define_expr_method(_name)
    define_expr_method(_name, reverse=True)


def resolve_type_annotation(tp: object) -> Ref | TypeVar | RuntimeClass:
    """Keep unresolved Python generics as language state; concrete types are refs."""
    if tp is object:
        tp = _BASE_CLASSES["PyObject"]
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
        if isinstance(tp, _RuntimeClassAlias) and tp.__parameters__:
            return RuntimeClass(origin.__egg_definition__, arguments=get_args(tp))
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


def create_callable(
    reference: Ref,
    *,
    owner_sort: Ref | None = None,
    view_index: int | None = 0,
    type_bindings: tuple[tuple[int, Ref], ...] = (),
) -> RuntimeClass | RuntimeFunction | RuntimeExpr:
    """Recover the first canonical Python callable view without evaluating it."""
    record: pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive | pb.Primitive = reference.read()
    if view_index is None or not presentation(reference).views:
        return RuntimeFunction(reference, type_bindings=type_bindings)
    view = presentation(reference).views[view_index]
    owner = None
    if view.owner is not None:
        owner = _class_for_sort(owner_sort) if owner_sort is not None else _class_for_sort(view.owner)
    if view.kind == CallKind.INITIALIZER:
        if owner is None:
            msg = "An initializer view requires a Python owner"
            raise ValueError(msg)
        return owner
    if view.kind not in {CallKind.CONSTANT, CallKind.CLASS_VARIABLE}:
        return RuntimeFunction(reference, view_index, owner, type_bindings=type_bindings)
    if not isinstance(record, pb.Function | pb.Constructor | pb.UserDefinedPrimitive) or record.inputs:
        msg = "A constant view requires a nullary concrete value callable"
        raise ValueError(msg)
    signature = record
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
