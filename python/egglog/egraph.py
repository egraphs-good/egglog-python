from __future__ import annotations

import contextlib
import dis
import inspect
import json
import re
import sys
from collections.abc import Callable, Generator, Iterable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import partial
from inspect import Parameter, currentframe, getmodule, signature
from operator import index as index_integer
from types import CellType, FrameType, FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Never,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    get_type_hints,
    overload,
)
from uuid import uuid4

if sys.version_info >= (3, 14):
    import annotationlib

from egglog_proto.egglog.v1 import egglog_pb as pb
from opentelemetry import trace
from protobuf import Oneof
from typing_extensions import ParamSpec, TypeForm

from ._catalog import Catalog, builtin_catalog
from ._program import (
    Builder,
    Packed,
    Ref,
    StructuralView,
    _copy_record,
    _structural_parts,
    action_equal,
    pack,
    rule_equal,
)
from .conversion import *
from .conversion import convert_to_same_type, resolve_literal
from .declarations import Ident, RuleEvalMode
from .egraph_state import *
from .egraph_state import _CostModelObservation
from .pretty import _format_pyobject, pretty_ref
from .run_report import RunReport
from .runtime import *
from .runtime import _ClassContext, _definition_at
from .thunk import *

if TYPE_CHECKING:
    import graphviz

    from .builtins import Unit, i64, i64Like
    from .visualizer_widget import _Graph


_TRACER = trace.get_tracer(__name__)


__all__ = [
    "Action",
    "ActionLike",
    "BackOff",
    "BaseExpr",
    "BuiltinExpr",
    "Command",
    "CostModel",
    "DagCostModel",
    "EGraph",
    "Expr",
    "ExprCallable",
    "ExtractionMode",
    "Fact",
    "GraphvizKwargs",
    "RewriteOrRule",
    "RuleEvalMode",
    "Ruleset",
    "RunReport",
    "Schedule",
    "TreeCostModel",
    "_BirewriteBuilder",
    "_EqBuilder",
    "_NeBuilder",
    "_RewriteBuilder",
    "_SetBuilder",
    "_UnionBuilder",
    "back_off",
    "birewrite",
    "check",
    "check_eq",
    "constant",
    "default_cost_model",
    "delete",
    "eq",
    "expr_action",
    "expr_fact",
    "expr_parts",
    "function",
    "get_cost",
    "let",
    "method",
    "ne",
    "panic",
    "relation",
    "rewrite",
    "rule",
    "ruleset",
    "run",
    "seq",
    "set_",
    "set_cost",
    "set_current_ruleset",
    "subsume",
    "union",
    "unstable_combine_rulesets",
    "var",
    "vars_",
]


T = TypeVar("T")
P = ParamSpec("P")
EXPR_TYPE = TypeVar("EXPR_TYPE", bound="type[Expr]")
BASE_EXPR_TYPE = TypeVar("BASE_EXPR_TYPE", bound="type[BaseExpr]")
EXPR = TypeVar("EXPR", bound="Expr")
BASE_EXPR = TypeVar("BASE_EXPR", bound="BaseExpr")
BE1 = TypeVar("BE1", bound="BaseExpr")
BE2 = TypeVar("BE2", bound="BaseExpr")
BE3 = TypeVar("BE3", bound="BaseExpr")
BE4 = TypeVar("BE4", bound="BaseExpr")
_MISSING = object()
# Attributes which are sometimes added to classes by the interpreter or the dataclass decorator, or by ipython.
# We ignore these when inspecting the class.

IGNORED_ATTRIBUTES = {
    "__module__",
    "__doc__",
    "__dict__",
    "__weakref__",
    "__orig_bases__",
    "__annotations__",
    "__qualname__",
    "__firstlineno__",
    "__static_attributes__",
    "__match_args__",
    # Added in 3.14
    # https://egraphs.zulipchat.com/#narrow/channel/375765-egg.2Fegglog/topic/Cost.20function.3A.20using.20function.20values.20of.20subtrees/near/577236488
    "__classdictcell__",
    # Ignore all reflected binary method
    *(f"__r{m[2:]}" for m in NUMERIC_BINARY_METHODS),
}


def check_eq(x: BASE_EXPR, y: BASE_EXPR, schedule: Schedule | None = None, *, add_second=True, display=False) -> EGraph:
    """
    Verifies that two expressions are equal after running the schedule.

    If add_second is true, then the second expression is added to the egraph before running the schedule.
    """
    egraph = EGraph()
    x_var = egraph.let("__check_eq_x", x)
    y_var: BASE_EXPR = egraph.let("__check_eq_y", y) if add_second else y
    if schedule:
        try:
            egraph.run(schedule)
        finally:
            if display:
                egraph.display()
    fact = eq(x_var).to(y_var)
    try:
        egraph.check(fact)
    except EggSmolError as err:
        if display:
            egraph.display()
        err.add_note(f"Failed:\n{eq(x).to(y)}\n\nExtracted:\n {eq(egraph.extract(x)).to(egraph.extract(y))})")
        raise
    return egraph


def check(x: FactLike, schedule: Schedule | None = None, *given: ActionLike) -> None:
    """
    Verifies that the fact is true given some assumptions and after running the schedule.
    """
    egraph = EGraph()
    if given:
        egraph.register(*given)
    if schedule:
        egraph.run(schedule)
    egraph.check(x)


# We seperate the function and method overloads to make it simpler to know if we are modifying a function or method,
# So that we can add the functions eagerly to the registry and wait on the methods till we process the class.


def _resolve_merge(
    return_type: Ref,
    merge: Callable[[object, object], object] | None,
) -> Ref | None:
    if merge is None:
        return None
    builder = Builder()
    parameters = [
        builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(return_type),
                kind=Oneof[Literal["var"], str]("var", name),
            ),
        )
        for name in ("old", "new")
    ]
    owner = builder.publish()
    with set_current_ruleset(None):
        resolved = resolve_literal(
            return_type, merge(*(RuntimeExpr(owner.ref("nodes", index)) for index in parameters))
        )
    return resolved.__egg_ref__


def _function_has_body(fn: FunctionType) -> bool:
    """Return whether a function does more than implicitly return ``None``."""
    instructions = [
        (instruction.opname, instruction.argval)
        for instruction in dis.get_instructions(fn)
        if instruction.opname not in {"CACHE", "NOP", "RESUME"}
    ]
    return instructions not in (
        [("RETURN_CONST", None)],
        [("LOAD_CONST", None), ("RETURN_VALUE", None)],
    )


CALLABLE = TypeVar("CALLABLE", bound=Callable)
CONSTRUCTOR_CALLABLE = TypeVar("CONSTRUCTOR_CALLABLE", bound=Callable[..., "Expr | None"])

EXPR_NONE = TypeVar("EXPR_NONE", bound="Expr | None")
BASE_EXPR_NONE = TypeVar("BASE_EXPR_NONE", bound="BaseExpr | None")
USER_EXPR = TypeVar("USER_EXPR", bound="Expr")
BUILTIN_EXPR = TypeVar("BUILTIN_EXPR", bound="BuiltinExpr")


@overload
def method(
    *,
    preserve: Literal[True],
) -> Callable[[CALLABLE], CALLABLE]: ...


# function wihout merge
@overload
def method(
    *,
    egg_fn: str | None = ...,
    reverse_args: bool = ...,
    mutates_self: bool = ...,
) -> Callable[[CALLABLE], CALLABLE]: ...


# function
@overload
def method(
    *,
    egg_fn: str | None = ...,
    merge: Callable[[BASE_EXPR, BASE_EXPR], BASE_EXPR] | None = ...,
    mutates_self: bool = ...,
) -> Callable[[Callable[P, BASE_EXPR]], Callable[P, BASE_EXPR]]: ...


# constructor
@overload
def method(
    *,
    egg_fn: str | None = ...,
    cost: int | None = ...,
    mutates_self: bool = ...,
    unextractable: bool = ...,
    subsume: bool = ...,
) -> Callable[[Callable[P, EXPR_NONE]], Callable[P, EXPR_NONE]]: ...


def method(
    *,
    egg_fn: str | None = None,
    cost: int | None = None,
    merge: Callable[[BASE_EXPR, BASE_EXPR], BASE_EXPR] | None = None,
    preserve: bool = False,
    mutates_self: bool = False,
    unextractable: bool = False,
    subsume: bool = False,
    reverse_args: bool = False,
) -> Callable[[Callable[P, BASE_EXPR_NONE]], Callable[P, BASE_EXPR_NONE]]:
    """
    Any method can be decorated with this to customize it's behavior. This is only supported in classes which subclass :class:`Expr`.
    """
    merge = cast("Callable[[object, object], object]", merge)
    return lambda fn: _WrappedMethod(
        egg_fn, cost, merge, fn, preserve, mutates_self, unextractable, subsume, reverse_args
    )


@overload
def function(fn: CALLABLE, /) -> CALLABLE: ...


# function without merge
@overload
def function(
    *,
    egg_fn: str | None = ...,
    builtin: bool = ...,
    mutates_first_arg: bool = ...,
) -> Callable[[CALLABLE], CALLABLE]: ...


# function
@overload
def function(
    *,
    egg_fn: str | None = ...,
    merge: Callable[[BASE_EXPR, BASE_EXPR], BASE_EXPR] | None = ...,
    mutates_first_arg: bool = ...,
) -> Callable[[Callable[P, BASE_EXPR]], Callable[P, BASE_EXPR]]: ...


@overload
def function(
    *,
    egg_fn: str | None = ...,
    cost: int | None = ...,
    mutates_first_arg: bool = ...,
    unextractable: bool = ...,
    ruleset: Ruleset | None = ...,
    subsume: bool = ...,
) -> Callable[[CONSTRUCTOR_CALLABLE], CONSTRUCTOR_CALLABLE]: ...


def function(*args, **kwargs) -> Any:
    """
    Decorate a function typing stub to create an egglog function for it.

    Python automatically infers whether this lowers to a `function`, `constructor`,
    or eager `primitive` from the return type and whether a body/default exists.

    Bodies and defaults lower eagerly unless an explicit `ruleset` is provided,
    in which case they are added as rewrite-backed defaults in that ruleset.
    Direct top-level functions can only use an explicit `ruleset` when they
    provide a body to lower as a rewrite. `subsume` is only valid for eqsort-
    returning bodies on that explicit-ruleset rewrite path.
    """
    # Resolve annotations on the defining thread: reading its live locals from
    # another thread is unsafe on free-threaded Python.
    # https://docs.python.org/3.14/howto/free-threading-python.html#frame-objects
    fn_locals = currentframe().f_back.f_locals  # type: ignore[union-attr]

    # If we have any positional args, then we are calling it directly on a function
    if args:
        if len(args) != 1 or kwargs:
            msg = "function() accepts either a function or keyword options, not both"
            raise TypeError(msg)
        return _FunctionConstructor(fn_locals)(args[0])
    # otherwise, we are passing some keyword args, so save those, and then return a partial
    return _FunctionConstructor(fn_locals, **kwargs)


class _ExprMetaclass(type):
    """
    Metaclass of Expr.

    Used to override isistance checks, so that runtime expressions are instances of Expr at runtime.
    """

    def __new__(
        cls: type[_ExprMetaclass],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        egg_sort: str | None = None,
        ruleset: Ruleset | None = None,
    ) -> RuntimeClass | type:
        # If this is the Expr subclass, just return the class
        if not bases or bases == (BaseExpr,):
            return super().__new__(cls, name, bases, namespace)
        frame = currentframe()
        assert frame
        prev_frame = frame.f_back
        assert prev_frame
        hooks = {
            key: value.fn if isinstance(value, _WrappedMethod) else value
            for key, value in namespace.items()
            if key in ALWAYS_PRESERVED or (isinstance(value, _WrappedMethod) and value.preserve)
        }
        context = _ClassContext(
            partial(_prepare_expr_class, namespace=namespace, frame=prev_frame, bases=bases, ruleset=ruleset),
            hooks,
            namespace.get("__match_args__", ()),
        )
        module = namespace.get("__module__", "")
        path = [*module.split("."), name] if module else [name]
        builder = Builder()
        definition = builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["eq_sort"], pb.EqSort](
                    "eq_sort",
                    pb.EqSort(
                        name=egg_sort or ".".join(path), bindings=pb.SortBindings(python=pb.TypeBinding(path=path))
                    ),
                ),
                doc=namespace.get("__doc__"),
            ),
        )
        return RuntimeClass(builder.publish(keepalive=(context,)).ref("declarations", definition))

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, RuntimeExpr)


class BaseExpr(metaclass=_ExprMetaclass):
    """
    Either a builtin or a user defined expression type.
    """

    # these methods are only provided for type checking
    # The real implementations are in the runtime system

    def __ne__(self, other: Self) -> Unit: ...  # type: ignore[override, empty-body]

    # not currently dissalowing other types of equality https://github.com/python/typeshed/issues/8217#issuecomment-3140873292
    def __eq__(self, other: Self) -> Fact: ...  # type: ignore[override, empty-body]

    def __replace_expr__(self, new_expr: Self) -> None:
        """
        Replace the current expression with the new expression in place.
        """

    def __hash__(self) -> int: ...  # type: ignore[empty-body]


class BuiltinExpr(BaseExpr, metaclass=_ExprMetaclass):
    """
    A builtin expr type, not an eqsort.
    """


class Expr(BaseExpr, metaclass=_ExprMetaclass):
    """
    Subclass this to define a custom expression type.
    """


def _prepare_expr_class(  # noqa: C901, PLR0912
    runtime_cls: RuntimeClass,
    *,
    namespace: dict[str, Any],
    frame: FrameType,
    bases: tuple[type, ...],
    ruleset: Ruleset | None,
) -> Catalog:
    """Stage member headers before source-ordered value/body completion."""
    if bases != (Expr,):
        msg = "Only direct nongeneric Expr classes are migrated"
        raise NotImplementedError(msg)
    context = runtime_cls.__egg_context__
    assert context is not None
    definitions = [runtime_cls.__egg_definition__]
    context.catalog = Catalog.from_refs(definitions)
    runtime_cls.__egg_attr_cache__.clear()
    hint_locals = {**frame.f_locals, runtime_cls.__name__: runtime_cls}
    annotations = namespace.get("__annotations__", {})
    if (
        sys.version_info >= (3, 14)
        and (annotate := annotationlib.get_annotate_from_class_namespace(namespace)) is not None
    ):
        annotations = annotationlib.call_annotate_function(annotate, annotationlib.Format.FORWARDREF)
    hints = get_type_hints(
        type("_ClassAnnotations", (), {"__annotations__": annotations}),
        globalns=frame.f_globals,
        localns=hint_locals,
    )
    assert runtime_cls.__egg_sort__ is not None
    for name, annotation in hints.items():
        if getattr(annotation, "__origin__", None) is not ClassVar:
            raise NotImplementedError(f"On class {runtime_cls}, for attribute {name!r}, expected a ClassVar")
        sort = resolve_type_annotation(annotation.__args__[0])
        if not isinstance(sort, Ref) or sort.role != "sorts":
            raise NotImplementedError(f"Class variable {runtime_cls}.{name} requires a concrete sort")
        builder = Builder()
        output = builder.import_ref(sort)
        sort_record: pb.Sort = sort.read()
        if name in namespace and ruleset is not None and (sort_record.kind is None or sort_record.kind.field != "eq"):
            msg = "Primitive-returning defaults cannot use an explicit ruleset"
            raise ValueError(msg)
        qualified = f"{runtime_cls.__module__}.{runtime_cls.__name__}.{name}"
        egg_name = re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", qualified)
        body = None
        if name in namespace and ruleset is None:
            with set_current_ruleset(None):
                default = resolve_literal(sort, namespace[name])
            body = _lower_function_body(builder, default.__egg_ref__, [])
        kind = (
            Oneof[Literal["primitive"], pb.Primitive](
                "primitive", pb.Primitive(name=egg_name, output=output, body=body)
            )
            if body is not None
            else Oneof[Literal["constructor"], pb.Constructor](
                "constructor", pb.Constructor(name=egg_name, output=output)
            )
            if sort_record.kind is not None and sort_record.kind.field == "eq"
            else Oneof[Literal["function"], pb.Function]("function", pb.Function(name=egg_name, output=output))
        )
        index = builder.add(
            "declarations",
            pb.Declaration(
                kind=kind,
                bindings=pb.CallableBindings(
                    python=pb.PythonBindings(
                        views=[
                            pb.PythonCallable(
                                kind=pb.PythonCallKind.CLASS_VARIABLE,
                                path=[name],
                                owner=pb.BindingOwner(
                                    kind=Oneof[Literal["sort"], int](
                                        "sort", builder.import_ref(runtime_cls.__egg_sort__)
                                    )
                                ),
                            )
                        ]
                    )
                ),
            ),
        )
        definitions.append(builder.publish().ref("declarations", index))
        context.catalog = Catalog.from_refs(definitions)
        if name in namespace and ruleset is not None:
            default = resolve_literal(sort, namespace[name])
            lhs = RuntimeFunction(definitions[-1], 0)()
            rewrite(cast("Expr", lhs), ruleset=ruleset).to(cast("Expr", default))
    bodies: list[tuple[FunctionType, RuntimeFunction, bool]] = []
    pending: list[RuntimeFunction] = []
    values_pending = False
    for name, value in namespace.items():
        if name in hints:
            continue
        if name in {"__annotate__", "__annotate_func__"} and not isinstance(value, _WrappedMethod):
            continue
        if name in IGNORED_ATTRIBUTES and not isinstance(value, _WrappedMethod):
            continue
        if isinstance(value, _WrappedMethod):
            if (
                (value.preserve or name == "__init__")
                and (
                    (value.cost is not None and name != "__init__")
                    or value.merge is not None
                    or value.mutates_self
                    or value.unextractable
                    or value.subsume
                )
            ) or (value.preserve and (value.egg_fn is not None or name == "__init__")):
                raise NotImplementedError(f"Method options for {runtime_cls}.{name} are not migrated")
            fn = value.fn
            preserve = value.preserve
            compiler = _FunctionConstructor(
                hint_locals,
                egg_fn=value.egg_fn,
                cost=value.cost,
                merge=value.merge,
                mutates_first_arg=value.mutates_self,
                unextractable=value.unextractable,
                subsume=value.subsume,
                ruleset=ruleset,
            )
        else:
            fn, preserve = value, False
            compiler = _FunctionConstructor(hint_locals, mutates_first_arg=name in ALWAYS_MUTATES_SELF, ruleset=ruleset)
        if preserve or name in ALWAYS_PRESERVED:
            continue
        if name in IGNORED_ATTRIBUTES:
            raise NotImplementedError(f"Symbolic reflected member {runtime_cls}.{name} is not migrated")
        view_kind = pb.PythonCallKind.INITIALIZER if name == "__init__" else pb.PythonCallKind.METHOD
        if isinstance(fn, classmethod):
            fn, view_kind = fn.__func__, pb.PythonCallKind.CLASS_METHOD
        elif isinstance(fn, property):
            fn, view_kind = fn.fget, pb.PythonCallKind.PROPERTY
        if isinstance(fn, _WrappedMethod):
            msg = f"{runtime_cls}.{name} Add the @method(...) decorator above @classmethod or @property"
            raise ValueError(msg)  # noqa: TRY004 -- preserve the public decorator-order error
        function = compiler.prepare(
            fn,
            owner=runtime_cls,
            view_kind=view_kind,
            member_name=name,
            reverse_args=value.reverse_args if isinstance(value, _WrappedMethod) else False,
        )
        header = function.__egg_pending_declaration__
        assert header is not None
        assert header.kind is not None
        key = ("callable", header.kind.value.name)
        duplicate = key in context.catalog.definitions
        for previous in pending:
            previous_header = previous.__egg_pending_declaration__ or previous.__egg_ref__.read()
            assert previous_header.kind is not None
            duplicate |= previous_header.kind.value.name == key[1]
        if duplicate:
            raise ValueError(f"Duplicate catalog definition: {key}")
        assert isinstance(fn, FunctionType)
        if _function_has_body(fn) and ruleset is not None:
            bodies.append((fn, function, compiler.subsume))
        # Keep the source prefix of complete headers, but never materialize a
        # later header across a pending conversion before the static barrier.
        values_pending |= function.__egg_pending_values__
        if values_pending or header.kind.field == "primitive":
            runtime_cls.__egg_attr_cache__[name] = function
            pending.append(function)
        else:
            _ = function.__egg_ref__
    for function in pending:
        _ = function.__egg_ref__
    for fn, function, subsume_body in bodies:
        assert ruleset is not None
        reference = function.__egg_ref__
        declaration: pb.Declaration = reference.read()
        assert declaration.kind is not None
        assert isinstance(declaration.kind.value, pb.Constructor)
        assert declaration.bindings is not None
        assert declaration.bindings.python is not None
        view = declaration.bindings.python.views[0]
        builder = Builder()
        arguments: list[int] = []
        for argument in declaration.kind.value.inputs:
            index = builder.add(
                "nodes",
                pb.Node(
                    sort_id=builder.import_ref(reference.owner.ref("sorts", argument.sort)),
                    kind=Oneof[Literal["var"], str]("var", argument.name),
                ),
            )
            # All callback arguments are views over the canonical input sorts.
            arguments.append(index)
        owner = builder.publish()
        values = [RuntimeExpr(owner.ref("nodes", index)) for index in arguments]
        callback_args: list[RuntimeExpr | RuntimeClass] = [values[parameter.core_input] for parameter in view.params]
        if view.has_field("receiver"):
            callback_args.insert(0, values[view.receiver])
        elif view.kind == pb.PythonCallKind.CLASS_METHOD:
            callback_args.insert(0, runtime_cls)
        with set_current_ruleset(ruleset):
            result = fn(*callback_args)
        if view.has_field("mutates"):
            if result is not None:
                msg = "Mutating callable bodies must return None"
                raise ValueError(msg)
            result = values[view.mutates]
        if result is None:
            msg = "Callable bodies must return a value"
            raise ValueError(msg)
        output_sort = reference.owner.ref("sorts", declaration.kind.value.output)
        rhs = resolve_literal(output_sort, result)
        # A body may replace its argument wrappers. The pattern still binds the
        # original variables; reconstruct it only after body/result validation.
        values = [RuntimeExpr(owner.ref("nodes", index)) for index in arguments]
        # Calling the selected view reuses its canonical receiver/slot ordering.
        lhs = RuntimeFunction(reference, 0)(
            *([values[view.receiver]] if view.has_field("receiver") else []),
            *(values[parameter.core_input] for parameter in view.params),
        )
        if view.has_field("mutates"):
            lhs = values[view.mutates]
        rewrite(cast("Expr", lhs), ruleset=ruleset, subsume=subsume_body).to(cast("Expr", rhs))
    return context.catalog


@dataclass
class _FunctionConstructor:
    hint_locals: dict[str, Any]
    builtin: bool = False
    mutates_first_arg: bool = False
    egg_fn: str | None = None
    cost: int | None = None
    merge: Callable[[object, object], object] | None = None
    unextractable: bool = False
    ruleset: Ruleset | None = None
    subsume: bool = False

    def __call__(self, fn: Callable) -> RuntimeFunction:
        return RuntimeFunction(partial(self.prepare, fn), 0)

    @overload
    def prepare(
        self,
        fn: object,
        *,
        owner: None = None,
        view_kind: pb.PythonCallKind = pb.PythonCallKind.FUNCTION,
        member_name: str | None = None,
        reverse_args: bool = False,
    ) -> Ref: ...

    @overload
    def prepare(
        self,
        fn: object,
        *,
        owner: RuntimeClass,
        view_kind: pb.PythonCallKind = pb.PythonCallKind.FUNCTION,
        member_name: str | None = None,
        reverse_args: bool = False,
    ) -> RuntimeFunction: ...

    def prepare(  # noqa: C901, PLR0912
        self,
        fn: object,
        *,
        owner: RuntimeClass | None = None,
        view_kind: pb.PythonCallKind = pb.PythonCallKind.FUNCTION,
        member_name: str | None = None,
        reverse_args: bool = False,
    ) -> Ref | RuntimeFunction:
        """Compile raw Python source directly, never a legacy declaration projection."""
        if isinstance(fn, RuntimeFunction):
            msg = "Inside of classes, wrap methods with the `method` decorator, not `function`"
            raise ValueError(msg)  # noqa: TRY004
        if not isinstance(fn, FunctionType):
            raise NotImplementedError(f"Can only generate function declarations for functions, not {type(fn)}")
        module_globals = vars(sys.modules[fn.__module__]) if fn.__module__ in sys.modules else {}
        hint_namespace = {**module_globals, **fn.__globals__, **self.hint_locals}
        hints = get_type_hints(fn, globalns=hint_namespace, localns=hint_namespace)
        parameters = list(signature(fn).parameters.values())
        if owner is not None:
            if not parameters or any(parameter.kind != Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters):
                msg = "Member parameters must be fixed positional-or-keyword parameters including self or cls"
                raise NotImplementedError(msg)
            if view_kind in {pb.PythonCallKind.INITIALIZER, pb.PythonCallKind.CLASS_METHOD}:
                parameters = parameters[1:]
            elif view_kind == pb.PythonCallKind.PROPERTY and len(parameters) != 1:
                msg = "A symbolic property must have only its receiver"
                raise NotImplementedError(msg)
        variable = _last_param_variable(parameters)
        if self.mutates_first_arg:
            if owner is not None and view_kind != pb.PythonCallKind.METHOD:
                msg = "Mutating non-instance views are not migrated"
                raise NotImplementedError(msg)
            if not parameters:
                msg = "Mutating callables require a first argument"
                raise ValueError(msg)
            if hints.get("return") is not type(None):
                msg = "Mutating callables must be annotated to return None"
                raise ValueError(msg)
        receiver = owner is not None and view_kind in {pb.PythonCallKind.METHOD, pb.PythonCallKind.PROPERTY}
        input_sorts = [
            owner.__egg_sort__
            if owner is not None and index == 0 and receiver
            else resolve_type_annotation(hints[parameter.name])
            for index, parameter in enumerate(parameters)
        ]
        output = (
            owner.__egg_sort__
            if owner is not None and view_kind == pb.PythonCallKind.INITIALIZER
            else input_sorts[0]
            if self.mutates_first_arg and input_sorts
            else resolve_type_annotation(hints["return"])
        )
        if not isinstance(output, Ref) or output.role != "sorts":
            msg = "Free functions require a concrete output sort"
            raise NotImplementedError(msg)
        output_record: pb.Sort = output.read()
        if output_record.kind is None or output_record.kind.field not in {"eq", "family", "func"}:
            msg = "Free functions require an equality, host-family or function output sort"
            raise NotImplementedError(msg)
        equality = output_record.kind.field == "eq"
        has_body = _function_has_body(fn)
        has_merge = self.merge is not None
        if self.builtin and has_merge:
            msg = "Builtin callables cannot use merge"
            raise ValueError(msg)
        if self.subsume:
            if not equality:
                msg = "Primitive-returning callables cannot use subsume"
                raise ValueError(msg)
            if self.ruleset is None:
                msg = "subsume requires an explicit ruleset"
                raise ValueError(msg)
        if equality and self.builtin:
            msg = "Eqsort-returning callables cannot be builtin"
            raise ValueError(msg)
        if not equality:
            if self.cost is not None:
                msg = "Primitive-returning callables cannot use cost"
                raise ValueError(msg)
            if self.unextractable:
                msg = "Primitive-returning callables cannot be unextractable"
                raise ValueError(msg)
            if output_record.kind.field == "family" and output_record.kind.value.name == "Unit" and has_merge:
                msg = "Functions that return Unit cannot use merge"
                raise ValueError(msg)
        elif has_merge:
            if self.cost is not None:
                msg = "Cost can only be set for constructors"
                raise ValueError(msg)
            if self.unextractable:
                msg = "Unextractable can only be set for constructors"
                raise ValueError(msg)
        if self.builtin and self.ruleset is not None:
            msg = "Builtin callables cannot use an explicit ruleset"
            raise ValueError(msg)
        if self.builtin and has_body:
            msg = "Builtin callables cannot have a body"
            raise ValueError(msg)
        if has_body:
            if equality:
                if has_merge:
                    msg = "Eqsort-returning callables with bodies cannot use merge"
                    raise ValueError(msg)
                if self.ruleset is None and self.cost is not None:
                    msg = "Eqsort-returning eager bodies cannot use cost"
                    raise ValueError(msg)
                if self.ruleset is None and self.unextractable:
                    msg = "Eqsort-returning eager bodies cannot be unextractable"
                    raise ValueError(msg)
            else:
                if self.ruleset is not None:
                    msg = "Primitive-returning callables with bodies cannot use an explicit ruleset"
                    raise ValueError(msg)
                if has_merge:
                    msg = "Primitive-returning callables with bodies cannot use merge"
                    raise ValueError(msg)
            if owner is not None and (
                (self.ruleset is None and self.mutates_first_arg)
                or (self.ruleset is not None and view_kind == pb.PythonCallKind.INITIALIZER)
            ):
                msg = "Eager member and initializer bodies are not migrated"
                raise NotImplementedError(msg)
        if self.ruleset is not None and owner is None:
            if not has_body:
                msg = "Explicit rulesets require a body"
                raise ValueError(msg)
            if not isinstance(cast("object", self.ruleset), Ruleset):
                msg = "Rewrite destinations must be Rulesets"
                raise TypeError(msg)
        if self.subsume and not has_body:
            msg = "subsume requires a body"
            raise ValueError(msg)
        if variable or (self.builtin and (owner is not None or self.mutates_first_arg or self.egg_fn is not None)):
            msg = "Builtin, mutation and variadic free functions are not migrated"
            raise NotImplementedError(msg)
        if any(not isinstance(sort, Ref) or sort.role != "sorts" for sort in input_sorts):
            msg = "Free functions require concrete input sorts"
            raise NotImplementedError(msg)
        builder = Builder()
        inputs = []
        surface = []
        pending_values = self.merge is not None or self.cost is not None
        for index, (parameter, sort) in enumerate(zip(parameters, input_sorts, strict=True)):
            assert isinstance(sort, Ref)
            inputs.append(pb.Arg(sort=builder.import_ref(sort), name=parameter.name))
            if receiver and index == 0:
                continue
            parameter_record = pb.PythonParameter(
                core_input=len(parameters) - index - 1 if reverse_args else index, name=parameter.name
            )
            if parameter.default is not Parameter.empty:
                pending_values = True
                if owner is not None and isinstance(parameter.default, RuntimeExpr):
                    # Defaults are captured before any dependency body can
                    # replace a supplied expression's Python wrapper.
                    parameters[index] = parameter.replace(default=RuntimeExpr(parameter.default.__egg_ref__))
            surface.append(parameter_record)
        if reverse_args:
            inputs.reverse()
            input_sorts.reverse()
        output_index = builder.import_ref(output)
        if owner is None:
            path = [*fn.__module__.split("."), fn.__name__] if fn.__module__ else [fn.__name__]
        else:
            assert member_name is not None
            path = [*owner.__module__.split("."), owner.__name__, member_name]
        name = self.egg_fn or re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", ".".join(path))
        view = pb.PythonCallable(kind=view_kind, path=path, params=surface)
        python_first_input = len(parameters) - 1 if reverse_args else 0
        if self.mutates_first_arg:
            view.mutates = python_first_input
        if owner is not None:
            assert member_name is not None
            assert owner.__egg_sort__ is not None
            view.owner = pb.BindingOwner(
                kind=Oneof[Literal["sort"], int]("sort", builder.import_ref(owner.__egg_sort__))
            )
            view.path = [] if view_kind == pb.PythonCallKind.INITIALIZER else [member_name]
            if receiver:
                view.receiver = python_first_input
        kind = (
            Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name=name,
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature", pb.GenericSignature(inputs=inputs, output=output_index)
                    ),
                ),
            )
            if self.builtin
            else Oneof[Literal["primitive"], pb.Primitive](
                "primitive", pb.Primitive(name=name, inputs=inputs, output=output_index)
            )
            if has_body and self.ruleset is None
            else Oneof[Literal["constructor"], pb.Constructor](
                "constructor",
                pb.Constructor(name=name, inputs=inputs, output=output_index, unextractable=self.unextractable),
            )
            if equality and not has_merge
            else Oneof[Literal["function"], pb.Function](
                "function",
                pb.Function(
                    name=name,
                    inputs=inputs,
                    output=output_index,
                ),
            )
        )
        declaration = pb.Declaration(
            kind=kind,
            bindings=pb.CallableBindings(python=pb.PythonBindings(views=[view])),
            doc=fn.__doc__,
        )

        def finish() -> Ref:  # noqa: C901, PLR0912
            for parameter, parameter_record in zip(
                parameters[1:] if receiver else parameters, view.params, strict=True
            ):
                if parameter.default is not Parameter.empty:
                    sort = input_sorts[parameter_record.core_input]
                    assert isinstance(sort, Ref)
                    default = resolve_literal(sort, parameter.default)
                    parameter_record.default_expr = builder.import_ref(default.__egg_ref__)
            merge = _resolve_merge(output, self.merge)
            if merge is not None:
                assert isinstance(kind.value, pb.Function)
                kind.value.merge = builder.import_ref(merge)
            if self.cost is not None:
                from .builtins import i64  # noqa: PLC0415

                cost_sort = resolve_type_annotation(i64)
                assert isinstance(cost_sort, Ref)
                assert isinstance(kind.value, pb.Constructor)
                kind.value.cost = builder.import_ref(resolve_literal(cost_sort, self.cost).__egg_ref__)
            body = None
            if has_body and (owner is None or self.ruleset is None):
                formals = Builder()
                formal_indices = []
                for argument, sort in zip(inputs, input_sorts, strict=True):
                    assert isinstance(sort, Ref)
                    formal_indices.append(
                        formals.add(
                            "nodes",
                            pb.Node(
                                sort_id=formals.import_ref(sort),
                                kind=Oneof[Literal["var"], str]("var", argument.name),
                            ),
                        )
                    )
                formal_owner = formals.publish()
                formal_refs = [formal_owner.ref("nodes", index) for index in formal_indices]
                values = [RuntimeExpr(reference) for reference in formal_refs]
                callback_args: list[RuntimeExpr | RuntimeClass] = [
                    values[parameter.core_input] for parameter in view.params
                ]
                if view.has_field("receiver"):
                    callback_args.insert(0, values[view.receiver])
                elif owner is not None and view.kind == pb.PythonCallKind.CLASS_METHOD:
                    callback_args.insert(0, owner)
                elif owner is not None and view.kind == pb.PythonCallKind.INITIALIZER:
                    # A source-only self template, never a placeholder declaration.
                    # Retaining this call in the final result is eager recursion.
                    initial = Builder()
                    initial_index = initial.add(
                        "nodes",
                        pb.Node(
                            sort_id=initial.import_ref(output),
                            kind=Oneof[Literal["call"], pb.Call](
                                "call", pb.Call(func=name, args=[initial.import_ref(formal) for formal in formal_refs])
                            ),
                        ),
                    )
                    callback_args.insert(0, RuntimeExpr(initial.publish().ref("nodes", initial_index)))
                with set_current_ruleset(self.ruleset):
                    result = fn(*callback_args)
                    if self.mutates_first_arg:
                        if result is not None:
                            msg = "Mutating callable bodies must return None"
                            raise ValueError(msg)
                        result = values[view.mutates]
                    if result is None:
                        msg = "Callable bodies must return a value"
                        raise ValueError(msg)
                    if self.ruleset is None:
                        resolved = resolve_literal(output, result)
                if self.ruleset is not None:
                    resolved = resolve_literal(output, result)
                body = _lower_function_body(
                    builder,
                    resolved.__egg_ref__,
                    formal_refs,
                    rewrite_body=self.ruleset is not None,
                    recursive_name=name if owner is not None and view.kind == pb.PythonCallKind.INITIALIZER else None,
                )
                if isinstance(kind.value, pb.Primitive):
                    kind.value.body = body
            index = builder.add("declarations", declaration)
            reference = builder.publish().ref("declarations", index)
            Catalog.from_refs([reference])
            if body is not None and self.ruleset is not None:
                values = [RuntimeExpr(formal) for formal in formal_refs]
                lhs = RuntimeFunction(reference, 0)(*values)
                if self.mutates_first_arg:
                    lhs = values[0]
                assert isinstance(lhs, RuntimeExpr)
                lhs_builder = Builder()
                lhs_index = _lower_function_body(lhs_builder, lhs.__egg_ref__, formal_refs, rewrite_body=True)
                rewrite(
                    cast("Expr", RuntimeExpr(lhs_builder.publish().ref("nodes", lhs_index))),
                    ruleset=self.ruleset,
                    subsume=self.subsume,
                ).to(cast("Expr", RuntimeExpr(reference.owner.ref("nodes", body))))
            return reference

        if owner is None:
            return finish()
        return RuntimeFunction(finish, 0, owner, pending_declaration=declaration, pending_values=pending_values)


def _lower_function_body(  # noqa: C901
    builder: Builder, body: Ref, formals: list[Ref], *, rewrite_body: bool = False, recursive_name: str | None = None
) -> int:
    """Lower this body's variables to backend names, importing callee closures intact."""
    parameters = {reference: f"_{index}" for index, reference in enumerate(formals)}
    indices: dict[Ref, int] = {body: builder.reserve("nodes")}
    pending = [body]
    while pending:
        source = pending.pop()
        node: pb.Node = source.read()
        if node.kind is None:
            msg = "Named eager body node has no kind"
            raise ValueError(msg)
        if node.kind.field == "var":
            if source in parameters:
                name = parameters[source]
            elif rewrite_body:
                name = f"_{node.kind.value}"
            else:
                raise ValueError(f"Free variable {node.kind.value!r} in named eager body")
            node.kind = Oneof[Literal["var"], str]("var", name)
        elif node.kind.field == "union" and not rewrite_body:
            msg = "Union in named eager bodies is not supported"
            raise NotImplementedError(msg)
        elif (
            node.kind.field == "primitive_value"
            and node.kind.value.value is not None
            and node.kind.value.value.field == "lambda"
        ):
            msg = (
                "Lambda in named rewrite bodies is not migrated"
                if rewrite_body
                else "Lambda in named eager bodies is not supported"
            )
            raise NotImplementedError(msg)

        def relocate(role: str, index: int, origin: Ref = source) -> int:
            reference = origin.owner.ref(role, index)
            if role != "nodes":
                return builder.import_ref(reference)
            if reference not in indices:
                indices[reference] = builder.reserve("nodes")
                pending.append(reference)
            return indices[reference]

        def dependency(namespace: str, name: str, origin: Ref = source) -> None:
            if namespace == "callable" and name == recursive_name:
                msg = "Recursively resolving function declarations"
                raise ValueError(msg)
            builder.import_ref(_definition_at(origin, namespace, name))

        builder.fill("nodes", indices[source], _copy_record(node, relocate=relocate, dependency=dependency))
    return indices[body]


def _compile_python_function(
    function: FunctionType, parameters: list[Ref], output: Ref | None
) -> tuple[RuntimeFunction, tuple[RuntimeExpr, ...]]:
    """Lift lexical symbolic values into a named primitive's positional prefix."""
    closure_values = [cell.cell_contents for cell in function.__closure__ or ()]
    global_values = {
        name: value
        for name in function.__code__.co_names
        if isinstance(value := function.__globals__.get(name), RuntimeExpr)
    }
    captured = {
        value.__egg_ref__: value
        for value in [*closure_values, *global_values.values()]
        if isinstance(value, RuntimeExpr)
    }
    captures = tuple(captured.values())
    capture_sorts = [value.__egg_ref__.owner.ref("sorts", value.__egg_ref__.read().sort_id) for value in captures]
    sorts = [*capture_sorts, *parameters]
    formals = Builder()
    indices = [
        formals.add(
            "nodes",
            pb.Node(
                sort_id=formals.import_ref(sort),
                kind=Oneof[Literal["var"], str]("var", f"_{index}"),
            ),
        )
        for index, sort in enumerate(sorts)
    ]
    owner = formals.publish()
    references = [owner.ref("nodes", index) for index in indices]
    replacements = {
        reference: RuntimeExpr(formal) for reference, formal in zip(captured, references[: len(captures)], strict=True)
    }
    symbolic_globals = {
        **function.__globals__,
        **{name: replacements[value.__egg_ref__] for name, value in global_values.items()},
    }
    symbolic_closure = tuple(
        CellType(replacements[value.__egg_ref__] if isinstance(value, RuntimeExpr) else value)
        for value in closure_values
    )
    symbolic = FunctionType(
        function.__code__, symbolic_globals, function.__name__, function.__defaults__, symbolic_closure or None
    )
    symbolic.__kwdefaults__ = function.__kwdefaults__
    with set_current_ruleset(None):
        result = symbolic(*(RuntimeExpr(reference) for reference in references[len(captures) :]))
        body = resolve_literal(output, result) if output is not None else to_runtime_expr(result)
    body_node: pb.Node = body.__egg_ref__.read()
    builder = Builder()
    body_index = _lower_function_body(builder, body.__egg_ref__, references)
    name = f"__egglog_python_lambda_{uuid4().hex}"
    argument_names = list(signature(function).parameters)
    if len(argument_names) != len(parameters):
        argument_names = [f"arg{index}" for index in range(len(parameters))]
    parameter_names = [*(f"_capture{index}" for index in range(len(captures))), *argument_names]
    if len(set(parameter_names)) != len(parameter_names):
        parameter_names = [f"arg{index}" for index in range(len(sorts))]
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["primitive"], pb.Primitive](
                "primitive",
                pb.Primitive(
                    name=name,
                    inputs=[
                        pb.Arg(sort=builder.import_ref(sort), name=f"_{index}") for index, sort in enumerate(sorts)
                    ],
                    output=builder.import_ref(body.__egg_ref__.owner.ref("sorts", body_node.sort_id)),
                    body=body_index,
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.FUNCTION,
                            path=["egglog", "egraph", name],
                            params=[
                                pb.PythonParameter(name=parameter, core_input=index)
                                for index, parameter in enumerate(parameter_names)
                            ],
                        )
                    ]
                )
            ),
        ),
    )
    return RuntimeFunction(builder.publish().ref("declarations", declaration), 0), captures


# Overload to support aritys 0-4 until variadic generic support map, so we can map from type to value
@overload
def relation(
    name: str, tp1: type[BE1], tp2: type[BE2], tp3: type[BE3], tp4: type[BE4], /
) -> Callable[[BE1, BE2, BE3, BE4], Unit]: ...


@overload
def relation(name: str, tp1: type[BE1], tp2: type[BE2], tp3: type[BE3], /) -> Callable[[BE1, BE2, BE3], Unit]: ...


@overload
def relation(name: str, tp1: type[BE1], tp2: type[BE2], /) -> Callable[[BE1, BE2], Unit]: ...


@overload
def relation(name: str, tp1: type[BE1], /, *, egg_fn: str | None = None) -> Callable[[BE1], Unit]: ...


@overload
def relation(name: str, /, *, egg_fn: str | None = None) -> Callable[[], Unit]: ...


def relation(name: str, /, *tps: type, egg_fn: str | None = None) -> Callable[..., Unit]:
    """
    Creates a function whose return type is `Unit` and has a default value.
    """
    module = _get_module()
    path = [*module.split("."), name] if module else [name]
    builder = Builder()
    inputs = []
    params = []
    for index, tp in enumerate(tps):
        sort = resolve_type_annotation(tp)
        if not isinstance(sort, Ref) or sort.role != "sorts":
            msg = "Relations require concrete input sorts"
            raise TypeError(msg)
        label = f"arg{index}"
        inputs.append(pb.Arg(sort=builder.import_ref(sort), name=label))
        params.append(pb.PythonParameter(core_input=index, name=label))
    logical_name = egg_fn or re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", ".".join(path))
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["relation"], pb.Relation]("relation", pb.Relation(name=logical_name, inputs=inputs)),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[pb.PythonCallable(kind=pb.PythonCallKind.FUNCTION, path=path, params=params)]
                )
            ),
        ),
    )
    reference = builder.publish().ref("declarations", declaration)
    Catalog.from_refs([reference])
    return cast("Callable[..., Unit]", RuntimeFunction(reference, 0))


def _get_module(frame: FrameType | None = None) -> str | None:
    if frame is None:
        frame = currentframe()
        assert frame is not None
        frame = frame.f_back
        assert frame is not None
        frame = frame.f_back
        assert frame is not None

    module = getmodule(frame)
    return module.__name__ if module is not None else None


@overload
def constant(
    name: str,
    tp: type[USER_EXPR],
    /,
    *,
    egg_name: str | None = ...,
    merge: Callable[[USER_EXPR, USER_EXPR], USER_EXPR] | None = ...,
) -> USER_EXPR: ...


@overload
def constant(
    name: str,
    tp: type[BUILTIN_EXPR],
    /,
    *,
    egg_name: str | None = ...,
    merge: Callable[[BUILTIN_EXPR, BUILTIN_EXPR], BUILTIN_EXPR] | None = ...,
) -> BUILTIN_EXPR: ...


@overload
def constant(
    name: str,
    tp: type[USER_EXPR],
    default_replacement: USER_EXPR,
    /,
    *,
    egg_name: str | None = ...,
    ruleset: Ruleset | None = ...,
) -> USER_EXPR: ...


@overload
def constant(
    name: str,
    tp: type[BUILTIN_EXPR],
    default_replacement: BUILTIN_EXPR | None,
    /,
    *,
    egg_name: str | None = ...,
) -> BUILTIN_EXPR: ...


def constant(
    name: str,
    tp: type[BASE_EXPR],
    default_replacement: BASE_EXPR | None | object = _MISSING,
    /,
    *,
    egg_name: str | None = None,
    ruleset: Ruleset | None = None,
    merge: Callable[[BASE_EXPR, BASE_EXPR], BASE_EXPR] | None = None,
) -> BASE_EXPR:
    """
    A "constant" is implemented as the instantiation of a value that takes no args.
    This creates a function with `name` and return type `tp` and returns a value of it being called.
    """
    sort = resolve_type_annotation(tp)
    if not isinstance(sort, Ref) or sort.role != "sorts":
        msg = "Constants require a concrete sort"
        raise NotImplementedError(msg)
    record: pb.Sort = sort.read()
    if record.kind is None or record.kind.field not in {"eq", "family"}:
        msg = "Constants require an equality or host-family sort"
        raise NotImplementedError(msg)
    has_default = default_replacement is not _MISSING
    return_type_is_eqsort = record.kind.field == "eq"
    if ruleset is not None and not has_default:
        msg = "Explicit rulesets require a default"
        raise ValueError(msg)
    if has_default and ruleset is not None and not return_type_is_eqsort:
        msg = "Primitive-returning defaults cannot use an explicit ruleset"
        raise ValueError(msg)
    has_merge = merge is not None
    if record.kind.field == "family" and record.kind.value.name == "Unit" and has_merge:
        msg = "Functions that return Unit cannot use merge"
        raise ValueError(msg)
    if has_default and has_merge:
        msg = (
            "Eqsort-returning callables with bodies cannot use merge"
            if return_type_is_eqsort
            else "Primitive-returning callables with bodies cannot use merge"
        )
        raise ValueError(msg)
    if ruleset is not None and not isinstance(cast("object", ruleset), Ruleset):
        msg = "Rewrite destinations must be Rulesets"
        raise TypeError(msg)
    module = _get_module()
    path = [*module.split("."), name] if module else [name]
    builder = Builder()
    output = builder.import_ref(sort)
    body = None
    if has_default:
        with set_current_ruleset(None):
            default = resolve_literal(sort, default_replacement)
        body = _lower_function_body(builder, default.__egg_ref__, [])
    merge_expression = _resolve_merge(sort, cast("Callable[[object, object], object] | None", merge))
    logical_name = egg_name or re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", "%" + ".".join(path))
    kind = (
        Oneof[Literal["primitive"], pb.Primitive](
            "primitive", pb.Primitive(name=logical_name, output=output, body=body)
        )
        if body is not None and ruleset is None
        else Oneof[Literal["constructor"], pb.Constructor](
            "constructor", pb.Constructor(name=logical_name, output=output)
        )
        if return_type_is_eqsort and not has_merge
        else Oneof[Literal["function"], pb.Function](
            "function",
            pb.Function(
                name=logical_name,
                output=output,
                merge=builder.import_ref(merge_expression) if merge_expression is not None else None,
            ),
        )
    )
    index = builder.add(
        "declarations",
        pb.Declaration(
            kind=kind,
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(views=[pb.PythonCallable(kind=pb.PythonCallKind.CONSTANT, path=path)])
            ),
        ),
    )
    reference = builder.publish().ref("declarations", index)
    Catalog.from_refs([reference])  # Validate the persisted Python view before exposing the expression.
    value = RuntimeFunction(reference, 0)()
    if body is not None and ruleset is not None:
        rewrite(cast("Expr", value), ruleset=ruleset).to(cast("Expr", RuntimeExpr(reference.owner.ref("nodes", body))))
    return cast("BASE_EXPR", value)


def _last_param_variable(params: list[Parameter]) -> bool:
    """
    Checks if the last paramater is a variable arg.

    Raises an error if any of the other params are not positional or keyword.
    """
    found_var_arg = False
    for param in params:
        if found_var_arg:
            msg = "Can only have a single var arg at the end"
            raise ValueError(msg)
        kind = param.kind
        if kind == Parameter.VAR_POSITIONAL:
            found_var_arg = True
        elif kind != Parameter.POSITIONAL_OR_KEYWORD:
            raise ValueError(f"Can only register functions with positional or keyword args, not {param.kind}")
    return found_var_arg


class GraphvizKwargs(TypedDict, total=False):
    max_functions: int | None
    max_calls_per_function: int | None
    n_inline_leaves: int
    split_primitive_outputs: bool
    split_functions: list[ExprCallable]
    include_temporary_functions: bool


@dataclass
class EGraph:
    """
    A collection of expressions where each expression is part of a distinct equivalence class.

    Can run actions, check facts, run schedules, or extract minimal cost expressions.
    """

    _state: EGraphState = field(init=False, repr=False)
    # For pushing/popping with egglog
    _state_stack: list[EGraphState] = field(default_factory=list, repr=False)
    # For storing the global "current" egraph
    _token_stack: list[EGraph] = field(default_factory=list, repr=False)
    _cost_model_frames: list[tuple[_CostModelObservation, frozenset[Ref]]] = field(default_factory=list, repr=False)

    def __init__(
        self,
        *actions: ActionLike,
        seminaive: bool = True,
        save_egglog_string: bool = False,
        num_threads: int = 1,
        no_decomp: bool = False,
    ) -> None:
        with _TRACER.start_as_current_span("create"):
            with _TRACER.start_as_current_span("create_bindings"):
                self._state = EGraphState(
                    num_threads=num_threads,
                    save_egglog_string=save_egglog_string,
                    seminaive=seminaive,
                    no_decomp=no_decomp,
                )
            self._state_stack = []
            self._token_stack = []
            self._cost_model_frames = []
            if actions:
                self.register(*actions)

    def set_report_level(self, level: pb.ReportLevel) -> None:
        """Set native report detail using the generated protocol enum."""
        if self._cost_model_frames or self._state._native_running:
            msg = "Cannot execute engine operations during a cost-model callback"
            raise RuntimeError(msg)
        if type(level) is not pb.ReportLevel or level == pb.ReportLevel.UNSPECIFIED:
            msg = "Expected a defined protobuf report level"
            raise TypeError(msg)
        if self._state.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        request = pb.ConfigureEGraphResourcesRequest(
            egraph_id=self._state.handle,
            operation=Oneof[Literal["report_level"], pb.ReportLevel]("report_level", level),
        )
        pb.ConfigureEGraphResourcesResponse.from_binary(self._state.transport.configure_resources(request.to_binary()))

    def num_threads(self) -> int:
        """Return the number of worker threads configured for this e-graph."""
        return self._state.configure_resources().threads

    def set_num_threads(self, num_threads: int) -> None:
        """
        Set the number of worker threads used by this e-graph.

        Passing ``1`` keeps execution serial. Passing ``0`` uses available
        parallelism.
        """
        self._state.configure_resources(num_threads)

    def no_decomp(self) -> bool:
        """Query whether decomposition is disabled for future rule compilation."""
        return self._state.configure_resources().no_decomp

    def set_no_decomp(self, no_decomp: bool) -> None:
        """
        Configure decomposition for subsequently compiled rules, not installed plans.

        This runtime policy is excluded from logical Egglog transcripts.
        """
        self._state.configure_resources(no_decomp=no_decomp)

    @property
    def as_egglog_string(self) -> str:
        """
        Return the replayable Egglog transcript for this e-graph.

        If a failure may have partially changed the graph and cannot be
        represented by a replayable command, the transcript is invalidated and
        this property raises ``RuntimeError``.
        """
        return self._state.transport.recorded_source()

    def close(self) -> None:
        """
        Close and remove the optional saved Egglog transcript.

        For an e-graph created with ``save_egglog_string=True``, subsequent
        commands are rejected because they could no longer be recorded.
        Otherwise this method has no effect and the e-graph remains usable.
        """
        # With transcripts disabled, close historically leaves the graph usable.
        # Native handles are released when their owning state is disposed.
        if not self._state._recording:
            return
        if self._cost_model_frames or self._state._native_running:
            msg = "Cannot close a transcript during engine execution"
            raise RuntimeError(msg)
        self._state.transport.close_recording()

    def _ipython_display_(self) -> None:
        self.display()

    def input(self, fn: Callable[..., BASE_EXPR], path: str) -> None:
        """Load tab-separated data; saved transcripts replay without the input file."""
        reference = resolve_callable(fn)
        declaration: pb.Declaration = reference.read()
        if declaration.kind is None or not isinstance(
            declaration.kind.value, pb.Function | pb.Constructor | pb.Relation
        ):
            msg = "CSV input requires a table-backed relation, constructor, or bodyless function"
            raise ValueError(msg)
        descriptor = pack(roots=[reference])
        self._state.run_program(descriptor)
        encoded = self._state.transport.produce_input(descriptor.program.to_binary(), declaration.kind.value.name, path)
        rows = pb.Program.from_binary(encoded)
        self._state.run_program(Packed(rows, (), descriptor.definitions, descriptor.rulesets))

    def let(self, name: str, expr: BASE_EXPR) -> BASE_EXPR:
        """
        Set an exact-name binding and return its ordinary nullary call.

        The returned expression resolves that name in whichever graph uses it.
        See :func:`let` for repeated-write and name policies.
        """
        action = let(name, expr)
        self.register(action)
        command: pb.Command = action.__egg_ref__.read()
        assert command.kind is not None
        assert command.kind.field == "action"
        assert command.kind.value.kind is not None
        assert command.kind.value.kind.field == "set"
        target = command.kind.value.kind.value.target
        assert target is not None
        declaration = _definition_at(action.__egg_ref__, "callable", target.func)
        return cast("BASE_EXPR", RuntimeFunction(declaration, 0)())

    def include(self, path: str) -> None:
        """
        Include a file of rules.
        """
        msg = "Not implemented yet, because we don't have a way of registering the types with Python"
        raise NotImplementedError(msg)

    def output(self) -> None:
        msg = "Not imeplemented yet, because there are no examples in the egglog repo"
        raise NotImplementedError(msg)

    @overload
    def run(
        self, limit: int, /, *until: Fact, ruleset: Ruleset | UnstableCombinedRuleset | None = None
    ) -> RunReport: ...

    @overload
    def run(self, schedule: Schedule, /) -> RunReport: ...

    @_TRACER.start_as_current_span("run")
    def run(
        self,
        limit_or_schedule: int | Schedule,
        /,
        *until: Fact,
        ruleset: Ruleset | UnstableCombinedRuleset | None = None,
    ) -> RunReport:
        """
        Run the egraph until the given limit or until the given facts are true.
        """
        if isinstance(limit_or_schedule, int):
            limit_or_schedule = run(ruleset, *until) * limit_or_schedule
        return self._run_schedule(limit_or_schedule)

    def _run_schedule(self, schedule: Schedule) -> RunReport:
        if isinstance(schedule, Ruleset | UnstableCombinedRuleset):
            schedule = run(schedule)
        command = schedule._resolve(self._state)
        packed = pack(commands=[command], ambient=self._state.definitions)
        response, _ = self._state.run_program(packed, include_run_reports=True)
        if len(response.outputs) != 1 or response.outputs[0].run_report is None:
            msg = "Run response omitted its complete native report"
            raise ValueError(msg)
        output = response.outputs[0]
        report = output.run_report
        assert report is not None
        expected = "run" if command.read().kind.field == "run" else "loop"
        if (
            output.location != pb.CommandLocation(path=[0])
            or output.kind is None
            or output.kind.field not in ("run", "loop")
            or output.kind.field != expected
            or output.kind.value.updated != report.updated
            or (output.kind.field == "run" and output.kind.value.can_stop != report.can_stop)
        ):
            msg = "Run report does not match its requested top-level control"
            raise ValueError(msg)
        return RunReport._from_proto(report, response, packed, self._state)

    def stats(self) -> RunReport:
        """
        Returns the overall run report for the egraph.
        """
        builder = Builder()
        command = builder.add("commands", pb.Command(kind=Oneof[Literal["stats"], pb.Unit]("stats", pb.Unit())))
        packed = pack(commands=[builder.publish().ref("commands", command)])
        response, _ = self._state.run_program(packed)
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "statistics"
            or response.outputs[0].location != pb.CommandLocation(path=[0])
            or response.outputs[0].run_report is not None
        ):
            msg = "Stats response omitted its native statistics"
            raise ValueError(msg)
        return RunReport._from_proto(response.outputs[0].kind.value, response, packed, self._state)

    def check_bool(self, *facts: FactLike) -> bool:
        """
        Returns true if the facts are true in the egraph.
        """
        return self._check(facts, allow_failed_check=True)

    def check(self, *facts: FactLike) -> None:
        """
        Check if a fact is true in the egraph.
        """
        self._check(facts)

    @_TRACER.start_as_current_span("check")
    def _check(self, facts: Iterable[FactLike], *, allow_failed_check: bool = False) -> bool:
        """Build one canonical query and interpret only this response's check result."""
        builder = Builder()
        indices = [builder.import_ref(fact.__egg_ref__) for fact in _fact_likes(facts)]
        command = builder.add(
            "commands", pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=indices)))
        )
        response, _ = self._state.run_program(
            [builder.publish().ref("commands", command)], allow_failed_check=allow_failed_check
        )
        return response.error is None

    def check_fail(self, *facts: FactLike) -> None:
        """
        Expect the native check to fail, retaining completed effects.

        Invalid facts and prerequisite or trailing helper failures propagate.
        """
        builder = Builder()
        indices = [builder.import_ref(fact.__egg_ref__) for fact in _fact_likes(facts)]
        command = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["fail"], pb.Command](
                    "fail", pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=indices)))
                )
            ),
        )
        self._state.run_program([builder.publish().ref("commands", command)])

    @overload
    def extract(
        self,
        expr: BASE_EXPR,
        /,
        include_cost: Literal[False] = False,
        cost_model: TreeCostModel | DagCostModel | None = None,
        *,
        extractor: ExtractionMode = "tree",
    ) -> BASE_EXPR: ...

    @overload
    def extract(
        self,
        expr: BASE_EXPR,
        /,
        include_cost: Literal[True],
        cost_model: None = None,
        *,
        extractor: ExtractionMode = "tree",
    ) -> tuple[BASE_EXPR, int]: ...

    @overload
    def extract(
        self,
        expr: BASE_EXPR,
        /,
        include_cost: Literal[True],
        cost_model: TreeCostModel[COST],
        *,
        extractor: ExtractionMode = "tree",
    ) -> tuple[BASE_EXPR, COST]: ...

    @overload
    def extract(
        self,
        expr: BASE_EXPR,
        /,
        include_cost: Literal[True],
        cost_model: DagCostModel[DAG_COST],
        *,
        extractor: ExtractionMode = "tree",
    ) -> tuple[BASE_EXPR, DAG_COST]: ...

    @_TRACER.start_as_current_span("extract")
    def extract(
        self,
        expr: BASE_EXPR,
        /,
        include_cost: bool = False,
        cost_model: TreeCostModel[Any] | DagCostModel[Any] | None = None,
        *,
        extractor: ExtractionMode = "tree",
    ) -> BASE_EXPR | tuple[BASE_EXPR, Any]:
        """
        Extract the lowest cost expression from the egraph.
        """
        if extractor not in {"tree", "greedy-dag"}:
            raise ValueError(f"Unknown extractor {extractor!r}; expected 'tree' or 'greedy-dag'")
        if cost_model is not None and extractor == "greedy-dag" and not isinstance(cost_model, DagCostModel):
            msg = "greedy-dag extraction requires a DagCostModel"
            raise TypeError(msg)
        registration = (
            self._state.cost_model(
                _HostModelFactory(self, cast("TreeCostModel[Comparable] | DagCostModel[ComparableAdd]", cost_model)),
                additive=isinstance(cost_model, DagCostModel),
            )
            if cost_model is not None
            else contextlib.nullcontext("")
        )
        with registration as model_name:
            runtime_expr = to_runtime_expr(expr)
            builder = Builder()
            root = builder.import_ref(runtime_expr.__egg_ref__)
            index = builder.add(
                "commands",
                pb.Command(
                    kind=Oneof[Literal["extract"], pb.Extract](
                        "extract",
                        pb.Extract(
                            roots=[root],
                            variants=1,
                            cost_model=model_name,
                            extractor=pb.Extractor.TREE if extractor == "tree" else pb.Extractor.GREEDY_DAG,
                        ),
                    )
                ),
            )
            response, owner = self._state.run_program([builder.publish().ref("commands", index)])
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "extraction"
        ):
            msg = "Shared adapter did not return the requested extraction"
            raise ValueError(msg)
        roots = response.outputs[0].kind.value.roots
        if len(roots) != 1 or not roots[0].variants:
            msg = "Unextractable root"
            raise ValueError(msg)
        variant = roots[0].variants[0]
        result = cast("BASE_EXPR", RuntimeExpr(owner.ref("nodes", variant.term)))
        if not include_cost:
            return result
        if not variant.has_field("cost"):
            msg = "The extraction cost has no portable value"
            raise ValueError(msg)
        from .deconstruct import get_literal_value  # noqa: PLC0415

        cost = get_literal_value(RuntimeExpr(owner.ref("nodes", variant.cost)))
        if cost_model is None and not isinstance(cost, int):
            msg = "Expected an i64 extraction cost"
            raise TypeError(msg)
        return result, cost

    @overload
    def extract_multiple(
        self,
        expr: BASE_EXPR,
        n: int,
        *,
        include_cost: Literal[False] = False,
        cost_model: TreeCostModel | DagCostModel | None = None,
        extractor: ExtractionMode = "tree",
    ) -> list[BASE_EXPR]: ...

    @overload
    def extract_multiple(
        self,
        expr: Sequence[BASE_EXPR],
        n: int,
        *,
        include_cost: Literal[False] = False,
        cost_model: TreeCostModel | DagCostModel | None = None,
        extractor: ExtractionMode = "tree",
    ) -> list[list[BASE_EXPR]]: ...

    @overload
    def extract_multiple(
        self,
        expr: BASE_EXPR,
        n: int,
        *,
        include_cost: Literal[True],
        cost_model: None = None,
        extractor: ExtractionMode = "tree",
    ) -> list[tuple[BASE_EXPR, int]]: ...

    @overload
    def extract_multiple(
        self,
        expr: Sequence[BASE_EXPR],
        n: int,
        *,
        include_cost: Literal[True],
        cost_model: None = None,
        extractor: ExtractionMode = "tree",
    ) -> list[list[tuple[BASE_EXPR, int]]]: ...

    @overload
    def extract_multiple(
        self,
        expr: BASE_EXPR,
        n: int,
        *,
        include_cost: Literal[True],
        cost_model: TreeCostModel[COST],
        extractor: ExtractionMode = "tree",
    ) -> list[tuple[BASE_EXPR, COST]]: ...

    @overload
    def extract_multiple(
        self,
        expr: Sequence[BASE_EXPR],
        n: int,
        *,
        include_cost: Literal[True],
        cost_model: TreeCostModel[COST],
        extractor: ExtractionMode = "tree",
    ) -> list[list[tuple[BASE_EXPR, COST]]]: ...

    @overload
    def extract_multiple(
        self,
        expr: BASE_EXPR,
        n: int,
        *,
        include_cost: Literal[True],
        cost_model: DagCostModel[DAG_COST],
        extractor: ExtractionMode = "tree",
    ) -> list[tuple[BASE_EXPR, DAG_COST]]: ...

    @overload
    def extract_multiple(
        self,
        expr: Sequence[BASE_EXPR],
        n: int,
        *,
        include_cost: Literal[True],
        cost_model: DagCostModel[DAG_COST],
        extractor: ExtractionMode = "tree",
    ) -> list[list[tuple[BASE_EXPR, DAG_COST]]]: ...

    def extract_multiple(  # noqa: C901, PLR0912
        self,
        expr: BASE_EXPR | Sequence[BASE_EXPR],
        n: int,
        *,
        include_cost: bool = False,
        cost_model: TreeCostModel[Any] | DagCostModel[Any] | None = None,
        extractor: ExtractionMode = "tree",
    ) -> list[BASE_EXPR] | list[list[BASE_EXPR]] | list[tuple[BASE_EXPR, Any]] | list[list[tuple[BASE_EXPR, Any]]]:
        """Extract ordered variants for one root or a sequence of roots over bytes."""
        n = index_integer(n)
        if n <= 0:
            msg = "The number of variants must be positive"
            raise ValueError(msg)
        if n > 2**32 - 1:
            msg = "The number of variants exceeds the protocol's uint32 range"
            raise OverflowError(msg)
        if extractor not in {"tree", "greedy-dag"}:
            raise ValueError(f"Unknown extractor {extractor!r}; expected 'tree' or 'greedy-dag'")
        if cost_model is not None and extractor == "greedy-dag" and not isinstance(cost_model, DagCostModel):
            msg = "greedy-dag extraction requires a DagCostModel"
            raise TypeError(msg)
        if isinstance(expr, Sequence):
            batched = True
            expressions = list(expr)
        else:
            batched = False
            expressions = [expr]
        if not expressions:
            msg = "extract_multiple requires at least one expression"
            raise ValueError(msg)
        registration = (
            self._state.cost_model(
                _HostModelFactory(self, cast("TreeCostModel[Comparable] | DagCostModel[ComparableAdd]", cost_model)),
                additive=isinstance(cost_model, DagCostModel),
            )
            if cost_model is not None
            else contextlib.nullcontext("")
        )
        with registration as model_name:
            builder = Builder()
            roots = [builder.import_ref(to_runtime_expr(item).__egg_ref__) for item in expressions]
            command = builder.add(
                "commands",
                pb.Command(
                    kind=Oneof[Literal["extract"], pb.Extract](
                        "extract",
                        pb.Extract(
                            roots=roots,
                            variants=n,
                            cost_model=model_name,
                            extractor=pb.Extractor.TREE if extractor == "tree" else pb.Extractor.GREEDY_DAG,
                        ),
                    )
                ),
            )
            response, owner = self._state.run_program([builder.publish().ref("commands", command)])
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "extraction"
            or len(response.outputs[0].kind.value.roots) != len(expressions)
        ):
            msg = "Shared adapter did not return the requested root extractions"
            raise ValueError(msg)
        if not include_cost:
            results = [
                [cast("BASE_EXPR", RuntimeExpr(owner.ref("nodes", variant.term))) for variant in root.variants]
                for root in response.outputs[0].kind.value.roots
            ]
            return results if batched else results[0]
        from .deconstruct import get_literal_value  # noqa: PLC0415

        cost_results: list[list[tuple[BASE_EXPR, Any]]] = []
        for root in response.outputs[0].kind.value.roots:
            root_results: list[tuple[BASE_EXPR, Any]] = []
            for variant in root.variants:
                if not variant.has_field("cost"):
                    msg = "The extraction cost has no portable value"
                    raise ValueError(msg)
                cost = get_literal_value(RuntimeExpr(owner.ref("nodes", variant.cost)))
                if cost_model is None and not isinstance(cost, int):
                    msg = "Expected an i64 extraction cost"
                    raise TypeError(msg)
                root_results.append((cast("BASE_EXPR", RuntimeExpr(owner.ref("nodes", variant.term))), cost))
            cost_results.append(root_results)
        return cost_results if batched else cost_results[0]

    def keep_best(
        self,
        fn: ExprCallable,
        /,
        *fns: ExprCallable,
        extractor: ExtractionMode = "tree",
        cost_model: TreeCostModel[COST] | DagCostModel[DAG_COST] | None = None,
    ) -> None:
        """Compact selected tables through the shared KeepBest command."""
        if extractor not in {"tree", "greedy-dag"}:
            raise ValueError(f"Unknown extractor {extractor!r}; expected 'tree' or 'greedy-dag'")
        if cost_model is not None and extractor == "greedy-dag" and not isinstance(cost_model, DagCostModel):
            msg = "greedy-dag extraction requires a DagCostModel"
            raise TypeError(msg)
        references = [resolve_callable(callable_) for callable_ in (fn, *fns)]
        builder = Builder()
        tables = []
        for reference in references:
            declaration: pb.Declaration = reference.read()
            if declaration.kind is None or not isinstance(
                declaration.kind.value, pb.Function | pb.Constructor | pb.Relation
            ):
                msg = "This operation requires a table-backed relation, constructor, or bodyless function"
                raise ValueError(msg)
            builder.import_ref(reference)
            tables.append(declaration.kind.value.name)
        registration = (
            self._state.cost_model(
                _HostModelFactory(self, cast("TreeCostModel[Comparable] | DagCostModel[ComparableAdd]", cost_model)),
                additive=isinstance(cost_model, DagCostModel),
            )
            if cost_model is not None
            else contextlib.nullcontext("")
        )
        with registration as model_name:
            command = builder.add(
                "commands",
                pb.Command(
                    kind=Oneof[Literal["keep_best"], pb.KeepBest](
                        "keep_best",
                        pb.KeepBest(
                            tables=tables,
                            cost_model=model_name,
                            extractor=pb.Extractor.TREE if extractor == "tree" else pb.Extractor.GREEDY_DAG,
                        ),
                    )
                ),
            )
            self._state.run_program([builder.publish().ref("commands", command)])

    @_TRACER.start_as_current_span("push")
    def push(self) -> None:
        """
        Push the current state of the egraph, so that it can be popped later and reverted back.
        """
        if self._cost_model_frames or self._state._native_running:
            msg = "Cannot push during a cost-model callback"
            raise RuntimeError(msg)
        copied = self._state.copy()
        self._state_stack.append(self._state)
        self._state = copied

    @_TRACER.start_as_current_span("pop")
    def pop(self) -> None:
        """
        Pop the current state of the egraph, reverting back to the previous state.
        """
        if self._cost_model_frames or self._state._native_running:
            msg = "Cannot pop during a cost-model callback"
            raise RuntimeError(msg)
        if not self._state_stack:
            msg = "Cannot pop an EGraph without a pushed state"
            raise IndexError(msg)
        self._state.return_to(self._state_stack[-1])
        self._state = self._state_stack.pop()

    def __enter__(self) -> Self:
        """
        Copy the egraph state, so that it can be reverted back to the original state at the end.

        Also sets the current egraph to this one.
        """
        self.push()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        try:
            self.pop()
        except Exception:
            if exc_type is None:
                raise

    def _serialize(self, **kwargs: Unpack[GraphvizKwargs]) -> _Graph:
        """Read logical state over bytes and project a separate, filtered display."""
        from .visualizer_widget import _snapshot_to_data  # noqa: PLC0415

        split = {resolve_callable(fn).read().kind.value.name for fn in kwargs.get("split_functions", [])}
        # Backend temporaries are not part of the logical snapshot inventory.
        return _snapshot_to_data(
            self.freeze(),
            split_functions=split,
            max_functions=kwargs.get("max_functions"),
            max_calls_per_function=kwargs.get("max_calls_per_function"),
            n_inline_leaves=kwargs.get("n_inline_leaves", 0),
            split_primitive_outputs=kwargs.get("split_primitive_outputs", True),
        )

    def _graphviz(self, **kwargs: Unpack[GraphvizKwargs]) -> graphviz.Digraph:
        import graphviz  # noqa: PLC0415

        data = self._serialize(**kwargs)
        dot = graphviz.Digraph(graph_attr={"compound": "true"})
        groups: dict[str, list[str]] = {}
        for nid, node in data["nodes"].items():
            groups.setdefault(node["eclass"], []).append(nid)
        for cid, members in groups.items():
            with dot.subgraph(name=f"cluster_{cid}") as cluster:
                cluster.attr(label=data["class_data"].get(cid, {}).get("type", ""), style="rounded,dashed")
                for nid in members:
                    node = data["nodes"][nid]
                    cluster.node(
                        nid,
                        node["op"],
                        style="dashed" if node["subsumed"] else "solid",
                        tooltip=f"cost: {node['cost']}" if "cost" in node else "",
                    )
        for nid, node in data["nodes"].items():
            for position, child in enumerate(node["children"]):
                dot.edge(nid, child, label=str(position), lhead=f"cluster_{data['nodes'][child]['eclass']}")
        return dot

    def display(self, graphviz: bool = False, **kwargs: Unpack[GraphvizKwargs]) -> None:
        """Display the protobuf snapshot with the interactive widget or Graphviz."""
        from IPython.display import SVG, display  # noqa: PLC0415

        from .visualizer_widget import IN_IPYTHON, VisualizerWidget  # noqa: PLC0415

        if graphviz:
            dot = self._graphviz(**kwargs)
            if IN_IPYTHON:
                display(SVG(dot.pipe(format="svg", quiet=True, encoding="utf-8")))
            else:
                dot.render(view=True, format="svg", quiet=True)
        else:
            VisualizerWidget(egraphs=[json.dumps(self._serialize(**kwargs))]).display_or_open()

    def saturate(  # noqa: C901
        self,
        schedule: Schedule | None = None,
        *,
        expr: Expr | None = None,
        max: int = 1000,
        visualize: bool = True,
        print_frozen: bool = False,
        **kwargs: Unpack[GraphvizKwargs],
    ) -> None:
        """Run repeated byte schedules until the adapter reports saturation."""
        from .visualizer_widget import VisualizerWidget  # noqa: PLC0415

        history = [json.dumps(self._serialize(**kwargs))] if visualize else []
        failure: BaseException | None = None
        try:
            for iteration in range(max):
                report = self.run(schedule or 1)
                if expr is not None:
                    print(self.extract(expr), "\n")
                if visualize:
                    history.append(json.dumps(self._serialize(**kwargs)))
                if print_frozen:
                    print(f"After iteration {iteration + 1}:")
                    print(self.freeze(), "\n")
                if report.can_stop:
                    break
        except BaseException as error:
            failure = error
            if visualize:
                try:
                    history.append(json.dumps(self._serialize(**kwargs)))
                except Exception as observation_error:
                    error.add_note(f"Could not visualize the failure prefix ({type(observation_error).__name__})")
            raise
        finally:
            if visualize:
                try:
                    VisualizerWidget(egraphs=history).display_or_open()
                except Exception as display_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Could not display snapshot history ({type(display_error).__name__})")

    @_TRACER.start_as_current_span("register")
    def register(
        self,
        /,
        command_or_generator: ActionLike | RewriteOrRule | RewriteOrRuleGenerator,
        *command_likes: ActionLike | RewriteOrRule,
    ) -> None:
        """
        Registers any number of rewrites or rules.
        """
        if isinstance(command_or_generator, FunctionType):
            assert not command_likes
            frame = inspect.currentframe()
            assert frame is not None
            assert frame.f_back is not None
            command_likes = tuple(_rewrite_or_rule_generator(command_or_generator, frame.f_back))
        else:
            command_likes = (cast("CommandLike", command_or_generator), *command_likes)
        commands = [_command_like(c) for c in command_likes]
        self._register_commands(commands)

    def _register_commands(self, cmds: list[Command]) -> None:
        if all(isinstance(command, Action) for command in cmds):
            self._state.run_program([cast("Action", command).__egg_ref__ for command in cmds])
            return
        # Mixed registration preserves each original command's failure prefix.
        # Splitting is forbidden if direct Action uses bridge a Union identity
        # across requests. Query facts and invoked code bodies have other units.
        uses: dict[Ref, int] = {}
        for position, command in enumerate(cmds):
            if not isinstance(command, Action):
                continue
            pending = [command.__egg_ref__]
            seen: set[Ref] = set()
            while pending:
                reference = pending.pop()
                if reference in seen:
                    continue
                seen.add(reference)
                record = reference.read()
                if isinstance(record, pb.Node) and record.kind is not None and record.kind.field == "union":
                    if reference in uses and uses[reference] != position:
                        msg = "Mixed registration cannot split a shared allocating Union across Actions"
                        raise NotImplementedError(msg)
                    uses[reference] = position
                _, children = _structural_parts(reference, identity=False)
                pending.extend(
                    child for child, path in children if child.role == "nodes" and "/lambda/0/body/" not in path
                )
        for command in cmds:
            if isinstance(command, Action):
                self._state.run_program([command.__egg_ref__])
            else:
                builder = Builder()
                index = builder.add(
                    "rulesets",
                    pb.Ruleset(
                        kind=Oneof[Literal["rules"], pb.RuleList](
                            "rules", pb.RuleList(rules=[builder.import_ref(command.__egg_ref__)])
                        )
                    ),
                )
                root = builder.publish().ref("rulesets", index)
                self._state.publish_ruleset(root, {root: Ident("")}, direct=True)

    def function_size(self, fn: ExprCallable) -> int:
        """
        Return the number of rows in a table-backed callable.

        Relations, constructors, and bodyless functions have tables. Eager and
        builtin primitives do not.
        """
        reference = resolve_callable(fn)
        declaration: pb.Declaration = reference.read()
        if declaration.kind is None or not isinstance(
            declaration.kind.value, pb.Function | pb.Constructor | pb.Relation
        ):
            msg = (
                "This operation requires a table-backed relation, constructor, or bodyless function; "
                "eager and builtin primitives do not have tables"
            )
            raise ValueError(msg)
        table = declaration.kind.value.name
        builder = Builder()
        builder.import_ref(reference)
        command = builder.add(
            "commands",
            pb.Command(kind=Oneof[Literal["print_size"], pb.PrintSize]("print_size", pb.PrintSize(tables=[table]))),
        )
        response, _ = self._state.run_program([builder.publish().ref("commands", command)])
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "sizes"
            or response.outputs[0].location != pb.CommandLocation(path=[0])
            or response.outputs[0].run_report is not None
            or len(response.outputs[0].kind.value.sizes) != 1
            or response.outputs[0].kind.value.sizes[0].table != table
        ):
            raise ValueError(f"Shared adapter did not return the requested table size for {table!r}")
        return response.outputs[0].kind.value.sizes[0].rows

    def all_function_sizes(self) -> list[tuple[ExprCallable, int]]:
        """
        Return the sizes of all registered function tables.
        """
        builder = Builder()
        command = builder.add(
            "commands", pb.Command(kind=Oneof[Literal["print_size"], pb.PrintSize]("print_size", pb.PrintSize()))
        )
        response, _ = self._state.run_program([builder.publish().ref("commands", command)])
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "sizes"
            or response.outputs[0].location != pb.CommandLocation(path=[0])
            or response.outputs[0].run_report is not None
        ):
            msg = "Shared adapter did not return all table sizes"
            raise ValueError(msg)
        names: set[str] = set()
        result: list[tuple[ExprCallable, int]] = []
        for entry in response.outputs[0].kind.value.sizes:
            if not entry.table or entry.table in names:
                msg = "Shared adapter returned invalid or duplicate table sizes"
                raise ValueError(msg)
            names.add(entry.table)
            if (reference := self._state.definitions.get(("callable", entry.table))) is not None:
                result.append((cast("ExprCallable", create_callable(reference)), entry.rows))
        return result

    def function_values(  # noqa: C901, PLR0912
        self, fn: Callable[..., BASE_EXPR] | BASE_EXPR, length: int | None = None
    ) -> dict[BASE_EXPR, BASE_EXPR]:
        """
        Return the rows of a table-backed callable as calls mapped to values.

        Relations, constructors, and bodyless functions have tables. Eager and
        builtin primitives do not.

        If length is specified, only the first `length` rows will be returned;
        zero returns no rows. The callable is registered before length validation.

        Raises ValueError if any selected argument or output has no extractable
        representative. The error identifies the table, zero-based row, and
        argument index or output. No partial dictionary or placeholder is returned.
        """
        reference = resolve_callable(fn)
        declaration: pb.Declaration = reference.read()
        if declaration.kind is None or not isinstance(
            declaration.kind.value, pb.Function | pb.Constructor | pb.Relation
        ):
            msg = (
                "This operation requires a table-backed relation, constructor, or bodyless function; "
                "eager and builtin primitives do not have tables"
            )
            raise ValueError(msg)
        signature = declaration.kind.value
        if length is not None:
            self._state.run_program([], roots=[reference])
            try:
                length = index_integer(length)
                if length < 0:
                    msg = "can't convert negative int to unsigned"
                    raise OverflowError(msg)  # noqa: TRY301
                if length > 2**64 - 1:
                    msg = "int too big to convert"
                    raise OverflowError(msg)  # noqa: TRY301
            except BaseException as error:
                error.add_note("while processing 'length'")
                raise
        builder = Builder()
        builder.import_ref(reference)
        command = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["print_function"], pb.PrintFunction](
                    "print_function", pb.PrintFunction(table=signature.name, max_rows=length)
                )
            ),
        )
        response, owner = self._state.run_program([builder.publish().ref("commands", command)])
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "printed_function"
            or response.outputs[0].location != pb.CommandLocation(path=[0])
            or response.outputs[0].kind.value.table != signature.name
            or response.outputs[0].run_report is not None
        ):
            raise ValueError(f"Shared adapter did not return the requested table {signature.name!r}")
        builder = Builder()
        builder.import_ref(reference)
        rows: list[tuple[int, Ref]] = []
        for row_index, row in enumerate(response.outputs[0].kind.value.rows):
            if len(row.args) != len(signature.inputs) or row.output is None:
                raise ValueError(f"Malformed table {signature.name!r} row {row_index}")
            values = []
            for position, cell in enumerate([*row.args, row.output]):
                location = f"argument {position}" if position < len(row.args) else "output"
                context = f"Table {signature.name!r} row {row_index} {location}"
                if cell.kind is None:
                    raise ValueError(f"{context} omitted its cell kind")
                if cell.kind.field == "term":
                    if cell.kind.value >= len(response.nodes):
                        raise ValueError(f"{context} references an invalid term")
                    value = owner.ref("nodes", cell.kind.value)
                    node: pb.Node = value.read()
                    if node.kind is None or node.sort_id >= len(response.sorts):
                        raise ValueError(f"{context} references an invalid node")
                    actual_sort = owner.ref("sorts", node.sort_id)
                    values.append(value)
                else:
                    if cell.kind.value >= len(response.sorts):
                        raise ValueError(f"{context} references an invalid unavailable sort")
                    actual_sort = owner.ref("sorts", cell.kind.value)
                expected = (
                    signature.inputs[position].sort
                    if position < len(row.args)
                    else signature.output
                    if not isinstance(signature, pb.Relation)
                    else None
                )
                valid_sort = (
                    actual_sort.read().kind == Oneof("family", pb.HostSort(name="Unit"))
                    if expected is None
                    else StructuralView(actual_sort) == StructuralView(reference.owner.ref("sorts", expected))
                )
                if not valid_sort:
                    raise ValueError(f"{context} has the wrong sort")
                if cell.kind.field == "unavailable_sort":
                    raise ValueError(f"{context} has no extractable representative")
            key = builder.add(
                "nodes",
                pb.Node(
                    sort_id=builder.import_ref(actual_sort),
                    kind=Oneof[Literal["call"], pb.Call](
                        "call",
                        pb.Call(func=signature.name, args=[builder.import_ref(value) for value in values[:-1]]),
                    ),
                ),
            )
            rows.append((key, values[-1]))
        result_owner = builder.publish()
        return {
            cast("BASE_EXPR", RuntimeExpr(result_owner.ref("nodes", key))): cast("BASE_EXPR", RuntimeExpr(value))
            for key, value in rows
        }

    def lookup_function_value(self, expr: BASE_EXPR) -> BASE_EXPR | None:
        """
        Read an existing table row without evaluating or materializing keys.

        The table must already be registered. Standalone calls resolve portable
        values and existing table-call keys in a fresh snapshot; a missing key
        or row returns None. Opaque class values belong to that observation, not
        to a later lookup or live root.
        Use ``self.let(name, original_call)`` to retain a live value instead.
        Portable primitive results remain usable across observations.

        During extraction, lookup uses the active callback observation and only
        its supplied values. It never executes a command or initializes a binding.
        This deliberately replaces the prototype's key materialization and
        transparent raw-value retention; register or bind expressions explicitly.
        """
        runtime_expr = to_runtime_expr(expr)
        node: pb.Node = runtime_expr.__egg_ref__.read()
        if node.kind is not None and node.kind.field == "call":
            declaration: pb.Declaration = _definition_at(
                runtime_expr.__egg_ref__, "callable", node.kind.value.func
            ).read()
            if declaration.kind is None or not isinstance(
                declaration.kind.value, pb.Function | pb.Constructor | pb.Relation
            ):
                msg = "Lookup requires a table-backed callable; eager and builtin primitives do not have tables"
                raise ValueError(msg)
        if self._cost_model_frames:
            observation, supplied = self._cost_model_frames[-1]
        else:
            snapshot = self.freeze()
            assert snapshot.program is not None
            observation, supplied = _CostModelObservation(self._state, snapshot.program), None
        value = observation.lookup(runtime_expr.__egg_ref__, supplied=supplied)
        return None if value is None else cast("BASE_EXPR", RuntimeExpr(value))

    def has_custom_cost(self, fn: ExprCallable) -> bool:
        """Inspect assigned row costs from a complete shared snapshot."""
        reference = resolve_callable(fn)
        declaration: pb.Declaration = reference.read()
        assert declaration.kind is not None
        name = declaration.kind.value.name
        if self._cost_model_frames:
            return any(table == name for table, _ in self._cost_model_frames[-1][0].assigned)
        snapshot = self.freeze()
        assert snapshot.program is not None
        return any(
            command.kind is not None
            and command.kind.field == "action"
            and command.kind.value.kind is not None
            and command.kind.value.kind.field == "set_cost"
            and command.kind.value.kind.value.target is not None
            and command.kind.value.kind.value.target.func == name
            for command in snapshot.program.commands
        )

    def freeze(self) -> pb.EGraphSnapshot:
        """Export the complete logical state as an owned protobuf snapshot."""
        builder = Builder()
        command = builder.add("commands", pb.Command(kind=Oneof[Literal["freeze"], pb.Freeze]("freeze", pb.Freeze())))
        response, _ = self._state.run_program([builder.publish().ref("commands", command)])
        if (
            len(response.outputs) != 1
            or response.outputs[0].kind is None
            or response.outputs[0].kind.field != "snapshot"
            or response.outputs[0].kind.value.options is None
            or response.outputs[0].kind.value.program is None
        ):
            msg = "Shared adapter did not return the requested logical snapshot"
            raise ValueError(msg)
        return pb.EGraphSnapshot.from_binary(response.outputs[0].kind.value.to_binary())


# Either a constant or a function.
ExprCallable: TypeAlias = Callable[..., BaseExpr] | BaseExpr


@dataclass(frozen=True)
class _WrappedMethod:
    """
    Used to wrap a method and store some extra options on it before processing it when processing the class.
    """

    egg_fn: str | None
    cost: int | None
    merge: Callable[[object, object], object] | None
    fn: Callable
    preserve: bool
    mutates_self: bool
    unextractable: bool
    subsume: bool
    reverse_args: bool

    def __call__(self, *args, **kwargs) -> Never:
        msg = "We should never call a wrapped method. Did you forget to wrap the class?"
        raise NotImplementedError(msg)


def ruleset(
    rule_or_generator: RewriteOrRule | RewriteOrRuleGenerator | None = None,
    *rules: RewriteOrRule,
    name: None | str = None,
) -> Ruleset:
    """
    Creates a ruleset with the following rules.

    If no name is provided, try using the name of the funciton.
    """
    module: str | None
    if isinstance(rule_or_generator, FunctionType):
        name = name or rule_or_generator.__name__
        module = rule_or_generator.__module__
    else:
        module = _get_module()
    r = Ruleset(Ident(name, module if name else None) if name is not None else None)
    if rule_or_generator is not None:
        r.register(rule_or_generator, *rules, _increase_frame=True)
    return r


class Schedule:
    """
    A composable e-graph schedule.

    Use ``left + right`` to run schedules in sequence, ``schedule * n`` to
    repeat a schedule a fixed number of times, and ``schedule.saturate()`` to
    repeat until the schedule stops changing the e-graph.
    """

    def __init__(
        self,
        reference: Ref,
        bindings: dict[str, Ruleset | UnstableCombinedRuleset],
        schedulers: dict[str, BackOff] | None = None,
    ) -> None:
        if reference.role != "commands":
            msg = "Schedules require a canonical command reference"
            raise TypeError(msg)
        self._command = reference
        # Resolution/lifetime ownership only; all control topology is in Command.
        self._bindings = bindings.copy()
        self._schedulers = {} if schedulers is None else schedulers.copy()

    def _resolve(self, state: EGraphState | None) -> Ref:  # noqa: C901, PLR0912
        """Materialize live sources, then relocate this one canonical command."""
        builder = Builder()
        command: pb.Command = _copy_record(
            self._command.read(),
            relocate=lambda role, index: builder.import_ref(self._command.owner.ref(role, index)),
        )
        pending = [command]
        leaves: list[pb.Run] = []
        local_schedulers: dict[str, BackOff] = {}
        persistent_schedulers: dict[str, BackOff] = {}
        while pending:
            current = pending.pop()
            if current.kind is None:
                msg = "Schedule command has no kind"
                raise ValueError(msg)
            if current.kind.field == "run":
                scheduler = self._schedulers.get(current.kind.value.scheduler)
                if scheduler is not None:
                    selected = persistent_schedulers if scheduler._persistent else local_schedulers
                    selected.setdefault(current.kind.value.scheduler, scheduler)
                leaves.append(current.kind.value)
            elif current.kind.field in {"repeat", "saturate"}:
                control = current.kind.value
                if state is not None:
                    count = (
                        control.times
                        if isinstance(control, pb.Repeat)
                        else (control.max_iterations if control.has_field("max_iterations") else None)
                    )
                    if count is not None and not 0 <= count < 2**64:
                        msg = "can't convert schedule count to unsigned native size"
                        raise OverflowError(msg)
                pending.extend(reversed(control.body))
            elif current.kind.field == "bind_scheduler":
                continue
            else:
                msg = f"Schedule command is not migrated: {current.kind.field}"
                raise NotImplementedError(msg)
        identifiers: dict[Ref, Ident] = {}
        snapshots: dict[str, Ref] = {}
        # Source precedence is authored leaf order, not binding-map insertion
        # order: seq(a, b, a) must select a last even when its token is reused.
        for leaf in leaves:
            if leaf.ruleset is None or leaf.ruleset.kind is None or leaf.ruleset.kind.field != "name":
                msg = "Schedule template requires a live named target"
                raise ValueError(msg)
            name = leaf.ruleset.kind.value
            if name:
                snapshots[name] = self._bindings[name]._snapshot(identifiers)
        for leaf in leaves:
            if leaf.ruleset is None or leaf.ruleset.kind is None or leaf.ruleset.kind.field != "name":
                msg = "Schedule template requires a live named target"
                raise ValueError(msg)
            name = leaf.ruleset.kind.value
            if name:
                reference = snapshots[name]
                if state is not None:
                    reference = state.publish_ruleset(reference, identifiers)
            elif state is None:
                continue
            else:
                default = Builder()
                index = default.add(
                    "rulesets", pb.Ruleset(kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList()))
                )
                reference = default.publish().ref("rulesets", index)
                reference = state.publish_ruleset(reference, {reference: Ident("")})
            leaf.ruleset = pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", builder.import_ref(reference)))
        if state is not None:
            for scheduler in persistent_schedulers.values():
                state.bind_scheduler(scheduler.__egg_ref__)
        if local_schedulers:
            command = pb.Command(
                kind=Oneof[Literal["repeat"], pb.Repeat](
                    "repeat",
                    pb.Repeat(
                        times=1,
                        body=[
                            *(_copy_record(scheduler.__egg_ref__.read()) for scheduler in local_schedulers.values()),
                            command,
                        ],
                    ),
                )
            )
        index = builder.add("commands", command)
        return builder.publish().ref("commands", index)

    def __str__(self) -> str:
        return pretty_ref(self._resolve(None), custom_literal=_format_pyobject)

    def __repr__(self) -> str:
        return str(self)

    def __mul__(self, length: int) -> Schedule:
        """
        Repeat the schedule a number of times.
        """
        source = run(self) if isinstance(self, Ruleset | UnstableCombinedRuleset) else self
        length = index_integer(length)
        builder = Builder()
        command: pb.Command = _copy_record(
            source._command.read(),
            relocate=lambda role, index: builder.import_ref(source._command.owner.ref(role, index)),
        )
        kind = (
            Oneof[Literal["repeat"], pb.Repeat]("repeat", pb.Repeat(times=0, body=[command]))
            if length == 0
            else Oneof[Literal["saturate"], pb.Saturate]("saturate", pb.Saturate(max_iterations=length, body=[command]))
        )
        index = builder.add("commands", pb.Command(kind=kind))
        return Schedule(builder.publish().ref("commands", index), source._bindings, source._schedulers)

    def saturate(self) -> Schedule:
        """
        Run the schedule until the e-graph is saturated.
        """
        source = run(self) if isinstance(self, Ruleset | UnstableCombinedRuleset) else self
        builder = Builder()
        command: pb.Command = _copy_record(
            source._command.read(),
            relocate=lambda role, index: builder.import_ref(source._command.owner.ref(role, index)),
        )
        index = builder.add(
            "commands",
            pb.Command(kind=Oneof[Literal["saturate"], pb.Saturate]("saturate", pb.Saturate(body=[command]))),
        )
        return Schedule(builder.publish().ref("commands", index), source._bindings, source._schedulers)

    def __add__(self, other: Schedule) -> Schedule:
        """
        Run two schedules in sequence.
        """
        return seq(self, other)


class Ruleset(Schedule):
    """
    A collection of rules, which can be run as a schedule.
    """

    def __init__(self, ident: Ident | None) -> None:
        self.ident = ident
        self.deferred_rule_gens: list[Callable[[], Iterable[RewriteOrRule]]] = []
        builder = Builder()
        index = builder.add("rulesets", pb.Ruleset(kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList())))
        self.__egg_ref__ = builder.publish().ref("rulesets", index)

    @property
    def __egg_decls__(self) -> Never:
        msg = "Canonical rulesets have no legacy declaration projection"
        raise NotImplementedError(msg)

    def _snapshot(self, identifiers: dict[Ref, Ident]) -> Ref:
        while self.deferred_rule_gens:
            generator = self.deferred_rule_gens.pop()
            with set_current_ruleset(self):
                rules = list(generator())
            for rule in rules:
                self.append(rule)
        identifiers.pop(self.__egg_ref__, None)
        identifiers[self.__egg_ref__] = self.__egg_ident__
        return self.__egg_ref__

    def append(self, rule: RewriteOrRule) -> None:
        """
        Register a rule with the ruleset.
        """
        if not isinstance(cast("object", rule), RewriteOrRule):
            msg = "Rulesets require canonical rules or rewrites"
            raise TypeError(msg)
        record: pb.Ruleset = self.__egg_ref__.read()
        assert record.kind is not None
        assert record.kind.field == "rules"
        builder = Builder()
        record.kind.value.rules = [
            builder.import_ref(self.__egg_ref__.owner.ref("rules", index)) for index in record.kind.value.rules
        ]
        record.kind.value.rules.append(builder.import_ref(rule.__egg_ref__))
        index = builder.add("rulesets", record)
        self.__egg_ref__ = builder.publish().ref("rulesets", index)

    def register(
        self,
        /,
        rule_or_generator: RewriteOrRule | RewriteOrRuleGenerator,
        *rules: RewriteOrRule,
        _increase_frame: bool = False,
    ) -> None:
        """
        Register rewrites or rules, either as a function or as values.
        """
        if isinstance(rule_or_generator, RewriteOrRule):
            self.append(rule_or_generator)
            for r in rules:
                self.append(r)
        else:
            assert not rules
            current_frame = inspect.currentframe()
            assert current_frame
            original_frame = current_frame.f_back
            assert original_frame
            if _increase_frame:
                original_frame = original_frame.f_back
                assert original_frame
            self.deferred_rule_gens.append(Thunk.fn(_rewrite_or_rule_generator, rule_or_generator, original_frame))

    def __str__(self) -> str:
        return pretty_ref(self._snapshot({}), custom_literal=_format_pyobject)

    def __repr__(self) -> str:
        return str(self)

    def __or__(self, other: Ruleset | UnstableCombinedRuleset) -> UnstableCombinedRuleset:
        return unstable_combine_rulesets(self, other)

    # Create a unique name if we didn't pass one from the user
    @property
    def __egg_ident__(self) -> Ident:
        return self.ident or Ident(f"ruleset_{id(self)}")


class UnstableCombinedRuleset(Schedule):
    _next_generated_ident: ClassVar[int] = 0

    def __init__(self, ident: Ident | None, rulesets: list[Ruleset | UnstableCombinedRuleset]) -> None:
        self.ident = ident
        self.rulesets = tuple(rulesets)
        self.__egg_ref__: Ref | None = None
        self._generated_ident: Ident | None = None
        if self.ident is None:
            self._generated_ident = Ident(f"_combined_ruleset_{UnstableCombinedRuleset._next_generated_ident}")
            UnstableCombinedRuleset._next_generated_ident += 1

    @property
    def __egg_decls__(self) -> Never:
        msg = "Canonical rulesets have no legacy declaration projection"
        raise NotImplementedError(msg)

    @property
    def __egg_ident__(self) -> Ident:
        if self.ident:
            return self.ident
        assert self._generated_ident is not None
        return self._generated_ident

    def _snapshot(self, identifiers: dict[Ref, Ident]) -> Ref:
        children = tuple(child._snapshot(identifiers) for child in self.rulesets)
        previous: tuple[Ref, ...] | None = None
        if self.__egg_ref__ is not None:
            record: pb.Ruleset = self.__egg_ref__.read()
            assert record.kind is not None
            assert record.kind.field == "combined"
            previous = tuple(
                self.__egg_ref__.owner.ref("rulesets", child.kind.value)
                for child in record.kind.value.rulesets
                if child.kind is not None and child.kind.field == "index"
            )
        if previous != children:
            builder = Builder()
            index = builder.add(
                "rulesets",
                pb.Ruleset(
                    kind=Oneof[Literal["combined"], pb.CombinedRuleset](
                        "combined",
                        pb.CombinedRuleset(
                            rulesets=[
                                pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", builder.import_ref(child)))
                                for child in children
                            ]
                        ),
                    )
                ),
            )
            self.__egg_ref__ = builder.publish().ref("rulesets", index)
        assert self.__egg_ref__ is not None
        identifiers.pop(self.__egg_ref__, None)
        identifiers[self.__egg_ref__] = self.__egg_ident__
        return self.__egg_ref__

    def __str__(self) -> str:
        return pretty_ref(self._snapshot({}), custom_literal=_format_pyobject)

    def __repr__(self) -> str:
        return str(self)

    def __or__(self, other: Ruleset | UnstableCombinedRuleset) -> UnstableCombinedRuleset:
        return unstable_combine_rulesets(self, other)


def unstable_combine_rulesets(
    *rulesets: Ruleset | UnstableCombinedRuleset, name: str | None = None
) -> UnstableCombinedRuleset:
    """
    Combine multiple rulesets into a single ruleset.
    """
    return UnstableCombinedRuleset(Ident(name, _get_module()) if name else None, list(rulesets))


class RewriteOrRule:
    __slots__ = ("__egg_ref__",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, reference: Ref) -> None:
        if not isinstance(reference, Ref) or reference.role != "rules":
            msg = "Rules require a canonical rule reference"
            raise TypeError(msg)
        self.__egg_ref__ = reference.owner.ref(reference.role, reference.index)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RewriteOrRule):
            return NotImplemented
        if self is other:
            return True
        return rule_equal(self.__egg_ref__, other.__egg_ref__)

    def __str__(self) -> str:
        return pretty_ref(self.__egg_ref__, custom_literal=_format_pyobject)

    def __repr__(self) -> str:
        return str(self)


class Fact:
    """
    A query on an EGraph, either by an expression or an equivalence between multiple expressions.
    """

    __slots__ = ("__egg_ref__",)

    def __init__(self, reference: Ref) -> None:
        if reference.role != "nodes":
            msg = "Facts require an owned protobuf node"
            raise TypeError(msg)
        self.__egg_ref__ = reference

    def __str__(self) -> str:
        return pretty_ref(self.__egg_ref__, custom_literal=_format_pyobject)

    def __repr__(self) -> str:
        return str(self)

    def __bool__(self) -> bool:
        """
        Compare equality or inequality structurally, without querying an EGraph.
        """
        node: pb.Node = self.__egg_ref__.read()
        if node.kind is not None and node.kind.field == "union" and len(node.kind.value.members) == 2:
            left, right = node.kind.value.members
            return StructuralView(self.__egg_ref__.owner.ref("nodes", left)) == StructuralView(
                self.__egg_ref__.owner.ref("nodes", right)
            )
        if (
            node.kind is not None
            and node.kind.field == "call"
            and node.kind.value.func == "egglog.core.ne"
            and len(node.kind.value.args) == 2
        ):
            left, right = node.kind.value.args
            return StructuralView(self.__egg_ref__.owner.ref("nodes", left)) != StructuralView(
                self.__egg_ref__.owner.ref("nodes", right)
            )
        msg = f"Can only check equality for == or != not {self}"
        raise ValueError(msg)


class Action:
    """
    A change to an EGraph, either unioning multiple expressions, setting the value of a function call, deleting an expression, or panicing.
    """

    __slots__ = ("__egg_ref__",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, reference: Ref) -> None:
        if not isinstance(reference, Ref) or reference.role != "commands":
            msg = "Actions require a canonical command reference"
            raise TypeError(msg)
        record: pb.Command = reference.read()
        if record.kind is None or record.kind.field != "action":
            msg = "Actions require a command containing an action"
            raise TypeError(msg)
        self.__egg_ref__ = reference.owner.ref(reference.role, reference.index)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return action_equal(self.__egg_ref__, other.__egg_ref__)

    def __str__(self) -> str:
        return pretty_ref(self.__egg_ref__, custom_literal=_format_pyobject)

    def __repr__(self) -> str:
        return str(self)


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


def rewrite(lhs: EXPR, ruleset: Ruleset | None = None, *, subsume: bool = False) -> _RewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression."""
    return _RewriteBuilder(lhs, ruleset, subsume)


def birewrite(lhs: EXPR, ruleset: Ruleset | None = None) -> _BirewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression and vice versa."""
    return _BirewriteBuilder(lhs, ruleset)


def eq(expr: BASE_EXPR) -> _EqBuilder[BASE_EXPR]:
    """Check if the given expression is equal to the given value."""
    return _EqBuilder(expr)


def ne(expr: BASE_EXPR) -> _NeBuilder[BASE_EXPR]:
    """Check if the given expression is not equal to the given value."""
    return _NeBuilder(expr)


def panic(message: str) -> Action:
    """Raise an error with the given message."""
    builder = Builder()
    index = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["panic"], str]("panic", message))
            )
        ),
    )
    return Action(builder.publish().ref("commands", index))


def set_cost(expr: BaseExpr, cost: i64Like) -> Action:
    """Set the cost of the given expression."""
    runtime_expr = to_runtime_expr(expr)
    reference = runtime_expr.__egg_ref__
    node: pb.Node = reference.read()
    if node.kind is None or node.kind.field != "call":
        msg = "Can only set cost of function calls"
        raise TypeError(msg)
    builder = Builder()
    builder.import_ref(_definition_at(reference, "callable", node.kind.value.func))
    target = _copy_record(
        node.kind.value, relocate=lambda role, index: builder.import_ref(reference.owner.ref(role, index))
    )
    cost_sort = Builder()
    sort = cost_sort.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    )
    value = builder.import_ref(resolve_literal(cost_sort.publish().ref("sorts", sort), cost).__egg_ref__)
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action",
                pb.Action(
                    kind=Oneof[Literal["set_cost"], pb.SetCost]("set_cost", pb.SetCost(target=target, cost=value))
                ),
            )
        ),
    )
    return Action(builder.publish().ref("commands", command))


def let(name: str, expr: BaseExpr) -> Action:
    """
    Bind an expression once using the exact name; ``x`` and ``$x`` are distinct.

    Equal repeated writes are idempotent; unequal writes use ordinary no-merge
    errors, including their completed operand effects. Names are graph-local,
    not fresh identities tied to a call or scope.
    """
    runtime_expr = to_runtime_expr(expr)
    node: pb.Node = runtime_expr.__egg_ref__.read()
    builder = Builder()
    logical_name = f"__egglog_python_let_{name.encode('utf-8').hex()}"
    builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function](
                "function",
                pb.Function(
                    name=logical_name,
                    output=builder.import_ref(runtime_expr.__egg_ref__.owner.ref("sorts", node.sort_id)),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.CONSTANT,
                            path=["egglog", "egraph", logical_name, name],
                        )
                    ]
                )
            ),
        ),
    )
    index = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action",
                pb.Action(
                    kind=Oneof[Literal["set"], pb.Set](
                        "set",
                        pb.Set(target=pb.Call(func=logical_name), value=builder.import_ref(runtime_expr.__egg_ref__)),
                    )
                ),
            )
        ),
    )
    return Action(builder.publish().ref("commands", index))


def expr_action(expr: BaseExpr | Fact) -> Action:
    reference = expr.__egg_ref__ if isinstance(expr, Fact) else to_runtime_expr(expr).__egg_ref__
    builder = Builder()
    node = builder.import_ref(reference)
    index = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", node))
            )
        ),
    )
    return Action(builder.publish().ref("commands", index))


def delete(expr: BaseExpr) -> Action:
    """Create a delete expression."""
    reference = to_runtime_expr(expr).__egg_ref__
    node: pb.Node = reference.read()
    if node.kind is None or node.kind.field != "call":
        msg = "Delete requires a table call"
        raise TypeError(msg)
    builder = Builder()
    builder.import_ref(_definition_at(reference, "callable", node.kind.value.func))
    target = _copy_record(
        node.kind.value, relocate=lambda role, index: builder.import_ref(reference.owner.ref(role, index))
    )
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["delete"], pb.Call]("delete", target))
            )
        ),
    )
    return Action(builder.publish().ref("commands", command))


def subsume(expr: Expr) -> Action:
    """Subsume an expression so it cannot be matched against or extracted"""
    reference = to_runtime_expr(expr).__egg_ref__
    node: pb.Node = reference.read()
    if node.kind is None or node.kind.field != "call":
        msg = "Subsume requires a constructor call"
        raise TypeError(msg)
    builder = Builder()
    builder.import_ref(_definition_at(reference, "callable", node.kind.value.func))
    target = _copy_record(
        node.kind.value, relocate=lambda role, index: builder.import_ref(reference.owner.ref(role, index))
    )
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["subsume"], pb.Call]("subsume", target))
            )
        ),
    )
    return Action(builder.publish().ref("commands", command))


def expr_fact(expr: BaseExpr) -> Fact:
    runtime_expr = to_runtime_expr(expr)
    return Fact(runtime_expr.__egg_ref__)


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    """Create a union of the given expression."""
    return _UnionBuilder(lhs=lhs)


def set_(lhs: BASE_EXPR) -> _SetBuilder[BASE_EXPR]:
    """Create a set of the given expression."""
    return _SetBuilder(lhs=lhs)


def rule(
    *facts: FactLike,
    ruleset: Ruleset | None = None,
    name: str | None = None,
    eval_mode: RuleEvalMode = "seminaive",
    no_decomp: bool = False,
) -> _RuleBuilder:
    """Create a rule with the given facts."""
    return _RuleBuilder(
        facts=_fact_likes(facts),
        name=name,
        ruleset=ruleset,
        eval_mode=eval_mode,
        no_decomp=no_decomp,
    )


def var(name: str, bound: TypeForm[T], egg_name: str | None = None) -> T:
    """Create a typed query variable named ``egg_name or name or '_'``."""
    return cast("T", _var(name, bound, egg_name=egg_name))


def _var(name: object, bound: object, egg_name: object) -> RuntimeExpr:
    """Create a new variable with the given name and type."""
    if not isinstance(name, str):
        msg = "Variable names must be strings"
        raise TypeError(msg)
    if egg_name is not None and not isinstance(egg_name, str):
        msg = "Variable aliases must be strings"
        raise TypeError(msg)
    sort = resolve_type_annotation(bound)
    if not isinstance(sort, Ref) or sort.role != "sorts":
        msg = "Variables require a concrete sort"
        raise TypeError(msg)
    builder = Builder()
    index = builder.add(
        "nodes",
        pb.Node(sort_id=builder.import_ref(sort), kind=Oneof[Literal["var"], str]("var", egg_name or name or "_")),
    )
    return RuntimeExpr(builder.publish().ref("nodes", index))


def vars_(names: str, bound: type[BASE_EXPR]) -> Iterable[BASE_EXPR]:
    """Create variables with the given names and type."""
    for name in names.split(" "):
        yield var(name, bound)


@dataclass
class _RewriteBuilder(Generic[EXPR]):
    lhs: EXPR
    ruleset: Ruleset | None
    subsume: bool

    def to(self, rhs: EXPR, *conditions: FactLike) -> RewriteOrRule:
        if self.ruleset is not None and not isinstance(cast("object", self.ruleset), Ruleset):
            msg = "Rewrite destinations must be Rulesets"
            raise TypeError(msg)
        lhs = to_runtime_expr(self.lhs)
        facts = _fact_likes(conditions)
        rhs = convert_to_same_type(rhs, lhs)
        builder = Builder()
        index = builder.add(
            "rules",
            pb.RuleDecl(
                kind=Oneof[Literal["rewrite"], pb.Rewrite](
                    "rewrite",
                    pb.Rewrite(
                        lhs=builder.import_ref(lhs.__egg_ref__),
                        rhs=builder.import_ref(rhs.__egg_ref__),
                        conditions=[builder.import_ref(fact.__egg_ref__) for fact in facts],
                        subsume=self.subsume,
                    ),
                ),
                eval_mode=pb.RuleEvalMode.SEMINAIVE,
            ),
        )
        result = RewriteOrRule(builder.publish().ref("rules", index))
        if self.ruleset is not None:
            self.ruleset.append(result)
        return result

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return f"rewrite({lhs})"


@dataclass
class _BirewriteBuilder(Generic[EXPR]):
    lhs: EXPR
    ruleset: Ruleset | None

    def to(self, rhs: EXPR, *conditions: FactLike) -> RewriteOrRule:
        lhs = to_runtime_expr(self.lhs)
        facts = _fact_likes(conditions)
        rhs = convert_to_same_type(rhs, lhs)
        if self.ruleset is not None and not isinstance(cast("object", self.ruleset), Ruleset):
            msg = "Birewrite destinations must be Rulesets"
            raise TypeError(msg)
        builder = Builder()
        index = builder.add(
            "rules",
            pb.RuleDecl(
                kind=Oneof[Literal["birewrite"], pb.BiRewrite](
                    "birewrite",
                    pb.BiRewrite(
                        lhs=builder.import_ref(lhs.__egg_ref__),
                        rhs=builder.import_ref(rhs.__egg_ref__),
                        conditions=[builder.import_ref(fact.__egg_ref__) for fact in facts],
                    ),
                ),
                eval_mode=pb.RuleEvalMode.SEMINAIVE,
            ),
        )
        result = RewriteOrRule(builder.publish().ref("rules", index))
        if self.ruleset is not None:
            self.ruleset.append(result)
        return result

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return f"birewrite({lhs})"


@dataclass
class _EqBuilder(Generic[BASE_EXPR]):
    expr: BASE_EXPR

    def to(self, other: BASE_EXPR) -> Fact:
        expr = to_runtime_expr(self.expr)
        other = convert_to_same_type(other, expr)
        node: pb.Node = expr.__egg_ref__.read()
        builder = Builder()
        index = builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(expr.__egg_ref__.owner.ref("sorts", node.sort_id)),
                kind=Oneof[Literal["union"], pb.Union](
                    "union",
                    pb.Union(members=[builder.import_ref(expr.__egg_ref__), builder.import_ref(other.__egg_ref__)]),
                ),
            ),
        )
        return Fact(builder.publish().ref("nodes", index))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = to_runtime_expr(self.expr)
        return f"eq({expr})"


@dataclass
class _NeBuilder(Generic[BASE_EXPR]):
    lhs: BASE_EXPR

    def to(self, rhs: BASE_EXPR) -> Unit:
        lhs = to_runtime_expr(self.lhs)
        rhs = convert_to_same_type(rhs, lhs)
        return cast("Unit", RuntimeFunction(builtin_catalog().definitions["callable", "egglog.core.ne"])(lhs, rhs))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = to_runtime_expr(self.lhs)
        return f"ne({expr})"


@dataclass
class _SetBuilder(Generic[BASE_EXPR]):
    lhs: BASE_EXPR

    def to(self, rhs: BASE_EXPR) -> Action:
        lhs = to_runtime_expr(self.lhs)
        node: pb.Node = lhs.__egg_ref__.read()
        if node.kind is None or node.kind.field != "call":
            msg = "Can only set function calls"
            raise TypeError(msg)
        call = node.kind.value
        declaration = _definition_at(lhs.__egg_ref__, "callable", call.func)
        declared: pb.Declaration = declaration.read()
        if declared.kind is None or declared.kind.field != "function":
            msg = "Can only set table-backed Function calls"
            raise TypeError(msg)
        rhs_expr = resolve_literal(declaration.owner.ref("sorts", declared.kind.value.output), rhs)
        builder = Builder()
        builder.import_ref(declaration)
        target = pb.Call(
            func=call.func,
            args=[builder.import_ref(lhs.__egg_ref__.owner.ref("nodes", index)) for index in call.args],
        )
        index = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["set"], pb.Set](
                            "set", pb.Set(target=target, value=builder.import_ref(rhs_expr.__egg_ref__))
                        )
                    ),
                )
            ),
        )
        return Action(builder.publish().ref("commands", index))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return f"set_({pretty_ref(lhs.__egg_ref__, custom_literal=_format_pyobject)})"


@dataclass
class _UnionBuilder(Generic[EXPR]):
    lhs: EXPR

    def with_(self, rhs: EXPR) -> Action:
        lhs = to_runtime_expr(self.lhs)
        rhs_expr = convert_to_same_type(rhs, lhs)
        node: pb.Node = lhs.__egg_ref__.read()
        builder = Builder()
        union = builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(lhs.__egg_ref__.owner.ref("sorts", node.sort_id)),
                kind=Oneof[Literal["union"], pb.Union](
                    "union",
                    pb.Union(members=[builder.import_ref(lhs.__egg_ref__), builder.import_ref(rhs_expr.__egg_ref__)]),
                ),
            ),
        )
        command = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", union))
                )
            ),
        )
        return Action(builder.publish().ref("commands", command))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return f"union({pretty_ref(lhs.__egg_ref__, custom_literal=_format_pyobject)})"


@dataclass
class _RuleBuilder:
    facts: tuple[Fact, ...]
    name: str | None
    ruleset: Ruleset | None
    eval_mode: RuleEvalMode
    no_decomp: bool

    def then(self, *actions: ActionLike) -> RewriteOrRule:
        if self.ruleset is not None and not isinstance(cast("object", self.ruleset), Ruleset):
            msg = "Rule destinations must be Rulesets"
            raise TypeError(msg)
        modes = {
            "seminaive": pb.RuleEvalMode.SEMINAIVE,
            "naive": pb.RuleEvalMode.NAIVE,
            "unsafe-seminaive": pb.RuleEvalMode.UNSAFE_SEMINAIVE,
        }
        if self.eval_mode not in modes:
            raise ValueError(f"Unknown rule evaluation mode: {self.eval_mode!r}")
        if self.name is not None and not isinstance(cast("object", self.name), str):
            msg = "Rule names must be strings when supplied"
            raise TypeError(msg)
        builder = Builder()
        query = [builder.import_ref(fact.__egg_ref__) for fact in self.facts]
        head: list[pb.Action] = []
        dependencies: set[tuple[str, str]] = set()
        for action in _action_likes(actions):
            reference = action.__egg_ref__
            command: pb.Command = reference.read()
            assert command.kind is not None
            assert command.kind.field == "action"
            # Inline actions must retain both their owner-local indices and
            # named closure, even for a Set target with no argument nodes.
            head.append(
                _copy_record(
                    command.kind.value,
                    # _copy_record invokes this synchronously; it never retains
                    # the callback beyond this action's owner.
                    relocate=lambda role, index: builder.import_ref(reference.owner.ref(role, index)),  # noqa: B023
                    dependency=lambda namespace, name: dependencies.add((namespace, name)),
                )
            )
            for namespace, name in dependencies:
                builder.import_ref(_definition_at(reference, namespace, name))
            dependencies.clear()
        record = pb.RuleDecl(
            kind=Oneof[Literal["rule"], pb.Rule]("rule", pb.Rule(query=query, head=head)),
            eval_mode=modes[self.eval_mode],
            no_decomp=self.no_decomp,
        )
        if self.name:
            record.name = self.name
        index = builder.add("rules", record)
        result = RewriteOrRule(builder.publish().ref("rules", index))
        if self.ruleset is not None:
            self.ruleset.append(result)
        return result

    def __str__(self) -> str:
        # TODO: Figure out how to stringify rulebuilder that preserves statements
        args = list(map(str, self.facts))
        if self.name is not None:
            args.append(f"name={self.name!r}")
        if self.eval_mode != "seminaive":
            args.append(f"eval_mode={self.eval_mode!r}")
        if self.no_decomp:
            args.append("no_decomp=True")
        if self.ruleset is not None:
            args.append(f"ruleset={self.ruleset}")
        return f"rule({', '.join(args)})"


def expr_parts(expr: BaseExpr) -> StructuralView:
    """
    Returns the underlying type and decleration of the expression. Useful for testing structural equality or debugging.
    """
    if not isinstance(cast("object", expr), RuntimeExpr):
        raise TypeError(f"Expected a RuntimeExpr not {expr}")
    return StructuralView(cast("RuntimeExpr", expr).__egg_ref__)


def to_runtime_expr(expr: BaseExpr) -> RuntimeExpr:
    if not isinstance(cast("object", expr), RuntimeExpr):
        raise TypeError(f"Expected a RuntimeExpr not {expr}")
    return cast("RuntimeExpr", expr)


def run(
    ruleset: Ruleset | UnstableCombinedRuleset | None = None, *until: FactLike, scheduler: BackOff | None = None
) -> Schedule:
    """
    Create a one-step run schedule.

    ``run()`` runs the default ruleset once. ``run(ruleset)`` runs one named
    ruleset once. Additional facts are conjunctive pre-check stop conditions.
    Scheduler configuration and conditions are authored without an engine;
    their execution still requires the shared adapter scheduler/loop bridge.
    """
    if ruleset is not None and not isinstance(cast("object", ruleset), Ruleset | UnstableCombinedRuleset):
        msg = "Run target must be a ruleset"
        raise TypeError(msg)
    if scheduler is not None and not isinstance(cast("object", scheduler), BackOff):
        msg = "Run scheduler must be a BackOff configuration"
        raise TypeError(msg)
    name = uuid4().hex if ruleset is not None else ""
    builder = Builder()
    scheduler_name = "" if scheduler is None else scheduler.__egg_ref__.read().kind.value.name
    command = pb.Command(
        kind=Oneof[Literal["run"], pb.Run](
            "run",
            pb.Run(
                ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", name)),
                scheduler=scheduler_name,
            ),
        )
    )
    if until:
        command = pb.Command(
            kind=Oneof[Literal["repeat"], pb.Repeat](
                "repeat",
                pb.Repeat(
                    times=1,
                    until=[builder.import_ref(fact.__egg_ref__) for fact in _fact_likes(until)],
                    body=[command],
                ),
            )
        )
    index = builder.add("commands", command)
    return Schedule(
        builder.publish().ref("commands", index),
        {} if ruleset is None else {name: ruleset},
        {} if scheduler is None else {scheduler_name: scheduler},
    )


def back_off(match_limit: None | int = None, ban_length: None | int = None) -> BackOff:
    """
    Author a backoff scheduler configuration without creating an engine.

    ```python
    schedule = run(analysis_ruleset).saturate() + run(ruleset, scheduler=back_off(match_limit=1000, ban_length=5)) * 10
    ```
    This will run the `analysis_ruleset` until saturation, then run `ruleset` 10 times, using a backoff scheduler.

    The backend defaults are ``match_limit=1000`` and ``ban_length=5``. Each time
    a rule is banned, both effective values double for that rule.
    Execution still requires the shared adapter's scheduler/loop bridge.
    """
    options = pb.BackOff()
    for name, value in (("match_limit", match_limit), ("ban_length", ban_length)):
        if value is not None:
            numeric = index_integer(value)
            if not 0 <= numeric < 2**64:
                raise OverflowError(f"Scheduler {name} must fit unsigned uint64")
            setattr(options, name, numeric)
    builder = Builder()
    index = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["bind_scheduler"], pb.BindScheduler](
                "bind_scheduler",
                pb.BindScheduler(
                    name=f"__egglog_python_scheduler_{uuid4().hex}",
                    scheduler=pb.Scheduler(kind=Oneof[Literal["back_off"], pb.BackOff]("back_off", options)),
                ),
            )
        ),
    )
    return BackOff(builder.publish().ref("commands", index))


@dataclass(frozen=True)
class BackOff:
    """An owned canonical configuration with frontend-only lifetime intent."""

    __egg_ref__: Ref
    _persistent: bool = False

    def __post_init__(self) -> None:
        command = self.__egg_ref__.read()
        if (
            self.__egg_ref__.role != "commands"
            or not isinstance(command, pb.Command)
            or command.kind is None
            or command.kind.field != "bind_scheduler"
            or not command.kind.value.name
            or command.kind.value.scheduler is None
            or command.kind.value.scheduler.kind is None
            or command.kind.value.scheduler.kind.field != "back_off"
        ):
            msg = "BackOff requires a canonical BindScheduler with back-off configuration"
            raise TypeError(msg)

    def persistent(self) -> BackOff:
        """
        Author a distinct identity, bound once on first use in each e-graph scope.

        Acknowledged bindings and scheduler progress follow push/pop. A failed
        deployment is not retried after transport without restoring a saved graph.
        """
        command = self.__egg_ref__.read()
        command.kind.value.name = f"__egglog_python_scheduler_{uuid4().hex}"
        builder = Builder()
        index = builder.add("commands", command)
        return BackOff(builder.publish().ref("commands", index), _persistent=True)

    def scope(self, schedule: Schedule) -> Schedule:
        """
        Author a fresh local binding immediately around this schedule.

        Placement inside a loop creates a fresh instance per iteration. Explicit
        scope overrides persistent intent for these references; no engine is bound.
        """
        source = run(schedule) if isinstance(schedule, Ruleset | UnstableCombinedRuleset) else schedule
        builder = Builder()
        command: pb.Command = _copy_record(
            source._command.read(),
            relocate=lambda role, index: builder.import_ref(source._command.owner.ref(role, index)),
        )
        binding = self.__egg_ref__.read()
        original = binding.kind.value.name
        binding.kind.value.name = f"__egglog_python_scheduler_scope_{uuid4().hex}"
        pending = [command]
        while pending:
            current = pending.pop()
            if current.kind is not None:
                if current.kind.field == "run" and current.kind.value.scheduler == original:
                    current.kind.value.scheduler = binding.kind.value.name
                elif current.kind.field in {"repeat", "saturate"}:
                    pending.extend(current.kind.value.body)
        index = builder.add(
            "commands",
            pb.Command(kind=Oneof[Literal["repeat"], pb.Repeat]("repeat", pb.Repeat(times=1, body=[binding, command]))),
        )
        schedulers = source._schedulers.copy()
        schedulers.pop(original, None)
        return Schedule(builder.publish().ref("commands", index), source._bindings, schedulers)

    def __str__(self) -> str:
        return pretty_ref(self.__egg_ref__) + (".persistent()" if self._persistent else "")

    def __repr__(self) -> str:
        return str(self)


def seq(*schedules: Schedule) -> Schedule:
    """
    Run any number of schedules in order.

    For two schedules, ``left + right`` is equivalent to ``seq(left, right)``.
    """
    sources = [
        run(schedule) if isinstance(schedule, Ruleset | UnstableCombinedRuleset) else schedule for schedule in schedules
    ]
    builder = Builder()
    body: list[pb.Command] = [
        _copy_record(
            source._command.read(),
            relocate=lambda role, index: builder.import_ref(source._command.owner.ref(role, index)),  # noqa: B023
        )
        for source in sources
    ]
    index = builder.add(
        "commands",
        pb.Command(kind=Oneof[Literal["repeat"], pb.Repeat]("repeat", pb.Repeat(times=1, body=body))),
    )
    return Schedule(
        builder.publish().ref("commands", index),
        {name: owner for source in sources for name, owner in source._bindings.items()},
        {name: scheduler for source in sources for name, scheduler in source._schedulers.items()},
    )


def _action_likes(action_likes: Iterable[ActionLike]) -> tuple[Action, ...]:
    return tuple(map(_action_like, action_likes))


def _action_like(action_like: ActionLike) -> Action:
    if isinstance(action_like, Action):
        return action_like
    return expr_action(action_like)


Command: TypeAlias = Action | RewriteOrRule


def _command_like(command_like: CommandLike) -> Command:
    if isinstance(command_like, RewriteOrRule):
        return command_like
    return _action_like(command_like)


RewriteOrRuleGenerator = Callable[..., Iterable[RewriteOrRule]]


def _rewrite_or_rule_generator(gen: RewriteOrRuleGenerator, frame: FrameType) -> Iterable[RewriteOrRule]:
    """
    Returns a thunk which will call the function with variables of the type and name of the arguments.
    """
    # Need to manually pass in the frame locals from the generator, because otherwise classes defined within function
    # will not be available in the annotations
    # combine locals and globals so that they are the same dict. Otherwise get_type_hints will go through the wrong
    # path and give an error for the test
    # python/tests/test_no_import_star.py::test_no_import_star_rulesset
    # Materialize shared rulesets before starting worker threads.
    combined = {**gen.__globals__, **frame.f_locals}
    hints = get_type_hints(gen, combined, combined)
    args = [_var(p.name, hints[p.name], egg_name=None) for p in signature(gen).parameters.values()]
    return list(gen(*args))


FactLike = Fact | BaseExpr
ActionLike: TypeAlias = Action | BaseExpr | Fact
CommandLike: TypeAlias = ActionLike | RewriteOrRule


def _fact_likes(fact_likes: Iterable[FactLike]) -> tuple[Fact, ...]:
    return tuple(map(_fact_like, fact_likes))


def _fact_like(fact_like: FactLike) -> Fact:
    if isinstance(fact_like, Fact):
        return fact_like
    return expr_fact(fact_like)


_CURRENT_RULESET = ContextVar[Ruleset | None]("CURRENT_RULESET", default=None)


def get_current_ruleset() -> Ruleset | None:
    return _CURRENT_RULESET.get()


@contextlib.contextmanager
def set_current_ruleset(r: Ruleset | None) -> Generator[None, None, None]:
    token: Token[Ruleset | None] = _CURRENT_RULESET.set(r)
    try:
        yield
    finally:
        _CURRENT_RULESET.reset(token)


def get_cost(expr: BaseExpr) -> i64:
    """
    Return a lookup of the cost of an expression. If not set, won't match.
    """
    runtime_expr = to_runtime_expr(expr)
    builder = Builder()
    reference = runtime_expr.__egg_ref__
    node: pb.Node = reference.read()
    if node.kind is None or node.kind.field != "call":
        msg = "Can only get cost of function calls, not literals or variables"
        raise TypeError(msg)
    builder.import_ref(_definition_at(reference, "callable", node.kind.value.func))
    target = _copy_record(
        node.kind.value, relocate=lambda role, index: builder.import_ref(reference.owner.ref(role, index))
    )
    sort = builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    index = builder.add("nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["get_cost"], pb.Call]("get_cost", target)))
    return cast("i64", RuntimeExpr(builder.publish().ref("nodes", index)))


class Comparable(Protocol):
    def __lt__(self, other: Self) -> bool: ...
    def __le__(self, other: Self) -> bool: ...
    def __gt__(self, other: Self) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...


COST = TypeVar("COST", bound=Comparable)
ExtractionMode: TypeAlias = Literal["tree", "greedy-dag"]


class TreeCostModel(Protocol, Generic[COST]):
    """
    A cost model for an e-graph. Used to determine the cost of an expression based on its structure and the costs of its sub-expressions.

    Called with an expression and the costs of its children, returns the total cost of the expression.

    Additionally, the cost model should guarantee that a term has a no-smaller cost
    than its subterms to avoid cycles in the extracted terms for common case usages.
    For more niche usages, a term can have a cost less than its subterms.
    As long as there is no negative cost cycle, the default extractor is guaranteed to terminate in computing the costs.
    However, the user needs to be careful to guarantee acyclicity in the extracted terms.
    """

    def __call__(self, egraph: EGraph, expr: BaseExpr, children_costs: list[COST]) -> COST:
        """
        The total cost of a term given the cost of the root e-node and its immediate children's total costs.
        """
        raise NotImplementedError


CostModel: TypeAlias = TreeCostModel


class ComparableAdd(Comparable, Protocol):
    def __add__(self, other: Self) -> Self: ...


DAG_COST = TypeVar("DAG_COST", bound=ComparableAdd)


@dataclass(frozen=True)
class DagCostModel(Generic[DAG_COST]):
    """
    An additive marginal cost model for tree or greedy-DAG extraction.

    ``marginal_cost`` excludes selected children and container elements.
    Costs are combined with Python ``+``. The operation must be associative,
    commutative, monotone, and have ``identity`` as a two-sided identity.
    Cost values must also have a total order and be effectively immutable.
    The identity is required because a generic Python ``+`` operation does not
    provide a way to construct its neutral value from the cost type.
    With tree extraction, recursive e-classes must not contain a
    cost-improving cycle or cost computation may not terminate.
    """

    marginal_cost: Callable[[EGraph, BaseExpr], DAG_COST]
    identity: DAG_COST


def default_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
    """Apply assigned C, declared C, then the ordinary node/container cost."""
    from .deconstruct import get_callable_fn  # noqa: PLC0415

    callable_fn = get_callable_fn(expr)
    if callable_fn is not None and (assigned := egraph.lookup_function_value(get_cost(expr))) is not None:
        self_cost = int(assigned)
    elif callable_fn is not None and (declared := get_callable_cost(callable_fn)) is not None:
        self_cost = declared
    else:
        node: pb.Node = to_runtime_expr(expr).__egg_ref__.read()
        primitive = node.kind.value.value if node.kind is not None and node.kind.field == "primitive_value" else None
        self_cost = (
            0
            if primitive is not None
            and primitive.field in {"vec", "set", "multiset", "map", "pair", "maybe", "partial_call", "lambda"}
            else 1
        )
    return sum(children_costs, start=self_cost)


@dataclass(frozen=True)
class _HostModelFactory:
    egraph: EGraph
    model: TreeCostModel[Comparable] | DagCostModel[ComparableAdd]

    def __call__(self, encoded: bytes) -> _HostModelRuntime:
        observation = _CostModelObservation(self.egraph._state, pb.Program.from_binary(encoded))
        return _HostModelRuntime(self.egraph, self.model, observation)


@dataclass(frozen=True)
class _HostModelRuntime:
    egraph: EGraph
    model: TreeCostModel[Comparable] | DagCostModel[ComparableAdd]
    observation: _CostModelObservation

    @property
    def identity(self) -> object:
        if not isinstance(self.model, DagCostModel):
            msg = "Tree folds do not provide an additive identity"
            raise TypeError(msg)
        return self.model.identity

    def cost(self, node: int, children: list[object]) -> object:
        frame = self.observation, self.observation.supplied_values(node)
        state = self.egraph._state
        self.egraph._cost_model_frames.append(frame)
        state._cost_model_active += 1
        try:
            expr = cast("BaseExpr", RuntimeExpr(self.observation.owner.ref("nodes", node)))
            if isinstance(self.model, DagCostModel):
                return self.model.marginal_cost(self.egraph, expr)
            return self.model(self.egraph, expr, cast("list[Comparable]", children))
        finally:
            state._cost_model_active -= 1
            assert self.egraph._cost_model_frames.pop() is frame


def get_callable_cost(fn: ExprCallable) -> int | None:
    """
    Returns the cost of a callable, if it has one set. Otherwise returns None.
    """
    reference = resolve_callable(fn)
    declaration: pb.Declaration = reference.read()
    if declaration.kind is None or declaration.kind.field != "constructor":
        return 1
    if not declaration.kind.value.has_field("cost"):
        return None
    from .deconstruct import get_literal_value  # noqa: PLC0415

    cost = get_literal_value(RuntimeExpr(reference.owner.ref("nodes", declaration.kind.value.cost)))
    if not isinstance(cost, int):
        msg = "The constructor's Python cost must be an i64 literal"
        raise TypeError(msg)
    return cost
