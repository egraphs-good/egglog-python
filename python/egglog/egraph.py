from __future__ import annotations

import contextlib
import dis
import inspect
import pathlib
import re
import sys
import tempfile
from collections.abc import Callable, Generator, Iterable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from functools import partial
from inspect import Parameter, currentframe, getmodule, signature
from operator import index as index_integer
from types import FrameType, FunctionType
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
    assert_never,
    cast,
    get_type_hints,
    overload,
)
from uuid import uuid4
from warnings import warn

if sys.version_info >= (3, 14):
    import annotationlib

import graphviz
from egglog_proto.egglog.v1 import egglog_pb as pb
from opentelemetry import trace
from protobuf import Oneof
from typing_extensions import ParamSpec, TypeForm

from . import bindings
from ._catalog import Catalog, builtin_catalog
from ._program import (
    Builder,
    Ref,
    StructuralView,
    _copy_record,
    _structural_parts,
    action_equal,
    pack,
    rule_equal,
)
from ._tracing import call_with_current_trace
from .conversion import *
from .conversion import convert_to_same_type, resolve_literal
from .declarations import *
from .declarations import is_callable_decl_constructor
from .egraph_state import *
from .ipython_magic import IN_IPYTHON
from .pretty import pretty_decl, pretty_ref
from .run_report import RunReport
from .runtime import *
from .runtime import _ClassContext, _definition_at
from .thunk import *

if TYPE_CHECKING:
    from .builtins import String, Unit, i64, i64Like


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
    except bindings.EggSmolError as err:
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
    decls: Declarations,
    return_type: TypeOrVarRef,
    merge: Callable[[object, object], object] | None,
) -> ExprDecl | None:
    if merge is None:
        return None
    old = RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), UnboundVarDecl("old", "old")))
    new = RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), UnboundVarDecl("new", "new")))
    resolved_merge = resolve_literal(return_type, merge(old, new), lambda: decls)
    decls |= resolved_merge
    return resolved_merge.__egg_typed_expr__.expr


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
    """Publish canonical headers in order, then author explicit rewrite defaults."""
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
        if name in namespace:
            if ruleset is None:
                raise NotImplementedError(f"Class variable defaults are not migrated: {runtime_cls}.{name}")
            if sort_record.kind is None or sort_record.kind.field != "eq":
                msg = "Primitive-returning defaults cannot use an explicit ruleset"
                raise ValueError(msg)
        qualified = f"{runtime_cls.__module__}.{runtime_cls.__name__}.{name}"
        egg_name = re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", qualified)
        kind = (
            Oneof[Literal["constructor"], pb.Constructor]("constructor", pb.Constructor(name=egg_name, output=output))
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
        if name in namespace:
            default = resolve_literal(sort, namespace[name])
            lhs = RuntimeFunction(definitions[-1], 0)()
            rewrite(cast("Expr", lhs), ruleset=ruleset).to(cast("Expr", default))
    bodies: list[tuple[FunctionType, Ref, bool]] = []
    for name, value in namespace.items():
        if name in hints:
            continue
        if name in {"__annotate__", "__annotate_func__"} and not isinstance(value, _WrappedMethod):
            continue
        if name in IGNORED_ATTRIBUTES and not isinstance(value, _WrappedMethod):
            continue
        if isinstance(value, _WrappedMethod):
            if (
                (
                    (value.preserve or name == "__init__")
                    and (
                        value.cost is not None
                        or value.merge is not None
                        or value.mutates_self
                        or value.unextractable
                        or value.subsume
                    )
                )
                or value.reverse_args
                or (value.preserve and (value.egg_fn is not None or name == "__init__"))
            ):
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
        definitions.append(compiler.prepare(fn, owner=runtime_cls, view_kind=view_kind, member_name=name))
        context.catalog = Catalog.from_refs(definitions)
        assert isinstance(fn, FunctionType)
        if _function_has_body(fn):
            bodies.append((fn, definitions[-1], compiler.subsume))
    for fn, reference, subsume_body in bodies:
        assert ruleset is not None
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

    def prepare(  # noqa: C901, PLR0912
        self,
        fn: object,
        *,
        owner: RuntimeClass | None = None,
        view_kind: pb.PythonCallKind = pb.PythonCallKind.FUNCTION,
        member_name: str | None = None,
    ) -> Ref:
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
            if owner is not None and (self.ruleset is None or view_kind == pb.PythonCallKind.INITIALIZER):
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
        if (
            self.builtin
            or has_merge
            or self.mutates_first_arg
            or self.cost is not None
            or self.unextractable
            or variable
        ):
            msg = "Builtin, merge, mutation, cost, unextractable and variadic free functions are not migrated"
            raise NotImplementedError(msg)
        if any(not isinstance(sort, Ref) or sort.role != "sorts" for sort in input_sorts):
            msg = "Free functions require concrete input sorts"
            raise NotImplementedError(msg)
        builder = Builder()
        inputs = []
        surface = []
        for index, (parameter, sort) in enumerate(zip(parameters, input_sorts, strict=True)):
            assert isinstance(sort, Ref)
            inputs.append(pb.Arg(sort=builder.import_ref(sort), name=parameter.name))
            if receiver and index == 0:
                continue
            parameter_record = pb.PythonParameter(core_input=index, name=parameter.name)
            if parameter.default is not Parameter.empty:
                default = resolve_literal(sort, parameter.default)
                parameter_record.default_expr = builder.import_ref(default.__egg_ref__)
            surface.append(parameter_record)
        output_index = builder.import_ref(output)
        if owner is None:
            path = [*fn.__module__.split("."), fn.__name__] if fn.__module__ else [fn.__name__]
        else:
            assert member_name is not None
            path = [*owner.__module__.split("."), owner.__name__, member_name]
        name = self.egg_fn or re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", ".".join(path))
        view = pb.PythonCallable(kind=view_kind, path=path, params=surface)
        if owner is not None:
            assert member_name is not None
            assert owner.__egg_sort__ is not None
            view.owner = pb.BindingOwner(
                kind=Oneof[Literal["sort"], int]("sort", builder.import_ref(owner.__egg_sort__))
            )
            view.path = [] if view_kind == pb.PythonCallKind.INITIALIZER else [member_name]
            if receiver:
                view.receiver = 0
        body = None
        if has_body and owner is None:
            formals = Builder()
            formal_indices = []
            for parameter, sort in zip(parameters, input_sorts, strict=True):
                assert isinstance(sort, Ref)
                formal_indices.append(
                    formals.add(
                        "nodes",
                        pb.Node(
                            sort_id=formals.import_ref(sort),
                            kind=Oneof[Literal["var"], str]("var", parameter.name),
                        ),
                    )
                )
            formal_owner = formals.publish()
            formal_refs = [formal_owner.ref("nodes", index) for index in formal_indices]
            with set_current_ruleset(self.ruleset):
                result = fn(*(RuntimeExpr(reference) for reference in formal_refs))
                if result is None:
                    msg = "Callable bodies must return a value"
                    raise ValueError(msg)
                if self.ruleset is None:
                    resolved = resolve_literal(output, result)
            if self.ruleset is not None:
                resolved = resolve_literal(output, result)
            body = _lower_function_body(
                builder, resolved.__egg_ref__, formal_refs, rewrite_body=self.ruleset is not None
            )
        kind = (
            Oneof[Literal["primitive"], pb.Primitive](
                "primitive", pb.Primitive(name=name, inputs=inputs, output=output_index, body=body)
            )
            if body is not None and self.ruleset is None
            else Oneof[Literal["constructor"], pb.Constructor](
                "constructor", pb.Constructor(name=name, inputs=inputs, output=output_index)
            )
            if equality
            else Oneof[Literal["function"], pb.Function](
                "function", pb.Function(name=name, inputs=inputs, output=output_index)
            )
        )
        index = builder.add(
            "declarations",
            pb.Declaration(
                kind=kind,
                bindings=pb.CallableBindings(python=pb.PythonBindings(views=[view])),
                doc=fn.__doc__,
            ),
        )
        reference = builder.publish().ref("declarations", index)
        Catalog.from_refs([reference])
        if body is not None and self.ruleset is not None:
            lhs = RuntimeFunction(reference, 0)(*(RuntimeExpr(formal) for formal in formal_refs))
            assert isinstance(lhs, RuntimeExpr)
            lhs_builder = Builder()
            lhs_index = _lower_function_body(lhs_builder, lhs.__egg_ref__, formal_refs, rewrite_body=True)
            rewrite(
                cast("Expr", RuntimeExpr(lhs_builder.publish().ref("nodes", lhs_index))),
                ruleset=self.ruleset,
                subsume=self.subsume,
            ).to(cast("Expr", RuntimeExpr(reference.owner.ref("nodes", body))))
        return reference


def _lower_function_body(  # noqa: C901
    builder: Builder, body: Ref, formals: list[Ref], *, rewrite_body: bool = False
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
            builder.import_ref(_definition_at(origin, namespace, name))

        builder.fill("nodes", indices[source], _copy_record(node, relocate=relocate, dependency=dependency))
    return indices[body]


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
    ident = Ident(name, _get_module())

    decls_thunk = Thunk.fn(_relation_decls, ident, tps, egg_fn)
    return cast("Callable[..., Unit]", RuntimeFunction(decls_thunk, Thunk.value(FunctionRef(ident))))


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


def _relation_decls(ident: Ident, tps: tuple[type, ...], egg_fn: str | None) -> Declarations:
    from .builtins import Unit  # noqa: PLC0415

    decls = Declarations()
    decls |= cast("RuntimeClass", Unit)
    arg_types = tuple(resolve_type_annotation_mutate(decls, tp).to_just() for tp in tps)
    decls._functions[ident] = RelationDecl(arg_types, tuple(None for _ in tps), egg_fn)
    return decls


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
    if has_default or has_merge or ruleset is not None:
        msg = "Constant defaults, merges and rulesets are not migrated"
        raise NotImplementedError(msg)
    module = _get_module()
    path = [*module.split("."), name] if module else [name]
    builder = Builder()
    output = builder.import_ref(sort)
    logical_name = egg_name or re.sub(r"[^\w\-+*/?!=<>&|^/%]", "_", "%" + ".".join(path))
    kind = (
        Oneof[Literal["constructor"], pb.Constructor]("constructor", pb.Constructor(name=logical_name, output=output))
        if return_type_is_eqsort
        else Oneof[Literal["function"], pb.Function]("function", pb.Function(name=logical_name, output=output))
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
    return cast("BASE_EXPR", RuntimeFunction(reference, 0)())


def _add_default_rewrite_function(  # noqa: C901, PLR0912
    decls: Declarations,
    ref: FunctionRef | MethodRef | PropertyRef | ClassMethodRef | InitRef,
    fn: Callable,
    args: Iterable[TypedExprDecl],
    ruleset: Ruleset | None,
    subsume: bool,
    res_type: TypeOrVarRef,
    mutates_first_arg: bool,
    has_body: bool,
) -> None:
    args = list(args)
    arg_exprs: list[RuntimeExpr | RuntimeClass] = [RuntimeExpr.__from_values__(decls, a) for a in args]
    # If this is a classmethod, add the class as the first arg
    if isinstance(ref, ClassMethodRef):
        tp = decls.get_parameterized_class(ref.ident)
        arg_exprs.insert(0, RuntimeClass(Thunk.value(decls), tp))
    elif isinstance(ref, InitRef):
        tp = decls.get_parameterized_class(ref.ident)
        arg_exprs.insert(
            0,
            RuntimeExpr.__from_values__(decls, TypedExprDecl(tp.to_just(), CallDecl(ref, tuple(args)))),
        )
    with set_current_ruleset(ruleset):
        res = fn(*arg_exprs)
    # If this function mutates its first argument and that argument changed, use it as the result.
    if mutates_first_arg and arg_exprs[0].__egg_typed_expr__ != args[0]:
        res = arg_exprs[0]
    decl = decls.get_callable_decl(ref)
    assert isinstance(decl, ConstructorDecl | FunctionDecl)
    if res is None:
        if has_body:
            msg = "Callable bodies must return a value"
            raise ValueError(msg)
        if ruleset is not None and isinstance(ref, FunctionRef):
            msg = "Explicit rulesets require a body"
            raise ValueError(msg)
        if subsume:
            msg = "subsume requires a body"
            raise ValueError(msg)
        return
    return_type_is_eqsort = isinstance(res_type, TypeRefWithVars) and not decls._classes[res_type.ident].builtin
    if return_type_is_eqsort and isinstance(decl, FunctionDecl) and decl.merge is not None:
        msg = "Eqsort-returning callables with bodies cannot use merge"
        raise ValueError(msg)
    if return_type_is_eqsort and ruleset is not None:
        _add_default_rewrite(decls, ref, res_type, res, ruleset, subsume)
        return

    if isinstance(decl, ConstructorDecl):
        if decl.cost is not None:
            msg = "Eqsort-returning eager bodies cannot use cost"
            raise ValueError(msg)
        if decl.unextractable:
            msg = "Eqsort-returning eager bodies cannot be unextractable"
            raise ValueError(msg)
    else:
        if ruleset is not None:
            msg = "Primitive-returning callables with bodies cannot use an explicit ruleset"
            raise ValueError(msg)
        if decl.builtin:
            msg = "Builtin callables cannot have a body"
            raise ValueError(msg)
        if decl.merge is not None:
            msg = "Primitive-returning callables with bodies cannot use merge"
            raise ValueError(msg)

    resolved_value = resolve_literal(res_type, res, Thunk.value(decls))
    decls |= resolved_value
    match decl:
        case ConstructorDecl(signature, egg_name, _, _, doc):
            decls.set_function_decl(
                ref,
                FunctionDecl(signature=signature, egg_name=egg_name, body=resolved_value.__egg_typed_expr__, doc=doc),
            )
        case FunctionDecl():
            decls.set_function_decl(ref, replace(decl, body=resolved_value.__egg_typed_expr__))


def _add_default_rewrite(
    decls: Declarations,
    ref: FunctionRef | ConstantRef | MethodRef | ClassMethodRef | InitRef | ClassVariableRef | PropertyRef,
    type_ref: TypeOrVarRef,
    default_rewrite: object,
    ruleset: Ruleset,
    subsume: bool,
) -> None:
    """
    Adds a default rewrite for the callable when an explicit ruleset is provided.
    """
    msg = "Canonical class/default rewrite authoring is not migrated"
    raise NotImplementedError(msg)


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
    # Raw values supplied while extraction holds this e-graph read-only.
    _cost_callback_values: dict[TypedExprDecl, bindings.Value] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __init__(
        self,
        *actions: ActionLike,
        seminaive: bool = True,
        save_egglog_string: bool = False,
        num_threads: int = 1,
        no_decomp: bool = False,
    ) -> None:
        if not seminaive or no_decomp or save_egglog_string:
            msg = "Nondefault native execution flags and source transcripts are not migrated to the shared adapter"
            raise NotImplementedError(msg)
        with _TRACER.start_as_current_span("create"):
            with _TRACER.start_as_current_span("create_bindings"):
                self._state = EGraphState(num_threads=num_threads)
            self._state_stack = []
            self._token_stack = []
            self._cost_callback_values = None
            if actions:
                self.register(*actions)

    def _add_decls(self, *decls: DeclarationsLike) -> None:
        msg = "Legacy declaration submission is removed; this operation is not migrated"
        raise NotImplementedError(msg)

    def set_report_level(self, level: bindings._ReportLevel) -> None:
        """
        Set the level of detail recorded in subsequent run reports.
        """
        levels = {
            bindings.TimeOnly: pb.ReportLevel.TIME_ONLY,
            bindings.WithPlan: pb.ReportLevel.WITH_PLAN,
            bindings.StageInfo: pb.ReportLevel.STAGE_INFO,
        }
        if type(level) not in levels:
            msg = "Expected a native report level"
            raise TypeError(msg)
        if self._state.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        request = pb.ConfigureEGraphResourcesRequest(
            egraph_id=self._state.handle,
            operation=Oneof[Literal["report_level"], pb.ReportLevel]("report_level", levels[type(level)]),
        )
        pb.ConfigureEGraphResourcesResponse.from_binary(self._state.transport.configure_resources(request.to_binary()))

    def num_threads(self) -> int:
        """Return the number of worker threads configured for this e-graph."""
        return self._state.configure_resources()

    def set_num_threads(self, num_threads: int) -> None:
        """
        Set the number of worker threads used by this e-graph.

        Passing ``1`` keeps execution serial. Passing ``0`` uses available
        parallelism.
        """
        self._state.configure_resources(num_threads)

    def no_decomp(self) -> bool:
        """Return whether rule decomposition is disabled for this e-graph."""
        return False

    def set_no_decomp(self, no_decomp: bool) -> None:
        """Set whether subsequently registered rules skip decomposition."""
        self._egraph.set_no_decomp(no_decomp)

    @property
    def as_egglog_string(self) -> str:
        """
        Return the replayable Egglog transcript for this e-graph.

        If a failure may have partially changed the graph and cannot be
        represented by a replayable command, the transcript is invalidated and
        this property raises ``RuntimeError``.
        """
        return self._state.egglog_string()

    def close(self) -> None:
        """
        Close and remove the optional saved Egglog transcript.

        For an e-graph created with ``save_egglog_string=True``, subsequent
        commands are rejected because they could no longer be recorded.
        Otherwise this method has no effect and the e-graph remains usable.
        """
        # With transcripts disabled, close historically leaves the graph usable.
        # Native handles are released when their owning state is disposed.

    def _ipython_display_(self) -> None:
        self.display()

    def input(self, fn: Callable[..., String], path: str) -> None:
        """
        Load a CSV file into a table-backed callable.

        Eager and builtin primitives do not have tables and cannot be input
        targets.
        """
        ref, decls = resolve_callable(fn)
        self._add_decls(decls)
        self._require_table_backed(ref)
        self._state.run_program(bindings.Input(span(1), self._state.callable_ref_to_egg(ref)[0], path))

    def _callable_to_egg(self, fn: ExprCallable) -> tuple[CallableRef, str]:
        ref, decls = resolve_callable(fn)
        self._add_decls(decls)
        return ref, self._state.callable_ref_to_egg(ref)[0]

    # TODO: Change let to be action...
    def let(self, name: str, expr: BASE_EXPR) -> BASE_EXPR:
        """
        Define a new expression in the egraph and return a reference to it.
        """
        action = let(name, expr)
        self.register(action)
        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr)
        return cast(
            "BASE_EXPR",
            RuntimeExpr.__from_values__(
                self.__egg_decls__, TypedExprDecl(runtime_expr.__egg_typed_expr__.tp, LetRefDecl(name))
            ),
        )

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
        try:
            self.check(*facts)
        except bindings.EggSmolError as e:
            error = e.__dict__.get("_egglog_proto_error")
            if (
                isinstance(error, pb.Error)
                and error.code == pb.ErrorCode.CHECK_FAILED
                and error.location is not None
                and error.location.path == [0]
                and not error.location.iterations
            ):
                return False
            raise
        return True

    @_TRACER.start_as_current_span("check")
    def check(self, *facts: FactLike) -> None:
        """
        Check if a fact is true in the egraph.
        """
        builder = Builder()
        indices = [builder.import_ref(fact.__egg_ref__) for fact in _fact_likes(facts)]
        command = builder.add(
            "commands", pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=indices)))
        )
        self._state.run_program([builder.publish().ref("commands", command)])

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
        _extractor_options(extractor)
        if cost_model is not None or extractor != "tree":
            msg = "Custom Python cost models and non-tree extraction are not migrated"
            raise NotImplementedError(msg)
        runtime_expr = to_runtime_expr(expr)
        builder = Builder()
        root = builder.import_ref(runtime_expr.__egg_ref__)
        index = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["extract"], pb.Extract](
                    "extract",
                    pb.Extract(roots=[root], variants=1, extractor=pb.Extractor.TREE),
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
        if not isinstance(cost, int):
            msg = "Expected an i64 extraction cost"
            raise TypeError(msg)
        return result, cost

    def _from_termdag(self, termdag: bindings.TermDag, term: int, tp: JustTypeRef) -> Any:
        (new_typed_expr,) = self._state.exprs_from_egg(termdag, [term], tp)
        return RuntimeExpr.__from_values__(self.__egg_decls__, new_typed_expr)

    @overload
    def extract_multiple(self, expr: BASE_EXPR, n: int, *, extractor: ExtractionMode = "tree") -> list[BASE_EXPR]: ...

    @overload
    def extract_multiple(
        self, expr: Sequence[BASE_EXPR], n: int, *, extractor: ExtractionMode = "tree"
    ) -> list[list[BASE_EXPR]]: ...

    def extract_multiple(
        self,
        expr: BASE_EXPR | Sequence[BASE_EXPR],
        n: int,
        *,
        extractor: ExtractionMode = "tree",
    ) -> list[BASE_EXPR] | list[list[BASE_EXPR]]:
        """
        Extract up to ``n`` variants of one expression or each expression in a sequence.

        Sequence results preserve the roots' order and may contain an empty
        list when a root has no extractable representation.
        """
        if n <= 0:
            msg = "The number of variants must be positive"
            raise ValueError(msg)
        extractor_options = _extractor_options(extractor)
        if isinstance(expr, Sequence):
            if not expr:
                msg = "extract_multiple requires at least one expression"
                raise ValueError(msg)
            runtime_exprs = [to_runtime_expr(item) for item in expr]
            self._add_decls(*runtime_exprs)
            egg_exprs = [self._state.typed_expr_to_egg(item.__egg_typed_expr__) for item in runtime_exprs]
            cmd = bindings.UserDefined(
                span(2),
                "multi-extract",
                [bindings.Lit(span(2), bindings.Int(n)), *egg_exprs, *extractor_options],
            )
            try:
                outputs = self._state.run_program(cmd)
            except BaseException as e:
                e.add_note("while extracting: " + ", ".join(map(str, runtime_exprs)))
                raise
            if len(outputs) != 1 or not isinstance(outputs[0], bindings.UserDefinedOutput):
                msg = "multi-extract returned unexpected command outputs"
                raise RuntimeError(msg)
            output = outputs[0].output.as_multi_extract()
            if output is None:
                msg = "multi-extract returned an unexpected user-defined output"
                raise RuntimeError(msg)
            termdag = output.termdag
            terms_by_root = output.terms
            if len(terms_by_root) != len(runtime_exprs):
                msg = "multi-extract returned an unexpected user-defined output"
                raise RuntimeError(msg)
            results: list[list[BASE_EXPR]] = []
            for runtime_expr, terms in zip(runtime_exprs, terms_by_root, strict=True):
                typed_exprs = self._state.exprs_from_egg(termdag, terms, runtime_expr.__egg_typed_expr__.tp)
                results.append([
                    cast("BASE_EXPR", RuntimeExpr.__from_values__(self.__egg_decls__, typed_expr))
                    for typed_expr in typed_exprs
                ])
            return results

        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr)
        extract_report = self._run_extract(runtime_expr, n, extractor)
        assert isinstance(extract_report, bindings.ExtractVariants)
        new_exprs = self._state.exprs_from_egg(
            extract_report.termdag, extract_report.terms, runtime_expr.__egg_typed_expr__.tp
        )
        return [cast("BASE_EXPR", RuntimeExpr.__from_values__(self.__egg_decls__, expr)) for expr in new_exprs]

    def keep_best(
        self,
        fn: ExprCallable,
        /,
        *fns: ExprCallable,
        extractor: ExtractionMode = "tree",
    ) -> None:
        """
        Keep the best rows of selected callables and discard unreferenced rows.

        This destructively clears all old rows, then rebuilds the selected rows
        and the constructor rows needed by their extracted representations.
        Declarations and dynamic-cost table identities remain available for
        subsequent iteration.
        """
        extractor_options = _extractor_options(extractor)
        resolved = [resolve_callable(callable_) for callable_ in (fn, *fns)]
        self._add_decls(*(decls for _, decls in resolved))
        for ref, _ in resolved:
            self._require_table_backed(ref)
        table_names = [self._state.callable_ref_to_egg(ref)[0] for ref, _ in resolved]
        args: list[bindings._Expr] = [bindings.Lit(span(2), bindings.String(table_name)) for table_name in table_names]
        self._state.run_program(bindings.UserDefined(span(2), "keep-best", [*args, *extractor_options]))

        # Compaction rebuilds the tables with fresh raw value IDs. Reject every
        # opaque value captured before it, including values inherited from a
        # parent push scope; popping restores that parent's owner generation.
        self._state.value_owner = object()
        self._state.valid_value_owners = frozenset((self._state.value_owner,))

        # keep-best clears every table, including synthetic and user let rows.
        # Do not let later lowering reuse references to those now-empty tables.
        self._state.expr_to_letref_cache.clear()
        self._state.expr_to_egg_cache = {
            expr: egg_expr
            for expr, egg_expr in self._state.expr_to_egg_cache.items()
            if not isinstance(expr, LetRefDecl)
        }

    def _run_extract(self, expr: RuntimeExpr, n: int, extractor: ExtractionMode = "tree") -> bindings._CommandOutput:
        egg_expr = self._state.typed_expr_to_egg(expr.__egg_typed_expr__)
        # If we have defined any cost tables use the custom extraction
        args: tuple[bindings._Expr, ...] = (egg_expr, bindings.Lit(span(2), bindings.Int(n)))
        args += _extractor_options(extractor)
        if self._state.cost_table_names or extractor == "greedy-dag":
            cmd: bindings._Command = bindings.UserDefined(span(2), "extract", list(args))
        else:
            cmd = bindings.Extract(span(2), *args)
        try:
            return self._state.run_program(cmd)[0]
        except BaseException as e:
            e.add_note("while extracting: " + str(expr))
            raise

    @_TRACER.start_as_current_span("push")
    def push(self) -> None:
        """
        Push the current state of the egraph, so that it can be popped later and reverted back.
        """
        copied = self._state.copy()
        self._state_stack.append(self._state)
        self._state = copied

    @_TRACER.start_as_current_span("pop")
    def pop(self) -> None:
        """
        Pop the current state of the egraph, reverting back to the previous state.
        """
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

    def _serialize(
        self,
        **kwargs: Unpack[GraphvizKwargs],
    ) -> bindings.SerializedEGraph:
        max_functions = kwargs.pop("max_functions", None)
        max_calls_per_function = kwargs.pop("max_calls_per_function", None)
        split_primitive_outputs = kwargs.pop("split_primitive_outputs", True)
        split_functions = kwargs.pop("split_functions", [])
        include_temporary_functions = kwargs.pop("include_temporary_functions", False)
        n_inline_leaves = kwargs.pop("n_inline_leaves", 0)
        serialized = call_with_current_trace(
            self._egraph.serialize,
            [],
            max_functions=max_functions,
            max_calls_per_function=max_calls_per_function,
            include_temporary_functions=include_temporary_functions,
        )
        if serialized.discarded_functions:
            msg = ", ".join(set(self._state.possible_egglog_functions(serialized.discarded_functions)))
            warn(f"Omitted: {msg}", stacklevel=3)
        if serialized.truncated_functions:
            msg = ", ".join(set(self._state.possible_egglog_functions(serialized.truncated_functions)))
            warn(f"Truncated: {msg}", stacklevel=3)
        if split_primitive_outputs or split_functions:
            additional_ops = {self._callable_to_egg(f)[1] for f in split_functions}
            serialized.split_classes(self._egraph, additional_ops)
        serialized.map_ops(self._state.op_mapping())

        for _ in range(n_inline_leaves):
            serialized.inline_leaves()

        return serialized

    def _graphviz(self, **kwargs: Unpack[GraphvizKwargs]) -> graphviz.Source:
        serialized = self._serialize(**kwargs)

        original = serialized.to_dot()
        # Add link to stylesheet to the graph, so that edges light up on hover
        # https://gist.github.com/sverweij/93e324f67310f66a8f5da5c2abe94682
        styles = """/* the lines within the edges */
      .edge:active path,
      .edge:hover path {
        stroke: fuchsia;
        stroke-width: 3;
        stroke-opacity: 1;
      }
      /* arrows are typically drawn with a polygon */
      .edge:active polygon,
      .edge:hover polygon {
        stroke: fuchsia;
        stroke-width: 3;
        fill: fuchsia;
        stroke-opacity: 1;
        fill-opacity: 1;
      }
      /* If you happen to have text and want to color that as well... */
      .edge:active text,
      .edge:hover text {
        fill: fuchsia;
      }"""
        p = pathlib.Path(tempfile.gettempdir()) / "graphviz-styles.css"
        p.write_text(styles)
        with_stylesheet = original.replace("{", f'{{stylesheet="{p!s}"', 1)
        return graphviz.Source(with_stylesheet)

    def display(self, graphviz: bool = False, **kwargs: Unpack[GraphvizKwargs]) -> None:
        """
        Displays the e-graph.

        If in IPython it will display it inline, otherwise it will write it to a file and open it.
        """
        from IPython.display import SVG, display  # noqa: PLC0415

        from .visualizer_widget import VisualizerWidget  # noqa: PLC0415

        if graphviz:
            if IN_IPYTHON:
                svg = self._graphviz(**kwargs).pipe(format="svg", quiet=True, encoding="utf-8")
                display(SVG(svg))
            else:
                self._graphviz(**kwargs).render(view=True, format="svg", quiet=True)
        else:
            serialized = self._serialize(**kwargs)
            VisualizerWidget(egraphs=[serialized.to_json()]).display_or_open()

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
        """
        Saturate the egraph, running the given schedule until the egraph is saturated.
        It serializes the egraph at each step and returns a widget to visualize the egraph.

        If an `expr` is passed, it's also extracted after each run and printed
        """
        from .visualizer_widget import VisualizerWidget  # noqa: PLC0415

        def to_json() -> str:
            if expr is not None:
                print(self.extract(expr), "\n")
            return self._serialize(**kwargs).to_json()

        if visualize:
            egraphs = [to_json()]
        i = 0
        # Always visualize, even if we encounter an error
        try:
            while i < max:
                report = self.run(schedule or 1)
                if report.can_stop:
                    break
                i += 1
                if visualize:
                    egraphs.append(to_json())
                if print_frozen:
                    print(f"After iteration {i}:")
                    print(self.freeze())
                    print("\n")
        except:
            if visualize:
                egraphs.append(to_json())
            if print_frozen:
                print(self.freeze())
            raise
        finally:
            if visualize:
                VisualizerWidget(egraphs=egraphs).display_or_open()

    @property
    def _egraph(self) -> bindings.EGraph:
        msg = "Direct native-AST access is removed; this operation is not migrated"
        raise NotImplementedError(msg)

    @property
    def __egg_decls__(self) -> Declarations:
        msg = "Legacy declaration projections are removed"
        raise NotImplementedError(msg)

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
        Look up the current value of a table-backed call, if that row exists.

        Cost lookups use their associated cost table. Eager and builtin
        primitive calls cannot be inspected with this method.

        During a custom cost-model callback, same-e-graph lookups may use
        values supplied to that callback but cannot evaluate newly derived
        arguments while extraction holds the e-graph read-only.

        Opaque values returned for user-defined sorts belong to this EGraph
        and cannot be used to query another EGraph.
        """
        runtime_expr = to_runtime_expr(expr)
        typed_expr = runtime_expr.__egg_typed_expr__
        assert isinstance(typed_expr.expr, CallDecl | GetCostDecl)
        callback_values = self._cost_callback_values
        in_cost_model_callback = callback_values is not None
        if in_cost_model_callback:
            ref = typed_expr.expr.callable
            table_is_registered = (
                ref in self._state.callable_ref_to_egg_fn
                if isinstance(typed_expr.expr, CallDecl)
                else ref in self._state.cost_table_names
            )
            if not table_is_registered:
                msg = "Tables queried by cost-model callbacks must be registered before extraction starts"
                raise ValueError(msg)
        else:
            self._add_decls(runtime_expr)
        if isinstance(typed_expr.expr, CallDecl):
            self._require_table_backed(typed_expr.expr.callable)
        egg_fn, typed_args = self._state.translate_call(typed_expr.expr)
        values_args = self._lookup_argument_values(typed_args, callback_values)
        possible_value = self._egraph.lookup_function(egg_fn, values_args)
        if possible_value is None:
            return None
        return cast(
            "BASE_EXPR",
            RuntimeExpr.__from_values__(
                self.__egg_decls__,
                TypedExprDecl(typed_expr.tp, self._state.value_to_expr(typed_expr.tp, possible_value)),
            ),
        )

    def _lookup_argument_values(
        self,
        typed_args: list[TypedExprDecl],
        callback_values: dict[TypedExprDecl, bindings.Value] | None,
    ) -> list[bindings.Value]:
        """Resolve lookup keys without mutating an e-graph borrowed by extraction."""
        if callback_values is None:
            args_to_materialize = list(typed_args)
            while args_to_materialize:
                arg = args_to_materialize.pop()
                match arg.expr:
                    case CallDecl():
                        self.register(cast("BaseExpr", RuntimeExpr.__from_values__(self.__egg_decls__, arg)))
                    case PartialCallDecl(CallDecl(args=nested_args)):
                        # The unstable-fn value itself has no table row. Only
                        # evaluating its captured calls can mutate the e-graph.
                        args_to_materialize.extend(nested_args)
            return [self._state.typed_expr_to_value(arg) for arg in typed_args]

        values = []
        for arg in typed_args:
            if arg in callback_values:
                values.append(callback_values[arg])
                continue
            if isinstance(arg.expr, ValueDecl) and arg.expr.owner in self._state.valid_value_owners:
                values.append(arg.expr.value)
                continue
            msg = (
                "Cost-model callbacks can only look up tables using values supplied to the callback; "
                "evaluating new expressions would require mutating the borrowed e-graph"
            )
            raise ValueError(msg)
        return values

    def _require_table_backed(self, ref: CallableRef) -> None:
        """Reject callable shapes that Egglog lowers without a queryable table."""
        if self._state._callable_is_table_backed(ref):
            return
        msg = (
            "This operation requires a table-backed relation, constructor, or bodyless function; "
            "eager and builtin primitives do not have tables"
        )
        raise ValueError(msg)

    def has_custom_cost(self, fn: ExprCallable) -> bool:
        """
        Checks if the any custom costs have been set for this expression callable.
        """
        resolved, _ = resolve_callable(fn)
        return resolved in self._state.cost_table_names

    def freeze(self) -> FrozenEGraph:  # noqa: C901,PLR0912
        """
        Freeze the current e-graph state for debugging.

        The returned :class:`FrozenEGraph` contains a snapshot of the current
        declarations and can be pretty-printed back into replayable high-level
        actions with ``str(...)``.

        This is useful when debugging unexpected unions, sets, subsumptions, or
        costs after a run:

        >>> from egglog import *
        >>> class Math(Expr):
        ...     def __init__(self, value: i64Like) -> None: ...
        ...
        >>> egraph = EGraph(Math(1))
        >>> str(egraph.freeze())
        'EGraph(Math(1)).freeze()'
        """
        frozen = self._egraph.freeze()
        let_bindings: dict[str, TypedExprDecl] = {}
        e_classes: dict[bindings.Value, tuple[JustTypeRef, list[CallDecl]]] = {}
        sets: dict[CallDecl, TypedExprDecl] = {}
        costs: dict[CallDecl, tuple[JustTypeRef, int]] = {}
        expr_actions: list[TypedExprDecl] = []
        subsumed: list[tuple[JustTypeRef, CallDecl]] = []

        def append_e_class_row(output: bindings.Value, tp: JustTypeRef, call: CallDecl, is_subsumed: bool) -> None:
            if output not in e_classes:
                e_classes[output] = (tp, [])
            e_class_tp, exprs = e_classes[output]
            assert e_class_tp == tp
            exprs.append(call)
            if is_subsumed:
                subsumed.append((tp, call))

        synthetic_let_names = {var.name for var in self._state.expr_to_letref_cache.values()}
        cost_target_inputs: dict[CallableRef, set[tuple[bindings.Value, ...]] | None] = {}
        for cost_callable_ref in self._state.cost_table_names:
            target_name = self._state.callable_ref_to_egg_fn[cost_callable_ref][0]
            target_fn = frozen.functions.get(target_name)
            # Table-backed targets need a live row before replay can set its
            # cost. Eager and builtin targets have no frozen table and can be
            # reconstructed directly from the cost-table arguments.
            cost_target_inputs[cost_callable_ref] = (
                None if target_fn is None else {tuple(target_row.inputs) for target_row in target_fn.rows}
            )
        for name, fn in frozen.functions.items():
            if fn.is_let_binding:
                if name in synthetic_let_names:
                    continue
                for row in fn.rows:
                    output_tp = self._state.egg_sort_to_type_ref[fn.output_sort]
                    let_bindings[name] = TypedExprDecl(output_tp, self._state.value_to_expr(output_tp, row.output))
                continue
            is_cost = name in self._state.cost_table_names.values()
            cost_callable_refs = tuple(
                ref for ref, cost_name in self._state.cost_table_names.items() if name == cost_name
            )
            raw_callable_refs = self._state.egg_fn_to_callable_refs.get(name, set())
            if is_cost and not raw_callable_refs:
                callable_ref = cost_callable_refs[0]
            else:
                (callable_ref,) = raw_callable_refs
            callable_decl = self.__egg_decls__.get_callable_decl(callable_ref)
            signature = callable_decl.signature
            assert isinstance(signature, FunctionSignature), (
                f"Cannot freeze special callable {callable_ref} with signature {signature}"
            )
            assert signature.var_arg_type is None, f"Frozen calls do not support var args: {callable_ref}"

            for row in fn.rows:
                python_inputs = row.inputs[::-1] if signature.reverse_args else row.inputs
                arg_exprs = tuple(
                    TypedExprDecl(tp, self._state.value_to_expr(tp, value))
                    for arg_type, value in zip(signature.arg_types, python_inputs, strict=True)
                    for tp in (arg_type.to_just(),)
                )
                if is_cost:
                    cost_tp = self._state.egg_sort_to_type_ref[fn.output_sort]
                    cost_expr = TypedExprDecl(cost_tp, self._state.value_to_expr(cost_tp, row.output))
                    for cost_callable_ref in cost_callable_refs:
                        target_inputs = cost_target_inputs[cost_callable_ref]
                        if target_inputs is not None and tuple(row.inputs) not in target_inputs:
                            continue
                        cost_signature = self.__egg_decls__.get_callable_decl(cost_callable_ref).signature
                        assert isinstance(cost_signature, FunctionSignature)
                        cost_inputs = row.inputs[::-1] if cost_signature.reverse_args else row.inputs
                        cost_arg_exprs = tuple(
                            TypedExprDecl(tp, self._state.value_to_expr(tp, value))
                            for arg_type, value in zip(cost_signature.arg_types, cost_inputs, strict=True)
                            for tp in (arg_type.to_just(),)
                        )
                        cost_call = CallDecl(cost_callable_ref, cost_arg_exprs)
                        match cost_expr.expr:
                            case LitDecl(int(value)):
                                costs[cost_call] = (cost_signature.semantic_return_type.to_just(), value)
                            case _:
                                raise TypeError(f"Expected integer cost for {cost_callable_ref}, got {cost_expr.expr}")
                    # A user-declared bodyless function may intentionally own the
                    # canonical table that also serves as the dynamic-cost table.
                    # Preserve its public rows as ordinary sets as well as costs.
                    if not raw_callable_refs:
                        continue

                call = CallDecl(callable_ref, arg_exprs)
                output_tp = signature.semantic_return_type.to_just()
                match callable_decl:
                    case ConstructorDecl():
                        append_e_class_row(row.output, output_tp, call, row.subsumed)
                    case FunctionDecl():
                        set_tp = self._state.egg_sort_to_type_ref[fn.output_sort]
                        sets[call] = TypedExprDecl(set_tp, self._state.value_to_expr(set_tp, row.output))
                    case ConstantDecl():
                        if is_callable_decl_constructor(self.__egg_decls__, callable_decl):
                            append_e_class_row(row.output, output_tp, call, row.subsumed)
                        else:
                            set_tp = self._state.egg_sort_to_type_ref[fn.output_sort]
                            sets[call] = TypedExprDecl(set_tp, self._state.value_to_expr(set_tp, row.output))
                    case RelationDecl():
                        if row.subsumed:
                            raise TypeError(f"Cannot freeze subsumed relation row for {callable_ref}")
                        expr_actions.append(TypedExprDecl(output_tp, call))
                    case _:
                        assert_never(callable_decl)

        return FrozenEGraph(
            self.__egg_decls__.copy(),
            EGraphDecl(
                let_bindings=let_bindings,
                e_classes={value: (tp, tuple(exprs)) for value, (tp, exprs) in e_classes.items()},
                sets=sets,
                expr_actions=tuple(expr_actions),
                costs=costs,
                subsumed=tuple(subsumed),
            ),
        )

    def _values_to_expr_and_callback_values(
        self, args: list[bindings.Value], name: str
    ) -> tuple[RuntimeExpr, dict[TypedExprDecl, bindings.Value]] | None:
        """Reconstruct a callback call and map its Python-order arguments to raw backend values."""
        if name not in self._state.egg_fn_to_callable_refs:
            return None
        (callable_ref,) = self._state.egg_fn_to_callable_refs[name]
        signature = self.__egg_decls__.get_callable_decl(callable_ref).signature
        assert isinstance(signature, FunctionSignature)
        python_args = args[::-1] if signature.reverse_args else args
        arg_exprs = tuple(
            TypedExprDecl(tp, self._state.value_to_expr(tp, arg))
            for arg_type, arg in zip(signature.arg_types, python_args, strict=True)
            for tp in (arg_type.to_just(),)
        )
        # Interned calls retain structurally equal argument declarations from
        # earlier uses, so callback values cannot be keyed by Python identity.
        callback_values = dict(zip(arg_exprs, python_args, strict=True))
        stack = list(arg_exprs)
        while stack:
            arg = stack.pop()
            match arg.expr:
                case ValueDecl(value, owner):
                    # Reconstructed nested e-class values also belong to this
                    # callback borrow, but structurally equal values from
                    # another e-graph must not be accepted.
                    if owner in self._state.valid_value_owners:
                        callback_values[arg] = value
                case CallDecl(args=nested_args) | PartialCallDecl(CallDecl(args=nested_args)):
                    stack.extend(nested_args)
        res_type = signature.semantic_return_type.to_just()
        return (
            RuntimeExpr.__from_values__(
                self.__egg_decls__,
                TypedExprDecl(res_type, CallDecl(callable_ref, arg_exprs)),
            ),
            callback_values,
        )


@dataclass(frozen=True)
class FrozenEGraph:
    __egg_decls__: Declarations
    decl: EGraphDecl

    def __str__(self) -> str:
        return pretty_decl(self.__egg_decls__, self.decl)

    def __repr__(self) -> str:
        return str(self)


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

    def __init__(self, reference: Ref, bindings: dict[str, Ruleset | UnstableCombinedRuleset]) -> None:
        if reference.role != "commands":
            msg = "Schedules require a canonical command reference"
            raise TypeError(msg)
        self._command = reference
        # Resolution/lifetime ownership only; all control topology is in Command.
        self._bindings = bindings.copy()

    def _resolve(self, state: EGraphState | None) -> Ref:  # noqa: C901, PLR0912
        """Materialize live sources, then relocate this one canonical command."""
        builder = Builder()
        command: pb.Command = _copy_record(
            self._command.read(),
            relocate=lambda role, index: builder.import_ref(self._command.owner.ref(role, index)),
        )
        pending = [command]
        leaves: list[pb.Run] = []
        while pending:
            current = pending.pop()
            if current.kind is None:
                msg = "Schedule command has no kind"
                raise ValueError(msg)
            if current.kind.field == "run":
                if current.kind.value.scheduler:
                    msg = "Custom schedulers are not migrated"
                    raise NotImplementedError(msg)
                leaves.append(current.kind.value)
            elif current.kind.field in {"repeat", "saturate"}:
                control = current.kind.value
                if control.until or (isinstance(control, pb.Repeat) and control.times not in (0, 1)):
                    msg = "Only ordinary no-until schedules and exact Repeat0/1 are migrated"
                    raise NotImplementedError(msg)
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
        index = builder.add("commands", command)
        return builder.publish().ref("commands", index)

    def __str__(self) -> str:
        return pretty_ref(self._resolve(None))

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
        return Schedule(builder.publish().ref("commands", index), source._bindings)

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
        return Schedule(builder.publish().ref("commands", index), source._bindings)

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
    def __egg_decls__(self) -> Declarations:
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
        return pretty_ref(self._snapshot({}))

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
    def __egg_decls__(self) -> Declarations:
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
        return pretty_ref(self._snapshot({}))

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
        return pretty_ref(self.__egg_ref__)

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
        return pretty_ref(self.__egg_ref__)

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
        return pretty_ref(self.__egg_ref__)

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
    msg = "Panic action authoring is not migrated"
    raise NotImplementedError(msg)


def set_cost(expr: BaseExpr, cost: i64Like) -> Action:
    """Set the cost of the given expression."""
    msg = "Dynamic cost action authoring is not migrated"
    raise NotImplementedError(msg)


def let(name: str, expr: BaseExpr) -> Action:
    """Create a let binding."""
    msg = "Retained local-let actions are not migrated"
    raise NotImplementedError(msg)


def expr_action(expr: BaseExpr) -> Action:
    runtime_expr = to_runtime_expr(expr)
    builder = Builder()
    node = builder.import_ref(runtime_expr.__egg_ref__)
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
    msg = "Delete action authoring is not migrated"
    raise NotImplementedError(msg)


def subsume(expr: Expr) -> Action:
    """Subsume an expression so it cannot be matched against or extracted"""
    msg = "Subsume action authoring is not migrated"
    raise NotImplementedError(msg)


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
    """Create a new variable with the given name and type."""
    return cast("T", _var(name, bound, egg_name=egg_name))


def _var(name: str, bound: object, egg_name: str | None) -> RuntimeExpr:
    """Create a new variable with the given name and type."""
    if egg_name is not None:
        msg = "Variable backend aliases are not migrated"
        raise NotImplementedError(msg)
    if not isinstance(cast("object", name), str):
        msg = "Variable names must be strings"
        raise TypeError(msg)
    if not name:
        msg = "Empty variable names are not migrated"
        raise NotImplementedError(msg)
    sort = resolve_type_annotation(bound)
    if not isinstance(sort, Ref) or sort.role != "sorts":
        msg = "Variables require a concrete sort"
        raise TypeError(msg)
    builder = Builder()
    index = builder.add(
        "nodes",
        pb.Node(sort_id=builder.import_ref(sort), kind=Oneof[Literal["var"], str]("var", name)),
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
        return f"set_({pretty_ref(lhs.__egg_ref__)})"


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
        return f"union({pretty_ref(lhs.__egg_ref__)})"


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
    ruleset once. Additional facts become ``:until`` stop conditions. A custom
    ``scheduler`` currently supports at most one non-equality stop fact.
    """
    if until or scheduler is not None:
        msg = "Until and custom schedulers require a separate shared execution migration"
        raise NotImplementedError(msg)
    if ruleset is not None and not isinstance(cast("object", ruleset), Ruleset | UnstableCombinedRuleset):
        msg = "Run target must be a ruleset"
        raise TypeError(msg)
    name = uuid4().hex if ruleset is not None else ""
    builder = Builder()
    index = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["run"], pb.Run](
                "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", name)))
            )
        ),
    )
    return Schedule(builder.publish().ref("commands", index), {} if ruleset is None else {name: ruleset})


def back_off(match_limit: None | int = None, ban_length: None | int = None) -> BackOff:
    """
    Create a backoff scheduler configuration.

    ```python
    schedule = run(analysis_ruleset).saturate() + run(ruleset, scheduler=back_off(match_limit=1000, ban_length=5)) * 10
    ```
    This will run the `analysis_ruleset` until saturation, then run `ruleset` 10 times, using a backoff scheduler.

    The backend defaults are ``match_limit=1000`` and ``ban_length=5``. Each time
    a rule is banned, both effective values double for that rule.
    """
    return BackOff(BackOffDecl(id=uuid4(), match_limit=match_limit, ban_length=ban_length))


@dataclass(frozen=True)
class BackOff:
    scheduler: BackOffDecl

    def persistent(self) -> BackOff:
        """
        Reuse this scheduler across repeated runs on the same egraph.
        """
        return BackOff(replace(self.scheduler, id=uuid4(), persistent=True))

    def scope(self, schedule: Schedule) -> Schedule:
        """
        Defines the scheduler to be created directly before the inner schedule, instead of the default which is at the
        most outer scope.
        """
        msg = "Custom scheduler scopes are not migrated"
        raise NotImplementedError(msg)

    def __str__(self) -> str:
        return pretty_decl(Declarations(), self.scheduler)

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
    )


def _action_likes(action_likes: Iterable[ActionLike]) -> tuple[Action, ...]:
    return tuple(map(_action_like, action_likes))


def _action_like(action_like: ActionLike) -> Action:
    if isinstance(action_like, Action):
        return action_like
    if isinstance(action_like, Fact):
        msg = "Fact-to-action conversion is not migrated"
        raise NotImplementedError(msg)
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


@contextlib.contextmanager
def _cost_model_callback_values(
    egraph: EGraph,
    values: dict[TypedExprDecl, bindings.Value],
) -> Generator[None, None, None]:
    """Make raw callback values available to read-only table lookups without evaluating expressions."""
    # Values belong to the physical e-graph borrow, not Python's ambient context.
    # Restore any outer callback scope if callback setup is nested.
    previous = egraph._cost_callback_values
    egraph._cost_callback_values = values
    try:
        yield
    finally:
        egraph._cost_callback_values = previous


def get_cost(expr: BaseExpr) -> i64:
    """
    Return a lookup of the cost of an expression. If not set, won't match.
    """
    runtime_expr = to_runtime_expr(expr)
    expr_decl = runtime_expr.__egg_typed_expr__.expr
    if not isinstance(expr_decl, CallDecl):
        msg = "Can only get cost of function calls, not literals or variables"
        raise TypeError(msg)
    return cast(
        "i64",
        RuntimeExpr.__from_values__(
            runtime_expr.__egg_decls__,
            TypedExprDecl(JustTypeRef(Ident.builtin("i64")), GetCostDecl(expr_decl.callable, expr_decl.args)),
        ),
    )


class Comparable(Protocol):
    def __lt__(self, other: Self) -> bool: ...
    def __le__(self, other: Self) -> bool: ...
    def __gt__(self, other: Self) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...


COST = TypeVar("COST", bound=Comparable)
ExtractionMode: TypeAlias = Literal["tree", "greedy-dag"]


def _extractor_options(extractor: str) -> tuple[bindings._Expr, ...]:
    """Validate an extraction mode and lower its optional command selector."""
    match extractor:
        case "tree":
            return ()
        case "greedy-dag":
            return (bindings.Var(span(2), ":extractor"), bindings.Var(span(2), "greedy-dag"))
        case _:
            msg = f"Unknown extractor {extractor!r}; expected 'tree' or 'greedy-dag'"
            raise ValueError(msg)


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
    """
    A default cost model for an e-graph, which looks up costs set on function calls, or uses 1 as the default cost.
    """
    from .builtins import Container  # noqa: PLC0415
    from .deconstruct import get_callable_fn  # noqa: PLC0415

    # 1. First prefer if the expr has a custom cost set on it
    if (
        (callable_fn := get_callable_fn(expr)) is not None
        and egraph.has_custom_cost(callable_fn)
        and (i := egraph.lookup_function_value(get_cost(expr))) is not None
    ):
        self_cost = int(i)
    # 2. Else, check if this is a callable and it has a cost set on its declaration
    elif callable_fn is not None and (callable_cost := get_callable_cost(callable_fn)) is not None:
        self_cost = callable_cost
    # 3. Else, if this is a container, it has no cost, otherwise it has a cost of 1
    else:
        # By default, all nodes have a cost of 1 except for containers which have a cost of 0
        self_cost = 0 if isinstance(expr, Container) else 1
    # Sum up the costs of the children and our own cost
    return sum(children_costs, start=self_cost)


def get_callable_cost(fn: ExprCallable) -> int | None:
    """
    Returns the cost of a callable, if it has one set. Otherwise returns None.
    """
    callable_ref, decls = resolve_callable(fn)
    callable_decl = decls.get_callable_decl(callable_ref)
    return callable_decl.cost if isinstance(callable_decl, ConstructorDecl) else 1


@dataclass
class _CostModel(Generic[COST]):
    """
    Implements the methods compatible with the bindings for the cost model.
    """

    model: CostModel[COST]
    egraph: EGraph
    enode_cost_results: dict[tuple[str, tuple[bindings.Value, ...]], int] = field(default_factory=dict)
    enode_cost_expressions: list[RuntimeExpr] = field(default_factory=list)
    enode_cost_argument_values: list[dict[TypedExprDecl, bindings.Value]] = field(default_factory=list)
    base_value_cost_results: dict[tuple[str, bindings.Value], COST] = field(default_factory=dict)

    def call_model(self, expr: RuntimeExpr, children_costs: list[COST]) -> COST:
        return self.model(self.egraph, cast("BaseExpr", expr), children_costs)
        # if __debug__:
        #     for c in children_costs:
        #         if res <= c:
        #             msg = f"Cost model {self.model} produced a cost {res} less than or equal to a child cost {c} for {expr}"
        #             raise ValueError(msg)

    def fold(self, _fn: str, index: int, children_costs: list[COST]) -> COST:
        expr = self.enode_cost_expressions[index]
        with _cost_model_callback_values(self.egraph, self.enode_cost_argument_values[index]):
            return self.call_model(expr, children_costs)

    # enode cost is only ever called right before fold, for the head_cost
    def enode_cost(self, name: str, args: list[bindings.Value]) -> int:
        try:
            return self.enode_cost_results[(name, tuple(args))]
        except KeyError:
            pass
        callback = self.egraph._values_to_expr_and_callback_values(args, name)
        if callback is None:
            msg = f"Cannot compute custom cost for unknown egg function {name!r}"
            raise ValueError(msg)
        res, argument_values = callback
        index = len(self.enode_cost_expressions)
        self.enode_cost_expressions.append(res)
        self.enode_cost_argument_values.append(argument_values)
        self.enode_cost_results[(name, tuple(args))] = index
        return index

    def base_value_cost(self, tp: str, value: bindings.Value) -> COST:
        try:
            return self.base_value_cost_results[(tp, value)]
        except KeyError:
            pass
        type_ref = self.egraph._state.egg_sort_to_type_ref[tp]
        expr = RuntimeExpr.__from_values__(
            self.egraph.__egg_decls__,
            TypedExprDecl(type_ref, self.egraph._state.value_to_expr(type_ref, value)),
        )
        with _cost_model_callback_values(self.egraph, {expr.__egg_typed_expr__: value}):
            res = self.call_model(expr, [])
        self.base_value_cost_results[(tp, value)] = res
        return res

    def container_cost(self, tp: str, value: bindings.Value, element_costs: list[COST]) -> COST:
        type_ref = self.egraph._state.egg_sort_to_type_ref[tp]
        expr = RuntimeExpr.__from_values__(
            self.egraph.__egg_decls__,
            TypedExprDecl(type_ref, self.egraph._state.value_to_expr(type_ref, value)),
        )
        with _cost_model_callback_values(self.egraph, {expr.__egg_typed_expr__: value}):
            return self.call_model(expr, element_costs)

    def to_bindings_cost_model(self) -> bindings.CostModel[COST, int]:
        return bindings.CostModel(self.fold, self.enode_cost, self.container_cost, self.base_value_cost)


@dataclass
class _DagCostModel(Generic[DAG_COST]):
    """Adapts a high-level marginal model to the raw binding callbacks."""

    model: DagCostModel[DAG_COST]
    egraph: EGraph
    enode_cost_results: dict[tuple[str, tuple[bindings.Value, ...]], DAG_COST] = field(default_factory=dict)
    value_cost_results: dict[tuple[str, bindings.Value], DAG_COST] = field(default_factory=dict)

    def enode_cost(self, name: str, args: list[bindings.Value]) -> DAG_COST:
        key = (name, tuple(args))
        try:
            return self.enode_cost_results[key]
        except KeyError:
            pass
        callback = self.egraph._values_to_expr_and_callback_values(args, name)
        if callback is None:
            msg = f"Cannot compute custom cost for unknown egg function {name!r}"
            raise ValueError(msg)
        expr, argument_values = callback
        with _cost_model_callback_values(self.egraph, argument_values):
            result = self.model.marginal_cost(self.egraph, cast("BaseExpr", expr))
        self.enode_cost_results[key] = result
        return result

    def value_cost(self, tp: str, value: bindings.Value) -> DAG_COST:
        key = (tp, value)
        try:
            return self.value_cost_results[key]
        except KeyError:
            pass
        type_ref = self.egraph._state.egg_sort_to_type_ref[tp]
        expr = RuntimeExpr.__from_values__(
            self.egraph.__egg_decls__,
            TypedExprDecl(type_ref, self.egraph._state.value_to_expr(type_ref, value)),
        )
        with _cost_model_callback_values(self.egraph, {expr.__egg_typed_expr__: value}):
            result = self.model.marginal_cost(self.egraph, cast("BaseExpr", expr))
        self.value_cost_results[key] = result
        return result

    def to_bindings_cost_model(self) -> bindings.DagCostModel[DAG_COST]:
        return bindings.DagCostModel(
            self.model.identity,
            self.enode_cost,
            self.value_cost,
            self.value_cost,
        )
