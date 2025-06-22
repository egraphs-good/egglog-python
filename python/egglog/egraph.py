from __future__ import annotations

import contextlib
import inspect
import pathlib
import tempfile
from collections.abc import Callable, Generator, Iterable
from contextvars import ContextVar
from dataclasses import InitVar, dataclass, field
from functools import partial
from inspect import Parameter, currentframe, signature
from types import FrameType, FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

import graphviz
from typing_extensions import Never, ParamSpec, Self, Unpack, assert_never

from . import bindings
from .conversion import *
from .declarations import *
from .egraph_state import *
from .ipython_magic import IN_IPYTHON
from .pretty import pretty_decl
from .runtime import *
from .thunk import *
from .version_compat import *

if TYPE_CHECKING:
    from .builtins import String, Unit


__all__ = [
    "Action",
    "BaseExpr",
    "BuiltinExpr",
    "Command",
    "Command",
    "EGraph",
    "Expr",
    "Fact",
    "Fact",
    "GraphvizKwargs",
    "RewriteOrRule",
    "Ruleset",
    "Schedule",
    "_BirewriteBuilder",
    "_EqBuilder",
    "_NeBuilder",
    "_RewriteBuilder",
    "_SetBuilder",
    "_UnionBuilder",
    "birewrite",
    "check",
    "check_eq",
    "constant",
    "delete",
    "eq",
    "expr_action",
    "expr_fact",
    "expr_parts",
    "function",
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
    "simplify",
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
# Attributes which are sometimes added to classes by the interpreter or the dataclass decorator, or by ipython.
# We ignore these when inspecting the class.

IGNORED_ATTRIBUTES = {
    "__module__",
    "__doc__",
    "__dict__",
    "__weakref__",
    "__orig_bases__",
    "__annotations__",
    "__hash__",
    "__qualname__",
    "__firstlineno__",
    "__static_attributes__",
    # Ignore all reflected binary method
    *REFLECTED_BINARY_METHODS.keys(),
}


# special methods that return none and mutate self
ALWAYS_MUTATES_SELF = {"__setitem__", "__delitem__"}
# special methods which must return real python values instead of lazy expressions
ALWAYS_PRESERVED = {
    "__repr__",
    "__str__",
    "__bytes__",
    "__format__",
    "__hash__",
    "__bool__",
    "__len__",
    "__length_hint__",
    "__iter__",
    "__reversed__",
    "__contains__",
    "__index__",
    "__bufer__",
}


def simplify(x: EXPR, schedule: Schedule | None = None) -> EXPR:
    """
    Simplify an expression by running the schedule.
    """
    if schedule:
        return EGraph().simplify(x, schedule)
    return EGraph().extract(x)


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
        raise add_note(
            f"Failed:\n{eq(x).to(y)}\n\nExtracted:\n {eq(egraph.extract(x)).to(egraph.extract(y))})", err
        ) from None
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


CALLABLE = TypeVar("CALLABLE", bound=Callable)
CONSTRUCTOR_CALLABLE = TypeVar("CONSTRUCTOR_CALLABLE", bound=Callable[..., "Expr | None"])

EXPR_NONE = TypeVar("EXPR_NONE", bound="Expr | None")
BASE_EXPR_NONE = TypeVar("BASE_EXPR_NONE", bound="BaseExpr | None")


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
    builtin: bool = ...,
    mutates_first_arg: bool = ...,
) -> Callable[[Callable[P, BASE_EXPR]], Callable[P, BASE_EXPR]]: ...


# constructor
@overload
def function(
    *,
    egg_fn: str | None = ...,
    cost: int | None = ...,
    mutates_first_arg: bool = ...,
    unextractable: bool = ...,
    ruleset: Ruleset | None = ...,
    use_body_as_name: bool = ...,
    subsume: bool = ...,
) -> Callable[[CONSTRUCTOR_CALLABLE], CONSTRUCTOR_CALLABLE]: ...


def function(*args, **kwargs) -> Any:
    """
    Decorate a function typing stub to create an egglog function for it.

    If a body is included, it will be added to the `ruleset` passed in as a default rewrite.

    This will default to creating a "constructor" in egglog, unless a merge function is passed in or the return
    type is a primtive, then it will be a "function".
    """
    fn_locals = currentframe().f_back.f_locals  # type: ignore[union-attr]

    # If we have any positional args, then we are calling it directly on a function
    if args:
        assert len(args) == 1
        return _FunctionConstructor(fn_locals)(args[0])
    # otherwise, we are passing some keyword args, so save those, and then return a partial
    return _FunctionConstructor(fn_locals, **kwargs)


class _ExprMetaclass(type):
    """
    Metaclass of Expr.

    Used to override isistance checks, so that runtime expressions are instances of Expr at runtime.
    """

    def __new__(  # type: ignore[misc]
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
        builtin = BuiltinExpr in bases
        # TODO: Raise error on subclassing or multiple inheritence

        frame = currentframe()
        assert frame
        prev_frame = frame.f_back
        assert prev_frame

        # Pass in an instance of the class so that when we are generating the decls
        # we can update them eagerly so that we can access the methods in the class body
        runtime_cls = RuntimeClass(None, TypeRefWithVars(name))  # type: ignore[arg-type]

        # Store frame so that we can get live access to updated locals/globals
        # Otherwise, f_locals returns a copy
        # https://peps.python.org/pep-0667/
        runtime_cls.__egg_decls_thunk__ = Thunk.fn(
            _generate_class_decls,
            namespace,
            prev_frame,
            builtin,
            egg_sort,
            name,
            ruleset,
            runtime_cls,
        )
        return runtime_cls

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, RuntimeExpr)


class BaseExpr(metaclass=_ExprMetaclass):
    """
    Either a builtin or a user defined expression type.
    """

    def __ne__(self, other: Self) -> Unit: ...  # type: ignore[override, empty-body]

    def __eq__(self, other: Self) -> Fact: ...  # type: ignore[override, empty-body]


class BuiltinExpr(BaseExpr, metaclass=_ExprMetaclass):
    """
    A builtin expr type, not an eqsort.
    """


class Expr(BaseExpr, metaclass=_ExprMetaclass):
    """
    Subclass this to define a custom expression type.
    """


def _generate_class_decls(  # noqa: C901,PLR0912
    namespace: dict[str, Any],
    frame: FrameType,
    builtin: bool,
    egg_sort: str | None,
    cls_name: str,
    ruleset: Ruleset | None,
    runtime_cls: RuntimeClass,
) -> Declarations:
    """
    Lazy constructor for class declerations to support classes with methods whose types are not yet defined.
    """
    parameters: list[TypeVar] = (
        # Get the generic params from the orig bases generic class
        namespace["__orig_bases__"][1].__parameters__ if "__orig_bases__" in namespace else []
    )
    type_vars = tuple(ClassTypeVarRef.from_type_var(p) for p in parameters)
    del parameters
    cls_decl = ClassDecl(egg_sort, type_vars, builtin)
    decls = Declarations(_classes={cls_name: cls_decl})
    # Update class think eagerly when resolving so that lookups work in methods
    runtime_cls.__egg_decls_thunk__ = Thunk.value(decls)

    ##
    # Register class variables
    ##
    # Create a dummy type to pass to get_type_hints to resolve the annotations we have
    _Dummytype = type("_DummyType", (), {"__annotations__": namespace.get("__annotations__", {})})
    for k, v in get_type_hints(_Dummytype, globalns=frame.f_globals, localns=frame.f_locals).items():
        if getattr(v, "__origin__", None) == ClassVar:
            (inner_tp,) = v.__args__
            type_ref = resolve_type_annotation_mutate(decls, inner_tp)
            cls_decl.class_variables[k] = ConstantDecl(type_ref.to_just())
            _add_default_rewrite(
                decls, ClassVariableRef(cls_name, k), type_ref, namespace.pop(k, None), ruleset, subsume=False
            )
        else:
            msg = f"On class {cls_name}, for attribute '{k}', expected a ClassVar, but got {v}"
            raise NotImplementedError(msg)

    ##
    # Register methods, classmethods, preserved methods, and properties
    ##
    # Get all the methods from the class
    filtered_namespace: list[tuple[str, Any]] = [
        (k, v) for k, v in namespace.items() if k not in IGNORED_ATTRIBUTES or isinstance(v, _WrappedMethod)
    ]

    # all methods we should try adding default functions for
    add_default_funcs: list[Callable[[], None]] = []
    # Then register each of its methods
    for method_name, method in filtered_namespace:
        is_init = method_name == "__init__"
        # Don't register the init methods for literals, since those don't use the type checking mechanisms
        if is_init and cls_name in LIT_CLASS_NAMES:
            continue
        match method:
            case _WrappedMethod(egg_fn, cost, merge, fn, preserve, mutates, unextractable, subsume, reverse_args):
                pass
            case _:
                egg_fn, cost, merge = None, None, None
                fn = method
                unextractable, preserve, subsume = False, False, False
                mutates = method_name in ALWAYS_MUTATES_SELF
                reverse_args = False
        if preserve:
            cls_decl.preserved_methods[method_name] = fn
            continue
        locals = frame.f_locals
        ref: ClassMethodRef | MethodRef | PropertyRef | InitRef
        match fn:
            case classmethod():
                ref = ClassMethodRef(cls_name, method_name)
                fn = fn.__func__
            case property():
                ref = PropertyRef(cls_name, method_name)
                fn = fn.fget
            case _:
                ref = InitRef(cls_name) if is_init else MethodRef(cls_name, method_name)
        if isinstance(fn, _WrappedMethod):
            msg = f"{cls_name}.{method_name} Add the @method(...) decorator above @classmethod or @property"

            raise ValueError(msg)  # noqa: TRY004
        special_function_name: SpecialFunctions | None = (
            "fn-partial" if egg_fn == "unstable-fn" else "fn-app" if egg_fn == "unstable-app" else None
        )
        if special_function_name:
            decl = FunctionDecl(special_function_name, builtin=True, egg_name=egg_fn)
            decls.set_function_decl(ref, decl)
            continue
        try:
            _, add_rewrite = _fn_decl(
                decls,
                egg_fn,
                ref,
                fn,
                locals,
                cost,
                merge,
                mutates,
                builtin,
                ruleset=ruleset,
                unextractable=unextractable,
                subsume=subsume,
                reverse_args=reverse_args,
            )
        except Exception as e:
            raise add_note(f"Error processing {cls_name}.{method_name}", e) from None

        if not builtin and not isinstance(ref, InitRef) and not mutates:
            add_default_funcs.append(add_rewrite)

    # Add all rewrite methods at the end so that all methods are registered first and can be accessed
    # in the bodies
    for add_rewrite in add_default_funcs:
        add_rewrite()
    return decls


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
    use_body_as_name: bool = False
    subsume: bool = False

    def __call__(self, fn: Callable) -> RuntimeFunction:
        return RuntimeFunction(*split_thunk(Thunk.fn(self.create_decls, fn)))

    def create_decls(self, fn: Callable) -> tuple[Declarations, CallableRef]:
        decls = Declarations()
        ref = None if self.use_body_as_name else FunctionRef(fn.__name__)
        ref, add_rewrite = _fn_decl(
            decls,
            self.egg_fn,
            ref,
            fn,
            self.hint_locals,
            self.cost,
            self.merge,
            self.mutates_first_arg,
            self.builtin,
            ruleset=self.ruleset,
            subsume=self.subsume,
            unextractable=self.unextractable,
        )
        add_rewrite()
        return decls, ref


def _fn_decl(
    decls: Declarations,
    egg_name: str | None,
    # If ref is Callable, then generate the ref from the function name
    ref: FunctionRef | MethodRef | PropertyRef | ClassMethodRef | InitRef | None,
    fn: object,
    # Pass in the locals, retrieved from the frame when wrapping,
    # so that we support classes and function defined inside of other functions (which won't show up in the globals)
    hint_locals: dict[str, Any],
    cost: int | None,
    merge: Callable[[object, object], object] | None,
    mutates_first_arg: bool,
    is_builtin: bool,
    subsume: bool,
    ruleset: Ruleset | None = None,
    unextractable: bool = False,
    reverse_args: bool = False,
) -> tuple[CallableRef, Callable[[], None]]:
    """
    Sets the function decl for the function object and returns the ref as well as a thunk that sets the default callable.
    """
    if isinstance(fn, RuntimeFunction):
        msg = "Inside of classes, wrap methods with the `method` decorator, not `function`"
        raise ValueError(msg)  # noqa: TRY004
    if not isinstance(fn, FunctionType):
        raise NotImplementedError(f"Can only generate function decls for functions not {fn}  {type(fn)}")

    # Instead of passing both globals and locals, just pass the globals. Otherwise, for some reason forward references
    # won't be resolved correctly
    # We need this to be false so it returns "__forward_value__" https://github.com/python/cpython/blob/440ed18e08887b958ad50db1b823e692a747b671/Lib/typing.py#L919
    # https://github.com/egraphs-good/egglog-python/issues/210
    hint_globals = {**fn.__globals__, **hint_locals}
    hints = get_type_hints(fn, hint_globals)

    params = list(signature(fn).parameters.values())

    # If this is an init function, or a classmethod, the first arg is not used
    if isinstance(ref, ClassMethodRef | InitRef):
        params = params[1:]

    if _last_param_variable(params):
        *params, var_arg_param = params
        # For now, we don't use the variable arg name
        var_arg_type = resolve_type_annotation_mutate(decls, hints[var_arg_param.name])
    else:
        var_arg_type = None
    arg_types = tuple(
        decls.get_paramaterized_class(ref.class_name)
        if i == 0 and isinstance(ref, MethodRef | PropertyRef)
        else resolve_type_annotation_mutate(decls, hints[t.name])
        for i, t in enumerate(params)
    )

    # Resolve all default values as arg types
    arg_defaults = [
        resolve_literal(t, p.default, Thunk.value(decls)) if p.default is not Parameter.empty else None
        for (t, p) in zip(arg_types, params, strict=True)
    ]

    decls.update(*arg_defaults)

    return_type = (
        decls.get_paramaterized_class(ref.class_name)
        if isinstance(ref, InitRef)
        else arg_types[0]
        if mutates_first_arg
        else resolve_type_annotation_mutate(decls, hints["return"])
    )

    arg_names = tuple(t.name for t in params)

    merged = (
        None
        if merge is None
        else resolve_literal(
            return_type,
            merge(
                RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), VarDecl("old", False))),
                RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), VarDecl("new", False))),
            ),
            lambda: decls,
        )
    )
    decls |= merged

    # defer this in generator so it doesn't resolve for builtins eagerly
    args = (TypedExprDecl(tp.to_just(), VarDecl(name, False)) for name, tp in zip(arg_names, arg_types, strict=True))
    res_ref: FunctionRef | MethodRef | ClassMethodRef | PropertyRef | InitRef | UnnamedFunctionRef
    res_thunk: Callable[[], object]
    # If we were not passed in a ref, this is an unnamed funciton, so eagerly compute the value and use that to refer to it
    if not ref:
        tuple_args = tuple(args)
        res = _create_default_value(decls, ref, fn, tuple_args, ruleset)
        assert isinstance(res, RuntimeExpr)
        res_ref = UnnamedFunctionRef(tuple_args, res.__egg_typed_expr__)
        decls._unnamed_functions.add(res_ref)
        res_thunk = Thunk.value(res)

    else:
        return_type_is_eqsort = (
            not decls._classes[return_type.name].builtin if isinstance(return_type, TypeRefWithVars) else False
        )
        is_constructor = not is_builtin and return_type_is_eqsort and merged is None
        signature_ = FunctionSignature(
            return_type=None if mutates_first_arg else return_type,
            var_arg_type=var_arg_type,
            arg_types=arg_types,
            arg_names=arg_names,
            arg_defaults=tuple(a.__egg_typed_expr__.expr if a is not None else None for a in arg_defaults),
            reverse_args=reverse_args,
        )
        decl: ConstructorDecl | FunctionDecl
        if is_constructor:
            decl = ConstructorDecl(signature_, egg_name, cost, unextractable)
        else:
            if cost is not None:
                msg = "Cost can only be set for constructors"
                raise ValueError(msg)
            if unextractable:
                msg = "Unextractable can only be set for constructors"
                raise ValueError(msg)
            decl = FunctionDecl(
                signature=signature_,
                egg_name=egg_name,
                merge=merged.__egg_typed_expr__.expr if merged is not None else None,
                builtin=is_builtin,
            )
        res_ref = ref
        decls.set_function_decl(ref, decl)
        res_thunk = Thunk.fn(_create_default_value, decls, ref, fn, args, ruleset, context=f"creating {ref}")
    return res_ref, Thunk.fn(_add_default_rewrite_function, decls, res_ref, return_type, ruleset, res_thunk, subsume)


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
    decls_thunk = Thunk.fn(_relation_decls, name, tps, egg_fn)
    return cast("Callable[..., Unit]", RuntimeFunction(decls_thunk, Thunk.value(FunctionRef(name))))


def _relation_decls(name: str, tps: tuple[type, ...], egg_fn: str | None) -> Declarations:
    from .builtins import Unit

    decls = Declarations()
    decls |= cast("RuntimeClass", Unit)
    arg_types = tuple(resolve_type_annotation_mutate(decls, tp).to_just() for tp in tps)
    decls._functions[name] = RelationDecl(arg_types, tuple(None for _ in tps), egg_fn)
    return decls


def constant(
    name: str,
    tp: type[BASE_EXPR],
    default_replacement: BASE_EXPR | None = None,
    /,
    *,
    egg_name: str | None = None,
    ruleset: Ruleset | None = None,
) -> BASE_EXPR:
    """
    A "constant" is implemented as the instantiation of a value that takes no args.
    This creates a function with `name` and return type `tp` and returns a value of it being called.
    """
    return cast(
        "BASE_EXPR",
        RuntimeExpr(*split_thunk(Thunk.fn(_constant_thunk, name, tp, egg_name, default_replacement, ruleset))),
    )


def _constant_thunk(
    name: str, tp: type, egg_name: str | None, default_replacement: object, ruleset: Ruleset | None
) -> tuple[Declarations, TypedExprDecl]:
    decls = Declarations()
    type_ref = resolve_type_annotation_mutate(decls, tp)
    callable_ref = ConstantRef(name)
    decls._constants[name] = ConstantDecl(type_ref.to_just(), egg_name)
    _add_default_rewrite(decls, callable_ref, type_ref, default_replacement, ruleset, subsume=False)
    return decls, TypedExprDecl(type_ref.to_just(), CallDecl(callable_ref))


def _create_default_value(
    decls: Declarations,
    ref: CallableRef | None,
    fn: Callable,
    args: Iterable[TypedExprDecl],
    ruleset: Ruleset | None,
) -> object:
    args: list[object] = [RuntimeExpr.__from_values__(decls, a) for a in args]

    # If this is a classmethod, add the class as the first arg
    if isinstance(ref, ClassMethodRef):
        tp = decls.get_paramaterized_class(ref.class_name)
        args.insert(0, RuntimeClass(Thunk.value(decls), tp))
    with set_current_ruleset(ruleset):
        return fn(*args)


def _add_default_rewrite_function(
    decls: Declarations,
    ref: CallableRef,
    res_type: TypeOrVarRef,
    ruleset: Ruleset | None,
    value_thunk: Callable[[], object],
    subsume: bool,
) -> None:
    """
    Helper functions that resolves a value thunk to create the default value.
    """
    _add_default_rewrite(decls, ref, res_type, value_thunk(), ruleset, subsume)


def _add_default_rewrite(
    decls: Declarations,
    ref: CallableRef,
    type_ref: TypeOrVarRef,
    default_rewrite: object,
    ruleset: Ruleset | None,
    subsume: bool,
) -> None:
    """
    Adds a default rewrite for the callable, if the default rewrite is not None

    Will add it to the ruleset if it is passed in, or add it to the default ruleset on the passed in decls if not.
    """
    if default_rewrite is None:
        return
    resolved_value = resolve_literal(type_ref, default_rewrite, Thunk.value(decls))
    rewrite_decl = DefaultRewriteDecl(ref, resolved_value.__egg_typed_expr__.expr, subsume)
    if ruleset:
        ruleset_decls = ruleset._current_egg_decls
        ruleset_decl = ruleset.__egg_ruleset__
    else:
        ruleset_decls = decls
        ruleset_decl = decls.default_ruleset
    ruleset_decl.rules.append(rewrite_decl)
    ruleset_decls |= resolved_value


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
    split_functions: list[object]
    include_temporary_functions: bool


@dataclass
class EGraph:
    """
    A collection of expressions where each expression is part of a distinct equivalence class.

    Can run actions, check facts, run schedules, or extract minimal cost expressions.
    """

    seminaive: InitVar[bool] = True
    save_egglog_string: InitVar[bool] = False

    _state: EGraphState = field(init=False, repr=False)
    # For pushing/popping with egglog
    _state_stack: list[EGraphState] = field(default_factory=list, repr=False)
    # For storing the global "current" egraph
    _token_stack: list[EGraph] = field(default_factory=list, repr=False)

    def __post_init__(self, seminaive: bool, save_egglog_string: bool) -> None:
        egraph = bindings.EGraph(GLOBAL_PY_OBJECT_SORT, seminaive=seminaive, record=save_egglog_string)
        self._state = EGraphState(egraph)

    def _add_decls(self, *decls: DeclerationsLike) -> None:
        for d in decls:
            self._state.__egg_decls__ |= d

    @property
    def as_egglog_string(self) -> str:
        """
        Returns the egglog string for this module.
        """
        cmds = self._egraph.commands()
        if cmds is None:
            msg = "Can't get egglog string unless EGraph created with save_egglog_string=True"
            raise ValueError(msg)
        return cmds

    def _ipython_display_(self) -> None:
        self.display()

    def input(self, fn: Callable[..., String], path: str) -> None:
        """
        Loads a CSV file and sets it as *input, output of the function.
        """
        self._egraph.run_program(bindings.Input(span(1), self._callable_to_egg(fn), path))

    def _callable_to_egg(self, fn: object) -> str:
        ref, decls = resolve_callable(fn)
        self._add_decls(decls)
        return self._state.callable_ref_to_egg(ref)[0]

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
                self.__egg_decls__, TypedExprDecl(runtime_expr.__egg_typed_expr__.tp, VarDecl(name, True))
            ),
        )

    @overload
    def simplify(self, expr: BASE_EXPR, limit: int, /, *until: Fact, ruleset: Ruleset | None = None) -> BASE_EXPR: ...

    @overload
    def simplify(self, expr: BASE_EXPR, schedule: Schedule, /) -> BASE_EXPR: ...

    def simplify(
        self, expr: BASE_EXPR, limit_or_schedule: int | Schedule, /, *until: Fact, ruleset: Ruleset | None = None
    ) -> BASE_EXPR:
        """
        Simplifies the given expression.
        """
        schedule = run(ruleset, *until) * limit_or_schedule if isinstance(limit_or_schedule, int) else limit_or_schedule
        del limit_or_schedule, until, ruleset
        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr, schedule)
        egg_schedule = self._state.schedule_to_egg(schedule.schedule)
        typed_expr = runtime_expr.__egg_typed_expr__
        # Must also register type
        egg_expr = self._state.typed_expr_to_egg(typed_expr)
        self._egraph.run_program(bindings.Simplify(span(1), egg_expr, egg_schedule))
        extract_report = self._egraph.extract_report()
        if not isinstance(extract_report, bindings.Best):
            msg = "No extract report saved"
            raise ValueError(msg)  # noqa: TRY004
        (new_typed_expr,) = self._state.exprs_from_egg(extract_report.termdag, [extract_report.term], typed_expr.tp)
        return cast("BASE_EXPR", RuntimeExpr.__from_values__(self.__egg_decls__, new_typed_expr))

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
    def run(self, limit: int, /, *until: Fact, ruleset: Ruleset | None = None) -> bindings.RunReport: ...

    @overload
    def run(self, schedule: Schedule, /) -> bindings.RunReport: ...

    def run(
        self, limit_or_schedule: int | Schedule, /, *until: Fact, ruleset: Ruleset | None = None
    ) -> bindings.RunReport:
        """
        Run the egraph until the given limit or until the given facts are true.
        """
        if isinstance(limit_or_schedule, int):
            limit_or_schedule = run(ruleset, *until) * limit_or_schedule
        return self._run_schedule(limit_or_schedule)

    def _run_schedule(self, schedule: Schedule) -> bindings.RunReport:
        self._add_decls(schedule)
        egg_schedule = self._state.schedule_to_egg(schedule.schedule)
        self._egraph.run_program(bindings.RunSchedule(egg_schedule))
        run_report = self._egraph.run_report()
        if not run_report:
            msg = "No run report saved"
            raise ValueError(msg)
        return run_report

    def check_bool(self, *facts: FactLike) -> bool:
        """
        Returns true if the facts are true in the egraph.
        """
        try:
            self.check(*facts)
        # TODO: Make a separate exception class for this
        except Exception as e:
            if "Check failed" in str(e):
                return False
            raise
        return True

    def check(self, *facts: FactLike) -> None:
        """
        Check if a fact is true in the egraph.
        """
        self._egraph.run_program(self._facts_to_check(facts))

    def check_fail(self, *facts: FactLike) -> None:
        """
        Checks that one of the facts is not true
        """
        self._egraph.run_program(bindings.Fail(span(1), self._facts_to_check(facts)))

    def _facts_to_check(self, fact_likes: Iterable[FactLike]) -> bindings.Check:
        facts = _fact_likes(fact_likes)
        self._add_decls(*facts)
        egg_facts = [self._state.fact_to_egg(f.fact) for f in _fact_likes(facts)]
        return bindings.Check(span(2), egg_facts)

    @overload
    def extract(self, expr: BASE_EXPR, /, include_cost: Literal[False] = False) -> BASE_EXPR: ...

    @overload
    def extract(self, expr: BASE_EXPR, /, include_cost: Literal[True]) -> tuple[BASE_EXPR, int]: ...

    def extract(self, expr: BASE_EXPR, include_cost: bool = False) -> BASE_EXPR | tuple[BASE_EXPR, int]:
        """
        Extract the lowest cost expression from the egraph.
        """
        runtime_expr = to_runtime_expr(expr)
        extract_report = self._run_extract(runtime_expr, 0)

        if not isinstance(extract_report, bindings.Best):
            msg = "No extract report saved"
            raise ValueError(msg)  # noqa: TRY004
        (new_typed_expr,) = self._state.exprs_from_egg(
            extract_report.termdag, [extract_report.term], runtime_expr.__egg_typed_expr__.tp
        )

        res = cast("BASE_EXPR", RuntimeExpr.__from_values__(self.__egg_decls__, new_typed_expr))
        if include_cost:
            return res, extract_report.cost
        return res

    def extract_multiple(self, expr: BASE_EXPR, n: int) -> list[BASE_EXPR]:
        """
        Extract multiple expressions from the egraph.
        """
        runtime_expr = to_runtime_expr(expr)
        extract_report = self._run_extract(runtime_expr, n)
        if not isinstance(extract_report, bindings.Variants):
            msg = "Wrong extract report type"
            raise ValueError(msg)  # noqa: TRY004
        new_exprs = self._state.exprs_from_egg(
            extract_report.termdag, extract_report.terms, runtime_expr.__egg_typed_expr__.tp
        )
        return [cast("BASE_EXPR", RuntimeExpr.__from_values__(self.__egg_decls__, expr)) for expr in new_exprs]

    def _run_extract(self, expr: RuntimeExpr, n: int) -> bindings._ExtractReport:
        self._add_decls(expr)
        expr = self._state.typed_expr_to_egg(expr.__egg_typed_expr__)
        try:
            self._egraph.run_program(
                bindings.ActionCommand(bindings.Extract(span(2), expr, bindings.Lit(span(2), bindings.Int(n))))
            )
        except BaseException as e:
            raise add_note("Extracting: " + str(expr), e)  # noqa: B904
        extract_report = self._egraph.extract_report()
        if not extract_report:
            msg = "No extract report saved"
            raise ValueError(msg)
        return extract_report

    def push(self) -> None:
        """
        Push the current state of the egraph, so that it can be popped later and reverted back.
        """
        self._egraph.run_program(bindings.Push(1))
        self._state_stack.append(self._state)
        self._state = self._state.copy()

    def pop(self) -> None:
        """
        Pop the current state of the egraph, reverting back to the previous state.
        """
        self._egraph.run_program(bindings.Pop(span(1), 1))
        self._state = self._state_stack.pop()

    def __enter__(self) -> Self:
        """
        Copy the egraph state, so that it can be reverted back to the original state at the end.

        Also sets the current egraph to this one.
        """
        self.push()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.pop()

    def _serialize(
        self,
        **kwargs: Unpack[GraphvizKwargs],
    ) -> bindings.SerializedEGraph:
        max_functions = kwargs.pop("max_functions", None)
        max_calls_per_function = kwargs.pop("max_calls_per_function", None)
        split_primitive_outputs = kwargs.pop("split_primitive_outputs", True)
        split_functions = kwargs.pop("split_functions", [])
        include_temporary_functions = kwargs.pop("include_temporary_functions", False)
        n_inline_leaves = kwargs.pop("n_inline_leaves", 1)
        serialized = self._egraph.serialize(
            [],
            max_functions=max_functions,
            max_calls_per_function=max_calls_per_function,
            include_temporary_functions=include_temporary_functions,
        )
        if split_primitive_outputs or split_functions:
            additional_ops = set(map(self._callable_to_egg, split_functions))
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
        from IPython.display import SVG, display

        from .visualizer_widget import VisualizerWidget

        if graphviz:
            if IN_IPYTHON:
                svg = self._graphviz(**kwargs).pipe(format="svg", quiet=True, encoding="utf-8")
                display(SVG(svg))
            else:
                self._graphviz(**kwargs).render(view=True, format="svg", quiet=True)
        else:
            serialized = self._serialize(**kwargs)
            VisualizerWidget(egraphs=[serialized.to_json()]).display_or_open()

    def saturate(
        self,
        schedule: Schedule | None = None,
        *,
        expr: Expr | None = None,
        max: int = 1000,
        visualize: bool = True,
        **kwargs: Unpack[GraphvizKwargs],
    ) -> None:
        """
        Saturate the egraph, running the given schedule until the egraph is saturated.
        It serializes the egraph at each step and returns a widget to visualize the egraph.

        If an `expr` is passed, it's also extracted after each run and printed
        """
        from .visualizer_widget import VisualizerWidget

        def to_json() -> str:
            if expr is not None:
                print(self.extract(expr), "\n")
            return self._serialize(**kwargs).to_json()

        if visualize:
            egraphs = [to_json()]
        i = 0
        # Always visualize, even if we encounter an error
        try:
            while (self.run(schedule or 1).updated) and i < max:
                i += 1
                if visualize:
                    egraphs.append(to_json())
        except:
            if visualize:
                egraphs.append(to_json())
            raise
        finally:
            if visualize:
                VisualizerWidget(egraphs=egraphs).display_or_open()

    @property
    def _egraph(self) -> bindings.EGraph:
        return self._state.egraph

    @property
    def __egg_decls__(self) -> Declarations:
        return self._state.__egg_decls__

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
            current_frame = inspect.currentframe()
            assert current_frame
            original_frame = current_frame.f_back
            assert original_frame
            command_likes = tuple(_rewrite_or_rule_generator(command_or_generator, original_frame))
        else:
            command_likes = (cast("CommandLike", command_or_generator), *command_likes)
        commands = [_command_like(c) for c in command_likes]
        self._register_commands(commands)

    def _register_commands(self, cmds: list[Command]) -> None:
        self._add_decls(*cmds)
        egg_cmds = [egg_cmd for cmd in cmds if (egg_cmd := self._command_to_egg(cmd)) is not None]
        self._egraph.run_program(*egg_cmds)

    def _command_to_egg(self, cmd: Command) -> bindings._Command | None:
        ruleset_name = ""
        cmd_decl: CommandDecl
        match cmd:
            case RewriteOrRule(_, cmd_decl, ruleset):
                if ruleset:
                    ruleset_name = ruleset.__egg_name__
            case Action(_, action):
                cmd_decl = ActionCommandDecl(action)
            case _:
                assert_never(cmd)
        return self._state.command_to_egg(cmd_decl, ruleset_name)


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
    if isinstance(rule_or_generator, FunctionType):
        name = name or rule_or_generator.__name__
    r = Ruleset(name)
    if rule_or_generator is not None:
        r.register(rule_or_generator, *rules, _increase_frame=True)
    return r


@dataclass
class Schedule(DelayedDeclerations):
    """
    A composition of some rulesets, either composing them sequentially, running them repeatedly, running them till saturation, or running until some facts are met
    """

    # Defer declerations so that we can have rule generators that used not yet defined yet
    schedule: ScheduleDecl

    def __str__(self) -> str:
        return pretty_decl(self.__egg_decls__, self.schedule)

    def __repr__(self) -> str:
        return str(self)

    def __mul__(self, length: int) -> Schedule:
        """
        Repeat the schedule a number of times.
        """
        return Schedule(self.__egg_decls_thunk__, RepeatDecl(self.schedule, length))

    def saturate(self) -> Schedule:
        """
        Run the schedule until the e-graph is saturated.
        """
        return Schedule(self.__egg_decls_thunk__, SaturateDecl(self.schedule))

    def __add__(self, other: Schedule) -> Schedule:
        """
        Run two schedules in sequence.
        """
        return Schedule(Thunk.fn(Declarations.create, self, other), SequenceDecl((self.schedule, other.schedule)))


@dataclass
class Ruleset(Schedule):
    """
    A collection of rules, which can be run as a schedule.
    """

    __egg_decls_thunk__: Callable[[], Declarations] = field(init=False)
    schedule: RunDecl = field(init=False)
    name: str | None

    # Current declerations we have accumulated
    _current_egg_decls: Declarations = field(default_factory=Declarations)
    # Current rulesets we have accumulated
    __egg_ruleset__: RulesetDecl = field(init=False)
    # Rule generator functions that have been deferred, to allow for late type binding
    deferred_rule_gens: list[Callable[[], Iterable[RewriteOrRule]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schedule = RunDecl(self.__egg_name__, ())
        self.__egg_ruleset__ = self._current_egg_decls._rulesets[self.__egg_name__] = RulesetDecl([])
        self.__egg_decls_thunk__ = self._update_egg_decls

    def _update_egg_decls(self) -> Declarations:
        """
        To return the egg decls, we go through our deferred rules and add any we haven't yet
        """
        while self.deferred_rule_gens:
            with set_current_ruleset(self):
                rules = self.deferred_rule_gens.pop()()
            self._current_egg_decls.update(*rules)
            self.__egg_ruleset__.rules.extend(r.decl for r in rules)
        return self._current_egg_decls

    def append(self, rule: RewriteOrRule) -> None:
        """
        Register a rule with the ruleset.
        """
        self._current_egg_decls |= rule
        self.__egg_ruleset__.rules.append(rule.decl)

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
        return pretty_decl(self._current_egg_decls, self.__egg_ruleset__, ruleset_name=self.name)

    def __repr__(self) -> str:
        return str(self)

    def __or__(self, other: Ruleset | UnstableCombinedRuleset) -> UnstableCombinedRuleset:
        return unstable_combine_rulesets(self, other)

    # Create a unique name if we didn't pass one from the user
    @property
    def __egg_name__(self) -> str:
        return self.name or f"ruleset_{id(self)}"


@dataclass
class UnstableCombinedRuleset(Schedule):
    __egg_decls_thunk__: Callable[[], Declarations] = field(init=False)
    schedule: RunDecl = field(init=False)
    name: str | None
    rulesets: InitVar[list[Ruleset | UnstableCombinedRuleset]]

    def __post_init__(self, rulesets: list[Ruleset | UnstableCombinedRuleset]) -> None:
        self.schedule = RunDecl(self.__egg_name__, ())
        # Don't use thunk so that this is re-evaluated each time its requsted, so that additions inside will
        # be added after its been evaluated once.
        self.__egg_decls_thunk__ = partial(self._create_egg_decls, *rulesets)

    @property
    def __egg_name__(self) -> str:
        return self.name or f"combined_ruleset_{id(self)}"

    def _create_egg_decls(self, *rulesets: Ruleset | UnstableCombinedRuleset) -> Declarations:
        decls = Declarations.create(*rulesets)
        decls._rulesets[self.__egg_name__] = CombinedRulesetDecl(tuple(r.__egg_name__ for r in rulesets))
        return decls

    def __or__(self, other: Ruleset | UnstableCombinedRuleset) -> UnstableCombinedRuleset:
        return unstable_combine_rulesets(self, other)


def unstable_combine_rulesets(
    *rulesets: Ruleset | UnstableCombinedRuleset, name: str | None = None
) -> UnstableCombinedRuleset:
    """
    Combine multiple rulesets into a single ruleset.
    """
    return UnstableCombinedRuleset(name, list(rulesets))


@dataclass
class RewriteOrRule:
    __egg_decls__: Declarations
    decl: RewriteOrRuleDecl
    ruleset: Ruleset | None = None

    def __str__(self) -> str:
        return pretty_decl(self.__egg_decls__, self.decl)

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Fact:
    """
    A query on an EGraph, either by an expression or an equivalence between multiple expressions.
    """

    __egg_decls__: Declarations
    fact: FactDecl

    def __str__(self) -> str:
        return pretty_decl(self.__egg_decls__, self.fact)

    def __repr__(self) -> str:
        return str(self)

    def __bool__(self) -> bool:
        """
        Returns True if the two sides of an equality are structurally equal.
        """
        if not isinstance(self.fact, EqDecl):
            msg = "Can only check equality facts"
            raise TypeError(msg)
        return self.fact.left == self.fact.right


@dataclass
class Action:
    """
    A change to an EGraph, either unioning multiple expressions, setting the value of a function call, deleting an expression, or panicing.
    """

    __egg_decls__: Declarations
    action: ActionDecl

    def __str__(self) -> str:
        return pretty_decl(self.__egg_decls__, self.action)

    def __repr__(self) -> str:
        return str(self)


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


def rewrite(lhs: EXPR, ruleset: None = None, *, subsume: bool = False) -> _RewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression."""
    return _RewriteBuilder(lhs, ruleset, subsume)


def birewrite(lhs: EXPR, ruleset: None = None) -> _BirewriteBuilder[EXPR]:
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
    return Action(Declarations(), PanicDecl(message))


def let(name: str, expr: BaseExpr) -> Action:
    """Create a let binding."""
    runtime_expr = to_runtime_expr(expr)
    return Action(runtime_expr.__egg_decls__, LetDecl(name, runtime_expr.__egg_typed_expr__))


def expr_action(expr: BaseExpr) -> Action:
    runtime_expr = to_runtime_expr(expr)
    return Action(runtime_expr.__egg_decls__, ExprActionDecl(runtime_expr.__egg_typed_expr__))


def delete(expr: BaseExpr) -> Action:
    """Create a delete expression."""
    runtime_expr = to_runtime_expr(expr)
    typed_expr = runtime_expr.__egg_typed_expr__
    call_decl = typed_expr.expr
    assert isinstance(call_decl, CallDecl), "Can only delete calls, not literals or vars"
    return Action(runtime_expr.__egg_decls__, ChangeDecl(typed_expr.tp, call_decl, "delete"))


def subsume(expr: Expr) -> Action:
    """Subsume an expression so it cannot be matched against or extracted"""
    runtime_expr = to_runtime_expr(expr)
    typed_expr = runtime_expr.__egg_typed_expr__
    call_decl = typed_expr.expr
    assert isinstance(call_decl, CallDecl), "Can only subsume calls, not literals or vars"
    return Action(runtime_expr.__egg_decls__, ChangeDecl(typed_expr.tp, call_decl, "subsume"))


def expr_fact(expr: BaseExpr) -> Fact:
    runtime_expr = to_runtime_expr(expr)
    return Fact(runtime_expr.__egg_decls__, ExprFactDecl(runtime_expr.__egg_typed_expr__))


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    """Create a union of the given expression."""
    return _UnionBuilder(lhs=lhs)


def set_(lhs: BASE_EXPR) -> _SetBuilder[BASE_EXPR]:
    """Create a set of the given expression."""
    return _SetBuilder(lhs=lhs)


def rule(*facts: FactLike, ruleset: None = None, name: str | None = None) -> _RuleBuilder:
    """Create a rule with the given facts."""
    return _RuleBuilder(facts=_fact_likes(facts), name=name, ruleset=ruleset)


def var(name: str, bound: type[T]) -> T:
    """Create a new variable with the given name and type."""
    return cast("T", _var(name, bound))


def _var(name: str, bound: object) -> RuntimeExpr:
    """Create a new variable with the given name and type."""
    decls_like, type_ref = resolve_type_annotation(bound)
    return RuntimeExpr(
        Thunk.fn(Declarations.create, decls_like), Thunk.value(TypedExprDecl(type_ref.to_just(), VarDecl(name, False)))
    )


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
        lhs = to_runtime_expr(self.lhs)
        facts = _fact_likes(conditions)
        rhs = convert_to_same_type(rhs, lhs)
        rule = RewriteOrRule(
            Declarations.create(lhs, rhs, *facts, self.ruleset),
            RewriteDecl(
                lhs.__egg_typed_expr__.tp,
                lhs.__egg_typed_expr__.expr,
                rhs.__egg_typed_expr__.expr,
                tuple(f.fact for f in facts),
                self.subsume,
            ),
        )
        if self.ruleset:
            self.ruleset.append(rule)
        return rule

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return lhs.__egg_pretty__("rewrite")


@dataclass
class _BirewriteBuilder(Generic[EXPR]):
    lhs: EXPR
    ruleset: Ruleset | None

    def to(self, rhs: EXPR, *conditions: FactLike) -> RewriteOrRule:
        lhs = to_runtime_expr(self.lhs)
        facts = _fact_likes(conditions)
        rhs = convert_to_same_type(rhs, lhs)
        rule = RewriteOrRule(
            Declarations.create(lhs, rhs, *facts, self.ruleset),
            BiRewriteDecl(
                lhs.__egg_typed_expr__.tp,
                lhs.__egg_typed_expr__.expr,
                rhs.__egg_typed_expr__.expr,
                tuple(f.fact for f in facts),
            ),
        )
        if self.ruleset:
            self.ruleset.append(rule)
        return rule

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return lhs.__egg_pretty__("birewrite")


@dataclass
class _EqBuilder(Generic[BASE_EXPR]):
    expr: BASE_EXPR

    def to(self, other: BASE_EXPR) -> Fact:
        expr = to_runtime_expr(self.expr)
        other = convert_to_same_type(other, expr)
        return Fact(
            Declarations.create(expr, other),
            EqDecl(expr.__egg_typed_expr__.tp, expr.__egg_typed_expr__.expr, other.__egg_typed_expr__.expr),
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = to_runtime_expr(self.expr)
        return expr.__egg_pretty__("eq")


@dataclass
class _NeBuilder(Generic[BASE_EXPR]):
    lhs: BASE_EXPR

    def to(self, rhs: BASE_EXPR) -> Unit:
        from .builtins import Unit

        lhs = to_runtime_expr(self.lhs)
        rhs = convert_to_same_type(rhs, lhs)
        res = RuntimeExpr.__from_values__(
            Declarations.create(cast("RuntimeClass", Unit), lhs, rhs),
            TypedExprDecl(
                JustTypeRef("Unit"), CallDecl(FunctionRef("!="), (lhs.__egg_typed_expr__, rhs.__egg_typed_expr__))
            ),
        )
        return cast("Unit", res)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = to_runtime_expr(self.lhs)
        return expr.__egg_pretty__("ne")


@dataclass
class _SetBuilder(Generic[BASE_EXPR]):
    lhs: BASE_EXPR

    def to(self, rhs: BASE_EXPR) -> Action:
        lhs = to_runtime_expr(self.lhs)
        rhs = convert_to_same_type(rhs, lhs)
        lhs_expr = lhs.__egg_typed_expr__.expr
        assert isinstance(lhs_expr, CallDecl), "Can only set function calls"
        return Action(
            Declarations.create(lhs, rhs),
            SetDecl(lhs.__egg_typed_expr__.tp, lhs_expr, rhs.__egg_typed_expr__.expr),
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return lhs.__egg_pretty__("set_")


@dataclass
class _UnionBuilder(Generic[EXPR]):
    lhs: EXPR

    def with_(self, rhs: EXPR) -> Action:
        lhs = to_runtime_expr(self.lhs)
        rhs = convert_to_same_type(rhs, lhs)
        return Action(
            Declarations.create(lhs, rhs),
            UnionDecl(lhs.__egg_typed_expr__.tp, lhs.__egg_typed_expr__.expr, rhs.__egg_typed_expr__.expr),
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        lhs = to_runtime_expr(self.lhs)
        return lhs.__egg_pretty__("union")


@dataclass
class _RuleBuilder:
    facts: tuple[Fact, ...]
    name: str | None
    ruleset: Ruleset | None

    def then(self, *actions: ActionLike) -> RewriteOrRule:
        actions = _action_likes(actions)
        rule = RewriteOrRule(
            Declarations.create(self.ruleset, *actions, *self.facts),
            RuleDecl(tuple(a.action for a in actions), tuple(f.fact for f in self.facts), self.name),
        )
        if self.ruleset:
            self.ruleset.append(rule)
        return rule

    def __str__(self) -> str:
        # TODO: Figure out how to stringify rulebuilder that preserves statements
        args = list(map(str, self.facts))
        if self.name is not None:
            args.append(f"name={self.name}")
        if ruleset is not None:
            args.append(f"ruleset={self.ruleset}")
        return f"rule({', '.join(args)})"


def expr_parts(expr: BaseExpr) -> TypedExprDecl:
    """
    Returns the underlying type and decleration of the expression. Useful for testing structural equality or debugging.
    """
    if not isinstance(expr, RuntimeExpr):
        raise TypeError(f"Expected a RuntimeExpr not {expr}")
    return expr.__egg_typed_expr__


def to_runtime_expr(expr: BaseExpr) -> RuntimeExpr:
    if not isinstance(expr, RuntimeExpr):
        raise TypeError(f"Expected a RuntimeExpr not {expr}")
    return expr


def run(ruleset: Ruleset | None = None, *until: FactLike) -> Schedule:
    """
    Create a run configuration.
    """
    facts = _fact_likes(until)
    return Schedule(
        Thunk.fn(Declarations.create, ruleset, *facts),
        RunDecl(ruleset.__egg_name__ if ruleset else "", tuple(f.fact for f in facts) or None),
    )


def seq(*schedules: Schedule) -> Schedule:
    """
    Run a sequence of schedules.
    """
    return Schedule(Thunk.fn(Declarations.create, *schedules), SequenceDecl(tuple(s.schedule for s in schedules)))


ActionLike: TypeAlias = Action | BaseExpr


def _action_likes(action_likes: Iterable[ActionLike]) -> tuple[Action, ...]:
    return tuple(map(_action_like, action_likes))


def _action_like(action_like: ActionLike) -> Action:
    if isinstance(action_like, Action):
        return action_like
    return expr_action(action_like)


Command: TypeAlias = Action | RewriteOrRule

CommandLike: TypeAlias = ActionLike | RewriteOrRule


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
    combined = {**gen.__globals__, **frame.f_locals}
    hints = get_type_hints(gen, combined, combined)
    args = [_var(p.name, hints[p.name]) for p in signature(gen).parameters.values()]
    return list(gen(*args))  # type: ignore[misc]


FactLike = Fact | BaseExpr


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
    token = _CURRENT_RULESET.set(r)
    try:
        yield
    finally:
        _CURRENT_RULESET.reset(token)
