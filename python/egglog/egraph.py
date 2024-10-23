from __future__ import annotations

import contextlib
import inspect
import pathlib
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Generator, Iterable
from contextvars import ContextVar, Token
from dataclasses import InitVar, dataclass, field
from inspect import Parameter, currentframe, signature
from types import FrameType, FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

import graphviz
from typing_extensions import ParamSpec, Self, Unpack, assert_never, deprecated

from . import bindings
from .conversion import *
from .declarations import *
from .egraph_state import *
from .ipython_magic import IN_IPYTHON
from .pretty import pretty_decl
from .runtime import *
from .thunk import *

if TYPE_CHECKING:
    from .builtins import Bool, PyObject, String, f64, i64


__all__ = [
    "EGraph",
    "Module",
    "function",
    "ruleset",
    "method",
    "relation",
    "Expr",
    "Unit",
    "rewrite",
    "birewrite",
    "eq",
    "ne",
    "panic",
    "let",
    "constant",
    "delete",
    "subsume",
    "union",
    "set_",
    "rule",
    "var",
    "vars_",
    "Fact",
    "expr_parts",
    "expr_action",
    "expr_fact",
    "action_command",
    "Schedule",
    "run",
    "seq",
    "Command",
    "simplify",
    "unstable_combine_rulesets",
    "check",
    "GraphvizKwargs",
    "Ruleset",
    "_RewriteBuilder",
    "_BirewriteBuilder",
    "_EqBuilder",
    "_NeBuilder",
    "_SetBuilder",
    "_UnionBuilder",
    "RewriteOrRule",
    "Fact",
    "Action",
    "Command",
    "check_eq",
]

T = TypeVar("T")
P = ParamSpec("P")
TYPE = TypeVar("TYPE", bound="type[Expr]")
CALLABLE = TypeVar("CALLABLE", bound=Callable)
EXPR = TypeVar("EXPR", bound="Expr")
E1 = TypeVar("E1", bound="Expr")
E2 = TypeVar("E2", bound="Expr")
E3 = TypeVar("E3", bound="Expr")
E4 = TypeVar("E4", bound="Expr")
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


def check_eq(x: EXPR, y: EXPR, schedule: Schedule | None = None) -> EGraph:
    """
    Verifies that two expressions are equal after running the schedule.
    """
    egraph = EGraph()
    x_var = egraph.let("__check_eq_x", x)
    y_var = egraph.let("__check_eq_y", y)
    if schedule:
        egraph.run(schedule)
    fact = eq(x_var).to(y_var)
    try:
        egraph.check(fact)
    except bindings.EggSmolError as err:
        raise AssertionError(f"Failed {eq(x).to(y)}\n -> {ne(egraph.extract(x)).to(egraph.extract(y))})") from err
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


@dataclass
class _BaseModule:
    """
    Base Module which provides methods to register sorts, expressions, actions etc.

    Inherited by:
    - EGraph: Holds a live EGraph instance
    - Builtins: Stores a list of the builtins which have already been pre-regsietered
    - Module: Stores a list of commands and additional declerations
    """

    # TODO: If we want to preserve existing semantics, then we use the module to find the default schedules
    # and add them to the

    modules: InitVar[list[Module]] = []  # noqa: RUF008

    # TODO: Move commands to Decleraration instance. Pass in is_builtins to declerations so we can skip adding commands for those. Pass in from module, set as argument of module and subclcass

    # Any modules you want to depend on
    # # All dependencies flattened
    _flatted_deps: list[Module] = field(init=False, default_factory=list)
    # _mod_decls: ModuleDeclarations = field(init=False)

    def __post_init__(self, modules: list[Module]) -> None:
        for mod in modules:
            for child_mod in [*mod._flatted_deps, mod]:
                if child_mod not in self._flatted_deps:
                    self._flatted_deps.append(child_mod)

    @deprecated("Remove this decorator and move the egg_sort to the class statement, i.e. E(Expr, egg_sort='MySort').")
    @overload
    def class_(self, *, egg_sort: str) -> Callable[[TYPE], TYPE]: ...

    @deprecated("Remove this decorator. Simply subclassing Expr is enough now.")
    @overload
    def class_(self, cls: TYPE, /) -> TYPE: ...

    def class_(self, *args, **kwargs) -> Any:
        """
        Registers a class.
        """
        if kwargs:
            msg = "Switch to subclassing from Expr and passing egg_sort as a keyword arg to the class constructor"
            raise NotImplementedError(msg)

        assert len(args) == 1
        return args[0]

    @overload
    def method(
        self,
        *,
        preserve: Literal[True],
    ) -> Callable[[CALLABLE], CALLABLE]: ...

    @overload
    def method(
        self,
        *,
        egg_fn: str | None = None,
        cost: int | None = None,
        merge: Callable[[Any, Any], Any] | None = None,
        on_merge: Callable[[Any, Any], Iterable[ActionLike]] | None = None,
        mutates_self: bool = False,
        unextractable: bool = False,
    ) -> Callable[[CALLABLE], CALLABLE]: ...

    @overload
    def method(
        self,
        *,
        egg_fn: str | None = None,
        cost: int | None = None,
        default: EXPR | None = None,
        merge: Callable[[EXPR, EXPR], EXPR] | None = None,
        on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None = None,
        mutates_self: bool = False,
        unextractable: bool = False,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]: ...

    @deprecated("Use top level method function instead")
    def method(
        self,
        *,
        egg_fn: str | None = None,
        cost: int | None = None,
        default: EXPR | None = None,
        merge: Callable[[EXPR, EXPR], EXPR] | None = None,
        on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None = None,
        preserve: bool = False,
        mutates_self: bool = False,
        unextractable: bool = False,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        return lambda fn: _WrappedMethod(
            egg_fn, cost, default, merge, on_merge, fn, preserve, mutates_self, unextractable
        )

    @overload
    def function(self, fn: CALLABLE, /) -> CALLABLE: ...

    @overload
    def function(
        self,
        *,
        egg_fn: str | None = None,
        cost: int | None = None,
        merge: Callable[[Any, Any], Any] | None = None,
        on_merge: Callable[[Any, Any], Iterable[ActionLike]] | None = None,
        mutates_first_arg: bool = False,
        unextractable: bool = False,
    ) -> Callable[[CALLABLE], CALLABLE]: ...

    @overload
    def function(
        self,
        *,
        egg_fn: str | None = None,
        cost: int | None = None,
        default: EXPR | None = None,
        merge: Callable[[EXPR, EXPR], EXPR] | None = None,
        on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None = None,
        mutates_first_arg: bool = False,
        unextractable: bool = False,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]: ...

    @deprecated("Use top level function `function` instead")
    def function(self, *args, **kwargs) -> Any:
        """
        Registers a function.
        """
        fn_locals = currentframe().f_back.f_back.f_locals  # type: ignore[union-attr]
        # If we have any positional args, then we are calling it directly on a function
        if args:
            assert len(args) == 1
            return _FunctionConstructor(fn_locals)(args[0])
        # otherwise, we are passing some keyword args, so save those, and then return a partial
        return _FunctionConstructor(fn_locals, **kwargs)

    @deprecated("Use top level `ruleset` function instead")
    def ruleset(self, name: str) -> Ruleset:
        return Ruleset(name)

    # Overload to support aritys 0-4 until variadic generic support map, so we can map from type to value
    @overload
    def relation(
        self, name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], tp4: type[E4], /
    ) -> Callable[[E1, E2, E3, E4], Unit]: ...

    @overload
    def relation(self, name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], /) -> Callable[[E1, E2, E3], Unit]: ...

    @overload
    def relation(self, name: str, tp1: type[E1], tp2: type[E2], /) -> Callable[[E1, E2], Unit]: ...

    @overload
    def relation(self, name: str, tp1: type[T], /, *, egg_fn: str | None = None) -> Callable[[T], Unit]: ...

    @overload
    def relation(self, name: str, /, *, egg_fn: str | None = None) -> Callable[[], Unit]: ...

    @deprecated("Use top level relation function instead")
    def relation(self, name: str, /, *tps: type, egg_fn: str | None = None) -> Callable[..., Unit]:
        """
        Defines a relation, which is the same as a function which returns unit.
        """
        return relation(name, *tps, egg_fn=egg_fn)

    @deprecated("Use top level constant function instead")
    def constant(self, name: str, tp: type[EXPR], egg_name: str | None = None) -> EXPR:
        """

        Defines a named constant of a certain type.

        This is the same as defining a nullary function with a high cost.
        # TODO: Rename as declare to match eggglog?
        """
        return constant(name, tp, egg_name=egg_name)

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
            command_likes = (cast(CommandLike, command_or_generator), *command_likes)
        commands = [_command_like(c) for c in command_likes]
        self._register_commands(commands)

    @abstractmethod
    def _register_commands(self, cmds: list[Command]) -> None:
        raise NotImplementedError


# We seperate the function and method overloads to make it simpler to know if we are modifying a function or method,
# So that we can add the functions eagerly to the registry and wait on the methods till we process the class.


@overload
def method(
    *,
    preserve: Literal[True],
) -> Callable[[CALLABLE], CALLABLE]: ...


# We have to seperate method/function overloads for those that use the T params and those that don't
# Otherwise, if you say just pass in `cost` then the T param is inferred as `Nothing` and
# It will break the typing.


@overload
def method(
    *,
    egg_fn: str | None = None,
    cost: int | None = None,
    merge: Callable[[Any, Any], Any] | None = None,
    on_merge: Callable[[Any, Any], Iterable[ActionLike]] | None = None,
    mutates_self: bool = False,
    unextractable: bool = False,
) -> Callable[[CALLABLE], CALLABLE]: ...


@overload
def method(
    *,
    egg_fn: str | None = None,
    cost: int | None = None,
    default: EXPR | None = None,
    merge: Callable[[EXPR, EXPR], EXPR] | None = None,
    on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None = None,
    mutates_self: bool = False,
    unextractable: bool = False,
) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]: ...


def method(
    *,
    egg_fn: str | None = None,
    cost: int | None = None,
    default: EXPR | None = None,
    merge: Callable[[EXPR, EXPR], EXPR] | None = None,
    on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None = None,
    preserve: bool = False,
    mutates_self: bool = False,
    unextractable: bool = False,
) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
    """
    Any method can be decorated with this to customize it's behavior. This is only supported in classes which subclass :class:`Expr`.
    """
    return lambda fn: _WrappedMethod(egg_fn, cost, default, merge, on_merge, fn, preserve, mutates_self, unextractable)


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
        builtin: bool = False,
        ruleset: Ruleset | None = None,
    ) -> RuntimeClass | type:
        # If this is the Expr subclass, just return the class
        if not bases:
            return super().__new__(cls, name, bases, namespace)
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


def _generate_class_decls(  # noqa: C901
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
    type_vars = tuple(p.__name__ for p in parameters)
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
            type_ref = resolve_type_annotation(decls, inner_tp)
            cls_decl.class_variables[k] = ConstantDecl(type_ref.to_just())
            _add_default_rewrite(decls, ClassVariableRef(cls_name, k), type_ref, namespace.pop(k, None), ruleset)
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
            case _WrappedMethod(egg_fn, cost, default, merge, on_merge, fn, preserve, mutates, unextractable):
                pass
            case _:
                egg_fn, cost, default, merge, on_merge = None, None, None, None, None
                fn = method
                unextractable, preserve = False, False
                mutates = method_name in ALWAYS_MUTATES_SELF
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
        special_function_name: SpecialFunctions | None = (
            "fn-partial" if egg_fn == "unstable-fn" else "fn-app" if egg_fn == "unstable-app" else None
        )
        if special_function_name:
            decl = FunctionDecl(special_function_name, builtin=True, egg_name=egg_fn)
            decls.set_function_decl(ref, decl)
            continue

        _, add_rewrite = _fn_decl(
            decls, egg_fn, ref, fn, locals, default, cost, merge, on_merge, mutates, builtin, ruleset, unextractable
        )

        if not builtin and not isinstance(ref, InitRef) and not mutates:
            add_default_funcs.append(add_rewrite)

    # Add all rewrite methods at the end so that all methods are registered first and can be accessed
    # in the bodies
    for add_rewrite in add_default_funcs:
        add_rewrite()

    return decls


@overload
def function(fn: CALLABLE, /) -> CALLABLE: ...


@overload
def function(
    *,
    egg_fn: str | None = None,
    cost: int | None = None,
    merge: Callable[[Any, Any], Any] | None = None,
    on_merge: Callable[[Any, Any], Iterable[ActionLike]] | None = None,
    mutates_first_arg: bool = False,
    unextractable: bool = False,
    builtin: bool = False,
    ruleset: Ruleset | None = None,
    use_body_as_name: bool = False,
) -> Callable[[CALLABLE], CALLABLE]: ...


@overload
def function(
    *,
    egg_fn: str | None = None,
    cost: int | None = None,
    default: EXPR | None = None,
    merge: Callable[[EXPR, EXPR], EXPR] | None = None,
    on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None = None,
    mutates_first_arg: bool = False,
    unextractable: bool = False,
    ruleset: Ruleset | None = None,
    use_body_as_name: bool = False,
) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]: ...


def function(*args, **kwargs) -> Any:
    """
    Defined by a unique name and a typing relation that will specify the return type based on the types of the argument expressions.


    """
    fn_locals = currentframe().f_back.f_locals  # type: ignore[union-attr]

    # If we have any positional args, then we are calling it directly on a function
    if args:
        assert len(args) == 1
        return _FunctionConstructor(fn_locals)(args[0])
    # otherwise, we are passing some keyword args, so save those, and then return a partial
    return _FunctionConstructor(fn_locals, **kwargs)


@dataclass
class _FunctionConstructor:
    hint_locals: dict[str, Any]
    builtin: bool = False
    mutates_first_arg: bool = False
    egg_fn: str | None = None
    cost: int | None = None
    default: RuntimeExpr | None = None
    merge: Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr] | None = None
    on_merge: Callable[[RuntimeExpr, RuntimeExpr], Iterable[ActionLike]] | None = None
    unextractable: bool = False
    ruleset: Ruleset | None = None
    use_body_as_name: bool = False

    def __call__(self, fn: Callable[..., RuntimeExpr]) -> RuntimeFunction:
        return RuntimeFunction(*split_thunk(Thunk.fn(self.create_decls, fn)))

    def create_decls(self, fn: Callable[..., RuntimeExpr]) -> tuple[Declarations, CallableRef]:
        decls = Declarations()
        ref = None if self.use_body_as_name else FunctionRef(fn.__name__)
        ref, add_rewrite = _fn_decl(
            decls,
            self.egg_fn,
            ref,
            fn,
            self.hint_locals,
            self.default,
            self.cost,
            self.merge,
            self.on_merge,
            self.mutates_first_arg,
            self.builtin,
            self.ruleset,
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
    default: RuntimeExpr | None,
    cost: int | None,
    merge: Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr] | None,
    on_merge: Callable[[RuntimeExpr, RuntimeExpr], Iterable[ActionLike]] | None,
    mutates_first_arg: bool,
    is_builtin: bool,
    ruleset: Ruleset | None = None,
    unextractable: bool = False,
) -> tuple[CallableRef, Callable[[], None]]:
    """
    Sets the function decl for the function object and returns the ref as well as a thunk that sets the default callable.
    """
    if not isinstance(fn, FunctionType):
        raise NotImplementedError(f"Can only generate function decls for functions not {fn}  {type(fn)}")

    hint_globals = fn.__globals__.copy()
    # Copy Callable into global if not present bc sometimes it gets automatically removed by ruff to type only block
    # https://docs.astral.sh/ruff/rules/typing-only-standard-library-import/
    if "Callable" not in hint_globals:
        hint_globals["Callable"] = Callable

    hints = get_type_hints(fn, hint_globals, hint_locals)

    params = list(signature(fn).parameters.values())

    # If this is an init function, or a classmethod, the first arg is not used
    if isinstance(ref, ClassMethodRef | InitRef):
        params = params[1:]

    if _last_param_variable(params):
        *params, var_arg_param = params
        # For now, we don't use the variable arg name
        var_arg_type = resolve_type_annotation(decls, hints[var_arg_param.name])
    else:
        var_arg_type = None
    arg_types = tuple(
        decls.get_paramaterized_class(ref.class_name)
        if i == 0 and isinstance(ref, MethodRef | PropertyRef)
        else resolve_type_annotation(decls, hints[t.name])
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
        else resolve_type_annotation(decls, hints["return"])
    )

    arg_names = tuple(t.name for t in params)

    decls |= default
    merged = (
        None
        if merge is None
        else merge(
            RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), VarDecl("old", False))),
            RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), VarDecl("new", False))),
        )
    )
    decls |= merged

    merge_action = (
        []
        if on_merge is None
        else _action_likes(
            on_merge(
                RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), VarDecl("old", False))),
                RuntimeExpr.__from_values__(decls, TypedExprDecl(return_type.to_just(), VarDecl("new", False))),
            )
        )
    )
    decls.update(*merge_action)
    # defer this in generator so it doesnt resolve for builtins eagerly
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
        signature_ = FunctionSignature(
            return_type=None if mutates_first_arg else return_type,
            var_arg_type=var_arg_type,
            arg_types=arg_types,
            arg_names=arg_names,
            arg_defaults=tuple(a.__egg_typed_expr__.expr if a is not None else None for a in arg_defaults),
        )
        decl = FunctionDecl(
            signature=signature_,
            cost=cost,
            egg_name=egg_name,
            merge=merged.__egg_typed_expr__.expr if merged is not None else None,
            unextractable=unextractable,
            builtin=is_builtin,
            default=None if default is None else default.__egg_typed_expr__.expr,
            on_merge=tuple(a.action for a in merge_action),
        )
        res_ref = ref
        decls.set_function_decl(ref, decl)
        res_thunk = Thunk.fn(_create_default_value, decls, ref, fn, args, ruleset)
    return res_ref, Thunk.fn(_add_default_rewrite_function, decls, res_ref, return_type, ruleset, res_thunk)


# Overload to support aritys 0-4 until variadic generic support map, so we can map from type to value
@overload
def relation(
    name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], tp4: type[E4], /
) -> Callable[[E1, E2, E3, E4], Unit]: ...


@overload
def relation(name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], /) -> Callable[[E1, E2, E3], Unit]: ...


@overload
def relation(name: str, tp1: type[E1], tp2: type[E2], /) -> Callable[[E1, E2], Unit]: ...


@overload
def relation(name: str, tp1: type[T], /, *, egg_fn: str | None = None) -> Callable[[T], Unit]: ...


@overload
def relation(name: str, /, *, egg_fn: str | None = None) -> Callable[[], Unit]: ...


def relation(name: str, /, *tps: type, egg_fn: str | None = None) -> Callable[..., Unit]:
    """
    Creates a function whose return type is `Unit` and has a default value.
    """
    decls_thunk = Thunk.fn(_relation_decls, name, tps, egg_fn)
    return cast(Callable[..., Unit], RuntimeFunction(decls_thunk, Thunk.value(FunctionRef(name))))


def _relation_decls(name: str, tps: tuple[type, ...], egg_fn: str | None) -> Declarations:
    decls = Declarations()
    decls |= cast(RuntimeClass, Unit)
    arg_types = tuple(resolve_type_annotation(decls, tp).to_just() for tp in tps)
    decls._functions[name] = RelationDecl(arg_types, tuple(None for _ in tps), egg_fn)
    return decls


def constant(
    name: str,
    tp: type[EXPR],
    default_replacement: EXPR | None = None,
    /,
    *,
    egg_name: str | None = None,
    ruleset: Ruleset | None = None,
) -> EXPR:
    """
    A "constant" is implemented as the instantiation of a value that takes no args.
    This creates a function with `name` and return type `tp` and returns a value of it being called.
    """
    return cast(
        EXPR, RuntimeExpr(*split_thunk(Thunk.fn(_constant_thunk, name, tp, egg_name, default_replacement, ruleset)))
    )


def _constant_thunk(
    name: str, tp: type, egg_name: str | None, default_replacement: object, ruleset: Ruleset | None
) -> tuple[Declarations, TypedExprDecl]:
    decls = Declarations()
    type_ref = resolve_type_annotation(decls, tp)
    callable_ref = ConstantRef(name)
    decls._constants[name] = ConstantDecl(type_ref.to_just(), egg_name)
    _add_default_rewrite(decls, callable_ref, type_ref, default_replacement, ruleset)
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
) -> None:
    """
    Helper functions that resolves a value thunk to create the default value.
    """
    _add_default_rewrite(decls, ref, res_type, value_thunk(), ruleset)


def _add_default_rewrite(
    decls: Declarations, ref: CallableRef, type_ref: TypeOrVarRef, default_rewrite: object, ruleset: Ruleset | None
) -> None:
    """
    Adds a default rewrite for the callable, if the default rewrite is not None

    Will add it to the ruleset if it is passed in, or add it to the default ruleset on the passed in decls if not.
    """
    if default_rewrite is None:
        return
    resolved_value = resolve_literal(type_ref, default_rewrite, Thunk.value(decls))
    rewrite_decl = DefaultRewriteDecl(ref, resolved_value.__egg_typed_expr__.expr)
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


@deprecated(
    "Modules are deprecated, use top level functions to register classes/functions and rulesets to register rules"
)
@dataclass
class Module(_BaseModule):
    cmds: list[Command] = field(default_factory=list)

    def _register_commands(self, cmds: list[Command]) -> None:
        self.cmds.extend(cmds)

    def without_rules(self) -> Module:
        return Module()

    # Use identity for hash and equility, so we don't have to compare commands and compare expressions
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Module):
            return NotImplemented
        return self is other


class GraphvizKwargs(TypedDict, total=False):
    max_functions: int | None
    max_calls_per_function: int | None
    n_inline_leaves: int
    split_primitive_outputs: bool
    split_functions: list[object]
    include_temporary_functions: bool


@dataclass
class EGraph(_BaseModule):
    """
    A collection of expressions where each expression is part of a distinct equivalence class.

    Can run actions, check facts, run schedules, or extract minimal cost expressions.
    """

    seminaive: InitVar[bool] = True
    save_egglog_string: InitVar[bool] = False

    _state: EGraphState = field(init=False)
    # For pushing/popping with egglog
    _state_stack: list[EGraphState] = field(default_factory=list, repr=False)
    # For storing the global "current" egraph
    _token_stack: list[Token[EGraph]] = field(default_factory=list, repr=False)

    def __post_init__(self, modules: list[Module], seminaive: bool, save_egglog_string: bool) -> None:
        egraph = bindings.EGraph(GLOBAL_PY_OBJECT_SORT, seminaive=seminaive, record=save_egglog_string)
        self._state = EGraphState(egraph)
        super().__post_init__(modules)

        for m in self._flatted_deps:
            self._register_commands(m.cmds)

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
        self._egraph.run_program(bindings.Input(bindings.DUMMY_SPAN, self._callable_to_egg(fn), path))

    def _callable_to_egg(self, fn: object) -> str:
        ref, decls = resolve_callable(fn)
        self._add_decls(decls)
        return self._state.callable_ref_to_egg(ref)

    def let(self, name: str, expr: EXPR) -> EXPR:
        """
        Define a new expression in the egraph and return a reference to it.
        """
        action = let(name, expr)
        self.register(action)
        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr)
        return cast(
            EXPR,
            RuntimeExpr.__from_values__(
                self.__egg_decls__, TypedExprDecl(runtime_expr.__egg_typed_expr__.tp, VarDecl(name, True))
            ),
        )

    @overload
    def simplify(self, expr: EXPR, limit: int, /, *until: Fact, ruleset: Ruleset | None = None) -> EXPR: ...

    @overload
    def simplify(self, expr: EXPR, schedule: Schedule, /) -> EXPR: ...

    def simplify(
        self, expr: EXPR, limit_or_schedule: int | Schedule, /, *until: Fact, ruleset: Ruleset | None = None
    ) -> EXPR:
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
        self._egraph.run_program(bindings.Simplify(bindings.DUMMY_SPAN, egg_expr, egg_schedule))
        extract_report = self._egraph.extract_report()
        if not isinstance(extract_report, bindings.Best):
            msg = "No extract report saved"
            raise ValueError(msg)  # noqa: TRY004
        (new_typed_expr,) = self._state.exprs_from_egg(extract_report.termdag, [extract_report.term], typed_expr.tp)
        return cast(EXPR, RuntimeExpr.__from_values__(self.__egg_decls__, new_typed_expr))

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

    def check(self, *facts: FactLike) -> None:
        """
        Check if a fact is true in the egraph.
        """
        self._egraph.run_program(self._facts_to_check(facts))

    def check_fail(self, *facts: FactLike) -> None:
        """
        Checks that one of the facts is not true
        """
        self._egraph.run_program(bindings.Fail(bindings.DUMMY_SPAN, self._facts_to_check(facts)))

    def _facts_to_check(self, fact_likes: Iterable[FactLike]) -> bindings.Check:
        facts = _fact_likes(fact_likes)
        self._add_decls(*facts)
        egg_facts = [self._state.fact_to_egg(f.fact) for f in _fact_likes(facts)]
        return bindings.Check(bindings.DUMMY_SPAN, egg_facts)

    @overload
    def extract(self, expr: EXPR, /, include_cost: Literal[False] = False) -> EXPR: ...

    @overload
    def extract(self, expr: EXPR, /, include_cost: Literal[True]) -> tuple[EXPR, int]: ...

    def extract(self, expr: EXPR, include_cost: bool = False) -> EXPR | tuple[EXPR, int]:
        """
        Extract the lowest cost expression from the egraph.
        """
        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr)
        typed_expr = runtime_expr.__egg_typed_expr__
        extract_report = self._run_extract(typed_expr, 0)

        if not isinstance(extract_report, bindings.Best):
            msg = "No extract report saved"
            raise ValueError(msg)  # noqa: TRY004
        (new_typed_expr,) = self._state.exprs_from_egg(extract_report.termdag, [extract_report.term], typed_expr.tp)

        res = cast(EXPR, RuntimeExpr.__from_values__(self.__egg_decls__, new_typed_expr))
        if include_cost:
            return res, extract_report.cost
        return res

    def extract_multiple(self, expr: EXPR, n: int) -> list[EXPR]:
        """
        Extract multiple expressions from the egraph.
        """
        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr)
        typed_expr = runtime_expr.__egg_typed_expr__

        extract_report = self._run_extract(typed_expr, n)
        if not isinstance(extract_report, bindings.Variants):
            msg = "Wrong extract report type"
            raise ValueError(msg)  # noqa: TRY004
        new_exprs = self._state.exprs_from_egg(extract_report.termdag, extract_report.terms, typed_expr.tp)
        return [cast(EXPR, RuntimeExpr.__from_values__(self.__egg_decls__, expr)) for expr in new_exprs]

    def _run_extract(self, typed_expr: TypedExprDecl, n: int) -> bindings._ExtractReport:
        expr = self._state.typed_expr_to_egg(typed_expr)
        self._egraph.run_program(
            bindings.ActionCommand(
                bindings.Extract(bindings.DUMMY_SPAN, expr, bindings.Lit(bindings.DUMMY_SPAN, bindings.Int(n)))
            )
        )
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
        self._egraph.run_program(bindings.Pop(bindings.DUMMY_SPAN, 1))
        self._state = self._state_stack.pop()

    def __enter__(self) -> Self:
        """
        Copy the egraph state, so that it can be reverted back to the original state at the end.

        Also sets the current egraph to this one.
        """
        self._token_stack.append(CURRENT_EGRAPH.set(self))
        self.push()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # noqa: ANN001
        CURRENT_EGRAPH.reset(self._token_stack.pop())
        self.pop()

    @overload
    def eval(self, expr: i64) -> int: ...

    @overload
    def eval(self, expr: f64) -> float: ...

    @overload
    def eval(self, expr: Bool) -> bool: ...

    @overload
    def eval(self, expr: String) -> str: ...

    @overload
    def eval(self, expr: PyObject) -> object: ...

    def eval(self, expr: Expr) -> object:
        """
        Evaluates the given expression (which must be a primitive type), returning the result.
        """
        runtime_expr = to_runtime_expr(expr)
        self._add_decls(runtime_expr)
        typed_expr = runtime_expr.__egg_typed_expr__
        egg_expr = self._state.typed_expr_to_egg(typed_expr)
        match typed_expr.tp:
            case JustTypeRef("i64"):
                return self._egraph.eval_i64(egg_expr)
            case JustTypeRef("f64"):
                return self._egraph.eval_f64(egg_expr)
            case JustTypeRef("Bool"):
                return self._egraph.eval_bool(egg_expr)
            case JustTypeRef("String"):
                return self._egraph.eval_string(egg_expr)
            case JustTypeRef("PyObject"):
                return self._egraph.eval_py_object(egg_expr)
        raise TypeError(f"Eval not implemented for {typed_expr.tp}")

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
        **kwargs: Unpack[GraphvizKwargs],
    ) -> None:
        """
        Saturate the egraph, running the given schedule until the egraph is saturated.
        It serializes the egraph at each step and returns a widget to visualize the egraph.
        """
        from .visualizer_widget import VisualizerWidget

        def to_json() -> str:
            if expr:
                print(self.extract(expr))
            return self._serialize(**kwargs).to_json()

        egraphs = [to_json()]
        i = 0
        while self.run(schedule or 1).updated and i < max:
            i += 1
            egraphs.append(to_json())
        VisualizerWidget(egraphs=egraphs).display_or_open()

    @classmethod
    def current(cls) -> EGraph:
        """
        Returns the current egraph, which is the one in the context.
        """
        try:
            return CURRENT_EGRAPH.get()
        except LookupError:
            return cls(save_egglog_string=True)

    @property
    def _egraph(self) -> bindings.EGraph:
        return self._state.egraph

    @property
    def __egg_decls__(self) -> Declarations:
        return self._state.__egg_decls__

    def _register_commands(self, cmds: list[Command]) -> None:
        self._add_decls(*cmds)
        egg_cmds = list(map(self._command_to_egg, cmds))
        self._egraph.run_program(*egg_cmds)

    def _command_to_egg(self, cmd: Command) -> bindings._Command:
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


CURRENT_EGRAPH = ContextVar[EGraph]("CURRENT_EGRAPH")


@dataclass(frozen=True)
class _WrappedMethod(Generic[P, EXPR]):
    """
    Used to wrap a method and store some extra options on it before processing it when processing the class.
    """

    egg_fn: str | None
    cost: int | None
    default: EXPR | None
    merge: Callable[[EXPR, EXPR], EXPR] | None
    on_merge: Callable[[EXPR, EXPR], Iterable[ActionLike]] | None
    fn: Callable[P, EXPR]
    preserve: bool
    mutates_self: bool
    unextractable: bool

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> EXPR:
        msg = "We should never call a wrapped method. Did you forget to wrap the class?"
        raise NotImplementedError(msg)


class Expr(metaclass=_ExprMetaclass):
    """
    Either a function called with some number of argument expressions or a literal integer, float, or string, with a particular type.
    """

    def __ne__(self, other: NoReturn) -> NoReturn:  # type: ignore[override, empty-body]
        ...

    def __eq__(self, other: NoReturn) -> NoReturn:  # type: ignore[override, empty-body]
        ...


class Unit(Expr, egg_sort="Unit", builtin=True):
    """
    The unit type. This is also used to reprsent if a value exists, if it is resolved or not.
    """

    def __init__(self) -> None: ...


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
        self.__egg_decls_thunk__ = Thunk.fn(self._create_egg_decls, *rulesets)

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


@dataclass
class Action:
    """
    A change to an EGraph, either unioning multiple expressing, setting the value of a function call, deleting an expression, or panicking.
    """

    __egg_decls__: Declarations
    action: ActionDecl

    def __str__(self) -> str:
        return pretty_decl(self.__egg_decls__, self.action)

    def __repr__(self) -> str:
        return str(self)


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


@deprecated("Use <ruleset>.register(<rewrite>) instead of passing rulesets as arguments to rewrites.")
@overload
def rewrite(lhs: EXPR, ruleset: Ruleset, *, subsume: bool = False) -> _RewriteBuilder[EXPR]: ...


@overload
def rewrite(lhs: EXPR, ruleset: None = None, *, subsume: bool = False) -> _RewriteBuilder[EXPR]: ...


def rewrite(lhs: EXPR, ruleset: Ruleset | None = None, *, subsume: bool = False) -> _RewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression."""
    return _RewriteBuilder(lhs, ruleset, subsume)


@deprecated("Use <ruleset>.register(<birewrite>) instead of passing rulesets as arguments to birewrites.")
@overload
def birewrite(lhs: EXPR, ruleset: Ruleset) -> _BirewriteBuilder[EXPR]: ...


@overload
def birewrite(lhs: EXPR, ruleset: None = None) -> _BirewriteBuilder[EXPR]: ...


def birewrite(lhs: EXPR, ruleset: Ruleset | None = None) -> _BirewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression and vice versa."""
    return _BirewriteBuilder(lhs, ruleset)


def eq(expr: EXPR) -> _EqBuilder[EXPR]:
    """Check if the given expression is equal to the given value."""
    return _EqBuilder(expr)


def ne(expr: EXPR) -> _NeBuilder[EXPR]:
    """Check if the given expression is not equal to the given value."""
    return _NeBuilder(expr)


def panic(message: str) -> Action:
    """Raise an error with the given message."""
    return Action(Declarations(), PanicDecl(message))


def let(name: str, expr: Expr) -> Action:
    """Create a let binding."""
    runtime_expr = to_runtime_expr(expr)
    return Action(runtime_expr.__egg_decls__, LetDecl(name, runtime_expr.__egg_typed_expr__))


def expr_action(expr: Expr) -> Action:
    runtime_expr = to_runtime_expr(expr)
    return Action(runtime_expr.__egg_decls__, ExprActionDecl(runtime_expr.__egg_typed_expr__))


def delete(expr: Expr) -> Action:
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


def expr_fact(expr: Expr) -> Fact:
    runtime_expr = to_runtime_expr(expr)
    return Fact(runtime_expr.__egg_decls__, ExprFactDecl(runtime_expr.__egg_typed_expr__))


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    """Create a union of the given expression."""
    return _UnionBuilder(lhs=lhs)


def set_(lhs: EXPR) -> _SetBuilder[EXPR]:
    """Create a set of the given expression."""
    return _SetBuilder(lhs=lhs)


@deprecated("Use <ruleset>.register(<rule>) instead of passing rulesets as arguments to rules.")
@overload
def rule(*facts: FactLike, ruleset: Ruleset, name: str | None = None) -> _RuleBuilder: ...


@overload
def rule(*facts: FactLike, ruleset: None = None, name: str | None = None) -> _RuleBuilder: ...


def rule(*facts: FactLike, ruleset: Ruleset | None = None, name: str | None = None) -> _RuleBuilder:
    """Create a rule with the given facts."""
    return _RuleBuilder(facts=_fact_likes(facts), name=name, ruleset=ruleset)


@deprecated("This function is now a no-op, you can remove it and use actions as commands")
def action_command(action: Action) -> Action:
    return action


def var(name: str, bound: type[T]) -> T:
    """Create a new variable with the given name and type."""
    return cast(T, _var(name, bound))


def _var(name: str, bound: object) -> RuntimeExpr:
    """Create a new variable with the given name and type."""
    decls = Declarations()
    type_ref = resolve_type_annotation(decls, bound)
    return RuntimeExpr.__from_values__(decls, TypedExprDecl(type_ref.to_just(), VarDecl(name, False)))


def vars_(names: str, bound: type[EXPR]) -> Iterable[EXPR]:
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
class _EqBuilder(Generic[EXPR]):
    expr: EXPR

    def to(self, *exprs: EXPR) -> Fact:
        expr = to_runtime_expr(self.expr)
        args = [expr, *(convert_to_same_type(e, expr) for e in exprs)]
        return Fact(
            Declarations.create(*args),
            EqDecl(expr.__egg_typed_expr__.tp, tuple(a.__egg_typed_expr__.expr for a in args)),
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = to_runtime_expr(self.expr)
        return expr.__egg_pretty__("eq")


@dataclass
class _NeBuilder(Generic[EXPR]):
    lhs: EXPR

    def to(self, rhs: EXPR) -> Unit:
        lhs = to_runtime_expr(self.lhs)
        rhs = convert_to_same_type(rhs, lhs)
        assert isinstance(Unit, RuntimeClass)
        res = RuntimeExpr.__from_values__(
            Declarations.create(Unit, lhs, rhs),
            TypedExprDecl(
                JustTypeRef("Unit"), CallDecl(FunctionRef("!="), (lhs.__egg_typed_expr__, rhs.__egg_typed_expr__))
            ),
        )
        return cast(Unit, res)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = to_runtime_expr(self.lhs)
        return expr.__egg_pretty__("ne")


@dataclass
class _SetBuilder(Generic[EXPR]):
    lhs: EXPR

    def to(self, rhs: EXPR) -> Action:
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


def expr_parts(expr: Expr) -> TypedExprDecl:
    """
    Returns the underlying type and decleration of the expression. Useful for testing structural equality or debugging.
    """
    if not isinstance(expr, RuntimeExpr):
        raise TypeError(f"Expected a RuntimeExpr not {expr}")
    return expr.__egg_typed_expr__


def to_runtime_expr(expr: Expr) -> RuntimeExpr:
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


ActionLike: TypeAlias = Action | Expr


def _action_likes(action_likes: Iterable[ActionLike]) -> tuple[Action, ...]:
    return tuple(map(_action_like, action_likes))


def _action_like(action_like: ActionLike) -> Action:
    if isinstance(action_like, Expr):
        return expr_action(action_like)
    return action_like


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
    # Get the local scope from where the function is defined, so that we can get any type hints that are in the scope
    # but not in the globals
    globals = gen.__globals__.copy()
    if "Callable" not in globals:
        globals["Callable"] = Callable
    hints = get_type_hints(gen, globals, frame.f_locals)
    args = [_var(p.name, hints[p.name]) for p in signature(gen).parameters.values()]
    return list(gen(*args))  # type: ignore[misc]


FactLike = Fact | Expr


def _fact_likes(fact_likes: Iterable[FactLike]) -> tuple[Fact, ...]:
    return tuple(map(_fact_like, fact_likes))


def _fact_like(fact_like: FactLike) -> Fact:
    if isinstance(fact_like, Expr):
        return expr_fact(fact_like)
    return fact_like


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
