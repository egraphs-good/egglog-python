from __future__ import annotations

import inspect
import pathlib
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from inspect import Parameter, currentframe, signature
from types import FrameType, FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    TypedDict,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

import graphviz
from typing_extensions import ParamSpec, Self, Unpack, deprecated

from egglog.declarations import REFLECTED_BINARY_METHODS, Declarations

from . import bindings
from .declarations import *
from .ipython_magic import IN_IPYTHON
from .runtime import *

if TYPE_CHECKING:
    import ipywidgets

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
    "union",
    "set_",
    "rule",
    "var",
    "vars_",
    "Fact",
    "expr_parts",
    "Schedule",
    "run",
    "seq",
    "Command",
    "simplify",
    "check",
    "GraphvizKwargs",
    "Ruleset",
    "_RewriteBuilder",
    "_BirewriteBuilder",
    "_EqBuilder",
    "_NeBuilder",
    "_SetBuilder",
    "_UnionBuilder",
    "Rule",
    "Rewrite",
    "BiRewrite",
    "Union_",
    "Action",
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


ALWAYS_MUTATES_SELF = {"__setitem__", "__delitem__"}


def simplify(x: EXPR, schedule: Schedule | None = None) -> EXPR:
    """
    Simplify an expression by running the schedule.
    """
    if schedule:
        return EGraph().simplify(x, schedule)
    return EGraph().extract(x)


def check(x: FactLike, schedule: Schedule | None = None, *given: Union_ | Expr | Set) -> None:
    """
    Verifies that the fact is true given some assumptions and after running the schedule.
    """
    egraph = EGraph()
    if given:
        egraph.register(*given)
    if schedule:
        egraph.run(schedule)
    egraph.check(x)


# def extract(res: )


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
    def class_(self, *, egg_sort: str) -> Callable[[TYPE], TYPE]:
        ...

    @deprecated("Remove this decorator. Simply subclassing Expr is enough now.")
    @overload
    def class_(self, cls: TYPE, /) -> TYPE:
        ...

    def class_(self, *args, **kwargs) -> Any:
        """
        Registers a class.
        """
        if kwargs:
            assert set(kwargs.keys()) == {"egg_sort"}

            def _inner(cls: object, egg_sort: str = kwargs["egg_sort"]):
                assert isinstance(cls, RuntimeClass)
                assert isinstance(cls.lazy_decls, _ClassDeclerationsConstructor)
                cls.lazy_decls.egg_sort = egg_sort
                return cls

            return _inner

        assert len(args) == 1
        return args[0]

    @overload
    def method(
        self,
        *,
        preserve: Literal[True],
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

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
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

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
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

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
    def function(self, fn: CALLABLE, /) -> CALLABLE:
        ...

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
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

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
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    @deprecated("Use top level function `function` instead")
    def function(self, *args, **kwargs) -> Any:
        """
        Registers a function.
        """
        fn_locals = currentframe().f_back.f_back.f_locals  # type: ignore[union-attr]
        # If we have any positional args, then we are calling it directly on a function
        if args:
            assert len(args) == 1
            return _function(args[0], fn_locals, False)
        # otherwise, we are passing some keyword args, so save those, and then return a partial
        return lambda fn: _function(fn, fn_locals, False, **kwargs)

    @deprecated("Use top level `ruleset` function instead")
    def ruleset(self, name: str) -> Ruleset:
        return Ruleset(name)

    # Overload to support aritys 0-4 until variadic generic support map, so we can map from type to value
    @overload
    def relation(
        self, name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], tp4: type[E4], /
    ) -> Callable[[E1, E2, E3, E4], Unit]:
        ...

    @overload
    def relation(self, name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], /) -> Callable[[E1, E2, E3], Unit]:
        ...

    @overload
    def relation(self, name: str, tp1: type[E1], tp2: type[E2], /) -> Callable[[E1, E2], Unit]:
        ...

    @overload
    def relation(self, name: str, tp1: type[T], /, *, egg_fn: str | None = None) -> Callable[[T], Unit]:
        ...

    @overload
    def relation(self, name: str, /, *, egg_fn: str | None = None) -> Callable[[], Unit]:
        ...

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
        return constant(name, tp, egg_name)

    def register(self, /, command_or_generator: CommandLike | CommandGenerator, *command_likes: CommandLike) -> None:
        """
        Registers any number of rewrites or rules.
        """
        if isinstance(command_or_generator, FunctionType):
            assert not command_likes
            command_likes = tuple(_command_generator(command_or_generator))
        else:
            command_likes = (cast(CommandLike, command_or_generator), *command_likes)

        self._register_commands(list(map(_command_like, command_likes)))

    @abstractmethod
    def _register_commands(self, cmds: list[Command]) -> None:
        raise NotImplementedError


# We seperate the function and method overloads to make it simpler to know if we are modifying a function or method,
# So that we can add the functions eagerly to the registry and wait on the methods till we process the class.


@overload
def method(
    *,
    preserve: Literal[True],
) -> Callable[[CALLABLE], CALLABLE]:
    ...


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
) -> Callable[[CALLABLE], CALLABLE]:
    ...


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
) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
    ...


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
    ) -> RuntimeClass | type:
        # If this is the Expr subclass, just return the class
        if not bases:
            return super().__new__(cls, name, bases, namespace)

        frame = currentframe()
        assert frame
        prev_frame = frame.f_back
        assert prev_frame
        return _ClassDeclerationsConstructor(
            namespace=namespace,
            # Store frame so that we can get live access to updated locals/globals
            # Otherwise, f_locals returns a copy
            # https://peps.python.org/pep-0667/
            frame=prev_frame,
            builtin=builtin,
            egg_sort=egg_sort,
            cls_name=name,
        ).current_cls

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, RuntimeExpr)


@dataclass
class _ClassDeclerationsConstructor:
    """
    Lazy constructor for class declerations to support classes with methods whose types are not yet defined.
    """

    namespace: dict[str, Any]
    frame: FrameType
    builtin: bool
    egg_sort: str | None
    cls_name: str
    current_cls: RuntimeClass = field(init=False)

    def __post_init__(self) -> None:
        self.current_cls = RuntimeClass(self, self.cls_name)

    def __call__(self, decls: Declarations) -> None:  # noqa: PLR0912
        # Get all the methods from the class
        cls_dict: dict[str, Any] = {
            k: v for k, v in self.namespace.items() if k not in IGNORED_ATTRIBUTES or isinstance(v, _WrappedMethod)
        }
        parameters: list[TypeVar] = (
            # Get the generic params from the orig bases generic class
            self.namespace["__orig_bases__"][1].__parameters__ if "__orig_bases__" in self.namespace else []
        )
        type_vars = tuple(p.__name__ for p in parameters)
        del parameters

        decls.register_class(self.cls_name, type_vars, self.builtin, self.egg_sort)
        # The type ref of self is paramterized by the type vars
        slf_type_ref = TypeRefWithVars(self.cls_name, tuple(map(ClassTypeVarRef, type_vars)))

        # Create a dummy type to pass to get_type_hints to resolve the annotations we have
        class _Dummytype:
            pass

        _Dummytype.__annotations__ = self.namespace.get("__annotations__", {})
        # Make lazy update to locals, so we keep a live handle on them after class creation
        locals = self.frame.f_locals.copy()
        locals[self.cls_name] = self.current_cls
        for k, v in get_type_hints(_Dummytype, globalns=self.frame.f_globals, localns=locals).items():
            if v.__origin__ == ClassVar:
                (inner_tp,) = v.__args__
                _register_constant(decls, ClassVariableRef(self.cls_name, k), inner_tp, None)
            else:
                msg = "The only supported annotations on class attributes are class vars"
                raise NotImplementedError(msg)

        # Then register each of its methods
        for method_name, method in cls_dict.items():
            is_init = method_name == "__init__"
            # Don't register the init methods for literals, since those don't use the type checking mechanisms
            if is_init and self.cls_name in LIT_CLASS_NAMES:
                continue
            if isinstance(method, _WrappedMethod):
                fn = method.fn
                egg_fn = method.egg_fn
                cost = method.cost
                default = method.default
                merge = method.merge
                on_merge = method.on_merge
                mutates_first_arg = method.mutates_self
                unextractable = method.unextractable
                if method.preserve:
                    decls.register_preserved_method(self.cls_name, method_name, fn)
                    continue
            else:
                fn = method
                egg_fn, cost, default, merge, on_merge = None, None, None, None, None
                unextractable = False
                mutates_first_arg = False
            if isinstance(fn, classmethod):
                fn = fn.__func__
                is_classmethod = True
            else:
                # We count __init__ as a classmethod since it is called on the class
                is_classmethod = is_init

            if isinstance(fn, property):
                fn = fn.fget
                is_property = True
                if is_classmethod:
                    msg = "Can't have a classmethod property"
                    raise NotImplementedError(msg)
            else:
                is_property = False
            ref: FunctionCallableRef = (
                ClassMethodRef(self.cls_name, method_name)
                if is_classmethod
                else PropertyRef(self.cls_name, method_name)
                if is_property
                else MethodRef(self.cls_name, method_name)
            )
            _register_function(
                decls,
                ref,
                egg_fn,
                fn,
                locals,
                default,
                cost,
                merge,
                on_merge,
                mutates_first_arg or method_name in ALWAYS_MUTATES_SELF,
                self.builtin,
                "cls" if is_classmethod and not is_init else slf_type_ref,
                is_init,
                unextractable=unextractable,
            )


@overload
def function(fn: CALLABLE, /) -> CALLABLE:
    ...


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
) -> Callable[[CALLABLE], CALLABLE]:
    ...


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
) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
    ...


def function(*args, **kwargs) -> Any:
    """
    Defined by a unique name and a typing relation that will specify the return type based on the types of the argument expressions.


    """
    fn_locals = currentframe().f_back.f_locals  # type: ignore[union-attr]

    # If we have any positional args, then we are calling it directly on a function
    if args:
        assert len(args) == 1
        return _function(args[0], fn_locals, False)
    # otherwise, we are passing some keyword args, so save those, and then return a partial
    return lambda fn: _function(fn, fn_locals, **kwargs)


def _function(
    fn: Callable[..., RuntimeExpr],
    hint_locals: dict[str, Any],
    builtin: bool = False,
    mutates_first_arg: bool = False,
    egg_fn: str | None = None,
    cost: int | None = None,
    default: RuntimeExpr | None = None,
    merge: Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr] | None = None,
    on_merge: Callable[[RuntimeExpr, RuntimeExpr], Iterable[ActionLike]] | None = None,
    unextractable: bool = False,
) -> RuntimeFunction:
    """
    Uncurried version of function decorator
    """
    name = fn.__name__
    decls = Declarations()
    _register_function(
        decls,
        FunctionRef(name),
        egg_fn,
        fn,
        hint_locals,
        default,
        cost,
        merge,
        on_merge,
        mutates_first_arg,
        builtin,
        unextractable=unextractable,
    )
    return RuntimeFunction(decls, name)


def _register_function(
    decls: Declarations,
    ref: FunctionCallableRef,
    egg_name: str | None,
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
    # The first arg is either cls, for a classmethod, a self type, or none for a function
    first_arg: Literal["cls"] | TypeOrVarRef | None = None,
    is_init: bool = False,
    unextractable: bool = False,
) -> None:
    if not isinstance(fn, FunctionType):
        raise NotImplementedError(f"Can only generate function decls for functions not {fn}  {type(fn)}")

    hint_globals = fn.__globals__.copy()

    hints = get_type_hints(fn, hint_globals, hint_locals)

    params = list(signature(fn).parameters.values())

    # If this is an init function, or a classmethod, remove the first arg name
    if is_init or first_arg == "cls":
        params = params[1:]

    if _last_param_variable(params):
        *params, var_arg_param = params
        # For now, we don't use the variable arg name
        var_arg_type = resolve_type_annotation(decls, hints[var_arg_param.name])
    else:
        var_arg_type = None
    arg_types = tuple(
        first_arg
        # If the first arg is a self, and this not an __init__ fn, add this as a typeref
        if i == 0 and isinstance(first_arg, ClassTypeVarRef | TypeRefWithVars) and not is_init
        else resolve_type_annotation(decls, hints[t.name])
        for i, t in enumerate(params)
    )

    # Resolve all default values as arg types
    arg_defaults = [
        resolve_literal(t, p.default) if p.default is not Parameter.empty else None
        for (t, p) in zip(arg_types, params, strict=True)
    ]

    decls.update(*arg_defaults)

    # If this is an init fn use the first arg as the return type
    if is_init:
        assert not mutates_first_arg
        if not isinstance(first_arg, ClassTypeVarRef | TypeRefWithVars):
            msg = "Init function must have a self type"
            raise ValueError(msg)
        return_type = first_arg
    elif mutates_first_arg:
        return_type = arg_types[0]
    else:
        return_type = resolve_type_annotation(decls, hints["return"])

    decls |= default
    merged = (
        None
        if merge is None
        else merge(
            RuntimeExpr(decls, TypedExprDecl(return_type.to_just(), VarDecl("old"))),
            RuntimeExpr(decls, TypedExprDecl(return_type.to_just(), VarDecl("new"))),
        )
    )
    decls |= merged

    merge_action = (
        []
        if on_merge is None
        else _action_likes(
            on_merge(
                RuntimeExpr(decls, TypedExprDecl(return_type.to_just(), VarDecl("old"))),
                RuntimeExpr(decls, TypedExprDecl(return_type.to_just(), VarDecl("new"))),
            )
        )
    )
    decls.update(*merge_action)
    fn_decl = FunctionDecl(
        return_type=return_type,
        var_arg_type=var_arg_type,
        arg_types=arg_types,
        arg_names=tuple(t.name for t in params),
        arg_defaults=tuple(a.__egg_typed_expr__.expr if a is not None else None for a in arg_defaults),
        mutates_first_arg=mutates_first_arg,
    )
    decls.register_function_callable(
        ref,
        fn_decl,
        egg_name,
        cost,
        None if default is None else default.__egg_typed_expr__.expr,
        merged.__egg_typed_expr__.expr if merged is not None else None,
        [a._to_egg_action() for a in merge_action],
        unextractable,
        is_builtin,
    )


# Overload to support aritys 0-4 until variadic generic support map, so we can map from type to value
@overload
def relation(
    name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], tp4: type[E4], /
) -> Callable[[E1, E2, E3, E4], Unit]:
    ...


@overload
def relation(name: str, tp1: type[E1], tp2: type[E2], tp3: type[E3], /) -> Callable[[E1, E2, E3], Unit]:
    ...


@overload
def relation(name: str, tp1: type[E1], tp2: type[E2], /) -> Callable[[E1, E2], Unit]:
    ...


@overload
def relation(name: str, tp1: type[T], /, *, egg_fn: str | None = None) -> Callable[[T], Unit]:
    ...


@overload
def relation(name: str, /, *, egg_fn: str | None = None) -> Callable[[], Unit]:
    ...


def relation(name: str, /, *tps: type, egg_fn: str | None = None) -> Callable[..., Unit]:
    """
    Creates a function whose return type is `Unit` and has a default value.
    """
    decls = Declarations()
    decls |= cast(RuntimeClass, Unit)
    arg_types = tuple(resolve_type_annotation(decls, tp) for tp in tps)
    fn_decl = FunctionDecl(arg_types, None, tuple(None for _ in tps), TypeRefWithVars("Unit"), mutates_first_arg=False)
    decls.register_function_callable(
        FunctionRef(name),
        fn_decl,
        egg_fn,
        cost=None,
        default=None,
        merge=None,
        merge_action=[],
        unextractable=False,
        builtin=False,
        is_relation=True,
    )
    return cast(Callable[..., Unit], RuntimeFunction(decls, name))


def constant(name: str, tp: type[EXPR], egg_name: str | None = None) -> EXPR:
    """

    A "constant" is implemented as the instantiation of a value that takes no args.
    This creates a function with `name` and return type `tp` and returns a value of it being called.
    """
    ref = ConstantRef(name)
    decls = Declarations()
    type_ref = _register_constant(decls, ref, tp, egg_name)
    return cast(EXPR, RuntimeExpr(decls, TypedExprDecl(type_ref, CallDecl(ref))))


def _register_constant(
    decls: Declarations,
    ref: ConstantRef | ClassVariableRef,
    tp: object,
    egg_name: str | None,
) -> JustTypeRef:
    """
    Register a constant, returning its typeref().
    """
    type_ref = resolve_type_annotation(decls, tp).to_just()
    decls.register_constant_callable(ref, type_ref, egg_name)
    return type_ref


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


@dataclass
class _EGraphState:
    """
    State of the EGraph declerations and rulesets, so when we pop/push the stack we know whats defined.
    """

    # The decleratons we have added. The _cmds represent all the symbols we have added
    decls: Declarations = field(default_factory=Declarations)
    # List of rulesets already added, so we don't re-add them if they are passed again
    added_rulesets: set[str] = field(default_factory=set)

    def add_decls(self, new_decls: Declarations) -> Iterable[bindings._Command]:
        new_cmds = [v for k, v in new_decls._cmds.items() if k not in self.decls._cmds]
        self.decls |= new_decls
        return new_cmds

    def add_rulesets(self, rulesets: Iterable[Ruleset]) -> Iterable[bindings._Command]:
        for ruleset in rulesets:
            if ruleset.egg_name not in self.added_rulesets:
                self.added_rulesets.add(ruleset.egg_name)
                yield from ruleset._cmds


@dataclass
class EGraph(_BaseModule):
    """
    A collection of expressions where each expression is part of a distinct equivalence class.

    Can run actions, check facts, run schedules, or extract minimal cost expressions.
    """

    seminaive: InitVar[bool] = True
    save_egglog_string: InitVar[bool] = False

    default_ruleset: Ruleset | None = None
    _egraph: bindings.EGraph = field(repr=False, init=False)
    _egglog_string: str | None = field(default=None, repr=False, init=False)
    _state: _EGraphState = field(default_factory=_EGraphState, repr=False)
    # For pushing/popping with egglog
    _state_stack: list[_EGraphState] = field(default_factory=list, repr=False)
    # For storing the global "current" egraph
    _token_stack: list[Token[EGraph]] = field(default_factory=list, repr=False)

    def __post_init__(self, modules: list[Module], seminaive: bool, save_egglog_string: bool) -> None:
        self._egraph = bindings.EGraph(GLOBAL_PY_OBJECT_SORT, seminaive=seminaive)
        super().__post_init__(modules)

        for m in self._flatted_deps:
            self._add_decls(*m.cmds)
            self._register_commands(m.cmds)
        if save_egglog_string:
            self._egglog_string = ""

    def _register_commands(self, commands: list[Command]) -> None:
        for c in commands:
            if c.ruleset:
                self._add_schedule(c.ruleset)

        self._add_decls(*commands)
        self._process_commands(command._to_egg_command(self._default_ruleset_name) for command in commands)

    def _process_commands(self, commands: Iterable[bindings._Command]) -> None:
        commands = list(commands)
        self._egraph.run_program(*commands)
        if isinstance(self._egglog_string, str):
            self._egglog_string += "\n".join(str(c) for c in commands) + "\n"

    def _add_decls(self, *decls: DeclerationsLike) -> None:
        for d in upcast_decleratioons(decls):
            self._process_commands(self._state.add_decls(d))

    def _add_schedule(self, schedule: Schedule) -> None:
        self._add_decls(schedule)
        self._process_commands(self._state.add_rulesets(schedule._rulesets()))

    @property
    def as_egglog_string(self) -> str:
        """
        Returns the egglog string for this module.
        """
        if self._egglog_string is None:
            msg = "Can't get egglog string unless EGraph created with save_egglog_string=True"
            raise ValueError(msg)
        return self._egglog_string

    def _repr_mimebundle_(self, *args, **kwargs):
        """
        Returns the graphviz representation of the e-graph.
        """
        return {"image/svg+xml": self.graphviz().pipe(format="svg", quiet=True, encoding="utf-8")}

    def graphviz(self, **kwargs: Unpack[GraphvizKwargs]) -> graphviz.Source:
        # By default we want to split primitive outputs
        kwargs.setdefault("split_primitive_outputs", True)
        n_inline = kwargs.pop("n_inline_leaves", 0)
        serialized = self._egraph.serialize(**kwargs)  # type: ignore[misc]
        serialized.map_ops(self._state.decls.op_mapping())
        for _ in range(n_inline):
            serialized.inline_leaves()
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

    def graphviz_svg(self, **kwargs: Unpack[GraphvizKwargs]) -> str:
        return self.graphviz(**kwargs).pipe(format="svg", quiet=True, encoding="utf-8")

    def _repr_html_(self) -> str:
        """
        Add a _repr_html_ to be an SVG to work with sphinx gallery.

        ala https://github.com/xflr6/graphviz/pull/121
        until this PR is merged and released
        https://github.com/sphinx-gallery/sphinx-gallery/pull/1138
        """
        return self.graphviz_svg()

    def display(self, **kwargs: Unpack[GraphvizKwargs]) -> None:
        """
        Displays the e-graph in the notebook.
        """
        graphviz = self.graphviz(**kwargs)
        if IN_IPYTHON:
            from IPython.display import SVG, display

            display(SVG(self.graphviz_svg(**kwargs)))
        else:
            graphviz.render(view=True, format="svg", quiet=True)

    def input(self, fn: Callable[..., String], path: str) -> None:
        """
        Loads a CSV file and sets it as *input, output of the function.
        """
        ref, decls = resolve_callable(fn)
        fn_name = decls.get_egg_fn(ref)
        self._process_commands(decls.list_cmds())
        self._process_commands([bindings.Input(fn_name, path)])

    def let(self, name: str, expr: EXPR) -> EXPR:
        """
        Define a new expression in the egraph and return a reference to it.
        """
        self._register_commands([let(name, expr)])
        expr = to_runtime_expr(expr)
        return cast(EXPR, RuntimeExpr(expr.__egg_decls__, TypedExprDecl(expr.__egg_typed_expr__.tp, VarDecl(name))))

    @overload
    def simplify(self, expr: EXPR, limit: int, /, *until: Fact, ruleset: Ruleset | None = None) -> EXPR:
        ...

    @overload
    def simplify(self, expr: EXPR, schedule: Schedule, /) -> EXPR:
        ...

    def simplify(
        self, expr: EXPR, limit_or_schedule: int | Schedule, /, *until: Fact, ruleset: Ruleset | None = None
    ) -> EXPR:
        """
        Simplifies the given expression.
        """
        schedule = run(ruleset, *until) * limit_or_schedule if isinstance(limit_or_schedule, int) else limit_or_schedule
        del limit_or_schedule
        expr = to_runtime_expr(expr)
        self._add_decls(expr)
        self._add_schedule(schedule)

        # decls = Declarations.create(expr, schedule)
        self._process_commands([bindings.Simplify(expr.__egg__, schedule._to_egg_schedule(self._default_ruleset_name))])
        extract_report = self._egraph.extract_report()
        if not isinstance(extract_report, bindings.Best):
            msg = "No extract report saved"
            raise ValueError(msg)  # noqa: TRY004
        new_typed_expr = TypedExprDecl.from_egg(
            self._egraph, self._state.decls, expr.__egg_typed_expr__.tp, extract_report.termdag, extract_report.term, {}
        )
        return cast(EXPR, RuntimeExpr(self._state.decls, new_typed_expr))

    @property
    def _default_ruleset_name(self) -> str:
        if self.default_ruleset:
            self._add_schedule(self.default_ruleset)
            return self.default_ruleset.egg_name
        return ""

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
    def run(self, limit: int, /, *until: Fact, ruleset: Ruleset | None = None) -> bindings.RunReport:
        ...

    @overload
    def run(self, schedule: Schedule, /) -> bindings.RunReport:
        ...

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
        self._add_schedule(schedule)
        self._process_commands([bindings.RunSchedule(schedule._to_egg_schedule(self._default_ruleset_name))])
        run_report = self._egraph.run_report()
        if not run_report:
            msg = "No run report saved"
            raise ValueError(msg)
        return run_report

    def check(self, *facts: FactLike) -> None:
        """
        Check if a fact is true in the egraph.
        """
        self._process_commands([self._facts_to_check(facts)])

    def check_fail(self, *facts: FactLike) -> None:
        """
        Checks that one of the facts is not true
        """
        self._process_commands([bindings.Fail(self._facts_to_check(facts))])

    def _facts_to_check(self, facts: Iterable[FactLike]) -> bindings.Check:
        facts = _fact_likes(facts)
        self._add_decls(*facts)
        egg_facts = [f._to_egg_fact() for f in _fact_likes(facts)]
        return bindings.Check(egg_facts)

    @overload
    def extract(self, expr: EXPR, /, include_cost: Literal[False] = False) -> EXPR:
        ...

    @overload
    def extract(self, expr: EXPR, /, include_cost: Literal[True]) -> tuple[EXPR, int]:
        ...

    def extract(self, expr: EXPR, include_cost: bool = False) -> EXPR | tuple[EXPR, int]:
        """
        Extract the lowest cost expression from the egraph.
        """
        assert isinstance(expr, RuntimeExpr)
        self._add_decls(expr)
        extract_report = self._run_extract(expr.__egg__, 0)
        if not isinstance(extract_report, bindings.Best):
            msg = "No extract report saved"
            raise ValueError(msg)  # noqa: TRY004
        new_typed_expr = TypedExprDecl.from_egg(
            self._egraph, self._state.decls, expr.__egg_typed_expr__.tp, extract_report.termdag, extract_report.term, {}
        )
        res = cast(EXPR, RuntimeExpr(self._state.decls, new_typed_expr))
        if include_cost:
            return res, extract_report.cost
        return res

    def extract_multiple(self, expr: EXPR, n: int) -> list[EXPR]:
        """
        Extract multiple expressions from the egraph.
        """
        assert isinstance(expr, RuntimeExpr)
        self._add_decls(expr)

        extract_report = self._run_extract(expr.__egg__, n)
        if not isinstance(extract_report, bindings.Variants):
            msg = "Wrong extract report type"
            raise ValueError(msg)  # noqa: TRY004
        new_exprs = [
            TypedExprDecl.from_egg(
                self._egraph, self._state.decls, expr.__egg_typed_expr__.tp, extract_report.termdag, term, {}
            )
            for term in extract_report.terms
        ]
        return [cast(EXPR, RuntimeExpr(self._state.decls, expr)) for expr in new_exprs]

    def _run_extract(self, expr: bindings._Expr, n: int) -> bindings._ExtractReport:
        self._process_commands([bindings.ActionCommand(bindings.Extract(expr, bindings.Lit(bindings.Int(n))))])
        extract_report = self._egraph.extract_report()
        if not extract_report:
            msg = "No extract report saved"
            raise ValueError(msg)
        return extract_report

    def push(self) -> None:
        """
        Push the current state of the egraph, so that it can be popped later and reverted back.
        """
        self._process_commands([bindings.Push(1)])
        self._state_stack.append(self._state)
        self._state = deepcopy(self._state)

    def pop(self) -> None:
        """
        Pop the current state of the egraph, reverting back to the previous state.
        """
        self._process_commands([bindings.Pop(1)])
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
    def eval(self, expr: i64) -> int:
        ...

    @overload
    def eval(self, expr: f64) -> float:
        ...

    @overload
    def eval(self, expr: Bool) -> bool:
        ...

    @overload
    def eval(self, expr: String) -> str:
        ...

    @overload
    def eval(self, expr: PyObject) -> object:
        ...

    def eval(self, expr: Expr) -> object:
        """
        Evaluates the given expression (which must be a primitive type), returning the result.
        """
        assert isinstance(expr, RuntimeExpr)
        typed_expr = expr.__egg_typed_expr__
        egg_expr = expr.__egg__
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
        raise NotImplementedError(f"Eval not implemented for {typed_expr.tp.name}")

    def saturate(
        self, *, max: int = 1000, performance: bool = False, **kwargs: Unpack[GraphvizKwargs]
    ) -> ipywidgets.Widget:
        from .graphviz_widget import graphviz_widget_with_slider

        dots = [str(self.graphviz(**kwargs))]
        i = 0
        while self.run(1).updated and i < max:
            i += 1
            dots.append(str(self.graphviz(**kwargs)))
        return graphviz_widget_with_slider(dots, performance=performance)

    def saturate_to_html(
        self, file: str = "tmp.html", performance: bool = False, **kwargs: Unpack[GraphvizKwargs]
    ) -> None:
        # raise NotImplementedError("Upstream bugs prevent rendering to HTML")

        # import panel

        # panel.extension("ipywidgets")

        widget = self.saturate(performance=performance, **kwargs)
        # panel.panel(widget).save(file)

        from ipywidgets.embed import embed_minimal_html

        embed_minimal_html("tmp.html", views=[widget], drop_defaults=False)
        # Use panel while this issue persists
        # https://github.com/jupyter-widgets/ipywidgets/issues/3761#issuecomment-1755563436

    @classmethod
    def current(cls) -> EGraph:
        """
        Returns the current egraph, which is the one in the context.
        """
        return CURRENT_EGRAPH.get()


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

    def __init__(self) -> None:
        ...


def ruleset(
    rule_or_generator: CommandLike | CommandGenerator | None = None, *rules: Rule | Rewrite, name: None | str = None
) -> Ruleset:
    """
    Creates a ruleset with the following rules.

    If no name is provided, one is generated based on the current module
    """
    r = Ruleset(name=name)
    if rule_or_generator is not None:
        r.register(rule_or_generator, *rules)
    return r


class Schedule(ABC):
    """
    A composition of some rulesets, either composing them sequentially, running them repeatedly, running them till saturation, or running until some facts are met
    """

    def __mul__(self, length: int) -> Schedule:
        """
        Repeat the schedule a number of times.
        """
        return Repeat(length, self)

    def saturate(self) -> Schedule:
        """
        Run the schedule until the e-graph is saturated.
        """
        return Saturate(self)

    def __add__(self, other: Schedule) -> Schedule:
        """
        Run two schedules in sequence.
        """
        return Sequence((self, other))

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _to_egg_schedule(self, default_ruleset_name: str) -> bindings._Schedule:
        raise NotImplementedError

    @abstractmethod
    def _rulesets(self) -> Iterable[Ruleset]:
        """
        Mapping of all the rulesets used to commands.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def __egg_decls__(self) -> Declarations:
        raise NotImplementedError


@dataclass
class Ruleset(Schedule):
    """
    A collection of rules, which can be run as a schedule.
    """

    name: str | None
    rules: list[Rule | Rewrite] = field(default_factory=list)

    def append(self, rule: Rule | Rewrite) -> None:
        """
        Register a rule with the ruleset.
        """
        self.rules.append(rule)

    def register(self, /, rule_or_generator: CommandLike | CommandGenerator, *rules: Rule | Rewrite) -> None:
        """
        Register rewrites or rules, either as a function or as values.
        """
        if isinstance(rule_or_generator, FunctionType):
            assert not rules
            rules = tuple(_command_generator(rule_or_generator))
        else:
            rules = (cast(Rule | Rewrite, rule_or_generator), *rules)
        for r in rules:
            self.append(r)

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(*self.rules)

    @property
    def _cmds(self) -> list[bindings._Command]:
        cmds = [r._to_egg_command(self.egg_name) for r in self.rules]
        if self.egg_name:
            cmds.insert(0, bindings.AddRuleset(self.egg_name))
        return cmds

    def __str__(self) -> str:
        return f"ruleset(name={self.egg_name!r})"

    def __repr__(self) -> str:
        if not self.rules:
            return str(self)
        rules = ", ".join(map(repr, self.rules))
        return f"ruleset({rules}, name={self.egg_name!r})"

    def _to_egg_schedule(self, default_ruleset_name: str) -> bindings._Schedule:
        return bindings.Run(self._to_egg_config())

    def _to_egg_config(self) -> bindings.RunConfig:
        return bindings.RunConfig(self.egg_name, None)

    def _rulesets(self) -> Iterable[Ruleset]:
        yield self

    @property
    def egg_name(self) -> str:
        return self.name or f"_ruleset_{id(self)}"


class Command(ABC):
    """
    A command that can be executed in the egg interpreter.

    We only use this for commands which return no result and don't create new Python objects.

    Anything that can be passed to the `register` function in a Module is a Command.
    """

    ruleset: Ruleset | None

    @property
    @abstractmethod
    def __egg_decls__(self) -> Declarations:
        raise NotImplementedError

    @abstractmethod
    def _to_egg_command(self, default_ruleset_name: str) -> bindings._Command:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass
class Rewrite(Command):
    ruleset: Ruleset | None
    _lhs: RuntimeExpr
    _rhs: RuntimeExpr
    _conditions: tuple[Fact, ...]
    _fn_name: ClassVar[str] = "rewrite"

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self._rhs, *self._conditions]))
        return f"{self._fn_name}({self._lhs}).to({args_str})"

    def _to_egg_command(self, default_ruleset_name: str) -> bindings._Command:
        return bindings.RewriteCommand(
            self.ruleset.egg_name if self.ruleset else default_ruleset_name, self._to_egg_rewrite()
        )

    def _to_egg_rewrite(self) -> bindings.Rewrite:
        return bindings.Rewrite(
            self._lhs.__egg_typed_expr__.expr.to_egg(self._lhs.__egg_decls__),
            self._rhs.__egg_typed_expr__.expr.to_egg(self._rhs.__egg_decls__),
            [c._to_egg_fact() for c in self._conditions],
        )

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(self._lhs, self._rhs, *self._conditions)

    def with_ruleset(self, ruleset: Ruleset) -> Rewrite:
        return Rewrite(ruleset, self._lhs, self._rhs, self._conditions)


@dataclass
class BiRewrite(Rewrite):
    _fn_name: ClassVar[str] = "birewrite"

    def _to_egg_command(self, default_ruleset_name: str) -> bindings._Command:
        return bindings.BiRewriteCommand(
            self.ruleset.egg_name if self.ruleset else default_ruleset_name, self._to_egg_rewrite()
        )


@dataclass
class Fact(ABC):
    """
    A query on an EGraph, either by an expression or an equivalence between multiple expressions.
    """

    @abstractmethod
    def _to_egg_fact(self) -> bindings._Fact:
        raise NotImplementedError

    @property
    @abstractmethod
    def __egg_decls__(self) -> Declarations:
        raise NotImplementedError


@dataclass
class Eq(Fact):
    _exprs: list[RuntimeExpr]

    def __str__(self) -> str:
        first, *rest = self._exprs
        args_str = ", ".join(map(str, rest))
        return f"eq({first}).to({args_str})"

    def _to_egg_fact(self) -> bindings.Eq:
        return bindings.Eq([e.__egg__ for e in self._exprs])

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(*self._exprs)


@dataclass
class ExprFact(Fact):
    _expr: RuntimeExpr

    def __str__(self) -> str:
        return str(self._expr)

    def _to_egg_fact(self) -> bindings.Fact:
        return bindings.Fact(self._expr.__egg__)

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return self._expr.__egg_decls__


@dataclass
class Rule(Command):
    head: tuple[Action, ...]
    body: tuple[Fact, ...]
    name: str
    ruleset: Ruleset | None

    def __str__(self) -> str:
        head_str = ", ".join(map(str, self.head))
        body_str = ", ".join(map(str, self.body))
        return f"rule({body_str}).then({head_str})"

    def _to_egg_command(self, default_ruleset_name: str) -> bindings.RuleCommand:
        return bindings.RuleCommand(
            self.name,
            self.ruleset.egg_name if self.ruleset else default_ruleset_name,
            bindings.Rule(
                [a._to_egg_action() for a in self.head],
                [f._to_egg_fact() for f in self.body],
            ),
        )

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(*self.head, *self.body)


class Action(Command, ABC):
    """
    A change to an EGraph, either unioning multiple expressing, setting the value of a function call, deleting an expression, or panicking.
    """

    @abstractmethod
    def _to_egg_action(self) -> bindings._Action:
        raise NotImplementedError

    def _to_egg_command(self, default_ruleset_name: str) -> bindings._Command:
        return bindings.ActionCommand(self._to_egg_action())

    @property
    def ruleset(self) -> None | Ruleset:  # type: ignore[override]
        return None


@dataclass
class Let(Action):
    _name: str
    _value: RuntimeExpr

    def __str__(self) -> str:
        return f"let({self._name}, {self._value})"

    def _to_egg_action(self) -> bindings.Let:
        return bindings.Let(self._name, self._value.__egg__)

    @property
    def __egg_decls__(self) -> Declarations:
        return self._value.__egg_decls__


@dataclass
class Set(Action):
    """
    Similar to union, except can be used on primitive expressions, whereas union can only be used on user defined expressions.
    """

    _call: RuntimeExpr
    _rhs: RuntimeExpr

    def __str__(self) -> str:
        return f"set({self._call}).to({self._rhs})"

    def _to_egg_action(self) -> bindings.Set:
        egg_call = self._call.__egg__
        if not isinstance(egg_call, bindings.Call):
            raise ValueError(f"Can only create a set with a call for the lhs, got {self._call}")  # noqa: TRY004
        return bindings.Set(
            egg_call.name,
            egg_call.args,
            self._rhs.__egg__,
        )

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(self._call, self._rhs)


@dataclass
class ExprAction(Action):
    _expr: RuntimeExpr

    def __str__(self) -> str:
        return str(self._expr)

    def _to_egg_action(self) -> bindings.Expr_:
        return bindings.Expr_(self._expr.__egg__)

    @property
    def __egg_decls__(self) -> Declarations:
        return self._expr.__egg_decls__


@dataclass
class Delete(Action):
    """
    Remove a function call from an EGraph.
    """

    _call: RuntimeExpr

    def __str__(self) -> str:
        return f"delete({self._call})"

    def _to_egg_action(self) -> bindings.Delete:
        egg_call = self._call.__egg__
        if not isinstance(egg_call, bindings.Call):
            raise ValueError(f"Can only create a call with a call for the lhs, got {self._call}")  # noqa: TRY004
        return bindings.Delete(egg_call.name, egg_call.args)

    @property
    def __egg_decls__(self) -> Declarations:
        return self._call.__egg_decls__


@dataclass
class Union_(Action):  # noqa: N801
    """
    Merges two equivalence classes of two expressions.
    """

    _lhs: RuntimeExpr
    _rhs: RuntimeExpr

    def __str__(self) -> str:
        return f"union({self._lhs}).with_({self._rhs})"

    def _to_egg_action(self) -> bindings.Union:
        return bindings.Union(self._lhs.__egg__, self._rhs.__egg__)

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(self._lhs, self._rhs)


@dataclass
class Panic(Action):
    message: str

    def __str__(self) -> str:
        return f"panic({self.message})"

    def _to_egg_action(self) -> bindings.Panic:
        return bindings.Panic(self.message)

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations()


@dataclass
class Run(Schedule):
    """Configuration of a run"""

    # None if using default ruleset
    ruleset: Ruleset | None
    until: tuple[Fact, ...]

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self.ruleset, *self.until]))
        return f"run({args_str})"

    def _to_egg_schedule(self, default_ruleset_name: str) -> bindings._Schedule:
        return bindings.Run(self._to_egg_config(default_ruleset_name))

    def _to_egg_config(self, default_ruleset_name: str) -> bindings.RunConfig:
        return bindings.RunConfig(
            self.ruleset.egg_name if self.ruleset else default_ruleset_name,
            [fact._to_egg_fact() for fact in self.until] if self.until else None,
        )

    def _rulesets(self) -> Iterable[Ruleset]:
        if self.ruleset:
            yield self.ruleset

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(self.ruleset, *self.until)


@dataclass
class Saturate(Schedule):
    schedule: Schedule

    def __str__(self) -> str:
        return f"{self.schedule}.saturate()"

    def _to_egg_schedule(self, default_ruleset_name: str) -> bindings._Schedule:
        return bindings.Saturate(self.schedule._to_egg_schedule(default_ruleset_name))

    def _rulesets(self) -> Iterable[Ruleset]:
        return self.schedule._rulesets()

    @property
    def __egg_decls__(self) -> Declarations:
        return self.schedule.__egg_decls__


@dataclass
class Repeat(Schedule):
    length: int
    schedule: Schedule

    def __str__(self) -> str:
        return f"{self.schedule} * {self.length}"

    def _to_egg_schedule(self, default_ruleset_name: str) -> bindings._Schedule:
        return bindings.Repeat(self.length, self.schedule._to_egg_schedule(default_ruleset_name))

    def _rulesets(self) -> Iterable[Ruleset]:
        return self.schedule._rulesets()

    @property
    def __egg_decls__(self) -> Declarations:
        return self.schedule.__egg_decls__


@dataclass
class Sequence(Schedule):
    schedules: tuple[Schedule, ...]

    def __str__(self) -> str:
        return f"sequence({', '.join(map(str, self.schedules))})"

    def _to_egg_schedule(self, default_ruleset_name: str) -> bindings._Schedule:
        return bindings.Sequence([schedule._to_egg_schedule(default_ruleset_name) for schedule in self.schedules])

    def _rulesets(self) -> Iterable[Ruleset]:
        for s in self.schedules:
            yield from s._rulesets()

    @cached_property
    def __egg_decls__(self) -> Declarations:
        return Declarations.create(*self.schedules)


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


@deprecated("Use <ruleset>.register(<rewrite>) instead of passing rulesets as arguments to rewrites.")
@overload
def rewrite(lhs: EXPR, ruleset: Ruleset) -> _RewriteBuilder[EXPR]:
    ...


@overload
def rewrite(lhs: EXPR, ruleset: None = None) -> _RewriteBuilder[EXPR]:
    ...


def rewrite(lhs: EXPR, ruleset: Ruleset | None = None) -> _RewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression."""
    return _RewriteBuilder(lhs, ruleset)


@deprecated("Use <ruleset>.register(<birewrite>) instead of passing rulesets as arguments to birewrites.")
@overload
def birewrite(lhs: EXPR, ruleset: Ruleset) -> _BirewriteBuilder[EXPR]:
    ...


@overload
def birewrite(lhs: EXPR, ruleset: None = None) -> _BirewriteBuilder[EXPR]:
    ...


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
    return Panic(message)


def let(name: str, expr: Expr) -> Action:
    """Create a let binding."""
    return Let(name, to_runtime_expr(expr))


def expr_action(expr: Expr) -> Action:
    return ExprAction(to_runtime_expr(expr))


def delete(expr: Expr) -> Action:
    """Create a delete expression."""
    return Delete(to_runtime_expr(expr))


def expr_fact(expr: Expr) -> Fact:
    return ExprFact(to_runtime_expr(expr))


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    """Create a union of the given expression."""
    return _UnionBuilder(lhs=lhs)


def set_(lhs: EXPR) -> _SetBuilder[EXPR]:
    """Create a set of the given expression."""
    return _SetBuilder(lhs=lhs)


@deprecated("Use <ruleset>.register(<rule>) instead of passing rulesets as arguments to rules.")
@overload
def rule(*facts: FactLike, ruleset: Ruleset, name: str | None = None) -> _RuleBuilder:
    ...


@overload
def rule(*facts: FactLike, ruleset: None = None, name: str | None = None) -> _RuleBuilder:
    ...


def rule(*facts: FactLike, ruleset: Ruleset | None = None, name: str | None = None) -> _RuleBuilder:
    """Create a rule with the given facts."""
    return _RuleBuilder(facts=_fact_likes(facts), name=name, ruleset=ruleset)


def var(name: str, bound: type[EXPR]) -> EXPR:
    """Create a new variable with the given name and type."""
    return cast(EXPR, _var(name, bound))


def _var(name: str, bound: object) -> RuntimeExpr:
    """Create a new variable with the given name and type."""
    if not isinstance(bound, RuntimeClass | RuntimeParamaterizedClass):
        raise TypeError(f"Unexpected type {type(bound)}")
    return RuntimeExpr(bound.__egg_decls__, TypedExprDecl(class_to_ref(bound), VarDecl(name)))


def vars_(names: str, bound: type[EXPR]) -> Iterable[EXPR]:
    """Create variables with the given names and type."""
    for name in names.split(" "):
        yield var(name, bound)


@dataclass
class _RewriteBuilder(Generic[EXPR]):
    lhs: EXPR
    ruleset: Ruleset | None

    def to(self, rhs: EXPR, *conditions: FactLike) -> Rewrite:
        lhs = to_runtime_expr(self.lhs)
        rule = Rewrite(self.ruleset, lhs, convert_to_same_type(rhs, lhs), _fact_likes(conditions))
        if self.ruleset:
            self.ruleset.append(rule)
        return rule

    def __str__(self) -> str:
        return f"rewrite({self.lhs})"


@dataclass
class _BirewriteBuilder(Generic[EXPR]):
    lhs: EXPR
    ruleset: Ruleset | None

    def to(self, rhs: EXPR, *conditions: FactLike) -> Command:
        lhs = to_runtime_expr(self.lhs)
        rule = BiRewrite(self.ruleset, lhs, convert_to_same_type(rhs, lhs), _fact_likes(conditions))
        if self.ruleset:
            self.ruleset.append(rule)
        return rule

    def __str__(self) -> str:
        return f"birewrite({self.lhs})"


@dataclass
class _EqBuilder(Generic[EXPR]):
    expr: EXPR

    def to(self, *exprs: EXPR) -> Fact:
        expr = to_runtime_expr(self.expr)
        return Eq([expr] + [convert_to_same_type(e, expr) for e in exprs])

    def __str__(self) -> str:
        return f"eq({self.expr})"


@dataclass
class _NeBuilder(Generic[EXPR]):
    expr: EXPR

    def to(self, expr: EXPR) -> Unit:
        assert isinstance(self.expr, RuntimeExpr)
        args = (self.expr, convert_to_same_type(expr, self.expr))
        decls = Declarations.create(*args)
        res = RuntimeExpr(
            decls,
            TypedExprDecl(JustTypeRef("Unit"), CallDecl(FunctionRef("!="), tuple(a.__egg_typed_expr__ for a in args))),
        )
        return cast(Unit, res)

    def __str__(self) -> str:
        return f"ne({self.expr})"


@dataclass
class _SetBuilder(Generic[EXPR]):
    lhs: Expr

    def to(self, rhs: EXPR) -> Set:
        lhs = to_runtime_expr(self.lhs)
        return Set(lhs, convert_to_same_type(rhs, lhs))

    def __str__(self) -> str:
        return f"set_({self.lhs})"


@dataclass
class _UnionBuilder(Generic[EXPR]):
    lhs: Expr

    def with_(self, rhs: EXPR) -> Action:
        lhs = to_runtime_expr(self.lhs)
        return Union_(lhs, convert_to_same_type(rhs, lhs))

    def __str__(self) -> str:
        return f"union({self.lhs})"


@dataclass
class _RuleBuilder:
    facts: tuple[Fact, ...]
    name: str | None
    ruleset: Ruleset | None

    def then(self, *actions: ActionLike) -> Rule:
        rule = Rule(_action_likes(actions), self.facts, self.name or "", self.ruleset)
        if self.ruleset:
            self.ruleset.append(rule)
        return rule


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


def run(ruleset: Ruleset | None = None, *until: Fact) -> Run:
    """
    Create a run configuration.
    """
    return Run(ruleset, tuple(until))


def seq(*schedules: Schedule) -> Schedule:
    """
    Run a sequence of schedules.
    """
    return Sequence(tuple(schedules))


CommandLike = Command | Expr


def _command_like(command_like: CommandLike) -> Command:
    if isinstance(command_like, Expr):
        return expr_action(command_like)
    return command_like


CommandGenerator = Callable[..., Iterable[Rule | Rewrite]]


def _command_generator(gen: CommandGenerator) -> Iterable[Command]:
    """
    Calls the function with variables of the type and name of the arguments.
    """
    # Get the local scope from where the function is defined, so that we can get any type hints that are in the scope
    # but not in the globals
    current_frame = inspect.currentframe()
    assert current_frame
    register_frame = current_frame.f_back
    assert register_frame
    original_frame = register_frame.f_back
    assert original_frame
    hints = get_type_hints(gen, gen.__globals__, original_frame.f_locals)
    args = (_var(p.name, hints[p.name]) for p in signature(gen).parameters.values())
    return gen(*args)


ActionLike = Action | Expr


def _action_likes(action_likes: Iterable[ActionLike]) -> tuple[Action, ...]:
    return tuple(map(_action_like, action_likes))


def _action_like(action_like: ActionLike) -> Action:
    if isinstance(action_like, Expr):
        return expr_action(action_like)
    return action_like


FactLike = Fact | Expr


def _fact_likes(fact_likes: Iterable[FactLike]) -> tuple[Fact, ...]:
    return tuple(map(_fact_like, fact_likes))


def _fact_like(fact_like: FactLike) -> Fact:
    if isinstance(fact_like, Expr):
        return expr_fact(fact_like)
    return fact_like
