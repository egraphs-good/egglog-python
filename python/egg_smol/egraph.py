from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from inspect import Parameter, currentframe, signature
from types import FunctionType
from typing import _GenericAlias  # type: ignore[attr-defined]
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    NoReturn,
    Optional,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    overload,
)

from typing_extensions import (
    ParamSpec,
    TypeVarTuple,
    Unpack,
    assert_never,
    get_args,
    get_origin,
)

from . import bindings
from .declarations import *
from .monkeypatch import monkeypatch_forward_ref
from .runtime import *
from .runtime import _resolve_callable, class_to_ref

if TYPE_CHECKING:
    from .builtins import String

monkeypatch_forward_ref()

__all__ = [
    "EGraph",
    "BUILTINS",
    "BaseExpr",
    "Unit",
    "rewrite",
    "eq",
    "panic",
    "let",
    "delete",
    "union",
    "set_",
    "rule",
    "var",
    "vars_",
    "Fact",
    "expr_parts",
    "Schedule",
    "config",
    "sequence",
]

T = TypeVar("T")
TS = TypeVarTuple("TS")
P = ParamSpec("P")
TYPE = TypeVar("TYPE", bound="type[BaseExpr]")
CALLABLE = TypeVar("CALLABLE", bound=Callable)
EXPR = TypeVar("EXPR", bound="BaseExpr")

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
}


@dataclass
class EGraph:
    """
    An expression graph.
    """

    _egraph: bindings.EGraph | None = field(repr=False)
    _decls: Declarations = field(repr=False)
    # The current declarations which have been pushed to the stack
    _decl_stack: list[Declarations] = field(default_factory=list, repr=False)
    _BUILTIN_DECLS: ClassVar[Declarations | None] = None

    def __init__(self, *, _for_builtins: bool = False):
        """
        Creates a new e-graph.
        """
        # If this is the e-graph to register declerations for our builtins, don't actually create a runtime e-graph
        if _for_builtins:
            self._egraph = None
            self._decls = Declarations()
            # Set the global builtins to use these declerations
            EGraph._BUILTIN_DECLS = self._decls
        # Otherwise, we are creating a normal e-graph and use the builtin declerations as the base
        else:
            self._egraph = bindings.EGraph()
            if not EGraph._BUILTIN_DECLS:
                raise RuntimeError("Builtin values not registered yet")
            self._decls = deepcopy(EGraph._BUILTIN_DECLS)
        self._decl_stack = []

    def _run_program(self, commands: Iterable[bindings._Command]) -> None:
        if self._egraph:
            self._egraph.run_program(*commands)

    def _get_egraph(self) -> bindings.EGraph:
        if not self._egraph:
            raise RuntimeError("Cannot get the e-graph")
        return self._egraph

    def simplify(self, expr: EXPR, limit: int, *until: Fact) -> EXPR:
        """
        Simplifies the given expression.
        """
        return self._simplify(expr, limit, None, until)

    def _simplify(self, expr: EXPR, limit: int, ruleset: Optional[Ruleset], until: tuple[Fact, ...]) -> EXPR:
        tp, decl = expr_parts(expr)
        egg_expr = decl.to_egg(self._decls)
        self._run_program([bindings.Simplify(egg_expr, Config(limit, ruleset, until)._to_egg_config(self._decls))])
        extract_report = self._get_egraph().extract_report()
        if not extract_report:
            raise ValueError("No extract report saved")
        new_tp, new_decl = tp_and_expr_decl_from_egg(self._decls, extract_report.expr)
        return cast(EXPR, RuntimeExpr(self._decls, new_tp, new_decl))

    def relation(self, name: str, *tps: Unpack[TS], egg_fn: Optional[str] = None) -> Callable[[Unpack[TS]], Unit]:
        """
        Defines a relation, which is the same as a function which returns unit.
        """
        arg_types = tuple(self._resolve_type_annotation(cast(object, tp), [], None) for tp in tps)
        fn_decl = FunctionDecl(arg_types, TypeRefWithVars("unit"))
        commands = self._decls.register_callable(FunctionRef(name), fn_decl, egg_fn)
        self._run_program(commands)
        return cast(Callable[[Unpack[TS]], Unit], RuntimeFunction(self._decls, name))

    def include(self, path: str) -> None:
        """
        Include a file of rules.
        """
        raise NotImplementedError(
            "Not implemented yet, because we don't have a way of registering the types with Python"
        )

    def input(self, fn: Callable[..., String], path: str) -> None:
        """
        Loads a CSV file and sets it as *input, output of the function.
        """
        fn_name = self._decls.get_egg_fn(_resolve_callable(fn))
        self._run_program([bindings.Input(fn_name, path)])

    def output(self) -> None:
        raise NotImplementedError("Not imeplemented yet, because there are no examples in the egg-smol repo")

    def calc(self) -> None:
        raise NotImplementedError("Not implemented yet")

    @overload
    def run(self, limit: int, /, *until: Fact) -> bindings.RunReport:
        ...

    @overload
    def run(self, schedule: Schedule, /) -> bindings.RunReport:
        ...

    def run(self, limit_or_schedule: int | Schedule, /, *until: Fact) -> bindings.RunReport:
        """
        Run the egraph until the given limit or until the given facts are true.
        """
        if isinstance(limit_or_schedule, int):
            limit_or_schedule = config(limit_or_schedule, None, *until)
        return self._run_schedule(limit_or_schedule)

    def _run_schedule(self, schedule: Schedule) -> bindings.RunReport:
        self._run_program([bindings.RunScheduleCommand(schedule._to_egg(self._decls))])
        run_report = self._get_egraph().run_report()
        if not run_report:
            raise ValueError("No run report saved")
        return run_report

    def check(self, *facts: Fact) -> None:
        """
        Check if a fact is true in the egraph.
        """
        self._run_program([self._facts_to_check(facts)])

    def check_fail(self, *facts) -> None:
        """
        Checks that one of the facts is not true
        """
        self._run_program([bindings.Fail(self._facts_to_check(facts))])

    def _facts_to_check(self, facts: Iterable[Fact]) -> bindings.Check:
        egg_facts = [fact_decl_to_egg(self._decls, _fact_to_decl(f)) for f in facts]
        return bindings.Check(egg_facts)

    def extract(self, expr: EXPR) -> EXPR:
        """
        Extract the lowest cost expression from the egraph.
        """
        tp, decl = expr_parts(expr)
        egg_expr = decl.to_egg(self._decls)
        extract_report = self._run_extract(egg_expr, 0)
        new_tp, new_decl = tp_and_expr_decl_from_egg(self._decls, extract_report.expr)
        if new_tp != tp:
            raise RuntimeError(f"Type mismatch: {tp} != {new_tp}")
        return cast(EXPR, RuntimeExpr(self._decls, tp, new_decl))

    def extract_multiple(self, expr: EXPR, n: int) -> list[EXPR]:
        """
        Extract multiple expressions from the egraph.
        """
        tp, decl = expr_parts(expr)
        egg_expr = decl.to_egg(self._decls)
        extract_report = self._run_extract(egg_expr, n + 1)
        new_decls = [tp_and_expr_decl_from_egg(self._decls, egg_expr)[1] for egg_expr in extract_report.variants]
        return [cast(EXPR, RuntimeExpr(self._decls, tp, new_decl)) for new_decl in new_decls]

    def _run_extract(self, expr: bindings._Expr, n: int) -> bindings.ExtractReport:
        self._run_program([bindings.Extract(n, expr)])
        extract_report = self._get_egraph().extract_report()
        if not extract_report:
            raise ValueError("No extract report saved")
        return extract_report

    def constant(self, name: str, tp: type[EXPR], egg_name: Optional[str] = None) -> EXPR:
        """
        Defines a named constant of a certain type.

        This is the same as defining a nullary function with a high cost.
        """
        ref = ConstantRef(name)
        type_ref, commands = self._register_constant(ref, tp, egg_name, None)
        self._run_program(commands)
        return cast(EXPR, RuntimeExpr(self._decls, type_ref, CallDecl(ref)))

    def _register_constant(
        self,
        ref: ConstantRef | ClassVariableRef,
        tp: object,
        egg_name: Optional[str],
        cls_type_and_name: Optional[tuple[type | RuntimeClass, str]],
    ) -> tuple[JustTypeRef, Iterable[bindings._Command]]:
        """
        Register a constant, returning its typeref.
        """
        type_ref = self._resolve_type_annotation(tp, [], cls_type_and_name).to_just()
        fn_decl = constant_function_decl(type_ref)
        return type_ref, self._decls.register_callable(ref, fn_decl, egg_name)

    def define(self, name: str, expr: EXPR) -> EXPR:
        """
        Define a new expression in the egraph and return a reference to it.
        """
        # Don't support cost and maybe will be removed in favor of let
        # https://github.com/mwillsey/egg-smol/issues/128#issuecomment-1523760578
        tp, decl = expr_parts(expr)
        self._run_program([bindings.Define(name, decl.to_egg(self._decls), None)])
        return cast(EXPR, RuntimeExpr(self._decls, tp, VarDecl(name)))

    def push(self) -> None:
        """
        Push the current state of the egraph, so that it can be popped later and reverted back.
        """
        self._run_program([bindings.Push(1)])
        self._decl_stack.append(self._decls)
        self._decls = deepcopy(self._decls)

    def pop(self) -> None:
        """
        Pop the current state of the egraph, reverting back to the previous state.
        """
        self._run_program([bindings.Pop(1)])
        self._decls = self._decl_stack.pop()

    def __enter__(self):
        """
        Copy the egraph state, so that it can be reverted back to the original state at the end.
        """
        self.push()

    def __exit__(self, exc_type, exc, exc_tb):
        self.pop()

    @overload
    def class_(self, *, egg_sort: str) -> Callable[[TYPE], TYPE]:
        ...

    @overload
    def class_(self, cls: TYPE, /) -> TYPE:
        ...

    def class_(self, *args, **kwargs) -> Any:
        """
        Registers a class.
        """
        frame = currentframe()
        assert frame
        prev_frame = frame.f_back
        assert prev_frame

        if kwargs:
            assert set(kwargs.keys()) == {"egg_sort"}
            return lambda cls: self._class(cls, prev_frame.f_locals, prev_frame.f_globals, kwargs["egg_sort"])
        assert len(args) == 1
        return self._class(args[0], prev_frame.f_locals, prev_frame.f_globals)

    def _class(
        self,
        cls: type[BaseExpr],
        hint_locals: dict[str, Any],
        hint_globals: dict[str, Any],
        egg_sort: Optional[str] = None,
    ) -> RuntimeClass:
        """
        Registers a class.
        """
        cls_name = cls.__name__
        # Get all the methods from the class
        cls_dict: dict[str, Any] = {k: v for k, v in cls.__dict__.items() if k not in IGNORED_ATTRIBUTES}
        parameters: list[TypeVar] = cls_dict.pop("__parameters__", [])

        n_type_vars = len(parameters)
        commands = list(self._decls.register_class(cls_name, n_type_vars, egg_sort))
        # The type ref of self is paramterized by the type vars
        slf_type_ref = TypeRefWithVars(cls_name, tuple(ClassTypeVarRef(i) for i in range(n_type_vars)))

        # First register any class vars as constants
        hint_globals = hint_globals.copy()
        hint_globals[cls_name] = cls
        for k, v in get_type_hints(cls, globalns=hint_globals, localns=hint_locals).items():
            if v.__origin__ == ClassVar:
                (inner_tp,) = v.__args__
                commands.extend(
                    self._register_constant(ClassVariableRef(cls_name, k), inner_tp, None, (cls, cls_name))[1]
                )
            else:
                raise NotImplementedError("The only supported annotations on class attributes are class vars")

        # Then register each of its methods
        for method_name, method in cls_dict.items():
            is_init = method_name == "__init__"
            # Don't register the init methods for literals, since those don't use the type checking mechanisms
            if is_init and cls_name in LIT_CLASS_NAMES:
                continue
            if isinstance(method, _WrappedMethod):
                fn = method.fn
                egg_fn = method.egg_fn
                cost = method.cost
                default = method.default
                merge = method.merge
                on_merge = method.on_merge
            else:
                fn = method
                egg_fn, cost, default, merge, on_merge = None, None, None, None, None
            if isinstance(fn, classmethod):
                fn = fn.__func__
                is_classmethod = True
            else:
                # We count __init__ as a classmethod since it is called on the class
                is_classmethod = is_init

            fn_decl = self._generate_function_decl(
                fn,
                hint_locals,
                default,
                cost,
                merge,
                on_merge,
                "cls" if is_classmethod and not is_init else slf_type_ref,
                parameters,
                is_init,
                # If this is an i64, use the runtime class for the alias so that i64Like is resolved properly
                # Otherwise, this might be a Map in which case pass in the original cls so that we
                # can do Map[T, V] on it, which is not allowed on the runtime class
                cls_type_and_name=(RuntimeClass(self._decls, "i64") if cls_name == "i64" else cls, cls_name),
            )
            ref: ClassMethodRef | MethodRef = (
                ClassMethodRef(cls_name, method_name) if is_classmethod else MethodRef(cls_name, method_name)
            )
            commands.extend(
                self._decls.register_callable(ref, fn_decl, egg_fn, generate_commands=self._egraph is not None)
            )

        self._run_program(commands)
        # Register != as a method so we can print it as a string
        self._decls.register_callable_ref(MethodRef(cls_name, "__ne__"), "!=")
        return RuntimeClass(self._decls, cls_name)

    # We seperate the function and method overloads to make it simpler to know if we are modifying a function or method,
    # So that we can add the functions eagerly to the registry and wait on the methods till we process the class.

    # We have to seperate method/function overloads for those that use the T params and those that don't
    # Otherwise, if you say just pass in `cost` then the T param is inferred as `Nothing` and
    # It will break the typing.
    @overload
    def method(  # type: ignore
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        merge: Optional[Callable[[Any, Any], Any]] = None,
        on_merge: Optional[Callable[[Any, Any], Iterable[Action]]] = None,
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

    @overload
    def method(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
        on_merge: Optional[Callable[[EXPR, EXPR], Iterable[Action]]] = None,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    def method(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
        on_merge: Optional[Callable[[EXPR, EXPR], Iterable[Action]]] = None,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        return lambda fn: _WrappedMethod(egg_fn, cost, default, merge, on_merge, fn)

    @overload
    def function(self, fn: CALLABLE, /) -> CALLABLE:
        ...

    @overload
    def function(  # type: ignore
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        merge: Optional[Callable[[Any, Any], Any]] = None,
        on_merge: Optional[Callable[[Any, Any], Iterable[Action]]] = None,
    ) -> Callable[[CALLABLE], CALLABLE]:
        ...

    @overload
    def function(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
        on_merge: Optional[Callable[[EXPR, EXPR], Iterable[Action]]] = None,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    def function(self, *args, **kwargs) -> Any:
        """
        Registers a function.
        """
        fn_locals = currentframe().f_back.f_locals  # type: ignore

        # If we have any positional args, then we are calling it directly on a function
        if args:
            assert len(args) == 1
            return self._function(args[0], fn_locals)
        # otherwise, we are passing some keyword args, so save those, and then return a partial
        return lambda fn: self._function(fn, fn_locals, **kwargs)

    def _function(
        self,
        fn: Callable[..., RuntimeExpr],
        hint_locals: dict[str, Any],
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[RuntimeExpr] = None,
        merge: Optional[Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr]] = None,
        on_merge: Optional[Callable[[RuntimeExpr, RuntimeExpr], Iterable[Action]]] = None,
    ) -> RuntimeFunction:
        """
        Uncurried version of function decorator
        """
        name = fn.__name__
        # Save function decleartion
        fn_decl = self._generate_function_decl(fn, hint_locals, default, cost, merge, on_merge)
        commands = self._decls.register_callable(FunctionRef(name), fn_decl, egg_fn)
        self._run_program(commands)
        # Return a runtime function which will act like the decleration
        return RuntimeFunction(self._decls, name)

    def _generate_function_decl(
        self,
        fn: Any,
        # Pass in the locals, retrieved from the frame when wrapping,
        # so that we support classes and function defined inside of other functions (which won't show up in the globals)
        hint_locals: dict[str, Any],
        default: Optional[RuntimeExpr],
        cost: Optional[int],
        merge: Optional[Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr]],
        on_merge: Optional[Callable[[RuntimeExpr, RuntimeExpr], Iterable[Action]]],
        # The first arg is either cls, for a classmethod, a self type, or none for a function
        first_arg: Literal["cls"] | TypeOrVarRef | None = None,
        cls_typevars: list[TypeVar] = [],
        is_init: bool = False,
        cls_type_and_name: Optional[tuple[type | RuntimeClass, str]] = None,
    ) -> FunctionDecl:
        if not isinstance(fn, FunctionType):
            raise NotImplementedError(f"Can only generate function decls for functions not {fn}  {type(fn)}")

        hint_globals = fn.__globals__.copy()

        if cls_type_and_name:
            hint_globals[cls_type_and_name[1]] = cls_type_and_name[0]
        hints = get_type_hints(fn, hint_globals, hint_locals)
        # If this is an init fn use the first arg as the return type
        if is_init:
            if not isinstance(first_arg, (ClassTypeVarRef, TypeRefWithVars)):
                raise ValueError("Init function must have a self type")
            return_type = first_arg
        else:
            return_type = self._resolve_type_annotation(hints["return"], cls_typevars, cls_type_and_name)

        params = list(signature(fn).parameters.values())
        # Remove first arg if this is a classmethod or a method, since it won't have an annotation
        if first_arg is not None:
            first, *params = params
            if first.annotation != Parameter.empty:
                raise ValueError(f"First arg of a method must not have an annotation, not {first.annotation}")

        for param in params:
            if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
                raise ValueError(f"Can only register functions with positional or keyword args, not {param.kind}")

        arg_types = tuple(self._resolve_type_annotation(hints[t.name], cls_typevars, cls_type_and_name) for t in params)
        # If the first arg is a self, and this not an __init__ fn, add this as a typeref
        if isinstance(first_arg, (ClassTypeVarRef, TypeRefWithVars)) and not is_init:
            arg_types = (first_arg,) + arg_types

        default_decl = None if default is None else default.__egg_expr__
        merge_decl = (
            None
            if merge is None
            else merge(
                RuntimeExpr(self._decls, return_type.to_just(), VarDecl("old")),
                RuntimeExpr(self._decls, return_type.to_just(), VarDecl("new")),
            ).__egg_expr__
        )
        merge_action = (
            ()
            if on_merge is None
            else tuple(
                map(
                    _action_to_decl,
                    on_merge(
                        RuntimeExpr(self._decls, return_type.to_just(), VarDecl("old")),
                        RuntimeExpr(self._decls, return_type.to_just(), VarDecl("new")),
                    ),
                )
            )
        )
        decl = FunctionDecl(
            return_type=return_type,
            arg_types=arg_types,
            cost=cost,
            default=default_decl,
            merge=merge_decl,
            merge_action=merge_action,
        )
        return decl

    def _resolve_type_annotation(
        self,
        tp: object,
        cls_typevars: list[TypeVar],
        cls_type_and_name: Optional[tuple[type | RuntimeClass, str]],
    ) -> TypeOrVarRef:
        if isinstance(tp, TypeVar):
            return ClassTypeVarRef(cls_typevars.index(tp))
        # If there is a union, it should be of a literal and another type to allow type promotion
        if get_origin(tp) == Union:
            args = get_args(tp)
            if len(args) != 2:
                raise TypeError("Union types are only supported for type promotion")
            fst, snd = args
            if fst in {int, str, float}:
                return self._resolve_type_annotation(snd, cls_typevars, cls_type_and_name)
            if snd in {int, str, float}:
                return self._resolve_type_annotation(fst, cls_typevars, cls_type_and_name)
            raise TypeError("Union types are only supported for type promotion")

        # If this is the type for the class, use the class name
        if cls_type_and_name and tp == cls_type_and_name[0]:
            return TypeRefWithVars(cls_type_and_name[1])

        # If this is the class for this method and we have a paramaterized class, recurse
        if (
            cls_type_and_name
            and isinstance(tp, _GenericAlias)
            and tp.__origin__ == cls_type_and_name[0]  # type: ignore
        ):
            return TypeRefWithVars(
                cls_type_and_name[1],
                tuple(
                    self._resolve_type_annotation(a, cls_typevars, cls_type_and_name)
                    for a in tp.__args__  # type: ignore
                ),
            )

        if isinstance(tp, (RuntimeClass, RuntimeParamaterizedClass)):
            return class_to_ref(tp).to_var()
        raise TypeError(f"Unexpected type annotation {tp}")

    def register(self, *values: Rewrite | Birewrite | Rule | Action) -> None:
        """
        Registers any number of rewrites or rules.
        """
        self._run_program(_value_to_command(self._decls, v, ruleset="") for v in values)

    def ruleset(self, name: str) -> Ruleset:
        self._run_program([bindings.AddRuleset(name)])
        return Ruleset(self, name)


def _value_to_command(
    decls: Declarations, value: Rewrite | Birewrite | Rule | Action, ruleset: str
) -> bindings._Command:
    if isinstance(value, Rewrite):
        return bindings.RewriteCommand(ruleset, value._to_decl().to_egg(decls))
    if isinstance(value, Birewrite):
        return bindings.BiRewriteCommand(ruleset, value._to_decl().to_egg(decls))
    if isinstance(value, Rule):
        return bindings.RuleCommand(value.name or "", ruleset, value._to_decl().to_egg(decls))
    return bindings.ActionCommand(action_decl_to_egg(decls, _action_to_decl(value)))


@dataclass(frozen=True)
class _WrappedMethod(Generic[P, EXPR]):
    """
    Used to wrap a method and store some extra options on it before processing it.
    """

    egg_fn: Optional[str]
    cost: Optional[int]
    default: Optional[EXPR]
    merge: Optional[Callable[[EXPR, EXPR], EXPR]]
    on_merge: Optional[Callable[[EXPR, EXPR], Iterable[Action]]]
    fn: Callable[P, EXPR]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> EXPR:
        raise NotImplementedError("We should never call a wrapped method. Did you forget to wrap the class?")


class _BaseExprMetaclass(type):
    """
    Metaclass of BaseExpr, used to override isistance checks, so that runtime expressions are instances
    of BaseExpr at runtime, so this matches the intuition.
    """

    def __instancecheck__(self, instance: object) -> bool:
        return isinstance(instance, RuntimeExpr)


class BaseExpr(metaclass=_BaseExprMetaclass):
    """
    Expression base class, which adds suport for != to all expression types.
    """

    def __ne__(self: EXPR, other_expr: EXPR) -> Unit:  # type: ignore[override, empty-body]
        """
        Compare whether to expressions are not equal.

        :param self: The expression to compare.
        :param other_expr: The other expression to compare to, which must be of the same type.
        :meta public:
        """
        ...

    def __eq__(self, other: NoReturn) -> NoReturn:  # type: ignore[override, empty-body]
        """
        Equality is currently not supported. We only add this method so that
        if you try to use it MyPy will warn you.
        """
        ...


BUILTINS = EGraph(_for_builtins=True)


@BUILTINS.class_(egg_sort="Unit")
class Unit(BaseExpr):
    """
    The unit type. This is also used to reprsent if a value exists, if it is resolved or not.
    """

    def __init__(self) -> None:
        ...


@dataclass(frozen=True)
class Ruleset:
    _egraph: EGraph = field(repr=False)
    name: str

    def register(self, *values: Rewrite | Birewrite | Rule) -> None:
        """
        Registers any number of rewrites or rules.
        """
        self._egraph._run_program(_value_to_command(self._egraph._decls, v, ruleset=self.name) for v in values)

    def run(self, limit: int, *until: Fact) -> bindings.RunReport:
        """
        Run the e-graph with this ruleset.
        """
        return self._egraph._run_schedule(config(limit, self, *until))

    def simplify(self, expr: EXPR, limit: int, *until: Fact) -> EXPR:
        """
        Simplify the given expression with this ruleset.
        """
        return self._egraph._simplify(expr, limit, self, until)


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


def rewrite(lhs: EXPR) -> _RewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression."""
    return _RewriteBuilder(lhs=lhs)


def birewrite(lhs: EXPR) -> _BirewriteBuilder[EXPR]:
    """Rewrite the given expression to a new expression and vice versa."""
    return _BirewriteBuilder(lhs=lhs)


def eq(expr: EXPR) -> _EqBuilder[EXPR]:
    """Check if the given expression is equal to the given value."""
    return _EqBuilder(expr)


def panic(message: str) -> Panic:
    """Raise an error with the given message."""
    return Panic(message)


def let(name: str, expr: BaseExpr) -> Let:
    """Create a let binding."""
    return Let(name, expr)


def delete(expr: BaseExpr) -> Delete:
    """Create a delete expression."""
    return Delete(expr)


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    """Create a union of the given expression."""
    return _UnionBuilder(lhs=lhs)


def set_(lhs: EXPR) -> _SetBuilder[EXPR]:
    """Create a set of the given expression."""
    return _SetBuilder(lhs=lhs)


def rule(*facts: Fact, name: Optional[str] = None) -> _RuleBuilder:
    """Create a rule with the given facts."""
    return _RuleBuilder(facts=facts, name=name)


def var(name: str, bound: type[EXPR]) -> EXPR:
    """Create a new variable with the given name and type."""
    return cast(EXPR, _var(name, bound))


def _var(name: str, bound: Any) -> RuntimeExpr:
    """Create a new variable with the given name and type."""
    if not isinstance(bound, (RuntimeClass, RuntimeParamaterizedClass)):
        raise TypeError(f"Unexpected type {type(bound)}")
    return RuntimeExpr(bound.__egg_decls__, class_to_ref(bound), VarDecl(name))


def vars_(names: str, bound: type[EXPR]) -> Iterable[EXPR]:
    """Create variables with the given names and type."""
    for name in names.split(" "):
        yield var(name, bound)


@dataclass
class _RewriteBuilder(Generic[EXPR]):
    lhs: EXPR

    def to(self, rhs: EXPR, *conditions: Fact) -> Rewrite:
        return Rewrite(lhs=self.lhs, rhs=rhs, conditions=list(conditions))

    def __str__(self) -> str:
        return f"rewrite({self.lhs})"


@dataclass
class _BirewriteBuilder(Generic[EXPR]):
    lhs: EXPR

    def to(self, rhs: EXPR, *conditions: Fact) -> Birewrite:
        return Birewrite(lhs=self.lhs, rhs=rhs, conditions=list(conditions))

    def __str__(self) -> str:
        return f"birewrite({self.lhs})"


@dataclass
class _EqBuilder(Generic[EXPR]):
    expr: BaseExpr

    def to(self, *exprs: EXPR) -> Eq:
        return Eq([self.expr, *exprs])

    def __str__(self) -> str:
        return f"eq({self.expr})"


@dataclass
class _SetBuilder(Generic[EXPR]):
    lhs: BaseExpr

    def to(self, rhs: EXPR) -> Set:
        return Set(lhs=self.lhs, rhs=rhs)

    def __str__(self) -> str:
        return f"set_({self.lhs})"


@dataclass
class _UnionBuilder(Generic[EXPR]):
    lhs: BaseExpr

    def with_(self, rhs: EXPR) -> Union_:
        return Union_(lhs=self.lhs, rhs=rhs)

    def __str__(self) -> str:
        return f"union({self.lhs})"


@dataclass
class _RuleBuilder:
    facts: tuple[Fact, ...]
    name: Optional[str]

    def then(self, *actions: Action) -> Rule:
        return Rule(actions, self.facts, self.name)


def expr_parts(expr: BaseExpr) -> tuple[JustTypeRef, ExprDecl]:
    """
    Returns the underlying type and decleration of the expression. Useful for testing structural equality or debugging.

    :rtype: tuple[object, object]
    """
    assert isinstance(expr, RuntimeExpr)
    return expr.__egg_parts__


@dataclass
class Rewrite:
    lhs: BaseExpr
    rhs: BaseExpr
    conditions: list[Fact]

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self.rhs, *self.conditions]))
        return f"rewrite({self.lhs}).to({args_str})"

    def _to_decl(self) -> RewriteDecl:
        return RewriteDecl(
            expr_parts(self.lhs)[1],
            expr_parts(self.rhs)[1],
            tuple(_fact_to_decl(fact) for fact in self.conditions),
        )


@dataclass
class Birewrite:
    lhs: BaseExpr
    rhs: BaseExpr
    conditions: list[Fact]

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self.rhs, *self.conditions]))
        return f"birewrite({self.lhs}).to({args_str})"

    def _to_decl(self) -> RewriteDecl:
        return RewriteDecl(
            expr_parts(self.lhs)[1],
            expr_parts(self.rhs)[1],
            tuple(_fact_to_decl(fact) for fact in self.conditions),
        )


@dataclass
class Eq:
    exprs: list[BaseExpr]

    def __str__(self) -> str:
        first, *rest = self.exprs
        args_str = ", ".join(map(str, rest))
        return f"eq({first}).to({args_str})"

    def _to_decl(self) -> EqDecl:
        return EqDecl(tuple(expr_parts(expr)[1] for expr in self.exprs))


Fact = Union[Unit, Eq]


def _fact_to_decl(fact: Fact) -> FactDecl:
    if isinstance(fact, Eq):
        return fact._to_decl()
    elif isinstance(fact, BaseExpr):
        return expr_parts(fact)[1]
    assert_never(fact)


@dataclass
class Delete:
    expr: BaseExpr

    def __str__(self) -> str:
        return f"delete({self.expr})"

    def _to_decl(self) -> DeleteDecl:
        decl = expr_parts(self.expr)[1]
        if not isinstance(decl, CallDecl):
            raise ValueError(f"Can only delete calls not {decl}")
        return DeleteDecl(decl)


@dataclass
class Panic:
    message: str

    def __str__(self) -> str:
        return f"panic({self.message})"

    def _to_decl(self) -> PanicDecl:
        return PanicDecl(self.message)


@dataclass
class Union_:
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"union({self.lhs}).with_({self.rhs})"

    def _to_decl(self) -> UnionDecl:
        return UnionDecl(expr_parts(self.lhs)[1], expr_parts(self.rhs)[1])


@dataclass
class Set:
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"set_({self.lhs}).to({self.rhs})"

    def _to_decl(self) -> SetDecl:
        lhs = expr_parts(self.lhs)[1]
        if not isinstance(lhs, CallDecl):
            raise ValueError(f"Can only create a call with a call for the lhs, got {lhs}")
        return SetDecl(lhs, expr_parts(self.rhs)[1])


@dataclass
class Let:
    name: str
    value: BaseExpr

    def __str__(self) -> str:
        return f"let({self.name}, {self.value})"

    def _to_decl(self) -> LetDecl:
        return LetDecl(self.name, expr_parts(self.value)[1])


Action = Union[Let, Set, Delete, Union_, Panic, "BaseExpr"]


def _action_to_decl(action: Action) -> ActionDecl:
    if isinstance(action, BaseExpr):
        return expr_parts(action)[1]
    return action._to_decl()


@dataclass
class Rule:
    header: tuple[Action, ...]
    body: tuple[Fact, ...]
    name: Optional[str]

    def _to_decl(self) -> RuleDecl:
        return RuleDecl(
            tuple(_action_to_decl(action) for action in self.header),
            tuple(_fact_to_decl(fact) for fact in self.body),
        )


def config(limit: int, ruleset: Optional[Ruleset] = None, *until: Fact) -> Config:
    """
    Create a run configuration.
    """
    return Config(limit, ruleset, tuple(until))


def sequence(*schedules: Schedule) -> Schedule:
    """
    Run a sequence of schedules.
    """
    return Sequence(tuple(schedules))


class _BaseSchedule:
    def __mul__(self, length: int) -> Schedule:
        """
        Repeat the schedule a number of times.
        """
        if not isinstance(self, (Config, Repeat, Saturate, Sequence)):
            raise TypeError(f"Cannot multiply {type(self)}")
        return Repeat(length, self)

    def saturate(self) -> Schedule:
        """
        Run the schedule until the e-graph is saturated.
        """
        if not isinstance(self, (Config, Repeat, Saturate, Sequence)):
            raise TypeError(f"Cannot saturate {type(self)}")
        return Saturate(self)


@dataclass
class Config(_BaseSchedule):
    """Configuration of a run"""

    limit: int
    ruleset: Optional[Ruleset]
    until: tuple[Fact, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        args_str = ", ".join(map(str, [self.limit, self.ruleset, *self.until]))
        return f"config({args_str})"

    def _to_egg(self, decls: Declarations) -> bindings._Schedule:
        return bindings.Run(self._to_egg_config(decls))

    def _to_egg_config(self, decls: Declarations) -> bindings.RunConfig:
        return bindings.RunConfig(
            self.ruleset.name if self.ruleset else "",
            self.limit,
            [fact_decl_to_egg(decls, _fact_to_decl(fact)) for fact in self.until] if self.until else None,
        )


@dataclass
class Saturate(_BaseSchedule):
    schedule: Schedule

    def __str__(self) -> str:
        return f"{self.schedule}.saturate()"

    def _to_egg(self, declerations: Declarations) -> bindings._Schedule:
        return bindings.Saturate(self.schedule._to_egg(declerations))


@dataclass
class Repeat(_BaseSchedule):
    length: int
    schedule: Schedule

    def __str__(self) -> str:
        return f"{self.schedule} * {self.length}"

    def _to_egg(self, declerations: Declarations) -> bindings._Schedule:
        return bindings.Repeat(self.length, self.schedule._to_egg(declerations))


@dataclass
class Sequence(_BaseSchedule):
    schedules: tuple[Schedule, ...]

    def __str__(self) -> str:
        return f"sequence({', '.join(map(str, self.schedules))})"

    def _to_egg(self, declerations: Declarations) -> bindings._Schedule:
        return bindings.Sequence([schedule._to_egg(declerations) for schedule in self.schedules])


# Define Schedule union instead of using BaseSchedule b/c Python doesn't suppot
# closed types
Schedule = Union[Config, Repeat, Saturate, Sequence]
