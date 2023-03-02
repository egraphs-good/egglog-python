from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
from types import FunctionType, UnionType
from typing import _GenericAlias  # type: ignore[attr-defined]
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)

from .declarations import *
from .runtime import *
from .runtime import class_decls, class_to_ref

if TYPE_CHECKING:
    from .builtins import BaseExpr, Unit

__all__ = [
    "Registry",
    "rewrite",
    "eq",
    "panic",
    "let",
    "delete",
    "union",
    "set_",
    "if_",
    "var",
    "vars",
    "Fact",
]

T = TypeVar("T")
P = ParamSpec("P")
TYPE = TypeVar("TYPE", bound=type)
CALLABLE = TypeVar("CALLABLE", bound=Callable)
EXPR = TypeVar("EXPR", bound="BaseExpr")


@dataclass
class Registry:
    """
    A registry holds all the declerations of classes and functions as well as the mapping
    to and from egg names to python names.
    """

    _decls: Declarations = field(default_factory=Declarations)

    def _on_register_sort(self, name: str) -> None:
        """
        Called whenever a sort is registered.
        """
        pass

    def _on_register_function(self, ref: CallableRef, decl: FunctionDecl) -> None:
        """
        Called whenever a function is registered.
        """
        pass

    def _on_register_rewrite(self, decl: RewriteDecl) -> None:
        """
        Called whenever a rewrite is registered.
        """
        pass

    def _on_register_rule(self, decl: RuleDecl) -> None:
        """
        Called whenever a rule is registered.
        """
        pass

    def _on_register_action(self, decl: ActionDecl) -> None:
        pass

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

        if kwargs:
            assert set(kwargs.keys()) == {"egg_sort"}
            return lambda cls: self._class(cls, kwargs["egg_sort"])
        assert len(args) == 1
        return self._class(args[0])

    def _class(
        self, cls: type[BaseExpr], egg_sort: Optional[str] = None
    ) -> RuntimeClass:
        """
        Registers a class.
        """
        cls_name = cls.__name__
        # Get all the methods from the class
        cls_dict: dict[str, Any] = dict(cls.__dict__)
        del cls_dict["__module__"]
        del cls_dict["__doc__"]
        parameters: list[TypeVar]
        if "__orig_bases__" in cls_dict:
            del cls_dict["__orig_bases__"]
            parameters = cls_dict["__parameters__"]
        else:
            parameters = []

        # Register class first
        if cls_name in self._decls.classes:
            raise ValueError(f"Class {cls_name} already registered")
        n_type_vars = len(parameters)
        cls_decl = ClassDecl(n_type_vars=n_type_vars)
        self._decls.classes[cls_name] = cls_decl

        # The type ref of self is paramterized by the type vars
        slf_type_ref = TypeRefWithVars(
            cls_name, tuple(ClassTypeVarRef(i) for i in range(n_type_vars))
        )

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
            else:
                fn = method
                egg_fn, cost, default, merge = None, None, None, None
            if isinstance(fn, classmethod):
                fn = fn.__func__
                is_classmethod = True
            else:
                is_classmethod = False

            first_arg = "cls" if is_classmethod else slf_type_ref
            fn_decl = self._generate_function_decl(
                fn,
                default,
                cost,
                merge,
                first_arg,
                parameters,
                is_init,
                (cls, cls_name),
            )

            if is_classmethod:
                cls_decl.class_methods[method_name] = fn_decl
                ref = ClassMethodRef(cls_name, method_name)
            else:
                cls_decl.methods[method_name] = fn_decl
                ref = MethodRef(cls_name, method_name)
            self._register_callable_ref(egg_fn, ref)

        return RuntimeClass(self._decls, cls_name)

    # We seperate the function and method overloads to make it simpler to know if we are modifying a function or method,
    # So that we can add the functions eagerly to the registry and wait on the methods till we process the class.

    # We have to seperate method/function overloads for those that use the T params and those that don't
    # Otherwise, if you say just pass in `cost` then the T param is inferred as `Nothing` and
    # It will break the typing.
    @overload
    def method(  # type: ignore
        self, *, egg_fn: Optional[str] = None, cost: Optional[int] = None
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
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    def method(
        self,
        *,
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[EXPR] = None,
        merge: Optional[Callable[[EXPR, EXPR], EXPR]] = None,
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        return lambda fn: _WrappedMethod(egg_fn, cost, default, merge, fn)

    @overload
    def function(self, fn: CALLABLE, /) -> CALLABLE:
        ...

    @overload
    def function(
        self, *, egg_fn: Optional[str] = None, cost: Optional[int] = None
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
    ) -> Callable[[Callable[P, EXPR]], Callable[P, EXPR]]:
        ...

    def function(self, *args, **kwargs) -> Any:
        """
        Registers a function.
        """
        # If we have any positional args, then we are calling it directly on a function
        if args:
            assert len(args) == 1
            return self._function(args[0])
        # otherwise, we are passing some keyword args, so save those, and then return a partial
        return lambda fn: self._function(fn, **kwargs)

    def _function(
        self,
        fn: Callable[..., RuntimeExpr],
        egg_fn: Optional[str] = None,
        cost: Optional[int] = None,
        default: Optional[RuntimeExpr] = None,
        merge: Optional[Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr]] = None,
    ) -> RuntimeFunction:
        """
        Uncurried version of function decorator
        """
        name = fn.__name__
        if name in self._decls.functions:
            raise ValueError(f"Function {name} already registered")

        # Save function decleartion
        self._decls.functions[name] = self._generate_function_decl(
            fn, default, cost, merge
        )
        # Register it with the egg name
        self._register_callable_ref(egg_fn, FunctionRef(name))
        # Return a runtime function whcich will act like the decleration
        return RuntimeFunction(self._decls, name)

    def _generate_function_decl(
        self,
        fn: Any,
        default: Optional[RuntimeExpr],
        cost: Optional[int],
        merge: Optional[Callable[[RuntimeExpr, RuntimeExpr], RuntimeExpr]],
        # The first arg is either cls, for a classmethod, a self type, or none for a function
        first_arg: Literal["cls"] | TypeOrVarRef | None = None,
        cls_typevars: list[TypeVar] = [],
        is_init: bool = False,
        cls_type_and_name: Optional[tuple[type, str]] = None,
    ) -> FunctionDecl:
        if not isinstance(fn, FunctionType):
            raise NotImplementedError(
                f"Can only generate function decls for functions not {type(fn)}"
            )

        sig_globals = fn.__globals__
        if cls_type_and_name:
            sig_globals = sig_globals.copy()
            sig_globals[cls_type_and_name[1]] = cls_type_and_name[0]
        sig = signature(fn, eval_str=True, globals=sig_globals)

        # If this is an init fn use the first arg as the return type
        if is_init:
            if not isinstance(first_arg, TypeOrVarRef):
                raise ValueError("Init function must have a self type")
            return_type = first_arg
        else:
            return_type = self._resolve_type_annotation(
                sig.return_annotation, cls_typevars, cls_type_and_name
            )

        param_types = list(sig.parameters.values())
        # Remove first arg if this is a classmethod or a method, since it won't have an annotation
        if first_arg is not None:
            first, *param_types = param_types
            if first.annotation != Parameter.empty:
                raise ValueError(
                    f"First arg of a method must not have an annotation, not {first.annotation}"
                )

        for param in param_types:
            if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
                raise ValueError(
                    f"Can only register functions with positional or keyword args, not {param.kind}"
                )

        arg_types = tuple(
            self._resolve_type_annotation(t.annotation, cls_typevars, cls_type_and_name)
            for t in param_types
        )
        # If the first arg is a self, and this not an __init__ fn, add this as a typeref
        if isinstance(first_arg, TypeOrVarRef) and not is_init:
            arg_types = (first_arg,) + arg_types

        default_decl = None if default is None else default.expr
        merge_decl = (
            None
            if merge is None
            else merge(
                self._create_var(return_type, "old"),
                self._create_var(return_type, "new"),
            ).expr
        )
        decl = FunctionDecl(
            return_type=return_type,
            arg_types=arg_types,
            cost=cost,
            default=default_decl,
            merge=merge_decl,
        )
        return decl

    def _register_callable_ref(self, egg_fn: Optional[str], ref: CallableRef) -> None:
        egg_fn = egg_fn or ref.generate_egg_name()
        if egg_fn in self._decls.egg_fn_to_callable_ref:
            raise ValueError(f"Egg function {egg_fn} already registered")
        self._decls.callable_ref_to_egg_fn[ref] = egg_fn
        self._decls.egg_fn_to_callable_ref[egg_fn] = ref

    def _resolve_type_annotation(
        self,
        tp: object,
        cls_typevars: list[TypeVar],
        cls_type_and_name: Optional[tuple[type, str]],
    ) -> TypeOrVarRef:
        if isinstance(tp, TypeVar):
            return ClassTypeVarRef(cls_typevars.index(tp))
        # If there is a union, it should be of a literal and another type to allow type promotion
        if isinstance(tp, UnionType):
            args = get_args(tp)
            if len(args) != 2:
                raise TypeError("Union types are only supported for type promotion")
            fst, snd = args
            if fst in {int, str}:
                return self._resolve_type_annotation(snd, [], None)
            if snd in {int, str}:
                return self._resolve_type_annotation(fst, [], None)
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

        if isinstance(tp, RuntimeClass | RuntimeParamaterizedClass):
            return class_to_ref(tp).to_var()
        raise TypeError(f"Unexpected type annotation {tp}")

    def register(self, *values: Rewrite | Rule | Action) -> None:
        """
        Registers any number of rewrites or rules.
        """
        for value in values:
            self._register_single(value)

    def _register_single(self, value: Rewrite | Rule | Action) -> None:
        if isinstance(value, Rewrite):
            decl = value._to_decl()
            self._decls.rewrites.append(decl)
            self._on_register_rewrite(decl)
        elif isinstance(value, Rule):
            decl = value._to_decl()
            self._decls.rules.append(decl)
            self._on_register_rule(decl)
        elif isinstance(value, Action):
            decl = _action_to_decl(value)
            self._decls.actions.append(decl)
            self._on_register_action(decl)
        else:
            raise TypeError(f"Unexpected type {type(value)}")

    def _create_var(self, tp: TypeOrVarRef, name: str) -> RuntimeExpr:
        return RuntimeExpr(self._decls, tp.to_just(), VarDecl(name))


@dataclass(frozen=True)
class _WrappedMethod(Generic[P, EXPR]):
    """
    Used to wrap a method and store some extra options on it before processing it.
    """

    egg_fn: Optional[str]
    cost: Optional[int]
    default: Optional[EXPR]
    merge: Optional[Callable[[EXPR, EXPR], EXPR]]
    fn: Callable[P, EXPR]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> EXPR:
        raise NotImplementedError(
            "We should never call a wrapped method. Did you forget to wrap the class?"
        )


# We use these builders so that when creating these structures we can type check
# if the arguments are the same type of expression


def rewrite(lhs: EXPR) -> _RewriteBuilder[EXPR]:
    return _RewriteBuilder(lhs=lhs)


def eq(expr: EXPR) -> _EqBuilder[EXPR]:
    return _EqBuilder(expr)


def panic(message: str) -> Panic:
    return Panic(message)


def let(name: str, expr: BaseExpr) -> Let:
    return Let(name, expr)


def delete(expr: BaseExpr) -> Delete:
    return Delete(expr)


def union(lhs: EXPR) -> _UnionBuilder[EXPR]:
    return _UnionBuilder(lhs=lhs)


def set_(lhs: EXPR) -> _SetBuilder[EXPR]:
    return _SetBuilder(lhs=lhs)


def if_(*facts: Fact) -> _RuleBuilder:
    return _RuleBuilder(facts=facts)


def var(name: str, bound: type[EXPR]) -> EXPR:
    return cast(
        EXPR,
        RuntimeExpr(class_decls(bound), class_to_ref(cast(Any, bound)), VarDecl(name)),
    )


def vars(names: str, bound: type[EXPR]) -> Iterable[EXPR]:
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

    def then(self, *actions: Action) -> Rule:
        return Rule(actions, self.facts)


def _expr_to_decl(expr: BaseExpr) -> ExprDecl:
    assert isinstance(expr, RuntimeExpr)
    return expr.expr


def decl_to_expr(expr: ExprDecl, source_expr: EXPR) -> EXPR:
    assert isinstance(source_expr, RuntimeExpr)
    return RuntimeExpr(source_expr.decls, source_expr.tp, expr)


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
            _expr_to_decl(self.lhs),
            _expr_to_decl(self.rhs),
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
        return EqDecl(tuple(_expr_to_decl(expr) for expr in self.exprs))


Fact = Union["Unit", Eq]


def _fact_to_decl(fact: Fact) -> FactDecl:
    if isinstance(fact, Eq):
        return fact._to_decl()
    return _expr_to_decl(fact)


@dataclass
class Delete:
    expr: BaseExpr

    def __str__(self) -> str:
        return f"delete({self.expr})"

    def _to_decl(self) -> DeleteDecl:
        decl = _expr_to_decl(self.expr)
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
class Union_(Generic[EXPR]):
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"union({self.lhs}).with_({self.rhs})"

    def _to_decl(self) -> UnionDecl:
        return UnionDecl(_expr_to_decl(self.lhs), _expr_to_decl(self.rhs))


@dataclass
class Set:
    lhs: BaseExpr
    rhs: BaseExpr

    def __str__(self) -> str:
        return f"set_({self.lhs}).to({self.rhs})"

    def _to_decl(self) -> SetDecl:
        lhs = _expr_to_decl(self.lhs)
        if not isinstance(lhs, CallDecl):
            raise ValueError(
                f"Can only create a call with a call for the lhs, got {lhs}"
            )
        return SetDecl(lhs, _expr_to_decl(self.rhs))


@dataclass
class Let:
    name: str
    value: BaseExpr

    def __str__(self) -> str:
        return f"let({self.name}, {self.value})"

    def _to_decl(self) -> LetDecl:
        return LetDecl(self.name, _expr_to_decl(self.value))


Action = Union[Let, Set, Delete, Union_, Panic, "BaseExpr"]


def _action_to_decl(action: Action) -> ActionDecl:
    if isinstance(action, BaseExpr):
        return _expr_to_decl(action)
    return action._to_decl()


@dataclass
class Rule:
    header: tuple[Action, ...]
    body: tuple[Fact, ...]

    def _to_decl(self) -> RuleDecl:
        return RuleDecl(
            tuple(_action_to_decl(action) for action in self.header),
            tuple(_fact_to_decl(fact) for fact in self.body),
        )
