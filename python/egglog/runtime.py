"""
This module holds a number of types which are only used at runtime to emulate Python objects.

Users will not import anything from this module, and statically they won't know these are the types they are using.

But at runtime they will be exposed.

Note that all their internal fields are prefixed with __egg_ to avoid name collisions with user code, but will end in __
so they are not mangled by Python and can be accessed by the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import black
import black.parsing
from typing_extensions import assert_never

from . import bindings, config  # noqa: F401
from .declarations import *
from .declarations import BINARY_METHODS, REFLECTED_BINARY_METHODS, UNARY_METHODS
from .type_constraint_solver import *

if TYPE_CHECKING:
    from .egraph import Expr

__all__ = [
    "LIT_CLASS_NAMES",
    "RuntimeClass",
    "RuntimeParamaterizedClass",
    "RuntimeClassMethod",
    "RuntimeExpr",
    "RuntimeFunction",
    "convert",
    "converter",
]


BLACK_MODE = black.Mode(line_length=180)  # type: ignore

UNIT_CLASS_NAME = "Unit"
UNARY_LIT_CLASS_NAMES = {"i64", "f64", "String"}
LIT_CLASS_NAMES = UNARY_LIT_CLASS_NAMES | {UNIT_CLASS_NAME}


# Mapping of types, from Python or egglog types to egglog types
CONVERSIONS: dict[tuple[Type | JustTypeRef, JustTypeRef], Callable] = {}

T = TypeVar("T")
V = TypeVar("V", bound="Expr")


def converter(from_type: Type[T], to_type: Type[V], fn: Callable[[T], V]) -> None:
    """
    Register a converter from some type to an egglog type.
    """
    to_type_name = process_tp(to_type)
    if not isinstance(to_type_name, JustTypeRef):
        raise TypeError(f"Expected return type to be a egglog type, got {to_type_name}")
    _register_converter(process_tp(from_type), to_type_name, fn)


def _register_converter(a: Type | JustTypeRef, b: JustTypeRef, a_b: Callable) -> None:
    """
    Registers a converter from some type to an egglog type, if not already registered.

    Also adds transitive converters, i.e. if registering A->B and there is already B->C, then A->C will be registered.
    Also, if registering A->B and there is already D->A, then D->B will be registered.
    """
    if a == b or (a, b) in CONVERSIONS:
        return
    CONVERSIONS[(a, b)] = a_b
    for (c, d), c_d in list(CONVERSIONS.items()):
        if b == c:
            _register_converter(a, d, _ComposedConverter(a_b, c_d))
        if a == d:
            _register_converter(c, b, _ComposedConverter(c_d, a_b))


@dataclass
class _ComposedConverter:
    """
    A converter which is composed of multiple converters.

    _ComposeConverter(a_b, b_c) is equivalent to lambda x: b_c(a_b(x))

    We use the dataclass instead of the lambda to make it easier to debug.
    """

    a_b: Callable
    b_c: Callable

    def __call__(self, x: object) -> object:
        return self.b_c(self.a_b(x))


def convert(source: object, target: type[V]) -> V:
    """
    Convert a source object to a target type.
    """
    target_ref = class_to_ref(target)  # type: ignore
    return cast(V, _resolve_literal(target_ref.to_var(), source))


def process_tp(tp: type | RuntimeTypeArgType) -> JustTypeRef | type:
    if isinstance(tp, (RuntimeClass, RuntimeParamaterizedClass)):
        return class_to_ref(tp)
    return tp


def _resolve_literal(tp: TypeOrVarRef, arg: object) -> RuntimeExpr:
    arg_type: JustTypeRef | type
    if isinstance(arg, RuntimeExpr):
        arg_type = arg.__egg_typed_expr__.tp
    else:
        arg_type = type(arg)
        # If this value has a custom metaclass, let's use that as our index instead of the type
        if type(arg_type) != type:  # noqa: E721
            arg_type = type(arg_type)

    # If we have any type variables, dont bother trying to resolve the literal, just return the arg
    try:
        tp_just = tp.to_just()
    except Exception:
        return arg  # type: ignore
    if arg_type == tp_just:
        return arg  # type: ignore
    try:
        fn = CONVERSIONS[(arg_type, tp_just)]
    except KeyError:
        raise TypeError(f"Cannot convert {arg} ({repr(arg_type)}) to {tp}")
    return fn(arg)


@dataclass
class RuntimeClass:
    __egg_decls__: ModuleDeclarations
    __egg_name__: str

    def __call__(self, *args: object) -> Optional[RuntimeExpr]:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.__egg_name__ in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], (int, float, str))
            return RuntimeExpr(self.__egg_decls__, TypedExprDecl(JustTypeRef(self.__egg_name__), LitDecl(args[0])))
        if self.__egg_name__ == UNIT_CLASS_NAME:
            assert len(args) == 0
            return RuntimeExpr(self.__egg_decls__, TypedExprDecl(JustTypeRef(self.__egg_name__), LitDecl(None)))

        return RuntimeClassMethod(self.__egg_decls__, self.__egg_name__, "__init__")(*args)

    def __dir__(self) -> list[str]:
        cls_decl = self.__egg_decls__.get_class_decl(self.__egg_name__)
        possible_methods = (
            list(cls_decl.class_methods) + list(cls_decl.class_variables) + list(cls_decl.preserved_methods)
        )
        if "__init__" in possible_methods:
            possible_methods.remove("__init__")
            possible_methods.append("__call__")
        return possible_methods

    def __getitem__(self, args: tuple[RuntimeTypeArgType, ...] | RuntimeTypeArgType) -> RuntimeParamaterizedClass:
        if not isinstance(args, tuple):
            args = (args,)
        tp = JustTypeRef(self.__egg_name__, tuple(class_to_ref(arg) for arg in args))
        return RuntimeParamaterizedClass(self.__egg_decls__, tp)

    def __getattr__(self, name: str) -> RuntimeClassMethod | RuntimeExpr | Callable:
        cls_decl = self.__egg_decls__.get_class_decl(self.__egg_name__)

        preserved_methods = cls_decl.preserved_methods
        if name in preserved_methods:
            return preserved_methods[name].__get__(self)

        # if this is a class variable, return an expr for it, otherwise, assume it's a method
        if name in cls_decl.class_variables:
            return_tp = cls_decl.class_variables[name]
            return RuntimeExpr(
                self.__egg_decls__, TypedExprDecl(return_tp, CallDecl(ClassVariableRef(self.__egg_name__, name)))
            )
        return RuntimeClassMethod(self.__egg_decls__, self.__egg_name__, name)

    def __str__(self) -> str:
        return self.__egg_name__

    # Make hashable so can go in Union
    def __hash__(self) -> int:
        return hash((id(self.__egg_decls__), self.__egg_name__))


@dataclass
class RuntimeParamaterizedClass:
    __egg_decls__: ModuleDeclarations
    # Note that this will never be a typevar because we don't use RuntimeParamaterizedClass for maps on their own methods
    # which is the only time we define function which take typevars
    __egg_tp__: JustTypeRef

    def __post_init__(self):
        desired_args = self.__egg_decls__.get_class_decl(self.__egg_tp__.name).n_type_vars
        if len(self.__egg_tp__.args) != desired_args:
            raise ValueError(f"Expected {desired_args} type args, got {len(self.__egg_tp__.args)}")

    def __call__(self, *args: object) -> Optional[RuntimeExpr]:
        return RuntimeClassMethod(self.__egg_decls__, class_to_ref(self), "__init__")(*args)

    def __getattr__(self, name: str) -> RuntimeClassMethod:
        return RuntimeClassMethod(self.__egg_decls__, class_to_ref(self), name)

    def __str__(self) -> str:
        return self.__egg_tp__.pretty()


# Type args can either be typevars or classes
RuntimeTypeArgType = Union[RuntimeClass, RuntimeParamaterizedClass]


def class_to_ref(cls: RuntimeTypeArgType) -> JustTypeRef:
    if isinstance(cls, RuntimeClass):
        return JustTypeRef(cls.__egg_name__)
    if isinstance(cls, RuntimeParamaterizedClass):
        return cls.__egg_tp__
    assert_never(cls)


@dataclass
class RuntimeFunction:
    __egg_decls__: ModuleDeclarations
    __egg_name__: str
    __egg_fn_ref__: FunctionRef = field(init=False)
    __egg_fn_decl__: FunctionDecl = field(init=False)

    def __post_init__(self):
        self.__egg_fn_ref__ = FunctionRef(self.__egg_name__)
        self.__egg_fn_decl__ = self.__egg_decls__.get_function_decl(self.__egg_fn_ref__)

    def __call__(self, *args: object, **kwargs: object) -> Optional[RuntimeExpr]:
        return _call(self.__egg_decls__, self.__egg_fn_ref__, self.__egg_fn_decl__, args, kwargs)

    def __str__(self) -> str:
        return self.__egg_name__


def _call(
    decls: ModuleDeclarations,
    callable_ref: CallableRef,
    # Not included if this is the != method
    fn_decl: Optional[FunctionDecl],
    args: Collection[object],
    kwargs: dict[str, object],
    bound_params: Optional[tuple[JustTypeRef, ...]] = None,
) -> Optional[RuntimeExpr]:
    # Turn all keyword args into positional args

    if fn_decl:
        bound = fn_decl.to_signature(lambda expr: RuntimeExpr(decls, expr)).bind(*args, **kwargs)
        bound.apply_defaults()
        assert not bound.kwargs
        args = bound.args
        mutates_first_arg = fn_decl.mutates_first_arg
    else:
        assert not kwargs
        mutates_first_arg = False
    upcasted_args: list[RuntimeExpr]
    if fn_decl is not None:
        upcasted_args = [
            _resolve_literal(tp, arg)  # type: ignore
            for arg, tp in zip_longest(args, fn_decl.arg_types, fillvalue=fn_decl.var_arg_type)
        ]
    else:
        upcasted_args = cast("list[RuntimeExpr]", args)
    arg_decls = tuple(arg.__egg_typed_expr__ for arg in upcasted_args)

    arg_types = [decl.tp for decl in arg_decls]

    if bound_params is not None:
        tcs = TypeConstraintSolver.from_type_parameters(bound_params)
    else:
        tcs = TypeConstraintSolver()

    if fn_decl is not None:
        return_tp = tcs.infer_return_type(fn_decl.arg_types, fn_decl.return_type, fn_decl.var_arg_type, arg_types)
    else:
        return_tp = JustTypeRef("Unit")

    expr_decl = CallDecl(callable_ref, arg_decls, bound_params)
    typed_expr_decl = TypedExprDecl(return_tp, expr_decl)
    if mutates_first_arg:
        first_arg = upcasted_args[0]
        first_arg.__egg_typed_expr__ = typed_expr_decl
        first_arg.__egg_decls__ = decls
        return None
    return RuntimeExpr(decls, TypedExprDecl(return_tp, expr_decl))


@dataclass
class RuntimeClassMethod:
    __egg_decls__: ModuleDeclarations
    # Either a string if it isn't bound or a tp if it s
    __egg_tp__: JustTypeRef | str
    __egg_method_name__: str
    __egg_callable_ref__: ClassMethodRef = field(init=False)
    __egg_fn_decl__: FunctionDecl = field(init=False)

    def __post_init__(self):
        self.__egg_callable_ref__ = ClassMethodRef(self.class_name, self.__egg_method_name__)
        try:
            self.__egg_fn_decl__ = self.__egg_decls__.get_function_decl(self.__egg_callable_ref__)
        except KeyError:
            raise AttributeError(f"Class {self.class_name} does not have method {self.__egg_method_name__}")

    def __call__(self, *args: object, **kwargs) -> Optional[RuntimeExpr]:
        bound_params = self.__egg_tp__.args if isinstance(self.__egg_tp__, JustTypeRef) else None
        return _call(self.__egg_decls__, self.__egg_callable_ref__, self.__egg_fn_decl__, args, kwargs, bound_params)

    def __str__(self) -> str:
        return f"{self.class_name}.{self.__egg_method_name__}"

    @property
    def class_name(self) -> str:
        if isinstance(self.__egg_tp__, str):
            return self.__egg_tp__
        return self.__egg_tp__.name


@dataclass
class RuntimeMethod:
    __egg_self__: RuntimeExpr
    __egg_method_name__: str
    __egg_callable_ref__: MethodRef | PropertyRef = field(init=False)
    __egg_fn_decl__: Optional[FunctionDecl] = field(init=False)

    def __post_init__(self):
        if self.__egg_method_name__ in self.__egg_self__.__egg_decls__.get_class_decl(self.class_name).properties:
            self.__egg_callable_ref__ = PropertyRef(self.class_name, self.__egg_method_name__)
        else:
            self.__egg_callable_ref__ = MethodRef(self.class_name, self.__egg_method_name__)
        # Special case for __ne__ which does not have a normal function defintion since
        # it relies of type parameters
        if self.__egg_method_name__ == "__ne__":
            self.__egg_fn_decl__ = None
        else:
            try:
                self.__egg_fn_decl__ = self.__egg_self__.__egg_decls__.get_function_decl(self.__egg_callable_ref__)
            except KeyError:
                raise AttributeError(f"Class {self.class_name} does not have method {self.__egg_method_name__}")

    def __call__(self, *args: object, **kwargs) -> Optional[RuntimeExpr]:
        args = (self.__egg_self__, *args)
        return _call(self.__egg_self__.__egg_decls__, self.__egg_callable_ref__, self.__egg_fn_decl__, args, kwargs)

    @property
    def class_name(self) -> str:
        return self.__egg_self__.__egg_typed_expr__.tp.name


@dataclass
class RuntimeExpr:
    __egg_decls__: ModuleDeclarations
    __egg_typed_expr__: TypedExprDecl

    def __getattr__(self, name: str) -> RuntimeMethod | RuntimeExpr | Callable | None:
        class_decl = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name)

        preserved_methods = class_decl.preserved_methods
        if name in preserved_methods:
            return preserved_methods[name].__get__(self)

        method = RuntimeMethod(self, name)
        if isinstance(method.__egg_callable_ref__, PropertyRef):
            return method()
        return method

    def __repr__(self) -> str:
        """
        The repr of the expr is the pretty printed version of the expr.
        """
        return str(self)

    def __str__(self) -> str:
        context = PrettyContext(self.__egg_decls__)
        pretty_expr = self.__egg_typed_expr__.expr.pretty(context, parens=False)
        try:
            if config.SHOW_TYPES:
                raise NotImplementedError()
                # s = f"_: {self.__egg_typed_expr__.tp.pretty()} = {pretty_expr}"
                # return black.format_str(s, mode=black.FileMode()).strip()
            else:
                pretty_statements = context.render(pretty_expr)
                return black.format_str(pretty_statements, mode=BLACK_MODE).strip()
        except black.parsing.InvalidInput:
            return pretty_expr

    def __dir__(self) -> Iterable[str]:
        return list(self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).methods)

    def __to_egg__(self) -> bindings._Expr:
        return self.__egg_typed_expr__.expr.to_egg(self.__egg_decls__)

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    def __eq__(self, other: NoReturn) -> Expr:  # type: ignore
        raise NotImplementedError(
            "Do not use == on RuntimeExpr. Compare the __egg_typed_expr__ attribute instead for structural equality."
        )

    # Implement these so that copy() works on this object
    # otherwise copy will try to call `__getstate__` before object is initialized with properties which will cause inifinite recursion

    def __getstate__(self):
        return (self.__egg_decls__, self.__egg_typed_expr__)

    def __setstate__(self, d):
        self.__egg_decls__, self.__egg_typed_expr__ = d


# Define each of the special methods, since we have already declared them for pretty printing
for name in list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__", "__call__", "__setitem__", "__delitem__"]:

    def _special_method(self: RuntimeExpr, *args: object, __name: str = name) -> Optional[RuntimeExpr]:
        # First, try to resolve as preserved method
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
        except KeyError:
            return RuntimeMethod(self, __name)(*args)
        else:
            return method(self, *args)

    setattr(RuntimeExpr, name, _special_method)

# For each of the reflected binary methods, translate to the corresponding non-reflected method
for name, normal in REFLECTED_BINARY_METHODS.items():

    def _reflected_method(self: RuntimeExpr, other: object, __normal: str = normal) -> Optional[RuntimeExpr]:
        egg_tp = self.__egg_typed_expr__.tp.to_var()
        converted_other = _resolve_literal(egg_tp, other)
        return RuntimeMethod(converted_other, __normal)(self)

    setattr(RuntimeExpr, name, _reflected_method)

for name in ["__bool__", "__len__", "__complex__", "__int__", "__float__", "__hash__", "__iter__", "__index__"]:

    def _preserved_method(self: RuntimeExpr, __name: str = name):
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
        except KeyError:
            raise TypeError(f"{self.__egg_typed_expr__.tp.name} has no method {__name}")
        return method(self)

    setattr(RuntimeExpr, name, _preserved_method)


def _resolve_callable(callable: object) -> CallableRef:
    """
    Resolves a runtime callable into a ref
    """
    if isinstance(callable, RuntimeFunction):
        return FunctionRef(callable.__egg_name__)
    if isinstance(callable, RuntimeClassMethod):
        return ClassMethodRef(callable.class_name, callable.__egg_method_name__)
    if isinstance(callable, RuntimeMethod):
        return MethodRef(callable.__egg_self__.__egg_typed_expr__.tp.name, callable.__egg_method_name__)
    if isinstance(callable, RuntimeClass):
        return ClassMethodRef(callable.__egg_name__, "__init__")
    raise NotImplementedError(f"Cannot turn {callable} into a callable ref")
