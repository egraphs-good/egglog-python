"""
Holds a number of types which are only used at runtime to emulate Python objects.

Users will not import anything from this module, and statically they won't know these are the types they are using.

But at runtime they will be exposed.

Note that all their internal fields are prefixed with __egg_ to avoid name collisions with user code, but will end in __
so they are not mangled by Python and can be accessed by the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from typing import TYPE_CHECKING, NoReturn, TypeVar, cast

import black
import black.parsing
from typing_extensions import assert_never

from . import bindings, config  # noqa: F401
from .declarations import *
from .declarations import BINARY_METHODS, REFLECTED_BINARY_METHODS, UNARY_METHODS
from .type_constraint_solver import *

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

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


BLACK_MODE = black.Mode(line_length=180)

UNIT_CLASS_NAME = "Unit"
UNARY_LIT_CLASS_NAMES = {"i64", "f64", "Bool", "String"}
LIT_CLASS_NAMES = UNARY_LIT_CLASS_NAMES | {UNIT_CLASS_NAME, "PyObject"}

##
# Converters
##

# Mapping from (source type, target type) to and function which takes in the runtimes values of the source and return the target
CONVERSIONS: dict[tuple[type | JustTypeRef, JustTypeRef], tuple[int, Callable]] = {}

T = TypeVar("T")
V = TypeVar("V", bound="Expr")


class ConvertError(Exception):
    pass


def converter(from_type: type[T], to_type: type[V], fn: Callable[[T], V], cost: int = 1) -> None:
    """
    Register a converter from some type to an egglog type.
    """
    to_type_name = process_tp(to_type)
    if not isinstance(to_type_name, JustTypeRef):
        raise TypeError(f"Expected return type to be a egglog type, got {to_type_name}")
    _register_converter(process_tp(from_type), to_type_name, fn, cost)


def _register_converter(a: type | JustTypeRef, b: JustTypeRef, a_b: Callable, cost: int) -> None:
    """
    Registers a converter from some type to an egglog type, if not already registered.

    Also adds transitive converters, i.e. if registering A->B and there is already B->C, then A->C will be registered.
    Also, if registering A->B and there is already D->A, then D->B will be registered.
    """
    if a == b:
        return
    if (a, b) in CONVERSIONS and CONVERSIONS[(a, b)][0] <= cost:
        return
    CONVERSIONS[(a, b)] = (cost, a_b)
    for (c, d), (other_cost, c_d) in list(CONVERSIONS.items()):
        if b == c:
            _register_converter(a, d, _ComposedConverter(a_b, c_d), cost + other_cost)
        if a == d:
            _register_converter(c, b, _ComposedConverter(c_d, a_b), cost + other_cost)


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

    def __str__(self) -> str:
        return f"{self.b_c} ∘ {self.a_b}"


def convert(source: object, target: type[V]) -> V:
    """
    Convert a source object to a target type.
    """
    target_ref = class_to_ref(cast(RuntimeTypeArgType, target))
    return cast(V, _resolve_literal(target_ref.to_var(), source))


def convert_to_same_type(source: object, target: RuntimeExpr) -> RuntimeExpr:
    """
    Convert a source object to the same type as the target.
    """
    tp = target.__egg_typed_expr__.tp
    return _resolve_literal(tp.to_var(), source)


def process_tp(tp: type | RuntimeTypeArgType) -> JustTypeRef | type:
    if isinstance(tp, RuntimeClass | RuntimeParamaterizedClass):
        return class_to_ref(tp)
    return tp


def min_convertable_tp(decls: ModuleDeclarations, a: object, b: object, name: str) -> JustTypeRef:
    """
    Returns the minimum convertable type between a and b, that has a method `name`, raising a TypeError if no such type exists.
    """
    a_tp = _get_tp(a)
    b_tp = _get_tp(b)
    a_converts_to = {
        to: c for ((from_, to), (c, _)) in CONVERSIONS.items() if from_ == a_tp and decls.has_method(to.name, name)
    }
    b_converts_to = {
        to: c for ((from_, to), (c, _)) in CONVERSIONS.items() if from_ == b_tp and decls.has_method(to.name, name)
    }
    if isinstance(a_tp, JustTypeRef):
        a_converts_to[a_tp] = 0
    if isinstance(b_tp, JustTypeRef):
        b_converts_to[b_tp] = 0
    common = set(a_converts_to) & set(b_converts_to)
    if not common:
        raise ConvertError(f"Cannot convert {a_tp} and {b_tp} to a common type")
    return min(common, key=lambda tp: a_converts_to[tp] + b_converts_to[tp])


def identity(x: object) -> object:
    return x


def _resolve_literal(tp: TypeOrVarRef, arg: object) -> RuntimeExpr:
    arg_type = _get_tp(arg)

    # If we have any type variables, dont bother trying to resolve the literal, just return the arg
    try:
        tp_just = tp.to_just()
    except NotImplementedError:
        # If this is a var, it has to be a runtime exprssions
        assert isinstance(arg, RuntimeExpr)
        return arg
    if arg_type == tp_just:
        # If the type is an egg type, it has to be a runtime expr
        assert isinstance(arg, RuntimeExpr)
        return arg
    # Try all parent types as well, if we are converting from a Python type
    for arg_type_instance in arg_type.__mro__ if isinstance(arg_type, type) else [arg_type]:
        try:
            fn = CONVERSIONS[(cast(JustTypeRef | type, arg_type_instance), tp_just)][1]
        except KeyError:
            continue
        break
    else:
        arg_type_str = arg_type.pretty() if isinstance(arg_type, JustTypeRef) else arg_type.__name__
        raise ConvertError(f"Cannot convert {arg_type_str} to {tp_just.pretty()}")
    return fn(arg)


def _get_tp(x: object) -> JustTypeRef | type:
    if isinstance(x, RuntimeExpr):
        return x.__egg_typed_expr__.tp
    tp = type(x)
    # If this value has a custom metaclass, let's use that as our index instead of the type
    if type(tp) != type:
        return type(tp)
    return tp


##
# Runtime objects
##


@dataclass
class RuntimeClass:
    __egg_decls__: ModuleDeclarations
    __egg_name__: str

    def __call__(self, *args: object) -> RuntimeExpr | None:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.__egg_name__ == "PyObject":
            assert len(args) == 1
            return RuntimeExpr(self.__egg_decls__, TypedExprDecl(JustTypeRef(self.__egg_name__), PyObjectDecl(args[0])))
        if self.__egg_name__ in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], int | float | str | bool)
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

    def __post_init__(self) -> None:
        desired_args = self.__egg_decls__.get_class_decl(self.__egg_tp__.name).n_type_vars
        if len(self.__egg_tp__.args) != desired_args:
            raise ValueError(f"Expected {desired_args} type args, got {len(self.__egg_tp__.args)}")

    def __call__(self, *args: object) -> RuntimeExpr | None:
        return RuntimeClassMethod(self.__egg_decls__, class_to_ref(self), "__init__")(*args)

    def __getattr__(self, name: str) -> RuntimeClassMethod:
        return RuntimeClassMethod(self.__egg_decls__, class_to_ref(self), name)

    def __str__(self) -> str:
        return self.__egg_tp__.pretty()


# Type args can either be typevars or classes
RuntimeTypeArgType = RuntimeClass | RuntimeParamaterizedClass


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

    def __post_init__(self) -> None:
        self.__egg_fn_ref__ = FunctionRef(self.__egg_name__)
        self.__egg_fn_decl__ = self.__egg_decls__.get_function_decl(self.__egg_fn_ref__)

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        return _call(self.__egg_decls__, self.__egg_fn_ref__, self.__egg_fn_decl__, args, kwargs)

    def __str__(self) -> str:
        return self.__egg_name__


def _call(
    decls: ModuleDeclarations,
    callable_ref: CallableRef,
    # Not included if this is the != method
    fn_decl: FunctionDecl | None,
    args: Collection[object],
    kwargs: dict[str, object],
    bound_params: tuple[JustTypeRef, ...] | None = None,
) -> RuntimeExpr | None:
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
            _resolve_literal(cast(TypeOrVarRef, tp), arg)
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

    def __post_init__(self) -> None:
        self.__egg_callable_ref__ = ClassMethodRef(self.class_name, self.__egg_method_name__)
        try:
            self.__egg_fn_decl__ = self.__egg_decls__.get_function_decl(self.__egg_callable_ref__)
        except KeyError as e:
            raise AttributeError(f"Class {self.class_name} does not have method {self.__egg_method_name__}") from e

    def __call__(self, *args: object, **kwargs) -> RuntimeExpr | None:
        bound_params = self.__egg_tp__.args if isinstance(self.__egg_tp__, JustTypeRef) else None
        return _call(self.__egg_decls__, self.__egg_callable_ref__, self.__egg_fn_decl__, args, kwargs, bound_params)

    def __str__(self) -> str:
        return f"{self.class_name}.{self.__egg_method_name__}"

    @property
    def class_name(self) -> str:
        if isinstance(self.__egg_tp__, str):
            return self.__egg_tp__
        return self.__egg_tp__.name


# All methods which should return NotImplemented if they fail to resolve
# From https://docs.python.org/3/reference/datamodel.html
PARTIAL_METHODS = {
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
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
}


@dataclass
class RuntimeMethod:
    __egg_self__: RuntimeExpr
    __egg_method_name__: str
    __egg_callable_ref__: MethodRef | PropertyRef = field(init=False)
    __egg_fn_decl__: FunctionDecl | None = field(init=False)

    def __post_init__(self) -> None:
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
            except KeyError as e:
                raise AttributeError(f"Class {self.class_name} does not have method {self.__egg_method_name__}") from e

    def __call__(self, *args: object, **kwargs) -> RuntimeExpr | None:
        args = (self.__egg_self__, *args)
        try:
            return _call(self.__egg_self__.__egg_decls__, self.__egg_callable_ref__, self.__egg_fn_decl__, args, kwargs)
        except ConvertError as e:
            name = self.__egg_method_name__
            raise TypeError(f"Wrong types for {self.__egg_self__.__egg_typed_expr__.tp.pretty()}.{name}") from e

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
        context.traverse_for_parents(self.__egg_typed_expr__.expr)
        pretty_expr = self.__egg_typed_expr__.expr.pretty(context, parens=False)
        try:
            if config.SHOW_TYPES:
                raise NotImplementedError
                # s = f"_: {self.__egg_typed_expr__.tp.pretty()} = {pretty_expr}"
                # return black.format_str(s, mode=black.FileMode()).strip()
            pretty_statements = context.render(pretty_expr)
            return black.format_str(pretty_statements, mode=BLACK_MODE).strip()
        except black.parsing.InvalidInput:
            return pretty_expr

    def _ipython_display_(self) -> None:
        from IPython.display import Code, display

        display(Code(str(self), language="python"))

    def __dir__(self) -> Iterable[str]:
        return list(self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).methods)

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    def __eq__(self, other: NoReturn) -> Expr:  # type: ignore[override]
        msg = "Do not use == on RuntimeExpr. Compare the __egg_typed_expr__ attribute instead for structural equality."
        raise NotImplementedError(msg)

    # Implement these so that copy() works on this object
    # otherwise copy will try to call `__getstate__` before object is initialized with properties which will cause inifinite recursion

    def __getstate__(self) -> tuple[ModuleDeclarations, TypedExprDecl]:
        return (self.__egg_decls__, self.__egg_typed_expr__)

    def __setstate__(self, d: tuple[ModuleDeclarations, TypedExprDecl]) -> None:
        self.__egg_decls__, self.__egg_typed_expr__ = d

    def __hash__(self) -> int:
        return hash(self.__egg_typed_expr__)


# Define each of the special methods, since we have already declared them for pretty printing
for name in list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__", "__call__", "__setitem__", "__delitem__"]:

    def _special_method(self: RuntimeExpr, *args: object, __name: str = name) -> RuntimeExpr | None:
        # First, try to resolve as preserved method
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
        except KeyError:
            # If this is a "partial" method meaning that it can return NotImplemented,
            # we want to find the "best" superparent (lowest cost) of the arg types to call with it, instead of just
            # using the arg type of the self arg.
            # This is neccesary so if we add like an int to a ndarray, it will upcast the int to an ndarray, instead of vice versa.
            if __name in PARTIAL_METHODS:
                try:
                    return call_method_min_conversion(self, args[0], __name)
                except ConvertError:
                    return NotImplemented
            return RuntimeMethod(self, __name)(*args)
        else:
            return method(self, *args)

    setattr(RuntimeExpr, name, _special_method)

# For each of the reflected binary methods, translate to the corresponding non-reflected method
for reflected, non_reflected in REFLECTED_BINARY_METHODS.items():

    def _reflected_method(self: RuntimeExpr, other: object, __non_reflected: str = non_reflected) -> RuntimeExpr | None:
        # All binary methods are also "partial" meaning we should try to upcast first.
        return call_method_min_conversion(other, self, __non_reflected)

    setattr(RuntimeExpr, reflected, _reflected_method)


def call_method_min_conversion(slf: object, other: object, name: str) -> RuntimeExpr | None:
    # Use the mod decls that is most general between the args, if both of them are expressions
    mod_decls = get_general_decls(slf, other)
    # find a minimum type that both can be converted to
    # This is so so that calls like `-0.1 * Int("x")` work by upcasting both to floats.
    min_tp = min_convertable_tp(mod_decls, slf, other, name)
    slf = _resolve_literal(min_tp.to_var(), slf)
    other = _resolve_literal(min_tp.to_var(), other)
    method = RuntimeMethod(slf, name)
    return method(other)


def get_general_decls(a: object, b: object) -> ModuleDeclarations:
    """
    Returns the more general module declerations between the two, if both are expressions.
    """
    if isinstance(a, RuntimeExpr) and isinstance(b, RuntimeExpr):
        return ModuleDeclarations.parent_decl(a.__egg_decls__, b.__egg_decls__)
    if isinstance(a, RuntimeExpr):
        return a.__egg_decls__
    assert isinstance(b, RuntimeExpr)
    return b.__egg_decls__


for name in ["__bool__", "__len__", "__complex__", "__int__", "__float__", "__iter__", "__index__"]:

    def _preserved_method(self: RuntimeExpr, __name: str = name):
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
        except KeyError as e:
            raise TypeError(f"{self.__egg_typed_expr__.tp.name} has no method {__name}") from e
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
