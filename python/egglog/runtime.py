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
from typing import TYPE_CHECKING, NoReturn, TypeVar, Union, cast, get_args, get_origin

import black
import black.parsing
from typing_extensions import assert_never

from . import bindings, config
from .declarations import *
from .declarations import BINARY_METHODS, REFLECTED_BINARY_METHODS, UNARY_METHODS
from .type_constraint_solver import *

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

    from .egraph import Expr

__all__ = [
    "LIT_CLASS_NAMES",
    "class_to_ref",
    "resolve_literal",
    "resolve_callable",
    "resolve_type_annotation",
    "convert_to_same_type",
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

# Set this globally so we can get access to PyObject when we have a type annotation of just object.
# This is the only time a type annotation doesn't need to include the egglog type b/c object is top so that would be redundant statically.
_PY_OBJECT_CLASS: RuntimeClass | None = None

##
# Converters
##

# Mapping from (source type, target type) to and function which takes in the runtimes values of the source and return the target
CONVERSIONS: dict[tuple[type | JustTypeRef, JustTypeRef], tuple[int, Callable]] = {}
# Global declerations to store all convertable types so we can query if they have certain methods or not
CONVERSIONS_DECLS = Declarations()

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
    return cast(V, resolve_literal(target_ref.to_var(), source))


def convert_to_same_type(source: object, target: RuntimeExpr) -> RuntimeExpr:
    """
    Convert a source object to the same type as the target.
    """
    tp = target.__egg_typed_expr__.tp
    return resolve_literal(tp.to_var(), source)


def process_tp(tp: type | RuntimeTypeArgType) -> JustTypeRef | type:
    """
    Process a type before converting it, to add it to the global declerations and resolve to a ref.
    """
    global CONVERSIONS_DECLS
    if isinstance(tp, RuntimeClass | RuntimeParamaterizedClass):
        CONVERSIONS_DECLS |= tp
        return class_to_ref(tp)
    return tp


def min_convertable_tp(a: object, b: object, name: str) -> JustTypeRef:
    """
    Returns the minimum convertable type between a and b, that has a method `name`, raising a TypeError if no such type exists.
    """
    a_tp = _get_tp(a)
    b_tp = _get_tp(b)
    a_converts_to = {
        to: c
        for ((from_, to), (c, _)) in CONVERSIONS.items()
        if from_ == a_tp and CONVERSIONS_DECLS.has_method(to.name, name)
    }
    b_converts_to = {
        to: c
        for ((from_, to), (c, _)) in CONVERSIONS.items()
        if from_ == b_tp and CONVERSIONS_DECLS.has_method(to.name, name)
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


def resolve_type_annotation(decls: Declarations, tp: object) -> TypeOrVarRef:
    """
    Resolves a type object into a type reference.
    """
    if isinstance(tp, TypeVar):
        return ClassTypeVarRef(tp.__name__)
    # If there is a union, then we assume the first item is the type we want, and the others are types that can be converted to that type.
    if get_origin(tp) == Union:
        first, *_rest = get_args(tp)
        return resolve_type_annotation(decls, first)

    # If the type is `object` then this is assumed to be a PyObjectLike, i.e. converted into a PyObject
    if tp == object:
        assert _PY_OBJECT_CLASS
        return resolve_type_annotation(decls, _PY_OBJECT_CLASS)
    if isinstance(tp, RuntimeClass):
        decls |= tp
        return tp.__egg_tp__.to_var()
    if isinstance(tp, RuntimeParamaterizedClass):
        decls |= tp
        return tp.__egg_tp__
    raise TypeError(f"Unexpected type annotation {tp}")


def resolve_literal(tp: TypeOrVarRef, arg: object) -> RuntimeExpr:
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
    # Pass in a constructor to make the declarations lazy, so we can have classes reference each other in their type constructors
    # This function should mutate the declerations and add to them
    # Used this instead of a lazy property so we can have a reference to the decls in the class as its computing
    lazy_decls: Callable[[Declarations], None] = field(repr=False)
    # Cached declerations
    _inner_decls: Declarations | None = field(init=False, repr=False, default=None)
    __egg_name__: str

    def __post_init__(self) -> None:
        global _PY_OBJECT_CLASS
        if self.__egg_name__ == "PyObject":
            _PY_OBJECT_CLASS = self

    @property
    def __egg_decls__(self) -> Declarations:
        if self._inner_decls is None:
            # Set it like this so we can have a reference to the decls in the class as its computing
            self._inner_decls = Declarations()
            self.lazy_decls(self._inner_decls)
        return self._inner_decls

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.__egg_name__ == "PyObject":
            assert len(args) == 1
            return RuntimeExpr(self.__egg_decls__, TypedExprDecl(self.__egg_tp__, PyObjectDecl(args[0])))
        if self.__egg_name__ in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], int | float | str | bool)
            return RuntimeExpr(self.__egg_decls__, TypedExprDecl(self.__egg_tp__, LitDecl(args[0])))
        if self.__egg_name__ == UNIT_CLASS_NAME:
            assert len(args) == 0
            return RuntimeExpr(self.__egg_decls__, TypedExprDecl(self.__egg_tp__, LitDecl(None)))

        return RuntimeClassMethod(self.__egg_decls__, self.__egg_tp__, "__init__")(*args, **kwargs)

    def __dir__(self) -> list[str]:
        cls_decl = self.__egg_decls__.get_class_decl(self.__egg_name__)
        possible_methods = (
            list(cls_decl.class_methods) + list(cls_decl.class_variables) + list(cls_decl.preserved_methods)
        )
        if "__init__" in possible_methods:
            possible_methods.remove("__init__")
            possible_methods.append("__call__")
        return possible_methods

    def __getitem__(self, args: object) -> RuntimeParamaterizedClass:
        if not isinstance(args, tuple):
            args = (args,)
        decls = self.__egg_decls__.copy()
        tp = TypeRefWithVars(self.__egg_name__, tuple(resolve_type_annotation(decls, arg) for arg in args))
        return RuntimeParamaterizedClass(self.__egg_decls__, tp)

    def __getattr__(self, name: str) -> RuntimeClassMethod | RuntimeExpr | Callable:
        # Special case some names that don't exist so we can exit early without resolving decls
        # Important so if we take union of RuntimeClass it won't try to resolve decls
        if name in {
            "__typing_subst__",
            "__parameters__",
            # Origin is used in get_type_hints which is used when resolving the class itself
            "__origin__",
        }:
            raise AttributeError

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
        return RuntimeClassMethod(self.__egg_decls__, self.__egg_tp__, name)

    def __str__(self) -> str:
        return self.__egg_name__

    # Make hashable so can go in Union
    def __hash__(self) -> int:
        return hash((id(self.lazy_decls), self.__egg_name__))

    # Support unioning like types
    def __or__(self, __value: type) -> object:
        return Union[self, __value]  # noqa: UP007

    @property
    def __egg_tp__(self) -> JustTypeRef:
        return JustTypeRef(self.__egg_name__)


@dataclass
class RuntimeParamaterizedClass:
    __egg_decls__: Declarations
    __egg_tp__: TypeRefWithVars

    def __post_init__(self) -> None:
        desired_args = self.__egg_decls__.get_class_decl(self.__egg_tp__.name).type_vars
        if len(self.__egg_tp__.args) != len(desired_args):
            raise ValueError(f"Expected {desired_args} type args, got {len(self.__egg_tp__.args)}")

    def __call__(self, *args: object) -> RuntimeExpr | None:
        return RuntimeClassMethod(self.__egg_decls__, class_to_ref(self), "__init__")(*args)

    def __getattr__(self, name: str) -> RuntimeClassMethod | RuntimeClass:
        # Special case so when get_type_annotations proccessed it can work
        if name in {"__origin__"}:
            return RuntimeClass(self.__egg_decls__.update_other, self.__egg_tp__.name)
        return RuntimeClassMethod(self.__egg_decls__, class_to_ref(self), name)

    def __str__(self) -> str:
        return self.__egg_tp__.pretty()

    # Support unioning
    def __or__(self, __value: type) -> object:
        return Union[self, __value]  # noqa: UP007


# Type args can either be typevars or classes
RuntimeTypeArgType = RuntimeClass | RuntimeParamaterizedClass


def class_to_ref(cls: RuntimeTypeArgType) -> JustTypeRef:
    if isinstance(cls, RuntimeClass):
        return JustTypeRef(cls.__egg_name__)
    if isinstance(cls, RuntimeParamaterizedClass):
        # Currently this is used when calling methods on a parametrized class, which is only possible when we
        # have actualy types currently, not typevars, currently.
        return cls.__egg_tp__.to_just()
    assert_never(cls)


@dataclass
class RuntimeFunction:
    __egg_decls__: Declarations
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
    decls_from_fn: Declarations,
    callable_ref: CallableRef,
    fn_decl: FunctionDecl,
    args: Collection[object],
    kwargs: dict[str, object],
    bound_class: JustTypeRef | None = None,
) -> RuntimeExpr | None:
    # Turn all keyword args into positional args
    bound = fn_decl.to_signature(lambda expr: RuntimeExpr(decls_from_fn, expr)).bind(*args, **kwargs)
    bound.apply_defaults()
    assert not bound.kwargs
    del args, kwargs

    upcasted_args = [
        resolve_literal(cast(TypeOrVarRef, tp), arg)
        for arg, tp in zip_longest(bound.args, fn_decl.arg_types, fillvalue=fn_decl.var_arg_type)
    ]

    arg_exprs = tuple(arg.__egg_typed_expr__ for arg in upcasted_args)
    decls = Declarations.create(decls_from_fn, *upcasted_args)

    tcs = TypeConstraintSolver(decls)
    if bound_class is not None and bound_class.args:
        tcs.bind_class(bound_class)

    if fn_decl is not None:
        arg_types = [expr.tp for expr in arg_exprs]
        cls_name = bound_class.name if bound_class is not None else None
        return_tp = tcs.infer_return_type(
            fn_decl.arg_types, fn_decl.return_type, fn_decl.var_arg_type, arg_types, cls_name
        )
    else:
        return_tp = JustTypeRef("Unit")
    bound_params = cast(JustTypeRef, bound_class).args if isinstance(callable_ref, ClassMethodRef) else None
    expr_decl = CallDecl(callable_ref, arg_exprs, bound_params)
    typed_expr_decl = TypedExprDecl(return_tp, expr_decl)
    # Register return type sort in case it's a variadic generic that needs to be created
    decls.register_sort(return_tp, False)
    if fn_decl.mutates_first_arg:
        first_arg = upcasted_args[0]
        first_arg.__egg_typed_expr__ = typed_expr_decl
        first_arg.__egg_decls__ = decls
        return None
    return RuntimeExpr(decls, TypedExprDecl(return_tp, expr_decl))


@dataclass
class RuntimeClassMethod:
    __egg_decls__: Declarations
    __egg_tp__: JustTypeRef
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
        return _call(self.__egg_decls__, self.__egg_callable_ref__, self.__egg_fn_decl__, args, kwargs, self.__egg_tp__)

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
    __egg_fn_decl__: FunctionDecl = field(init=False, repr=False)
    __egg_decls__: Declarations = field(init=False)

    def __post_init__(self) -> None:
        self.__egg_decls__ = self.__egg_self__.__egg_decls__
        if self.__egg_method_name__ in self.__egg_decls__.get_class_decl(self.class_name).properties:
            self.__egg_callable_ref__ = PropertyRef(self.class_name, self.__egg_method_name__)
        else:
            self.__egg_callable_ref__ = MethodRef(self.class_name, self.__egg_method_name__)
        try:
            self.__egg_fn_decl__ = self.__egg_decls__.get_function_decl(self.__egg_callable_ref__)
        except KeyError:
            msg = f"Class {self.class_name} does not have method {self.__egg_method_name__}"
            if self.__egg_method_name__ == "__ne__":
                msg += ". Did you mean to use the ne(...).to(...)?"
            raise AttributeError(msg) from None

    def __call__(self, *args: object, **kwargs) -> RuntimeExpr | None:
        args = (self.__egg_self__, *args)
        try:
            return _call(
                self.__egg_decls__,
                self.__egg_callable_ref__,
                self.__egg_fn_decl__,
                args,
                kwargs,
                self.__egg_self__.__egg_typed_expr__.tp,
            )
        except ConvertError as e:
            name = self.__egg_method_name__
            raise TypeError(f"Wrong types for {self.__egg_self__.__egg_typed_expr__.tp.pretty()}.{name}") from e

    @property
    def class_name(self) -> str:
        return self.__egg_self__.__egg_typed_expr__.tp.name


@dataclass
class RuntimeExpr:
    __egg_decls__: Declarations
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

    @property
    def __egg__(self) -> bindings._Expr:
        return self.__egg_typed_expr__.to_egg(self.__egg_decls__)

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    def __eq__(self, other: NoReturn) -> Expr:  # type: ignore[override]
        msg = "Do not use == on RuntimeExpr. Compare the __egg_typed_expr__ attribute instead for structural equality."
        raise NotImplementedError(msg)

    # Implement these so that copy() works on this object
    # otherwise copy will try to call `__getstate__` before object is initialized with properties which will cause inifinite recursion

    def __getstate__(self) -> tuple[Declarations, TypedExprDecl]:
        return (self.__egg_decls__, self.__egg_typed_expr__)

    def __setstate__(self, d: tuple[Declarations, TypedExprDecl]) -> None:
        self.__egg_decls__, self.__egg_typed_expr__ = d

    def __hash__(self) -> int:
        return hash(self.__egg_typed_expr__)


# Define each of the special methods, since we have already declared them for pretty printing
for name in list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__", "__call__", "__setitem__", "__delitem__"]:

    def _special_method(
        self: RuntimeExpr,
        *args: object,
        __name: str = name,
        **kwargs: object,
    ) -> RuntimeExpr | None:
        # First, try to resolve as preserved method
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
            return method(self, *args, **kwargs)
        except KeyError:
            pass
        # If this is a "partial" method meaning that it can return NotImplemented,
        # we want to find the "best" superparent (lowest cost) of the arg types to call with it, instead of just
        # using the arg type of the self arg.
        # This is neccesary so if we add like an int to a ndarray, it will upcast the int to an ndarray, instead of vice versa.
        if __name in PARTIAL_METHODS:
            try:
                return call_method_min_conversion(self, args[0], __name)
            except ConvertError:
                return NotImplemented
        return RuntimeMethod(self, __name)(*args, **kwargs)

    setattr(RuntimeExpr, name, _special_method)

# For each of the reflected binary methods, translate to the corresponding non-reflected method
for reflected, non_reflected in REFLECTED_BINARY_METHODS.items():

    def _reflected_method(self: RuntimeExpr, other: object, __non_reflected: str = non_reflected) -> RuntimeExpr | None:
        # All binary methods are also "partial" meaning we should try to upcast first.
        return call_method_min_conversion(other, self, __non_reflected)

    setattr(RuntimeExpr, reflected, _reflected_method)


def call_method_min_conversion(slf: object, other: object, name: str) -> RuntimeExpr | None:
    # find a minimum type that both can be converted to
    # This is so so that calls like `-0.1 * Int("x")` work by upcasting both to floats.
    min_tp = min_convertable_tp(slf, other, name)
    slf = resolve_literal(min_tp.to_var(), slf)
    other = resolve_literal(min_tp.to_var(), other)
    method = RuntimeMethod(slf, name)
    return method(other)


for name in ["__bool__", "__len__", "__complex__", "__int__", "__float__", "__iter__", "__index__"]:

    def _preserved_method(self: RuntimeExpr, __name: str = name):
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
        except KeyError as e:
            raise TypeError(f"{self.__egg_typed_expr__.tp.name} has no method {__name}") from e
        return method(self)

    setattr(RuntimeExpr, name, _preserved_method)


def resolve_callable(callable: object) -> tuple[CallableRef, Declarations]:
    """
    Resolves a runtime callable into a ref
    """
    # TODO: Fix these typings.
    ref: CallableRef
    decls: Declarations
    if isinstance(callable, RuntimeFunction):
        ref = FunctionRef(callable.__egg_name__)
        decls = callable.__egg_decls__
    elif isinstance(callable, RuntimeClassMethod):
        ref = ClassMethodRef(callable.class_name, callable.__egg_method_name__)
        decls = callable.__egg_decls__
    elif isinstance(callable, RuntimeMethod):
        ref = MethodRef(callable.__egg_self__.__egg_typed_expr__.tp.name, callable.__egg_method_name__)
        decls = callable.__egg_decls__
    elif isinstance(callable, RuntimeClass):
        ref = ClassMethodRef(callable.__egg_name__, "__init__")
        decls = callable.__egg_decls__
    else:
        raise NotImplementedError(f"Cannot turn {callable} into a callable ref")
    return (ref, decls)
