"""
Holds a number of types which are only used at runtime to emulate Python objects.

Users will not import anything from this module, and statically they won't know these are the types they are using.

But at runtime they will be exposed.

Note that all their internal fields are prefixed with __egg_ to avoid name collisions with user code, but will end in __
so they are not mangled by Python and can be accessed by the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, Signature
from itertools import zip_longest
from typing import TYPE_CHECKING, NoReturn, TypeVar, Union, cast, get_args, get_origin

from .declarations import *
from .pretty import *
from .thunk import Thunk
from .type_constraint_solver import *

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .egraph import Expr

__all__ = [
    "LIT_CLASS_NAMES",
    "resolve_callable",
    "resolve_type_annotation",
    "RuntimeClass",
    "RuntimeExpr",
    "RuntimeFunction",
    "REFLECTED_BINARY_METHODS",
]


UNIT_CLASS_NAME = "Unit"
UNARY_LIT_CLASS_NAMES = {"i64", "f64", "Bool", "String"}
LIT_CLASS_NAMES = UNARY_LIT_CLASS_NAMES | {UNIT_CLASS_NAME, "PyObject"}

REFLECTED_BINARY_METHODS = {
    "__radd__": "__add__",
    "__rsub__": "__sub__",
    "__rmul__": "__mul__",
    "__rmatmul__": "__matmul__",
    "__rtruediv__": "__truediv__",
    "__rfloordiv__": "__floordiv__",
    "__rmod__": "__mod__",
    "__rpow__": "__pow__",
    "__rlshift__": "__lshift__",
    "__rrshift__": "__rshift__",
    "__rand__": "__and__",
    "__rxor__": "__xor__",
    "__ror__": "__or__",
}

# Set this globally so we can get access to PyObject when we have a type annotation of just object.
# This is the only time a type annotation doesn't need to include the egglog type b/c object is top so that would be redundant statically.
_PY_OBJECT_CLASS: RuntimeClass | None = None

T = TypeVar("T")


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
        return tp.__egg_tp__
    raise TypeError(f"Unexpected type annotation {tp}")


##
# Runtime objects
##


@dataclass
class RuntimeClass(DelayedDeclerations):
    __egg_tp__: TypeRefWithVars

    def __post_init__(self) -> None:
        global _PY_OBJECT_CLASS
        if self.__egg_tp__.name == "PyObject":
            _PY_OBJECT_CLASS = self

    def verify(self) -> None:
        if not self.__egg_tp__.args:
            return

        # Raise error if we have args, but they are the wrong number
        desired_args = self.__egg_decls__.get_class_decl(self.__egg_tp__.name).type_vars
        if len(self.__egg_tp__.args) != len(desired_args):
            raise ValueError(f"Expected {desired_args} type args, got {len(self.__egg_tp__.args)}")

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.__egg_tp__.name == "PyObject":
            assert len(args) == 1
            return RuntimeExpr.__from_value__(
                self.__egg_decls__, TypedExprDecl(self.__egg_tp__.to_just(), PyObjectDecl(args[0]))
            )
        if self.__egg_tp__.name in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], int | float | str | bool)
            return RuntimeExpr.__from_value__(
                self.__egg_decls__, TypedExprDecl(self.__egg_tp__.to_just(), LitDecl(args[0]))
            )
        if self.__egg_tp__.name == UNIT_CLASS_NAME:
            assert len(args) == 0
            return RuntimeExpr.__from_value__(
                self.__egg_decls__, TypedExprDecl(self.__egg_tp__.to_just(), LitDecl(None))
            )

        return RuntimeFunction(
            Thunk.value(self.__egg_decls__), ClassMethodRef(self.__egg_tp__.name, "__init__"), self.__egg_tp__.to_just()
        )(*args, **kwargs)

    def __dir__(self) -> list[str]:
        cls_decl = self.__egg_decls__.get_class_decl(self.__egg_tp__.name)
        possible_methods = (
            list(cls_decl.class_methods) + list(cls_decl.class_variables) + list(cls_decl.preserved_methods)
        )
        if "__init__" in possible_methods:
            possible_methods.remove("__init__")
            possible_methods.append("__call__")
        return possible_methods

    def __getitem__(self, args: object) -> RuntimeClass:
        if self.__egg_tp__.args:
            raise TypeError(f"Cannot index into a paramaterized class {self}")
        if not isinstance(args, tuple):
            args = (args,)
        decls = self.__egg_decls__.copy()
        tp = TypeRefWithVars(self.__egg_tp__.name, tuple(resolve_type_annotation(decls, arg) for arg in args))
        return RuntimeClass(Thunk.value(decls), tp)

    def __getattr__(self, name: str) -> RuntimeFunction | RuntimeExpr | Callable:
        if name == "__origin__" and self.__egg_tp__.args:
            return RuntimeClass(self.__egg_decls_thunk__, TypeRefWithVars(self.__egg_tp__.name))

        # Special case some names that don't exist so we can exit early without resolving decls
        # Important so if we take union of RuntimeClass it won't try to resolve decls
        if name in {
            "__typing_subst__",
            "__parameters__",
            # Origin is used in get_type_hints which is used when resolving the class itself
            "__origin__",
        }:
            raise AttributeError

        cls_decl = self.__egg_decls__._classes[self.__egg_tp__.name]

        preserved_methods = cls_decl.preserved_methods
        if name in preserved_methods:
            return preserved_methods[name].__get__(self)

        # if this is a class variable, return an expr for it, otherwise, assume it's a method
        if name in cls_decl.class_variables:
            return_tp = cls_decl.class_variables[name]
            return RuntimeExpr.__from_value__(
                self.__egg_decls__,
                TypedExprDecl(return_tp.type_ref, CallDecl(ClassVariableRef(self.__egg_tp__.name, name))),
            )
        if name in cls_decl.class_methods:
            return RuntimeFunction(
                Thunk.value(self.__egg_decls__), ClassMethodRef(self.__egg_tp__.name, name), self.__egg_tp__.to_just()
            )
        msg = f"Class {self.__egg_tp__.name} has no method {name}"
        if name == "__ne__":
            msg += ". Did you mean to use the ne(...).to(...)?"
        raise AttributeError(msg) from None

    def __str__(self) -> str:
        return str(self.__egg_tp__)

    # Make hashable so can go in Union
    def __hash__(self) -> int:
        return hash((id(self.__egg_decls_thunk__), self.__egg_tp__))

    # Support unioning like types
    def __or__(self, __value: type) -> object:
        return Union[self, __value]  # noqa: UP007


@dataclass
class RuntimeFunction(DelayedDeclerations):
    __egg_ref__: CallableRef
    # bound methods need to store RuntimeExpr not just TypedExprDecl, so they can mutate the expr if required on self
    __egg_bound__: JustTypeRef | RuntimeExpr | None = None

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        from .conversion import resolve_literal

        if isinstance(self.__egg_bound__, RuntimeExpr):
            args = (self.__egg_bound__, *args)
        fn_decl = self.__egg_decls__.get_callable_decl(self.__egg_ref__).to_function_decl()
        # Turn all keyword args into positional args
        bound = callable_decl_to_signature(fn_decl, self.__egg_decls__).bind(*args, **kwargs)
        bound.apply_defaults()
        assert not bound.kwargs
        del args, kwargs

        upcasted_args = [
            resolve_literal(cast(TypeOrVarRef, tp), arg)
            for arg, tp in zip_longest(bound.args, fn_decl.arg_types, fillvalue=fn_decl.var_arg_type)
        ]

        decls = Declarations.create(self, *upcasted_args)

        tcs = TypeConstraintSolver(decls)
        bound_tp = (
            None
            if self.__egg_bound__ is None
            else self.__egg_bound__.__egg_typed_expr__.tp
            if isinstance(self.__egg_bound__, RuntimeExpr)
            else self.__egg_bound__
        )
        if bound_tp and bound_tp.args:
            tcs.bind_class(bound_tp)
        arg_exprs = tuple(arg.__egg_typed_expr__ for arg in upcasted_args)
        arg_types = [expr.tp for expr in arg_exprs]
        cls_name = bound_tp.name if bound_tp else None
        return_tp = tcs.infer_return_type(
            fn_decl.arg_types, fn_decl.return_type or fn_decl.arg_types[0], fn_decl.var_arg_type, arg_types, cls_name
        )
        bound_params = cast(JustTypeRef, bound_tp).args if isinstance(self.__egg_ref__, ClassMethodRef) else None
        expr_decl = CallDecl(self.__egg_ref__, arg_exprs, bound_params)
        typed_expr_decl = TypedExprDecl(return_tp, expr_decl)
        # If there is not return type, we are mutating the first arg
        if not fn_decl.return_type:
            first_arg = upcasted_args[0]
            first_arg.__egg_thunk__ = Thunk.value((decls, typed_expr_decl))
            return None
        return RuntimeExpr.__from_value__(decls, typed_expr_decl)

    def __str__(self) -> str:
        first_arg, bound_tp_params = None, None
        match self.__egg_bound__:
            case RuntimeExpr(_):
                first_arg = self.__egg_bound__.__egg_typed_expr__.expr
            case JustTypeRef(_, args):
                bound_tp_params = args
        return pretty_callable_ref(self.__egg_decls__, self.__egg_ref__, first_arg, bound_tp_params)


def callable_decl_to_signature(
    decl: FunctionDecl,
    decls: Declarations,
) -> Signature:
    parameters = [
        Parameter(
            n,
            Parameter.POSITIONAL_OR_KEYWORD,
            default=RuntimeExpr.__from_value__(decls, TypedExprDecl(t.to_just(), d)) if d else Parameter.empty,
        )
        for n, d, t in zip(decl.arg_names, decl.arg_defaults, decl.arg_types, strict=True)
    ]
    if isinstance(decl, FunctionDecl) and decl.var_arg_type is not None:
        parameters.append(Parameter("__rest", Parameter.VAR_POSITIONAL))
    return Signature(parameters)


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
class RuntimeExpr:
    # Defer needing decls/expr so we can make constants that don't resolve their class types
    __egg_thunk__: Callable[[], tuple[Declarations, TypedExprDecl]]

    @classmethod
    def __from_value__(cls, d: Declarations, e: TypedExprDecl) -> RuntimeExpr:
        return cls(Thunk.value((d, e)))

    @property
    def __egg_decls__(self) -> Declarations:
        return self.__egg_thunk__()[0]

    @property
    def __egg_typed_expr__(self) -> TypedExprDecl:
        return self.__egg_thunk__()[1]

    def __getattr__(self, name: str) -> RuntimeFunction | RuntimeExpr | Callable | None:
        cls_name = self.__egg_class_name__
        class_decl = self.__egg_class_decl__

        if name in (preserved_methods := class_decl.preserved_methods):
            return preserved_methods[name].__get__(self)

        if name in class_decl.methods:
            return RuntimeFunction(Thunk.value(self.__egg_decls__), MethodRef(cls_name, name), self)
        if name in class_decl.properties:
            return RuntimeFunction(Thunk.value(self.__egg_decls__), PropertyRef(cls_name, name), self)()
        raise AttributeError(f"{cls_name} has no method {name}") from None

    def __repr__(self) -> str:
        """
        The repr of the expr is the pretty printed version of the expr.
        """
        return str(self)

    def __str__(self) -> str:
        return self.__egg_pretty__(None)

    def __egg_pretty__(self, wrapping_fn: str | None) -> str:
        return pretty_decl(self.__egg_decls__, self.__egg_typed_expr__.expr, wrapping_fn=wrapping_fn)

    def _ipython_display_(self) -> None:
        from IPython.display import Code, display

        display(Code(str(self), language="python"))

    def __dir__(self) -> Iterable[str]:
        class_decl = self.__egg_class_decl__
        return list(class_decl.methods) + list(class_decl.properties) + list(class_decl.preserved_methods)

    @property
    def __egg_class_name__(self) -> str:
        return self.__egg_typed_expr__.tp.name

    @property
    def __egg_class_decl__(self) -> ClassDecl:
        return self.__egg_decls__.get_class_decl(self.__egg_class_name__)

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    # At runtime, this will resolve if there is a defined egg function for `__eq__`
    def __eq__(self, other: NoReturn) -> Expr: ...  # type: ignore[override, empty-body]

    # Implement these so that copy() works on this object
    # otherwise copy will try to call `__getstate__` before object is initialized with properties which will cause inifinite recursion

    def __getstate__(self) -> tuple[Declarations, TypedExprDecl]:
        return self.__egg_thunk__()

    def __setstate__(self, d: tuple[Declarations, TypedExprDecl]) -> None:
        self.__egg_thunk__ = Thunk.value(d)

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
        from .conversion import ConvertError

        class_name = self.__egg_class_name__
        class_decl = self.__egg_class_decl__
        # First, try to resolve as preserved method
        try:
            method = class_decl.preserved_methods[__name]
        except KeyError:
            pass
        else:
            return method(self, *args, **kwargs)
        # If this is a "partial" method meaning that it can return NotImplemented,
        # we want to find the "best" superparent (lowest cost) of the arg types to call with it, instead of just
        # using the arg type of the self arg.
        # This is neccesary so if we add like an int to a ndarray, it will upcast the int to an ndarray, instead of vice versa.
        if __name in PARTIAL_METHODS:
            try:
                return call_method_min_conversion(self, args[0], __name)
            except ConvertError:
                return NotImplemented
        if __name in class_decl.methods:
            fn = RuntimeFunction(Thunk.value(self.__egg_decls__), MethodRef(class_name, __name), self)
            return fn(*args, **kwargs)
        raise TypeError(f"{class_name!r} object does not support {__name}")

    setattr(RuntimeExpr, name, _special_method)

# For each of the reflected binary methods, translate to the corresponding non-reflected method
for reflected, non_reflected in REFLECTED_BINARY_METHODS.items():

    def _reflected_method(self: RuntimeExpr, other: object, __non_reflected: str = non_reflected) -> RuntimeExpr | None:
        # All binary methods are also "partial" meaning we should try to upcast first.
        return call_method_min_conversion(other, self, __non_reflected)

    setattr(RuntimeExpr, reflected, _reflected_method)


def call_method_min_conversion(slf: object, other: object, name: str) -> RuntimeExpr | None:
    from .conversion import min_convertable_tp, resolve_literal

    # find a minimum type that both can be converted to
    # This is so so that calls like `-0.1 * Int("x")` work by upcasting both to floats.
    min_tp = min_convertable_tp(slf, other, name)
    slf = resolve_literal(min_tp.to_var(), slf)
    other = resolve_literal(min_tp.to_var(), other)
    method = RuntimeFunction(Thunk.value(slf.__egg_decls__), MethodRef(slf.__egg_class_name__, name), slf)
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
    match callable:
        case RuntimeFunction(decls, ref, _):
            return ref, decls()
        case RuntimeClass(thunk, tp):
            return ClassMethodRef(tp.name, "__init__"), thunk()
    raise NotImplementedError(f"Cannot turn {callable} into a callable ref")
