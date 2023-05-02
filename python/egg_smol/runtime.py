"""
This module holds a number of types which are only used at runtime to emulate Python objects.

Users will not import anything from this module, and statically they won't know these are the types they are using.

But at runtime they will be exposed.

Note that all their internal fields are prefixed with __egg_ to avoid name collisions with user code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Optional, Union

import black
from typing_extensions import assert_never

from . import config as configuration  # noqa: F401
from .declarations import *
from .declarations import BINARY_METHODS, UNARY_METHODS
from .type_constraint_solver import *

__all__ = [
    "LIT_CLASS_NAMES",
    "RuntimeClass",
    "RuntimeParamaterizedClass",
    "RuntimeClassMethod",
    "RuntimeExpr",
    "RuntimeFunction",
    "ArgType",
]


BLACK_MODE = black.Mode(line_length=120)  # type: ignore

UNIT_CLASS_NAME = "unit"
UNARY_LIT_CLASS_NAMES = {"i64", "f64", "String"}
LIT_CLASS_NAMES = UNARY_LIT_CLASS_NAMES | {UNIT_CLASS_NAME}


@dataclass
class RuntimeClass:
    __egg_decls__: Declarations
    __egg_name__: str

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.__egg_name__ in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], (int, float, str))
            return RuntimeExpr(self.__egg_decls__, JustTypeRef(self.__egg_name__), LitDecl(args[0]))
        if self.__egg_name__ == UNIT_CLASS_NAME:
            assert len(args) == 0
            return RuntimeExpr(self.__egg_decls__, JustTypeRef(self.__egg_name__), LitDecl(None))

        return RuntimeClassMethod(self.__egg_decls__, self.__egg_name__, "__init__")(*args)

    def __dir__(self) -> list[str]:
        cls_decl = self.__egg_decls__._classes[self.__egg_name__]
        possible_methods = list(cls_decl.class_methods) + list(cls_decl.class_variables)
        if "__init__" in possible_methods:
            possible_methods.remove("__init__")
            possible_methods.append("__call__")
        return possible_methods

    def __getitem__(self, args: tuple[RuntimeTypeArgType, ...]) -> RuntimeParamaterizedClass:
        tp = JustTypeRef(self.__egg_name__, tuple(class_to_ref(arg) for arg in args))
        return RuntimeParamaterizedClass(self.__egg_decls__, tp)

    def __getattr__(self, name: str) -> RuntimeClassMethod | RuntimeExpr:
        cls_decl = self.__egg_decls__._classes[self.__egg_name__]
        # if this is a class variable, return an expr for it, otherwise, assume it's a method
        if name in cls_decl.class_variables:
            return_tp = cls_decl.class_variables[name].return_type.to_just()
            return RuntimeExpr(self.__egg_decls__, return_tp, CallDecl(ClassVariableRef(self.__egg_name__, name)))
        return RuntimeClassMethod(self.__egg_decls__, self.__egg_name__, name)

    def __str__(self) -> str:
        return self.__egg_name__

    # Make hashable so can go in Union
    def __hash__(self) -> int:
        return hash((id(self.__egg_decls__), self.__egg_name__))


@dataclass
class RuntimeParamaterizedClass:
    __egg_decls__: Declarations
    # Note that this will never be a typevar because we don't use RuntimeParamaterizedClass for maps on their own methods
    # which is the only time we define function which take typevars
    __egg_tp__: JustTypeRef

    def __post_init__(self):
        desired_args = self.__egg_decls__._classes[self.__egg_tp__.name].n_type_vars
        if len(self.__egg_tp__.args) != desired_args:
            raise ValueError(f"Expected {desired_args} type args, got {len(self.__egg_tp__.args)}")

    def __call__(self, *args: ArgType) -> RuntimeExpr:
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
    __egg_decls__: Declarations
    __egg_name__: str

    def __post_init__(self):
        if self.__egg_name__ not in self.__egg_decls__._functions:
            raise ValueError(f"Function {self.__egg_name__} does not exist")

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        return _call(
            self.__egg_decls__,
            FunctionRef(self.__egg_name__),
            self.__egg_decls__._functions[self.__egg_name__],
            args,
        )

    def __str__(self) -> str:
        return self.__egg_name__


def _call(
    decls: Declarations,
    callable_ref: CallableRef,
    # Not included if this is the != method
    fn_decl: Optional[FunctionDecl],
    args: Collection[ArgType],
    bound_params: Optional[tuple[JustTypeRef, ...]] = None,
) -> RuntimeExpr:
    upcasted_args = [_resolve_literal(decls, arg) for arg in args]

    arg_types = [arg.__egg_tp__ for arg in upcasted_args]

    if bound_params is not None:
        tcs = TypeConstraintSolver.from_type_parameters(bound_params)
    else:
        tcs = TypeConstraintSolver()

    if fn_decl is not None:
        return_tp = tcs.infer_return_type(fn_decl.arg_types, fn_decl.return_type, arg_types)
    else:
        return_tp = JustTypeRef("unit")

    arg_decls = tuple(arg.__egg_expr__ for arg in upcasted_args)
    expr_decl = CallDecl(callable_ref, arg_decls, bound_params)
    return RuntimeExpr(decls, return_tp, expr_decl)


@dataclass
class RuntimeClassMethod:
    __egg_decls__: Declarations
    # Either a string if it isn't bound or a tp if it s
    __egg_tp__: JustTypeRef | str
    __egg_method_name__: str

    def __post_init__(self):
        if self.__egg_method_name__ not in self.__egg_decls__._classes[self.class_name].class_methods:
            raise AttributeError(f"Class {self.class_name} does not have method {self.__egg_method_name__}")

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        fn_decl = self.__egg_decls__._classes[self.class_name].class_methods[self.__egg_method_name__]
        bound_params = self.__egg_tp__.args if isinstance(self.__egg_tp__, JustTypeRef) else None
        return _call(
            self.__egg_decls__,
            ClassMethodRef(self.class_name, self.__egg_method_name__),
            fn_decl,
            args,
            bound_params,
        )

    def __str__(self) -> str:
        return f"{self.class_name}.{self.__egg_method_name__}"

    @property
    def class_name(self) -> str:
        if isinstance(self.__egg_tp__, str):
            return self.__egg_tp__
        return self.__egg_tp__.name


@dataclass
class RuntimeMethod:
    __egg_decls__: Declarations
    __egg_tp__: JustTypeRef
    __egg_method_name__: str
    __egg_slf_arg__: ExprDecl

    def __post_init__(self):
        if (
            self.__egg_method_name__ not in self.__egg_decls__._classes[self.class_name].methods
            # Special case for __ne__ which does not have a normal function defintion since
            # it relies of type parameters
            and self.__egg_method_name__ != "__ne__"
        ):
            raise AttributeError(f"Class {self.class_name} does not have method {self.__egg_method_name__}")

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        fn_decl = (
            self.__egg_decls__._classes[self.class_name].methods[self.__egg_method_name__]
            if self.__egg_method_name__ != "__ne__"
            else None
        )

        first_arg = RuntimeExpr(self.__egg_decls__, self.__egg_tp__, self.__egg_slf_arg__)
        args = (first_arg, *args)

        return _call(
            self.__egg_decls__,
            MethodRef(self.class_name, self.__egg_method_name__),
            fn_decl,
            args,
        )

    @property
    def class_name(self) -> str:
        return self.__egg_tp__.name


@dataclass
class RuntimeExpr:
    __egg_decls__: Declarations
    __egg_tp__: JustTypeRef
    __egg_expr__: ExprDecl

    @property
    def __egg_parts__(self) -> tuple[JustTypeRef, ExprDecl]:
        return self.__egg_tp__, self.__egg_expr__

    def __getattr__(self, name: str) -> RuntimeMethod:
        return RuntimeMethod(self.__egg_decls__, self.__egg_tp__, name, self.__egg_expr__)

    def __repr__(self) -> str:
        """
        The repr of the expr is the pretty printed version of the expr.
        """
        return str(self)

    def __str__(self) -> str:
        pretty_expr = self.__egg_expr__.pretty(parens=False)
        if configuration.SHOW_TYPES:
            s = f"_: {self.__egg_tp__.pretty()} = {pretty_expr}"
            return black.format_str(s, mode=black.FileMode()).strip()
        else:
            return black.format_str(pretty_expr, mode=black.FileMode(line_length=180)).strip()

    def __dir__(self) -> Iterable[str]:
        return list(self.__egg_decls__._classes[self.__egg_tp__.name].methods)

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    def __eq__(self, other: NoReturn) -> Expr:  # type: ignore
        raise NotImplementedError("Compare the __parts__ attribute instead")


# Define each of the special methods, since we have already declared them for pretty printing
for name in list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__", "__call__"]:

    def _special_method(self: RuntimeExpr, *args: ArgType, __name: str = name) -> RuntimeExpr:
        return RuntimeMethod(self.__egg_decls__, self.__egg_tp__, __name, self.__egg_expr__)(*args)

    setattr(RuntimeExpr, name, _special_method)


# Args can either be expressions or literals which are automatically promoted
ArgType = Union[RuntimeExpr, int, str, float]


def _resolve_literal(decls: Declarations, arg: ArgType) -> RuntimeExpr:
    if isinstance(arg, int):
        return RuntimeExpr(decls, JustTypeRef("i64"), LitDecl(arg))
    elif isinstance(arg, float):
        return RuntimeExpr(decls, JustTypeRef("f64"), LitDecl(arg))
    elif isinstance(arg, str):
        return RuntimeExpr(decls, JustTypeRef("String"), LitDecl(arg))
    return arg


def _resolve_callable(callable: object) -> CallableRef:
    """
    Resolves a runtime callable into a ref
    """
    if isinstance(callable, RuntimeFunction):
        return FunctionRef(callable.__egg_name__)
    if isinstance(callable, RuntimeClassMethod):
        return ClassMethodRef(callable.class_name, callable.__egg_method_name__)
    if isinstance(callable, RuntimeMethod):
        return MethodRef(callable.__egg_tp__.name, callable.__egg_method_name__)
    if isinstance(callable, RuntimeClass):
        return ClassMethodRef(callable.__egg_name__, "__init__")
    raise NotImplementedError(f"Cannot turn {callable} into a callable ref")
