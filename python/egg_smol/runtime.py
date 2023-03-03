"""
This module holds a number of types which are only used at runtime to emulate Python objects.

Users will not import anything from this module, and statically they won't know these are the types they are using.

But at runtime they will be exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Optional, Union

import black
from typing_extensions import assert_never

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
UNARY_LIT_CLASS_NAMES = {"i64", "string"}
LIT_CLASS_NAMES = UNARY_LIT_CLASS_NAMES | {UNIT_CLASS_NAME}


@dataclass
class RuntimeClass:
    decls: Declarations
    name: str

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.name in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], (int, str))
            return RuntimeExpr(self.decls, JustTypeRef(self.name), LitDecl(args[0]))
        if self.name == UNIT_CLASS_NAME:
            assert len(args) == 0
            return RuntimeExpr(self.decls, JustTypeRef(self.name), LitDecl(None))

        return RuntimeClassMethod(self.decls, self.name, "__init__")(*args)

    def __getitem__(
        self, args: tuple[RuntimeTypeArgType, ...]
    ) -> RuntimeParamaterizedClass:
        tp = JustTypeRef(self.name, tuple(class_to_ref(arg) for arg in args))
        return RuntimeParamaterizedClass(self.decls, tp)

    def __getattr__(self, name: str) -> RuntimeClassMethod:
        return RuntimeClassMethod(self.decls, self.name, name)

    def __str__(self) -> str:
        return self.name


@dataclass
class RuntimeParamaterizedClass:
    decls: Declarations
    # Note that this will never be a typevar because we don't use RuntimeParamaterizedClass for maps on their own methods
    # which is the only time we define function which take typevars
    tp: JustTypeRef

    def __post_init__(self):
        desired_args = self.decls.classes[self.tp.name].n_type_vars
        if len(self.tp.args) != desired_args:
            raise ValueError(
                f"Expected {desired_args} type args, got {len(self.tp.args)}"
            )

    def __call__(self, *args: ArgType) -> RuntimeExpr:

        return RuntimeClassMethod(self.decls, class_to_ref(self), "__init__")(*args)

    def __getattr__(self, name: str) -> RuntimeClassMethod:
        return RuntimeClassMethod(self.decls, class_to_ref(self), name)

    def __str__(self) -> str:
        return self.tp.pretty()


# Type args can either be typevars or classes
RuntimeTypeArgType = Union[RuntimeClass, RuntimeParamaterizedClass]


def class_to_ref(cls: RuntimeTypeArgType) -> JustTypeRef:
    if isinstance(cls, RuntimeClass):
        return JustTypeRef(cls.name)
    if isinstance(cls, RuntimeParamaterizedClass):
        return cls.tp
    assert_never(cls)


@dataclass
class RuntimeFunction:
    decls: Declarations
    name: str

    def __post_init__(self):
        if self.name not in self.decls.functions:
            raise ValueError(f"Function {self.name} does not exist")

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        return _call(
            self.decls, FunctionRef(self.name), self.decls.functions[self.name], args
        )

    def __str__(self) -> str:
        return self.name


def _call(
    decls: Declarations,
    callable_ref: CallableRef,
    fn_decl: FunctionDecl,
    args: Collection[ArgType],
    bound_params: Optional[tuple[JustTypeRef, ...]] = None,
) -> RuntimeExpr:
    upcasted_args = [_resolve_literal(decls, arg) for arg in args]

    arg_types = [arg.tp for arg in upcasted_args]

    if bound_params is not None:
        tcs = TypeConstraintSolver.from_type_parameters(bound_params)
    else:
        tcs = TypeConstraintSolver()

    return_tp = tcs.infer_return_type(fn_decl.arg_types, fn_decl.return_type, arg_types)

    arg_decls = tuple(arg.expr for arg in upcasted_args)
    expr_decl = CallDecl(callable_ref, arg_decls, bound_params)
    return RuntimeExpr(decls, return_tp, expr_decl)


@dataclass
class RuntimeClassMethod:
    decls: Declarations
    # Either a string if it isn't bound or a tp if it s
    tp: JustTypeRef | str
    method_name: str

    def __post_init__(self):
        if self.method_name not in self.decls.classes[self.class_name].methods:
            raise ValueError(
                f"Class {self.class_name} does not have method {self.method_name}"
            )

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        fn_decl = self.decls.classes[self.class_name].class_methods[self.method_name]
        bound_params = self.tp.args if isinstance(self.tp, JustTypeRef) else None
        return _call(
            self.decls,
            ClassMethodRef(self.class_name, self.method_name),
            fn_decl,
            args,
            bound_params,
        )

    def __str__(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    @property
    def class_name(self) -> str:
        if isinstance(self.tp, str):
            return self.tp
        return self.tp.name


@dataclass
class RuntimeMethod:
    decls: Declarations
    tp: JustTypeRef
    method_name: str
    slf_arg: ExprDecl

    def __post_init__(self):
        if self.method_name not in self.decls.classes[self.class_name].methods:
            raise ValueError(
                f"Class {self.class_name} does not have method {self.method_name}"
            )

    def __call__(self, *args: ArgType) -> RuntimeExpr:
        fn_decl = self.decls.classes[self.class_name].methods[self.method_name]

        first_arg = RuntimeExpr(self.decls, JustTypeRef(self.class_name), self.slf_arg)
        args = (first_arg, *args)

        return _call(
            self.decls, MethodRef(self.class_name, self.method_name), fn_decl, args
        )

    @property
    def class_name(self) -> str:
        return self.tp.name


@dataclass
class RuntimeExpr:
    decls: Declarations
    tp: JustTypeRef
    expr: ExprDecl

    @property
    def parts(self) -> tuple[JustTypeRef, ExprDecl]:
        return self.tp, self.expr

    #     def __getattr__(self, name: str) -> BoundMethod:
    #         return BoundMethod(self, name)

    def __str__(self) -> str:
        s = f"_: {self.tp.pretty()} = {self.expr.pretty()}"
        return black.format_str(s[:-1], mode=black.FileMode())

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    def __eq__(self, other: NoReturn) -> Expr:  # type: ignore
        raise NotImplementedError("Compare the __parts__ attribute instead")


# Define each of the special methods, since we have already declared them for pretty printing
for name in list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__", "__call__"]:

    def _special_method(
        self: RuntimeExpr, *args: ArgType, __name: str = name
    ) -> RuntimeExpr:
        return RuntimeMethod(self.decls, self.tp, __name, self.expr)(*args)

    setattr(RuntimeExpr, name, _special_method)


# Args can either be expressions or literals which are automatically promoted
ArgType = Union[RuntimeExpr, int, str]

tp_to_lit: dict[type, str] = {
    int: "i64",
    str: "string",
}


def _resolve_literal(decls: Declarations, arg: ArgType) -> RuntimeExpr:
    if isinstance(arg, int):
        return RuntimeExpr(decls, JustTypeRef("i64"), LitDecl(arg))
    elif isinstance(arg, str):
        return RuntimeExpr(decls, JustTypeRef("string"), LitDecl(arg))
    return arg


def test_type_str():
    decls = Declarations(
        classes={
            "i64": ClassDecl(),
            "Map": ClassDecl(n_type_vars=2),
        }
    )
    i64 = RuntimeClass(decls, "i64")
    Map = RuntimeClass(decls, "Map")
    assert str(i64) == "i64"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


def test_function_call():
    decls = Declarations(
        classes={
            "i64": ClassDecl(),
        },
        functions={
            "one": FunctionDecl(
                (),
                TypeRefWithVars("i64"),
            ),
        },
    )
    one = RuntimeFunction(decls, "one")
    assert (
        one().parts
        == RuntimeExpr(decls, JustTypeRef("i64"), CallDecl(FunctionRef("one"))).parts
    )


def test_classmethod_call():
    from pytest import raises

    K, V = ClassTypeVarRef(0), ClassTypeVarRef(1)
    decls = Declarations(
        classes={
            "i64": ClassDecl(),
            "unit": ClassDecl(),
            "Map": ClassDecl(n_type_vars=2),
        },
        functions={
            "create": FunctionDecl(
                (),
                TypeRefWithVars("Map", (K, V)),
            )
        },
    )
    Map = RuntimeClass(decls, "Map")
    with raises(TypeConstraintError):
        Map.create()
    i64 = RuntimeClass(decls, "i64")
    unit = RuntimeClass(decls, "unit")
    assert (
        Map[i64, unit].create().parts
        == RuntimeExpr(
            decls,
            JustTypeRef("Map", (JustTypeRef("i64"), JustTypeRef("unit"))),
            CallDecl(
                ClassMethodRef("Map", "create"),
                (),
                (JustTypeRef("i64"), JustTypeRef("unit")),
            ),
        ).parts
    )


def test_expr_special():
    decls = Declarations(
        classes={
            "i64": ClassDecl(
                methods={
                    "__add__": FunctionDecl(
                        (TypeRefWithVars("i64"), TypeRefWithVars("i64")),
                        TypeRefWithVars("i64"),
                    )
                },
                class_methods={
                    "__init__": FunctionDecl(
                        (TypeRefWithVars("i64"),),
                        TypeRefWithVars("i64"),
                    )
                },
            ),
        },
    )
    i64 = RuntimeClass(decls, "i64")
    one = i64(1)  # type: ignore
    res = one + one  # type: ignore
    expected_res = RuntimeExpr(
        decls,
        JustTypeRef("i64"),
        CallDecl(MethodRef("i64", "__add__"), (LitDecl(1), LitDecl(1))),
    )
    assert res.parts == expected_res.parts


# def blacken_python_expression(expr: str) -> str:
#     """
#     Runs black on a Python expression, to remove excess paranthesis and wrap it.
#     """
#     return black.format_str("x = " + expr, mode=BLACK_MODE)[4:-1]
