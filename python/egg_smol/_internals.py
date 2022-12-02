from __future__ import annotations

from collections.abc import Collection, Hashable
from dataclasses import dataclass, field
from typing import NewType, Union, cast

import black
import egg_smol.bindings as py


@dataclass(frozen=True)
class Kind:
    python_name: str
    egg_name: str
    typevariables: tuple[TypeVariable, ...] = ()
    classmethods: dict[str, Function] = field(default_factory=dict)
    methods: dict[str, Function] = field(default_factory=dict)
    # Init function. If defined, return value should be of this Kind
    init: Function | None = None

    def __call__(self, *args: Expr) -> Expr:
        if self.init is None:
            raise TypeError(f"{self} has no __init__ defined")
        return BoundClassMethod(self, (), self.init)(*args)

    def __getitem__(self, args: tuple[Type_, ...]) -> Type:
        if len(args) != len(self.typevariables):
            raise TypeError(
                f"Expected {len(self.typevariables)} type variables, got {len(args)}"
            )
        return Type(self, args)

    def __getattr__(self, name: str) -> BoundClassMethod:
        if name in self.classmethods:
            return BoundClassMethod(self, (), self.classmethods[name])
        raise AttributeError(f"{self.kind} has no classmethod {name}")

    def __str__(self) -> str:
        return self.python_name

    def __repr__(self) -> str:
        return str(self)


def test_kind_str():
    assert str(Kind("i64", "i64")) == "i64"


@dataclass(frozen=True)
class Type:
    kind: Kind
    args: tuple[Type_, ...]

    def __str__(self) -> str:
        if self.args:
            return f"{self.kind}[{', '.join(map(str, self.args))}]"
        else:
            return str(self.kind)

    def __repr__(self) -> str:
        return str(self)

    def __getattr__(self, name: str) -> BoundClassMethod:
        if name in self.kind.classmethods:
            return self._bind_classmethod(self.kind.classmethods[name])
        raise AttributeError(f"{self.kind} has no classmethod {name}")

    def __call__(self, *args: Expr) -> Expr:
        if self.kind.init is None:
            raise TypeError(f"{self} has no __init__ defined")
        return self._bind_classmethod(self.kind.init)(*args)

    def _bind_classmethod(self, fn: Function) -> BoundClassMethod:
        # Bind the classmethod, verifying that all types are not typevars
        tp_args: list[Type] = []
        for a in self.args:
            if not isinstance(a, Type):
                raise TypeError(f"Cannot get method from class with unbound type {a}")
            tp_args.append(a)
        return BoundClassMethod(self.kind, tuple(tp_args), fn)


@dataclass(frozen=True)
class TypeVariable:
    identifier: Hashable

    def __str__(self) -> str:
        return str(self.identifier)

    def __repr__(self) -> str:
        return str(self)


Type_ = TypeVariable | Type


def test_type_str():
    i64 = Kind("i64", "i64")[()]
    K, V = TypeVariable("K"), TypeVariable("V")
    Map = Kind("Map", "Map", (K, V))
    assert str(i64) == "i64"
    assert str(K) == "K"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


@dataclass(frozen=True)
class BoundClassMethod:
    kind: Kind
    # Any args provided, if it was bound
    tp_args: tuple[Type, ...] | None
    fn: Function

    def __call__(self, *args: Expr) -> Expr:
        # If this class was not bound, then no additional types were inferred
        if not self.tp_args:
            return self.fn(*args)
        # Otherwise, this class was bound, before before we accessed the method,
        # so we should replace the type variables with the inferred types
        ti = TypeInference(dict(zip(self.kind.typevariables, self.tp_args)))
        return self.fn(*args, _ti=ti)

    def __str__(self) -> str:
        tp_str = (
            f"{self.kind}[{', '.join(map(str, self.tp_args))}]"
            if self.tp_args
            else str(self.kind)
        )
        return f"{tp_str}.{self.fn}"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class BoundMethod:
    slf: Expr
    fn: Function

    def __call__(self, *args: Expr) -> Expr:
        return self.fn(self.slf, *args)

    def __str__(self) -> str:
        return f"{self.slf}.{self.fn}"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class TypeInference:
    _typevar_to_value: dict[TypeVariable, Type] = field(default_factory=dict)

    def infer_return_type(
        self,
        unbound_arg_types: list[Type_],
        unbound_return_type: Type_,
        arg_types: list[Type],
    ) -> Type:
        self._infer_typevars_zip(unbound_arg_types, arg_types)
        return self._subtitute_typevars(unbound_return_type)

    def _infer_typevars_zip(
        self, unbound: Collection[Type_], bound: Collection[Type_]
    ) -> None:
        if len(unbound) != len(bound):
            raise TypeError(f"Expected {len(unbound)} arguments, got {len(bound)}")
        for unbound_arg, bound_arg in zip(unbound, bound):
            self._infer_typevars(unbound_arg, bound_arg)

    def _infer_typevars(self, unbound: Type_, bound: Type_) -> None:
        if isinstance(bound, TypeVariable):
            raise ValueError("Bound type should not be a typevar")
        if isinstance(unbound, Type):
            if unbound.kind != bound.kind:
                raise TypeError(f"Expected {unbound.kind}, got {bound.kind}")
            self._infer_typevars_zip(unbound.args, bound.args)
            return
        if unbound in self._typevar_to_value:
            if self._typevar_to_value[unbound] != bound:
                raise TypeError(f"Typevar {unbound} already bound to {bound}")
        else:
            self._typevar_to_value[unbound] = bound

    def _subtitute_typevars(self, type_: Type_) -> Type:
        if isinstance(type_, TypeVariable):
            if type_ not in self._typevar_to_value:
                raise TypeError(f"Typevar {type_} not bound")
            return self._typevar_to_value[type_]
        return Type(
            type_.kind, tuple(self._subtitute_typevars(arg) for arg in type_.args)
        )


def test_type_inference():
    import pytest

    i64 = Kind("i64", "i64")[()]
    unit = Kind("Unit", "Unit")[()]
    K, V = TypeVariable("K"), TypeVariable("V")
    Map = Kind("Map", "Map", (K, V))
    ti = TypeInference()
    assert ti.infer_return_type([i64], i64, [i64]) == i64
    with pytest.raises(TypeError):
        ti.infer_return_type([i64], i64, [unit])
    with pytest.raises(TypeError):
        ti.infer_return_type([], i64, [unit])

    assert ti.infer_return_type([Map[K, V], K], V, [Map[i64, unit], i64]) == unit

    with pytest.raises(TypeError):
        ti.infer_return_type([Map[K, V], K], V, [Map[i64, unit], unit])


@dataclass(frozen=True)
class Function:
    python_name: str
    egg_name: str
    arg_types: list[Type_]
    return_type: Type_
    cost: int = 0
    merge: Expr | None = None

    def __call__(self, *args: Expr, _ti: TypeInference | None = None) -> Expr:
        ti = _ti or TypeInference()
        bound_return_type = ti.infer_return_type(
            self.arg_types, self.return_type, [arg.type for arg in args]
        )
        return Expr(bound_return_type, py.Call(self.egg_name, [a.value for a in args]))

    def __str__(self) -> str:
        return self.python_name

    def __repr__(self) -> str:
        return str(self)


def test_function_call():
    i64 = Kind("i64", "i64")[()]
    one = Function("one", "one", [], i64)
    assert one()._parts == (i64, py.Call("one", []))


def test_classmethod_call():
    from pytest import raises

    K, V = TypeVariable("K"), TypeVariable("V")
    Map = Kind("Map", "Map", (K, V))
    create_fn = Function("create", "create", [], Map[K, V])
    Map.classmethods["create"] = create_fn
    with raises(TypeError):
        Map.create()

    i64 = Kind("i64", "i64")[()]
    unit = Kind("Unit", "Unit")[()]
    assert Map[i64, unit].create()._parts == (Map[i64, unit], py.Call("create", []))


# Ex:
# K, V = TypeVar(0), TypeVar(1)
# get = Function("get", "get", [Map[K, V], K], V)


# Nothing should ever be this value, as a way of disallowing methods
_Nothing = NewType("_Nothing", object)


@dataclass(frozen=True)
class Expr:
    type: Type
    value: py._Expr
    inventory: Inventory = field(default_factory=lambda: Inventory())

    def __getattr__(self, name: str) -> BoundMethod:
        return self._get_method(name)

    def __str__(self) -> str:
        return self.inventory.prett_print_expr(self.value)

    def __repr__(self) -> str:
        return str(self)

    def _get_method(self, name: str) -> BoundMethod:
        methods = self.type.kind.methods
        if name not in methods:
            raise AttributeError(f"{self.type.kind} has no method {name}")
        return BoundMethod(self, methods[name])

    # Use _Nothing so that == is not allowed on Expr objects according to MyPy
    # (so that in tests we don't try to use this method by accident)
    # so that __eq__ is reserved for any custom equality checks
    def __eq__(self, other: _Nothing) -> Expr:  # type: ignore
        return self._get_method("__eq__")(cast(Expr, other))

    def _eq(self, other: Expr) -> bool:
        return self._parts == other._parts

    @property
    def _parts(self) -> tuple[Type, py._Expr]:
        return (self.type, self.value)


# Special methods which we might want to use as functions
# Mapping to the operator they represent for pretty printing them
# https://docs.python.org/3/reference/datamodel.html
BINARY_METHODS = {
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
    # Numeric
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__matmul__": "@",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    "__divmod__": "divmod",
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
}
UNARY_METHODS = {
    "__pos__": "+",
    "__neg__": "-",
    "__invert__": "~",
}

SPECIAL_METHODS = list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__"]
for name in SPECIAL_METHODS:
    setattr(
        Expr,
        name,
        lambda self, *args, name=name: self._get_method(name)(*args),
    )


def test_expr_special():
    i64Kind = Kind("i64", "i64")
    i64 = i64Kind[()]
    add = Function("__add__", "add", [i64, i64], i64)
    i64Kind.methods["__add__"] = add
    one_egg = py.Lit(py.Int(1))
    one = Expr(i64, one_egg)
    res = one + one  # type: ignore
    assert res._parts == (i64, py.Call("add", [one_egg, one_egg]))


@dataclass
class Inventory:
    """
    Collection of all the types and functions in the program.
    """

    # Mapping of egg literal to name which constructs it.
    lit_constructors: dict[type[py._Literal], str] = field(default_factory=dict)
    # Mapping of egg name to function pointer
    fn_ptrs: dict[str, FunctionPointer] = field(default_factory=dict)

    def prett_print_expr(self, expr: py._Expr) -> str:
        return blacken_python_expression(self._print_expr(expr))

    def _print_expr(self, expr: py._Expr) -> str:
        """
        Prints an expression, converting all of the function calls to their Python counterpoints
        """
        if isinstance(expr, py.Var):
            return expr.name
        elif isinstance(expr, py.Lit):
            val = expr.value
            constructor_name = self.lit_constructors[type(val)]
            args = [] if isinstance(val, py.Unit) else [str(val.value)]
            return f"{constructor_name}({', '.join(args)})"
        elif isinstance(expr, py.Call):
            return self._print_function_call(self.fn_ptrs[expr.name], expr.args)

    def _print_function_call(self, ptr: FunctionPointer, args: list[py._Expr]) -> str:
        if isinstance(ptr, TopLevelFunctionPointer):
            fn_str = ptr.name
        elif ptr.is_classmethod:
            fn_str = f"{ptr.kind_name}.{ptr.name}"
        else:
            name = ptr.name
            slf, *args = args
            slf_str = self._print_expr(slf)
            if name in UNARY_METHODS:
                return f"{UNARY_METHODS[name]}{slf_str}"
            elif name in BINARY_METHODS:
                rhs_str = self._print_expr(args[0])
                return f"({slf_str} {BINARY_METHODS[name]} {rhs_str})"
            elif name == "__getitem__":
                rhs_str = self._print_expr(args[0])
                return f"{slf_str}[{rhs_str}]"
            fn_str = f"{slf_str}.{ptr.name}"
        return f"{fn_str}({', '.join(self._print_expr(a) for a in args)})"


def test_print_literal():
    inventory = Inventory()
    inventory.lit_constructors[py.Int] = "i64"
    assert inventory.prett_print_expr(py.Lit(py.Int(1))) == "i64(1)"


def test_print_function_call():
    inventory = Inventory()
    inventory.lit_constructors[py.Int] = "i64"
    inventory.fn_ptrs["add"] = TopLevelFunctionPointer("add")
    assert (
        inventory.prett_print_expr(
            py.Call("add", [py.Lit(py.Int(1)), py.Lit(py.Int(2))])
        )
        == "add(i64(1), i64(2))"
    )


def test_print_classmethod_call():
    inventory = Inventory()
    inventory.lit_constructors[py.Int] = "i64"
    inventory.fn_ptrs["add"] = MethodPointer("i64", "add", True)
    assert (
        inventory.prett_print_expr(
            py.Call("add", [py.Lit(py.Int(1)), py.Lit(py.Int(2))])
        )
        == "i64.add(i64(1), i64(2))"
    )


def test_print_method_call():
    inventory = Inventory()
    inventory.lit_constructors[py.Int] = "i64"
    inventory.fn_ptrs["add"] = MethodPointer("i64", "add", False)
    assert (
        inventory.prett_print_expr(
            py.Call("add", [py.Lit(py.Int(1)), py.Lit(py.Int(2))])
        )
        == "i64(1).add(i64(2))"
    )


def test_print_special_method_call():
    inventory = Inventory()
    inventory.lit_constructors[py.Int] = "i64"
    inventory.fn_ptrs["__add__"] = MethodPointer("i64", "__add__", False)
    assert (
        inventory.prett_print_expr(
            py.Call("__add__", [py.Lit(py.Int(1)), py.Lit(py.Int(2))])
        )
        == "i64(1) + i64(2)"
    )


BLACK_MODE = black.Mode(line_length=120)


def blacken_python_expression(expr: str) -> str:
    """
    Runs black on a Python expression, to remove excess paranthesis and wrap it.
    """
    return black.format_str("x = " + expr, mode=BLACK_MODE)[4:-1]


@dataclass
class TopLevelFunctionPointer:
    name: str


@dataclass
class MethodPointer:
    kind_name: str
    name: str
    is_classmethod: bool


FunctionPointer = TopLevelFunctionPointer | MethodPointer
