from __future__ import annotations

from collections.abc import Collection, Hashable
from dataclasses import dataclass, field
from typing import NewType, cast

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
        return Expr(bound_return_type, Call(self, args))

    def __str__(self) -> str:
        return self.python_name

    def __repr__(self) -> str:
        return str(self)


def test_function_call():
    i64 = Kind("i64", "i64")[()]
    one = Function("one", "one", [], i64)
    assert one()._parts == (i64, Call(one))


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
    assert Map[i64, unit].create()._parts == (Map[i64, unit], Call(create_fn))


# Ex:
# K, V = TypeVar(0), TypeVar(1)
# get = Function("get", "get", [Map[K, V], K], V)


# Nothing should ever be this value, as a way of disallowing methods
_Nothing = NewType("_Nothing", object)


@dataclass(frozen=True)
class Expr:
    type: Type
    # TODO: Switch to regular _Expr, because we can't convert recursively without type inference.
    value: ExprValue

    def __getattr__(self, name: str) -> BoundMethod:
        return self._get_method(name)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self)

    def _get_method(self, name: str) -> BoundMethod:
        methods = self.type.kind.methods
        if name not in methods:
            raise AttributeError(f"{self.type.kind} has no method {name}")
        return BoundMethod(self, methods[name])

    # Use _Nothing so that == is not allowed on Expr objects according to MyPy
    # so that __eq__ is reserved for any custom equality checks
    def __eq__(self, other: _Nothing) -> Expr:  # type: ignore
        return self._get_method("__eq__")(cast(Expr, other))

    def _eq(self, other: Expr) -> bool:
        return self._parts == other._parts

    @property
    def _parts(self) -> tuple[Type, ExprValue]:
        return (self.type, self.value)

    def _to_egg(self) -> py._Expr:
        return self.value.to_egg(self.type)


# Special methods which we might want to use as functions
# These should all be functional
# https://docs.python.org/3/reference/datamodel.html
SPECIAL_METHODS = [
    # Comparison
    "lt",
    "le",
    "eq",
    "ne",
    "gt",
    "ge",
    # Container
    "getitem",
    # Numeric binary
    "add",
    "sub",
    "mul",
    "matmul",
    "truediv",
    "floordiv",
    "mod",
    "divmod",
    "pow",
    "lshift",
    "rshift",
    "and",
    "xor",
    "or",
    # Numeric unary
    "neg",
    "pos",
    "abs",
    "invert",
]
for special_method in SPECIAL_METHODS:
    name = f"__{special_method}__"
    del special_method
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
    one = Expr(i64, Int(1))
    res = one + one  # type: ignore
    assert res._parts == (i64, Call(add, (one, one)))


@dataclass(frozen=True)
class Var:
    name: str

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def to_egg(self, type: Type) -> py._Expr:
        return py.Var(self.name)


@dataclass(frozen=True)
class Call:
    fn: Function
    args: tuple[Expr, ...] = ()

    def __str__(self) -> str:
        return f"{self.fn}({', '.join(str(arg) for arg in self.args)})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Call):
            return NotImplemented
        return self._parts == __o._parts

    @property
    def _parts(self):
        return (self.fn, [a._parts for a in self.args])

    def to_egg(self, type: Type) -> py._Expr:
        return py.Call(self.fn.egg_name, [a._to_egg() for a in self.args])


@dataclass(frozen=True)
class Int:
    value: int

    def __str__(self) -> str:
        return f"i64({self.value})"

    def __repr__(self) -> str:
        return str(self)

    def to_egg(self, type: Type) -> py._Expr:
        return py.Lit(py.Int(self.value))


@dataclass(frozen=True)
class Unit:
    def __str__(self) -> str:
        return "Unit()"

    def __repr__(self) -> str:
        return str(self)

    def to_egg(self, type: Type) -> py._Expr:
        return py.Lit(py.Unit())


@dataclass(frozen=True)
class String:
    value: str

    def __str__(self) -> str:
        return f"String({self.value})"

    def __repr__(self) -> str:
        return str(self)

    def to_egg(self, type: Type) -> py._Expr:
        return py.Lit(py.String(self.value))


Literal = Int | Unit | String
ExprValue = Var | Call | Literal


@dataclass
class Inventory:
    """
    Collection of all the types and functions in the program.
    """

    lit_to_type: dict[type[py._Literal], Type] = field(default_factory=dict)

    def register_literal_type(self, lit: type[py._Literal], type: Type) -> None:
        self.lit_to_type[lit] = type

    def from_egg_expr(self, expr: py._Expr) -> Expr:
        if isinstance(expr, py.Var):
            raise NotImplementedError(
                "Cannot convert Var to Python expression because we don't know its type"
            )
        elif isinstance(expr, py.Call):
            return self.from_egg_call(expr.name, expr.args)
        elif isinstance(expr, py.Lit):
            value = expr.value
            tp = self.lit_to_type[type(value)]
            val: Literal
            if isinstance(value, py.Int):
                val = Int(value.value)
            elif isinstance(value, py.Unit):
                val = Unit()
            elif isinstance(value, py.String):
                val = String(value.value)
            else:
                raise NotImplementedError(f"Unknown literal {value}")
            return Expr(tp, val)
        raise NotImplementedError(f"Unknown egg expr {expr}")

    def from_egg_call(self, egg_fn: str, args: list[py._Expr]) -> Expr:
        fn = self.get_function(egg_fn)
        args = tuple(self.from_egg_expr(arg) for arg in args)
        return Expr(fn.return_type, Call(fn, args))
