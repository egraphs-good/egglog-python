from __future__ import annotations

from collections.abc import Collection, Hashable
from dataclasses import dataclass, field
from typing import Iterable, NoReturn, Union, cast

import black

# TODO: Add support for __dir__ and costring to aid in runtime completion
# TODO: Implement string with pretty printing?


@dataclass
class Namespace:
    """
    Represents a mapping of names to kinds and functions.

    This is used so we can map back from egg to python values.
    """

    values: dict[str, Union[Kind, Function]] = field(default_factory=dict)

    def __post_init__(self):
        # Verify that all values have the same name as their key
        for name, value in self.values.items():
            if value.name != name:
                raise ValueError(f"Name mismatch: {name} != {value.name}")

    def add_kind(self, kind: Kind):
        if kind.name in self.values:
            raise ValueError(f"Kind {kind.name} already exists")
        self.values[kind.name] = kind

    def add_function(self, fn: Function):
        if fn.name in self.values:
            raise ValueError(f"Function {fn.name} already exists")
        self.values[fn.name] = fn

    def get_kind(self, name: str) -> Kind:
        if name not in self.values:
            raise ValueError(f"Kind {name} does not exist")
        val = self.values[name]
        if not isinstance(val, Kind):
            raise TypeError(f"Value {name} is not a kind")
        return val

    def get_function(self, name: str) -> Function:
        if name not in self.values:
            raise ValueError(f"Function {name} does not exist")
        val = self.values[name]
        if not isinstance(val, Function):
            raise TypeError(f"Value {name} is not a function")
        return val

@dataclass
class Kind:
    """
    A kind represents the type of a type.

    In Python, this is similar to a generic type and in egg, this is like a presort or a sort.

    However, instead of only being for types with parameters, it is for all types, most just
    have no parameters.

    Currently, the only kind with paramaters is a "Map", which is a builtin presort in egg-smol.

    If/when custom generic types/presorts are supported in egg-smol, this will be expanded to
    support them.
    """

    # The name of the python class, for use when printing
    name: str
    # Typevariables which are then used in any classmethod or methods. These are only defined if this is a generic
    # kind, which is only the case for Map for now.
    typevariables: tuple[TypeVariable, ...] = ()

    # A mapping of classmethods to their functions
    # If you want to define __init__, you should define it as a classmethod
    classmethods: dict[str, Function] = field(default_factory=dict)
    methods: dict[str, Function] = field(default_factory=dict)

    # Whether this a literal type
    is_lit: bool = False

    def __post_init__(self):
        # Verify that all classmethods and methods have the same name as their key
        for name, fn in self.classmethods.items():
            if fn.name != name:
                raise TypeError(f"Classmethod {fn} has name {fn.name}, expected {name}")
        for name, fn in self.methods.items():
            if fn.name != name:
                raise TypeError(f"Method {fn} has name {fn.name}, expected {name}")

        # Verify that the __init__ classmethod returns this kind
        if "__init__" in self.classmethods:
            return_type = self.classmethods["__init__"].return_type
            if isinstance(return_type, TypeVariable):
                raise TypeError(
                    f"{self}.__init__ classmethod should return a concrete type, not a typevariable "
                )
            if return_type.kind != self:
                raise TypeError(
                    f"__init__ classmethod has return type {return_type}, expected {self}"
                )

    def __call__(self, *args: Expr | LitType) -> Expr:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if self.is_lit:
            assert isinstance(args[0], (int, str, type(None)))
            return Expr(self[()], Lit(args[0]))

        return BoundClassMethod(self, "__init__")(*args)

    def __getitem__(self, args: tuple[Type_, ...]) -> Type:
        return Type(self, args)

    def __getattr__(self, name: str) -> BoundClassMethod:
        return BoundClassMethod(self, name)

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Type:
    """
    A type is a bound kind, with all type variables replaced with concrete types.
    """

    kind: Kind
    args: tuple[Type_, ...]

    def __post_init__(self):
        # Verify that the number of args match the number of typevariables
        if len(self.args) != len(self.kind.typevariables):
            raise TypeError(
                f"Expected {len(self.kind.typevariables)} type arguments, got {len(self.args)}"
            )

    def __str__(self) -> str:
        if self.args:
            return f"{self.kind}[{', '.join(map(str, self.args))}]"
        else:
            return str(self.kind)

    def __repr__(self) -> str:
        return str(self)

    def __getattr__(self, name: str) -> BoundClassMethod:
        return BoundClassMethod(self, name)

    def __call__(self, *args: Expr) -> Expr:
        return BoundClassMethod(self, "__init__")(*args)


@dataclass(frozen=True)
class TypeVariable:
    identifier: Hashable

    def __str__(self) -> str:
        return str(self.identifier)

    def __repr__(self) -> str:
        return str(self)


Type_ = TypeVariable | Type


def test_type_str():
    i64 = Kind("i64")[()]
    K, V = TypeVariable("K"), TypeVariable("V")
    Map = Kind("Map", (K, V))
    assert str(i64) == "i64"
    assert str(K) == "K"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


@dataclass
class Function:
    # The name of the function
    name: str
    # The type of each arg
    arg_types: tuple[Type_, ...]
    # The return type
    return_type: Type_
    # The cost of the function
    cost: int = 0
    # An optional expression which can return the `old` and `new` values to return the merged value.
    merge: Expr | None = None

    def __call__(self, *args: Expr | LitType) -> Expr:
        return self._resolve_call(self, args)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def _resolve_call(
        self,
        callable: Callable_,
        args: Iterable[Expr | LitType],
        bound_type: Type | None = None,
    ) -> Expr:
        type_inference = (
            TypeInference.from_bound_type(bound_type) if bound_type else TypeInference()
        )
        resolved_args = [resolve_literal(arg) for arg in args]
        arg_types = [arg.type for arg in resolved_args]
        # Add self method for type checking, but don't add as arg, since already stored in callable
        if isinstance(callable, BoundMethod):
            arg_types.insert(0, callable.self.type)
        return_tp = type_inference.infer_return_type(
            self.arg_types, self.return_type, arg_types
        )
        return Expr(return_tp, Call(callable, tuple(a.value for a in resolved_args)))

    def __eq__(self, other) -> bool:
        """
        Override eq to use __expr_eq__ on merge expressions
        """
        if not isinstance(other, Function):
            return False
        return (
            self.name == other.name
            and self.arg_types == other.arg_types
            and self.return_type == other.return_type
            and self.cost == other.cost
            and self.merge.__expr_eq__(other.merge)
            if isinstance(self.merge, Expr) and isinstance(other.merge, Expr)
            else (self.merge is None and other.merge is None)
        )


# Mapping of literal types to their corresponding Kind
# i.e. int to i64.
# Used so that we can pass in unwrapped literals to functions and they will automatically be wrapped
TYPE_TO_LIT_KIND: dict[type, Kind] = {}


def resolve_literal(arg: Expr | LitType) -> Expr:
    if isinstance(arg, Expr):
        return arg
    return TYPE_TO_LIT_KIND[type(arg)](arg)


@dataclass
class BoundMethod:
    """
    A method which has been bound to a specific type.
    """

    # The type of the self argument
    self: Expr
    # The function that is bound
    name: str

    def __post_init__(self):
        if self.name not in self._methods:
            raise TypeError(f"Method {self.name} not found on {self.self.type}")

    def __call__(self, *args: Expr | LitType) -> Expr:
        fn = self._methods[self.name]
        return fn._resolve_call(self, args)

    @property
    def _methods(self):
        return self.self.type.kind.methods

    def __eq__(self, other: object) -> bool:
        """
        Override == to use __expr_eq__ since __eq__ can be overriden on Exprs
        """
        if isinstance(other, BoundMethod):
            return self.self.__expr_eq__(other.self) and self.name == other.name
        return False


@dataclass
class BoundClassMethod:
    """
    A class method which has been bound to a specific type.
    """

    # Bound or unbound type (kind) that this class method is bound to
    type: Type | Kind
    # The class method that is bound
    name: str

    # Validate that this name is on the type or kind on init
    def __post_init__(self):
        if self.name not in self._classmethods:
            raise TypeError(f"Class method {self.name} not found on {self.type}")

    def __call__(self, *args: Expr | LitType) -> Expr:
        fn = self._classmethods[self.name]
        return fn._resolve_call(
            self, args, self.type if isinstance(self.type, Type) else None
        )

    @property
    def kind(self) -> Kind:
        """
        The kind of the type that this class method is bound to.
        """
        return self.type.kind if isinstance(self.type, Type) else self.type

    @property
    def _classmethods(self):
        return self.kind.classmethods


Callable_ = Union[Function, BoundMethod, BoundClassMethod]


def test_function_call():
    i64 = Kind("i64")[()]
    one = Function("one", (), i64)
    assert one().__expr_eq__(Expr(i64, Call(one, ())))


def test_classmethod_call():
    from pytest import raises

    K, V = TypeVariable("K"), TypeVariable("V")
    Map = Kind("Map", (K, V))
    Map.classmethods["create"] = Function("create", (), Map[K, V])
    with raises(TypeError):
        Map.create()

    i64 = Kind("i64")[()]
    unit = Kind("Unit")[()]
    assert (
        Map[i64, unit].create().__expr_parts__
        == Expr(
            Map[i64, unit], Call(BoundClassMethod(Map[i64, unit], "create"), ())
        ).__expr_parts__
    )


@dataclass
class TypeInference:
    _typevar_to_value: dict[TypeVariable, Type] = field(default_factory=dict)

    @classmethod
    def from_bound_type(cls, tp: Type) -> TypeInference:
        """
        Create a TypeInference from a bound type, where all type variables are replaced with concrete types.
        """
        d: dict[TypeVariable, Type] = {}
        for tpv, tpa in zip(tp.kind.typevariables, tp.args):
            if not isinstance(tpa, Type):
                raise TypeError(f"Expected type, got {tpa}")
            d[tpv] = tpa
        return cls(d)

    def infer_return_type(
        self,
        unbound_arg_types: Collection[Type_],
        unbound_return_type: Type_,
        arg_types: Collection[Type],
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

    i64 = Kind("i64")[()]
    unit = Kind("Unit")[()]
    K, V = TypeVariable("K"), TypeVariable("V")
    Map = Kind("Map", (K, V))
    ti = TypeInference()
    assert ti.infer_return_type([i64], i64, [i64]) == i64
    with pytest.raises(TypeError):
        ti.infer_return_type([i64], i64, [unit])
    with pytest.raises(TypeError):
        ti.infer_return_type([], i64, [unit])

    assert ti.infer_return_type([Map[K, V], K], V, [Map[i64, unit], i64]) == unit

    with pytest.raises(TypeError):
        ti.infer_return_type([Map[K, V], K], V, [Map[i64, unit], unit])


LitType = Union[int, str, None]


@dataclass(frozen=True)
class Lit:
    value: LitType

    def __str__(self) -> str:
        if self.value is None:
            return "unit()"
        if isinstance(self.value, int):
            return f"i64({self.value})"
        if isinstance(self.value, str):
            return f"string({repr(self.value)})"
        raise TypeError(f"Unexpected literal type: {type(self.value)}")

    def __repr__(self) -> str:
        return str(self)


def test_print_literal():
    assert str(Lit(1)) == "i64(1)"
    assert str(Lit(None)) == "unit()"
    assert str(Lit("hi")) == "string('hi')"


@dataclass(frozen=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class Call:
    fn: Callable_
    args: tuple[Expr_, ...]

    def __post_init__(self):
        # Verify that if its a bound method and its a special method, it has the write number of args
        if isinstance(self.fn, BoundMethod):
            name = self.fn.name
            n_args = len(self.args)
            if name in UNARY_METHODS and n_args != 0:
                raise TypeError(f"Expected 0 args, for method {name} got {n_args}")
            elif (name in BINARY_METHODS or name == "__getitem__") and n_args != 1:
                raise TypeError(f"Expected 1 arg, for method {name} got {n_args}")

    def __str__(self) -> str:
        fn, args = self.fn, self.args
        if isinstance(fn, Function):
            fn_str = fn.name
        elif isinstance(fn, BoundClassMethod):
            name = fn.name
            kind_name = fn.kind.name
            if name == "__init__":
                fn_str = kind_name
            else:
                fn_str = f"{kind_name}.{name}"
        else:
            name, slf = fn.name, fn.self
            if name in UNARY_METHODS:
                return f"{UNARY_METHODS[name]}{slf}"
            elif name in BINARY_METHODS:
                return f"({slf} {BINARY_METHODS[name]} {args[0]})"
            elif name == "__getitem__":
                return f"{slf}[{args[0]}]"
            fn_str = f"{slf}.{name}"
        return f"{fn_str}({', '.join(map(str, args))})"

    def __repr__(self) -> str:
        return str(self)


# An untyped expression
Expr_ = Union[Lit, Var, Call]


def test_expr_str():
    i64 = Kind("i64")[()]
    add = Function("add", (i64, i64), i64)
    add_call = Call(add, (Lit(1), Var("x")))
    assert str(add_call) == "add(i64(1), x)"


@dataclass
class Expr:
    """
    Create an expr object that behaves like a python object of the `type`, by overloading all of the dunder functions.
    """

    type: Type
    value: Expr_

    def __getattr__(self, name: str) -> BoundMethod:
        return BoundMethod(self, name)

    def __str__(self) -> str:
        return blacken_python_expression(str(self.value))

    def __repr__(self) -> str:
        return str(self)

    # Have __eq__ take no NoReturn (aka Never https://docs.python.org/3/library/typing.html#typing.Never) because
    # we don't wany any type that MyPy thinks is an expr to be used with __eq__.
    # That's because we want to reserve __eq__ for domain specific equality checks, overloading this method.
    # To check if two exprs are equal, use the expr_eq method.
    def __eq__(self, other: NoReturn) -> Expr:  # type: ignore
        return BoundMethod(self, "__eq__")(cast(Expr, other))

    def __expr_eq__(self, other: Expr) -> bool:
        """
        Check if two exprs are equal.
        """
        return self.__expr_parts__ == other.__expr_parts__

    @property
    def __expr_parts__(self) -> tuple[Type, Expr_]:
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
        lambda self, *args, name=name: BoundMethod(self, name)(*args),
    )


def test_expr_special():
    i64Kind = Kind("i64")
    i64 = i64Kind[()]
    add = Function("__add__", (i64, i64), i64)
    i64Kind.methods["__add__"] = add
    one = Expr(i64, Lit(1))
    res = one + one  # type: ignore
    expected_res = Expr(i64, Call(BoundMethod(one, "__add__"), (Lit(1),)))
    assert str(expected_res) == "i64(1) + i64(1)"
    assert str(res) == "i64(1) + i64(1)"
    assert res.__expr_parts__ == expected_res.__expr_parts__


def test_print_method_call():
    i64 = Kind("i64", is_lit=True)
    i64_ = i64[()]
    add = Function("add", (i64_, i64_), i64_)
    i64.methods["add"] = add

    one = i64(1)
    assert str(one.add(one)) == "i64(1).add(i64(1))"


def test_print_classmethod_call():
    i64 = Kind("i64", is_lit=True)
    i64_ = i64[()]
    add = Function("add", (i64_, i64_), i64_)
    i64.classmethods["add"] = add

    one = i64(1)
    assert str(i64.add(one, one)) == "i64.add(i64(1), i64(1))"


def test_print_special_method_call():
    i64 = Kind("i64", is_lit=True)
    i64_ = i64[()]
    add = Function("__add__", (i64_, i64_), i64_)
    i64.methods["__add__"] = add

    one = i64(1)
    res = one + one  # type: ignore
    assert str(res) == "i64(1) + i64(1)"


BLACK_MODE = black.Mode(line_length=120)  # type: ignore


def blacken_python_expression(expr: str) -> str:
    """
    Runs black on a Python expression, to remove excess paranthesis and wrap it.
    """
    return black.format_str("x = " + expr, mode=BLACK_MODE)[4:-1]
