# mypy: disable-error-code="empty-body"
"""
Builtin sorts and function to egg.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import partial, reduce
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypeVar, cast, overload

from typing_extensions import TypeVarTuple, Unpack

from .conversion import convert, converter, get_type_args
from .declarations import *
from .egraph import BaseExpr, BuiltinExpr, expr_fact, function, get_current_ruleset, method
from .functionalize import functionalize
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction
from .thunk import Thunk

if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "BigInt",
    "BigIntLike",
    "BigRat",
    "BigRatLike",
    "Bool",
    "BoolLike",
    "BuiltinEvalError",
    "Map",
    "MapLike",
    "MultiSet",
    "PyObject",
    "Rational",
    "Set",
    "SetLike",
    "String",
    "StringLike",
    "Unit",
    "UnstableFn",
    "Vec",
    "VecLike",
    "f64",
    "f64Like",
    "i64",
    "i64Like",
    "join",
    "py_eval",
    "py_eval_fn",
    "py_exec",
]


class BuiltinEvalError(Exception):
    """
    Raised when an builtin cannot be evaluated into a Python primitive because it is complex.

    Try extracting this expression first.
    """

    def __str__(self) -> str:
        return f"Cannot evaluate builtin expression into a Python primitive. Try extracting this expression first: {super().__str__()}"


class Unit(BuiltinExpr, egg_sort="Unit"):
    """
    The unit type. This is used to reprsent if a value exists in the e-graph or not.
    """

    def __init__(self) -> None: ...

    @method(preserve=True)
    def __bool__(self) -> bool:
        return bool(expr_fact(self))


class String(BuiltinExpr):
    @method(preserve=True)
    def eval(self) -> str:
        value = _extract_lit(self)
        assert isinstance(value, str)
        return value

    def __init__(self, value: str) -> None: ...

    @method(egg_fn="replace")
    def replace(self, old: StringLike, new: StringLike) -> String: ...


StringLike: TypeAlias = String | str


@function(egg_fn="+", builtin=True)
def join(*strings: StringLike) -> String: ...


converter(str, String, String)


class Bool(BuiltinExpr, egg_sort="bool"):
    @method(preserve=True)
    def eval(self) -> bool:
        value = _extract_lit(self)
        assert isinstance(value, bool)
        return value

    @method(preserve=True)
    def __bool__(self) -> bool:
        return self.eval()

    def __init__(self, value: bool) -> None: ...

    @method(egg_fn="not")
    def __invert__(self) -> Bool: ...

    @method(egg_fn="and")
    def __and__(self, other: BoolLike) -> Bool: ...

    @method(egg_fn="or")
    def __or__(self, other: BoolLike) -> Bool: ...

    @method(egg_fn="xor")
    def __xor__(self, other: BoolLike) -> Bool: ...

    @method(egg_fn="=>")
    def implies(self, other: BoolLike) -> Bool: ...


BoolLike: TypeAlias = Bool | bool


converter(bool, Bool, Bool)


class i64(BuiltinExpr):  # noqa: N801
    @method(preserve=True)
    def eval(self) -> int:
        value = _extract_lit(self)
        assert isinstance(value, int)
        return value

    @method(preserve=True)
    def __index__(self) -> int:
        return self.eval()

    @method(preserve=True)
    def __int__(self) -> int:
        return self.eval()

    def __init__(self, value: int) -> None: ...

    @method(egg_fn="+")
    def __add__(self, other: i64Like) -> i64: ...

    @method(egg_fn="-")
    def __sub__(self, other: i64Like) -> i64: ...

    @method(egg_fn="*")
    def __mul__(self, other: i64Like) -> i64: ...

    @method(egg_fn="/")
    def __truediv__(self, other: i64Like) -> i64: ...

    @method(egg_fn="%")
    def __mod__(self, other: i64Like) -> i64: ...

    @method(egg_fn="&")
    def __and__(self, other: i64Like) -> i64: ...

    @method(egg_fn="|")
    def __or__(self, other: i64Like) -> i64: ...

    @method(egg_fn="^")
    def __xor__(self, other: i64Like) -> i64: ...

    @method(egg_fn="<<")
    def __lshift__(self, other: i64Like) -> i64: ...

    @method(egg_fn=">>")
    def __rshift__(self, other: i64Like) -> i64: ...

    def __radd__(self, other: i64Like) -> i64: ...

    def __rsub__(self, other: i64Like) -> i64: ...

    def __rmul__(self, other: i64Like) -> i64: ...

    def __rtruediv__(self, other: i64Like) -> i64: ...

    def __rmod__(self, other: i64Like) -> i64: ...

    def __rand__(self, other: i64Like) -> i64: ...

    def __ror__(self, other: i64Like) -> i64: ...

    def __rxor__(self, other: i64Like) -> i64: ...

    def __rlshift__(self, other: i64Like) -> i64: ...

    def __rrshift__(self, other: i64Like) -> i64: ...

    @method(egg_fn="log2")
    def log2(self) -> i64: ...

    @method(egg_fn="not-i64")
    def __invert__(self) -> i64: ...

    @method(egg_fn="<")
    def __lt__(self, other: i64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @method(egg_fn=">")
    def __gt__(self, other: i64Like) -> Unit: ...

    @method(egg_fn="<=")
    def __le__(self, other: i64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @method(egg_fn=">=")
    def __ge__(self, other: i64Like) -> Unit: ...

    @method(egg_fn="min")
    def min(self, other: i64Like) -> i64: ...

    @method(egg_fn="max")
    def max(self, other: i64Like) -> i64: ...

    @method(egg_fn="to-string")
    def to_string(self) -> String: ...

    @method(egg_fn="bool-<")
    def bool_lt(self, other: i64Like) -> Bool: ...

    @method(egg_fn="bool->")
    def bool_gt(self, other: i64Like) -> Bool: ...

    @method(egg_fn="bool-<=")
    def bool_le(self, other: i64Like) -> Bool: ...

    @method(egg_fn="bool->=")
    def bool_ge(self, other: i64Like) -> Bool: ...


# The types which can be convertered into an i64
i64Like: TypeAlias = i64 | int  # noqa: N816, PYI042

converter(int, i64, i64)


@function(builtin=True, egg_fn="count-matches")
def count_matches(s: StringLike, pattern: StringLike) -> i64: ...


class f64(BuiltinExpr):  # noqa: N801
    @method(preserve=True)
    def eval(self) -> float:
        value = _extract_lit(self)
        assert isinstance(value, float)
        return value

    @method(preserve=True)
    def __float__(self) -> float:
        return self.eval()

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.eval())

    def __init__(self, value: float) -> None: ...

    @method(egg_fn="neg")
    def __neg__(self) -> f64: ...

    @method(egg_fn="+")
    def __add__(self, other: f64Like) -> f64: ...

    @method(egg_fn="-")
    def __sub__(self, other: f64Like) -> f64: ...

    @method(egg_fn="*")
    def __mul__(self, other: f64Like) -> f64: ...

    @method(egg_fn="/")
    def __truediv__(self, other: f64Like) -> f64: ...

    @method(egg_fn="%")
    def __mod__(self, other: f64Like) -> f64: ...

    @method(egg_fn="^")
    def __pow__(self, other: f64Like) -> f64: ...

    def __radd__(self, other: f64Like) -> f64: ...

    def __rsub__(self, other: f64Like) -> f64: ...

    def __rmul__(self, other: f64Like) -> f64: ...

    def __rtruediv__(self, other: f64Like) -> f64: ...

    def __rmod__(self, other: f64Like) -> f64: ...

    @method(egg_fn="<")
    def __lt__(self, other: f64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @method(egg_fn=">")
    def __gt__(self, other: f64Like) -> Unit: ...

    @method(egg_fn="<=")
    def __le__(self, other: f64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @method(egg_fn=">=")
    def __ge__(self, other: f64Like) -> Unit: ...

    @method(egg_fn="min")
    def min(self, other: f64Like) -> f64: ...

    @method(egg_fn="max")
    def max(self, other: f64Like) -> f64: ...

    @method(egg_fn="to-i64")
    def to_i64(self) -> i64: ...

    @method(egg_fn="to-f64")
    @classmethod
    def from_i64(cls, i: i64) -> f64: ...

    @method(egg_fn="to-string")
    def to_string(self) -> String: ...


f64Like: TypeAlias = f64 | float  # noqa: N816, PYI042


converter(float, f64, f64)


T = TypeVar("T", bound=BaseExpr)
V = TypeVar("V", bound=BaseExpr)


class Map(BuiltinExpr, Generic[T, V]):
    @method(preserve=True)
    def eval(self) -> dict[T, V]:
        call = _extract_call(self)
        expr = cast("RuntimeExpr", self)
        d = {}
        while call.callable != ClassMethodRef("Map", "empty"):
            msg = "Map can only be evaluated if it is empty or a series of inserts."
            if call.callable != MethodRef("Map", "insert"):
                raise BuiltinEvalError(msg)
            call_typed, k_typed, v_typed = call.args
            if not isinstance(call_typed.expr, CallDecl):
                raise BuiltinEvalError(msg)
            k = cast("T", expr.__with_expr__(k_typed))
            v = cast("V", expr.__with_expr__(v_typed))
            d[k] = v
            call = call_typed.expr
        return d

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.eval())

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.eval())

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.eval()

    @method(egg_fn="map-empty")
    @classmethod
    def empty(cls) -> Map[T, V]: ...

    @method(egg_fn="map-insert")
    def insert(self, key: T, value: V) -> Map[T, V]: ...

    @method(egg_fn="map-get")
    def __getitem__(self, key: T) -> V: ...

    @method(egg_fn="map-not-contains")
    def not_contains(self, key: T) -> Unit: ...

    @method(egg_fn="map-contains")
    def contains(self, key: T) -> Unit: ...

    @method(egg_fn="map-remove")
    def remove(self, key: T) -> Map[T, V]: ...

    @method(egg_fn="rebuild")
    def rebuild(self) -> Map[T, V]: ...


TO = TypeVar("TO")
VO = TypeVar("VO")

converter(
    dict,
    Map,
    lambda t: reduce(
        (lambda acc, kv: acc.insert(convert(kv[0], get_type_args()[0]), convert(kv[1], get_type_args()[1]))),
        t.items(),
        Map[get_type_args()].empty(),  # type: ignore[misc]
    ),
)

MapLike: TypeAlias = Map[T, V] | dict[TO, VO]


class Set(BuiltinExpr, Generic[T]):
    @method(preserve=True)
    def eval(self) -> set[T]:
        call = _extract_call(self)
        if call.callable != InitRef("Set"):
            msg = "Set can only be initialized with the Set constructor."
            raise BuiltinEvalError(msg)
        return {cast("T", cast("RuntimeExpr", self).__with_expr__(x)) for x in call.args}

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.eval())

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.eval())

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.eval()

    @method(egg_fn="set-of")
    def __init__(self, *args: T) -> None: ...

    @method(egg_fn="set-empty")
    @classmethod
    def empty(cls) -> Set[T]: ...

    @method(egg_fn="set-insert")
    def insert(self, value: T) -> Set[T]: ...

    @method(egg_fn="set-not-contains")
    def not_contains(self, value: T) -> Unit: ...

    @method(egg_fn="set-contains")
    def contains(self, value: T) -> Unit: ...

    @method(egg_fn="set-remove")
    def remove(self, value: T) -> Set[T]: ...

    @method(egg_fn="set-union")
    def __or__(self, other: Set[T]) -> Set[T]: ...

    @method(egg_fn="set-diff")
    def __sub__(self, other: Set[T]) -> Set[T]: ...

    @method(egg_fn="set-intersect")
    def __and__(self, other: Set[T]) -> Set[T]: ...

    @method(egg_fn="rebuild")
    def rebuild(self) -> Set[T]: ...


converter(
    set,
    Set,
    lambda t: Set[get_type_args()[0]](  # type: ignore[misc,operator]
        *(convert(x, get_type_args()[0]) for x in t)
    ),
)

SetLike: TypeAlias = Set[T] | set[TO]


class MultiSet(BuiltinExpr, Generic[T]):
    @method(preserve=True)
    def eval(self) -> list[T]:
        call = _extract_call(self)
        if call.callable != InitRef("MultiSet"):
            msg = "MultiSet can only be initialized with the MultiSet constructor."
            raise BuiltinEvalError(msg)
        return [cast("T", cast("RuntimeExpr", self).__with_expr__(x)) for x in call.args]

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.eval())

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.eval())

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.eval()

    @method(egg_fn="multiset-of")
    def __init__(self, *args: T) -> None: ...

    @method(egg_fn="multiset-insert")
    def insert(self, value: T) -> MultiSet[T]: ...

    @method(egg_fn="multiset-not-contains")
    def not_contains(self, value: T) -> Unit: ...

    @method(egg_fn="multiset-contains")
    def contains(self, value: T) -> Unit: ...

    @method(egg_fn="multiset-remove")
    def remove(self, value: T) -> MultiSet[T]: ...

    @method(egg_fn="multiset-length")
    def length(self) -> i64: ...

    @method(egg_fn="multiset-pick")
    def pick(self) -> T: ...

    @method(egg_fn="multiset-sum")
    def __add__(self, other: MultiSet[T]) -> MultiSet[T]: ...

    @method(egg_fn="unstable-multiset-map", reverse_args=True)
    def map(self, f: Callable[[T], T]) -> MultiSet[T]: ...


class Rational(BuiltinExpr):
    @method(preserve=True)
    def eval(self) -> Fraction:
        call = _extract_call(self)
        if call.callable != InitRef("Rational"):
            msg = "Rational can only be initialized with the Rational constructor."
            raise BuiltinEvalError(msg)

        def _to_int(e: TypedExprDecl) -> int:
            expr = e.expr
            if not isinstance(expr, LitDecl):
                msg = "Rational can only be initialized with literals"
                raise BuiltinEvalError(msg)
            assert isinstance(expr.value, int)
            return expr.value

        num, den = call.args
        return Fraction(_to_int(num), _to_int(den))

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.eval())

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.eval())

    @method(egg_fn="rational")
    def __init__(self, num: i64Like, den: i64Like) -> None: ...

    @method(egg_fn="to-f64")
    def to_f64(self) -> f64: ...

    @method(egg_fn="+")
    def __add__(self, other: Rational) -> Rational: ...

    @method(egg_fn="-")
    def __sub__(self, other: Rational) -> Rational: ...

    @method(egg_fn="*")
    def __mul__(self, other: Rational) -> Rational: ...

    @method(egg_fn="/")
    def __truediv__(self, other: Rational) -> Rational: ...

    @method(egg_fn="min")
    def min(self, other: Rational) -> Rational: ...

    @method(egg_fn="max")
    def max(self, other: Rational) -> Rational: ...

    @method(egg_fn="neg")
    def __neg__(self) -> Rational: ...

    @method(egg_fn="abs")
    def __abs__(self) -> Rational: ...

    @method(egg_fn="floor")
    def floor(self) -> Rational: ...

    @method(egg_fn="ceil")
    def ceil(self) -> Rational: ...

    @method(egg_fn="round")
    def round(self) -> Rational: ...

    @method(egg_fn="pow")
    def __pow__(self, other: Rational) -> Rational: ...

    @method(egg_fn="log")
    def log(self) -> Rational: ...

    @method(egg_fn="sqrt")
    def sqrt(self) -> Rational: ...

    @method(egg_fn="cbrt")
    def cbrt(self) -> Rational: ...

    @method(egg_fn="numer")  # type: ignore[misc]
    @property
    def numer(self) -> i64: ...

    @method(egg_fn="denom")  # type: ignore[misc]
    @property
    def denom(self) -> i64: ...


class BigInt(BuiltinExpr):
    @method(preserve=True)
    def eval(self) -> int:
        call = _extract_call(self)
        if call.callable != ClassMethodRef("BigInt", "from_string"):
            msg = "BigInt can only be initialized with the BigInt constructor."
            raise BuiltinEvalError(msg)
        (s,) = call.args
        if not isinstance(s.expr, LitDecl):
            msg = "BigInt can only be initialized with literals"
            raise BuiltinEvalError(msg)
        assert isinstance(s.expr.value, str)
        return int(s.expr.value)

    @method(preserve=True)
    def __index__(self) -> int:
        return self.eval()

    @method(preserve=True)
    def __int__(self) -> int:
        return self.eval()

    @method(egg_fn="from-string")
    @classmethod
    def from_string(cls, s: StringLike) -> BigInt: ...

    @method(egg_fn="bigint")
    def __init__(self, value: i64Like) -> None: ...

    @method(egg_fn="+")
    def __add__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="-")
    def __sub__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="*")
    def __mul__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="/")
    def __truediv__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="%")
    def __mod__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="&")
    def __and__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="|")
    def __or__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="^")
    def __xor__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="<<")
    def __lshift__(self, other: i64Like) -> BigInt: ...

    @method(egg_fn=">>")
    def __rshift__(self, other: i64Like) -> BigInt: ...

    def __radd__(self, other: BigIntLike) -> BigInt: ...

    def __rsub__(self, other: BigIntLike) -> BigInt: ...

    def __rmul__(self, other: BigIntLike) -> BigInt: ...

    def __rtruediv__(self, other: BigIntLike) -> BigInt: ...

    def __rmod__(self, other: BigIntLike) -> BigInt: ...

    def __rand__(self, other: BigIntLike) -> BigInt: ...

    def __ror__(self, other: BigIntLike) -> BigInt: ...

    def __rxor__(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="not-Z")
    def __invert__(self) -> BigInt: ...

    @method(egg_fn="bits")
    def bits(self) -> BigInt: ...

    @method(egg_fn="<")
    def __lt__(self, other: BigIntLike) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @method(egg_fn=">")
    def __gt__(self, other: BigIntLike) -> Unit: ...

    @method(egg_fn="<=")
    def __le__(self, other: BigIntLike) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @method(egg_fn=">=")
    def __ge__(self, other: BigIntLike) -> Unit: ...

    @method(egg_fn="min")
    def min(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="max")
    def max(self, other: BigIntLike) -> BigInt: ...

    @method(egg_fn="to-string")
    def to_string(self) -> String: ...

    @method(egg_fn="bool-=")
    def bool_eq(self, other: BigIntLike) -> Bool: ...

    @method(egg_fn="bool-<")
    def bool_lt(self, other: BigIntLike) -> Bool: ...

    @method(egg_fn="bool->")
    def bool_gt(self, other: BigIntLike) -> Bool: ...

    @method(egg_fn="bool-<=")
    def bool_le(self, other: BigIntLike) -> Bool: ...

    @method(egg_fn="bool->=")
    def bool_ge(self, other: BigIntLike) -> Bool: ...


converter(i64, BigInt, lambda i: BigInt(i))

BigIntLike: TypeAlias = BigInt | i64Like


class BigRat(BuiltinExpr):
    @method(preserve=True)
    def eval(self) -> Fraction:
        call = _extract_call(self)
        if call.callable != InitRef("BigRat"):
            msg = "BigRat can only be initialized with the BigRat constructor."
            raise BuiltinEvalError(msg)

        def _to_fraction(e: TypedExprDecl) -> Fraction:
            expr = e.expr
            if not isinstance(expr, CallDecl) or expr.callable != ClassMethodRef("BigInt", "from_string"):
                msg = "BigRat can only be initialized BigInt strings"
                raise BuiltinEvalError(msg)
            (s,) = expr.args
            if not isinstance(s.expr, LitDecl):
                msg = "BigInt can only be initialized with literals"
                raise BuiltinEvalError(msg)
            assert isinstance(s.expr.value, str)
            return Fraction(s.expr.value)

        num, den = call.args
        return Fraction(_to_fraction(num), _to_fraction(den))

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.eval())

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.eval())

    @method(egg_fn="bigrat")
    def __init__(self, num: BigIntLike, den: BigIntLike) -> None: ...

    @method(egg_fn="to-f64")
    def to_f64(self) -> f64: ...

    @method(egg_fn="+")
    def __add__(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="-")
    def __sub__(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="*")
    def __mul__(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="/")
    def __truediv__(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="min")
    def min(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="max")
    def max(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="neg")
    def __neg__(self) -> BigRat: ...

    @method(egg_fn="abs")
    def __abs__(self) -> BigRat: ...

    @method(egg_fn="floor")
    def floor(self) -> BigRat: ...

    @method(egg_fn="ceil")
    def ceil(self) -> BigRat: ...

    @method(egg_fn="round")
    def round(self) -> BigRat: ...

    @method(egg_fn="pow")
    def __pow__(self, other: BigRatLike) -> BigRat: ...

    @method(egg_fn="log")
    def log(self) -> BigRat: ...

    @method(egg_fn="sqrt")
    def sqrt(self) -> BigRat: ...

    @method(egg_fn="cbrt")
    def cbrt(self) -> BigRat: ...

    @method(egg_fn="numer")  # type: ignore[misc]
    @property
    def numer(self) -> BigInt: ...

    @method(egg_fn="denom")  # type: ignore[misc]
    @property
    def denom(self) -> BigInt: ...

    @method(egg_fn="<")
    def __lt__(self, other: BigRatLike) -> Unit: ...  # type: ignore[has-type]

    @method(egg_fn=">")
    def __gt__(self, other: BigRatLike) -> Unit: ...

    @method(egg_fn=">=")
    def __ge__(self, other: BigRatLike) -> Unit: ...  # type: ignore[has-type]

    @method(egg_fn="<=")
    def __le__(self, other: BigRatLike) -> Unit: ...


converter(Fraction, BigRat, lambda f: BigRat(f.numerator, f.denominator))
BigRatLike: TypeAlias = BigRat | Fraction


class Vec(BuiltinExpr, Generic[T]):
    @method(preserve=True)
    def eval(self) -> tuple[T, ...]:
        call = _extract_call(self)
        if call.callable == ClassMethodRef("Vec", "empty"):
            return ()

        if call.callable != InitRef("Vec"):
            msg = "Vec can only be initialized with the Vec constructor."
            raise BuiltinEvalError(msg)
        return tuple(cast("T", cast("RuntimeExpr", self).__with_expr__(x)) for x in call.args)

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.eval())

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.eval())

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.eval()

    @method(egg_fn="vec-of")
    def __init__(self, *args: T) -> None: ...

    @method(egg_fn="vec-empty")
    @classmethod
    def empty(cls) -> Vec[T]: ...

    @method(egg_fn="vec-append")
    def append(self, *others: Vec[T]) -> Vec[T]: ...

    @method(egg_fn="vec-push")
    def push(self, value: T) -> Vec[T]: ...

    @method(egg_fn="vec-pop")
    def pop(self) -> Vec[T]: ...

    @method(egg_fn="vec-not-contains")
    def not_contains(self, value: T) -> Unit: ...

    @method(egg_fn="vec-contains")
    def contains(self, value: T) -> Unit: ...

    @method(egg_fn="vec-length")
    def length(self) -> i64: ...

    @method(egg_fn="vec-get")
    def __getitem__(self, index: i64Like) -> T: ...

    @method(egg_fn="rebuild")
    def rebuild(self) -> Vec[T]: ...

    @method(egg_fn="vec-remove")
    def remove(self, index: i64Like) -> Vec[T]: ...

    @method(egg_fn="vec-set")
    def set(self, index: i64Like, value: T) -> Vec[T]: ...


for sequence_type in (list, tuple):
    converter(
        sequence_type,
        Vec,
        lambda t: Vec[get_type_args()[0]](  # type: ignore[misc,operator]
            *(convert(x, get_type_args()[0]) for x in t)
        ),
    )

VecLike: TypeAlias = Vec[T] | tuple[TO, ...] | list[TO]


class PyObject(BuiltinExpr):
    @method(preserve=True)
    def eval(self) -> object:
        expr = cast("RuntimeExpr", self).__egg_typed_expr__.expr
        if not isinstance(expr, PyObjectDecl):
            msg = "PyObject can only be evaluated if it is a PyObject literal"
            raise BuiltinEvalError(msg)
        return expr.value

    def __init__(self, value: object) -> None: ...

    @method(egg_fn="py-from-string")
    @classmethod
    def from_string(cls, s: StringLike) -> PyObject: ...

    @method(egg_fn="py-to-string")
    def to_string(self) -> String: ...

    @method(egg_fn="py-to-bool")
    def to_bool(self) -> Bool: ...

    @method(egg_fn="py-dict-update")
    def dict_update(self, *keys_and_values: object) -> PyObject: ...

    @method(egg_fn="py-from-int")
    @classmethod
    def from_int(cls, i: i64Like) -> PyObject: ...

    @method(egg_fn="py-dict")
    @classmethod
    def dict(cls, *keys_and_values: object) -> PyObject: ...


converter(object, PyObject, PyObject)


@function(builtin=True, egg_fn="py-eval")
def py_eval(code: StringLike, globals: object = PyObject.dict(), locals: object = PyObject.dict()) -> PyObject: ...


class PyObjectFunction(Protocol):
    def __call__(self, *__args: PyObject) -> PyObject: ...


def py_eval_fn(fn: Callable) -> PyObjectFunction:
    """
    Takes a python callable and maps it to a callable which takes and returns PyObjects.

    It translates it to a call which uses `py_eval` to call the function, passing in the
    args as locals, and using the globals from function.
    """

    def inner(*__args: PyObject, __fn: Callable = fn) -> PyObject:
        new_kvs: list[object] = []
        eval_str = "__fn("
        for i, arg in enumerate(__args):
            new_kvs.extend((f"__arg_{i}", arg))
            eval_str += f"__arg_{i}, "
        eval_str += ")"
        return py_eval(eval_str, PyObject({"__fn": __fn}).dict_update(*new_kvs), __fn.__globals__)

    return inner


@function(builtin=True, egg_fn="py-exec")
def py_exec(code: StringLike, globals: object = PyObject.dict(), locals: object = PyObject.dict()) -> PyObject:
    """
    Copies the locals, execs the Python code, and returns the locals with any updates.
    """


TS = TypeVarTuple("TS")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class UnstableFn(BuiltinExpr, Generic[T, Unpack[TS]]):
    @overload
    def __init__(self, f: Callable[[Unpack[TS]], T]) -> None: ...

    @overload
    def __init__(self, f: Callable[[T1, Unpack[TS]], T], _a: T1, /) -> None: ...

    @overload
    def __init__(self, f: Callable[[T1, T2, Unpack[TS]], T], _a: T1, _b: T2, /) -> None: ...

    # Removing due to bug in MyPy
    # https://github.com/python/mypy/issues/17212
    # @overload
    # def __init__(self, f: Callable[[T1, T2, T3, Unpack[TS]], T], _a: T1, _b: T2, _c: T3, /) -> None: ...

    # etc, for partial application

    @method(egg_fn="unstable-fn")
    def __init__(self, f, *partial) -> None: ...

    @method(egg_fn="unstable-app")
    def __call__(self, *args: Unpack[TS]) -> T: ...


# Method Type is for builtins like __getitem__
converter(MethodType, UnstableFn, lambda m: UnstableFn(m.__func__, m.__self__))
converter(RuntimeFunction, UnstableFn, UnstableFn)
converter(partial, UnstableFn, lambda p: UnstableFn(p.func, *p.args))


def _convert_function(a: FunctionType) -> UnstableFn:
    """
    Converts a function type to an unstable function

    Would just be UnstableFn(function(a)) but we have to look for any nonlocals and globals
    which are runtime expressions with `var`s in them and add them as args to the function
    """
    # Update annotations of a to be the type we are trying to convert to
    return_tp, *arg_tps = get_type_args()
    a.__annotations__ = {
        "return": return_tp,
        # The first varnames should always be the arg names
        **dict(zip(a.__code__.co_varnames, arg_tps, strict=False)),
    }
    # Modify name to make it unique
    # a.__name__ = f"{a.__name__} {hash(a.__code__)}"
    transformed_fn = functionalize(a, value_to_annotation)
    assert isinstance(transformed_fn, partial)
    return UnstableFn(
        function(ruleset=get_current_ruleset(), use_body_as_name=True, subsume=True)(transformed_fn.func),
        *transformed_fn.args,
    )


def value_to_annotation(a: object) -> type | None:
    # only lift runtime expressions (which could contain vars) not any other nonlocals/globals we use in the function
    if not isinstance(a, RuntimeExpr):
        return None
    return cast("type", RuntimeClass(Thunk.value(a.__egg_decls__), a.__egg_typed_expr__.tp.to_var()))


converter(FunctionType, UnstableFn, _convert_function)


def _extract_lit(e: BaseExpr) -> LitType:
    """
    Special case extracting literals to make this faster by using termdag directly.
    """
    expr = cast("RuntimeExpr", e).__egg_typed_expr__.expr
    if not isinstance(expr, LitDecl):
        msg = "Expected a literal"
        raise BuiltinEvalError(msg)
    return expr.value


def _extract_call(e: BaseExpr) -> CallDecl:
    """
    Extracts the call form of an expression
    """
    expr = cast("RuntimeExpr", e).__egg_typed_expr__.expr
    if not isinstance(expr, CallDecl):
        msg = "Expected a call expression"
        raise BuiltinEvalError(msg)
    return expr
