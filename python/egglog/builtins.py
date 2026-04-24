# mypy: disable-error-code="empty-body"
"""
Builtin sorts and function to egg.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial, reduce
from inspect import signature
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypeVar, cast, overload

import cloudpickle
from typing_extensions import TypeVarTuple, Unpack, deprecated

from .conversion import convert, converter, get_type_args, resolve_literal
from .declarations import *
from .deconstruct import get_callable_args, get_literal_value
from .egraph import (
    BaseExpr,
    BuiltinExpr,
    expr_fact,
    function,
    method,
)
from .runtime import RuntimeExpr, RuntimeFunction, resolve_type_annotation_mutate
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
    "Container",
    "ExprValueError",
    "Map",
    "MapLike",
    "Maybe",
    "MultiSet",
    "MultiSetLike",
    "Pair",
    "Primitive",
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
    "catch",
    "collapse_floats_with_tol",
    "f64",
    "f64Like",
    "i64",
    "i64Like",
    "join",
    "map_bigrat_intersect_min",
    "map_bigrat_subtract",
    "map_contains_key_swapped",
    "map_divide_all_values_by_f64",
    "map_drop_zero_values",
    "map_filter_defined_kv",
    "map_filter_kv",
    "map_fold_kv",
    "map_intersect_with",
    "map_keys",
    "map_map_values",
    "map_merge_with",
    "map_merge_with_swapped",
    "map_nonconst_nonunit_f64_values",
    "map_not_contains_key_swapped",
    "map_remove_keys",
    "map_restrict_keys",
    "map_shared_factor_atoms",
    "map_subtract_bigrat_from_keys",
    "maybe_f64_merge_with_tol",
    "multiset_contains_swapped",
    "multiset_flat_map",
    "multiset_fold",
    "multiset_not_contains_swapped",
    "multiset_remove_swapped",
    "multiset_subtract_swapped",
    "multiset_union_values",
    "present",
    "py_eval",
    "py_eval_fn",
    "py_exec",
    "set_union_values",
]


@dataclass
class ExprValueError(AttributeError):
    """
    Raised when an expression cannot be converted to a Python value because the value is not a constructor.
    """

    expr: BaseExpr
    allowed: str

    def __str__(self) -> str:
        return f"Cannot get Python value of {self.expr}, must be of form {self.allowed}. Try calling `extract` on it to get the underlying value."


class Unit(BuiltinExpr, egg_sort="Unit"):
    """
    The unit type. This is used to represent if a value exists in the e-graph or not.
    """

    def __init__(self) -> None: ...

    @method(preserve=True)
    def __bool__(self) -> bool:
        return bool(expr_fact(self))


class String(BuiltinExpr, egg_sort="String"):
    def __init__(self, value: str) -> None: ...

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> str:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> str:
        if (value := get_literal_value(self)) is not None:
            return value
        raise ExprValueError(self, "String")

    __match_args__ = ("value",)

    @method(egg_fn="replace")
    def replace(self, old: StringLike, new: StringLike) -> String: ...

    @method(preserve=True)
    def __add__(self, other: StringLike) -> String:
        return join(self, other)

    @method(egg_fn="log")
    def log(self) -> Unit: ...


StringLike: TypeAlias = String | str


@function(egg_fn="+", builtin=True)
def join(*strings: StringLike) -> String: ...


converter(str, String, String)


class Bool(BuiltinExpr, egg_sort="bool"):
    def __init__(self, value: bool) -> None: ...

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> bool:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> bool:
        if (value := get_literal_value(self)) is not None:
            return value
        raise ExprValueError(self, "Bool")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __bool__(self) -> bool:
        return self.value

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


class i64(BuiltinExpr, egg_sort="i64"):  # noqa: N801
    def __init__(self, value: int) -> None: ...

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> int:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> int:
        if (value := get_literal_value(self)) is not None:
            return value
        raise ExprValueError(self, "i64")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __index__(self) -> int:
        return self.value

    @method(preserve=True)
    def __int__(self) -> int:
        return self.value

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
    def __lt__(self, other: i64Like) -> Unit:  # type: ignore[has-type]
        ...

    @method(egg_fn=">")
    def __gt__(self, other: i64Like) -> Unit: ...

    @method(egg_fn="<=")
    def __le__(self, other: i64Like) -> Unit:  # type: ignore[has-type]
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

    @method(egg_fn="abs")
    def __abs__(self) -> i64: ...

    @method(egg_fn="vec-range")
    def range(self) -> Vec[i64]: ...


# The types which can be converted into an i64
i64Like: TypeAlias = i64 | int  # noqa: N816, PYI042

converter(int, i64, i64)


@function(builtin=True, egg_fn="count-matches")
def count_matches(s: StringLike, pattern: StringLike) -> i64: ...


class f64(BuiltinExpr, egg_sort="f64"):  # noqa: N801
    def __init__(self, value: float) -> None: ...

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> float:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> float:
        if (value := get_literal_value(self)) is not None:
            return value
        raise ExprValueError(self, "f64")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __float__(self) -> float:
        return self.value

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

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

    @method(egg_fn="abs")
    def __abs__(self) -> f64: ...

    @method(egg_fn="exp")
    def exp(self) -> f64: ...

    @method(egg_fn="log")
    def log(self) -> f64: ...

    @method(egg_fn="sqrt")
    def sqrt(self) -> f64: ...

    @method(egg_fn="<")
    def __lt__(self, other: f64Like) -> Unit:  # type: ignore[has-type]
        ...

    @method(egg_fn=">")
    def __gt__(self, other: f64Like) -> Unit: ...

    @method(egg_fn="<=")
    def __le__(self, other: f64Like) -> Unit:  # type: ignore[has-type]
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

    @method(egg_fn="bigrat-pow-const-value")
    def pow_bigrat(self, exponent: BigRat) -> f64: ...


f64Like: TypeAlias = f64 | float  # noqa: N816, PYI042

converter(int, f64, lambda i: f64(float(i)))
converter(float, f64, f64)


T = TypeVar("T", bound=BaseExpr)
V = TypeVar("V", bound=BaseExpr)


class Maybe(BuiltinExpr, Generic[T], egg_sort="Maybe"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> T | None:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> T | None:
        if get_callable_args(self, Maybe.none) is not None:
            return None
        match get_callable_args(self, Maybe.some):
            case (value,):
                return value  # type: ignore[has-type]
        raise ExprValueError(self, "Maybe.none() or Maybe.some(value)")

    __match_args__ = ("value",)

    @method(egg_fn="maybe-none")
    @classmethod
    def none(cls) -> Maybe[T]: ...

    @method(egg_fn="maybe-some")
    @classmethod
    def some(cls, value: T) -> Maybe[T]: ...

    @method(egg_fn="maybe-unwrap")
    def unwrap(self) -> T: ...

    @method(egg_fn="maybe-unwrap-or")
    def unwrap_or(self, default: T) -> T: ...

    @method(egg_fn="unstable-maybe-match")
    def match(self, f: Callable[[T], V], n: V) -> V: ...

    # @method(egg_fn="TODO-unstable-maybe-map")
    # def map(self, f: Callable[[T], V]) -> Maybe[V]: ...

    # @method(egg_fn="TODO-unstable-maybe-flat-map")
    # def flat_map(self, f: Callable[[T], Maybe[V]]) -> Maybe[V]: ...



converter(type(None), Maybe, lambda _: Maybe[get_type_args()[0]].none())
# converter(object, Maybe, lambda x: Maybe[get_type_args()[0]].some(convert(x, get_type_args()[0])))


@function(egg_fn="maybe-f64-merge-with-tol", builtin=True)
def maybe_f64_merge_with_tol(old: Maybe[f64], new: Maybe[f64], tol: f64Like) -> Maybe[f64]: ...


@function(egg_fn="collapse-floats-with-tol", builtin=True)
def collapse_floats_with_tol(old: f64Like, new: f64Like, tol: f64Like) -> f64: ...


L = TypeVar("L", bound=BaseExpr)
R = TypeVar("R", bound=BaseExpr)
L2 = TypeVar("L2", bound=BaseExpr)
R2 = TypeVar("R2", bound=BaseExpr)


class Pair(BuiltinExpr, Generic[L, R], egg_sort="Pair"):
    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> tuple[L, R]:
        match get_callable_args(self, Pair[L, R]):
            case (left, right):
                return (left, right)
        raise ExprValueError(self, "Pair(left, right)")

    __match_args__ = ("value",)

    @method(egg_fn="pair")
    def __init__(self, left: L, right: R) -> None: ...

    @method(egg_fn="pair-left")  # type: ignore[prop-decorator]
    @property
    def left(self) -> L: ...

    @method(egg_fn="pair-right")  # type: ignore[prop-decorator]
    @property
    def right(self) -> R: ...

    @method(egg_fn="unstable-pair-match")
    def match(self, f: Callable[[L, R], V]) -> V: ...

    @method(egg_fn="unstable-pair-map-left")
    def map_left(self, f: Callable[[L], L2]) -> Pair[L2, R]: ...

    @method(egg_fn="unstable-pair-map-right")
    def map_right(self, f: Callable[[R], R2]) -> Pair[L, R2]: ...


converter(tuple, Pair, lambda t: Pair(convert(t[0], get_type_args()[0]), convert(t[1], get_type_args()[1])))


@function(egg_fn="present", builtin=True)
def present(value: T) -> Unit: ...


@function(egg_fn="unstable-catch", builtin=True)
def catch(f: Callable[[], T]) -> Maybe[T]: ...


class Map(BuiltinExpr, Generic[T, V], egg_sort="Map"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> dict[T, V]:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> dict[T, V]:
        items = []
        while args := get_callable_args(self, Map.insert):  # type: ignore[var-annotated]
            self, k, v = args  # noqa: PLW0642
            items.append((k, v))
        if get_callable_args(self, Map.empty) is None:
            raise ExprValueError(self, "Map.empty or Map.insert")
        d = {}
        for k, v in reversed(items):
            d[k] = v
        return d

    __match_args__ = ("value",)

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.value

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

    @method(egg_fn="map-length")
    def length(self) -> i64: ...

    @method(egg_fn="map-pick-key")
    def pick_key(self) -> T: ...

    @method(egg_fn="map-keys")
    def keys(self) -> MultiSet[T]: ...

    @method(egg_fn="rebuild")
    def rebuild(self) -> Map[T, V]: ...


TO = TypeVar("TO")
VO = TypeVar("VO")
A = TypeVar("A")
V2 = TypeVar("V2")

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


@function(egg_fn="map-fold-kv", builtin=True)
def map_fold_kv(f: Callable[[A, T, V], A], initial: A, xs: Map[T, V]) -> A: ...


@function(egg_fn="map-keys", builtin=True)
def map_keys(xs: Map[T, V]) -> MultiSet[T]: ...


@function(egg_fn="map-filter-kv", builtin=True)
def map_filter_kv(f: Callable[[T, V], Unit], xs: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-filter-defined-kv", builtin=True)
def map_filter_defined_kv(f: Callable[[T, V], A], xs: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-map-values", builtin=True)
def map_map_values(f: Callable[[T, V], V2], xs: Map[T, V]) -> Map[T, V2]: ...


@function(egg_fn="map-merge-with", builtin=True)
def map_merge_with(f: Callable[[V, V], V], left: Map[T, V], right: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-merge-with-swapped", builtin=True)
def map_merge_with_swapped(f: Callable[[V, V], V], right: Map[T, V], left: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-intersect-with", builtin=True)
def map_intersect_with(f: Callable[[V, V], V], left: Map[T, V], right: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-drop-zero-values", builtin=True)
def map_drop_zero_values(xs: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-bigrat-subtract", builtin=True)
def map_bigrat_subtract(right: Map[T, BigRat], left: Map[T, BigRat]) -> Map[T, BigRat]: ...


@function(egg_fn="map-bigrat-intersect-min", builtin=True)
def map_bigrat_intersect_min(left: Map[T, BigRat], right: Map[T, BigRat]) -> Map[T, BigRat]: ...


@function(egg_fn="map-contains-key-swapped", builtin=True)
def map_contains_key_swapped(x: T, xs: Map[T, V]) -> Unit: ...


@function(egg_fn="map-not-contains-key-swapped", builtin=True)
def map_not_contains_key_swapped(x: T, xs: Map[T, V]) -> Unit: ...


@function(egg_fn="map-restrict-keys", builtin=True)
def map_restrict_keys(keys: MultiSet[T], xs: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-remove-keys", builtin=True)
def map_remove_keys(keys: MultiSet[T], xs: Map[T, V]) -> Map[T, V]: ...


@function(egg_fn="map-subtract-bigrat-from-keys", builtin=True)
def map_subtract_bigrat_from_keys(factor: Map[T, BigRat], xs: Map[Map[T, BigRat], V]) -> Map[Map[T, BigRat], V]: ...


@function(egg_fn="map-nonconst-nonunit-f64-values", builtin=True)
def map_nonconst_nonunit_f64_values(xs: Map[Map[T, V], f64]) -> MultiSet[f64]: ...


@function(egg_fn="map-divide-all-values-by-f64", builtin=True)
def map_divide_all_values_by_f64(factor: f64, xs: Map[T, f64]) -> Map[T, f64]: ...


@function(egg_fn="map-shared-factor-atoms", builtin=True)
def map_shared_factor_atoms(xs: Map[Map[T, BigRat], V]) -> Set[T]: ...


class Set(BuiltinExpr, Generic[T], egg_sort="Set"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> set[T]:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> set[T]:
        if (args := get_callable_args(self, Set[T])) is not None:
            return set(args)
        raise ExprValueError(self, "Set(*xs)")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        if (args := get_callable_args(self, Set[T])) is not None:
            return iter(args)
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.value

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

    @method(egg_fn="set-length")
    def length(self) -> i64: ...

    @method(egg_fn="unstable-set-map", reverse_args=True)
    def map(self, f: Callable[[T], V]) -> Set[V]: ...

    @method(egg_fn="rebuild")
    def rebuild(self) -> Set[T]: ...


converter(
    set,
    Set,
    lambda t: Set(*(convert(x, get_type_args()[0]) for x in t)) if t else Set[get_type_args()[0]].empty(),  # type: ignore[misc]
)

SetLike: TypeAlias = Set[T] | set[TO]


class MultiSet(BuiltinExpr, Generic[T], egg_sort="MultiSet"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> list[T]:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> list[T]:
        if (args := get_callable_args(self, MultiSet[T])) is not None:
            return list(args)
        raise ExprValueError(self, "MultiSet")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.value

    @method(egg_fn="multiset-of")
    def __init__(self, *args: T) -> None: ...

    @method(egg_fn="multiset-intersection")
    def __and__(self, other: MultiSet[T]) -> MultiSet[T]: ...

    @method(egg_fn="multiset-single")
    @classmethod
    def single(cls, x: T, i: i64Like) -> MultiSet[T]: ...

    @method(egg_fn="multiset-sum-multisets")
    @classmethod
    def sum_multisets(cls, xs: MultiSet[MultiSet[T]]) -> MultiSet[T]: ...

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

    @method(egg_fn="multiset-subtract")
    def __sub__(self, other: MultiSet[T]) -> MultiSet[T]: ...

    @method(egg_fn="unstable-multiset-map", reverse_args=True)
    def map(self, f: Callable[[T], V]) -> MultiSet[V]: ...

    @method(egg_fn="unstable-multiset-fill-index")
    def fill_index(self, f: Callable[[MultiSet[T], T], i64]) -> Unit: ...

    @method(egg_fn="unstable-multiset-clear-index")
    def clear_index(self, f: Callable[[MultiSet[T], T], i64]) -> Unit: ...

    @method(egg_fn="multiset-pick-max")
    def pick_max(self) -> T: ...

    @method(egg_fn="multiset-count")
    def count(self, value: T) -> i64: ...

    @method(egg_fn="unstable-multiset-filter", reverse_args=True)
    def filter(self, f: Callable[[T], Unit]) -> MultiSet[T]: ...

    @method(egg_fn="unstable-multiset-filter-not", reverse_args=True)
    def filter_not(self, f: Callable[[T], Unit]) -> MultiSet[T]: ...

    @method(egg_fn="multiset-reset-counts")
    def reset_counts(self) -> MultiSet[T]: ...


# TODO: Move to method when partial supports reverse_args
@function(egg_fn="unstable-multiset-flat-map", builtin=True)
def multiset_flat_map(f: Callable[[T], MultiSet[T]], xs: MultiSet[T]) -> MultiSet[T]: ...


@function(egg_fn="multiset-remove-swapped", builtin=True)
def multiset_remove_swapped(x: T, xs: MultiSet[T]) -> MultiSet[T]: ...


@function(egg_fn="multiset-subtract-swapped", builtin=True)
def multiset_subtract_swapped(x: MultiSet[T], xs: MultiSet[T]) -> MultiSet[T]: ...


@function(egg_fn="multiset-not-contains-swapped", builtin=True)
def multiset_not_contains_swapped(x: T, xs: MultiSet[T]) -> Unit: ...


@function(egg_fn="multiset-contains-swapped", builtin=True)
def multiset_contains_swapped(x: T, xs: MultiSet[T]) -> Unit: ...


@function(egg_fn="unstable-multiset-reduce", builtin=True)
def multiset_fold(f: Callable[[T, T], T], initial: T, xs: MultiSet[T]) -> T: ...


@function(egg_fn="multiset-union-values", builtin=True)
def multiset_union_values(xs: MultiSet[T]) -> T: ...


@function(egg_fn="set-union-values", builtin=True)
def set_union_values(xs: Set[T]) -> T: ...


converter(
    tuple,
    MultiSet,
    lambda t: MultiSet(*(convert(x, get_type_args()[0]) for x in t)) if t else MultiSet[get_type_args()[0]](),  # type: ignore[operator,misc]
)

MultiSetLike: TypeAlias = MultiSet[T] | tuple[TO, ...]


class Rational(BuiltinExpr, egg_sort="Rational"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> Fraction:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> Fraction:
        match get_callable_args(self, Rational):
            case (i64(num), i64(den)):
                return Fraction(num, den)
        raise ExprValueError(self, "Rational(i64(num), i64(den))")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.value)

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

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

    @method(egg_fn="numer")  # type: ignore[prop-decorator]
    @property
    def numer(self) -> i64: ...

    @method(egg_fn="denom")  # type: ignore[prop-decorator]
    @property
    def denom(self) -> i64: ...


class BigInt(BuiltinExpr, egg_sort="BigInt"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> int:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> int:
        match get_callable_args(self, BigInt.from_string):
            case (String(s),):
                return int(s)
        raise ExprValueError(self, "BigInt.from_string(String(s))")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __index__(self) -> int:
        return self.value

    @method(preserve=True)
    def __int__(self) -> int:
        return self.value

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
    def __lt__(self, other: BigIntLike) -> Unit:  # type: ignore[has-type]
        ...

    @method(egg_fn=">")
    def __gt__(self, other: BigIntLike) -> Unit: ...

    @method(egg_fn="<=")
    def __le__(self, other: BigIntLike) -> Unit:  # type: ignore[has-type]
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


converter(i64, BigInt, BigInt)

BigIntLike: TypeAlias = BigInt | i64Like


class BigRat(BuiltinExpr, egg_sort="BigRat"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> Fraction:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> Fraction:
        match get_callable_args(self, BigRat):
            case (BigInt(num), BigInt(den)):
                return Fraction(num, den)
        raise ExprValueError(self, "BigRat(BigInt(num), BigInt(den))")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.value)

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

    @method(egg_fn="bigrat")
    def __init__(self, num: BigIntLike, den: BigIntLike) -> None: ...

    @method(egg_fn="to-f64")
    def to_f64(self) -> f64: ...

    @method(egg_fn="from-f64")
    @classmethod
    def from_f64(cls, f: f64Like) -> BigRat: ...

    @method(egg_fn="to-i64")
    def to_i64(self) -> i64: ...

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

    @method(egg_fn="numer")  # type: ignore[prop-decorator]
    @property
    def numer(self) -> BigInt: ...

    @method(egg_fn="denom")  # type: ignore[prop-decorator]
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


converter(i64, BigRat, lambda i: BigRat(BigInt(i), BigInt(1)))
converter(Fraction, BigRat, lambda f: BigRat(f.numerator, f.denominator))
BigRatLike: TypeAlias = BigRat | Fraction | i64Like


class Vec(BuiltinExpr, Generic[T], egg_sort="Vec"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> tuple[T, ...]:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> tuple[T, ...]:
        if get_callable_args(self, Vec.empty) is not None:
            return ()
        if (args := get_callable_args(self, Vec[T])) is not None:
            return args
        raise ExprValueError(self, "Vec(*xs) or Vec.empty()")

    __match_args__ = ("value",)

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, key: T) -> bool:
        return key in self.value

    @method(egg_fn="vec-of")
    def __init__(self, *args: T) -> None: ...

    @method(egg_fn="vec-empty")
    @classmethod
    def empty(cls) -> Vec[T]: ...

    @method(egg_fn="vec-append")
    def append(self, *others: VecLike[T, T]) -> Vec[T]: ...

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

    @method(egg_fn="vec-union")
    def __or__(self, other: Vec[T]) -> Vec[T]: ...

    @method(egg_fn="unstable-vec-map", reverse_args=True)
    def map(self, fn: Callable[[T], V]) -> Vec[V]: ...


for sequence_type in (list, tuple):
    converter(
        sequence_type,
        Vec,
        lambda t: Vec(*(convert(x, get_type_args()[0]) for x in t)) if t else Vec[get_type_args()[0]].empty(),  # type: ignore[misc]
    )

VecLike: TypeAlias = Vec[T] | tuple[TO, ...] | list[TO]


TS = TypeVarTuple("TS")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class UnstableFn(BuiltinExpr, Generic[T, *TS], egg_sort="UnstableFn"):
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

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> Callable[[Unpack[TS]], T]:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> Callable[[Unpack[TS]], T]:
        """
        If this is a constructor, returns either the callable directly or a `functools.partial` function if args are provided.
        """
        if (fn := get_literal_value(self)) is not None:
            return fn
        raise ExprValueError(self, "UnstableFn(f, *args)")

    __match_args__ = ("value",)

    @method(egg_fn="unstable-app")
    def __call__(self, *args: *TS) -> T: ...


# Method Type is for builtins like __getitem__
converter(MethodType, UnstableFn, lambda m: UnstableFn[*get_type_args()](m.__func__, m.__self__))  # type: ignore[operator, misc]
# Ignore PLW0108.
converter(RuntimeFunction, UnstableFn, lambda rf: UnstableFn[*get_type_args()](rf))  # type: ignore[operator, misc]
# converter(RuntimeClass, UnstableFn, lambda rc: UnstableFn[*get_type_args()](rc))  # type: ignore[operator, misc]
converter(partial, UnstableFn, lambda p: UnstableFn[*get_type_args()](p.func, *p.args))  # type: ignore[operator, misc]


def _convert_function(fn: FunctionType) -> UnstableFn:
    """
    Converts a function type to an unstable function. This function will be an anon function in egglog.

    Would just be UnstableFn(function(a)) but we have to account for unbound vars within the body.

    This means that we have to turn all of those unbound vars into args to the function, and then
    partially apply them, alongside storing the eager primitive body for the function.
    """
    decls = Declarations()
    return_type, *arg_types = [resolve_type_annotation_mutate(decls, tp) for tp in get_type_args()]
    arg_names = [p.name for p in signature(fn).parameters.values()]
    arg_decls = [
        TypedExprDecl(tp.to_just(), UnboundVarDecl(name)) for name, tp in zip(arg_names, arg_types, strict=True)
    ]
    res = resolve_literal(
        return_type, fn(*(RuntimeExpr.__from_values__(decls, a) for a in arg_decls)), Thunk.value(decls)
    )
    res_expr = res.__egg_typed_expr__
    decls |= res
    # these are all the args that appear in the body that are not bound by the args of the function
    unbound_vars = list(collect_unbound_vars(res_expr) - set(arg_decls))
    # prefix the args with them
    all_args = tuple(unbound_vars + arg_decls)
    normalized_args = tuple(
        TypedExprDecl(
            typed_arg.tp,
            UnboundVarDecl(cast("UnboundVarDecl", typed_arg.expr).name, f"_{i}"),
        )
        for i, typed_arg in enumerate(all_args)
    )
    res_expr = replace_typed_expr(res_expr, dict(zip(all_args, normalized_args, strict=True)))
    fn_ref = UnnamedFunctionRef(normalized_args, res_expr)
    fn = RuntimeFunction(Thunk.value(decls), Thunk.value(fn_ref))
    return UnstableFn(fn, *(RuntimeExpr.__from_values__(decls, v) for v in unbound_vars))


converter(FunctionType, UnstableFn, _convert_function)


class PyObject(BuiltinExpr, egg_sort="PyObject"):
    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> object:
        return self.value

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> object:
        expr = cast("RuntimeExpr", self).__egg_typed_expr__.expr
        if not isinstance(expr, PyObjectDecl):
            raise ExprValueError(self, "PyObject(x)")
        return cloudpickle.loads(expr.pickled)

    __match_args__ = ("value",)

    def __init__(self, value: object) -> None: ...

    @method(egg_fn="py-call")
    def __call__(self, *args: object) -> PyObject: ...

    @method(egg_fn="py-call-extended")
    def call_extended(self, args: PyObject, kwargs: PyObject) -> PyObject:
        """
        Call the PyObject with the given args and kwargs PyObjects.
        """

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
def py_eval(code: StringLike, globals_: object = PyObject.dict(), locals_: object = PyObject.dict()) -> PyObject: ...


class PyObjectFunction(Protocol):
    def __call__(self, *__args: PyObject) -> PyObject: ...


@deprecated("use PyObject(fn) directly")
def py_eval_fn(fn: Callable) -> PyObjectFunction:
    """
    Takes a python callable and maps it to a callable which takes and returns PyObjects.

    It translates it to a call which uses `py_eval` to call the function, passing in the
    args as locals, and using the globals from function.
    """
    return PyObject(fn)


@function(builtin=True, egg_fn="py-exec")
def py_exec(code: StringLike, globals_: object = PyObject.dict(), locals_: object = PyObject.dict()) -> PyObject:
    """
    Copies the locals, execs the Python code, and returns the locals with any updates.
    """


Container: TypeAlias = Map | Set | MultiSet | Vec | UnstableFn
Primitive: TypeAlias = String | Bool | i64 | f64 | Rational | BigInt | BigRat | PyObject | Unit
