"""Builtin Python expressions, validated against the native catalog."""
# mypy: disable-error-code="empty-body"
# ruff: noqa: N801, N816, PYI042 -- Canonical builtin names and operator stubs.

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from functools import partial
from types import FunctionType, MethodType
from typing import Generic, Never, Protocol, TypeAlias, TypeVar, TypeVarTuple, Unpack, overload

from typing_extensions import deprecated

from ._builtin_values import (
    ExprValueError,
    _collection_to_expr,
    _container_items,
    _dict_to_map,
    _function_to_expr,
    _map_value,
    _maybe_value,
    _none_to_maybe,
    _pair_value,
    _tuple_to_pair,
)
from ._catalog import builtin_catalog
from .conversion import convert, converter
from .egraph import BaseExpr, BuiltinExpr, expr_fact, function, method
from .runtime import RuntimeExpr, RuntimeFunction, _prepare_declarations

__all__ = [
    "BigInt",
    "BigIntLike",
    "BigRat",
    "BigRatLike",
    "Bool",
    "BoolLike",
    "ExprValueError",
    "Map",
    "MapLike",
    "Maybe",
    "MultiSet",
    "MultiSetLike",
    "Pair",
    "PyObject",
    "PyObjectLike",
    "Rational",
    "RationalLike",
    "Set",
    "SetLike",
    "String",
    "StringLike",
    "Unit",
    "UnstableFn",
    "Vec",
    "VecLike",
    "catch",
    "count_matches",
    "f64",
    "f64Like",
    "i64",
    "i64Like",
    "join",
    "map_fold_kv",
    "multiset_contains_swapped",
    "multiset_flat_map",
    "multiset_fold",
    "multiset_not_contains_swapped",
    "multiset_remove_swapped",
    "multiset_subtract_swapped",
    "py_eval",
    "py_eval_fn",
    "py_exec",
]


T = TypeVar("T", bound=BaseExpr)
U = TypeVar("U", bound=BaseExpr)
K = TypeVar("K", bound=BaseExpr)
V = TypeVar("V", bound=BaseExpr)
L = TypeVar("L", bound=BaseExpr)
R = TypeVar("R", bound=BaseExpr)
TS = TypeVarTuple("TS")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
H = TypeVar("H")
KH = TypeVar("KH")
VH = TypeVar("VH")


class UnstableFn(BuiltinExpr, Generic[T, *TS], egg_sort="UnstableFn"):
    __match_args__ = ("value",)

    @overload
    def __init__(self, f: Callable[[Unpack[TS]], T]) -> None: ...
    @overload
    def __init__(self, f: Callable[[T1, Unpack[TS]], T], _a: T1, /) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, Unpack[TS]], T],
        _a: T1,
        _b: T2,
        /,
    ) -> None: ...
    def __init__(self, f: Callable[..., BaseExpr], *partial: object) -> None: ...
    def __call__(self, *args: *TS) -> T: ...
    @property
    @method(preserve=True)
    def value(self) -> Callable[[Unpack[TS]], T]:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "UnstableFn(f, *args)")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> Callable[[Unpack[TS]], T]:
        return self.value


class Bool(BuiltinExpr, egg_sort="bool"):
    def __init__(self, value: bool, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> bool:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "Bool")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> bool:
        return self.value

    @method(preserve=True)
    def __bool__(self) -> bool:
        return bool(self.value)

    @method(egg_fn="egglog.core.bool.and")
    def __and__(self, other: BoolLike) -> Bool: ...
    @method(preserve=True)
    def __rand__(self, other: BoolLike) -> Bool:
        return convert(other, Bool).__and__(self)

    @method(egg_fn="egglog.core.bool.implies")
    def implies(self, other: BoolLike) -> Bool: ...
    @method(egg_fn="egglog.core.bool.not")
    def __invert__(self) -> Bool: ...
    @method(egg_fn="egglog.core.bool.or")
    def __or__(self, other: BoolLike) -> Bool: ...
    @method(preserve=True)
    def __ror__(self, other: BoolLike) -> Bool:
        return convert(other, Bool).__or__(self)

    @method(egg_fn="egglog.core.bool.xor")
    def __xor__(self, other: BoolLike) -> Bool: ...
    @method(preserve=True)
    def __rxor__(self, other: BoolLike) -> Bool:
        return convert(other, Bool).__xor__(self)


class i64(BuiltinExpr, egg_sort="i64"):
    def __init__(self, value: int, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> int:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "i64")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> int:
        return self.value

    @method(preserve=True)
    def __index__(self) -> int:
        return int(self.value)

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

    @method(egg_fn="egglog.core.i64.abs")
    def __abs__(self) -> i64: ...
    @method(egg_fn="egglog.core.i64.add")
    def __add__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __radd__(self, other: i64Like) -> i64:
        return convert(other, i64).__add__(self)

    @method(egg_fn="egglog.core.i64.bitand")
    def __and__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rand__(self, other: i64Like) -> i64:
        return convert(other, i64).__and__(self)

    @method(egg_fn="egglog.core.i64.bitor")
    def __or__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __ror__(self, other: i64Like) -> i64:
        return convert(other, i64).__or__(self)

    @method(egg_fn="egglog.core.i64.bitxor")
    def __xor__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rxor__(self, other: i64Like) -> i64:
        return convert(other, i64).__xor__(self)

    @method(egg_fn="egglog.core.i64.bool_ge")
    def bool_ge(self, other: i64Like) -> Bool: ...
    @method(egg_fn="egglog.core.i64.bool_gt")
    def bool_gt(self, other: i64Like) -> Bool: ...
    @method(egg_fn="egglog.core.i64.bool_le")
    def bool_le(self, other: i64Like) -> Bool: ...
    @method(egg_fn="egglog.core.i64.bool_lt")
    def bool_lt(self, other: i64Like) -> Bool: ...
    @method(egg_fn="egglog.core.i64.div")
    def __truediv__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rtruediv__(self, other: i64Like) -> i64:
        return convert(other, i64).__truediv__(self)

    def __ge__(self, other: i64Like) -> Unit: ...
    def __gt__(self, other: i64Like) -> Unit: ...
    def __le__(self, other: i64Like) -> Unit: ...
    @method(egg_fn="egglog.core.i64.log2")
    def log2(self) -> i64: ...
    def __lt__(self, other: i64Like) -> Unit: ...
    @method(egg_fn="egglog.core.i64.max")
    def max(self, other: i64Like) -> i64: ...
    @method(egg_fn="egglog.core.i64.min")
    def min(self, other: i64Like) -> i64: ...
    @method(egg_fn="egglog.core.i64.mul")
    def __mul__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rmul__(self, other: i64Like) -> i64:
        return convert(other, i64).__mul__(self)

    @method(egg_fn="egglog.core.i64.not")
    def __invert__(self) -> i64: ...
    @method(egg_fn="egglog.core.i64.rem")
    def __mod__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rmod__(self, other: i64Like) -> i64:
        return convert(other, i64).__mod__(self)

    @method(egg_fn="egglog.core.i64.shl")
    def __lshift__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rlshift__(self, other: i64Like) -> i64:
        return convert(other, i64).__lshift__(self)

    @method(egg_fn="egglog.core.i64.shr")
    def __rshift__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rrshift__(self, other: i64Like) -> i64:
        return convert(other, i64).__rshift__(self)

    @method(egg_fn="egglog.core.i64.sub")
    def __sub__(self, other: i64Like) -> i64: ...
    @method(preserve=True)
    def __rsub__(self, other: i64Like) -> i64:
        return convert(other, i64).__sub__(self)

    @method(egg_fn="egglog.core.i64.to_string")
    def to_string(self) -> String: ...
    @method(egg_fn="egglog.core.vec.range")
    def range(self) -> Vec[i64]: ...

    __lt__ = method(egg_fn="egglog.core.i64.lt")(__lt__)
    __gt__ = method(egg_fn="egglog.core.i64.gt")(__gt__)
    __le__ = method(egg_fn="egglog.core.i64.le")(__le__)
    __ge__ = method(egg_fn="egglog.core.i64.ge")(__ge__)


class String(BuiltinExpr, egg_sort="String"):
    def __init__(self, value: str, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> str:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "String")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> str:
        return self.value

    @method(egg_fn="egglog.core.string.concat")
    def __add__(self, other: StringLike) -> String: ...
    @method(preserve=True)
    def __radd__(self, other: StringLike) -> String:
        return convert(other, String).__add__(self)

    @method(egg_fn="egglog.core.string.replace")
    def replace(self, old: StringLike, new: StringLike) -> String: ...


class Unit(BuiltinExpr, egg_sort="Unit"):
    def __init__(self) -> None: ...
    @method(preserve=True)
    def __bool__(self) -> bool:
        return bool(expr_fact(self))


class f64(BuiltinExpr, egg_sort="f64"):
    def __init__(self, value: float, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> float:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "f64")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> float:
        return self.value

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.value)

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

    @method(egg_fn="egglog.core.f64.abs")
    def __abs__(self) -> f64: ...
    @method(egg_fn="egglog.core.f64.add")
    def __add__(self, other: f64Like) -> f64: ...
    @method(preserve=True)
    def __radd__(self, other: f64Like) -> f64:
        return convert(other, f64).__add__(self)

    @method(egg_fn="egglog.core.f64.div")
    def __truediv__(self, other: f64Like) -> f64: ...
    @method(preserve=True)
    def __rtruediv__(self, other: f64Like) -> f64:
        return convert(other, f64).__truediv__(self)

    @method(egg_fn="egglog.core.f64.exp")
    def exp(self) -> f64: ...
    @method(egg_fn="egglog.core.f64.from_i64")
    @classmethod
    def from_i64(cls, i: i64Like) -> f64: ...
    def __ge__(self, other: f64Like) -> Unit: ...
    def __gt__(self, other: f64Like) -> Unit: ...
    def __le__(self, other: f64Like) -> Unit: ...
    @method(egg_fn="egglog.core.f64.log")
    def log(self) -> f64: ...
    def __lt__(self, other: f64Like) -> Unit: ...
    @method(egg_fn="egglog.core.f64.max")
    def max(self, other: f64Like) -> f64: ...
    @method(egg_fn="egglog.core.f64.min")
    def min(self, other: f64Like) -> f64: ...
    @method(egg_fn="egglog.core.f64.mul")
    def __mul__(self, other: f64Like) -> f64: ...
    @method(preserve=True)
    def __rmul__(self, other: f64Like) -> f64:
        return convert(other, f64).__mul__(self)

    @method(egg_fn="egglog.core.f64.neg")
    def __neg__(self) -> f64: ...
    @method(egg_fn="egglog.core.f64.pow")
    def __pow__(self, other: f64Like) -> f64: ...
    @method(preserve=True)
    def __rpow__(self, other: f64Like) -> f64:
        return convert(other, f64).__pow__(self)

    @method(egg_fn="egglog.core.f64.rem")
    def __mod__(self, other: f64Like) -> f64: ...
    @method(preserve=True)
    def __rmod__(self, other: f64Like) -> f64:
        return convert(other, f64).__mod__(self)

    @method(egg_fn="egglog.core.f64.sqrt")
    def sqrt(self) -> f64: ...
    @method(egg_fn="egglog.core.f64.sub")
    def __sub__(self, other: f64Like) -> f64: ...
    @method(preserve=True)
    def __rsub__(self, other: f64Like) -> f64:
        return convert(other, f64).__sub__(self)

    @method(egg_fn="egglog.core.f64.to_i64")
    def to_i64(self) -> i64: ...
    @method(egg_fn="egglog.core.f64.to_string")
    def to_string(self) -> String: ...
    @method(egg_fn="egglog.experimental.f64.is_finite")
    def is_finite(self) -> Unit: ...

    __lt__ = method(egg_fn="egglog.core.f64.lt")(__lt__)
    __gt__ = method(egg_fn="egglog.core.f64.gt")(__gt__)
    __le__ = method(egg_fn="egglog.core.f64.le")(__le__)
    __ge__ = method(egg_fn="egglog.core.f64.ge")(__ge__)


class Map(BuiltinExpr, Generic[K, V], egg_sort="Map"):
    # No Python initializer is supplied; this remains a usable type token.
    def __init__(self, _unavailable: Never, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> dict[K, V]:
        return _map_value(self)

    @method(preserve=True)
    def __iter__(self) -> Iterator[K]:
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, key: K) -> bool:
        return key in self.value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> dict[K, V]:
        return self.value

    @method(egg_fn="egglog.core.map.contains")
    def contains(self, key: K) -> Unit: ...
    @method(egg_fn="egglog.core.map.empty")
    @classmethod
    def empty(cls) -> Map[K, V]: ...
    @method(egg_fn="egglog.core.map.get")
    def __getitem__(self, key: K) -> V: ...
    @method(egg_fn="egglog.core.map.insert")
    def insert(self, key: K, value: V) -> Map[K, V]: ...
    @method(egg_fn="egglog.core.map.length")
    def length(self) -> i64: ...
    @method(egg_fn="egglog.core.map.not_contains")
    def not_contains(self, key: K) -> Unit: ...
    @method(egg_fn="egglog.core.map.of")
    @classmethod
    def from_pairs(cls, entries: Vec[Pair[K, V]]) -> Map[K, V]: ...
    @method(egg_fn="egglog.core.map.remove")
    def remove(self, key: K) -> Map[K, V]: ...


class Pair(BuiltinExpr, Generic[L, R], egg_sort="Pair"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> tuple[L, R]:
        return _pair_value(self)

    @property
    @method(egg_fn="egglog.core.pair.first")
    def left(self) -> L: ...
    @method(egg_fn="egglog.core.pair.make")
    def __init__(self, left: L, right: R) -> None: ...
    @property
    @method(egg_fn="egglog.core.pair.second")
    def right(self) -> R: ...


class Vec(BuiltinExpr, Generic[T], egg_sort="Vec"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> tuple[T, ...]:
        return _container_items(self)

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, item: T) -> bool:
        return item in self.value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> tuple[T, ...]:
        return self.value

    @method(egg_fn="egglog.core.vec.append")
    def append(self, *others: VecLike[T, T]) -> Vec[T]: ...
    @method(egg_fn="egglog.core.vec.contains")
    def contains(self, value: T) -> Unit: ...
    @method(egg_fn="egglog.core.vec.empty")
    @classmethod
    def empty(cls) -> Vec[T]: ...
    @method(egg_fn="egglog.core.vec.get")
    def __getitem__(self, index: i64Like) -> T: ...
    @method(egg_fn="egglog.core.vec.length")
    def length(self) -> i64: ...
    @method(egg_fn="egglog.core.vec.map", reverse_args=True)
    def map(self, f: Callable[[T], U]) -> Vec[U]: ...
    @method(egg_fn="egglog.core.vec.not_contains")
    def not_contains(self, value: T) -> Unit: ...
    @method(egg_fn="egglog.core.vec.of")
    def __init__(self, *args: T) -> None: ...
    @method(egg_fn="egglog.core.vec.pop")
    def pop(self) -> Vec[T]: ...
    @method(egg_fn="egglog.core.vec.push")
    def push(self, value: T) -> Vec[T]: ...
    @method(egg_fn="egglog.core.vec.remove")
    def remove(self, index: i64Like) -> Vec[T]: ...
    @method(egg_fn="egglog.core.vec.set")
    def set(self, index: i64Like, value: T) -> Vec[T]: ...
    @method(egg_fn="egglog.core.vec.union")
    def __or__(self, other: Vec[T]) -> Vec[T]: ...


class MultiSet(BuiltinExpr, Generic[T], egg_sort="MultiSet"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> list[T]:
        return list(_container_items(self))

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(self.value)

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, item: T) -> bool:
        return item in self.value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> list[T]:
        return self.value

    @method(egg_fn="egglog.core.multiset.clear_index")
    def clear_index(self, f: Callable[[MultiSet[T], T], i64]) -> Unit: ...
    @method(egg_fn="egglog.core.multiset.contains")
    def contains(self, value: T) -> Unit: ...
    @method(egg_fn="egglog.core.multiset.count")
    def count(self, value: T) -> i64: ...
    @method(egg_fn="egglog.core.multiset.fill_index")
    def fill_index(self, f: Callable[[MultiSet[T], T], i64]) -> Unit: ...
    @method(egg_fn="egglog.core.multiset.filter", reverse_args=True)
    def filter(self, f: Callable[[T], Unit]) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.filter_not", reverse_args=True)
    def filter_not(self, f: Callable[[T], Unit]) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.insert")
    def insert(self, value: T) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.intersection")
    def __and__(self, other: MultiSet[T]) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.length")
    def length(self) -> i64: ...
    @method(egg_fn="egglog.core.multiset.map", reverse_args=True)
    def map(self, f: Callable[[T], U]) -> MultiSet[U]: ...
    @method(egg_fn="egglog.core.multiset.not_contains")
    def not_contains(self, value: T) -> Unit: ...
    @method(egg_fn="egglog.core.multiset.of")
    def __init__(self, *args: T) -> None: ...
    @method(egg_fn="egglog.core.multiset.pick")
    def pick(self) -> T: ...
    @method(egg_fn="egglog.core.multiset.pick_max")
    def pick_max(self) -> T: ...
    @method(egg_fn="egglog.core.multiset.remove")
    def remove(self, value: T) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.reset_counts")
    def reset_counts(self) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.single")
    @classmethod
    def single(cls, x: T, i: i64Like) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.subtract")
    def __sub__(self, other: MultiSet[T]) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.sum")
    def __add__(self, other: MultiSet[T]) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.sum_multisets")
    @classmethod
    def sum_multisets(cls, xs: MultiSet[MultiSet[T]]) -> MultiSet[T]: ...
    @method(egg_fn="egglog.core.multiset.union_values")
    def union_values(self) -> T: ...


class Set(BuiltinExpr, Generic[T], egg_sort="Set"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> set[T]:
        return set(_container_items(self))

    @method(preserve=True)
    def __iter__(self) -> Iterator[T]:
        return iter(dict.fromkeys(_container_items(self)))

    @method(preserve=True)
    def __len__(self) -> int:
        return len(self.value)

    @method(preserve=True)
    def __contains__(self, item: T) -> bool:
        return item in self.value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> set[T]:
        return self.value

    @method(egg_fn="egglog.core.set.contains")
    def contains(self, value: T) -> Unit: ...
    @method(egg_fn="egglog.core.set.diff")
    def __sub__(self, other: Set[T]) -> Set[T]: ...
    @method(egg_fn="egglog.core.set.empty")
    @classmethod
    def empty(cls) -> Set[T]: ...
    @method(egg_fn="egglog.core.set.insert")
    def insert(self, value: T) -> Set[T]: ...
    @method(egg_fn="egglog.core.set.intersect")
    def __and__(self, other: Set[T]) -> Set[T]: ...
    @method(egg_fn="egglog.core.set.length")
    def length(self) -> i64: ...
    @method(egg_fn="egglog.core.set.not_contains")
    def not_contains(self, value: T) -> Unit: ...
    @method(egg_fn="egglog.core.set.of")
    def __init__(self, *args: T) -> None: ...
    @method(egg_fn="egglog.core.set.remove")
    def remove(self, value: T) -> Set[T]: ...
    @method(egg_fn="egglog.core.set.union")
    def __or__(self, other: Set[T]) -> Set[T]: ...


class Maybe(BuiltinExpr, Generic[T], egg_sort="Maybe"):
    # No Python initializer is supplied; this remains a usable type token.
    def __init__(self, _unavailable: Never, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> T | None:
        return _maybe_value(self)

    @method(egg_fn="egglog.experimental.maybe.match")
    def match(self, f: Callable[[T], U], n: U) -> U: ...
    @method(egg_fn="egglog.experimental.maybe.none")
    @classmethod
    def none(cls) -> Maybe[T]: ...
    @method(egg_fn="egglog.experimental.maybe.some")
    @classmethod
    def some(cls, value: T) -> Maybe[T]: ...
    @method(egg_fn="egglog.experimental.maybe.unwrap")
    def unwrap(self) -> T: ...
    @method(egg_fn="egglog.experimental.maybe.unwrap_or")
    def unwrap_or(self, default: T) -> T: ...


class PyObject(BuiltinExpr, egg_sort="PyObject"):
    def __init__(self, value: object, /) -> None: ...

    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> object:
        from .deconstruct import _pyobject_value  # noqa: PLC0415 -- materialize only on explicit inspection

        if not isinstance(self, RuntimeExpr):
            msg = "Expected a runtime expression"
            raise TypeError(msg)
        return _pyobject_value(self)

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> object:
        return self.value

    @method(egg_fn="egglog.python.pyobject.call")
    def __call__(self, *args: PyObjectLike) -> PyObject: ...
    @method(egg_fn="egglog.python.pyobject.call_extended")
    def call_extended(self, args: PyObjectLike, kwargs: PyObjectLike) -> PyObject: ...
    @method(egg_fn="egglog.python.pyobject.dict")
    @classmethod
    def dict(cls, *keys_and_values: PyObjectLike) -> PyObject: ...
    @method(egg_fn="egglog.python.pyobject.dict_update")
    def dict_update(self, *keys_and_values: PyObjectLike) -> PyObject: ...
    @method(egg_fn="egglog.python.pyobject.from_int")
    @classmethod
    def from_int(cls, i: i64Like) -> PyObject: ...
    @method(egg_fn="egglog.python.pyobject.from_string")
    @classmethod
    def from_string(cls, s: StringLike) -> PyObject: ...
    @method(egg_fn="egglog.python.pyobject.to_bool")
    def to_bool(self) -> Bool: ...
    @method(egg_fn="egglog.python.pyobject.to_string")
    def to_string(self) -> String: ...


class BigInt(BuiltinExpr, egg_sort="BigInt"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> int:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "BigInt")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> int:
        return self.value

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

    @method(preserve=True)
    def __index__(self) -> int:
        return int(self.value)

    @method(egg_fn="egglog.core.bigint.add")
    def __add__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __radd__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__add__(self)

    @method(egg_fn="egglog.core.bigint.bitand")
    def __and__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __rand__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__and__(self)

    @method(egg_fn="egglog.core.bigint.bitor")
    def __or__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __ror__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__or__(self)

    @method(egg_fn="egglog.core.bigint.bits")
    def bits(self) -> BigInt: ...
    @method(egg_fn="egglog.core.bigint.bitxor")
    def __xor__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __rxor__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__xor__(self)

    @method(egg_fn="egglog.core.bigint.bool_eq")
    def bool_eq(self, other: BigIntLike) -> Bool: ...
    @method(egg_fn="egglog.core.bigint.bool_ge")
    def bool_ge(self, other: BigIntLike) -> Bool: ...
    @method(egg_fn="egglog.core.bigint.bool_gt")
    def bool_gt(self, other: BigIntLike) -> Bool: ...
    @method(egg_fn="egglog.core.bigint.bool_le")
    def bool_le(self, other: BigIntLike) -> Bool: ...
    @method(egg_fn="egglog.core.bigint.bool_lt")
    def bool_lt(self, other: BigIntLike) -> Bool: ...
    @method(egg_fn="egglog.core.bigint.div")
    def __truediv__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __rtruediv__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__truediv__(self)

    @method(egg_fn="egglog.core.bigint.from_i64")
    def __init__(self, value: i64Like, /) -> None: ...
    @method(egg_fn="egglog.core.bigint.from_string")
    @classmethod
    def from_string(cls, s: StringLike) -> BigInt: ...
    def __ge__(self, other: BigIntLike) -> Unit: ...
    def __gt__(self, other: BigIntLike) -> Unit: ...
    def __le__(self, other: BigIntLike) -> Unit: ...
    def __lt__(self, other: BigIntLike) -> Unit: ...
    @method(egg_fn="egglog.core.bigint.max")
    def max(self, other: BigIntLike) -> BigInt: ...
    @method(egg_fn="egglog.core.bigint.min")
    def min(self, other: BigIntLike) -> BigInt: ...
    @method(egg_fn="egglog.core.bigint.mul")
    def __mul__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __rmul__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__mul__(self)

    @method(egg_fn="egglog.core.bigint.not")
    def __invert__(self) -> BigInt: ...
    @method(egg_fn="egglog.core.bigint.rem")
    def __mod__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __rmod__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__mod__(self)

    @method(egg_fn="egglog.core.bigint.shl")
    def __lshift__(self, other: i64Like) -> BigInt: ...
    @method(egg_fn="egglog.core.bigint.shr")
    def __rshift__(self, other: i64Like) -> BigInt: ...
    @method(egg_fn="egglog.core.bigint.sub")
    def __sub__(self, other: BigIntLike) -> BigInt: ...
    @method(preserve=True)
    def __rsub__(self, other: BigIntLike) -> BigInt:
        return convert(other, BigInt).__sub__(self)

    @method(egg_fn="egglog.core.bigint.to_string")
    def to_string(self) -> String: ...

    __lt__ = method(egg_fn="egglog.core.bigint.lt")(__lt__)
    __gt__ = method(egg_fn="egglog.core.bigint.gt")(__gt__)
    __le__ = method(egg_fn="egglog.core.bigint.le")(__le__)
    __ge__ = method(egg_fn="egglog.core.bigint.ge")(__ge__)


class BigRat(BuiltinExpr, egg_sort="BigRat"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> Fraction:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "BigRat")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> Fraction:
        return self.value

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.value)

    @method(egg_fn="egglog.core.bigrat.abs")
    def __abs__(self) -> BigRat: ...
    @method(egg_fn="egglog.core.bigrat.add")
    def __add__(self, other: BigRatLike) -> BigRat: ...
    @method(preserve=True)
    def __radd__(self, other: BigRatLike) -> BigRat:
        return convert(other, BigRat).__add__(self)

    @method(egg_fn="egglog.core.bigrat.cbrt")
    def cbrt(self) -> BigRat: ...
    @method(egg_fn="egglog.core.bigrat.ceil")
    def ceil(self) -> BigRat: ...
    @property
    @method(egg_fn="egglog.core.bigrat.denom")
    def denom(self) -> BigInt: ...
    @method(egg_fn="egglog.core.bigrat.div")
    def __truediv__(self, other: BigRatLike) -> BigRat: ...
    @method(preserve=True)
    def __rtruediv__(self, other: BigRatLike) -> BigRat:
        return convert(other, BigRat).__truediv__(self)

    @method(egg_fn="egglog.core.bigrat.floor")
    def floor(self) -> BigRat: ...
    def __ge__(self, other: BigRatLike) -> Unit: ...
    def __gt__(self, other: BigRatLike) -> Unit: ...
    def __le__(self, other: BigRatLike) -> Unit: ...
    @method(egg_fn="egglog.core.bigrat.log")
    def log(self) -> BigRat: ...
    def __lt__(self, other: BigRatLike) -> Unit: ...
    @overload
    def __init__(self, value: int | Fraction, /) -> None: ...
    @overload
    def __init__(self, num: BigIntLike, den: BigIntLike, /) -> None: ...
    @method(egg_fn="egglog.core.bigrat.make")
    def __init__(self, num: BigIntLike | Fraction, den: BigIntLike | None = None, /) -> None: ...
    @method(egg_fn="egglog.core.bigrat.max")
    def max(self, other: BigRatLike) -> BigRat: ...
    @method(egg_fn="egglog.core.bigrat.min")
    def min(self, other: BigRatLike) -> BigRat: ...
    @method(egg_fn="egglog.core.bigrat.mul")
    def __mul__(self, other: BigRatLike) -> BigRat: ...
    @method(preserve=True)
    def __rmul__(self, other: BigRatLike) -> BigRat:
        return convert(other, BigRat).__mul__(self)

    @method(egg_fn="egglog.core.bigrat.neg")
    def __neg__(self) -> BigRat: ...
    @property
    @method(egg_fn="egglog.core.bigrat.numer")
    def numer(self) -> BigInt: ...
    @method(egg_fn="egglog.core.bigrat.pow")
    def __pow__(self, other: BigRatLike) -> BigRat: ...
    @method(preserve=True)
    def __rpow__(self, other: BigRatLike) -> BigRat:
        return convert(other, BigRat).__pow__(self)

    @method(egg_fn="egglog.core.bigrat.round")
    def round(self) -> BigRat: ...
    @method(egg_fn="egglog.core.bigrat.sqrt")
    def sqrt(self) -> BigRat: ...
    @method(egg_fn="egglog.core.bigrat.sub")
    def __sub__(self, other: BigRatLike) -> BigRat: ...
    @method(preserve=True)
    def __rsub__(self, other: BigRatLike) -> BigRat:
        return convert(other, BigRat).__sub__(self)

    @method(egg_fn="egglog.core.bigrat.to_f64")
    def to_f64(self) -> f64: ...
    @method(egg_fn="egglog.core.bigrat.to_i64")
    def to_i64(self) -> i64: ...

    __lt__ = method(egg_fn="egglog.core.bigrat.lt")(__lt__)
    __gt__ = method(egg_fn="egglog.core.bigrat.gt")(__gt__)
    __le__ = method(egg_fn="egglog.core.bigrat.le")(__le__)
    __ge__ = method(egg_fn="egglog.core.bigrat.ge")(__ge__)


class Rational(BuiltinExpr, egg_sort="Rational"):
    __match_args__ = ("value",)

    @property
    @method(preserve=True)
    def value(self) -> Fraction:
        from .deconstruct import get_literal_value  # noqa: PLC0415 -- avoid the inspection import cycle

        value = get_literal_value(self)
        if value is None:
            raise ExprValueError(self, "Rational")
        return value

    @method(preserve=True)
    @deprecated("use .value")
    def eval(self) -> Fraction:
        return self.value

    @method(preserve=True)
    def __int__(self) -> int:
        return int(self.value)

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.value)

    @method(egg_fn="egglog.experimental.rational.abs")
    def __abs__(self) -> Rational: ...
    @method(egg_fn="egglog.experimental.rational.add")
    def __add__(self, other: RationalLike) -> Rational: ...
    @method(preserve=True)
    def __radd__(self, other: RationalLike) -> Rational:
        return convert(other, Rational).__add__(self)

    @method(egg_fn="egglog.experimental.rational.cbrt")
    def cbrt(self) -> Rational: ...
    @method(egg_fn="egglog.experimental.rational.ceil")
    def ceil(self) -> Rational: ...
    @property
    @method(egg_fn="egglog.experimental.rational.denom")
    def denom(self) -> i64: ...
    @method(egg_fn="egglog.experimental.rational.div")
    def __truediv__(self, other: RationalLike) -> Rational: ...
    @method(preserve=True)
    def __rtruediv__(self, other: RationalLike) -> Rational:
        return convert(other, Rational).__truediv__(self)

    @method(egg_fn="egglog.experimental.rational.floor")
    def floor(self) -> Rational: ...
    def __ge__(self, other: RationalLike) -> Unit: ...
    def __gt__(self, other: RationalLike) -> Unit: ...
    def __le__(self, other: RationalLike) -> Unit: ...
    @method(egg_fn="egglog.experimental.rational.log")
    def log(self) -> Rational: ...
    def __lt__(self, other: RationalLike) -> Unit: ...
    @overload
    def __init__(self, value: int | Fraction, /) -> None: ...
    @overload
    def __init__(self, num: i64Like, den: i64Like, /) -> None: ...
    @method(egg_fn="egglog.experimental.rational.make")
    def __init__(self, num: i64Like | Fraction, den: i64Like | None = None, /) -> None: ...
    @method(egg_fn="egglog.experimental.rational.max")
    def max(self, other: RationalLike) -> Rational: ...
    @method(egg_fn="egglog.experimental.rational.min")
    def min(self, other: RationalLike) -> Rational: ...
    @method(egg_fn="egglog.experimental.rational.mul")
    def __mul__(self, other: RationalLike) -> Rational: ...
    @method(preserve=True)
    def __rmul__(self, other: RationalLike) -> Rational:
        return convert(other, Rational).__mul__(self)

    @method(egg_fn="egglog.experimental.rational.neg")
    def __neg__(self) -> Rational: ...
    @property
    @method(egg_fn="egglog.experimental.rational.numer")
    def numer(self) -> i64: ...
    @method(egg_fn="egglog.experimental.rational.pow")
    def __pow__(self, other: RationalLike) -> Rational: ...
    @method(preserve=True)
    def __rpow__(self, other: RationalLike) -> Rational:
        return convert(other, Rational).__pow__(self)

    @method(egg_fn="egglog.experimental.rational.round")
    def round(self) -> Rational: ...
    @method(egg_fn="egglog.experimental.rational.sqrt")
    def sqrt(self) -> Rational: ...
    @method(egg_fn="egglog.experimental.rational.sub")
    def __sub__(self, other: RationalLike) -> Rational: ...
    @method(preserve=True)
    def __rsub__(self, other: RationalLike) -> Rational:
        return convert(other, Rational).__sub__(self)

    @method(egg_fn="egglog.experimental.rational.to_f64")
    def to_f64(self) -> f64: ...

    __lt__ = method(egg_fn="egglog.experimental.rational.lt")(__lt__)
    __gt__ = method(egg_fn="egglog.experimental.rational.gt")(__gt__)
    __le__ = method(egg_fn="egglog.experimental.rational.le")(__le__)
    __ge__ = method(egg_fn="egglog.experimental.rational.ge")(__ge__)


BoolLike: TypeAlias = Bool | bool

i64Like: TypeAlias = i64 | int

StringLike: TypeAlias = String | str

f64Like: TypeAlias = f64 | float | int

MapLike: TypeAlias = Map[K, V] | dict[KH, VH]

VecLike: TypeAlias = Vec[T] | tuple[H, ...] | list[H]

MultiSetLike: TypeAlias = MultiSet[T] | tuple[H, ...]

SetLike: TypeAlias = Set[T] | set[H]

PyObjectLike: TypeAlias = PyObject | object

BigIntLike: TypeAlias = BigInt | i64Like

BigRatLike: TypeAlias = BigRat | Fraction | i64Like

RationalLike: TypeAlias = Rational | Fraction | i64Like

converter(RuntimeFunction, UnstableFn, _function_to_expr)
converter(FunctionType, UnstableFn, _function_to_expr)
converter(MethodType, UnstableFn, _function_to_expr)
converter(partial, UnstableFn, _function_to_expr)
converter(tuple, Pair, _tuple_to_pair)
converter(type(None), Maybe, _none_to_maybe)
converter(list, Vec, lambda value: _collection_to_expr(value, "Vec"))
converter(tuple, Vec, lambda value: _collection_to_expr(value, "Vec"))
converter(set, Set, lambda value: _collection_to_expr(value, "Set"))
converter(tuple, MultiSet, lambda value: _collection_to_expr(value, "MultiSet"))
converter(dict, Map, _dict_to_map)
converter(int, i64, i64)
converter(float, f64, lambda value: f64(float(value)))
converter(int, f64, lambda value: f64(float(value)))
converter(str, String, String)
converter(bool, Bool, Bool)
converter(object, PyObject, PyObject)
converter(int, Rational, Rational)
converter(Fraction, Rational, Rational)
converter(i64, Rational, lambda value: Rational(value, 1))
converter(int, BigInt, BigInt)
converter(i64, BigInt, BigInt)
converter(int, BigRat, BigRat)
converter(Fraction, BigRat, BigRat)
converter(i64, BigRat, lambda value: BigRat(value, 1))

_default_free_0 = PyObject.dict()
_default_free_1 = PyObject.dict()


@function(egg_fn="egglog.core.multiset.contains_swapped", builtin=True)
def multiset_contains_swapped(x: T, xs: MultiSet[T]) -> Unit: ...


@function(egg_fn="egglog.core.multiset.flat_map", builtin=True)
def multiset_flat_map(f: Callable[[T], MultiSet[T]], xs: MultiSet[T]) -> MultiSet[T]: ...


@function(egg_fn="egglog.core.multiset.not_contains_swapped", builtin=True)
def multiset_not_contains_swapped(x: T, xs: MultiSet[T]) -> Unit: ...


@function(egg_fn="egglog.core.multiset.reduce", builtin=True)
def multiset_fold(
    f: Callable[[T, T], T],
    initial: T,
    xs: MultiSet[T],
) -> T: ...


@function(egg_fn="egglog.core.multiset.remove_swapped", builtin=True)
def multiset_remove_swapped(x: T, xs: MultiSet[T]) -> MultiSet[T]: ...


@function(egg_fn="egglog.core.multiset.subtract_swapped", builtin=True)
def multiset_subtract_swapped(x: MultiSet[T], xs: MultiSet[T]) -> MultiSet[T]: ...


@function(egg_fn="egglog.core.string.concat", builtin=True)
def join(*strings: StringLike) -> String: ...


@function(egg_fn="egglog.core.string.count_matches", builtin=True)
def count_matches(s: StringLike, pattern: StringLike) -> i64: ...


@function(egg_fn="egglog.experimental.map.fold_kv", builtin=True)
def map_fold_kv(
    f: Callable[[U, K, V], U],
    initial: U,
    xs: Map[K, V],
) -> U: ...


@function(egg_fn="egglog.experimental.maybe.catch", builtin=True)
def catch(f: Callable[[], T]) -> Maybe[T]: ...


@function(egg_fn="egglog.python.pyobject.eval", builtin=True)
def py_eval(
    code: StringLike, globals_: PyObjectLike = _default_free_0, locals_: PyObjectLike = _default_free_0
) -> PyObject: ...


@function(egg_fn="egglog.python.pyobject.exec", builtin=True)
def py_exec(
    code: StringLike, globals_: PyObjectLike = _default_free_1, locals_: PyObjectLike = _default_free_1
) -> PyObject: ...


class PyObjectFunction(Protocol):
    def __call__(self, *__args: PyObject) -> PyObject: ...


@deprecated("use PyObject(fn) directly")
def py_eval_fn(fn: Callable) -> PyObjectFunction:
    return PyObject(fn)


# Complete the ordinary declaration metadata once all forward references exist.
# This validates native signatures and aliases; it does not create an EGraph.
_native_catalog = builtin_catalog()
_source_catalog = _prepare_declarations(globals()[name] for name in __all__)
_native_catalog.roots = tuple(
    _source_catalog.definitions.get(key, reference) for key, reference in _native_catalog.definitions.items()
)
_native_catalog.reindex()
del _native_catalog, _source_catalog
