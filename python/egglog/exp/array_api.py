"""
Experimental Array API support.
"""

# mypy: disable-error-code="empty-body"

from __future__ import annotations

import contextlib
import itertools
import math
import numbers
import os
import sys
from collections.abc import Callable
from copy import copy
from functools import partial
from tempfile import NamedTemporaryFile
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast

import numpy as np

from egglog import *
from egglog.runtime import RuntimeExpr

from .program_gen import *

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType


# Pretend that exprs are numbers b/c sklearn does isinstance checks
numbers.Integral.register(RuntimeExpr)

# Set this to 1 before scipy is ever imported
# https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#using-array-api-standard-support
os.environ["SCIPY_ARRAY_API"] = "1"

array_api_ruleset = ruleset(name="array_api_ruleset")


class Boolean(Expr, ruleset=array_api_ruleset):
    """
    A boolean expression
    """

    NEVER: ClassVar[Boolean]

    def __init__(self, value: BoolLike) -> None: ...

    @method(preserve=True)
    def __bool__(self) -> bool:
        """
        >>> bool(Boolean(True))
        True
        >>> bool(Boolean(False))
        False
        """
        return self.eval()

    @method(preserve=True)
    def eval(self) -> bool:
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_bool)

    @property
    def to_bool(self) -> Bool: ...

    def __or__(self, other: BooleanLike) -> Boolean: ...

    def __and__(self, other: BooleanLike) -> Boolean: ...

    def __invert__(self) -> Boolean: ...

    def __eq__(self, other: BooleanLike) -> Boolean: ...  # type: ignore[override]

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], Boolean], j: Callable[[], Boolean]) -> Boolean:
        """
        Returns i() if b is True, else j(). Wrapped in callables to avoid eager evaluation.

        >>> bool(Boolean.if_(TRUE, lambda: Boolean(True), lambda: Boolean(False)))
        True
        """


BooleanLike = Boolean | BoolLike

TRUE = Boolean(True)
FALSE = Boolean(False)
converter(Bool, Boolean, Boolean)


@array_api_ruleset.register
def _bool(x: Boolean, i: Int, j: Int, b: Bool, bt: Callable[[], Boolean], bf: Callable[[], Boolean]):
    return [
        rule(eq(x).to(Boolean(b))).then(set_(x.to_bool).to(b)),
        rewrite(TRUE | x).to(TRUE),
        rewrite(FALSE | x).to(x),
        rewrite(TRUE & x).to(x),
        rewrite(FALSE & x).to(FALSE),
        rewrite(~TRUE).to(FALSE),
        rewrite(~FALSE).to(TRUE),
        rule(eq(FALSE).to(TRUE)).then(panic("False cannot equal True")),
        rewrite(x == x).to(TRUE),  # noqa: PLR0124
        rewrite(FALSE == TRUE).to(FALSE),
        rewrite(TRUE == FALSE).to(FALSE),
        rewrite(Boolean.if_(TRUE, bt, bf), subsume=True).to(bt()),
        rewrite(Boolean.if_(FALSE, bt, bf), subsume=True).to(bf()),
    ]


class Int(Expr, ruleset=array_api_ruleset):
    # a never int is that should not exist. It could represent for example indexing into an array a value that is out of bounds
    # https://en.wikipedia.org/wiki/Bottom_type
    NEVER: ClassVar[Int]

    @classmethod
    def var(cls, name: StringLike) -> Int: ...

    def __init__(self, value: i64Like) -> None: ...

    def __invert__(self) -> Int: ...

    def __lt__(self, other: IntLike) -> Boolean: ...

    def __le__(self, other: IntLike) -> Boolean: ...
    def __abs__(self) -> Int: ...

    def __eq__(self, other: IntLike) -> Boolean:  # type: ignore[override]
        ...

    # add a hash so that this test can pass
    # https://github.com/scikit-learn/scikit-learn/blob/6fd23fca53845b32b249f2b36051c081b65e2fab/sklearn/utils/validation.py#L486-L487
    @method(preserve=True)
    def __hash__(self) -> int:
        egraph = _get_current_egraph()
        egraph.register(self)
        egraph.run(array_api_schedule)
        simplified = egraph.extract(self)
        return hash(cast("RuntimeExpr", simplified).__egg_typed_expr__)

    def __round__(self, ndigits: OptionalIntLike = None) -> Int: ...

    # TODO: Fix this?
    # Make != always return a Bool, so that numpy.unique works on a tuple of ints
    # In _unique1d
    @method(preserve=True)
    def __ne__(self, other: Int) -> bool:  # type: ignore[override]
        return not (self == other)

    def __gt__(self, other: IntLike) -> Boolean: ...

    def __ge__(self, other: IntLike) -> Boolean: ...

    def __add__(self, other: IntLike) -> Int: ...

    def __sub__(self, other: IntLike) -> Int: ...

    def __mul__(self, other: IntLike) -> Int: ...

    def __truediv__(self, other: IntLike) -> Int: ...

    def __floordiv__(self, other: IntLike) -> Int: ...

    def __mod__(self, other: IntLike) -> Int: ...

    def __divmod__(self, other: IntLike) -> Int: ...

    def __pow__(self, other: IntLike) -> Int: ...

    def __lshift__(self, other: IntLike) -> Int: ...

    def __rshift__(self, other: IntLike) -> Int: ...

    def __and__(self, other: IntLike) -> Int: ...

    def __xor__(self, other: IntLike) -> Int: ...

    def __or__(self, other: IntLike) -> Int: ...

    def __radd__(self, other: IntLike) -> Int: ...

    def __rsub__(self, other: IntLike) -> Int: ...

    def __rmul__(self, other: IntLike) -> Int: ...

    def __rmatmul__(self, other: IntLike) -> Int: ...

    def __rtruediv__(self, other: IntLike) -> Int: ...

    def __rfloordiv__(self, other: IntLike) -> Int: ...

    def __rmod__(self, other: IntLike) -> Int: ...

    def __rpow__(self, other: IntLike) -> Int: ...

    def __rlshift__(self, other: IntLike) -> Int: ...

    def __rrshift__(self, other: IntLike) -> Int: ...

    def __rand__(self, other: IntLike) -> Int: ...

    def __rxor__(self, other: IntLike) -> Int: ...

    def __ror__(self, other: IntLike) -> Int: ...

    @property
    def to_i64(self) -> i64: ...

    @method(preserve=True)
    def eval(self) -> int:
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_i64)

    @method(preserve=True)
    def __index__(self) -> int:
        return self.eval()

    @method(preserve=True)
    def __int__(self) -> int:
        return self.eval()

    @method(preserve=True)
    def __float__(self) -> float:
        return float(self.eval())

    @method(preserve=True)
    def __bool__(self) -> bool:
        return bool(self.eval())

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], Int], j: Callable[[], Int]) -> Int:
        """
        Returns i() if b is True, else j(). Wrapped in callables to avoid eager evaluation.

        >>> int(Int.if_(TRUE, lambda: Int(1), lambda: Int(2)))
        1
        """


@array_api_ruleset.register
def _int(i: i64, j: i64, r: Boolean, o: Int, b: Int, ot: Callable[[], Int], bt: Callable[[], Int]):
    yield rewrite(Int(i) == Int(i)).to(TRUE)
    yield rule(eq(r).to(Int(i) == Int(j)), ne(i).to(j)).then(union(r).with_(FALSE))

    yield rewrite(Int(i) >= Int(i)).to(TRUE)
    yield rule(eq(r).to(Int(i) >= Int(j)), i > j).then(union(r).with_(TRUE))
    yield rule(eq(r).to(Int(i) >= Int(j)), i < j).then(union(r).with_(FALSE))

    yield rewrite(Int(i) < Int(i)).to(FALSE)
    yield rule(eq(r).to(Int(i) < Int(j)), i < j).then(union(r).with_(TRUE))
    yield rule(eq(r).to(Int(i) < Int(j)), i > j).then(union(r).with_(FALSE))

    yield rewrite(Int(i) > Int(i)).to(FALSE)
    yield rule(eq(r).to(Int(i) > Int(j)), i > j).then(union(r).with_(TRUE))
    yield rule(eq(r).to(Int(i) > Int(j)), i < j).then(union(r).with_(FALSE))

    yield rule(eq(o).to(Int(j))).then(set_(o.to_i64).to(j))

    yield rule(eq(Int(i)).to(Int(j)), ne(i).to(j)).then(panic("Real ints cannot be equal to different ints"))

    yield rewrite(Int(i) + Int(j)).to(Int(i + j))
    yield rewrite(Int(i) - Int(j)).to(Int(i - j))
    yield rewrite(Int(i) * Int(j)).to(Int(i * j))
    yield rewrite(Int(i) // Int(j)).to(Int(i / j))
    yield rewrite(Int(i) % Int(j)).to(Int(i % j))
    yield rewrite(Int(i) & Int(j)).to(Int(i & j))
    yield rewrite(Int(i) | Int(j)).to(Int(i | j))
    yield rewrite(Int(i) ^ Int(j)).to(Int(i ^ j))
    yield rewrite(Int(i) << Int(j)).to(Int(i << j))
    yield rewrite(Int(i) >> Int(j)).to(Int(i >> j))
    yield rewrite(~Int(i)).to(Int(~i))
    yield rewrite(Int(i).__abs__()).to(Int(i.__abs__()))

    yield rewrite(Int.if_(TRUE, ot, bt), subsume=True).to(ot())
    yield rewrite(Int.if_(FALSE, ot, bt), subsume=True).to(bt())

    yield rewrite(o.__round__(OptionalInt.none)).to(o)

    # Never cannot be equal to anything real
    yield rule(eq(Int.NEVER).to(Int(i))).then(panic("Int.NEVER cannot be equal to any real int"))


converter(i64, Int, lambda x: Int(x))

IntLike: TypeAlias = Int | i64Like


@function(ruleset=array_api_ruleset)
def check_index(length: IntLike, idx: IntLike) -> Int:
    """
    Returns the index if 0 <= idx < length, otherwise returns Int.NEVER
    """
    length = cast("Int", length)
    idx = cast("Int", idx)
    return Int.if_(((idx >= 0) & (idx < length)), lambda: idx, lambda: Int.NEVER)


class OptionalInt(Expr, ruleset=array_api_ruleset):
    none: ClassVar[OptionalInt]

    @classmethod
    def some(cls, value: Int) -> OptionalInt: ...


OptionalIntLike: TypeAlias = OptionalInt | IntLike | None

converter(type(None), OptionalInt, lambda _: OptionalInt.none)
converter(Int, OptionalInt, OptionalInt.some)


# @array_api_ruleset.register
# def _check_index(i: i64, j: i64, x: Int):
#     yield rewrite(
#         check_index(Int(i), Int(j)),
#     ).to(
#         Int(j),
#         i >= 0,
#         i < j,
#     )

#     yield rewrite(
#         check_index(x, Int(i)),
#     ).to(
#         Int.NEVER,
#         i < 0,
#     )

#     yield rewrite(
#         check_index(Int(i), Int(j)),
#     ).to(
#         Int.NEVER,
#         i >= j,
#     )


class Float(Expr, ruleset=array_api_ruleset):
    # Differentiate costs of three constructors so extraction is deterministic if all three are present
    @method(cost=3)
    def __init__(self, value: f64Like) -> None: ...

    @property
    def to_f64(self) -> f64: ...

    @property
    def to_big_rat(self) -> BigRat: ...

    @method(preserve=True)
    def eval(self) -> float:
        try:
            return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_f64)
        except EggSmolError:
            return float(try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_big_rat))

    def abs(self) -> Float: ...

    @method(cost=2)
    @classmethod
    def rational(cls, r: BigRat) -> Float: ...

    @classmethod
    def from_int(cls, i: IntLike) -> Float: ...

    def __truediv__(self, other: FloatLike) -> Float: ...

    def __mul__(self, other: FloatLike) -> Float: ...

    def __add__(self, other: FloatLike) -> Float: ...

    def __sub__(self, other: FloatLike) -> Float: ...
    def __abs__(self) -> Float: ...

    def __pow__(self, other: FloatLike) -> Float: ...
    def __round__(self, ndigits: OptionalIntLike = None) -> Float: ...

    def __eq__(self, other: FloatLike) -> Boolean: ...  # type: ignore[override]
    def __ne__(self, other: FloatLike) -> Boolean: ...  # type: ignore[override]
    def __lt__(self, other: FloatLike) -> Boolean: ...
    def __le__(self, other: FloatLike) -> Boolean: ...
    def __gt__(self, other: FloatLike) -> Boolean: ...
    def __ge__(self, other: FloatLike) -> Boolean: ...


converter(float, Float, lambda x: Float(x))
converter(Int, Float, lambda x: Float.from_int(x))


FloatLike: TypeAlias = Float | float | IntLike


@array_api_ruleset.register
def _float(fl: Float, f: f64, f2: f64, i: i64, r: BigRat, r1: BigRat, i_: Int):
    return [
        rule(eq(fl).to(Float(f))).then(set_(fl.to_f64).to(f)),
        rule(eq(fl).to(Float.rational(r))).then(set_(fl.to_big_rat).to(r)),
        rewrite(Float.from_int(Int(i))).to(Float(f64.from_i64(i))),
        rewrite(Float(f).abs()).to(Float(f), f >= 0.0),
        rewrite(Float(f).abs()).to(Float(-f), f < 0.0),
        # Convert from float to rational, if its a whole number i.e. can be converted to int
        rewrite(Float(f)).to(Float.rational(BigRat(f.to_i64(), 1)), eq(f64.from_i64(f.to_i64())).to(f)),
        # always convert from int to rational
        rewrite(Float.from_int(Int(i))).to(Float.rational(BigRat(i, 1))),
        rewrite(Float(f) + Float(f2)).to(Float(f + f2)),
        rewrite(Float(f) - Float(f2)).to(Float(f - f2)),
        rewrite(Float(f) * Float(f2)).to(Float(f * f2)),
        rewrite(Float.rational(r) / Float.rational(r1)).to(Float.rational(r / r1)),
        rewrite(Float.rational(r) + Float.rational(r1)).to(Float.rational(r + r1)),
        rewrite(Float.rational(r) - Float.rational(r1)).to(Float.rational(r - r1)),
        rewrite(Float.rational(r) * Float.rational(r1)).to(Float.rational(r * r1)),
        rewrite(Float(f) ** Float(f2)).to(Float(f**f2)),
        # comparisons
        rewrite(Float(f) == Float(f)).to(TRUE),
        rewrite(Float(f) == Float(f2)).to(FALSE, ne(f).to(f2)),
        rewrite(Float(f) != Float(f2)).to(TRUE, f != f2),
        rewrite(Float(f) != Float(f)).to(FALSE),
        rewrite(Float(f) >= Float(f2)).to(TRUE, f >= f2),
        rewrite(Float(f) >= Float(f2)).to(FALSE, f < f2),
        rewrite(Float(f) <= Float(f2)).to(TRUE, f <= f2),
        rewrite(Float(f) <= Float(f2)).to(FALSE, f > f2),
        rewrite(Float(f) > Float(f2)).to(TRUE, f > f2),
        rewrite(Float(f) > Float(f2)).to(FALSE, f <= f2),
        rewrite(Float(f) < Float(f2)).to(TRUE, f < f2),
        rewrite(Float.rational(r) == Float.rational(r)).to(TRUE),
        rewrite(Float.rational(r) == Float.rational(r1)).to(FALSE, ne(r).to(r1)),
        rewrite(Float.rational(r).__round__()).to(Float.rational(r.round())),
        rewrite(Float(f).__abs__()).to(Float(f.__abs__())),
    ]


class TupleInt(Expr, ruleset=array_api_ruleset):
    """
    A tuple of integers.

    The following is true for all types of tuple:

    Tuples have two main constructors:

    - `Tuple[T](vs: Vec[T]=[])`
    - `Tuple.fn(length: Int, idx_fn: Callable[[Int], T])`

    This is so that they can be defined either with a known fixed integer length or a symbolic
    length that could not be resolved to an integer.

    Both constructors must implement two methods:

    * `l.length() -> Int`
    * `l.__getitem__(i: Int) -> T`

    Lists with a known length will be subsumed into the vector representation.

    Lists that have vecs that are equal will have the elements unified.

    Methods that transform lists should also subsume, so that the vector version will be preferred.

    Lists from a vec also have `l.to_vec` set to the original vec.
    """

    def __init__(self, vec: VecLike[Int, IntLike] = Vec[Int].empty()) -> None:
        """
        Create a TupleInt from a Vec of Ints.

        >>> list(TupleInt(Vec(i64(1), i64(2), i64(3))))
        [i64(1), i64(2), i64(3)]
        >>> list(TupleInt())
        []
        """

    @classmethod
    def fn(cls, length: IntLike, idx_fn: Callable[[Int], Int]) -> TupleInt:
        """
        Create a TupleInt from a length and an index function.

        >>> list(TupleInt.fn(3, lambda i: i * 10))
        [i64(0), i64(10), i64(20)]
        """

    def length(self) -> Int:
        """
        Return the length of the tuple.

        >>> int(TupleInt([1, 2, 3]).length())
        3
        >>> int(TupleInt.fn(5, lambda i: i).length())
        5
        """

    def __getitem__(self, i: IntLike) -> Int:
        """
        Return the integer at index i.

        >>> int(TupleInt([10, 20, 30])[1])
        20
        >>> int(TupleInt(3, lambda i: i * 10)[2])
        20
        """

    @method(preserve=True)
    def __len__(self) -> int:
        """
        >>> len(TupleInt([1, 2, 3]))
        3
        """
        return self.length().eval()

    @method(preserve=True)
    def __iter__(self) -> Iterator[Int]:
        return iter(self.eval())

    @method(merge=Vec.__or__)  # type: ignore[prop-decorator]
    @property
    def to_vec(self) -> Vec[Int]:
        """
        Returns the Vec[Int] representation of the TupleInt.
        """

    @method(preserve=True)
    def eval(self) -> tuple[Int, ...]:
        """
        Returns the evaluated tuple of Ints.
        """
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_vec)

    def append(self, i: IntLike) -> TupleInt:
        """
        Append an integer to the end of the tuple.

        >>> ti = TupleInt.range(3)
        >>> ti2 = ti.append(3)
        >>> list(ti2)
        [i64(0), i64(1), i64(2), i64(3)]
        """
        return TupleInt.fn(
            self.length() + 1, lambda j: Int.if_(j == self.length(), lambda: cast("Int", i), lambda: self[j])
        )

    def __add__(self, other: TupleIntLike) -> TupleInt:
        """
        Concatenate two TupleInts.
        >>> ti1 = TupleInt.range(3)
        >>> ti2 = TupleInt.range(2)
        >>> ti3 = ti1 + ti2
        >>> list(ti3)
        [Int(0), Int(1), Int(2), Int(0), Int(1)]
        """
        other = cast("TupleInt", other)
        return TupleInt.fn(
            self.length() + other.length(),
            lambda i: Int.if_(i < self.length(), lambda: self[i], lambda: other[i - self.length()]),
        )

    def drop(self, n: Int) -> TupleInt:
        """
        Return a new tuple with the first n elements dropped.

        >>> ti = TupleInt([1, 2, 3, 4])
        >>> list(ti.drop(2))
        [i64(3), i64(4)]
        """
        return TupleInt.fn(self.length() - n, lambda i: self[i + n])

    def last(self) -> Int:
        """
        Return the last element in the tuple.

        >>> ti = TupleInt([1, 2, 3])
        >>> int(ti.last())
        3
        """
        return self[self.length() - 1]

    def drop_last(self) -> TupleInt:
        """
        Return a new tuple with the last element dropped.

        >>> ti = TupleInt([1, 2, 3])
        >>> list(ti.drop_last())
        [i64(1), i64(2)]
        """
        return TupleInt.fn(self.length() - 1, self.__getitem__)

    @classmethod
    def range(cls, stop: IntLike) -> TupleInt:
        """
        Create a TupleInt with the integers from 0 to stop - 1.
        >>> list(TupleInt.range(5))
        [Int(0), Int(1), Int(2), Int(3), Int(4)]
        """
        return TupleInt.fn(stop, lambda i: i)

    def foldl(self, f: Callable[[Int, Int], Int], init: Int) -> Int:
        """
        Fold the tuple from the left with the given function and initial value.

        >>> ti = TupleInt([1, 2, 3])
        >>> int(ti.foldl(lambda acc, x: acc + x, i64(0)))
        6
        """
        return Int.if_(self.length() == 0, lambda: init, lambda: f(self.drop_last().foldl(f, init), self.last()))

    def foldl_boolean(self, f: Callable[[Boolean, Int], Boolean], init: Boolean) -> Boolean:
        """
        Fold the tuple from the left with the given boolean function and initial value.

        >>> ti = TupleInt([1, 2, 3])
        >>> bool(ti.foldl_boolean(lambda acc, x: acc | (x == i64(2)), FALSE))
        True
        >>> bool(ti.foldl_boolean(lambda acc, x: acc & (x < i64(3)), TRUE))
        False
        """
        return Boolean.if_(
            self.length() == 0, lambda: init, lambda: f(self.drop_last().foldl_boolean(f, init), self.last())
        )

    def foldl_tuple_int(self, f: Callable[[TupleInt, Int], TupleInt], init: TupleIntLike) -> TupleInt:
        """
        Fold the tuple from the left with the given tuple function and initial value.

        >>> ti = TupleInt([1, 2, 3])
        >>> ti2 = ti.foldl_tuple_int(lambda acc, x: acc.append(x * 2), TupleInt())
        >>> list(ti2)
        [Int(2), Int(4), Int(6)]
        """
        init = cast("TupleInt", init)
        return TupleInt.if_(
            self.length() == 0, lambda: init, lambda: f(self.drop_last().foldl_tuple_int(f, init), self.last())
        )

    def contains(self, i: Int) -> Boolean:
        """
        Returns True if the tuple contains the given integer.

        >>> ti = TupleInt([1, 2, 3])
        >>> bool(ti.contains(i64(2)))
        True
        >>> bool(ti.contains(i64(4)))
        False
        """
        return self.foldl_boolean(lambda acc, j: acc | (i == j), FALSE)

    def filter(self, f: Callable[[Int], Boolean]) -> TupleInt:
        """
        Returns a new tuple with only the elements that satisfy the given predicate.

        >>> ti = TupleInt([1, 2, 3, 4])
        >>> list(ti.filter(lambda x: x % i64(2) == i64(0)))
        [i64(2), i64(4)]
        >>> list(ti.filter(lambda x: x > i64(2)))
        [i64(3), i64(4)]
        """
        return self.foldl_tuple_int(
            lambda acc, v: TupleInt.if_(f(v), lambda: acc.append(v), lambda: acc),
            TupleInt(),
        )

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], TupleInt], j: Callable[[], TupleInt]) -> TupleInt:
        """
        Returns i() if b is True, else j(). Wrapped in callables to avoid eager evaluation.

        >>> ti1 = TupleInt([1, 2])
        >>> ti2 = TupleInt([3, 4])
        >>> ti = TupleInt.if_(TRUE, lambda: ti1, lambda: ti2)
        >>> list(ti)
        [i64(1), i64(2)]
        """

    def product(self) -> Int:
        """
        Return the product of all elements in the tuple.

        >>> ti = TupleInt([1, 2, 3, 4])
        >>> int(ti.product())
        24
        """
        return self.foldl(lambda acc, i: acc * i, Int(1))

    def select(self, indices: TupleIntLike) -> TupleInt:
        """
        Return a new tuple with the elements at the given indices

        >>> ti = TupleInt([10, 20, 30, 40])
        >>> indices = TupleInt([1, 3])
        >>> list(ti.select(indices))
        [Int(20), Int(40)]
        """
        indices = cast("TupleInt", indices)
        return indices.map(lambda i: self[i])

    def deselect(self, indices: TupleIntLike) -> TupleInt:
        """
        Return a new tuple with the elements not at the given indices

        >>> ti = TupleInt([10, 20, 30, 40])
        >>> indices = TupleInt([1, 3])
        >>> list(ti.deselect(indices))
        [i64(10), i64(30)]
        """
        indices = cast("TupleInt", indices)
        return TupleInt.range(self.length()).filter(lambda i: ~indices.contains(i)).map(lambda i: self[i])

    def reverse(self) -> TupleInt:
        """
        Return a new tuple with the elements in reverse order.

        >>> ti = TupleInt([1, 2, 3])
        >>> list(ti.reverse())
        [Int(3), Int(2), Int(1)]
        """
        return TupleInt.fn(self.length(), lambda i: self[self.length() - i - 1])

    def map(self, f: Callable[[Int], Int]) -> TupleInt:
        """
        Returns a new tuple with each element transformed by the given function.

        >>> ti = TupleInt([1, 2, 3])
        >>> list(ti.map(lambda x: x * i64(2)))
        [i64(2), i64(4), i64(6)]
        """
        return TupleInt.fn(self.length(), lambda i: f(self[i]))

    # Put at bottom so can use previous methods when resolving
    def map_tuple_int(self, f: Callable[[Int], TupleInt]) -> TupleTupleInt:
        """
        Returns a new tuple of TupleInts with each element transformed by the given function.

        >>> ti = TupleInt([1, 2])
        >>> tti = ti.map_tuple_int(lambda x: TupleInt([x, x + 10]))
        >>> list(tti[0])
        [i64(1), i64(11)]
        >>> list(tti[1])
        [i64(2), i64(12)]
        """
        return TupleTupleInt.fn(self.length(), lambda i: f(self[i]))

    def map_value(self, f: Callable[[Int], Value]) -> TupleValue:
        """
        Returns a new tuple of Values with each element transformed by the given function.

        >>> ti = TupleInt([1, 2])
        >>> tv = ti.map_value(lambda x: Value.from_int(x * i64(3)))
        >>> list(tv)
        [Value(i64(3)), Value(i64(6))]
        """
        return TupleValue.fn(self.length(), lambda i: f(self[i]))


converter(Vec[Int], TupleInt, TupleInt)
TupleIntLike: TypeAlias = TupleInt | VecLike[Int, IntLike]


@array_api_ruleset.register
def _tuple_int(
    i: Int,
    i2: Int,
    idx_fn: Callable[[Int], Int],
    vs: Vec[Int],
    ti: TupleInt,
    k: i64,
    lt: Callable[[], TupleInt],
    lf: Callable[[], TupleInt],
):
    yield rule(eq(ti).to(TupleInt(vs))).then(set_(ti.to_vec).to(vs))

    yield rewrite(TupleInt.fn(i2, idx_fn).length()).to(i2)
    yield rewrite(TupleInt.fn(i2, idx_fn)[i]).to(idx_fn(check_index(i, i2)))

    yield rewrite(TupleInt(vs).length()).to(Int(vs.length()))
    yield rewrite(TupleInt(vs)[Int(k)]).to(vs[k], k >= 0, k < vs.length())

    yield rewrite(TupleInt.fn(Int(k), idx_fn), subsume=True).to(TupleInt(k.range().map(lambda i: idx_fn(Int(i)))))

    yield rewrite(TupleInt.if_(TRUE, lt, lf)).to(lt())
    yield rewrite(TupleInt.if_(FALSE, lt, lf)).to(lf())


class TupleTupleInt(Expr, ruleset=array_api_ruleset):
    def __init__(self, vec: VecLike[TupleInt, TupleIntLike] = ()) -> None: ...
    @classmethod
    def fn(cls, length: IntLike, idx_fn: Callable[[Int], TupleInt]) -> TupleTupleInt: ...
    def length(self) -> Int: ...
    def __getitem__(self, i: IntLike) -> TupleInt: ...
    @method(preserve=True)
    def __len__(self) -> int:
        return self.length().eval()

    @method(preserve=True)
    def __iter__(self) -> Iterator[TupleInt]:
        return iter(self.eval())

    @method(merge=Vec.__or__)  # type: ignore[prop-decorator]
    @property
    def to_vec(self) -> Vec[TupleInt]: ...
    @method(preserve=True)
    def eval(self) -> tuple[TupleInt, ...]:
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_vec)

    def append(self, i: TupleIntLike) -> TupleTupleInt:
        return TupleTupleInt.fn(
            self.length() + 1, lambda j: TupleInt.if_(j == self.length(), lambda: cast("TupleInt", i), lambda: self[j])
        )

    def __add__(self, other: TupleTupleIntLike) -> TupleTupleInt:
        other = cast("TupleTupleInt", other)
        return TupleTupleInt.fn(
            self.length() + other.length(),
            lambda i: TupleInt.if_(i < self.length(), lambda: self[i], lambda: other[i - self.length()]),
        )

    def drop(self, n: Int) -> TupleTupleInt:
        return TupleTupleInt.fn(self.length() - n, lambda i: self[i + n])

    def map_int(self, f: Callable[[TupleInt], Int]) -> TupleInt:
        return TupleInt.fn(self.length(), lambda i: f(self[i]))

    def foldl_value(self, f: Callable[[Value, TupleInt], Value], init: ValueLike) -> Value:
        return Value.if_(
            self.length() == 0,
            lambda: cast("Value", init),
            lambda: f(self.drop_last().foldl_value(f, init), self.last()),
        )

    def last(self) -> TupleInt:
        return self[self.length() - 1]

    def drop_last(self) -> TupleTupleInt:
        return TupleTupleInt.fn(self.length() - 1, self.__getitem__)

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], TupleTupleInt], j: Callable[[], TupleTupleInt]) -> TupleTupleInt: ...

    @method(subsume=True)
    def product(self) -> TupleTupleInt:
        """
        Cartesian product of inputs

        https://docs.python.org/3/library/itertools.html#itertools.product

        https://github.com/saulshanabrook/saulshanabrook/discussions/39

        >>> list(TupleTupleInt([TupleInt([1, 2]), TupleInt([3, 4])]).product())
        [TupleInt([1, 3]), TupleInt([1, 4]), TupleInt([2, 3]), TupleInt([2, 4])]
        """
        return TupleTupleInt.fn(
            self.map_int(lambda x: x.length()).product(),
            lambda i: TupleInt.fn(
                self.length(),
                lambda j: self[j][(i // self.drop(j + 1).map_int(lambda x: x.length()).product()) % self[j].length()],
            ),
        )


converter(Vec[TupleInt], TupleTupleInt, lambda x: TupleTupleInt(x))

TupleTupleIntLike: TypeAlias = TupleTupleInt | VecLike[TupleInt, TupleIntLike]


@array_api_ruleset.register
def _tuple_tuple_int(
    i: Int,
    i2: Int,
    idx_fn: Callable[[Int], TupleInt],
    vs: Vec[TupleInt],
    ti: TupleTupleInt,
    k: i64,
    lt: Callable[[], TupleTupleInt],
    lf: Callable[[], TupleTupleInt],
):
    yield rule(eq(ti).to(TupleTupleInt(vs))).then(set_(ti.to_vec).to(vs))

    yield rewrite(TupleTupleInt.fn(i2, idx_fn).length()).to(i2)
    yield rewrite(TupleTupleInt.fn(i2, idx_fn)[i]).to(idx_fn(check_index(i, i2)))

    yield rewrite(TupleTupleInt(vs).length()).to(Int(vs.length()))
    yield rewrite(TupleTupleInt(vs)[Int(k)]).to(vs[k], k >= 0, k < vs.length())

    yield rewrite(TupleTupleInt.fn(Int(k), idx_fn), subsume=True).to(
        TupleTupleInt(k.range().map(lambda i: idx_fn(Int(i))))
    )

    yield rewrite(TupleTupleInt.if_(TRUE, lt, lf)).to(lt())
    yield rewrite(TupleTupleInt.if_(FALSE, lt, lf)).to(lf())


class DType(Expr, ruleset=array_api_ruleset):
    float64: ClassVar[DType]
    float32: ClassVar[DType]
    int64: ClassVar[DType]
    int32: ClassVar[DType]
    object: ClassVar[DType]
    bool: ClassVar[DType]

    def __eq__(self, other: DType) -> Boolean:  # type: ignore[override]
        ...


float64 = DType.float64
float32 = DType.float32
int32 = DType.int32
int64 = DType.int64

_DTYPES = [float64, float32, int32, int64, DType.object]

converter(type, DType, lambda x: convert(np.dtype(x), DType))
converter(np.dtype, DType, lambda x: getattr(DType, x.name))


@array_api_ruleset.register
def _():
    for l, r in itertools.product(_DTYPES, repeat=2):
        yield rewrite(l == r).to(TRUE if l is r else FALSE)


class IsDtypeKind(Expr, ruleset=array_api_ruleset):
    NULL: ClassVar[IsDtypeKind]

    @classmethod
    def string(cls, s: StringLike) -> IsDtypeKind: ...

    @classmethod
    def dtype(cls, d: DType) -> IsDtypeKind: ...

    @method(cost=10)
    def __or__(self, other: IsDtypeKind) -> IsDtypeKind: ...


# TODO: Make kind more generic to support tuples.
@function
def isdtype(dtype: DType, kind: IsDtypeKind) -> Boolean: ...


converter(DType, IsDtypeKind, lambda x: IsDtypeKind.dtype(x))
converter(str, IsDtypeKind, lambda x: IsDtypeKind.string(x))
converter(
    tuple, IsDtypeKind, lambda x: convert(x[0], IsDtypeKind) | convert(x[1:], IsDtypeKind) if x else IsDtypeKind.NULL
)


@array_api_ruleset.register
def _isdtype(d: DType, k1: IsDtypeKind, k2: IsDtypeKind):
    return [
        rewrite(isdtype(DType.float32, IsDtypeKind.string("integral"))).to(FALSE),
        rewrite(isdtype(DType.float64, IsDtypeKind.string("integral"))).to(FALSE),
        rewrite(isdtype(DType.object, IsDtypeKind.string("integral"))).to(FALSE),
        rewrite(isdtype(DType.int64, IsDtypeKind.string("integral"))).to(TRUE),
        rewrite(isdtype(DType.int32, IsDtypeKind.string("integral"))).to(TRUE),
        rewrite(isdtype(DType.float32, IsDtypeKind.string("real floating"))).to(TRUE),
        rewrite(isdtype(DType.float64, IsDtypeKind.string("real floating"))).to(TRUE),
        rewrite(isdtype(DType.object, IsDtypeKind.string("real floating"))).to(FALSE),
        rewrite(isdtype(DType.int64, IsDtypeKind.string("real floating"))).to(FALSE),
        rewrite(isdtype(DType.int32, IsDtypeKind.string("real floating"))).to(FALSE),
        rewrite(isdtype(DType.float32, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.float64, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.object, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.int64, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.int32, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(d, IsDtypeKind.NULL)).to(FALSE),
        rewrite(isdtype(d, IsDtypeKind.dtype(d))).to(TRUE),
        rewrite(isdtype(d, k1 | k2)).to(isdtype(d, k1) | isdtype(d, k2)),
        rewrite(k1 | IsDtypeKind.NULL).to(k1),
    ]


class Value(Expr, ruleset=array_api_ruleset):
    NEVER: ClassVar[Value]

    # TODO: Rename to avoid name conflicts
    @classmethod
    def int(cls, i: IntLike) -> Value: ...

    @classmethod
    def float(cls, f: FloatLike) -> Value: ...

    @classmethod
    def bool(cls, b: BooleanLike) -> Value: ...

    def isfinite(self) -> Boolean: ...

    # TODO: Fix
    def __lt__(self, other: ValueLike) -> Value: ...
    def __le__(self, other: ValueLike) -> Boolean: ...
    def __gt__(self, other: ValueLike) -> Boolean: ...
    def __ge__(self, other: ValueLike) -> Boolean: ...
    def __eq__(self, other: ValueLike) -> Boolean: ...  # type: ignore[override]

    def __truediv__(self, other: ValueLike) -> Value: ...
    def __mul__(self, other: ValueLike) -> Value: ...
    def __add__(self, other: ValueLike) -> Value: ...
    def __sub__(self, other: ValueLike) -> Value: ...
    def __pow__(self, other: ValueLike) -> Value: ...

    def __abs__(self) -> Value: ...

    def astype(self, dtype: DType) -> Value: ...

    # TODO: Add all operations

    @property
    def dtype(self) -> DType:
        """
        Default dtype for this scalar value
        """

    @property
    def to_bool(self) -> Boolean: ...

    @property
    def to_int(self) -> Int: ...

    @property
    def to_float(self) -> Float: ...

    @property
    def to_truthy_value(self) -> Value:
        """
        Converts the value to a bool, based on if its truthy.

        https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.any.html
        """

    def conj(self) -> Value: ...
    def real(self) -> Value: ...
    def sqrt(self) -> Value: ...

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], Value], j: Callable[[], Value]) -> Value: ...

    def __int__(self) -> int:  # type: ignore[valid-type]
        return self.to_int.eval()

    def __float__(self) -> float:  # type: ignore[valid-type]
        return self.to_float.eval()


ValueLike: TypeAlias = Value | IntLike | FloatLike | BooleanLike


converter(Int, Value, Value.int)
converter(Float, Value, Value.float)
converter(Boolean, Value, Value.bool)
converter(Value, Int, lambda x: x.to_int, 10)


@array_api_ruleset.register
def _value(
    i: Int,
    f: Float,
    b: Boolean,
    v: Value,
    v1: Value,
    v2: Value,
    i1: Int,
    f1: Float,
    b1: Boolean,
    vt: Callable[[], Value],
    v1t: Callable[[], Value],
):
    # Default dtypes
    # https://data-apis.org/array-api/latest/API_specification/data_types.html?highlight=dtype#default-data-types
    yield rewrite(Value.int(i).dtype).to(DType.int64)
    yield rewrite(Value.float(f).dtype).to(DType.float64)
    yield rewrite(Value.bool(b).dtype).to(DType.bool)

    yield rewrite(Value.bool(b).to_bool).to(b)
    yield rewrite(Value.int(i).to_int).to(i)
    yield rewrite(Value.float(f).to_float).to(f)

    yield rewrite(Value.bool(b).to_truthy_value).to(Value.bool(b))
    # TODO: Add more rules for to_bool_value

    yield rewrite(Value.float(f).conj()).to(Value.float(f))
    yield rewrite(Value.float(f).real()).to(Value.float(f))
    yield rewrite(Value.int(i).real()).to(Value.int(i))
    yield rewrite(Value.int(i).conj()).to(Value.int(i))

    yield rewrite(Value.float(f).sqrt()).to(Value.float(f ** (0.5)))

    yield rewrite(Value.float(Float.rational(BigRat(0, 1))) + v).to(v)

    yield rewrite(Value.if_(TRUE, vt, v1t)).to(vt())
    yield rewrite(Value.if_(FALSE, vt, v1t)).to(v1t())

    # ==
    yield rewrite(Value.int(i) == Value.int(i1)).to(i == i1)
    yield rewrite(Value.float(f) == Value.float(f1)).to(f == f1)
    yield rewrite(Value.bool(b) == Value.bool(b1)).to(b == b1)
    # >=
    yield rewrite(Value.int(i) >= Value.int(i1)).to(i >= i1)
    yield rewrite(Value.float(f) >= Value.float(f1)).to(f >= f1)
    # <=
    yield rewrite(Value.int(i) <= Value.int(i1)).to(i <= i1)
    yield rewrite(Value.float(f) <= Value.float(f1)).to(f <= f1)
    # >
    yield rewrite(Value.int(i) > Value.int(i1)).to(i > i1)
    yield rewrite(Value.float(f) > Value.float(f1)).to(f > f1)
    # <
    yield rewrite(Value.int(i) < Value.int(i1)).to(Value.bool(i < i1))
    yield rewrite(Value.float(f) < Value.float(f1)).to(Value.bool(f < f1))

    # /
    yield rewrite(Value.float(f) / Value.float(f1)).to(Value.float(f / f1))
    # *
    yield rewrite(Value.float(f) * Value.float(f1)).to(Value.float(f * f1))
    yield rewrite(Value.int(i) * Value.int(i1)).to(Value.int(i * i1))
    # +
    yield rewrite(Value.float(f) + Value.float(f1)).to(Value.float(f + f1))
    yield rewrite(Value.int(i) + Value.int(i1)).to(Value.int(i + i1))
    # -
    yield rewrite(Value.float(f) - Value.float(f1)).to(Value.float(f - f1))
    yield rewrite(Value.int(i) - Value.int(i1)).to(Value.int(i - i1))
    # **
    yield rewrite(Value.float(f) ** Value.float(f1)).to(Value.float(f**f1))
    yield rewrite(Value.int(i) ** Value.int(i1)).to(Value.int(i**i1))
    yield rewrite(Value.int(i) ** Value.float(f1)).to(Value.float(Float.from_int(i) ** f1))

    # abs
    yield rewrite(Value.int(i).__abs__()).to(Value.int(i.__abs__()))
    yield rewrite(Value.float(f).__abs__()).to(Value.float(f.__abs__()))
    # abs(x) **2 = x**2
    yield rewrite(v.__abs__() ** Value.float(Float.rational(BigRat(2, 1)))).to(v ** Value.float(2))

    # ** distributes over division
    yield rewrite((v1 / v) ** v2, subsume=True).to(v1**v2 / (v**v2))
    # x ** y ** z = x ** (y * z)
    yield rewrite((v**v1) ** v2, subsume=True).to(v ** (v1 * v2))
    yield rewrite(Value.float(f) * Value.int(i)).to(Value.float(f * Float.from_int(i)))
    yield rewrite(v ** Value.float(Float.rational(BigRat(1, 1)))).to(v)
    yield rewrite(Value.float(Float.from_int(i))).to(Value.int(i))


class TupleValue(Expr, ruleset=array_api_ruleset):
    def __init__(self, vec: VecLike[Value, ValueLike] = ()) -> None: ...
    @classmethod
    def fn(cls, length: IntLike, idx_fn: Callable[[Int], Value]) -> TupleValue: ...
    def length(self) -> Int: ...
    def __getitem__(self, i: IntLike) -> Value: ...
    @method(preserve=True)
    def __len__(self) -> int:
        return self.length().eval()

    @method(preserve=True)
    def __iter__(self) -> Iterator[Value]:
        return iter(self.eval())

    @method(merge=Vec.__or__)  # type: ignore[prop-decorator]
    @property
    def to_vec(self) -> Vec[Value]: ...
    @method(preserve=True)
    def eval(self) -> tuple[Value, ...]:
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_vec)

    def append(self, i: ValueLike) -> TupleValue:
        return TupleValue.fn(
            self.length() + 1, lambda j: Value.if_(j == self.length(), lambda: cast("Value", i), lambda: self[j])
        )

    def __add__(self, other: TupleValueLike) -> TupleValue:
        other = cast("TupleValue", other)
        return TupleValue.fn(
            self.length() + other.length(),
            lambda i: Value.if_(i < self.length(), lambda: self[i], lambda: other[i - self.length()]),
        )

    def last(self) -> Value:
        return self[self.length() - 1]

    def drop_last(self) -> TupleValue:
        return TupleValue.fn(self.length() - 1, self.__getitem__)

    def foldl_boolean(self, f: Callable[[Boolean, Value], Boolean], init: BooleanLike) -> Boolean:
        return Boolean.if_(
            self.length() == 0,
            lambda: cast("Boolean", init),
            lambda: f(self.drop_last().foldl_boolean(f, init), self.last()),
        )

    def foldl_value(self, f: Callable[[Value, Value], Value], init: ValueLike) -> Value:
        return Value.if_(
            self.length() == 0,
            lambda: cast("Value", init),
            lambda: f(self.drop_last().foldl_value(f, init), self.last()),
        )

    def map_value(self, f: Callable[[Value], Value]) -> TupleValue:
        return TupleValue.fn(self.length(), lambda i: f(self[i]))

    def contains(self, value: ValueLike) -> Boolean:
        value = cast("Value", value)
        return self.foldl_boolean(lambda acc, j: acc | (value == j), FALSE)

    @method(subsume=True)
    @classmethod
    def from_tuple_int(cls, ti: TupleIntLike) -> TupleValue:
        ti = cast("TupleInt", ti)
        return TupleValue.fn(ti.length(), lambda i: Value.int(ti[i]))

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], TupleValue], j: Callable[[], TupleValue]) -> TupleValue: ...


converter(Vec[Value], TupleValue, lambda x: TupleValue(x))
converter(TupleInt, TupleValue, lambda x: TupleValue.from_tuple_int(x))

TupleValueLike: TypeAlias = TupleValue | VecLike[Value, ValueLike] | TupleIntLike


@array_api_ruleset.register
def _tuple_value(
    i: Int,
    i2: Int,
    idx_fn: Callable[[Int], Value],
    vs: Vec[Value],
    ti: TupleValue,
    k: i64,
    lt: Callable[[], TupleValue],
    lf: Callable[[], TupleValue],
):
    yield rule(eq(ti).to(TupleValue(vs))).then(set_(ti.to_vec).to(vs))

    yield rewrite(TupleValue.fn(i2, idx_fn).length()).to(i2)
    yield rewrite(TupleValue.fn(i2, idx_fn)[i]).to(idx_fn(check_index(i, i2)))

    yield rewrite(TupleValue(vs).length()).to(Int(vs.length()))
    yield rewrite(TupleValue(vs)[Int(k)]).to(vs[k], k >= 0, k < vs.length())

    yield rewrite(TupleValue.fn(Int(k), idx_fn), subsume=True).to(TupleValue(k.range().map(lambda i: idx_fn(Int(i)))))

    yield rewrite(TupleValue.if_(TRUE, lt, lf)).to(lt())
    yield rewrite(TupleValue.if_(FALSE, lt, lf)).to(lf())


class TupleTupleValue(Expr, ruleset=array_api_ruleset):
    def __init__(self, vec: VecLike[TupleValue, TupleValueLike] = ()) -> None: ...
    @classmethod
    def fn(cls, length: IntLike, idx_fn: Callable[[Int], TupleValue]) -> TupleTupleValue: ...
    def length(self) -> Int: ...
    def __getitem__(self, i: IntLike) -> TupleValue: ...
    @method(preserve=True)
    def __len__(self) -> int:
        return self.length().eval()

    @method(preserve=True)
    def __iter__(self) -> Iterator[TupleValue]:
        return iter(self.eval())

    @method(merge=Vec.__or__)  # type: ignore[prop-decorator]
    @property
    def to_vec(self) -> Vec[TupleValue]: ...
    @method(preserve=True)
    def eval(self) -> tuple[TupleValue, ...]:
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_vec)


converter(tuple, TupleTupleValue, lambda x: TupleTupleValue(Vec(*(convert(i, TupleValue) for i in x))))
converter(list, TupleTupleValue, lambda x: TupleTupleValue(Vec(*(convert(i, TupleValue) for i in x))))
TupleTupleValueLike: TypeAlias = TupleTupleValue | list[TupleValueLike] | tuple[TupleValueLike, ...]


@array_api_ruleset.register
def _tuple_tuple_value(
    i: Int, i2: Int, idx_fn: Callable[[Int], TupleValue], vs: Vec[TupleValue], ti: TupleTupleValue, k: i64
):
    yield rule(eq(ti).to(TupleTupleValue(vs))).then(set_(ti.to_vec).to(vs))

    yield rewrite(TupleTupleValue.fn(i2, idx_fn).length()).to(i2)
    yield rewrite(TupleTupleValue.fn(i2, idx_fn)[i]).to(idx_fn(check_index(i, i2)))

    yield rewrite(TupleTupleValue(vs).length()).to(Int(vs.length()))
    yield rewrite(TupleTupleValue(vs)[Int(k)]).to(vs[k], k >= 0, k < vs.length())

    yield rewrite(TupleTupleValue.fn(Int(k), idx_fn), subsume=True).to(
        TupleTupleValue(k.range().map(lambda i: idx_fn(Int(i))))
    )


@function
def possible_values(values: Value) -> TupleValue:
    """
    All possible values in the input value.
    """


class Slice(Expr, ruleset=array_api_ruleset):
    def __init__(
        self,
        start: OptionalInt = OptionalInt.none,
        stop: OptionalInt = OptionalInt.none,
        step: OptionalInt = OptionalInt.none,
    ) -> None: ...


converter(
    slice,
    Slice,
    lambda x: Slice(convert(x.start, OptionalInt), convert(x.stop, OptionalInt), convert(x.step, OptionalInt)),
)

SliceLike: TypeAlias = Slice | slice


class MultiAxisIndexKeyItem(Expr, ruleset=array_api_ruleset):
    ELLIPSIS: ClassVar[MultiAxisIndexKeyItem]
    NONE: ClassVar[MultiAxisIndexKeyItem]

    @classmethod
    def int(cls, i: Int) -> MultiAxisIndexKeyItem: ...

    @classmethod
    def slice(cls, slice: Slice) -> MultiAxisIndexKeyItem: ...


converter(type(...), MultiAxisIndexKeyItem, lambda _: MultiAxisIndexKeyItem.ELLIPSIS)
converter(type(None), MultiAxisIndexKeyItem, lambda _: MultiAxisIndexKeyItem.NONE)
converter(Int, MultiAxisIndexKeyItem, MultiAxisIndexKeyItem.int)
converter(Slice, MultiAxisIndexKeyItem, MultiAxisIndexKeyItem.slice)

MultiAxisIndexKeyItemLike: TypeAlias = MultiAxisIndexKeyItem | EllipsisType | None | IntLike | SliceLike


class MultiAxisIndexKey(Expr, ruleset=array_api_ruleset):
    def __init__(self, length: IntLike, idx_fn: Callable[[Int], MultiAxisIndexKeyItem]) -> None: ...

    def __add__(self, other: MultiAxisIndexKey) -> MultiAxisIndexKey: ...

    @classmethod
    def from_vec(cls, vec: Vec[MultiAxisIndexKeyItem]) -> MultiAxisIndexKey: ...


MultiAxisIndexKeyLike: TypeAlias = "MultiAxisIndexKey | tuple[MultiAxisIndexKeyItemLike, ...] | TupleIntLike"


converter(
    tuple,
    MultiAxisIndexKey,
    lambda x: MultiAxisIndexKey.from_vec(Vec(*(convert(i, MultiAxisIndexKeyItem) for i in x))),
)
converter(
    TupleInt, MultiAxisIndexKey, lambda ti: MultiAxisIndexKey(ti.length(), lambda i: MultiAxisIndexKeyItem.int(ti[i]))
)


class IndexKey(Expr, ruleset=array_api_ruleset):
    """
    A key for indexing into an array

    https://data-apis.org/array-api/2022.12/API_specification/indexing.html

    It is equivalent to the following type signature:

    Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis, None], ...], array]
    """

    ELLIPSIS: ClassVar[IndexKey]

    @classmethod
    def int(cls, i: Int) -> IndexKey: ...

    @classmethod
    def slice(cls, slice: Slice) -> IndexKey: ...

    # Disabled until we support late binding
    # @classmethod
    # def boolean_array(cls, b: NDArray) -> IndexKey:
    #     ...

    @classmethod
    def multi_axis(cls, key: MultiAxisIndexKey) -> IndexKey: ...

    @classmethod
    def ndarray(cls, key: NDArray) -> IndexKey:
        """
        Indexes by a masked array
        """


IndexKeyLike: TypeAlias = "IndexKey | IntLike | SliceLike | MultiAxisIndexKeyLike | NDArrayLike"


converter(type(...), IndexKey, lambda _: IndexKey.ELLIPSIS)
converter(Int, IndexKey, lambda i: IndexKey.int(i))
converter(Slice, IndexKey, lambda s: IndexKey.slice(s))
converter(MultiAxisIndexKey, IndexKey, lambda m: IndexKey.multi_axis(m))


class Device(Expr, ruleset=array_api_ruleset): ...


ALL_INDICES: TupleInt = constant("ALL_INDICES", TupleInt)


class NDArray(Expr, ruleset=array_api_ruleset):
    """
    NDArray implementation following the Array API Standard.

    >>> NDArray.vector((1, 2, 3)).eval()
    (Value.int(Int(1)), Value.int(Int(2)), Value.int(Int(3)))
    >>> NDArray.vector((1, 2, 3)).eval_numpy("int64")
    array([1, 2, 3])
    >>> NDArray.matrix(((1, 2), (3, 4))).eval_numpy("int64")
    array([[1, 2],
           [3, 4]])
    """

    def __init__(self, shape: TupleIntLike, dtype: DType, idx_fn: Callable[[TupleInt], Value]) -> None: ...

    NEVER: ClassVar[NDArray]

    @method(cost=200)
    @classmethod
    def var(cls, name: StringLike) -> NDArray: ...

    @method(preserve=True)
    def __array_namespace__(self, api_version: object = None) -> ModuleType:
        return sys.modules[__name__]

    @property
    def ndim(self) -> Int: ...

    @property
    def dtype(self) -> DType: ...

    @property
    def device(self) -> Device: ...

    @property
    def shape(self) -> TupleInt: ...

    @method(preserve=True)
    def __bool__(self) -> bool:
        return self.to_value().to_bool.eval()

    @property
    def size(self) -> Int: ...

    @method(preserve=True)
    def __len__(self) -> int:
        return self.size.eval()

    @method(egg_fn="sum")
    def sum(self, axis: OptionalIntOrTupleLike = None) -> NDArray: ...

    @method(preserve=True)
    def __iter__(self) -> Iterator[NDArray]:
        for i in range(len(self)):
            yield self[IndexKey.int(Int(i))]

    def __getitem__(self, key: IndexKeyLike) -> NDArray: ...

    def __setitem__(self, key: IndexKeyLike, value: NDArray) -> None: ...

    def __lt__(self, other: NDArrayLike) -> NDArray: ...

    def __le__(self, other: NDArrayLike) -> NDArray: ...

    def __eq__(self, other: NDArrayLike) -> NDArray:  # type: ignore[override]
        ...

    # TODO: Add support for overloaded __ne__
    # def __ne__(self, other: NDArrayLike) -> NDArray:  # type: ignore[override]
    #     ...

    def __gt__(self, other: NDArrayLike) -> NDArray: ...

    def __ge__(self, other: NDArrayLike) -> NDArray: ...

    def __add__(self, other: NDArrayLike) -> NDArray: ...

    def __sub__(self, other: NDArrayLike) -> NDArray: ...

    def __mul__(self, other: NDArrayLike) -> NDArray: ...

    def __matmul__(self, other: NDArrayLike) -> NDArray: ...

    def __truediv__(self, other: NDArrayLike) -> NDArray: ...

    def __floordiv__(self, other: NDArrayLike) -> NDArray: ...

    def __mod__(self, other: NDArrayLike) -> NDArray: ...

    def __divmod__(self, other: NDArrayLike) -> NDArray: ...

    def __pow__(self, other: NDArrayLike) -> NDArray: ...

    def __lshift__(self, other: NDArrayLike) -> NDArray: ...

    def __rshift__(self, other: NDArrayLike) -> NDArray: ...

    def __and__(self, other: NDArrayLike) -> NDArray: ...

    def __xor__(self, other: NDArrayLike) -> NDArray: ...

    def __or__(self, other: NDArrayLike) -> NDArray: ...

    def __radd__(self, other: NDArray) -> NDArray: ...

    def __rsub__(self, other: NDArray) -> NDArray: ...

    def __rmul__(self, other: NDArray) -> NDArray: ...

    def __rmatmul__(self, other: NDArray) -> NDArray: ...

    def __rtruediv__(self, other: NDArray) -> NDArray: ...

    def __rfloordiv__(self, other: NDArray) -> NDArray: ...

    def __rmod__(self, other: NDArray) -> NDArray: ...

    def __rpow__(self, other: NDArray) -> NDArray: ...

    def __rlshift__(self, other: NDArray) -> NDArray: ...

    def __rrshift__(self, other: NDArray) -> NDArray: ...

    def __rand__(self, other: NDArray) -> NDArray: ...

    def __rxor__(self, other: NDArray) -> NDArray: ...

    def __ror__(self, other: NDArray) -> NDArray: ...

    def __abs__(self) -> NDArray: ...

    @classmethod
    def scalar(cls, value: ValueLike) -> NDArray:
        value = cast("Value", value)
        return NDArray(TupleInt(), value.dtype, lambda _: value)

    def to_value(self) -> Value:
        """
        Returns the value if this is a scalar.
        """

    def to_values(self) -> TupleValue:
        """
        Returns the value if this is a vector.
        """

    @property
    def T(self) -> NDArray:
        """
        https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.T.html#array_api.array.T
        """
        # Only works on 2D arrays
        return NDArray(
            (self.shape[1], self.shape[0]),
            self.dtype,
            lambda idx: self.index((idx[1], idx[0])),
        )

    @classmethod
    def vector(cls, values: TupleValueLike) -> NDArray:
        values = cast("TupleValue", values)
        return NDArray((values.length(),), values[0].dtype, lambda idx: values[idx[0]])

    @classmethod
    def matrix(cls, values: TupleTupleValueLike) -> NDArray:
        values = cast("TupleTupleValue", values)
        return NDArray(
            (values.length(), values[0].length()),
            values[0][0].dtype,
            lambda idx: values[idx[0]][idx[1]],
        )

    def index(self, indices: TupleIntLike) -> Value:
        """
        Return the value at the given indices.
        """

    @classmethod
    def if_(cls, b: BooleanLike, i: Callable[[], NDArray], j: Callable[[], NDArray]) -> NDArray: ...

    @method(preserve=True)
    def eval_vecs(self) -> VecValuesRecursive:
        """
        Evals to a nested Vec of values representing the array. It will be extracted and simplified by the e-graph.
        """
        # Share an e-graph for the computation of shape and eval
        egraph = _get_current_egraph()
        with set_array_api_egraph(egraph):
            shape = self.shape.eval()

        def _inner_values(current_index: tuple[int, ...], remaining_dims: tuple[Int, ...]) -> VecValuesRecursive:
            if not remaining_dims:
                return self.index(current_index)
            return Vec(*(_inner_values((*current_index, i), remaining_dims[1:]) for i in range(remaining_dims[0])))

        res = _inner_values((), shape)
        egraph.register(res)
        egraph.run(array_api_schedule)
        return egraph.extract(res)

    @method(preserve=True)
    def eval(self) -> PyTupleValuesRecursive:
        """
        Evals to a nested tuple of values representing the array.
        """

        def _to_tuple(v: VecValuesRecursive) -> PyTupleValuesRecursive:
            if isinstance(v, Value):
                return v
            return tuple(_to_tuple(i) for i in v)

        return _to_tuple(self.eval_vecs())

    @method(preserve=True)
    def eval_numpy(self, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Evals to a numpy ndarray.
        """
        return np.array(self.eval(), dtype=dtype)

    @method(preserve=True)
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if copy is False:
            msg = "NDArray.__array__ with copy=False is not supported"
            raise NotImplementedError(msg)
        return self.eval_numpy(dtype=dtype)


VecValuesRecursive: TypeAlias = "Value | Vec[VecValuesRecursive]"
PyTupleValuesRecursive: TypeAlias = Value | tuple["PyTupleValuesRecursive", ...]

NDArrayLike: TypeAlias = NDArray | ValueLike | TupleValueLike

converter(NDArray, IndexKey, lambda v: IndexKey.ndarray(v))
converter(Value, NDArray, lambda v: NDArray.scalar(v))
# Need this if we want to use ints in slices of arrays coming from 1d arrays, but make it more expensive
# to prefer upcasting in the other direction when we can, which is safer at runtime
converter(NDArray, Value, lambda n: n.to_value(), 100)
converter(TupleValue, NDArray, lambda v: NDArray.vector(v))
converter(TupleInt, TupleValue, lambda v: TupleValue.from_tuple_int(v))


@array_api_ruleset.register
def _ndarray(
    x: NDArray,
    x1: NDArray,
    shape: TupleInt,
    dtype: DType,
    idx_fn: Callable[[TupleInt], Value],
    idx: TupleInt,
    tv: TupleValue,
    v: Value,
    v1: Value,
    vi: Vec[Int],
    xt: Callable[[], NDArray],
    x1t: Callable[[], NDArray],
):
    return [
        rewrite(NDArray(shape, dtype, idx_fn).shape).to(shape),
        rewrite(NDArray(shape, dtype, idx_fn).dtype).to(dtype),
        rewrite(NDArray(shape, dtype, idx_fn).index(idx), subsume=True).to(idx_fn(idx)),
        rewrite(x.ndim).to(x.shape.length()),
        # rewrite(NDArray.scalar(Value.bool(b)).to_bool()).to(b),
        # Converting to a value requires a scalar bool value
        rewrite(x.to_value()).to(x.index(TupleInt())),
        rewrite(NDArray.vector(tv).to_values()).to(tv),
        rewrite(NDArray(TupleInt(vi), dtype, idx_fn).to_values(), subsume=True).to(
            TupleValue.fn(vi[0], lambda i: idx_fn(TupleInt([i]))),
            vi.length() == i64(1),
        ),
        # TODO: Push these down to float
        rewrite(NDArray.scalar(v) / NDArray.scalar(v)).to(NDArray.scalar(v / v)),
        rewrite(NDArray.scalar(v) + NDArray.scalar(v)).to(NDArray.scalar(v + v)),
        rewrite(NDArray.scalar(v) * NDArray.scalar(v)).to(NDArray.scalar(v * v)),
        # -
        rewrite(NDArray.scalar(v) - NDArray.scalar(v)).to(NDArray.scalar(v - v)),
        # Comparisons
        rewrite(NDArray.scalar(v) < NDArray.scalar(v1)).to(NDArray.scalar(v < v1)),
        rewrite(NDArray.scalar(v) <= NDArray.scalar(v1)).to(NDArray.scalar(v <= v1)),
        rewrite(NDArray.scalar(v) == NDArray.scalar(v1)).to(NDArray.scalar(v == v1)),
        rewrite(NDArray.scalar(v) > NDArray.scalar(v1)).to(NDArray.scalar(v > v1)),
        rewrite(NDArray.scalar(v) >= NDArray.scalar(v1)).to(NDArray.scalar(v >= v1)),
        # Transpose of transpose is the original array
        rewrite(x.T.T).to(x),
        # rewrite(reshape(x, shape).T, subsume=True).to(reshape(x, shape.reverse())),
        # rewrite(reshape(x, shape).shape).to(shape),
        # if_
        rewrite(NDArray.if_(TRUE, xt, x1t)).to(xt()),
        rewrite(NDArray.if_(FALSE, xt, x1t)).to(x1t()),
        # convert to scalar
        rewrite(NDArray(TupleInt(), dtype, idx_fn)).to(NDArray.scalar(idx_fn(TupleInt()))),
        # Scalars should push operators down
        rewrite(NDArray.scalar(v) / NDArray.scalar(v1)).to(NDArray.scalar(v / v1)),
        rewrite(NDArray.scalar(v) ** NDArray.scalar(v1)).to(NDArray.scalar(v**v1)),
    ]


class TupleNDArray(Expr, ruleset=array_api_ruleset):
    def __init__(self, vec: VecLike[NDArray, NDArrayLike] = ()) -> None: ...
    @classmethod
    def fn(cls, length: IntLike, idx_fn: Callable[[Int], NDArray]) -> TupleNDArray: ...
    def length(self) -> Int: ...
    def __getitem__(self, i: IntLike) -> NDArray: ...
    @method(preserve=True)
    def __len__(self) -> int:
        return self.length().eval()

    @method(preserve=True)
    def __iter__(self) -> Iterator[NDArray]:
        return iter(self.eval())

    @method(merge=Vec.__or__)  # type: ignore[prop-decorator]
    @property
    def to_vec(self) -> Vec[NDArray]: ...
    @method(preserve=True)
    def eval(self) -> tuple[NDArray, ...]:
        return try_evaling(_get_current_egraph(), array_api_schedule, self, self.to_vec)

    def append(self, i: NDArrayLike) -> TupleNDArray:
        return TupleNDArray.fn(
            self.length() + 1, lambda j: NDArray.if_(j == self.length(), lambda: cast("NDArray", i), lambda: self[j])
        )

    def __add__(self, other: TupleValueLike) -> TupleNDArray:
        other = cast("TupleNDArray", other)
        return TupleNDArray.fn(
            self.length() + other.length(),
            lambda i: NDArray.if_(i < self.length(), lambda: self[i], lambda: other[i - self.length()]),
        )


converter(Vec[NDArray], TupleNDArray, lambda x: TupleNDArray(x))

TupleNDArrayLike: TypeAlias = TupleNDArray | VecLike[NDArray, NDArrayLike]


@array_api_ruleset.register
def _tuple_ndarray(
    i: Int,
    i2: Int,
    idx_fn: Callable[[Int], NDArray],
    vs: Vec[NDArray],
    ti: TupleNDArray,
    k: i64,
    lt: Callable[[], TupleNDArray],
    lf: Callable[[], TupleNDArray],
):
    yield rule(eq(ti).to(TupleNDArray(vs))).then(set_(ti.to_vec).to(vs))

    yield rewrite(TupleNDArray.fn(i2, idx_fn).length()).to(i2)
    yield rewrite(TupleNDArray.fn(i2, idx_fn)[i]).to(idx_fn(check_index(i, i2)))

    yield rewrite(TupleNDArray(vs).length()).to(Int(vs.length()))
    yield rewrite(TupleNDArray(vs)[Int(k)]).to(vs[k], k >= 0, k < vs.length())

    yield rewrite(TupleNDArray.fn(Int(k), idx_fn), subsume=True).to(
        TupleNDArray(k.range().map(lambda i: idx_fn(Int(i))))
    )


class OptionalBool(Expr, ruleset=array_api_ruleset):
    none: ClassVar[OptionalBool]

    @classmethod
    def some(cls, value: Boolean) -> OptionalBool: ...


converter(type(None), OptionalBool, lambda _: OptionalBool.none)
converter(Boolean, OptionalBool, lambda x: OptionalBool.some(x))


class OptionalDType(Expr, ruleset=array_api_ruleset):
    none: ClassVar[OptionalDType]

    @classmethod
    def some(cls, value: DType) -> OptionalDType: ...


converter(type(None), OptionalDType, lambda _: OptionalDType.none)
converter(DType, OptionalDType, lambda x: OptionalDType.some(x))


class OptionalDevice(Expr, ruleset=array_api_ruleset):
    none: ClassVar[OptionalDevice]

    @classmethod
    def some(cls, value: Device) -> OptionalDevice: ...


converter(type(None), OptionalDevice, lambda _: OptionalDevice.none)
converter(Device, OptionalDevice, lambda x: OptionalDevice.some(x))


class OptionalTupleInt(Expr, ruleset=array_api_ruleset):
    none: ClassVar[OptionalTupleInt]

    @classmethod
    def some(cls, value: TupleIntLike) -> OptionalTupleInt: ...


converter(type(None), OptionalTupleInt, lambda _: OptionalTupleInt.none)
converter(TupleInt, OptionalTupleInt, lambda x: OptionalTupleInt.some(x))


class OptionalIntOrTuple(Expr, ruleset=array_api_ruleset):
    none: ClassVar[OptionalIntOrTuple]

    @classmethod
    def int(cls, value: Int) -> OptionalIntOrTuple: ...

    @classmethod
    def tuple(cls, value: TupleIntLike) -> OptionalIntOrTuple: ...


OptionalIntOrTupleLike: TypeAlias = OptionalIntOrTuple | None | IntLike | TupleIntLike

converter(type(None), OptionalIntOrTuple, lambda _: OptionalIntOrTuple.none)
converter(Int, OptionalIntOrTuple, lambda v: OptionalIntOrTuple.int(v))
converter(TupleInt, OptionalIntOrTuple, lambda v: OptionalIntOrTuple.tuple(v))


@function
def asarray(
    a: NDArray,
    dtype: OptionalDType = OptionalDType.none,
    copy: OptionalBool = OptionalBool.none,
    device: OptionalDevice = OptionalDevice.none,
) -> NDArray: ...


@array_api_ruleset.register
def _asarray(a: NDArray, d: OptionalDType, ob: OptionalBool):
    yield rewrite(asarray(a, d, ob).ndim).to(a.ndim)  # asarray doesn't change ndim
    yield rewrite(asarray(a)).to(a)


@function
def isfinite(x: NDArray) -> NDArray: ...


@function
def sum(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray:
    """
    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.sum.html?highlight=sum
    """


@array_api_ruleset.register
def _sum(x: NDArray, y: NDArray, v: Value, dtype: DType):
    return [
        rewrite(sum(x / NDArray.scalar(v))).to(sum(x) / NDArray.scalar(v)),
        # Sum of 0D array is
    ]


@function
def reshape(x: NDArray, shape: TupleIntLike, copy: OptionalBool = OptionalBool.none) -> NDArray: ...


# @function
# def reshape_transform_index(original_shape: TupleInt, shape: TupleInt, index: TupleInt) -> TupleInt:
#     """
#     Transforms an indexing operation on a reshaped array to an indexing operation on the original array.
#     """
#     ...


# @function
# def reshape_transform_shape(original_shape: TupleInt, shape: TupleInt) -> TupleInt:
#     """
#     Transforms the shape of an array to one that is reshaped, by replacing -1 with the correct value.
#     """
#     ...


# @array_api_ruleset.register
# def _reshape(
#     x: NDArray,
#     y: NDArray,
#     shape: TupleInt,
#     copy: OptionalBool,
#     i: Int,
#     s: String,
#     ix: TupleInt,
# ):
#     return [
#         # dtype of result is same as input
#         rewrite(reshape(x, shape, copy).dtype).to(x.dtype),
#         # Indexing into a reshaped array is the same as indexing into the original array with a transformed index
#         rewrite(reshape(x, shape, copy).index(ix)).to(x.index(reshape_transform_index(x.shape, shape, ix))),
#         rewrite(reshape(x, shape, copy).shape).to(reshape_transform_shape(x.shape, shape)),
#         # reshape_transform_shape recursively
#         # TODO: handle all cases
#         rewrite(reshape_transform_shape(TupleInt(i), TupleInt(Int(-1)))).to(TupleInt(i)),
#     ]


@function
def unique_values(x: NDArrayLike) -> NDArray:
    """
    Returns the unique elements of an input array x flattened with arbitrary ordering.

    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.unique_values.html
    """


@array_api_ruleset.register
def _unique_values(x: NDArray):
    return [
        rewrite(unique_values(unique_values(x))).to(unique_values(x)),
    ]


@function
def concat(arrays: TupleNDArrayLike, axis: OptionalInt = OptionalInt.none) -> NDArray: ...


@array_api_ruleset.register
def _concat(vs: Vec[NDArray]):
    return [
        # only support no-op concat for now
        rewrite(concat(TupleNDArray(vs))).to(vs[0], vs.length() == i64(1)),
    ]


@function
def astype(x: NDArray, dtype: DType) -> NDArray: ...


@array_api_ruleset.register
def _astype(x: NDArray, dtype: DType, i: i64):
    return [
        rewrite(astype(x, dtype).dtype).to(dtype),
        rewrite(astype(NDArray.scalar(Value.int(Int(i))), float64)).to(
            NDArray.scalar(Value.float(Float(f64.from_i64(i))))
        ),
    ]


@function
def unique_counts(x: NDArray) -> TupleNDArray:
    """
    Returns the unique elements of an input array x and the corresponding counts for each unique element in x.


    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.unique_counts.html
    """


@array_api_ruleset.register
def _unique_counts(x: NDArray, c: NDArray, tv: TupleValue, v: Value, dtype: DType):
    return [
        # rewrite(unique_counts(x).length()).to(Int(2)),
        rewrite(unique_counts(x)).to(TupleNDArray.fn(2, unique_counts(x).__getitem__)),
        # Sum of all unique counts is the size of the array
        rewrite(sum(unique_counts(x)[Int(1)])).to(NDArray.scalar(Value.int(x.size))),
        # Same but with astype in the middle
        # TODO: Replace
        rewrite(sum(astype(unique_counts(x)[Int(1)], dtype))).to(astype(NDArray.scalar(Value.int(x.size)), dtype)),
    ]


@function
def square(x: NDArray) -> NDArray: ...


@function
def any(x: NDArray) -> NDArray: ...


@function(egg_fn="ndarray-abs")
def abs(x: NDArray) -> NDArray: ...


@function(egg_fn="ndarray-log")
def log(x: NDArray) -> NDArray: ...


@array_api_ruleset.register
def _abs(f: Float):
    return [
        rewrite(abs(NDArray.scalar(Value.float(f)))).to(NDArray.scalar(Value.float(f.abs()))),
    ]


@function
def unique_inverse(x: NDArray) -> TupleNDArray:
    """
    Returns the unique elements of an input array x and the indices from the set of unique elements that reconstruct x.

    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.unique_inverse.html
    """


@array_api_ruleset.register
def _unique_inverse(x: NDArray, i: Int):
    return [
        # rewrite(unique_inverse(x).length()).to(Int(2)),
        rewrite(unique_inverse(x)).to(TupleNDArray.fn(2, unique_inverse(x).__getitem__)),
        # Shape of unique_inverse first element is same as shape of unique_values
        rewrite(unique_inverse(x)[Int(0)]).to(unique_values(x)),
    ]


@function
def zeros(
    shape: TupleIntLike, dtype: OptionalDType = OptionalDType.none, device: OptionalDevice = OptionalDevice.none
) -> NDArray: ...


@function
def expand_dims(x: NDArray, axis: Int = Int(0)) -> NDArray: ...


@function
def mean(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none, keepdims: Boolean = FALSE) -> NDArray: ...


# TODO: Possibly change names to include modules.
@function(egg_fn="ndarray-sqrt")
def sqrt(x: NDArray) -> NDArray: ...


@function
def std(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray: ...


@function
def real(x: NDArray) -> NDArray: ...


@function
def conj(x: NDArray) -> NDArray: ...


@function(ruleset=array_api_ruleset, subsume=True)
def vecdot(x1: NDArrayLike, x2: NDArrayLike) -> NDArray:
    """
    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.vecdot.html
    https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html

    TODO: Support axis, complex numbers, broadcasting, and more than matrix-vector

    >>> v = NDArray.matrix([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
    >>> n = NDArray.vector([0., 0.6, 0.8])
    >>> vecdot(v, n).eval_numpy("float64")
    array([ 3.,  8., 10.])
    """
    x1 = cast("NDArray", x1)
    x2 = cast("NDArray", x2)

    return NDArray(
        x1.shape.drop_last(),
        x1.dtype,
        lambda idx: (
            TupleInt.range(x1.shape.last())
            .map_value(lambda i: x1.index(idx.append(i)) * x2.index((i,)))
            .foldl_value(Value.__add__, Value.float(0))
        ),
    )


@function(ruleset=array_api_ruleset, subsume=True)
def vector_norm(x: NDArrayLike) -> NDArray:
    """
    https://data-apis.org/array-api/2022.12/extensions/generated/array_api.linalg.vector_norm.html
    TODO: support axis
    # >>> x = NDArray.vector([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # >>> vector_norm(x).eval_numpy("float64")
    # array(16.88194302)
    """
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm
    # sum(abs(x)**ord)**(1./ord) where ord=2
    x = cast("NDArray", x)
    # Only works on vectors
    return NDArray.scalar(
        x.to_values()
        .map_value(lambda v: v.__abs__() ** Value.float(Float(2.0)))
        .foldl_value(Value.__add__, Value.float(0))
        ** Value.float(Float(0.5))
    )


@function(ruleset=array_api_ruleset, subsume=True)
def cross(a: NDArrayLike, b: NDArrayLike) -> NDArray:
    """
    https://data-apis.org/array-api/2022.12/extensions/generated/array_api.linalg.cross.html
    TODO: support axis, and more than two vecs

    >>> x = NDArray.vector([1, 2, 3])
    >>> y = NDArray.vector([4, 5, 6])
    >>> cross(x, y).eval_numpy("int64")
    array([-3,  6, -3])
    """
    a = cast("NDArray", a)
    b = cast("NDArray", b)
    return NDArray(
        (3,),
        a.dtype,
        lambda idx: (
            (a.index(((idx[0] + 1) % 3,)) * b.index(((idx[0] + 2) % 3,)))
            - (a.index(((idx[0] + 2) % 3,)) * b.index(((idx[0] + 1) % 3,)))
        ),
    )


linalg = sys.modules[__name__]


@function
def svd(x: NDArray, full_matrices: Boolean = TRUE) -> TupleNDArray:
    """
    https://data-apis.org/array-api/2022.12/extensions/generated/array_api.linalg.svd.html
    """


@array_api_ruleset.register
def _linalg(x: NDArray, full_matrices: Boolean):
    return [
        # rewrite(svd(x, full_matrices).length()).to(Int(3)),
        rewrite(svd(x, full_matrices)).to(TupleNDArray.fn(3, svd(x, full_matrices).__getitem__)),
    ]


@function(ruleset=array_api_ruleset)
def ndindex(shape: TupleIntLike) -> TupleTupleInt:
    """
    https://numpy.org/doc/stable/reference/generated/numpy.ndindex.html
    """
    shape = cast("TupleInt", shape)
    return shape.map_tuple_int(TupleInt.range).product()


##
# Interval analysis
#
# to analyze `any(((astype(unique_counts(NDArray.var("y"))[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0))) < NDArray.scalar(Value.int(Int(0)))).bool()``
##

greater_zero = relation("greater_zero", Value)


# @function
# def ndarray_all_greater_0(x: NDArray) -> Unit:
#     ...


# @function
# def ndarray_all_false(x: NDArray) -> Unit:
#     ...


# @function
# def ndarray_all_true(x: NDArray) -> Unit:
#     ...


# any((astype(unique_counts(_NDArray_1)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))) < NDArray.scalar(Value.int(Int(0)))).to_bool()

# sum(astype(unique_counts(_NDArray_1)[Int(1)], DType.float64) / NDArray.scalar(Value.int(Int(150))))
# And also

# def


@function
def broadcast_index(from_shape: TupleIntLike, to_shape: TupleIntLike, index: TupleIntLike) -> TupleInt:
    """
    Returns the index in the original array of the given index in the broadcasted array.
    """


@function
def broadcast_shapes(shape1: TupleIntLike, shape2: TupleIntLike) -> TupleInt:
    """
    Returns the shape of the broadcasted array.
    """


@array_api_ruleset.register
def _interval_analaysis(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    dtype: DType,
    f: f64,
    i: i64,
    b: Boolean,
    idx: TupleInt,
    v: Value,
    v1: Value,
    v2: Value,
    float_: Float,
    int_: Int,
):
    res_shape = broadcast_shapes(x.shape, y.shape)
    x_value = x.index(broadcast_index(x.shape, res_shape, idx))
    y_value = y.index(broadcast_index(y.shape, res_shape, idx))
    return [
        # Calling any on an array gives back a sclar, which is true if any of the values are truthy
        rewrite(any(x)).to(
            NDArray.scalar(Value.bool(possible_values(x.index(ALL_INDICES).to_truthy_value).contains(Value.bool(TRUE))))
        ),
        # Indexing x < y is the same as broadcasting the index and then indexing both and then comparing
        rewrite((x < y).index(idx)).to(x_value < y_value),
        # Same for x / y
        rewrite((x / y).index(idx)).to(x_value / y_value),
        # Indexing a scalar is the same as the scalar
        rewrite(NDArray.scalar(v).index(idx)).to(v),
        # Indexing of astype is same as astype of indexing
        rewrite(astype(x, dtype).index(idx)).to(x.index(idx).astype(dtype)),
        # rule(eq(y).to(x < NDArray.scalar(Value.int(Int(0)))), ndarray_all_greater_0(x)).then(ndarray_all_false(y)),
        # rule(eq(y).to(any(x)), ndarray_all_false(x)).then(union(y).with_(NDArray.scalar(Value.bool(FALSE)))),
        # Indexing into unique counts counts are all positive
        rule(
            eq(v).to(unique_counts(x)[Int(1)].index(idx)),
        ).then(greater_zero(v)),
        # Min value preserved over astype
        rule(
            greater_zero(v),
            eq(v1).to(v.astype(dtype)),
        ).then(
            greater_zero(v1),
        ),
        # Min value of scalar is scalar itself
        rule(eq(v).to(Value.float(Float(f))), f > 0.0).then(greater_zero(v)),
        rule(eq(v).to(Value.int(Int(i))), i > 0).then(greater_zero(v)),
        # If we have divison of v and v1, and both greater than zero, then the result is greater than zero
        rule(
            greater_zero(v),
            greater_zero(v1),
            eq(v2).to(v / v1),
        ).then(
            greater_zero(v2),
        ),
        # Define v < 0 to be false, if greater_zero(v)
        rule(
            greater_zero(v),
            eq(v1).to(v < Value.int(Int(0))),
        ).then(
            union(v1).with_(Value.bool(FALSE)),
        ),
        # possible values of bool is bool
        rewrite(possible_values(Value.bool(b))).to(TupleValue([Value.bool(b)])),
        # casting to a type preserves if > 0
        rule(
            eq(v1).to(v.astype(dtype)),
            greater_zero(v),
        ).then(
            greater_zero(v1),
        ),
    ]


##
# Mathematical descriptions of arrays as:
# 1. A shape `.shape`
# 2. A dtype `.dtype`
# 3. A mapping from indices to values `x.index(idx)`
#
# For all operations that are supported mathematically, define each of the above.
##


def _demand_shape(compound: NDArray, inner: NDArray) -> Command:
    __a = var("__a", NDArray)
    return rule(eq(__a).to(compound)).then(inner.shape, inner.shape.length())


@array_api_ruleset.register
def _scalar_math(v: Value, vs: TupleValue, i: Int):
    yield rewrite(NDArray.scalar(v).shape).to(TupleInt())
    yield rewrite(NDArray.scalar(v).dtype).to(v.dtype)
    yield rewrite(NDArray.scalar(v).index(TupleInt())).to(v)


@array_api_ruleset.register
def _vector_math(v: Value, vs: TupleValue, ti: TupleInt):
    yield rewrite(NDArray.vector(vs).shape).to(TupleInt([vs.length()]))
    yield rewrite(NDArray.vector(vs).dtype).to(vs[Int(0)].dtype)
    yield rewrite(NDArray.vector(vs).index(ti)).to(vs[ti[0]])


@array_api_ruleset.register
def _reshape_math(x: NDArray, shape: TupleInt, copy: OptionalBool):
    res = reshape(x, shape, copy)

    yield _demand_shape(res, x)
    # Demand shape length and index
    yield rule(res).then(shape.length(), shape[0])

    # Reshaping a vec to a vec is the same as the vec
    yield rewrite(res).to(
        x,
        eq(x.shape.length()).to(Int(1)),
        eq(shape.length()).to(Int(1)),
        eq(shape[0]).to(Int(-1)),
    )


@array_api_ruleset.register
def _indexing_pushdown(x: NDArray, shape: TupleInt, copy: OptionalBool, i: Int):
    # rewrite full getitem to indexec
    yield rewrite(x[IndexKey.int(i)]).to(NDArray.scalar(x.index(TupleInt([i]))))
    # TODO: Multi index rewrite as well if all are ints


##
# Assumptions
##


@function(mutates_first_arg=True)
def assume_dtype(x: NDArray, dtype: DType) -> None:
    """
    Asserts that the dtype of x is dtype.
    """


@array_api_ruleset.register
def _assume_dtype(x: NDArray, dtype: DType, idx: TupleInt):
    orig_x = copy(x)
    assume_dtype(x, dtype)
    yield rewrite(x.dtype).to(dtype)
    yield rewrite(x.shape).to(orig_x.shape)
    yield rewrite(x.index(idx)).to(orig_x.index(idx))


@function(mutates_first_arg=True)
def assume_shape(x: NDArray, shape: TupleIntLike) -> None:
    """
    Asserts that the shape of x is shape.
    """


@array_api_ruleset.register
def _assume_shape(x: NDArray, shape: TupleInt, idx: TupleInt):
    orig_x = copy(x)
    assume_shape(x, shape)
    yield rewrite(x.shape).to(shape)
    yield rewrite(x.dtype).to(orig_x.dtype)
    yield rewrite(x.index(idx)).to(orig_x.index(idx))


@function(mutates_first_arg=True)
def assume_isfinite(x: NDArray) -> None:
    """
    Asserts that the scalar ndarray is non null and not infinite.
    """


@array_api_ruleset.register
def _isfinite(x: NDArray, ti: TupleInt):
    orig_x = copy(x)
    assume_isfinite(x)

    # pass through getitem, shape, index
    yield rewrite(x.shape).to(orig_x.shape)
    yield rewrite(x.dtype).to(orig_x.dtype)
    yield rewrite(x.index(ti)).to(orig_x.index(ti))
    # But say that any indixed value is finite
    yield rewrite(x.index(ti).isfinite()).to(TRUE)


@function(mutates_first_arg=True)
def assume_value_one_of(x: NDArray, values: TupleValueLike) -> None:
    """
    A value that is one of the values in the tuple.
    """


@array_api_ruleset.register
def _assume_value_one_of(x: NDArray, v: Value, vs: TupleValue, idx: TupleInt):
    x_orig = copy(x)
    assume_value_one_of(x, vs)
    # Pass through dtype and shape
    yield rewrite(x.shape).to(x_orig.shape)
    yield rewrite(x.dtype).to(x_orig.dtype)
    # The array vales passes through, but say that the possible_values are one of the values
    yield rule(eq(v).to(x.index(idx))).then(
        union(v).with_(x_orig.index(idx)),
        union(possible_values(v)).with_(vs),
    )


@array_api_ruleset.register
def _ndarray_value_isfinite(arr: NDArray, x: Value, xs: TupleValue, i: Int, f: f64, b: Boolean):
    yield rewrite(Value.int(i).isfinite()).to(TRUE)
    yield rewrite(Value.bool(b).isfinite()).to(TRUE)
    yield rewrite(Value.float(Float(f)).isfinite()).to(TRUE, ne(f).to(f64(math.nan)))

    # a sum of an array is finite if all the values are finite
    yield rewrite(isfinite(sum(arr))).to(NDArray.scalar(Value.bool(arr.index(ALL_INDICES).isfinite())))


@array_api_ruleset.register
def _unique(xs: TupleValue, a: NDArray, shape: TupleInt, copy: OptionalBool):
    yield rewrite(unique_values(x=a)).to(NDArray.vector(possible_values(a.index(ALL_INDICES))))
    # yield rewrite(
    #     possible_values(reshape(a.index(shape, copy), ALL_INDICES)),
    # ).to(possible_values(a.index(ALL_INDICES)))


@array_api_ruleset.register
def _size(x: NDArray):
    yield rewrite(x.size).to(x.shape.foldl(Int.__mul__, Int(1)))


@function(ruleset=array_api_ruleset)
def ravel_index(index: TupleIntLike, shape: TupleIntLike) -> Int:
    """
    Convert a multi-dimensional index to a flat index.

    https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index

    >>> int(ravel_index((3, 4), (7, 6)))
    22
    >>> int(ravel_index((6, 5), (7, 6)))
    41
    >>> int(ravel_index((6, 1), (7, 6)))
    37
    >>> int(ravel_index((3, 1, 4, 1), (6, 7, 8, 9)))
    1621
    """
    index = cast("TupleInt", index)
    shape = cast("TupleInt", shape)

    return TupleInt.range(shape.length()).foldl(lambda res, i: res * shape[i] + index[i], Int(0))


@function(ruleset=array_api_ruleset)
def unravel_index(flat_index: IntLike, shape: TupleIntLike) -> TupleInt:
    """
    Convert a flat index to a multi-dimensional index.

    https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html

    >>> tuple(map(int, unravel_index(22, (7, 6))))
    (3, 4)
    >>> tuple(map(int, unravel_index(41, (7, 6))))
    (6, 5)
    >>> tuple(map(int, unravel_index(37, (7, 6))))
    (6, 1)
    >>> tuple(map(int, unravel_index(1621, (6, 7, 8, 9))))
    (3, 1, 4, 1)
    """
    shape = cast("TupleInt", shape)

    return (
        shape.reverse()
        .foldl_tuple_int(
            # Store remainder as last item in accumulator
            lambda acc, dim: acc.drop_last().append((r := acc.last()) % dim).append(r // dim),
            TupleInt([flat_index]),
        )
        .drop_last()
        .reverse()
    )


@ruleset
def array_api_functional_ruleset(
    old_shape: TupleInt,
    shape: TupleInt,
    ob: OptionalBool,
    dtype: DType,
    idx_fn: Callable[[TupleInt], Value],
):
    # TODO: Support -1 in shape like numpy does
    yield rewrite(reshape(NDArray(old_shape, dtype, idx_fn), shape, ob), subsume=True).to(
        NDArray(shape, dtype, lambda idx: idx_fn(unravel_index(ravel_index(idx, shape), old_shape)))
    )


array_api_combined_ruleset = array_api_ruleset | array_api_functional_ruleset
array_api_schedule = (array_api_combined_ruleset + run()).saturate()

_CURRENT_EGRAPH: None | EGraph = None


@contextlib.contextmanager
def set_array_api_egraph(egraph: EGraph) -> Iterator[None]:
    """
    Context manager that will set the current egraph. It will be set back after.
    """
    global _CURRENT_EGRAPH
    assert _CURRENT_EGRAPH is None
    _CURRENT_EGRAPH = egraph
    yield
    _CURRENT_EGRAPH = None


def _get_current_egraph() -> EGraph:
    return _CURRENT_EGRAPH or EGraph(save_egglog_string=True)


def try_evaling(egraph: EGraph, schedule: Schedule, expr: Expr, prim_expr: BuiltinExpr) -> Any:
    """
    Try evaluating an expression that should produce a primitive (e.g., Bool/i64).
    If extraction fails, register the expr, run the schedule, and retry.
    On egglog panics we dump the .egg program for debugging.
    A common failure mode is that no rule ever sets the primitive output
    (e.g., `Boolean.to_bool` / `Int.to_i64`), so extraction fails.
    """
    try:
        return egraph.extract(prim_expr).value  # type: ignore[attr-defined]
    except EggSmolError:
        pass
    # If this primitive doesn't exist in the egraph, we need to try to create it by
    # registering the expression and running the schedule
    egraph.register(expr)
    try:
        egraph.run(schedule)
    except EggSmolError as e:
        # Write out the egraph for debugging
        with NamedTemporaryFile(mode="w", suffix=".egg", delete=False) as f:
            f.write(egraph.as_egglog_string)
            e.add_note(f"EGraph written to {f.name} for debugging")
        raise
    try:
        return egraph.extract(prim_expr).value  # type: ignore[attr-defined]
    except BaseException as e:
        # egraph.display(n_inline_leaves=1, split_primitive_outputs=True)
        e.add_note(f"Cannot evaluate {egraph.extract(expr)}")
        raise


##
# Polynomials
##


@function
def monomial(x: MultiSetLike[Value, ValueLike]) -> Value: ...


@function(merge=lambda old, new: new)
def get_monomial(x: Value) -> MultiSet[Value]:
    """
    Only defined on monomials:

        get_monomial(monomial(xs)) => xs
    """


@function(merge=lambda old, new: new)
def get_sole_polynomial(xs: MultiSet[Value]) -> MultiSet[MultiSet[Value]]:
    """
    Only defined on monomials that contain a single polynomial:

        get_sole_polynomial(MultiSet(polynomial(xs))) => xs
    """


@function
def polynomial(x: MultiSetLike[MultiSet[Value], MultiSetLike[Value, ValueLike]]) -> Value: ...


@ruleset
def to_polynomial_ruleset(
    n1: Value,
    n2: Value,
    n3: Value,
    mss: MultiSet[MultiSet[Value]],
    mss1: MultiSet[MultiSet[Value]],
):
    yield rewrite(n1 - n2, subsume=True).to(n1 + (Value.int(-1) * n2))
    yield rule(
        eq(n3).to(n1 + n2),
        name="add",
    ).then(
        union(n3).with_(polynomial(MultiSet(MultiSet(n1), MultiSet(n2)))),
        set_(get_sole_polynomial(MultiSet(n3))).to(MultiSet(MultiSet(n1), MultiSet(n2))),
        delete(n1 + n2),
    )
    yield rule(
        eq(n3).to(n1 * n2),
        name="mul",
    ).then(
        union(n3).with_(monomial(MultiSet(n1, n2))),
        set_(get_monomial(n3)).to(MultiSet(n1, n2)),
        delete(n1 * n2),
        # MultiSet(n1, n2).fill_index(ms_index),
    )
    yield rule(
        eq(n1).to(polynomial(mss)),
        mss1 == mss.map(partial(multiset_flat_map, get_monomial)),
        mss != mss1,
        name="unwrap monomial",
    ).then(
        union(n1).with_(polynomial(mss1)),
        delete(polynomial(mss)),
        set_(get_sole_polynomial(MultiSet(n1))).to(mss1),
    )
    yield rule(
        eq(n1).to(polynomial(mss)),
        mss1 == multiset_flat_map(UnstableFn(get_sole_polynomial), mss),
        mss != mss1,
        name="unwrap polynomial",
    ).then(
        union(n1).with_(polynomial(mss1)),
        delete(polynomial(mss)),
        set_(get_sole_polynomial(MultiSet(n1))).to(mss1),
    )


@ruleset
def factor_ruleset(
    n: Value,
    mss: MultiSet[MultiSet[Value]],
    counts: MultiSet[Value],
    factor: Value,
    divided: MultiSet[MultiSet[Value]],
    remainder: MultiSet[MultiSet[Value]],
):
    yield rule(
        eq(n).to(polynomial(mss)),
        # Find factor that shows up in most monomials, at least two of them
        counts == MultiSet.sum_multisets(mss.map(MultiSet.reset_counts)),
        eq(factor).to(counts.pick_max()),
        # Only factor out if it term appears in more than one monomial
        counts.count(factor) > 1,
        # map, including only those factor that contain the factor, removing them from that
        # other items are omitted from the map
        divided == mss.map(partial(multiset_remove_swapped, factor)),
        # remainder is those monomials that do not contain the factor
        remainder == mss.filter(partial(multiset_not_contains_swapped, factor)),
        name="factor",
    ).then(
        union(n).with_(polynomial(MultiSet(MultiSet(factor, polynomial(divided))) + remainder)),
        # delete(polynomial(mss)),
    )


@ruleset
def from_polynomial_ruleset(mss: MultiSet[MultiSet[Value]]):
    mul: Callable[[Value, Value], Value] = Value.__mul__
    yield rewrite(polynomial(mss)).to(
        multiset_fold(
            Value.__add__,
            Value.int(0),
            mss.map(
                partial(multiset_fold, mul, Value.int(1)),
            ),
        )
    )


polynomial_schedule = to_polynomial_ruleset.saturate() + factor_ruleset.saturate()
