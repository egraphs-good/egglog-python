# mypy: disable-error-code="empty-body"

from __future__ import annotations

import itertools
import math
import numbers
import sys
from copy import copy
from typing import TYPE_CHECKING, ClassVar, overload

import numpy as np

from egglog import *
from egglog.bindings import EggSmolError
from egglog.runtime import RuntimeExpr

from .program_gen import *

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

# Pretend that exprs are numbers b/c sklearn does isinstance checks
numbers.Integral.register(RuntimeExpr)


array_api_module = Module()


@array_api_module.class_
class Boolean(Expr):
    @array_api_module.method(preserve=True)
    def __bool__(self) -> bool:
        return try_evaling(self, self.bool)

    @property
    def bool(self) -> Bool:
        ...

    def __or__(self, other: Boolean) -> Boolean:
        ...

    def __and__(self, other: Boolean) -> Boolean:
        ...


TRUE = array_api_module.constant("TRUE", Boolean)
FALSE = array_api_module.constant("FALSE", Boolean)
converter(bool, Boolean, lambda x: TRUE if x else FALSE)


@array_api_module.register
def _bool(x: Boolean):
    return [
        rule(eq(x).to(TRUE)).then(set_(x.bool).to(Bool(True))),
        rule(eq(x).to(FALSE)).then(set_(x.bool).to(Bool(False))),
        rewrite(TRUE | x).to(TRUE),
        rewrite(FALSE | x).to(x),
        rewrite(TRUE & x).to(x),
        rewrite(FALSE & x).to(FALSE),
    ]


@array_api_module.class_
class DType(Expr):
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
converter(type(np.dtype), DType, lambda x: getattr(DType, x.name))  # type: ignore[call-overload]
array_api_module.register(
    *(
        rewrite(l == r).to(TRUE if expr_parts(l) == expr_parts(r) else FALSE)
        for l, r in itertools.product(_DTYPES, repeat=2)
    )
)


@array_api_module.class_
class IsDtypeKind(Expr):
    NULL: ClassVar[IsDtypeKind]

    @classmethod
    def string(cls, s: StringLike) -> IsDtypeKind:
        ...

    @classmethod
    def dtype(cls, d: DType) -> IsDtypeKind:
        ...

    @array_api_module.method(cost=10)
    def __or__(self, other: IsDtypeKind) -> IsDtypeKind:
        ...


# TODO: Make kind more generic to support tuples.
@array_api_module.function
def isdtype(dtype: DType, kind: IsDtypeKind) -> Boolean:
    ...


converter(DType, IsDtypeKind, lambda x: IsDtypeKind.dtype(x))
converter(str, IsDtypeKind, lambda x: IsDtypeKind.string(x))
converter(
    tuple, IsDtypeKind, lambda x: convert(x[0], IsDtypeKind) | convert(x[1:], IsDtypeKind) if x else IsDtypeKind.NULL
)


@array_api_module.register
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


@array_api_module.class_
class Int(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    def __invert__(self) -> Int:
        ...

    def __lt__(self, other: Int) -> Boolean:
        ...

    def __le__(self, other: Int) -> Boolean:
        ...

    def __eq__(self, other: Int) -> Boolean:  # type: ignore[override]
        ...

    # Make != always return a Bool, so that numpy.unique works on a tuple of ints
    # In _unique1d
    @array_api_module.method(preserve=True)
    def __ne__(self, other: Int) -> bool:  # type: ignore[override]
        return not (self == other)

    def __gt__(self, other: Int) -> Boolean:
        ...

    def __ge__(self, other: Int) -> Boolean:
        ...

    def __add__(self, other: Int) -> Int:
        ...

    def __sub__(self, other: Int) -> Int:
        ...

    def __mul__(self, other: Int) -> Int:
        ...

    def __truediv__(self, other: Int) -> Int:
        ...

    def __floordiv__(self, other: Int) -> Int:
        ...

    def __mod__(self, other: Int) -> Int:
        ...

    def __divmod__(self, other: Int) -> Int:
        ...

    def __pow__(self, other: Int) -> Int:
        ...

    def __lshift__(self, other: Int) -> Int:
        ...

    def __rshift__(self, other: Int) -> Int:
        ...

    def __and__(self, other: Int) -> Int:
        ...

    def __xor__(self, other: Int) -> Int:
        ...

    def __or__(self, other: Int) -> Int:
        ...

    def __radd__(self, other: Int) -> Int:
        ...

    def __rsub__(self, other: Int) -> Int:
        ...

    def __rmul__(self, other: Int) -> Int:
        ...

    def __rmatmul__(self, other: Int) -> Int:
        ...

    def __rtruediv__(self, other: Int) -> Int:
        ...

    def __rfloordiv__(self, other: Int) -> Int:
        ...

    def __rmod__(self, other: Int) -> Int:
        ...

    def __rpow__(self, other: Int) -> Int:
        ...

    def __rlshift__(self, other: Int) -> Int:
        ...

    def __rrshift__(self, other: Int) -> Int:
        ...

    def __rand__(self, other: Int) -> Int:
        ...

    def __rxor__(self, other: Int) -> Int:
        ...

    def __ror__(self, other: Int) -> Int:
        ...

    @property
    def i64(self) -> i64:
        ...

    @array_api_module.method(preserve=True)
    def __int__(self) -> int:
        return try_evaling(self, self.i64)

    @array_api_module.method(preserve=True)
    def __index__(self) -> int:
        return int(self)

    @array_api_module.method(preserve=True)
    def __float__(self) -> float:
        return float(int(self))

    @array_api_module.method(preserve=True)
    def __bool__(self) -> bool:
        return bool(int(self))


@array_api_module.register
def _int(i: i64, j: i64, r: Boolean, o: Int):
    yield rewrite(Int(i) == Int(i)).to(TRUE)
    yield rule(eq(r).to(Int(i) == Int(j)), i != j).then(union(r).with_(FALSE))

    yield rewrite(Int(i) >= Int(i)).to(TRUE)
    yield rule(eq(r).to(Int(i) >= Int(j)), i > j).then(union(r).with_(TRUE))
    yield rule(eq(r).to(Int(i) >= Int(j)), i < j).then(union(r).with_(FALSE))

    yield rewrite(Int(i) < Int(i)).to(FALSE)
    yield rule(eq(r).to(Int(i) < Int(j)), i < j).then(union(r).with_(TRUE))
    yield rule(eq(r).to(Int(i) < Int(j)), i > j).then(union(r).with_(FALSE))

    yield rewrite(Int(i) > Int(i)).to(FALSE)
    yield rule(eq(r).to(Int(i) > Int(j)), i > j).then(union(r).with_(TRUE))
    yield rule(eq(r).to(Int(i) > Int(j)), i < j).then(union(r).with_(FALSE))

    yield rule(eq(o).to(Int(j))).then(set_(o.i64).to(j))

    yield rewrite(Int(i) + Int(j)).to(Int(i + j))
    yield rewrite(Int(i) - Int(j)).to(Int(i - j))
    yield rewrite(Int(i) * Int(j)).to(Int(i * j))
    yield rewrite(Int(i) / Int(j)).to(Int(i / j))
    yield rewrite(Int(i) % Int(j)).to(Int(i % j))
    yield rewrite(Int(i) & Int(j)).to(Int(i & j))
    yield rewrite(Int(i) | Int(j)).to(Int(i | j))
    yield rewrite(Int(i) ^ Int(j)).to(Int(i ^ j))
    yield rewrite(Int(i) << Int(j)).to(Int(i << j))
    yield rewrite(Int(i) >> Int(j)).to(Int(i >> j))
    yield rewrite(~Int(i)).to(Int(~i))


converter(int, Int, lambda x: Int(x))


@array_api_module.class_
class Float(Expr):
    def __init__(self, value: f64Like) -> None:
        ...

    def abs(self) -> Float:
        ...

    # Make this more expensive than float, to get deterministic output
    @array_api_module.method(cost=5)
    @classmethod
    def rational(cls, r: Rational) -> Float:
        ...

    @classmethod
    def from_int(cls, i: Int) -> Float:
        ...

    def __truediv__(self, other: Float) -> Float:
        ...

    def __mul__(self, other: Float) -> Float:
        ...

    def __add__(self, other: Float) -> Float:
        ...

    def __sub__(self, other: Float) -> Float:
        ...


converter(float, Float, lambda x: Float(x))
converter(Int, Float, lambda x: Float.from_int(x))


@array_api_module.register
def _float(f: f64, f2: f64, i: i64, r: Rational, r1: Rational):
    return [
        rewrite(Float(f).abs()).to(Float(f), f >= 0.0),
        rewrite(Float(f).abs()).to(Float(-f), f < 0.0),
        # Convert from float to rationl, if its a whole number i.e. can ve converted to int
        rewrite(Float(f)).to(Float.rational(Rational(f.to_i64(), 1)), eq(f64.from_i64(f.to_i64())).to(f)),
        rewrite(Float.from_int(Int(i))).to(Float.rational(Rational(i, 1))),
        rewrite(Float(f) + Float(f2)).to(Float(f + f2)),
        rewrite(Float(f) - Float(f2)).to(Float(f - f2)),
        rewrite(Float(f) * Float(f2)).to(Float(f * f2)),
        rewrite(Float.rational(r) / Float.rational(r1)).to(Float.rational(r / r1)),
        rewrite(Float.rational(r) + Float.rational(r1)).to(Float.rational(r + r1)),
        rewrite(Float.rational(r) - Float.rational(r1)).to(Float.rational(r - r1)),
        rewrite(Float.rational(r) * Float.rational(r1)).to(Float.rational(r * r1)),
    ]


@array_api_module.class_
class TupleInt(Expr):
    EMPTY: ClassVar[TupleInt]

    def __init__(self, head: Int) -> None:
        ...

    def __add__(self, other: TupleInt) -> TupleInt:
        ...

    def length(self) -> Int:
        ...

    @array_api_module.method(preserve=True)
    def __len__(self) -> int:
        return int(self.length())

    @array_api_module.method(preserve=True)
    def __iter__(self) -> Iterator[Int]:
        return iter(self[Int(i)] for i in range(len(self)))

    def __getitem__(self, i: Int) -> Int:
        ...

    def product(self) -> Int:
        ...


converter(
    tuple,
    TupleInt,
    lambda x: TupleInt(convert(x[0], Int)) + convert(x[1:], TupleInt)
    if len(x) > 1
    else TupleInt(convert(x[0], Int))
    if x
    else TupleInt.EMPTY,
)


@array_api_module.register
def _tuple_int(ti: TupleInt, ti2: TupleInt, i: Int, i2: Int, k: i64):
    return [
        rewrite(ti + TupleInt.EMPTY).to(ti),
        rewrite(TupleInt(i).length()).to(Int(1)),
        rewrite((ti + ti2).length()).to(ti.length() + ti2.length()),
        rewrite(TupleInt(i)[Int(0)]).to(i),
        rewrite((TupleInt(i) + ti)[Int(0)]).to(i),
        # Rule for indexing > 0
        rule(eq(i).to((TupleInt(i2) + ti)[Int(k)]), k > 0).then(union(i).with_(ti[Int(k - 1)])),
        # Product
        rewrite(TupleInt(i).product()).to(i),
        rewrite((TupleInt(i) + ti).product()).to(i * ti.product()),
        rewrite(TupleInt.EMPTY.product()).to(Int(1)),
    ]


@array_api_module.class_
class OptionalInt(Expr):
    none: ClassVar[OptionalInt]

    @classmethod
    def some(cls, value: Int) -> OptionalInt:
        ...


converter(type(None), OptionalInt, lambda _: OptionalInt.none)
converter(Int, OptionalInt, OptionalInt.some)


@array_api_module.class_
class Slice(Expr):
    def __init__(
        self,
        start: OptionalInt = OptionalInt.none,
        stop: OptionalInt = OptionalInt.none,
        step: OptionalInt = OptionalInt.none,
    ) -> None:
        ...


converter(
    slice,
    Slice,
    lambda x: Slice(convert(x.start, OptionalInt), convert(x.stop, OptionalInt), convert(x.step, OptionalInt)),
)


@array_api_module.class_
class MultiAxisIndexKeyItem(Expr):
    ELLIPSIS: ClassVar[MultiAxisIndexKeyItem]
    NONE: ClassVar[MultiAxisIndexKeyItem]

    @classmethod
    def int(cls, i: Int) -> MultiAxisIndexKeyItem:
        ...

    @classmethod
    def slice(cls, slice: Slice) -> MultiAxisIndexKeyItem:
        ...


converter(type(...), MultiAxisIndexKeyItem, lambda _: MultiAxisIndexKeyItem.ELLIPSIS)
converter(type(None), MultiAxisIndexKeyItem, lambda _: MultiAxisIndexKeyItem.NONE)
converter(Int, MultiAxisIndexKeyItem, MultiAxisIndexKeyItem.int)
converter(Slice, MultiAxisIndexKeyItem, MultiAxisIndexKeyItem.slice)


@array_api_module.class_
class MultiAxisIndexKey(Expr):
    def __init__(self, item: MultiAxisIndexKeyItem) -> None:
        ...

    EMPTY: ClassVar[MultiAxisIndexKey]

    def __add__(self, other: MultiAxisIndexKey) -> MultiAxisIndexKey:
        ...


converter(
    tuple,
    MultiAxisIndexKey,
    lambda x: MultiAxisIndexKey(convert(x[0], MultiAxisIndexKeyItem)) + convert(x[1:], MultiAxisIndexKey)
    if len(x) > 1
    else MultiAxisIndexKey(convert(x[0], MultiAxisIndexKeyItem))
    if x
    else MultiAxisIndexKey.EMPTY,
)


@array_api_module.class_
class IndexKey(Expr):
    """
    A key for indexing into an array

    https://data-apis.org/array-api/2022.12/API_specification/indexing.html

    It is equivalent to the following type signature:

    Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis, None], ...], array]
    """

    ELLIPSIS: ClassVar[IndexKey]

    @classmethod
    def int(cls, i: Int) -> IndexKey:
        ...

    @classmethod
    def slice(cls, slice: Slice) -> IndexKey:
        ...

    # Disabled until we support late binding
    # @classmethod
    # def boolean_array(cls, b: NDArray) -> IndexKey:
    #     ...

    @classmethod
    def multi_axis(cls, key: MultiAxisIndexKey) -> IndexKey:
        ...


converter(type(...), IndexKey, lambda _: IndexKey.ELLIPSIS)
converter(Int, IndexKey, IndexKey.int)
converter(Slice, IndexKey, IndexKey.slice)
converter(MultiAxisIndexKey, IndexKey, IndexKey.multi_axis)


@array_api_module.class_
class Device(Expr):
    ...


ALL_INDICES: TupleInt = array_api_module.constant("ALL_INDICES", TupleInt)


# TODO: Add pushdown for math on scalars to values
# and add replacements
@array_api_module.class_
class Value(Expr):
    @classmethod
    def int(cls, i: Int) -> Value:
        ...

    @classmethod
    def float(cls, f: Float) -> Value:
        ...

    @classmethod
    def bool(cls, b: Boolean) -> Value:
        ...

    def isfinite(self) -> Boolean:
        ...

    def __lt__(self, other: Value) -> Value:
        ...

    def __truediv__(self, other: Value) -> Value:
        ...

    def astype(self, dtype: DType) -> Value:
        ...

    # TODO: Add all operations

    @property
    def dtype(self) -> DType:
        """
        Default dtype for this scalar value
        """
        ...

    @property
    def to_bool(self) -> Boolean:
        ...

    @property
    def to_int(self) -> Int:
        ...

    @property
    def to_truthy_value(self) -> Value:
        """
        Converts the value to a bool, based on if its truthy.

        https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.any.html
        """
        ...


converter(Int, Value, Value.int)
converter(Float, Value, Value.float)
converter(Boolean, Value, Value.bool)
converter(Value, Int, lambda x: x.to_int, 10)


@array_api_module.register
def _value(i: Int, f: Float, b: Boolean):
    # Default dtypes
    # https://data-apis.org/array-api/latest/API_specification/data_types.html?highlight=dtype#default-data-types
    yield rewrite(Value.int(i).dtype).to(DType.int64)
    yield rewrite(Value.float(f).dtype).to(DType.float64)
    yield rewrite(Value.bool(b).dtype).to(DType.bool)

    yield rewrite(Value.bool(b).to_bool).to(b)
    yield rewrite(Value.int(i).to_int).to(i)

    yield rewrite(Value.bool(b).to_truthy_value).to(Value.bool(b))
    # TODO: Add more rules for to_bool_value


@array_api_module.class_
class TupleValue(Expr):
    EMPTY: ClassVar[TupleValue]

    def __init__(self, head: Value) -> None:
        ...

    def __add__(self, other: TupleValue) -> TupleValue:
        ...

    def length(self) -> Int:
        ...

    def __getitem__(self, i: Int) -> Value:
        ...

    def includes(self, value: Value) -> Boolean:
        ...


converter(
    tuple,
    TupleValue,
    lambda x: TupleValue(convert(x[0], Value)) + convert(x[1:], TupleValue)
    if len(x) > 1
    else TupleValue(convert(x[0], Value))
    if x
    else TupleValue.EMPTY,
)


@array_api_module.register
def _tuple_value(
    ti: TupleValue,
    ti2: TupleValue,
    v: Value,
    i: Int,
    v2: Value,
    k: i64,
):
    return [
        rewrite(ti + TupleValue.EMPTY).to(ti),
        rewrite(TupleValue.EMPTY.length()).to(Int(0)),
        rewrite(TupleValue(v).length()).to(Int(1)),
        rewrite((ti + ti2).length()).to(ti.length() + ti2.length()),
        rewrite(TupleValue(v)[Int(0)]).to(v),
        rewrite((TupleValue(v) + ti)[Int(0)]).to(v),
        # Rule for indexing > 0
        rule(eq(v).to((TupleValue(v2) + ti)[Int(k)]), k > 0).then(union(v).with_(ti[Int(k - 1)])),
        # Includes
        rewrite(TupleValue.EMPTY.includes(v)).to(FALSE),
        rewrite(TupleValue(v).includes(v)).to(TRUE),
        rewrite(TupleValue(v).includes(v2)).to(FALSE, v != v2),
        rewrite((ti + ti2).includes(v)).to(ti.includes(v) | ti2.includes(v)),
    ]


@array_api_module.function
def possible_values(values: Value) -> TupleValue:
    """
    A value that is one of the values in the tuple.
    """
    ...


@array_api_module.class_
class NDArray(Expr):
    def __init__(self, py_array: PyObject) -> None:
        ...

    @array_api_module.method(cost=200)
    @classmethod
    def var(cls, name: StringLike) -> NDArray:
        ...

    @array_api_module.method(preserve=True)
    def __array_namespace__(self, api_version: object = None) -> ModuleType:
        return sys.modules[__name__]

    @property
    def ndim(self) -> Int:
        ...

    @property
    def dtype(self) -> DType:
        ...

    @property
    def device(self) -> Device:
        ...

    @property
    def shape(self) -> TupleInt:
        ...

    @array_api_module.method(preserve=True)
    def __bool__(self) -> bool:
        return bool(self.to_value().to_bool)

    @property
    def size(self) -> Int:
        ...

    @array_api_module.method(preserve=True)
    def __len__(self) -> int:
        return int(self.size)

    @array_api_module.method(preserve=True)
    def __iter__(self) -> Iterator[NDArray]:
        for i in range(len(self)):
            yield self[IndexKey.int(Int(i))]

    def __getitem__(self, key: IndexKey) -> NDArray:
        ...

    def __setitem__(self, key: IndexKey, value: NDArray) -> None:
        ...

    def __lt__(self, other: NDArray) -> NDArray:
        ...

    def __le__(self, other: NDArray) -> NDArray:
        ...

    def __eq__(self, other: NDArray) -> NDArray:  # type: ignore[override]
        ...

    # TODO: Add support for overloaded __ne__
    # def __ne__(self, other: NDArray) -> NDArray:  # type: ignore[override]
    #     ...

    def __gt__(self, other: NDArray) -> NDArray:
        ...

    def __ge__(self, other: NDArray) -> NDArray:
        ...

    def __add__(self, other: NDArray) -> NDArray:
        ...

    def __sub__(self, other: NDArray) -> NDArray:
        ...

    def __mul__(self, other: NDArray) -> NDArray:
        ...

    def __matmul__(self, other: NDArray) -> NDArray:
        ...

    def __truediv__(self, other: NDArray) -> NDArray:
        ...

    def __floordiv__(self, other: NDArray) -> NDArray:
        ...

    def __mod__(self, other: NDArray) -> NDArray:
        ...

    def __divmod__(self, other: NDArray) -> NDArray:
        ...

    def __pow__(self, other: NDArray) -> NDArray:
        ...

    def __lshift__(self, other: NDArray) -> NDArray:
        ...

    def __rshift__(self, other: NDArray) -> NDArray:
        ...

    def __and__(self, other: NDArray) -> NDArray:
        ...

    def __xor__(self, other: NDArray) -> NDArray:
        ...

    def __or__(self, other: NDArray) -> NDArray:
        ...

    def __radd__(self, other: NDArray) -> NDArray:
        ...

    def __rsub__(self, other: NDArray) -> NDArray:
        ...

    def __rmul__(self, other: NDArray) -> NDArray:
        ...

    def __rmatmul__(self, other: NDArray) -> NDArray:
        ...

    def __rtruediv__(self, other: NDArray) -> NDArray:
        ...

    def __rfloordiv__(self, other: NDArray) -> NDArray:
        ...

    def __rmod__(self, other: NDArray) -> NDArray:
        ...

    def __rpow__(self, other: NDArray) -> NDArray:
        ...

    def __rlshift__(self, other: NDArray) -> NDArray:
        ...

    def __rrshift__(self, other: NDArray) -> NDArray:
        ...

    def __rand__(self, other: NDArray) -> NDArray:
        ...

    def __rxor__(self, other: NDArray) -> NDArray:
        ...

    def __ror__(self, other: NDArray) -> NDArray:
        ...

    @classmethod
    def scalar(cls, value: Value) -> NDArray:
        ...

    def to_value(self) -> Value:
        ...

    @property
    def T(self) -> NDArray:  # noqa: N802
        """
        https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.T.html#array_api.array.T
        """
        ...

    @classmethod
    def vector(cls, values: TupleValue) -> NDArray:
        ...

    def index(self, indices: TupleInt) -> Value:
        """
        Return the value at the given indices.
        """
        ...


@array_api_module.function
def ndarray_index(x: NDArray) -> IndexKey:
    """
    Indexes by a masked array
    """
    ...


converter(NDArray, IndexKey, ndarray_index)
converter(Value, NDArray, NDArray.scalar)
# Need this if we want to use ints in slices of arrays coming from 1d arrays, but make it more expensive
# to prefer upcasting in the other direction when we can, which is safter at runtime
converter(NDArray, Value, lambda n: n.to_value(), 100)
converter(TupleValue, NDArray, NDArray.vector)


@array_api_module.register
def _ndarray(x: NDArray, b: Boolean, f: Float, fi1: f64, fi2: f64):
    return [
        rewrite(x.ndim).to(x.shape.length()),
        # rewrite(NDArray.scalar(Value.bool(b)).to_bool()).to(b),
        # Converting to a value requires a scalar bool value
        rewrite(x.to_value()).to(x.index(TupleInt.EMPTY)),
        # TODO: Push these down to float
        rewrite(NDArray.scalar(Value.float(f)) / NDArray.scalar(Value.float(f))).to(
            NDArray.scalar(Value.float(Float(1.0)))
        ),
        rewrite(NDArray.scalar(Value.float(f)) - NDArray.scalar(Value.float(f))).to(
            NDArray.scalar(Value.float(Float(0.0)))
        ),
        rewrite(NDArray.scalar(Value.float(Float(fi1))) > NDArray.scalar(Value.float(Float(fi2)))).to(
            NDArray.scalar(Value.bool(TRUE)), fi1 > fi2
        ),
        rewrite(NDArray.scalar(Value.float(Float(fi1))) > NDArray.scalar(Value.float(Float(fi2)))).to(
            NDArray.scalar(Value.bool(FALSE)), fi1 <= fi2
        ),
        # Transpose of tranpose is the original array
        rewrite(x.T.T).to(x),
    ]


@array_api_module.class_
class TupleNDArray(Expr):
    EMPTY: ClassVar[TupleNDArray]

    def __init__(self, head: NDArray) -> None:
        ...

    def __add__(self, other: TupleNDArray) -> TupleNDArray:
        ...

    def length(self) -> Int:
        ...

    @array_api_module.method(preserve=True)
    def __len__(self) -> int:
        return int(self.length())

    @array_api_module.method(preserve=True)
    def __iter__(self) -> Iterator[NDArray]:
        return iter(self[Int(i)] for i in range(len(self)))

    def __getitem__(self, i: Int) -> NDArray:
        ...


converter(
    tuple,
    TupleNDArray,
    lambda x: TupleNDArray(convert(x[0], NDArray)) + convert(x[1:], TupleNDArray)
    if len(x) > 1
    else TupleNDArray(convert(x[0], NDArray))
    if x
    else TupleNDArray.EMPTY,
)
converter(list, TupleNDArray, lambda x: convert(tuple(x), TupleNDArray))


@array_api_module.register
def _tuple_ndarray(ti: TupleNDArray, ti2: TupleNDArray, n: NDArray, i: Int, i2: Int, k: i64):
    return [
        rewrite(ti + TupleNDArray.EMPTY).to(ti),
        rewrite(TupleNDArray(n).length()).to(Int(1)),
        rewrite((ti + ti2).length()).to(ti.length() + ti2.length()),
        # rewrite(TupleNDArray(n)[Int(0)]).to(n),
        # rewrite((TupleNDArray(n) + ti)[Int(0)]).to(n),
        # Rule for indexing > 0
        # rule(eq(i).to((TupleInt(i2) + ti)[Int(k)]), k > 0).then(union(i).with_(ti[Int(k - 1)])),
    ]


@array_api_module.class_
class OptionalBool(Expr):
    none: ClassVar[OptionalBool]

    @classmethod
    def some(cls, value: Boolean) -> OptionalBool:
        ...


converter(type(None), OptionalBool, lambda _: OptionalBool.none)
converter(Boolean, OptionalBool, lambda x: OptionalBool.some(x))


@array_api_module.class_
class OptionalDType(Expr):
    none: ClassVar[OptionalDType]

    @classmethod
    def some(cls, value: DType) -> OptionalDType:
        ...


converter(type(None), OptionalDType, lambda _: OptionalDType.none)
converter(DType, OptionalDType, lambda x: OptionalDType.some(x))


@array_api_module.class_
class OptionalDevice(Expr):
    none: ClassVar[OptionalDevice]

    @classmethod
    def some(cls, value: Device) -> OptionalDevice:
        ...


converter(type(None), OptionalDevice, lambda _: OptionalDevice.none)
converter(Device, OptionalDevice, lambda x: OptionalDevice.some(x))


@array_api_module.class_
class OptionalTupleInt(Expr):
    none: ClassVar[OptionalTupleInt]

    @classmethod
    def some(cls, value: TupleInt) -> OptionalTupleInt:
        ...


converter(type(None), OptionalTupleInt, lambda _: OptionalTupleInt.none)
converter(TupleInt, OptionalTupleInt, lambda x: OptionalTupleInt.some(x))


@array_api_module.class_
class IntOrTuple(Expr):
    none: ClassVar[IntOrTuple]

    @classmethod
    def int(cls, value: Int) -> IntOrTuple:
        ...

    @classmethod
    def tuple(cls, value: TupleInt) -> IntOrTuple:
        ...


converter(Int, IntOrTuple, IntOrTuple.int)
converter(TupleInt, IntOrTuple, IntOrTuple.tuple)


@array_api_module.class_
class OptionalIntOrTuple(Expr):
    none: ClassVar[OptionalIntOrTuple]

    @classmethod
    def some(cls, value: IntOrTuple) -> OptionalIntOrTuple:
        ...


converter(type(None), OptionalIntOrTuple, lambda _: OptionalIntOrTuple.none)
converter(IntOrTuple, OptionalIntOrTuple, OptionalIntOrTuple.some)


@array_api_module.function
def asarray(a: NDArray, dtype: OptionalDType = OptionalDType.none, copy: OptionalBool = OptionalBool.none) -> NDArray:
    ...


@array_api_module.register
def _assarray(a: NDArray, d: OptionalDType, ob: OptionalBool):
    yield rewrite(asarray(a, d, ob).ndim).to(a.ndim)  # asarray doesn't change ndim
    yield rewrite(asarray(a)).to(a)


@array_api_module.function
def isfinite(x: NDArray) -> NDArray:
    ...


@array_api_module.function
def sum(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray:
    """
    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.sum.html?highlight=sum
    """
    ...


@array_api_module.register
def _sum(x: NDArray, y: NDArray, v: Value, dtype: DType):
    return [
        rewrite(sum(x / NDArray.scalar(v))).to(sum(x) / NDArray.scalar(v)),
        # Sum of 0D array is
    ]


@array_api_module.function
def reshape(x: NDArray, shape: TupleInt, copy: OptionalBool = OptionalBool.none) -> NDArray:
    ...


# @array_api_module.function
# def reshape_transform_index(original_shape: TupleInt, shape: TupleInt, index: TupleInt) -> TupleInt:
#     """
#     Transforms an indexing operation on a reshaped array to an indexing operation on the original array.
#     """
#     ...


# @array_api_module.function
# def reshape_transform_shape(original_shape: TupleInt, shape: TupleInt) -> TupleInt:
#     """
#     Transforms the shape of an array to one that is reshaped, by replacing -1 with the correct value.
#     """
#     ...


# @array_api_module.register
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


@array_api_module.function
def unique_values(x: NDArray) -> NDArray:
    ...


@array_api_module.register
def _unique_values(x: NDArray):
    return [
        rewrite(unique_values(unique_values(x))).to(unique_values(x)),
    ]


@array_api_module.function
def concat(arrays: TupleNDArray, axis: OptionalInt = OptionalInt.none) -> NDArray:
    ...


@array_api_module.register
def _concat(x: NDArray):
    return [
        rewrite(concat(TupleNDArray(x))).to(x),
    ]


@array_api_module.function
def astype(x: NDArray, dtype: DType) -> NDArray:
    ...


@array_api_module.register
def _astype(x: NDArray, dtype: DType, i: i64):
    return [
        rewrite(astype(x, dtype).dtype).to(dtype),
        rewrite(astype(NDArray.scalar(Value.int(Int(i))), float64)).to(
            NDArray.scalar(Value.float(Float(f64.from_i64(i))))
        ),
    ]


@array_api_module.function(cost=500)
def unique_counts(x: NDArray) -> TupleNDArray:
    ...


@array_api_module.register
def _unique_counts(x: NDArray, c: NDArray, tv: TupleValue, v: Value, dtype: DType):
    return [
        rewrite(unique_counts(x).length()).to(Int(2)),
        # Sum of all unique counts is the size of the array
        rewrite(sum(unique_counts(x)[Int(1)])).to(NDArray.scalar(Value.int(x.size))),
        # Same but with astype in the middle
        # TODO: Replace
        rewrite(sum(astype(unique_counts(x)[Int(1)], dtype))).to(astype(NDArray.scalar(Value.int(x.size)), dtype)),
    ]


@array_api_module.function
def square(x: NDArray) -> NDArray:
    ...


@array_api_module.function
def any(x: NDArray) -> NDArray:
    ...


@array_api_module.function(egg_fn="ndarray-abs")
def abs(x: NDArray) -> NDArray:
    ...


@array_api_module.function(egg_fn="ndarray-log")
def log(x: NDArray) -> NDArray:
    ...


@array_api_module.register
def _abs(f: Float):
    return [
        rewrite(abs(NDArray.scalar(Value.float(f)))).to(NDArray.scalar(Value.float(f.abs()))),
    ]


@array_api_module.function(cost=100)
def unique_inverse(x: NDArray) -> TupleNDArray:
    ...


@array_api_module.register
def _unique_inverse(x: NDArray, i: Int):
    return [
        rewrite(unique_inverse(x).length()).to(Int(2)),
        # Shape of unique_inverse first element is same as shape of unique_values
        rewrite(unique_inverse(x)[Int(0)]).to(unique_values(x)),
    ]


@array_api_module.function
def zeros(
    shape: TupleInt, dtype: OptionalDType = OptionalDType.none, device: OptionalDevice = OptionalDevice.none
) -> NDArray:
    ...


@array_api_module.function
def expand_dims(x: NDArray, axis: Int = Int(0)) -> NDArray:
    ...


@array_api_module.function(cost=100000)
def mean(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none, keepdims: Boolean = FALSE) -> NDArray:
    ...


# TODO: Possibly change names to include modules.
@array_api_module.function(egg_fn="ndarray-sqrt")
def sqrt(x: NDArray) -> NDArray:
    ...


@array_api_module.function(cost=100000)
def std(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray:
    ...


linalg = sys.modules[__name__]


@array_api_module.function
def svd(x: NDArray, full_matrices: Boolean = TRUE) -> TupleNDArray:
    """
    https://data-apis.org/array-api/2022.12/extensions/generated/array_api.linalg.svd.html
    """
    ...


@array_api_module.register
def _linalg(x: NDArray, full_matrices: Boolean):
    return [
        rewrite(svd(x, full_matrices).length()).to(Int(3)),
    ]


##
# Interval analysis
#
# to analyze `any(((astype(unique_counts(NDArray.var("y"))[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0))) < NDArray.scalar(Value.int(Int(0)))).bool()``
##

greater_zero = array_api_module.relation("greater_zero", Value)


# @array_api_module.function
# def ndarray_all_greater_0(x: NDArray) -> Unit:
#     ...


# @array_api_module.function
# def ndarray_all_false(x: NDArray) -> Unit:
#     ...


# @array_api_module.function
# def ndarray_all_true(x: NDArray) -> Unit:
#     ...


# any((astype(unique_counts(_NDArray_1)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))) < NDArray.scalar(Value.int(Int(0)))).to_bool()

# sum(astype(unique_counts(_NDArray_1)[Int(1)], DType.float64) / NDArray.scalar(Value.int(Int(150))))
# And also

# def


@array_api_module.function
def broadcast_index(from_shape: TupleInt, to_shape: TupleInt, index: TupleInt) -> TupleInt:
    """
    Returns the index in the original array of the given index in the broadcasted array.
    """
    ...


@array_api_module.function
def broadcast_shapes(shape1: TupleInt, shape2: TupleInt) -> TupleInt:
    """
    Returns the shape of the broadcasted array.
    """
    ...


@array_api_module.register
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
            NDArray.scalar(Value.bool(possible_values(x.index(ALL_INDICES).to_truthy_value).includes(Value.bool(TRUE))))
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
        rewrite(possible_values(Value.bool(b))).to(TupleValue(Value.bool(b))),
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


@array_api_module.register
def _scalar_math(v: Value, vs: TupleValue, i: Int):
    yield rewrite(NDArray.scalar(v).shape).to(TupleInt.EMPTY)
    yield rewrite(NDArray.scalar(v).dtype).to(v.dtype)
    yield rewrite(NDArray.scalar(v).index(TupleInt.EMPTY)).to(v)


@array_api_module.register
def _vector_math(v: Value, vs: TupleValue, i: Int):
    yield rewrite(NDArray.vector(vs).shape).to(TupleInt(vs.length()))
    yield rewrite(NDArray.vector(vs).dtype).to(vs[Int(0)].dtype)
    yield rewrite(NDArray.vector(vs).index(TupleInt(i))).to(vs[i])


@array_api_module.register
def _reshape_math(x: NDArray, shape: TupleInt, copy: OptionalBool):
    res = reshape(x, shape, copy)

    yield _demand_shape(res, x)

    # Reshaping a vec to a vec is the same as the vec
    yield rewrite(reshape(x, TupleInt(Int(-1)), copy)).to(x, eq(x.shape.length()).to(Int(1)))


@array_api_module.register
def _indexing_pushdown(x: NDArray, shape: TupleInt, copy: OptionalBool, i: Int):
    # rewrite full getitem to indexec
    yield rewrite(x[IndexKey.int(i)]).to(NDArray.scalar(x.index(TupleInt(i))))
    # TODO: Multi index rewrite as well if all are ints


##
# Assumptions
##


@array_api_module.function(mutates_first_arg=True)
def assume_dtype(x: NDArray, dtype: DType) -> None:
    """
    Asserts that the dtype of x is dtype.
    """
    ...


@array_api_module.register
def _assume_dtype(x: NDArray, dtype: DType, idx: TupleInt):
    orig_x = copy(x)
    assume_dtype(x, dtype)
    yield rewrite(x.dtype).to(dtype)
    yield rewrite(x.shape).to(orig_x.shape)
    yield rewrite(x.index(idx)).to(orig_x.index(idx))


@array_api_module.function(mutates_first_arg=True)
def assume_shape(x: NDArray, shape: TupleInt) -> None:
    """
    Asserts that the shape of x is shape.
    """
    ...


@array_api_module.register
def _assume_shape(x: NDArray, shape: TupleInt, idx: TupleInt):
    orig_x = copy(x)
    assume_shape(x, shape)
    yield rewrite(x.shape).to(shape)
    yield rewrite(x.dtype).to(orig_x.dtype)
    yield rewrite(x.index(idx)).to(orig_x.index(idx))


@array_api_module.function(mutates_first_arg=True)
def assume_isfinite(x: NDArray) -> None:
    """
    Asserts that the scalar ndarray is non null and not infinite.
    """
    ...


@array_api_module.register
def _isfinite(x: NDArray, ti: TupleInt):
    orig_x = copy(x)
    assume_isfinite(x)

    # pass through getitem, shape, index
    yield rewrite(x.shape).to(orig_x.shape)
    yield rewrite(x.dtype).to(orig_x.dtype)
    yield rewrite(x.index(ti)).to(orig_x.index(ti))
    # But say that any indixed value is finite
    yield rewrite(x.index(ti).isfinite()).to(TRUE)


@array_api_module.function(mutates_first_arg=True)
def assume_value_one_of(x: NDArray, values: TupleValue) -> None:
    """
    A value that is one of the values in the tuple.
    """
    ...


@array_api_module.register
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


@array_api_module.register
def _ndarray_value_isfinite(arr: NDArray, x: Value, xs: TupleValue, i: Int, f: f64, b: Boolean):
    yield rewrite(Value.int(i).isfinite()).to(TRUE)
    yield rewrite(Value.bool(b).isfinite()).to(TRUE)
    yield rewrite(Value.float(Float(f)).isfinite()).to(TRUE, f != f64(math.nan))

    # a sum of an array is finite if all the values are finite
    yield rewrite(isfinite(sum(arr))).to(NDArray.scalar(Value.bool(arr.index(ALL_INDICES).isfinite())))


@array_api_module.register
def _unique(xs: TupleValue, a: NDArray, shape: TupleInt, copy: OptionalBool):
    yield rewrite(unique_values(x=a)).to(NDArray.vector(possible_values(a.index(ALL_INDICES))))
    # yield rewrite(
    #     possible_values(reshape(a.index(shape, copy), ALL_INDICES)),
    # ).to(possible_values(a.index(ALL_INDICES)))


@array_api_module.register
def _size(x: NDArray):
    yield rewrite(x.size).to(x.shape.product())


@overload
def try_evaling(expr: Expr, prim_expr: i64) -> int:
    ...


@overload
def try_evaling(expr: Expr, prim_expr: Bool) -> bool:
    ...


def try_evaling(expr: Expr, prim_expr: i64 | Bool) -> int | bool:
    """
    Try evaling the expression, and if it fails, display the egraph and raise an error.
    """
    egraph = EGraph.current()
    egraph.register(expr)
    egraph.run((run() * 30).saturate())
    try:
        return egraph.eval(prim_expr)
    except EggSmolError as exc:
        egraph.display(n_inline_leaves=2, split_primitive_outputs=True)
        msg = "Cannot simplify:"
        raise ValueError(msg, egraph.extract(expr)) from exc
