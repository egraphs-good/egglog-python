# mypy: disable-error-code=empty-body
from __future__ import annotations

import itertools
import math
import numbers
import sys
from copy import copy
from dataclasses import field
from re import L
from typing import Any, Callable, ClassVar, Iterator, TypeVar

import numpy as np
from attr import dataclass
from egglog import *
from egglog.bindings import EggSmolError
from egglog.egraph import Action
from egglog.runtime import RuntimeExpr

# Pretend that exprs are numbers b/c scikit learn does isinstance checks
numbers.Integral.register(RuntimeExpr)

egraph = EGraph()

T = TypeVar("T", bound=Expr)

runtime_ruleset = egraph.ruleset("runtime")


# For now, have this global e-graph for this module, a bit hacky, but works as a proof of concept.
# We need a global e-graph so that we have the preserved methods reference it to extract when they are called.


def extract_py(e: Expr) -> Any:
    # print(e)
    egraph.register(e)
    egraph.run((run() * 10).saturate())
    final_object = egraph.extract(e)
    # print(f"  -> {final_object}")
    # with egraph:
    egraph.run((run(runtime_ruleset) * 10 + run() * 10).saturate())
    # Run saturation again b/c sometimes it doesn't work the first time.
    # final_object = egraph.extract(egraph.extract(final_object))
    # egraph.run(run(limit=10).saturate())
    # final_object: Expr = egraph.extract(final_object)
    # egraph.register(final_object.to_py())
    # egraph.run(run(limit=10).saturate())

    # try:
    #     x = str((Int(1) * Int(3)) == Int(2))
    # except Exception:
    #     pass
    # else:
    #     if str(egraph.extract(final_object)) == x:
    #         final_object = (Int(1) * Int(3)) == Int(2)
    # pass
    # raise Exception("Failed to extract")

    # final_object = egraph.extract(egraph.extract(final_object))
    # egraph.run(run(limit=10).saturate())

    # print(f"     -> {egraph.extract(final_object)}\n")
    try:
        res = egraph.load_object(egraph.extract(final_object.to_py()))  # type: ignore[attr-defined]
    except EggSmolError as error:
        res = egraph.extract(final_object)
        raise Exception(f"Failed to extract {res} to py") from error
    # print(res)
    return res


@egraph.class_
class Bool(Expr):
    @egraph.method(preserve=True)
    def __bool__(self) -> bool:
        return extract_py(self)

    def to_py(self) -> PyObject:
        ...

    def __or__(self, other: Bool) -> Bool:
        ...

    def __and__(self, other: Bool) -> Bool:
        ...


TRUE = egraph.constant("TRUE", Bool)
FALSE = egraph.constant("FALSE", Bool)
converter(bool, Bool, lambda x: TRUE if x else FALSE)


@egraph.register
def _bool(x: Bool):
    return [
        set_(TRUE.to_py()).to(egraph.save_object(True)),
        set_(FALSE.to_py()).to(egraph.save_object(False)),
        rewrite(TRUE | x).to(TRUE),
        rewrite(FALSE | x).to(x),
        rewrite(TRUE & x).to(x),
        rewrite(FALSE & x).to(FALSE),
    ]


@egraph.class_
class DType(Expr):
    float64: ClassVar[DType]
    float32: ClassVar[DType]
    int64: ClassVar[DType]
    int32: ClassVar[DType]
    object: ClassVar[DType]
    bool: ClassVar[DType]

    def __eq__(self, other: DType) -> Bool:  # type: ignore[override]
        ...


float64 = DType.float64
float32 = DType.float32
int32 = DType.int32
int64 = DType.int64

_DTYPES = [float64, float32, int32, int64, DType.object]

converter(type, DType, lambda x: convert(np.dtype(x), DType))
converter(type(np.dtype), DType, lambda x: getattr(DType, x.name))  # type: ignore[call-overload]
egraph.register(
    *(
        rewrite(l == r).to(TRUE if expr_parts(l) == expr_parts(r) else FALSE)
        for l, r in itertools.product(_DTYPES, repeat=2)
    )
)


@egraph.class_
class IsDtypeKind(Expr):
    NULL: ClassVar[IsDtypeKind]

    @classmethod
    def string(cls, s: StringLike) -> IsDtypeKind:
        ...

    @classmethod
    def dtype(cls, d: DType) -> IsDtypeKind:
        ...

    @egraph.method(cost=10)
    def __or__(self, other: IsDtypeKind) -> IsDtypeKind:
        ...


# TODO: Make kind more generic to support tuples.
@egraph.function
def isdtype(dtype: DType, kind: IsDtypeKind) -> Bool:
    ...


converter(DType, IsDtypeKind, lambda x: IsDtypeKind.dtype(x))
converter(str, IsDtypeKind, lambda x: IsDtypeKind.string(x))
converter(
    tuple, IsDtypeKind, lambda x: convert(x[0], IsDtypeKind) | convert(x[1:], IsDtypeKind) if x else IsDtypeKind.NULL
)


@egraph.register
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


# assert not bool(isdtype(DType.float32, IsDtypeKind.string("integral")))


@egraph.class_
class Float(Expr):
    def __init__(self, value: f64Like) -> None:
        ...

    def abs(self) -> Float:
        ...


converter(float, Float, lambda x: Float(x))


@egraph.register
def _float(f: f64, f2: f64, r: Bool, o: Float):
    return [
        rewrite(Float(f).abs()).to(Float(f), f >= 0.0),
        rewrite(Float(f).abs()).to(Float(-f), f < 0.0),
    ]


@egraph.class_
class Int(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    def __invert__(self) -> Int:
        ...

    def __lt__(self, other: Int) -> Bool:
        ...

    def __le__(self, other: Int) -> Bool:
        ...

    def __eq__(self, other: Int) -> Bool:  # type: ignore[override]
        ...

    # Make != always return a Bool, so that numpy.unique works on a tuple of ints
    # In _unique1d
    @egraph.method(preserve=True)
    def __ne__(self, other: Int) -> bool:  # type: ignore[override]
        return not extract_py(self == other)

    def __gt__(self, other: Int) -> Bool:
        ...

    def __ge__(self, other: Int) -> Bool:
        ...

    def __add__(self, other: Int) -> Int:
        ...

    def __sub__(self, other: Int) -> Int:
        ...

    def __mul__(self, other: Int) -> Int:
        ...

    def __matmul__(self, other: Int) -> Int:
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

    @egraph.method(preserve=True)
    def __int__(self) -> int:
        return extract_py(self)

    @egraph.method(preserve=True)
    def __index__(self) -> int:
        return extract_py(self)

    @egraph.method(preserve=True)
    def __float__(self) -> float:
        return float(int(self))

    def to_py(self) -> PyObject:
        ...

    @egraph.method(preserve=True)
    def __bool__(self) -> bool:
        return self != Int(0)


@egraph.register
def _int(i: i64, j: i64, r: Bool, o: Int):
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

    yield rule(eq(o).to(Int(j))).then(set_(o.to_py()).to(PyObject.from_int(j)))

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
# converter(float, Int, lambda x: Int(int(x)))


assert expr_parts(egraph.simplify(Int(1) == Int(1), 10)) == expr_parts(TRUE)
assert expr_parts(egraph.simplify(Int(1) == Int(2), 10)) == expr_parts(FALSE)
assert expr_parts(egraph.simplify(Int(1) >= Int(2), 10)) == expr_parts(FALSE)
assert expr_parts(egraph.simplify(Int(1) >= Int(1), 10)) == expr_parts(TRUE)
assert expr_parts(egraph.simplify(Int(2) >= Int(1), 10)) == expr_parts(TRUE)


@egraph.class_
class TupleInt(Expr):
    EMPTY: ClassVar[TupleInt]

    def __init__(self, head: Int) -> None:
        ...

    def __add__(self, other: TupleInt) -> TupleInt:
        ...

    def length(self) -> Int:
        ...

    @egraph.method(preserve=True)
    def __len__(self) -> int:
        return int(self.length())

    @egraph.method(preserve=True)
    def __iter__(self):
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


@egraph.register
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


@egraph.class_
class OptionalInt(Expr):
    none: ClassVar[OptionalInt]

    @classmethod
    def some(cls, value: Int) -> OptionalInt:
        ...


converter(type(None), OptionalInt, lambda x: OptionalInt.none)
converter(Int, OptionalInt, OptionalInt.some)


@egraph.class_
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


@egraph.class_
class MultiAxisIndexKeyItem(Expr):
    ELLIPSIS: ClassVar[MultiAxisIndexKeyItem]
    NONE: ClassVar[MultiAxisIndexKeyItem]

    @classmethod
    def int(cls, i: Int) -> MultiAxisIndexKeyItem:
        ...

    @classmethod
    def slice(cls, slice: Slice) -> MultiAxisIndexKeyItem:
        ...


converter(type(...), MultiAxisIndexKeyItem, lambda x: MultiAxisIndexKeyItem.ELLIPSIS)
converter(type(None), MultiAxisIndexKeyItem, lambda x: MultiAxisIndexKeyItem.NONE)
converter(Int, MultiAxisIndexKeyItem, MultiAxisIndexKeyItem.int)
converter(Slice, MultiAxisIndexKeyItem, MultiAxisIndexKeyItem.slice)


@egraph.class_
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


@egraph.class_
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


converter(type(...), IndexKey, lambda x: IndexKey.ELLIPSIS)
converter(Int, IndexKey, IndexKey.int)
converter(Slice, IndexKey, IndexKey.slice)
converter(MultiAxisIndexKey, IndexKey, IndexKey.multi_axis)


@egraph.class_
class Device(Expr):
    ...


# TODO: Add pushdown for math on scalars to values
# and add replacements
@egraph.class_
class Value(Expr):
    @classmethod
    def var(cls, name: StringLike) -> Value:
        ...

    @classmethod
    def int(cls, i: Int) -> Value:
        ...

    @classmethod
    def float(cls, f: Float) -> Value:
        ...

    @classmethod
    def bool(cls, b: Bool) -> Value:
        ...

    def isfinite(self) -> Bool:
        ...

    # TODO: Add all operations

    @property
    def dtype(self) -> DType:
        """
        Default dtype for this scalar value
        """
        ...


converter(Int, Value, Value.int)
converter(Float, Value, Value.float)
converter(Bool, Value, Value.bool)


@egraph.register
def _value(i: Int, f: Float, b: Bool):
    # Default dtypes
    # https://data-apis.org/array-api/latest/API_specification/data_types.html?highlight=dtype#default-data-types
    yield rewrite(Value.int(i).dtype).to(DType.int64)
    yield rewrite(Value.float(f).dtype).to(DType.float64)
    yield rewrite(Value.bool(b).dtype).to(DType.bool)


@egraph.class_
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


converter(
    tuple,
    TupleValue,
    lambda x: TupleValue(convert(x[0], Value)) + convert(x[1:], TupleValue)
    if len(x) > 1
    else TupleValue(convert(x[0], Value))
    if x
    else TupleValue.EMPTY,
)


@egraph.register
def _tuple_value(ti: TupleValue, ti2: TupleValue, v: Value, i: Int, v2: Value, k: i64):
    return [
        rewrite(ti + TupleValue.EMPTY).to(ti),
        rewrite(TupleValue.EMPTY.length()).to(Int(0)),
        rewrite(TupleValue(v).length()).to(Int(1)),
        rewrite((ti + ti2).length()).to(ti.length() + ti2.length()),
        rewrite(TupleValue(v)[Int(0)]).to(v),
        rewrite((TupleValue(v) + ti)[Int(0)]).to(v),
        # Rule for indexing > 0
        rule(eq(v).to((TupleValue(v2) + ti)[Int(k)]), k > 0).then(union(v).with_(ti[Int(k - 1)])),
    ]


@egraph.class_
class NDArray(Expr):
    def __init__(self, py_array: PyObject) -> None:
        ...

    @egraph.method(cost=100)
    @classmethod
    def var(cls, name: StringLike) -> NDArray:
        ...

    @egraph.method(preserve=True)
    def __array_namespace__(self, api_version=None):
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

    def to_bool(self) -> Bool:
        ...

    @egraph.method(preserve=True)
    def __bool__(self) -> bool:
        return bool(self.to_bool())

    @property
    def size(self) -> Int:
        ...

    @egraph.method(preserve=True)
    def __len__(self) -> int:
        return int(self.size)

    @egraph.method(preserve=True)
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

    def to_int(self) -> Int:
        ...

    @property
    def T(self) -> NDArray:
        """
        https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.T.html#array_api.array.T
        """
        ...

    @classmethod
    def vector(cls, values: TupleValue) -> NDArray:
        ...


@egraph.function
def ndarray_index(x: NDArray) -> IndexKey:
    ...


converter(NDArray, IndexKey, ndarray_index)
converter(Value, NDArray, NDArray.scalar)
converter(NDArray, Int, lambda n: n.to_int())
converter(TupleValue, NDArray, NDArray.vector)


@egraph.register
def _ndarray(x: NDArray, b: Bool, f: Float, fi1: f64, fi2: f64):
    return [
        rewrite(x.ndim).to(x.shape.length()),
        rewrite(NDArray.scalar(Value.bool(b)).to_bool()).to(b),
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
    ]


@egraph.class_
class TupleNDArray(Expr):
    EMPTY: ClassVar[TupleNDArray]

    def __init__(self, head: NDArray) -> None:
        ...

    def __add__(self, other: TupleNDArray) -> TupleNDArray:
        ...

    def length(self) -> Int:
        ...

    @egraph.method(preserve=True)
    def __len__(self) -> int:
        return int(self.length())

    @egraph.method(preserve=True)
    def __iter__(self):
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


@egraph.register
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


@egraph.class_
class OptionalBool(Expr):
    none: ClassVar[OptionalBool]

    @classmethod
    def some(cls, value: Bool) -> OptionalBool:
        ...


converter(type(None), OptionalBool, lambda x: OptionalBool.none)
converter(Bool, OptionalBool, lambda x: OptionalBool.some(x))


@egraph.class_
class OptionalDType(Expr):
    none: ClassVar[OptionalDType]

    @classmethod
    def some(cls, value: DType) -> OptionalDType:
        ...


converter(type(None), OptionalDType, lambda x: OptionalDType.none)
converter(DType, OptionalDType, lambda x: OptionalDType.some(x))


@egraph.class_
class OptionalDevice(Expr):
    none: ClassVar[OptionalDevice]

    @classmethod
    def some(cls, value: Device) -> OptionalDevice:
        ...


converter(type(None), OptionalDevice, lambda x: OptionalDevice.none)
converter(Device, OptionalDevice, lambda x: OptionalDevice.some(x))


@egraph.class_
class OptionalTupleInt(Expr):
    none: ClassVar[OptionalTupleInt]

    @classmethod
    def some(cls, value: TupleInt) -> OptionalTupleInt:
        ...


converter(type(None), OptionalTupleInt, lambda x: OptionalTupleInt.none)
converter(TupleInt, OptionalTupleInt, lambda x: OptionalTupleInt.some(x))


@egraph.class_
class OptionalIntOrTuple(Expr):
    none: ClassVar[OptionalIntOrTuple]

    @classmethod
    def int(cls, value: Int) -> OptionalIntOrTuple:
        ...

    @classmethod
    def tuple(cls, value: TupleInt) -> OptionalIntOrTuple:
        ...


converter(type(None), OptionalIntOrTuple, lambda x: OptionalIntOrTuple.none)
converter(Int, OptionalIntOrTuple, OptionalIntOrTuple.int)
converter(TupleInt, OptionalIntOrTuple, OptionalIntOrTuple.tuple)


@egraph.function
def asarray(a: NDArray, dtype: OptionalDType = OptionalDType.none, copy: OptionalBool = OptionalBool.none) -> NDArray:
    ...


@egraph.register
def _assarray(a: NDArray, d: OptionalDType, ob: OptionalBool):
    yield rewrite(asarray(a, d, ob).ndim).to(a.ndim)  # asarray doesn't change ndim
    yield rewrite(asarray(a)).to(a)  # asarray doesn't change to_py


@egraph.function
def isfinite(x: NDArray) -> NDArray:
    ...


@egraph.function
def sum(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray:
    """
    https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.sum.html?highlight=sum
    """
    ...


@egraph.register
def _sum(x: NDArray, y: NDArray, v: Value, dtype: DType):
    return [
        rewrite(sum(x / NDArray.scalar(v))).to(sum(x) / NDArray.scalar(v)),
    ]


@egraph.function
def reshape(x: NDArray, shape: TupleInt, copy: OptionalBool = OptionalBool.none) -> NDArray:
    ...


@egraph.register
def _reshape(x: NDArray, y: NDArray, shape: TupleInt, copy: OptionalBool, i: Int, s: String):
    return [
        # dtype of result is same as input
        rewrite(reshape(x, shape, copy).dtype).to(x.dtype),
        # dimensions of output are the same as length of shape
        rewrite(reshape(x, shape, copy).shape.length()).to(shape.length()),
        # Shape of single dimensions reshape is the total number of elements
        rewrite(reshape(x, TupleInt(Int(-1)), copy).shape).to(TupleInt(x.size)),
        # Reshaping something with just one dimensions doesn't change the shape
        rule(
            eq(y).to(reshape(x, TupleInt(Int(-1)), copy)),
            eq(x.shape).to(TupleInt(i)),
        ).then(union(x).with_(y)),
    ]


@egraph.function
def unique_values(x: NDArray) -> NDArray:
    ...


@egraph.register
def _unique_values(x: NDArray):
    return [
        rewrite(unique_values(unique_values(x))).to(unique_values(x)),
    ]


@egraph.function
def concat(arrays: TupleNDArray, axis: OptionalInt = OptionalInt.none) -> NDArray:
    ...


@egraph.register
def _concat(x: NDArray):
    return [
        rewrite(concat(TupleNDArray(x))).to(x),
    ]


@egraph.function
def unique_counts(x: NDArray) -> TupleNDArray:
    ...


@egraph.register
def _unique_counts(x: NDArray):
    return [
        rewrite(unique_counts(x).length()).to(Int(2)),
        # Sum of all unique counts is the size of the array
        rewrite(sum(unique_counts(x)[Int(1)])).to(NDArray.scalar(Value.int(x.size))),
    ]


@egraph.function
def astype(x: NDArray, dtype: DType) -> NDArray:
    ...


@egraph.register
def _astype(x: NDArray, dtype: DType, i: i64):
    return [
        rewrite(astype(x, dtype).dtype).to(dtype),
        rewrite(sum(astype(x, dtype))).to(astype(sum(x), dtype)),
        rewrite(astype(NDArray.scalar(Value.int(Int(i))), float64)).to(
            NDArray.scalar(Value.float(Float(f64.from_i64(i))))
        ),
    ]


@egraph.function
def std(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray:
    ...


@egraph.function
def any(x: NDArray) -> NDArray:
    ...


@egraph.function(egg_fn="ndarray-abs")
def abs(x: NDArray) -> NDArray:
    ...


@egraph.function(egg_fn="ndarray-log")
def log(x: NDArray) -> NDArray:
    ...


@egraph.register
def _abs(f: Float):
    return [
        rewrite(abs(NDArray.scalar(Value.float(f)))).to(NDArray.scalar(Value.float(f.abs()))),
    ]


@egraph.function
def unique_inverse(x: NDArray) -> TupleNDArray:
    ...


@egraph.register
def _unique_inverse(x: NDArray):
    return [
        rewrite(unique_inverse(x).length()).to(Int(2)),
        # Shape of unique_inverse first element is same as shape of unique_values
        rewrite(unique_inverse(x)[Int(0)].shape).to(unique_values(x).shape),
    ]


@egraph.function
def zeros(
    shape: TupleInt, dtype: OptionalDType = OptionalDType.none, device: OptionalDevice = OptionalDevice.none
) -> NDArray:
    ...


@egraph.function
def mean(x: NDArray, axis: OptionalIntOrTuple = OptionalIntOrTuple.none) -> NDArray:
    ...


# TODO: Possibly change names to include modules.
@egraph.function(egg_fn="ndarray-sqrt")
def sqrt(x: NDArray) -> NDArray:
    ...


linalg = sys.modules[__name__]


@egraph.function
def svd(x: NDArray, full_matrices: Bool = TRUE) -> TupleNDArray:
    """
    https://data-apis.org/array-api/2022.12/extensions/generated/array_api.linalg.svd.html
    """
    ...


@egraph.register
def _linalg(x: NDArray, full_matrices: Bool):
    return [
        rewrite(svd(x, full_matrices).length()).to(Int(3)),
    ]


##
# Interval analysis
#
# to analyze `any(((astype(unique_counts(NDArray.var("y"))[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0))) < NDArray.scalar(Value.int(Int(0)))).bool()``
##


@egraph.function
def ndarray_all_greater_0(x: NDArray) -> Unit:
    ...


@egraph.function
def ndarray_all_false(x: NDArray) -> Unit:
    ...


@egraph.function
def ndarray_all_true(x: NDArray) -> Unit:
    ...


# TODO: Redo as rewrites with assumptions?

# sum(astype(unique_counts(_NDArray_1)[Int(1)], DType.float64) / NDArray.scalar(Value.int(Int(150))))


@egraph.register
def _interval_analaysis(x: NDArray, y: NDArray, z: NDArray, dtype: DType, f: f64, i: i64, b: Bool):
    return [
        rule(eq(y).to(x < NDArray.scalar(Value.int(Int(0)))), ndarray_all_greater_0(x)).then(ndarray_all_false(y)),
        rule(eq(y).to(any(x)), ndarray_all_false(x)).then(union(y).with_(NDArray.scalar(Value.bool(FALSE)))),
        rule(
            eq(y).to(unique_counts(x)[Int(1)]),
        ).then(ndarray_all_greater_0(y)),
        rule(eq(y).to(astype(x, dtype)), ndarray_all_greater_0(x)).then(ndarray_all_greater_0(y)),
        rule(eq(z).to(x / y), ndarray_all_greater_0(x), ndarray_all_greater_0(y)).then(ndarray_all_greater_0(z)),
        rule(eq(z).to(NDArray.scalar(Value.float(Float(f)))), f > 0.0).then(ndarray_all_greater_0(z)),
        rule(eq(z).to(NDArray.scalar(Value.int(Int(i)))), i > 0).then(ndarray_all_greater_0(z)),
        # Also support abs(x) > 0
        rule(eq(y).to(abs(x))).then(ndarray_all_greater_0(y)),
        # And if all_greater_0(x) then x > 0 is all true
        rule(eq(y).to(x > NDArray.scalar(Value.int(Int(0)))), ndarray_all_greater_0(x)).then(ndarray_all_true(y)),
        rule(eq(b).to(x.to_bool()), ndarray_all_true(x)).then(union(b).with_(TRUE)),
    ]


##
# Mathematical descriptions of arrays as:
# 1. A shape `.shape`
# 2. A dtype `.dtype`
# 3. A mapping from indices to values `array_value(x, idx)`
#
# For all operations that are supported mathematically, define each of the above.
##


@egraph.function
def array_value(x: NDArray, idx: TupleInt) -> Value:
    """
    Indices into an ndarray to get a value.
    """
    ...


@egraph.register
def _array_math(v: Value, vs: TupleValue, i: Int):
    # Scalar values
    yield rewrite(NDArray.scalar(v).shape).to(TupleInt.EMPTY)
    yield rewrite(NDArray.scalar(v).dtype).to(v.dtype)
    yield rewrite(array_value(NDArray.scalar(v), TupleInt.EMPTY)).to(v)

    # Vector values
    yield rewrite(NDArray.vector(vs).shape).to(TupleInt(vs.length()))
    yield rewrite(NDArray.vector(vs).dtype).to(vs[Int(0)].dtype)
    yield rewrite(array_value(NDArray.vector(vs), TupleInt(i))).to(vs[i])


@egraph.function(mutates_first_arg=True)
def assume_dtype(x: NDArray, dtype: DType) -> None:
    """
    Asserts that the dtype of x is dtype.
    """
    ...


@egraph.register
def _assume_dtype(x: NDArray, dtype: DType, idx: TupleInt):
    orig_x = copy(x)
    assume_dtype(x, dtype)
    yield rewrite(x.dtype).to(dtype)
    yield rewrite(x.shape).to(orig_x.shape)
    yield rewrite(array_value(x, idx)).to(array_value(orig_x, idx))


@egraph.function(mutates_first_arg=True)
def assume_shape(x: NDArray, shape: TupleInt) -> None:
    """
    Asserts that the shape of x is shape.
    """
    ...


@egraph.register
def _assume_shape(x: NDArray, shape: TupleInt, idx: TupleInt):
    orig_x = copy(x)
    assume_shape(x, shape)
    yield rewrite(x.shape).to(shape)
    yield rewrite(x.dtype).to(orig_x.dtype)
    yield rewrite(array_value(x, idx)).to(array_value(orig_x, idx))


@egraph.function(mutates_first_arg=True)
def assume_value(x: NDArray, value: Value) -> None:
    """
    Asserts that all values (scalars) in x are equal to value.

    TODO: If we had first class lambdas, this would take an indexing function instead of a value.
    """
    ...


@egraph.register
def _assume_value(x: NDArray, value: Value, idx: TupleInt):
    orig_x = copy(x)
    assume_value(x, value)
    yield rewrite(array_value(x, idx)).to(value)
    yield rewrite(x.shape).to(orig_x.shape)
    yield rewrite(x.dtype).to(orig_x.dtype)


@egraph.function(mutates_first_arg=True)
def assume_isfinite(x: NDArray) -> None:
    """
    Asserts that the scalar ndarray is non null and not infinite.
    """
    ...


# @egraph.register
# def _isfinite(x: NDArray):
#     # orig_x = copy(x)
#     assume_isfinite(x)
#     isfinite(x)
# yield rewrite(x.isfinite()).to(TRUE)


@egraph.function(mutates_first_arg=True)
def assume_value_one_of(x: NDArray, values: TupleValue) -> None:
    """
    A value that is one of the values in the tuple.
    """
    ...


# @egraph.function
# def possible_values(v: Value) -> TupleValue:
#     """
#     Possible values of a value.
#     """
#     ...


# @egraph.register
# def _possible_values(v: Value, vs: TupleValue):
#     yield rewrite(possible_values(value_one_of(vs))).to(vs)


ALL_INDICES: TupleInt = egraph.constant("ALL_INDICES", TupleInt)


@egraph.register
def _ndarray_value_isfinite(arr: NDArray, x: Value, xs: TupleValue, i: Int, f: f64, b: Bool):
    # yield rewrite(value_one_of(TupleValue(x) + xs).isfinite()).to(x.isfinite() & value_one_of(xs).isfinite())
    # yield rewrite(value_one_of(TupleValue(x))).to(x)
    # yield rewrite(value_one_of(TupleValue.EMPTY).isfinite()).to(TRUE)

    yield rewrite(Value.int(i).isfinite()).to(TRUE)
    yield rewrite(Value.bool(b).isfinite()).to(TRUE)
    yield rewrite(Value.float(Float(f)).isfinite()).to(TRUE, f != f64(math.nan))

    # a sum of an array is finite if all the values are finite
    yield rewrite(isfinite(sum(arr))).to(NDArray.scalar(Value.bool(array_value(arr, ALL_INDICES).isfinite())))


# @egraph.register
# def _unique(xs: TupleValue, a: NDArray):
# unique_values should use value_one_of
# yield rewrite(unique_values(x=a)).to(NDArray.vector(possible_values(array_value(a, ALL_INDICES))))


@egraph.register
def _size(x: NDArray):
    yield rewrite(x.size).to(x.shape.product())


##
# Functionality to compile expression to strings of NumPy code.
# Depends on `np` as a global variable.
##


@egraph.function(merge=lambda old, new: new, default=i64(0))
def gensym() -> i64:
    ...


gensym_var = join("_", gensym().to_string())


def add_line(*v: StringLike) -> Action:
    return set_(statements()).to(join("    ", *v, "\n"))


incr_gensym = set_(gensym()).to(gensym() + 1)


@egraph.function(merge=lambda old, new: join(old, new), default=String(""))
def statements() -> String:
    ...


@egraph.function()
def ndarray_expr(x: NDArray) -> String:
    ...


@egraph.function()
def dtype_expr(x: DType) -> String:
    ...


@egraph.function()
def tuple_int_expr(x: TupleInt) -> String:
    ...


@egraph.function()
def int_expr(x: Int) -> String:
    ...


@egraph.function()
def tuple_value_expr(x: TupleValue) -> String:
    ...


@egraph.function()
def value_expr(x: Value) -> String:
    ...


egraph.register(
    set_(dtype_expr(DType.float64)).to(String("np.float64")),
    set_(dtype_expr(DType.int64)).to(String("np.int64")),
)


@egraph.function
def bool_expr(x: Bool) -> String:
    ...


egraph.register(
    set_(bool_expr(TRUE)).to(String("True")),
    set_(bool_expr(FALSE)).to(String("False")),
)


@egraph.function
def float_expr(x: Float) -> String:
    ...


@egraph.function(merge=lambda old, new: old | new)
def traversed() -> Set[NDArray]:
    """
    Global set of all traversed arrays.
    """
    ...


def traverse(x: NDArray) -> Action:
    return set_(traversed()).to(Set(x))


def not_traversed(x: NDArray) -> Unit:
    return traversed().not_contains(x)


egraph.register(
    set_(traversed()).to(Set[NDArray].empty()),
)


@egraph.function
def tuple_ndarray_expr(x: TupleNDArray) -> String:
    ...


@egraph.register
def _py_expr(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    s: String,
    y_str: String,
    z_str: String,
    dtype_str: String,
    dtype: DType,
    ti: TupleInt,
    ti1: TupleInt,
    ti2: TupleInt,
    ti_str: String,
    ti_str1: String,
    ti_str2: String,
    tv_str: String,
    tv1_str: String,
    tv2_str: String,
    i: Int,
    i_str: String,
    i64_: i64,
    tv: TupleValue,
    tv1: TupleValue,
    tv2: TupleValue,
    v: Value,
    v_str: String,
    b: Bool,
    f: Float,
    f_str: String,
    b_str: String,
    f64_: f64,
    ob: OptionalBool,
    tnd: TupleNDArray,
    tnd_str: String,
):
    # Var
    yield rule(
        eq(x).to(NDArray.var(s)),
    ).then(
        set_(lhs=ndarray_expr(x)).to(s),
    )

    # Asssume dtype
    z_assumed_dtype = copy(z)
    assume_dtype(z_assumed_dtype, dtype=dtype)
    yield rule(
        eq(x).to(z_assumed_dtype),
        eq(z_str).to(ndarray_expr(z)),
        eq(dtype_str).to(dtype_expr(dtype)),
    ).then(
        set_(ndarray_expr(x)).to(z_str),
        add_line("assert ", z_str, ".dtype == ", dtype_str),
    )

    # assume shape
    z_assumed_shape = copy(z)
    assume_shape(z_assumed_shape, ti)
    yield rule(
        eq(x).to(z_assumed_shape),
        eq(z_str).to(ndarray_expr(z)),
        eq(ti_str).to(tuple_int_expr(ti)),
    ).then(
        set_(ndarray_expr(x)).to(z_str),
        add_line("assert ", z_str, ".shape == ", ti_str),
    )
    # tuple int
    yield rule(
        eq(ti).to(ti1 + ti2),
        eq(ti_str1).to(tuple_int_expr(ti1)),
        eq(ti_str2).to(tuple_int_expr(ti2)),
    ).then(
        set_(tuple_int_expr(ti)).to(join(ti_str1, " + ", ti_str2)),
    )
    yield rule(
        eq(ti).to(TupleInt(i)),
        eq(i_str).to(int_expr(i)),
    ).then(
        set_(tuple_int_expr(ti)).to(join("(", i_str, ",)")),
    )
    # Int
    yield rule(
        eq(i).to(Int(i64_)),
    ).then(
        set_(int_expr(i)).to(i64_.to_string()),
    )

    # assume isfinite
    z_assumed_isfinite = copy(z)
    assume_isfinite(z_assumed_isfinite)
    yield rule(
        eq(x).to(z_assumed_isfinite),
        eq(z_str).to(ndarray_expr(z)),
    ).then(
        set_(ndarray_expr(x)).to(z_str),
        add_line("assert np.all(np.isfinite(", z_str, "))"),
    )

    # Assume value_one_of
    z_assumed_value_one_of = copy(z)
    assume_value_one_of(z_assumed_value_one_of, tv)
    yield rule(
        eq(x).to(z_assumed_value_one_of),
        not_traversed(x),
        eq(z_str).to(ndarray_expr(z)),
        eq(tv_str).to(tuple_value_expr(tv)),
    ).then(
        set_(ndarray_expr(x)).to(z_str),
        traverse(x),
        add_line("assert set(", z_str, ".flatten()) == set(", tv_str, ")"),
    )
    # print(r._to_egg_command(egraph._mod_decls))
    # yield r
    # tuple values
    yield rule(
        eq(tv).to(tv1 + tv2),
        eq(tv1_str).to(tuple_value_expr(tv1)),
        eq(tv2_str).to(tuple_value_expr(tv2)),
    ).then(
        set_(tuple_value_expr(tv)).to(join(tv1_str, " + ", tv2_str)),
    )
    yield rule(
        eq(tv).to(TupleValue(v)),
        eq(v_str).to(value_expr(v)),
    ).then(
        set_(tuple_value_expr(tv)).to(join("(", v_str, ",)")),
    )

    # Value
    yield rule(
        eq(v).to(Value.int(i)),
        eq(i_str).to(int_expr(i)),
    ).then(
        set_(value_expr(v)).to(i_str),
    )
    yield rule(
        eq(v).to(Value.bool(b)),
        eq(b_str).to(bool_expr(b)),
    ).then(
        set_(value_expr(v)).to(b_str),
    )
    yield rule(
        eq(v).to(Value.float(f)),
        eq(f_str).to(float_expr(f)),
    ).then(
        set_(value_expr(v)).to(f_str),
    )

    # Float
    yield rule(
        eq(f).to(Float(f64_)),
    ).then(
        set_(float_expr(f)).to(f64_.to_string()),
    )

    # reshape (don't include copy, since not present in numpy)
    yield rule(
        eq(x).to(reshape(y, ti, ob)),
        eq(y_str).to(ndarray_expr(y)),
        eq(ti_str).to(tuple_int_expr(ti)),
    ).then(
        set_(ndarray_expr(x)).to(gensym_var),
        add_line(gensym_var, " = ", y_str, ".reshape(", ti_str, ")"),
        incr_gensym,
    )

    # astype
    yield rule(
        eq(x).to(astype(y, dtype)),
        eq(y_str).to(ndarray_expr(y)),
        eq(dtype_str).to(dtype_expr(dtype)),
    ).then(
        set_(ndarray_expr(x)).to(gensym_var),
        add_line(gensym_var, " = ", y_str, ".astype(", dtype_str, ")"),
        incr_gensym,
    )

    # unique_counts(x) => unique(x, return_counts=True)
    yield rule(
        eq(tnd).to(unique_counts(y)),
        eq(y_str).to(ndarray_expr(y)),
    ).then(
        set_(tuple_ndarray_expr(tnd)).to(gensym_var),
        add_line(gensym_var, " = np.unique(", y_str, ", return_counts=True)"),
        incr_gensym,
    )
    # Tuple ndarray indexing
    yield rule(
        eq(x).to(tnd[i]),
        eq(tnd_str).to(tuple_ndarray_expr(tnd)),
        eq(i_str).to(int_expr(i)),
    ).then(
        set_(ndarray_expr(x)).to(join(tnd_str, "[", i_str, "]")),
    )

    # ndarray scalar
    # TODO: Use dtype and shape and indexing instead?
    yield rule(
        eq(x).to(NDArray.scalar(v)),
        eq(v_str).to(value_expr(v)),
    ).then(
        set_(ndarray_expr(x)).to(gensym_var),
        add_line(gensym_var, " = np.array(", v_str, ")"),
        incr_gensym,
    )

    # NDARRAy ops

    yield rule(
        eq(x).to(y + z),
        eq(y_str).to(ndarray_expr(y)),
        eq(z_str).to(ndarray_expr(z)),
    ).then(
        set_(ndarray_expr(x)).to(gensym_var),
        add_line(gensym_var, " = ", y_str, " + ", z_str),
        incr_gensym,
    )

    yield rule(
        eq(x).to(y / z),
        eq(y_str).to(ndarray_expr(y)),
        eq(z_str).to(ndarray_expr(z)),
    ).then(
        set_(ndarray_expr(x)).to(gensym_var),
        add_line(gensym_var, " = ", y_str, " / ", z_str),
        incr_gensym,
    )


@egraph.class_
class FunctionExprTwo(Expr):
    """
    Python expression that takes two NDArrays as arguments and returns an NDArray.
    """

    def __init__(self, name: StringLike, res: NDArray, arg_1: NDArray, arg_2: NDArray) -> None:
        ...

    @property
    def source(self) -> String:
        ...


fn_ruleset = egraph.ruleset("fn")


@egraph.register
def _function_expr(name: String, res: NDArray, arg1: String, arg2: String, f: FunctionExprTwo, s: String):
    yield rule(
        eq(f).to(FunctionExprTwo(name, res, NDArray.var(arg1), NDArray.var(arg2))),
        ruleset=fn_ruleset,
    ).then(
        set_(f.source).to(
            join("def ", name, "(", arg1, ", ", arg2, "):\n", statements(), "    return ", ndarray_expr(res), "\n")
        ),
    )


def test_ndarray_string():
    _NDArray_1 = NDArray.var("X")
    X_orig = copy(_NDArray_1)
    assume_dtype(_NDArray_1, DType.float64)
    assume_shape(_NDArray_1, TupleInt(Int(150)) + TupleInt(Int(4)))

    _NDArray_2 = NDArray.var("y")
    Y_orig = copy(_NDArray_2)

    assume_dtype(_NDArray_2, int64)
    assume_shape(_NDArray_2, TupleInt(150))  # type: ignore
    assume_value_one_of(_NDArray_2, (0, 1, 2))  # type: ignore

    _NDArray_3 = reshape(_NDArray_2, TupleInt(Int(-1)))
    _NDArray_4 = astype(unique_counts(_NDArray_3)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))

    res = _NDArray_4 + _NDArray_1
    fn = FunctionExprTwo("my_fn", res, X_orig, Y_orig)
    egraph.register(fn)

    egraph.run((run() * 20).saturate())
    # while egraph.run((run())).updated:
    #     print(egraph.load_object(egraph.extract(PyObject.from_string(statements()))))
    egraph.graphviz.render(view=True)

    egraph.run(run(fn_ruleset))

    fn_source = egraph.load_object(egraph.extract(PyObject.from_string(fn.source)))

    locals = {}
    globals = {"np": np}
    exec(fn_source, globals, locals) # type: ignore
    fn = locals["my_fn"]
    return fn_source


# print(fn(np.arange(10), np.arange(10)))
