# mypy: disable-error-code=empty-body

from __future__ import annotations

import itertools
import numbers
import sys
from typing import Any, ClassVar, TypeVar

import numpy as np
from egglog import *

# Pretend that exprs are numbers b/c scikit learn does isinstance checks
from egglog.runtime import RuntimeExpr

numbers.Integral.register(RuntimeExpr)

egraph = EGraph()

T = TypeVar("T", bound=Expr)

runtime_ruleset = egraph.ruleset("runtime")


def extract_py(e: Expr) -> Any:
    print(e)
    egraph.register(e)
    egraph.run(run(limit=10).saturate())
    final_object = egraph.extract(e)
    print(f"  -> {final_object}")
    with egraph:
        egraph.run((run(runtime_ruleset, limit=10) + run(limit=10)).saturate())
        print(f"     -> {egraph.extract(final_object)}")
        res = egraph.load_object(egraph.extract(final_object.to_py()))  # type: ignore[attr-defined]
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
    ]


@egraph.class_
class DType(Expr):
    float64: ClassVar[DType]
    float32: ClassVar[DType]
    int64: ClassVar[DType]
    object: ClassVar[DType]

    def __eq__(self, other: DType) -> Bool:  # type: ignore[override]
        ...


float64 = DType.float64
float32 = DType.float32
int64 = DType.int64

converter(type, DType, lambda x: convert(np.dtype(x), DType))
converter(type(np.dtype), DType, lambda x: getattr(DType, x.name))  # type: ignore[call-overload]
egraph.register(
    *(
        rewrite(l == r).to(TRUE if expr_parts(l) == expr_parts(r) else FALSE)
        for l, r in itertools.product([DType.float64, DType.float32, DType.object, DType.int64], repeat=2)
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


converter(np.dtype, IsDtypeKind, lambda x: IsDtypeKind.dtype(convert(x, DType)))
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
        rewrite(isdtype(DType.float32, IsDtypeKind.string("real floating"))).to(TRUE),
        rewrite(isdtype(DType.float64, IsDtypeKind.string("real floating"))).to(TRUE),
        rewrite(isdtype(DType.object, IsDtypeKind.string("real floating"))).to(FALSE),
        rewrite(isdtype(DType.int64, IsDtypeKind.string("real floating"))).to(FALSE),
        rewrite(isdtype(DType.float32, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.float64, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.object, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(DType.int64, IsDtypeKind.string("complex floating"))).to(FALSE),
        rewrite(isdtype(d, IsDtypeKind.NULL)).to(FALSE),
        rewrite(isdtype(d, IsDtypeKind.dtype(d))).to(TRUE),
        rewrite(isdtype(d, k1 | k2)).to(isdtype(d, k1) | isdtype(d, k2)),
        rewrite(k1 | IsDtypeKind.NULL).to(k1),
    ]


assert not bool(isdtype(DType.float32, IsDtypeKind.string("integral")))


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

    # Make != always return a Bool, so that numpy.unique works on a tuple of ints
    # In _unique1d
    @egraph.method(preserve=True)
    def __ne__(self, other: Int) -> bool:  # type: ignore[override]
        return not extract_py(self == other)

    def __eq__(self, other: Int) -> Bool:  # type: ignore[override]
        ...

    def __ge__(self, other: Int) -> Bool:
        ...

    def __lt__(self, other: Int) -> Bool:
        ...

    def __gt__(self, other: Int) -> Bool:
        ...

    def __add__(self, other: Int) -> Int:
        ...

    def __sub__(self, other: Int) -> Int:
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


converter(int, Int, lambda x: Int(x))

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


converter(tuple, TupleInt, lambda x: TupleInt(convert(x[0], Int)) + convert(x[1:], TupleInt) if x else TupleInt.EMPTY)


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
    ]


# HANDLED_FUNCTIONS = {}


@egraph.class_
class IndexKey(Expr):
    @classmethod
    def tuple_int(cls, ti: TupleInt) -> IndexKey:
        ...

    @classmethod
    def int(cls, i: Int) -> IndexKey:
        ...


converter(tuple, IndexKey, lambda x: IndexKey.tuple_int(convert(x, TupleInt)))
converter(int, IndexKey, lambda x: IndexKey.int(Int(x)))
converter(Int, IndexKey, lambda x: IndexKey.int(x))


@egraph.class_
class Device(Expr):
    ...


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

    def __getitem__(self, key: IndexKey) -> NDArray:
        ...

    def __setitem__(self, key: IndexKey, value: NDArray) -> None:
        ...

    def __truediv__(self, other: NDArray) -> NDArray:
        ...

    def __sub__(self, other: NDArray) -> NDArray:
        ...

    def __add__(self, other: NDArray) -> NDArray:
        ...

    def __lt__(self, other: NDArray) -> NDArray:
        ...

    def __gt__(self, other: NDArray) -> NDArray:
        ...

    def __eq__(self, other: NDArray) -> NDArray:  # type: ignore[override]
        ...

    @classmethod
    def scalar_float(cls, other: Float) -> NDArray:
        ...

    @classmethod
    def scalar_int(cls, other: Int) -> NDArray:
        ...

    @classmethod
    def scalar_bool(cls, other: Bool) -> NDArray:
        ...


@egraph.function
def ndarray_index(x: NDArray) -> IndexKey:
    ...


converter(NDArray, IndexKey, ndarray_index)


converter(float, NDArray, lambda x: NDArray.scalar_float(Float(x)))
converter(int, NDArray, lambda x: NDArray.scalar_int(Int(x)))


@egraph.register
def _ndarray(x: NDArray, b: Bool, f: Float, fi1: f64, fi2: f64):
    return [
        rewrite(x.ndim).to(x.shape.length()),
        rewrite(NDArray.scalar_bool(b).to_bool()).to(b),
        # TODO: Push these down to float
        rewrite(NDArray.scalar_float(f) / NDArray.scalar_float(f)).to(NDArray.scalar_float(Float(1.0))),
        rewrite(NDArray.scalar_float(f) - NDArray.scalar_float(f)).to(NDArray.scalar_float(Float(0.0))),
        rewrite(NDArray.scalar_float(Float(fi1)) > NDArray.scalar_float(Float(fi2))).to(
            NDArray.scalar_bool(TRUE), fi1 > fi2
        ),
        rewrite(NDArray.scalar_float(Float(fi1)) > NDArray.scalar_float(Float(fi2))).to(
            NDArray.scalar_bool(FALSE), fi1 <= fi2
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
    lambda x: TupleNDArray(convert(x[0], NDArray)) + convert(x[1:], TupleNDArray) if x else TupleNDArray.EMPTY,
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
converter(bool, OptionalBool, lambda x: OptionalBool.some(convert(x, Bool)))


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
converter(int, OptionalTupleInt, lambda x: OptionalTupleInt.some(TupleInt(Int(x))))


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
def sum(x: NDArray) -> NDArray:
    ...


@egraph.register
def _sum(x: NDArray, y: NDArray, f: Float, dtype: DType):
    return [
        rewrite(sum(x / NDArray.scalar_float(f))).to(sum(x) / NDArray.scalar_float(f)),
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
def concat(arrays: TupleNDArray) -> NDArray:
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
        rewrite(sum(unique_counts(x)[Int(1)])).to(NDArray.scalar_int(x.size)),
    ]


@egraph.function
def astype(x: NDArray, dtype: DType) -> NDArray:
    ...


@egraph.register
def _astype(x: NDArray, dtype: DType, i: i64):
    return [
        rewrite(astype(x, dtype).dtype).to(dtype),
        rewrite(sum(astype(x, dtype))).to(astype(sum(x), dtype)),
        rewrite(astype(NDArray.scalar_int(Int(i)), float64)).to(NDArray.scalar_float(Float(f64.from_i64(i)))),
    ]


@egraph.function
def any(x: NDArray) -> NDArray:
    ...


@egraph.function(egg_fn="ndarray-abs")
def abs(x: NDArray) -> NDArray:
    ...


@egraph.register
def _abs(f: Float):
    return [
        rewrite(abs(NDArray.scalar_float(f))).to(NDArray.scalar_float(f)),
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
def mean(x: NDArray, axis: OptionalTupleInt = OptionalTupleInt.none) -> NDArray:
    ...


linalg = sys.modules[__name__]


@egraph.function
def svd(x: NDArray) -> TupleNDArray:
    ...


@egraph.register
def _linalg(x: NDArray):
    return [
        rewrite(svd(x).length()).to(Int(3)),
    ]


##
# Interval analysis
#
# to analyze `any(((astype(unique_counts(NDArray.var("y"))[Int(1)], DType.float64) / NDArray.scalar_float(Float(150.0))) < NDArray.scalar_int(Int(0)))).bool()``
##


@egraph.function
def ndarray_all_greater_0(x: NDArray) -> Unit:
    ...


@egraph.function
def ndarray_all_false(x: NDArray) -> Unit:
    ...


@egraph.register
def _interval_analaysis(x: NDArray, y: NDArray, z: NDArray, dtype: DType, f: f64):
    return [
        rule(
            eq(y).to(x < NDArray.scalar_int(Int(0))),
            ndarray_all_greater_0(x),
        ).then(ndarray_all_false(y)),
        rule(
            eq(y).to(any(x)),
            ndarray_all_false(x),
        ).then(union(y).with_(NDArray.scalar_bool(FALSE))),
        rule(
            eq(y).to(unique_counts(x)[Int(1)]),
        ).then(ndarray_all_greater_0(y)),
        rule(eq(y).to(astype(x, dtype)), ndarray_all_greater_0(x)).then(ndarray_all_greater_0(y)),
        rule(eq(z).to(x / y), ndarray_all_greater_0(x), ndarray_all_greater_0(y)).then(ndarray_all_greater_0(z)),
        rule(eq(z).to(NDArray.scalar_float(Float(f))), f > 0.0).then(ndarray_all_greater_0(z)),
    ]
