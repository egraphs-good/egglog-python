_NDArray_1 = NDArray.var("X")
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt.from_vec(Vec[Int](Int(150), Int(4))))
assume_isfinite(_NDArray_1)
_NDArray_2 = NDArray.var("y")
assume_dtype(_NDArray_2, DType.int64)
assume_shape(_NDArray_2, TupleInt.from_vec(Vec[Int](Int(150))))
assume_value_one_of(_NDArray_2, TupleValue.from_vec(Vec[Value](Value.int(Int(0)), Value.int(Int(1)), Value.int(Int(2)))))
_NDArray_3 = astype(
    NDArray.vector(
        TupleValue.from_vec(
            Vec[Value](
                sum(_NDArray_2 == NDArray.scalar(Value.int(Int(0)))).to_value(),
                sum(_NDArray_2 == NDArray.scalar(Value.int(Int(1)))).to_value(),
                sum(_NDArray_2 == NDArray.scalar(Value.int(Int(2)))).to_value(),
            )
        )
    ),
    DType.float64,
) / NDArray.scalar(Value.float(Float.rational(BigRat(BigInt.from_string("150"), BigInt.from_string("1")))))
_NDArray_4 = zeros(TupleInt.from_vec(Vec[Int](Int(3), Int(4))), OptionalDType.some(DType.float64), OptionalDevice.some(_NDArray_1.device))
_MultiAxisIndexKeyItem_1 = MultiAxisIndexKeyItem.slice(Slice())
_IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(0)), _MultiAxisIndexKeyItem_1)))
_NDArray_5 = _NDArray_1[IndexKey.ndarray(_NDArray_2 == NDArray.scalar(Value.int(Int(0))))]
_OptionalIntOrTuple_1 = OptionalIntOrTuple.some(IntOrTuple.int(Int(0)))
_NDArray_4[_IndexKey_1] = sum(_NDArray_5, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_5.shape[Int(0)]))
_IndexKey_2 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(1)), _MultiAxisIndexKeyItem_1)))
_NDArray_6 = _NDArray_1[IndexKey.ndarray(_NDArray_2 == NDArray.scalar(Value.int(Int(1))))]
_NDArray_4[_IndexKey_2] = sum(_NDArray_6, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_6.shape[Int(0)]))
_IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(2)), _MultiAxisIndexKeyItem_1)))
_NDArray_7 = _NDArray_1[IndexKey.ndarray(_NDArray_2 == NDArray.scalar(Value.int(Int(2))))]
_NDArray_4[_IndexKey_3] = sum(_NDArray_7, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_7.shape[Int(0)]))
_NDArray_8 = concat(
    TupleNDArray.from_vec(Vec[NDArray](_NDArray_5 - _NDArray_4[_IndexKey_1], _NDArray_6 - _NDArray_4[_IndexKey_2], _NDArray_7 - _NDArray_4[_IndexKey_3])), OptionalInt.some(Int(0))
)
_NDArray_9 = square(_NDArray_8 - expand_dims(sum(_NDArray_8, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_8.shape[Int(0)]))))
_NDArray_10 = sqrt(sum(_NDArray_9, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_9.shape[Int(0)])))
_NDArray_11 = copy(_NDArray_10)
_NDArray_11[IndexKey.ndarray(_NDArray_10 == NDArray.scalar(Value.int(Int(0))))] = NDArray.scalar(
    Value.float(Float.rational(BigRat(BigInt.from_string("1"), BigInt.from_string("1"))))
)
_TupleNDArray_1 = svd(
    sqrt(asarray(NDArray.scalar(Value.float(Float.rational(BigRat(BigInt.from_string("1"), BigInt.from_string("147"))))), OptionalDType.some(DType.float64)))
    * (_NDArray_8 / _NDArray_11),
    Boolean(False),
)
_Slice_1 = Slice(OptionalInt.none, OptionalInt.some(sum(astype(_TupleNDArray_1[Int(1)] > NDArray.scalar(Value.float(Float(0.0001))), DType.int32)).to_value().to_int))
_NDArray_12 = (
    _TupleNDArray_1[Int(2)][IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.slice(_Slice_1), _MultiAxisIndexKeyItem_1)))]
    / _NDArray_11
).T / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
_TupleNDArray_2 = svd(
    (
        sqrt((NDArray.scalar(Value.int(Int(150))) * _NDArray_3) * NDArray.scalar(Value.float(Float.rational(BigRat(BigInt.from_string("1"), BigInt.from_string("2"))))))
        * (_NDArray_4 - (_NDArray_3 @ _NDArray_4)).T
    ).T
    @ _NDArray_12,
    Boolean(False),
)
(
    (_NDArray_1 - (_NDArray_3 @ _NDArray_4))
    @ (
        _NDArray_12
        @ _TupleNDArray_2[Int(2)].T[
            IndexKey.multi_axis(
                MultiAxisIndexKey.from_vec(
                    Vec[MultiAxisIndexKeyItem](
                        _MultiAxisIndexKeyItem_1,
                        MultiAxisIndexKeyItem.slice(
                            Slice(
                                OptionalInt.none,
                                OptionalInt.some(
                                    sum(astype(_TupleNDArray_2[Int(1)] > (NDArray.scalar(Value.float(Float(0.0001))) * _TupleNDArray_2[Int(1)][IndexKey.int(Int(0))]), DType.int32))
                                    .to_value()
                                    .to_int
                                ),
                            )
                        ),
                    )
                )
            )
        ]
    )
)[
    IndexKey.multi_axis(
        MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](_MultiAxisIndexKeyItem_1, MultiAxisIndexKeyItem.slice(Slice(OptionalInt.none, OptionalInt.some(Int(2))))))
    )
]