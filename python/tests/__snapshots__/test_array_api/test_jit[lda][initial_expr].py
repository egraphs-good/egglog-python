_NDArray_1 = NDArray.var("X")
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt.from_vec(Vec[Int](Int(150), Int(4))))
assume_isfinite(_NDArray_1)
_NDArray_2 = NDArray.var("y")
assume_dtype(_NDArray_2, DType.int64)
assume_shape(_NDArray_2, TupleInt.from_vec(Vec[Int](Int(150))))
_TupleValue_1 = TupleValue.from_vec(Vec[Value](Value.int(Int(0)), Value.int(Int(1)), Value.int(Int(2))))
assume_value_one_of(_NDArray_2, _TupleValue_1)
_NDArray_3 = zeros(
    TupleInt.from_vec(Vec[Int](NDArray.vector(_TupleValue_1).shape[Int(0)], asarray(_NDArray_1).shape[Int(1)])),
    OptionalDType.some(asarray(_NDArray_1).dtype),
    OptionalDevice.some(asarray(_NDArray_1).device),
)
_MultiAxisIndexKeyItem_1 = MultiAxisIndexKeyItem.slice(Slice())
_IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.int(Int(0)), _MultiAxisIndexKeyItem_1)))
_IndexKey_2 = IndexKey.ndarray(unique_inverse(_NDArray_2)[Int(1)] == NDArray.scalar(Value.int(Int(0))))
_OptionalIntOrTuple_1 = OptionalIntOrTuple.some(IntOrTuple.int(Int(0)))
_NDArray_3[_IndexKey_1] = mean(asarray(_NDArray_1)[_IndexKey_2], _OptionalIntOrTuple_1)
_IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.int(Int(1)), _MultiAxisIndexKeyItem_1)))
_IndexKey_4 = IndexKey.ndarray(unique_inverse(_NDArray_2)[Int(1)] == NDArray.scalar(Value.int(Int(1))))
_NDArray_3[_IndexKey_3] = mean(asarray(_NDArray_1)[_IndexKey_4], _OptionalIntOrTuple_1)
_IndexKey_5 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.int(Int(2)), _MultiAxisIndexKeyItem_1)))
_IndexKey_6 = IndexKey.ndarray(unique_inverse(_NDArray_2)[Int(1)] == NDArray.scalar(Value.int(Int(2))))
_NDArray_3[_IndexKey_5] = mean(asarray(_NDArray_1)[_IndexKey_6], _OptionalIntOrTuple_1)
_NDArray_4 = zeros(TupleInt.from_vec(Vec[Int](Int(3), Int(4))), OptionalDType.some(DType.float64), OptionalDevice.some(_NDArray_1.device))
_IndexKey_7 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(0)), _MultiAxisIndexKeyItem_1)))
_NDArray_4[_IndexKey_7] = mean(_NDArray_1[_IndexKey_2], _OptionalIntOrTuple_1)
_IndexKey_8 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(1)), _MultiAxisIndexKeyItem_1)))
_NDArray_4[_IndexKey_8] = mean(_NDArray_1[_IndexKey_4], _OptionalIntOrTuple_1)
_IndexKey_9 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(2)), _MultiAxisIndexKeyItem_1)))
_NDArray_4[_IndexKey_9] = mean(_NDArray_1[_IndexKey_6], _OptionalIntOrTuple_1)
_NDArray_5 = concat(
    TupleNDArray.from_vec(
        Vec[NDArray](
            _NDArray_1[IndexKey.ndarray(_NDArray_2 == NDArray.scalar(Value.int(Int(0))))] - _NDArray_4[_IndexKey_7],
            _NDArray_1[IndexKey.ndarray(_NDArray_2 == NDArray.scalar(Value.int(Int(1))))] - _NDArray_4[_IndexKey_8],
            _NDArray_1[IndexKey.ndarray(_NDArray_2 == NDArray.scalar(Value.int(Int(2))))] - _NDArray_4[_IndexKey_9],
        )
    ),
    OptionalInt.some(Int(0)),
)
_NDArray_6 = std(_NDArray_5, _OptionalIntOrTuple_1)
_NDArray_6[IndexKey.ndarray(std(_NDArray_5, _OptionalIntOrTuple_1) == NDArray.scalar(Value.int(Int(0))))] = NDArray.scalar(
    Value.float(Float.rational(BigRat(BigInt.from_string("1"), BigInt.from_string("1"))))
)
_TupleNDArray_1 = svd(
    sqrt(asarray(NDArray.scalar(Value.float(Float.rational(BigRat(BigInt.from_string("1"), BigInt.from_string("147"))))), OptionalDType.some(DType.float64)))
    * (_NDArray_5 / _NDArray_6),
    Boolean(False),
)
_Slice_1 = Slice(OptionalInt.none, OptionalInt.some(sum(astype(_TupleNDArray_1[Int(1)] > NDArray.scalar(Value.float(Float(0.0001))), DType.int32)).to_value().to_int))
_NDArray_7 = asarray(reshape(asarray(_NDArray_2), TupleInt.from_vec(Vec[Int](Int(-1)))))
_NDArray_8 = unique_values(concat(TupleNDArray.from_vec(Vec[NDArray](unique_values(asarray(_NDArray_7))))))
_NDArray_9 = std(
    concat(
        TupleNDArray.from_vec(
            Vec[NDArray](
                asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_7 == _NDArray_8[IndexKey.int(Int(0))])] - _NDArray_3[_IndexKey_1],
                asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_7 == _NDArray_8[IndexKey.int(Int(1))])] - _NDArray_3[_IndexKey_3],
                asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_7 == _NDArray_8[IndexKey.int(Int(2))])] - _NDArray_3[_IndexKey_5],
            )
        ),
        OptionalInt.some(Int(0)),
    ),
    _OptionalIntOrTuple_1,
)
_NDArray_10 = copy(_NDArray_9)
_NDArray_10[IndexKey.ndarray(_NDArray_9 == NDArray.scalar(Value.int(Int(0))))] = NDArray.scalar(Value.float(Float(1.0)))
_NDArray_11 = astype(unique_counts(_NDArray_2)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float.rational(BigRat(BigInt.from_string("150"), BigInt.from_string("1")))))
_TupleNDArray_2 = svd(
    (
        sqrt((NDArray.scalar(Value.int(Int(150))) * _NDArray_11) * NDArray.scalar(Value.float(Float.rational(BigRat(BigInt.from_string("1"), BigInt.from_string("2"))))))
        * (_NDArray_4 - (_NDArray_11 @ _NDArray_4)).T
    ).T
    @ (
        (
            _TupleNDArray_1[Int(2)][IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.slice(_Slice_1), _MultiAxisIndexKeyItem_1)))]
            / _NDArray_6
        ).T
        / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
    ),
    Boolean(False),
)
(
    (asarray(_NDArray_1) - ((astype(unique_counts(_NDArray_2)[Int(1)], asarray(_NDArray_1).dtype) / NDArray.scalar(Value.float(Float(150.0)))) @ _NDArray_3))
    @ (
        (
            (_TupleNDArray_1[Int(2)][IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.slice(_Slice_1), _MultiAxisIndexKeyItem_1)))] / _NDArray_10).T
            / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
        )
        @ _TupleNDArray_2[Int(2)].T[
            IndexKey.multi_axis(
                MultiAxisIndexKey.from_vec(
                    Vec(
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
)[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(_MultiAxisIndexKeyItem_1, MultiAxisIndexKeyItem.slice(Slice(OptionalInt.none, OptionalInt.some(Int(2)))))))]