_NDArray_1 = NDArray.var("X")
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt(Int(150)) + TupleInt(Int(4)))
assume_isfinite(_NDArray_1)
_NDArray_2 = NDArray.var("y")
assume_dtype(_NDArray_2, DType.int64)
assume_shape(_NDArray_2, TupleInt(Int(150)))
assume_value_one_of(_NDArray_2, TupleValue(Value.int(Int(0))) + (TupleValue(Value.int(Int(1))) + TupleValue(Value.int(Int(2)))))
_NDArray_3 = asarray(reshape(asarray(_NDArray_2), TupleInt(Int(-1))))
_NDArray_4 = astype(unique_counts(_NDArray_3)[Int(1)], asarray(_NDArray_1).dtype) / NDArray.scalar(Value.float(Float(150.0)))
_NDArray_5 = zeros(
    TupleInt(unique_inverse(_NDArray_3)[Int(0)].shape[Int(0)]) + TupleInt(asarray(_NDArray_1).shape[Int(1)]),
    OptionalDType.some(asarray(_NDArray_1).dtype),
    OptionalDevice.some(asarray(_NDArray_1).device),
)
_MultiAxisIndexKey_1 = MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice()))
_IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(0))) + _MultiAxisIndexKey_1)
_OptionalIntOrTuple_1 = OptionalIntOrTuple.some(IntOrTuple.int(Int(0)))
_NDArray_5[_IndexKey_1] = mean(asarray(_NDArray_1)[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(0))))], _OptionalIntOrTuple_1)
_IndexKey_2 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(1))) + _MultiAxisIndexKey_1)
_NDArray_5[_IndexKey_2] = mean(asarray(_NDArray_1)[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(1))))], _OptionalIntOrTuple_1)
_IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(2))) + _MultiAxisIndexKey_1)
_NDArray_5[_IndexKey_3] = mean(asarray(_NDArray_1)[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(2))))], _OptionalIntOrTuple_1)
_NDArray_6 = unique_values(concat(TupleNDArray(unique_values(asarray(_NDArray_3)))))
_NDArray_7 = concat(
    TupleNDArray(asarray(_NDArray_1)[ndarray_index(_NDArray_3 == _NDArray_6[IndexKey.int(Int(0))])] - _NDArray_5[_IndexKey_1])
    + (
        TupleNDArray(asarray(_NDArray_1)[ndarray_index(_NDArray_3 == _NDArray_6[IndexKey.int(Int(1))])] - _NDArray_5[_IndexKey_2])
        + TupleNDArray(asarray(_NDArray_1)[ndarray_index(_NDArray_3 == _NDArray_6[IndexKey.int(Int(2))])] - _NDArray_5[_IndexKey_3])
    ),
    OptionalInt.some(Int(0)),
)
_NDArray_8 = std(_NDArray_7, _OptionalIntOrTuple_1)
_NDArray_8[ndarray_index(std(_NDArray_7, _OptionalIntOrTuple_1) == NDArray.scalar(Value.int(Int(0))))] = NDArray.scalar(Value.float(Float(1.0)))
_TupleNDArray_1 = svd(
    sqrt(asarray(NDArray.scalar(Value.float(Float(1.0) / Float.from_int(asarray(_NDArray_1).shape[Int(0)] - _NDArray_6.shape[Int(0)]))))) * (_NDArray_7 / _NDArray_8), FALSE
)
_Slice_1 = Slice(OptionalInt.none, OptionalInt.some(sum(astype(_TupleNDArray_1[Int(1)] > NDArray.scalar(Value.float(Float(0.0001))), DType.int32)).to_value().to_int))
_NDArray_9 = (_TupleNDArray_1[Int(2)][IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(_Slice_1)) + _MultiAxisIndexKey_1)] / _NDArray_8).T / _TupleNDArray_1[
    Int(1)
][IndexKey.slice(_Slice_1)]
_TupleNDArray_2 = svd(
    (
        sqrt(
            (NDArray.scalar(Value.int(asarray(_NDArray_1).shape[Int(0)])) * _NDArray_4)
            * NDArray.scalar(Value.float(Float(1.0) / Float.from_int(_NDArray_6.shape[Int(0)] - Int(1))))
        )
        * (_NDArray_5 - (_NDArray_4 @ _NDArray_5)).T
    ).T
    @ _NDArray_9,
    FALSE,
)
(
    (asarray(_NDArray_1) - (_NDArray_4 @ _NDArray_5))
    @ (
        _NDArray_9
        @ _TupleNDArray_2[Int(2)].T[
            IndexKey.multi_axis(
                _MultiAxisIndexKey_1
                + MultiAxisIndexKey(
                    MultiAxisIndexKeyItem.slice(
                        Slice(
                            OptionalInt.none,
                            OptionalInt.some(
                                sum(astype(_TupleNDArray_2[Int(1)] > (NDArray.scalar(Value.float(Float(0.0001))) * _TupleNDArray_2[Int(1)][IndexKey.int(Int(0))]), DType.int32))
                                .to_value()
                                .to_int
                            ),
                        )
                    )
                )
            )
        ]
    )
)[IndexKey.multi_axis(_MultiAxisIndexKey_1 + MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice(OptionalInt.none, OptionalInt.some(Int(2))))))]