_NDArray_1 = NDArray.var("X")
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt(Int(150)) + TupleInt(Int(4)))
assume_isfinite(_NDArray_1)
_NDArray_2 = NDArray.var("y")
assume_dtype(_NDArray_2, DType.int64)
assume_shape(_NDArray_2, TupleInt(Int(150)))
_TupleValue_1 = TupleValue(Value.int(Int(0))) + (TupleValue(Value.int(Int(1))) + TupleValue(Value.int(Int(2))))
assume_value_one_of(_NDArray_2, _TupleValue_1)
_NDArray_3 = reshape(_NDArray_2, TupleInt(Int(-1)))
_NDArray_4 = astype(unique_counts(_NDArray_3)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))
_NDArray_5 = zeros(TupleInt(Int(3)) + TupleInt(Int(4)), OptionalDType.some(DType.float64), OptionalDevice.some(_NDArray_1.device))
_MultiAxisIndexKey_1 = MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice()))
_IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(0))) + _MultiAxisIndexKey_1)
_NDArray_5[_IndexKey_1] = mean(_NDArray_1[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(0))))], OptionalIntOrTuple.int(Int(0)))
_IndexKey_2 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(1))) + _MultiAxisIndexKey_1)
_NDArray_5[_IndexKey_2] = mean(_NDArray_1[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(1))))], OptionalIntOrTuple.int(Int(0)))
_IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(2))) + _MultiAxisIndexKey_1)
_NDArray_5[_IndexKey_3] = mean(_NDArray_1[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(2))))], OptionalIntOrTuple.int(Int(0)))
_NDArray_6 = concat(
    TupleNDArray(_NDArray_1[ndarray_index(_NDArray_3 == NDArray.vector(_TupleValue_1)[IndexKey.int(Int(0))])] - _NDArray_5[_IndexKey_1])
    + (
        TupleNDArray(_NDArray_1[ndarray_index(_NDArray_3 == NDArray.vector(_TupleValue_1)[IndexKey.int(Int(1))])] - _NDArray_5[_IndexKey_2])
        + TupleNDArray(_NDArray_1[ndarray_index(_NDArray_3 == NDArray.vector(_TupleValue_1)[IndexKey.int(Int(2))])] - _NDArray_5[_IndexKey_3])
    ),
    OptionalInt.some(Int(0)),
)
_NDArray_7 = std(_NDArray_6, OptionalIntOrTuple.int(Int(0)))
_NDArray_7[ndarray_index(std(_NDArray_6, OptionalIntOrTuple.int(Int(0))) == NDArray.scalar(Value.int(Int(0))))] = NDArray.scalar(Value.float(Float(1.0)))
_TupleNDArray_1 = svd(sqrt(NDArray.scalar(Value.int(NDArray.scalar(Value.float(Float(1.0))).to_int() / Int(147)))) * (_NDArray_6 / _NDArray_7), FALSE)
_Slice_1 = Slice(OptionalInt.none, OptionalInt.some(astype(sum(_TupleNDArray_1[Int(1)] > NDArray.scalar(Value.float(Float(0.0001)))), DType.int32).to_int()))
_NDArray_8 = (_TupleNDArray_1[Int(2)][IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(_Slice_1)) + _MultiAxisIndexKey_1)] / _NDArray_7).T / _TupleNDArray_1[
    Int(1)
][IndexKey.slice(_Slice_1)]
_TupleNDArray_2 = svd(
    (sqrt(NDArray.scalar(Value.int((Int(150) * _NDArray_4.to_int()) * (NDArray.scalar(Value.float(Float(1.0))).to_int() / Int(2))))) * (_NDArray_5 - (_NDArray_4 @ _NDArray_5)).T).T
    @ _NDArray_8,
    FALSE,
)
(
    (_NDArray_1 - (_NDArray_4 @ _NDArray_5))
    @ (
        _NDArray_8
        @ _TupleNDArray_2[Int(2)].T[
            IndexKey.multi_axis(
                _MultiAxisIndexKey_1
                + MultiAxisIndexKey(
                    MultiAxisIndexKeyItem.slice(
                        Slice(
                            OptionalInt.none,
                            OptionalInt.some(
                                astype(
                                    sum(_TupleNDArray_2[Int(1)] > (NDArray.scalar(Value.float(Float(0.0001))) * _TupleNDArray_2[Int(1)][IndexKey.int(Int(0))])), DType.int32
                                ).to_int()
                            ),
                        )
                    )
                )
            )
        ]
    )
)[IndexKey.multi_axis(_MultiAxisIndexKey_1 + MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice(OptionalInt.none, OptionalInt.some(Int(2))))))]