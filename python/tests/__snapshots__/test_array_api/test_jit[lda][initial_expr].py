_NDArray_1 = NDArray.var("X")
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt(Vec(Int(150), Int(4))))
assume_isfinite(_NDArray_1)
_NDArray_2 = NDArray.var("y")
assume_dtype(_NDArray_2, DType.int64)
assume_shape(_NDArray_2, TupleInt(Vec(Int(150))))
assume_value_one_of(_NDArray_2, TupleValue(Vec(Value.from_int(Int(0)), Value.from_int(Int(1)), Value.from_int(Int(2)))))
_NDArray_3 = astype(unique_counts_counts(_NDArray_2), asarray(_NDArray_1).dtype) / NDArray(RecursiveValue(Value.from_float(Float(150.0))))
_NDArray_4 = zeros(
    TupleInt(
        Vec(
            NDArray(RecursiveValue.vec(Vec(RecursiveValue(Value.from_int(Int(0))), RecursiveValue(Value.from_int(Int(1))), RecursiveValue(Value.from_int(Int(2)))))).shape[Int(0)],
            asarray(_NDArray_1).shape[Int(1)],
        )
    ),
    OptionalDType.some(asarray(_NDArray_1).dtype),
    OptionalDevice.some(asarray(_NDArray_1).device),
)
_MultiAxisIndexKeyItem_1 = MultiAxisIndexKeyItem.slice(Slice())
_IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.int(Int(0)), _MultiAxisIndexKeyItem_1)))
_NDArray_5 = NDArray(RecursiveValue(Value.from_int(Int(0))))
_NDArray_4[_IndexKey_1] = mean(asarray(_NDArray_1)[IndexKey.ndarray(unique_inverse_inverse_indices(_NDArray_2) == _NDArray_5)], OptionalIntOrTuple.int(Int(0)))
_IndexKey_2 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.int(Int(1)), _MultiAxisIndexKeyItem_1)))
_NDArray_4[_IndexKey_2] = mean(
    asarray(_NDArray_1)[IndexKey.ndarray(unique_inverse_inverse_indices(_NDArray_2) == NDArray(RecursiveValue(Value.from_int(Int(1)))))], OptionalIntOrTuple.int(Int(0))
)
_IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.int(Int(2)), _MultiAxisIndexKeyItem_1)))
_NDArray_4[_IndexKey_3] = mean(
    asarray(_NDArray_1)[IndexKey.ndarray(unique_inverse_inverse_indices(_NDArray_2) == NDArray(RecursiveValue(Value.from_int(Int(2)))))], OptionalIntOrTuple.int(Int(0))
)
_NDArray_6 = asarray(reshape(asarray(_NDArray_2), TupleInt(Vec(Int(-1)))))
_Int_1 = unique_values(concat(TupleNDArray(Vec(unique_values(asarray(_NDArray_6)))))).shape[Int(0)]
_NDArray_7 = concat(
    TupleNDArray(
        Vec(
            asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_6 == _NDArray_5)] - _NDArray_4[_IndexKey_1],
            asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_6 == NDArray(RecursiveValue(Value.from_int(Int(1)))))] - _NDArray_4[_IndexKey_2],
            asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_6 == NDArray(RecursiveValue(Value.from_int(Int(2)))))] - _NDArray_4[_IndexKey_3],
        )
    ),
    OptionalInt.some(Int(0)),
)
_NDArray_8 = std(_NDArray_7, OptionalIntOrTuple.int(Int(0)))
_NDArray_8[IndexKey.ndarray(std(_NDArray_7, OptionalIntOrTuple.int(Int(0))) == _NDArray_5)] = NDArray(RecursiveValue(Value.from_float(Float(1.0))))
_TupleNDArray_1 = svd_(
    sqrt(
        asarray(
            NDArray(RecursiveValue(Value.from_float(Float(1.0) / Float.from_int(Int(150) - _Int_1)))),
            OptionalDType.some(asarray(_NDArray_1).dtype),
            OptionalBool.none,
            OptionalDevice.some(asarray(_NDArray_1).device),
        )
    )
    * (_NDArray_7 / _NDArray_8),
    Boolean(False),
)
_Slice_1 = Slice(
    OptionalInt.none, OptionalInt.some(sum(astype(_TupleNDArray_1[Int(1)] > NDArray(RecursiveValue(Value.from_float(Float(0.0001)))), DType.int32)).index(TupleInt()).to_int)
)
_NDArray_9 = (
    _TupleNDArray_1[Int(2)][IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec(MultiAxisIndexKeyItem.slice(_Slice_1), _MultiAxisIndexKeyItem_1)))] / _NDArray_8
).T / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
_TupleNDArray_2 = svd_(
    (
        sqrt(NDArray(RecursiveValue(Value.from_int(Int(150)))) * _NDArray_3 * NDArray(RecursiveValue(Value.from_float(Float(1.0) / Float.from_int(_Int_1 - Int(1))))))
        * (_NDArray_4 - _NDArray_3 @ _NDArray_4).T
    ).T
    @ _NDArray_9,
    Boolean(False),
)
(
    (asarray(_NDArray_1) - _NDArray_3 @ _NDArray_4)
    @ (
        _NDArray_9
        @ _TupleNDArray_2[Int(2)].T[
            IndexKey.multi_axis(
                MultiAxisIndexKey.from_vec(
                    Vec(
                        _MultiAxisIndexKeyItem_1,
                        MultiAxisIndexKeyItem.slice(
                            Slice(
                                OptionalInt.none,
                                OptionalInt.some(
                                    sum(
                                        astype(
                                            _TupleNDArray_2[Int(1)] > NDArray(RecursiveValue(Value.from_float(Float(0.0001)))) * _TupleNDArray_2[Int(1)][IndexKey.int(Int(0))],
                                            DType.int32,
                                        )
                                    )
                                    .index(TupleInt())
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