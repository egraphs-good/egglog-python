_NDArray_1 = NDArray.var('X')
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt(Vec[Int](Int(150), Int(4))))
assume_isfinite(_NDArray_1)
_NDArray_2 = NDArray.var('y')
assume_dtype(_NDArray_2, DType.int64)
assume_shape(_NDArray_2, TupleInt(Vec[Int](Int(150))))
assume_value_one_of(_NDArray_2, TupleValue(Vec[Value](Value.from_int(Int(0)), Value.from_int(Int(1)), Value.from_int(Int(2)))))
_NDArray_3 = astype(unique_counts_counts(_NDArray_2), asarray(_NDArray_1).dtype) / NDArray(RecursiveValue(Value.from_float(Float(150.0))))
_NDArray_4 = zeros(
    TupleInt(
        Vec[Int](
            NDArray(
                RecursiveValue.vec(Vec[RecursiveValue](RecursiveValue(Value.from_int(Int(0))), RecursiveValue(Value.from_int(Int(1))), RecursiveValue(Value.from_int(Int(2)))))
            ).shape[Int(0)],
            asarray(_NDArray_1).shape[Int(1)],
        )
    ),
    OptionalDType.some(asarray(_NDArray_1).dtype),
    OptionalDevice.some(Device.cpu),
)
_NDArray_5 = NDArray.var('y')
assume_dtype(_NDArray_5, DType.int64)
assume_shape(_NDArray_5, TupleInt(Vec[Int](Int(150))))
assume_value_one_of(_NDArray_5, TupleValue(Vec[Value](Value.from_int(Int(0)), Value.from_int(Int(1)), Value.from_int(Int(2)))))
_NDArray_4[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(0)), MultiAxisIndexKeyItem.slice(Slice()))))] = mean(
    asarray(_NDArray_1)[IndexKey.ndarray(unique_inverse_inverse_indices(_NDArray_5) == NDArray(RecursiveValue(Value.from_int(Int(0)))))], OptionalIntOrTuple.int(Int(0))
)
_NDArray_4[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(1)), MultiAxisIndexKeyItem.slice(Slice()))))] = mean(
    asarray(_NDArray_1)[IndexKey.ndarray(unique_inverse_inverse_indices(_NDArray_5) == NDArray(RecursiveValue(Value.from_int(Int(1)))))], OptionalIntOrTuple.int(Int(0))
)
_NDArray_4[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(2)), MultiAxisIndexKeyItem.slice(Slice()))))] = mean(
    asarray(_NDArray_1)[IndexKey.ndarray(unique_inverse_inverse_indices(_NDArray_5) == NDArray(RecursiveValue(Value.from_int(Int(2)))))], OptionalIntOrTuple.int(Int(0))
)
_NDArray_6 = NDArray.var('y')
assume_dtype(_NDArray_6, DType.int64)
assume_shape(_NDArray_6, TupleInt(Vec[Int](Int(150))))
assume_value_one_of(_NDArray_6, TupleValue(Vec[Value](Value.from_int(Int(0)), Value.from_int(Int(1)), Value.from_int(Int(2)))))
_NDArray_7 = asarray(reshape(asarray(_NDArray_6), TupleInt(Vec[Int](Int(-1)))))
_NDArray_8 = concat(
    TupleNDArray(
        Vec[NDArray](
            asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_7 == NDArray(RecursiveValue(Value.from_int(Int(0)))))]
            - _NDArray_4[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(0)), MultiAxisIndexKeyItem.slice(Slice()))))],
            asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_7 == NDArray(RecursiveValue(Value.from_int(Int(1)))))]
            - _NDArray_4[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(1)), MultiAxisIndexKeyItem.slice(Slice()))))],
            asarray(_NDArray_1)[IndexKey.ndarray(_NDArray_7 == NDArray(RecursiveValue(Value.from_int(Int(2)))))]
            - _NDArray_4[IndexKey.multi_axis(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.int(Int(2)), MultiAxisIndexKeyItem.slice(Slice()))))],
        )
    ),
    OptionalInt.some(Int(0)),
)
_NDArray_9 = std(_NDArray_8, OptionalIntOrTuple.int(Int(0)))
_NDArray_9[IndexKey.ndarray(std(_NDArray_8, OptionalIntOrTuple.int(Int(0))) == NDArray(RecursiveValue(Value.from_int(Int(0)))))] = NDArray(
    RecursiveValue(Value.from_float(Float(1.0)))
)
_TupleNDArray_1 = svd_(
    sqrt(
        asarray(
            NDArray(RecursiveValue(Value.from_float(Float(1.0) / Float.from_int(Int(150))))),
            OptionalDType.some(asarray(_NDArray_1).dtype),
            OptionalBool.none,
            OptionalDevice.some(Device.cpu),
        )
    )
    * (_NDArray_8 / _NDArray_9),
    Boolean(False),
)
_OptionalInt_1 = OptionalInt.none
_NDArray_10 = sum(astype(_TupleNDArray_1[Int(1)] > NDArray(RecursiveValue(Value.from_float(Float(0.0001)))), DType.int32))
_NDArray_11 = (
    _TupleNDArray_1[Int(2)][
        IndexKey.multi_axis(
            MultiAxisIndexKey.from_vec(
                Vec[MultiAxisIndexKeyItem](
                    MultiAxisIndexKeyItem.slice(Slice(_OptionalInt_1, OptionalInt.some(_NDArray_10.index(TupleInt()).to_int))), MultiAxisIndexKeyItem.slice(Slice())
                )
            )
        )
    ]
    / _NDArray_9
).T / _TupleNDArray_1[Int(1)][IndexKey.slice(Slice(_OptionalInt_1, OptionalInt.some(_NDArray_10.index(TupleInt()).to_int)))]
_TupleNDArray_2 = svd_(
    (
        sqrt(
            (NDArray(RecursiveValue(Value.from_int(Int(150)))) * _NDArray_3)
            * NDArray(
                RecursiveValue(
                    Value.from_float(Float(1.0) / Float.from_int(unique_values(concat(TupleNDArray(Vec[NDArray](unique_values(asarray(_NDArray_7)))))).shape[Int(0)] - Int(1)))
                )
            )
        )
        * (_NDArray_4 - (_NDArray_3 @ _NDArray_4)).T
    ).T
    @ _NDArray_11,
    Boolean(False),
)
(
    (asarray(_NDArray_1) - (_NDArray_3 @ _NDArray_4))
    @ (
        _NDArray_11
        @ _TupleNDArray_2[Int(2)].T[
            IndexKey.multi_axis(
                MultiAxisIndexKey.from_vec(
                    Vec[MultiAxisIndexKeyItem](
                        MultiAxisIndexKeyItem.slice(Slice()),
                        MultiAxisIndexKeyItem.slice(
                            Slice(
                                _OptionalInt_1,
                                OptionalInt.some(
                                    sum(
                                        astype(
                                            _TupleNDArray_2[Int(1)] > (NDArray(RecursiveValue(Value.from_float(Float(0.0001)))) * _TupleNDArray_2[Int(1)][IndexKey.int(Int(0))]),
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
)[
    IndexKey.multi_axis(
        MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem](MultiAxisIndexKeyItem.slice(Slice()), MultiAxisIndexKeyItem.slice(Slice(_OptionalInt_1, OptionalInt.some(Int(2))))))
    )
]