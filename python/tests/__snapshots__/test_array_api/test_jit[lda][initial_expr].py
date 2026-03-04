_NDArray_1 = NDArray.var("X")
assume_dtype(_NDArray_1, DType.float64)
assume_shape(_NDArray_1, TupleInt(Vec(Int(150), Int(4))))
assume_isfinite(_NDArray_1)
_NDArray_2 = asarray(_NDArray_1)
_NDArray_3 = NDArray.var("y")
assume_dtype(_NDArray_3, DType.int64)
assume_shape(_NDArray_3, TupleInt(Vec(Int(150))))
assume_value_one_of(
    _NDArray_3,
    TupleValue(
        Vec(Value.from_int(Int(0)), Value.from_int(Int(1)), Value.from_int(Int(2)))
    ),
)
_NDArray_4 = astype(unique_counts_counts(_NDArray_3), _NDArray_2.dtype) / NDArray(
    RecursiveValue(Value.from_float(Float(150.0)))
)
_NDArray_5 = zeros(
    TupleInt(
        Vec(
            NDArray(
                RecursiveValue.vec(
                    Vec(
                        RecursiveValue(Value.from_int(Int(0))),
                        RecursiveValue(Value.from_int(Int(1))),
                        RecursiveValue(Value.from_int(Int(2))),
                    )
                )
            ).shape[Int(0)],
            _NDArray_2.shape[Int(1)],
        )
    ),
    OptionalDType.some(_NDArray_2.dtype),
    OptionalDevice.some(_NDArray_2.device),
)
_MultiAxisIndexKeyItem_1 = MultiAxisIndexKeyItem.slice(Slice())
_IndexKey_1 = IndexKey.multi_axis(
    MultiAxisIndexKey.from_vec(
        Vec(MultiAxisIndexKeyItem.int(Int(0)), _MultiAxisIndexKeyItem_1)
    )
)
_NDArray_6 = unique_inverse_inverse_indices(_NDArray_3)
_NDArray_7 = NDArray(RecursiveValue(Value.from_int(Int(0))))
_NDArray_5[_IndexKey_1] = mean(
    _NDArray_2[IndexKey.ndarray(_NDArray_6 == _NDArray_7)],
    OptionalIntOrTuple.int(Int(0)),
)
_IndexKey_2 = IndexKey.multi_axis(
    MultiAxisIndexKey.from_vec(
        Vec(MultiAxisIndexKeyItem.int(Int(1)), _MultiAxisIndexKeyItem_1)
    )
)
_NDArray_5[_IndexKey_2] = mean(
    _NDArray_2[
        IndexKey.ndarray(_NDArray_6 == NDArray(RecursiveValue(Value.from_int(Int(1)))))
    ],
    OptionalIntOrTuple.int(Int(0)),
)
_IndexKey_3 = IndexKey.multi_axis(
    MultiAxisIndexKey.from_vec(
        Vec(MultiAxisIndexKeyItem.int(Int(2)), _MultiAxisIndexKeyItem_1)
    )
)
_NDArray_5[_IndexKey_3] = mean(
    _NDArray_2[
        IndexKey.ndarray(_NDArray_6 == NDArray(RecursiveValue(Value.from_int(Int(2)))))
    ],
    OptionalIntOrTuple.int(Int(0)),
)
_NDArray_8 = asarray(reshape(asarray(_NDArray_3), TupleInt(Vec(Int(-1)))))
_Int_1 = unique_values(
    concat(TupleNDArray(Vec(unique_values(asarray(_NDArray_8)))))
).shape[Int(0)]
_NDArray_9 = concat(
    TupleNDArray(
        Vec(
            _NDArray_2[IndexKey.ndarray(_NDArray_8 == _NDArray_7)]
            - _NDArray_5[_IndexKey_1],
            _NDArray_2[
                IndexKey.ndarray(
                    _NDArray_8 == NDArray(RecursiveValue(Value.from_int(Int(1))))
                )
            ]
            - _NDArray_5[_IndexKey_2],
            _NDArray_2[
                IndexKey.ndarray(
                    _NDArray_8 == NDArray(RecursiveValue(Value.from_int(Int(2))))
                )
            ]
            - _NDArray_5[_IndexKey_3],
        )
    ),
    OptionalInt.some(Int(0)),
)
_NDArray_10 = std(_NDArray_9, OptionalIntOrTuple.int(Int(0)))
_NDArray_10[
    IndexKey.ndarray(std(_NDArray_9, OptionalIntOrTuple.int(Int(0))) == _NDArray_7)
] = NDArray(RecursiveValue(Value.from_float(Float(1.0))))
_TupleNDArray_1 = svd_(
    sqrt(
        asarray(
            NDArray(
                RecursiveValue(
                    Value.from_float(Float(1.0) / Float.from_int(Int(150) - _Int_1))
                )
            ),
            OptionalDType.some(_NDArray_2.dtype),
            OptionalBool.none,
            OptionalDevice.some(_NDArray_2.device),
        )
    )
    * (_NDArray_9 / _NDArray_10),
    Boolean(False),
)
_NDArray_11 = NDArray(RecursiveValue(Value.from_float(Float(0.0001))))
_Slice_1 = Slice(
    OptionalInt.none,
    OptionalInt.some(
        sum(astype(_TupleNDArray_1[Int(1)] > _NDArray_11, DType.int32))
        .index(TupleInt())
        .to_int
    ),
)
_NDArray_12 = (
    _TupleNDArray_1[Int(2)][
        IndexKey.multi_axis(
            MultiAxisIndexKey.from_vec(
                Vec(MultiAxisIndexKeyItem.slice(_Slice_1), _MultiAxisIndexKeyItem_1)
            )
        )
    ]
    / _NDArray_10
).T / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
_TupleNDArray_2 = svd_(
    (
        sqrt(
            NDArray(RecursiveValue(Value.from_int(Int(150))))
            * _NDArray_4
            * NDArray(
                RecursiveValue(
                    Value.from_float(Float(1.0) / Float.from_int(_Int_1 - Int(1)))
                )
            )
        )
        * (_NDArray_5 - _NDArray_4 @ _NDArray_5).T
    ).T
    @ _NDArray_12,
    Boolean(False),
)
(
    (_NDArray_2 - _NDArray_4 @ _NDArray_5)
    @ (
        _NDArray_12
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
                                            _TupleNDArray_2[Int(1)]
                                            > _NDArray_11
                                            * _TupleNDArray_2[Int(1)][
                                                IndexKey.int(Int(0))
                                            ],
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
        MultiAxisIndexKey.from_vec(
            Vec(
                _MultiAxisIndexKeyItem_1,
                MultiAxisIndexKeyItem.slice(
                    Slice(OptionalInt.none, OptionalInt.some(Int(2)))
                ),
            )
        )
    )
]