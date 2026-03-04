_Int_1 = check_index(NDArray.var("x").shape.length() + Int(2), Int(100))
NDArray.var("x")[
    IndexKey.int(
        Int.if_(
            _Int_1 < NDArray.var("x").shape.length(),
            lambda: NDArray.var("x").shape[_Int_1],
            lambda: TupleInt(Vec(Int(1), Int(2)))[
                _Int_1 - NDArray.var("x").shape.length()
            ],
        )
    )
]