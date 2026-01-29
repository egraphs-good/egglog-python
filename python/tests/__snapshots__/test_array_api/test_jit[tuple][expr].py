NDArray.var("x")[
    IndexKey.int(
        (NDArray.var("x").shape + TupleInt.from_vec(Vec(Int(1), Int(2))))[Int(100)]
    )
]