_Value_1 = NDArray.var("X").index(TupleInt.from_vec(Vec[Int](Int(0), Int(0), Int.var("i"), Int.var("j"))))
_Value_2 = NDArray.var("X").index(TupleInt.from_vec(Vec[Int](Int(0), Int(1), Int.var("i"), Int.var("j"))))
_Value_3 = NDArray.var("X").index(TupleInt.from_vec(Vec[Int](Int(1), Int(0), Int.var("i"), Int.var("j"))))
_Value_4 = NDArray.var("X").index(TupleInt.from_vec(Vec[Int](Int(1), Int(1), Int.var("i"), Int.var("j"))))
_Value_5 = NDArray.var("X").index(TupleInt.from_vec(Vec[Int](Int(2), Int(0), Int.var("i"), Int.var("j"))))
_Value_6 = NDArray.var("X").index(TupleInt.from_vec(Vec[Int](Int(2), Int(1), Int.var("i"), Int.var("j"))))
(
    (
        ((((_Value_1.conj() * _Value_1).real() + (_Value_2.conj() * _Value_2).real()) + (_Value_3.conj() * _Value_3).real()) + (_Value_4.conj() * _Value_4).real())
        + (_Value_5.conj() * _Value_5).real()
    )
    + (_Value_6.conj() * _Value_6).real()
).sqrt()