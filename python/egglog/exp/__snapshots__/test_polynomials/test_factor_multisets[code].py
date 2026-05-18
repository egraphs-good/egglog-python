_expr_1 = (
    Value.var("q2") * Value.var("bp1")
    + Value.var("q5") * Value.var("bp2")
    + Value.var("q8") * Value.var("bp3")
    + Value.var("bp4") * Value.var("q11")
)
_expr_2 = (
    Value.var("bpp2") * Value.var("q6")
    + Value.var("q3") * Value.var("bpp1")
    + Value.var("bpp3") * Value.var("q9")
    + Value.var("bpp4") * Value.var("q12")
)
_expr_3 = (
    Value.var("bp1") * Value.var("q3")
    + Value.var("bp2") * Value.var("q6")
    + Value.var("q12") * Value.var("bp4")
    + Value.var("bp3") * Value.var("q9")
)
_expr_4 = (
    Value.var("bpp2") * Value.var("q5")
    + Value.var("q2") * Value.var("bpp1")
    + Value.var("bpp3") * Value.var("q8")
    + Value.var("bpp4") * Value.var("q11")
)
_expr_5 = (
    Value.var("q4") * Value.var("bpp2")
    + Value.var("q7") * Value.var("bpp3")
    + Value.var("q10") * Value.var("bpp4")
    + Value.var("bpp1") * Value.var("q1")
)
_expr_6 = (
    Value.var("q4") * Value.var("bp2")
    + Value.var("q7") * Value.var("bp3")
    + Value.var("bp1") * Value.var("q1")
    + Value.var("q10") * Value.var("bp4")
)
NDArray(
    RecursiveValue(
        (
            (_expr_1 * _expr_2 + Value.from_int(Int(-1)) * (_expr_3 * _expr_4)) ** Value.from_int(Int(2))
            + (_expr_3 * _expr_5 + Value.from_int(Int(-1)) * (_expr_6 * _expr_2)) ** Value.from_int(Int(2))
            + (_expr_6 * _expr_4 + Value.from_int(Int(-1)) * (_expr_1 * _expr_5)) ** Value.from_int(Int(2))
        )
        / (_expr_6 ** Value.from_int(Int(2)) + _expr_1 ** Value.from_int(Int(2)) + _expr_3 ** Value.from_int(Int(2)))
        ** Value.from_int(Int(3))
    )
)
