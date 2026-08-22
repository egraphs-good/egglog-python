_Value_1 = Value.var("bp3") * Value.var("q9") + Value.var("bp2") * Value.var("q6") + Value.var("bp1") * Value.var("q3") + Value.var("bp4") * Value.var("q12")
_Value_2 = Value.var("bpp2") * Value.var("q5") + Value.var("bpp1") * Value.var("q2") + Value.var("bpp3") * Value.var("q8") + Value.var("bpp4") * Value.var("q11")
_Value_3 = Value.var("bp3") * Value.var("q8") + Value.var("bp2") * Value.var("q5") + Value.var("bp1") * Value.var("q2") + Value.var("bp4") * Value.var("q11")
_Value_4 = Value.var("bpp2") * Value.var("q6") + Value.var("bpp1") * Value.var("q3") + Value.var("bpp3") * Value.var("q9") + Value.var("bpp4") * Value.var("q12")
_Value_5 = Value.var("q1") * Value.var("bp1") + Value.var("q4") * Value.var("bp2") + Value.var("q7") * Value.var("bp3") + Value.var("q10") * Value.var("bp4")
_Value_6 = Value.var("q4") * Value.var("bpp2") + Value.var("q1") * Value.var("bpp1") + Value.var("q7") * Value.var("bpp3") + Value.var("q10") * Value.var("bpp4")
NDArray(
    RecursiveValue(
        (
            (Value.from_int(Int(-1)) * (_Value_1 * _Value_2) + _Value_3 * _Value_4) ** Value.from_int(Int(2))
            + (Value.from_int(Int(-1)) * (_Value_5 * _Value_4) + _Value_1 * _Value_6) ** Value.from_int(Int(2))
            + (Value.from_int(Int(-1)) * (_Value_3 * _Value_6) + _Value_5 * _Value_2) ** Value.from_int(Int(2))
        )
        / (_Value_5 ** Value.from_int(Int(2)) + _Value_3 ** Value.from_int(Int(2)) + _Value_1 ** Value.from_int(Int(2))) ** Value.from_int(Int(3))
    )
)