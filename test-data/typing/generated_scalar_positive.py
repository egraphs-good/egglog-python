"""Public consumer checks: typecheck this file, do not execute symbolic .value."""

from typing import assert_type

from egglog import (
    Bool,
    BoolLike,
    Expr,
    ExprValueError,
    Map,
    Pair,
    String,
    StringLike,
    Unit,
    Vec,
    constant,
    convert,
    count_matches,
    f64,
    f64Like,
    function,
    i64,
    i64Like,
    join,
    ne,
)

assert_type(String("x"), String)
assert_type(Bool(False), Bool)
assert_type(Unit(), Unit)
assert_type(join(), String)
assert_type(join("a", String("b"), "c"), String)
assert_type(String("a") + "b", String)
assert_type("a" + String("b"), String)
assert_type(String("abc").replace("b", String("x")), String)
assert_type(i64(-42).to_string(), String)
assert_type(~Bool(False), Bool)
assert_type(Bool(True) & False, Bool)
assert_type(True & Bool(False), Bool)
assert_type(Bool(True) | False, Bool)
assert_type(True | Bool(False), Bool)
assert_type(Bool(True) ^ False, Bool)
assert_type(True ^ Bool(False), Bool)
assert_type(Bool(True).implies(False), Bool)
assert_type(String("x").value, str)
assert_type(Bool(False).value, bool)
assert_type(bool(Bool(False)), bool)
assert_type(bool(Unit()), bool)
assert_type(ne(Bool(True)).to(Bool(False)), Unit)
assert_type(ne(i64(1)).to(i64(2)), Unit)
assert_type(ne(Vec[i64](i64(1))).to(Vec[i64](i64(2))), Unit)
assert_type(bool(ne(i64(1)).to(i64(2))), bool)
assert_type(convert("x", String), String)
assert_type(convert(False, Bool), Bool)
assert_type(Pair(String("key"), Bool(False)), Pair[String, Bool])
assert_type(Map[String, Bool].empty().insert(String("key"), Bool(False)), Map[String, Bool])


@function
def scalar_message(prefix: StringLike = String("a") + "b", enabled: BoolLike = False) -> String: ...


@function
def scalar_flag(value: BoolLike = Bool(False)) -> Bool: ...


assert_type(scalar_message(), String)
assert_type(scalar_message("x", Bool(True)), String)
assert_type(scalar_flag(), Bool)
assert_type(scalar_flag(False), Bool)

assert_type(i64(1), i64)
assert_type(f64(1.5), f64)
assert_type(f64(1), f64)
assert_type(i64(1) + 2, i64)
assert_type(i64(1) + i64(2), i64)
assert_type(2 + i64(1), i64)
assert_type(f64(1.5) + 2, f64)
assert_type(f64(1.5) + f64(2.5), f64)
assert_type(2 + f64(1.5), f64)
assert_type(i64(13) | 2, i64)
assert_type(2 | i64(13), i64)
assert_type(i64(13) ^ 2, i64)
assert_type(2 ^ i64(13), i64)
assert_type(i64(13) << 2, i64)
assert_type(2 << i64(13), i64)
assert_type(i64(13) >> 2, i64)
assert_type(2 >> i64(13), i64)
assert_type(~i64(13), i64)
assert_type(abs(i64(-13)), i64)
assert_type(i64(13).log2(), i64)
assert_type(i64(13) > 2, Unit)
assert_type(i64(13) <= 2, Unit)
assert_type(i64(13).bool_lt(2), Bool)
assert_type(i64(13).bool_gt(2), Bool)
assert_type(i64(13).bool_le(2), Bool)
assert_type(i64(13).bool_ge(2), Bool)
assert_type(-f64(4.0), f64)
assert_type(abs(f64(-4.0)), f64)
assert_type(f64(4.0) - 2, f64)
assert_type(2 - f64(4.0), f64)
assert_type(f64(4.0) * 2, f64)
assert_type(2 * f64(4.0), f64)
assert_type(f64(4.0) / 2, f64)
assert_type(2 / f64(4.0), f64)
assert_type(f64(4.0) % 2, f64)
assert_type(2 % f64(4.0), f64)
assert_type(f64(4.0) ** 2, f64)
assert_type(2 ** f64(4.0), f64)
assert_type(f64(4.0).exp(), f64)
assert_type(f64(4.0).log(), f64)
assert_type(f64(4.0).sqrt(), f64)
assert_type(f64(4.0).is_finite(), Unit)
assert_type(f64(4.0) < 2, Unit)
assert_type(f64(4.0) > 2, Unit)
assert_type(f64(4.0) <= 2, Unit)
assert_type(f64(4.0) >= 2, Unit)
assert_type(f64(4.0).min(2), f64)
assert_type(f64(4.0).max(2), f64)
assert_type(f64(4.0).to_i64(), i64)
assert_type(f64(4.0).to_string(), String)
assert_type(f64.from_i64(i=i64(4)), f64)
assert_type(f64.from_i64(4), f64)
assert_type(count_matches(s="a\0a", pattern="a"), i64)
assert_type(i64(1).value, int)
assert_type((i64(1) + 2).value, int)
assert_type(f64(1.5).value, float)
assert_type(int(i64(1)), int)
assert_type(float(f64(1.5)), float)
assert_type(i64(1).__index__(), int)
assert_type(convert(2, i64), i64)
assert_type(convert(2, f64), f64)
assert_type(constant("counter", i64), i64)
assert_type(ExprValueError(i64(1), "literal").allowed, str)


@function
def score(x: i64Like, offset: i64Like = 2) -> i64: ...


assert_type(score(1), i64)
assert_type(score(i64(1) + 2, offset=i64(3)), i64)


@function(egg_fn="exact measure")
def measure(x: f64Like) -> f64: ...


assert_type(measure(f64(1.5) + 2), f64)


class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...


assert_type(Num(1), Num)
assert_type(constant("number", Num), Num)


@function(cost=1)
def node(value: i64Like) -> Num: ...


assert_type(node(1), Num)


def nominal_vec(value: Vec[i64]) -> Vec[i64]:
    return value
