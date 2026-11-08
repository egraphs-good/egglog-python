"""Wrong sorts and implicit inexact input must fail static checking."""

from fractions import Fraction

from egglog import BigInt, BigIntLike, BigRat, BigRatLike, Rational, RationalLike, String, f64

Rational()  # error: call-overload
Rational(num=1, den=2)  # error: call-overload
BigInt(value=1)  # error: call-arg
Rational(Fraction(1, 2), 1)  # error: call-overload
Rational(1.5, 2)  # error: call-overload
Rational(f64(1.0), 2)  # error: call-overload
Rational(1, 2) + BigRat(1, 2)  # error: operator
Rational(1, 2) + 1.5  # error: operator
BigInt("1")  # error: arg-type
BigInt.from_string(1)  # error: arg-type
BigInt.from_string()  # error: call-arg
BigRat(String("1"), 2)  # error: call-overload
BigRat(Rational(1, 2), 1)  # error: call-overload
BigRat(Fraction(1, 2), 1)  # error: call-overload
wrong_r: int = Rational(1, 2).value  # error: assignment
wrong_z: Fraction = BigInt(1).value  # error: assignment
wrong_q: int = BigRat(1, 2).value  # error: assignment
r: RationalLike = 0.5  # error: assignment
z: BigIntLike = Fraction(1, 2)  # error: assignment
q: BigRatLike = BigInt(1)  # error: assignment
Rational.from_value(Fraction(1, 2))  # error: attr-defined
Rational(1, 2) - 1.5  # error: operator
Rational(1, 2) * f64(2.0)  # error: operator
Rational(1, 2) / BigRat(2, 1)  # error: operator
Rational(1, 2) ** BigInt(2)  # error: operator
Rational(1, 2).min(f64(1.0))  # error: arg-type
Rational(1, 2).max(1.0)  # error: arg-type
Rational(1, 2).floor(1)  # error: call-arg
Rational(1, 2).sqrt(1)  # error: call-arg
Rational(1, 2).to_f64(1)  # error: call-arg
Rational(1, 2).numer()  # error: operator
Rational(1, 2).denom(1)  # error: operator
wrong_rational_numerator: Rational = Rational(1, 2).numer  # error: assignment
wrong_rational_comparison: bool = Rational(1, 2) <= 1  # error: assignment
Rational(1, 2).__rlt__(1)  # error: attr-defined

integer = BigInt(1)
ratio = BigRat(1, 2)
integer << BigInt(2)  # error: operator
integer >> ratio  # error: operator
integer << f64(2.0)  # error: operator
integer + ratio  # error: operator
_ = integer < ratio  # error: operator
integer.min(ratio)  # error: arg-type
integer.bool_eq(ratio)  # error: arg-type
integer.bits(1)  # error: call-arg
integer.to_string(1)  # error: call-arg
ratio + integer  # error: operator
_ = ratio < integer  # error: operator
ratio.min(integer)  # error: arg-type
ratio**integer  # error: operator
ratio.numer()  # error: operator
ratio.denom(1)  # error: operator
ratio.to_i64(1)  # error: call-arg
ratio.to_f64(1)  # error: call-arg
ratio.sqrt(1)  # error: call-arg
wrong_numerator: BigRat = ratio.numer  # error: assignment
wrong_denominator: f64 = ratio.denom  # error: assignment
wrong_predicate: bool = ratio < 2  # error: assignment
wrong_bits: int = integer.bits()  # error: assignment
integer.__rrshift__(1)  # error: attr-defined
ratio.__rlt__(1)  # error: attr-defined
