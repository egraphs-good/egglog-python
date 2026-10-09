"""Wrong sorts and implicit inexact input must fail static checking."""

from fractions import Fraction

from egglog import BigInt, BigIntLike, BigRat, BigRatLike, Rational, RationalLike, String, f64

Rational(1)  # error: call-arg
Rational(Fraction(1, 2), 1)  # error: arg-type
Rational(1.5, 2)  # error: arg-type
Rational(f64(1.0), 2)  # error: arg-type
Rational(1, 2) + BigRat(1, 2)  # error: operator
Rational(1, 2) + 1.5  # error: operator
BigInt("1")  # error: arg-type
BigInt.from_string(1)  # error: arg-type
BigInt.from_string()  # error: call-arg
BigRat(String("1"), 2)  # error: arg-type
BigRat(Rational(1, 2), 1)  # error: arg-type
BigRat(Fraction(1, 2), 1)  # error: arg-type
wrong_r: int = Rational(1, 2).value  # error: assignment
wrong_z: Fraction = BigInt(1).value  # error: assignment
wrong_q: int = BigRat(1, 2).value  # error: assignment
r: RationalLike = 0.5  # error: assignment
z: BigIntLike = Fraction(1, 2)  # error: assignment
q: BigRatLike = BigInt(1)  # error: assignment
Rational.from_value(Fraction(1, 2))  # error: attr-defined
