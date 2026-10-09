"""Source-generated exact families; expression constructors retain native types."""

from fractions import Fraction
from operator import index
from typing import assert_type

from egglog import BigInt, BigIntLike, BigRat, BigRatLike, Rational, RationalLike, String, convert, i64

assert_type(Rational(1, 2), Rational)
assert_type(Rational(i64(1) + 2, 3), Rational)
assert_type(Rational(1, 2) + Fraction(1, 3), Rational)
assert_type(Fraction(1, 3) + Rational(1, 2), Rational)
assert_type(Rational(1, 2) + i64(1), Rational)
assert_type(BigInt(i64(1) + 2), BigInt)
assert_type(BigInt.from_string(String("123")), BigInt)
assert_type(BigRat(BigInt.from_string("123"), 2), BigRat)
assert_type(Rational(1, 2).value, Fraction)
assert_type(BigInt(1).value, int)
assert_type(BigRat(1, 2).value, Fraction)
assert_type(int(BigInt(1)), int)
assert_type(index(BigInt(1)), int)
assert_type(float(Rational(1, 2)), float)
assert_type(int(Rational(1, 2)), int)
assert_type(float(BigRat(1, 2)), float)
assert_type(int(BigRat(1, 2)), int)
assert_type(convert(Fraction(1, 2), Rational), Rational)
assert_type(convert(i64(1), BigInt), BigInt)
assert_type(convert(Fraction(1, 2), BigRat), BigRat)

r: RationalLike = Fraction(1, 2)
r = i64(1)
r = 1
z: BigIntLike = i64(1)
z = 1
q: BigRatLike = Fraction(1, 2)
q = i64(1)
q = 1
