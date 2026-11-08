"""
Python host codecs/protocols, not native callable signatures or sort records.

The Python declarations own their methods and host conversion hooks.
"""

from fractions import Fraction
from typing import NamedTuple


class ScalarProtocol(NamedTuple):
    literal_arm: str
    value_type: type[object]
    literal_types: tuple[type[object], ...]
    conversion_sort: str | None = None


SCALAR_PROTOCOLS = {
    "i64": ScalarProtocol("i64", int, (int,)),
    "f64": ScalarProtocol("f64_bits", float, (float, int)),
    "String": ScalarProtocol("string", str, (str,)),
    "bool": ScalarProtocol("bool", bool, (bool,)),
    "Unit": ScalarProtocol("unit", type(None), ()),
    "PyObject": ScalarProtocol("custom", object, (object,)),
    "Rational": ScalarProtocol("rational", Fraction, (int, Fraction), "i64"),
    "BigInt": ScalarProtocol("big_int", int, (int,), "i64"),
    "BigRat": ScalarProtocol("big_rat", Fraction, (int, Fraction), "i64"),
}

# Runtime forwarding policy, not evidence that a particular native method exists.
NUMERIC_BINARY_METHODS = {
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
}
