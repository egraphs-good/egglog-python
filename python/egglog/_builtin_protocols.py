"""
Python host codecs/protocols, not native callable signatures or sort records.

Kept import-independent so the offline stub generator can load this file without
executing egglog's package initializer, runtime, or native extension.
"""

from fractions import Fraction
from typing import NamedTuple


class ScalarProtocol(NamedTuple):
    literal_arm: str
    value_type: type[object]
    literal_types: tuple[type[object], ...]
    hooks: tuple[tuple[str, type[object]], ...] = ()
    literal_constructor: bool = True
    conversion_sort: str | None = None


SCALAR_PROTOCOLS = {
    "i64": ScalarProtocol("i64", int, (int,), (("__index__", int), ("__int__", int))),
    "f64": ScalarProtocol("f64_bits", float, (float, int), (("__float__", float), ("__int__", int))),
    "String": ScalarProtocol("string", str, (str,)),
    "bool": ScalarProtocol("bool", bool, (bool,), (("__bool__", bool),)),
    "Unit": ScalarProtocol("unit", type(None), (), (("__bool__", bool),)),
    "PyObject": ScalarProtocol("custom", object, (object,)),
    "Rational": ScalarProtocol(
        "rational", Fraction, (Fraction,), (("__int__", int), ("__float__", float)), False, "i64"
    ),
    "BigInt": ScalarProtocol("big_int", int, (), (("__int__", int), ("__index__", int)), False, "i64"),
    "BigRat": ScalarProtocol("big_rat", Fraction, (Fraction,), (("__int__", int), ("__float__", float)), False, "i64"),
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
