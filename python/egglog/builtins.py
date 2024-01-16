# mypy: disable-error-code="empty-body"
"""
Builtin sorts and function to egg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, Union

from .egraph import BUILTINS, Expr, Unit
from .runtime import converter

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BUILTINS",
    "i64",
    "i64Like",
    "f64",
    "f64Like",
    "Bool",
    "BoolLike",
    "String",
    "StringLike",
    "Map",
    "Rational",
    "Set",
    "Vec",
    "join",
    "PyObject",
    "py_eval",
    "py_exec",
    "py_eval_fn",
]


StringLike = Union["String", str]


@BUILTINS.class_
class String(Expr):
    def __init__(self, value: str) -> None:
        ...

    @BUILTINS.method(egg_fn="replace")
    def replace(self, old: StringLike, new: StringLike) -> String:
        ...


@BUILTINS.function(egg_fn="+")
def join(*strings: StringLike) -> String:
    ...


converter(str, String, String)

BoolLike = Union["Bool", bool]


@BUILTINS.class_(egg_sort="bool")
class Bool(Expr):
    def __init__(self, value: bool) -> None:
        ...

    @BUILTINS.method(egg_fn="not")
    def __invert__(self) -> Bool:
        ...

    @BUILTINS.method(egg_fn="and")
    def __and__(self, other: BoolLike) -> Bool:
        ...

    @BUILTINS.method(egg_fn="or")
    def __or__(self, other: BoolLike) -> Bool:
        ...

    @BUILTINS.method(egg_fn="xor")
    def __xor__(self, other: BoolLike) -> Bool:
        ...

    @BUILTINS.method(egg_fn="=>")
    def implies(self, other: BoolLike) -> Bool:
        ...


converter(bool, Bool, Bool)

# The types which can be convertered into an i64
i64Like = Union["i64", int]  # noqa: N816


@BUILTINS.class_(egg_sort="i64")
class i64(Expr):  # noqa: N801
    def __init__(self, value: int) -> None:
        ...

    @BUILTINS.method(egg_fn="+")
    def __add__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="-")
    def __sub__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="*")
    def __mul__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="/")
    def __truediv__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="%")
    def __mod__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="&")
    def __and__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="|")
    def __or__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="^")
    def __xor__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="<<")
    def __lshift__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn=">>")
    def __rshift__(self, other: i64Like) -> i64:
        ...

    def __radd__(self, other: i64Like) -> i64:
        ...

    def __rsub__(self, other: i64Like) -> i64:
        ...

    def __rmul__(self, other: i64Like) -> i64:
        ...

    def __rtruediv__(self, other: i64Like) -> i64:
        ...

    def __rmod__(self, other: i64Like) -> i64:
        ...

    def __rand__(self, other: i64Like) -> i64:
        ...

    def __ror__(self, other: i64Like) -> i64:
        ...

    def __rxor__(self, other: i64Like) -> i64:
        ...

    def __rlshift__(self, other: i64Like) -> i64:
        ...

    def __rrshift__(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="not-i64")
    def __invert__(self) -> i64:
        ...

    @BUILTINS.method(egg_fn="<")
    def __lt__(self, other: i64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @BUILTINS.method(egg_fn=">")
    def __gt__(self, other: i64Like) -> Unit:
        ...

    @BUILTINS.method(egg_fn="<=")
    def __le__(self, other: i64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @BUILTINS.method(egg_fn=">=")
    def __ge__(self, other: i64Like) -> Unit:
        ...

    @BUILTINS.method(egg_fn="min")
    def min(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="max")
    def max(self, other: i64Like) -> i64:
        ...

    @BUILTINS.method(egg_fn="to-string")
    def to_string(self) -> String:
        ...

    @BUILTINS.method(egg_fn="bool-<")
    def bool_lt(self, other: i64Like) -> Bool:
        ...

    @BUILTINS.method(egg_fn="bool->")
    def bool_gt(self, other: i64Like) -> Bool:
        ...

    @BUILTINS.method(egg_fn="bool-<=")
    def bool_le(self, other: i64Like) -> Bool:
        ...

    @BUILTINS.method(egg_fn="bool->=")
    def bool_ge(self, other: i64Like) -> Bool:
        ...


converter(int, i64, i64)


@BUILTINS.function(egg_fn="count-matches")
def count_matches(s: StringLike, pattern: StringLike) -> i64:
    ...


f64Like = Union["f64", float]  # noqa: N816


@BUILTINS.class_(egg_sort="f64")
class f64(Expr):  # noqa: N801
    def __init__(self, value: float) -> None:
        ...

    @BUILTINS.method(egg_fn="neg")
    def __neg__(self) -> f64:
        ...

    @BUILTINS.method(egg_fn="+")
    def __add__(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="-")
    def __sub__(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="*")
    def __mul__(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="/")
    def __truediv__(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="%")
    def __mod__(self, other: f64Like) -> f64:
        ...

    def __radd__(self, other: f64Like) -> f64:
        ...

    def __rsub__(self, other: f64Like) -> f64:
        ...

    def __rmul__(self, other: f64Like) -> f64:
        ...

    def __rtruediv__(self, other: f64Like) -> f64:
        ...

    def __rmod__(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="<")
    def __lt__(self, other: f64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @BUILTINS.method(egg_fn=">")
    def __gt__(self, other: f64Like) -> Unit:
        ...

    @BUILTINS.method(egg_fn="<=")
    def __le__(self, other: f64Like) -> Unit:  # type: ignore[empty-body,has-type]
        ...

    @BUILTINS.method(egg_fn=">=")
    def __ge__(self, other: f64Like) -> Unit:
        ...

    @BUILTINS.method(egg_fn="min")
    def min(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="max")
    def max(self, other: f64Like) -> f64:
        ...

    @BUILTINS.method(egg_fn="to-i64")
    def to_i64(self) -> i64:
        ...

    @BUILTINS.method(egg_fn="to-f64")
    @classmethod
    def from_i64(cls, i: i64) -> f64:
        ...

    @BUILTINS.method(egg_fn="to-string")
    def to_string(self) -> String:
        ...


converter(float, f64, f64)


T = TypeVar("T", bound=Expr)
V = TypeVar("V", bound=Expr)


@BUILTINS.class_(egg_sort="Map")
class Map(Expr, Generic[T, V]):
    @BUILTINS.method(egg_fn="map-empty")
    @classmethod
    def empty(cls) -> Map[T, V]:
        ...

    @BUILTINS.method(egg_fn="map-insert")
    def insert(self, key: T, value: V) -> Map[T, V]:
        ...

    @BUILTINS.method(egg_fn="map-get")
    def __getitem__(self, key: T) -> V:
        ...

    @BUILTINS.method(egg_fn="map-not-contains")
    def not_contains(self, key: T) -> Unit:
        ...

    @BUILTINS.method(egg_fn="map-contains")
    def contains(self, key: T) -> Unit:
        ...

    @BUILTINS.method(egg_fn="map-remove")
    def remove(self, key: T) -> Map[T, V]:
        ...

    @BUILTINS.method(egg_fn="rebuild")
    def rebuild(self) -> Map[T, V]:
        ...


@BUILTINS.class_(egg_sort="Set")
class Set(Expr, Generic[T]):
    @BUILTINS.method(egg_fn="set-of")
    def __init__(self, *args: T) -> None:
        ...

    @BUILTINS.method(egg_fn="set-empty")
    @classmethod
    def empty(cls) -> Set[T]:
        ...

    @BUILTINS.method(egg_fn="set-insert")
    def insert(self, value: T) -> Set[T]:
        ...

    @BUILTINS.method(egg_fn="set-not-contains")
    def not_contains(self, value: T) -> Unit:
        ...

    @BUILTINS.method(egg_fn="set-contains")
    def contains(self, value: T) -> Unit:
        ...

    @BUILTINS.method(egg_fn="set-remove")
    def remove(self, value: T) -> Set[T]:
        ...

    @BUILTINS.method(egg_fn="set-union")
    def __or__(self, other: Set[T]) -> Set[T]:
        ...

    @BUILTINS.method(egg_fn="set-diff")
    def __sub__(self, other: Set[T]) -> Set[T]:
        ...

    @BUILTINS.method(egg_fn="set-intersect")
    def __and__(self, other: Set[T]) -> Set[T]:
        ...

    @BUILTINS.method(egg_fn="rebuild")
    def rebuild(self) -> Set[T]:
        ...


@BUILTINS.class_(egg_sort="Rational")
class Rational(Expr):
    @BUILTINS.method(egg_fn="rational")
    def __init__(self, num: i64Like, den: i64Like) -> None:
        ...

    @BUILTINS.method(egg_fn="to-f64")
    def to_f64(self) -> f64:
        ...

    @BUILTINS.method(egg_fn="+")
    def __add__(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="-")
    def __sub__(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="*")
    def __mul__(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="/")
    def __truediv__(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="min")
    def min(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="max")
    def max(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="neg")
    def __neg__(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="abs")
    def __abs__(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="floor")
    def floor(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="ceil")
    def ceil(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="round")
    def round(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="pow")
    def __pow__(self, other: Rational) -> Rational:
        ...

    @BUILTINS.method(egg_fn="log")
    def log(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="sqrt")
    def sqrt(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="cbrt")
    def cbrt(self) -> Rational:
        ...

    @BUILTINS.method(egg_fn="numer")  # type: ignore[misc]
    @property
    def numer(self) -> i64:
        ...

    @BUILTINS.method(egg_fn="denom")  # type: ignore[misc]
    @property
    def denom(self) -> i64:
        ...


@BUILTINS.class_(egg_sort="Vec")
class Vec(Expr, Generic[T]):
    @BUILTINS.method(egg_fn="vec-of")
    def __init__(self, *args: T) -> None:
        ...

    @BUILTINS.method(egg_fn="vec-empty")
    @classmethod
    def empty(cls) -> Vec[T]:
        ...

    @BUILTINS.method(egg_fn="vec-append")
    def append(self, *others: Vec[T]) -> Vec[T]:
        ...

    @BUILTINS.method(egg_fn="vec-push")
    def push(self, value: T) -> Vec[T]:
        ...

    @BUILTINS.method(egg_fn="vec-pop")
    def pop(self) -> Vec[T]:
        ...

    @BUILTINS.method(egg_fn="vec-not-contains")
    def not_contains(self, value: T) -> Unit:
        ...

    @BUILTINS.method(egg_fn="vec-contains")
    def contains(self, value: T) -> Unit:
        ...

    @BUILTINS.method(egg_fn="vec-length")
    def length(self) -> i64:
        ...

    @BUILTINS.method(egg_fn="vec-get")
    def __getitem__(self, index: i64Like) -> T:
        ...

    @BUILTINS.method(egg_fn="rebuild")
    def rebuild(self) -> Vec[T]:
        ...


@BUILTINS.class_(egg_sort="PyObject")
class PyObject(Expr):
    def __init__(self, value: object) -> None:
        ...

    @BUILTINS.method(egg_fn="py-from-string")
    @classmethod
    def from_string(cls, s: StringLike) -> PyObject:
        ...

    @BUILTINS.method(egg_fn="py-to-string")
    def to_string(self) -> String:
        ...

    @BUILTINS.method(egg_fn="py-to-bool")
    def to_bool(self) -> Bool:
        ...

    @BUILTINS.method(egg_fn="py-dict-update")
    def dict_update(self, *keys_and_values: object) -> PyObject:
        ...

    @BUILTINS.method(egg_fn="py-from-int")
    @classmethod
    def from_int(cls, i: i64Like) -> PyObject:
        ...

    @BUILTINS.method(egg_fn="py-dict")
    @classmethod
    def dict(cls, *keys_and_values: object) -> PyObject:
        ...


converter(object, PyObject, PyObject)


@BUILTINS.function(egg_fn="py-eval")
def py_eval(code: StringLike, globals: object = PyObject.dict(), locals: object = PyObject.dict()) -> PyObject:
    ...


class PyObjectFunction(Protocol):
    def __call__(self, *__args: PyObject) -> PyObject:
        ...


def py_eval_fn(fn: Callable) -> PyObjectFunction:
    """
    Takes a python callable and maps it to a callable which takes and returns PyObjects.

    It translates it to a call which uses `py_eval` to call the function, passing in the
    args as locals, and using the globals from function.
    """

    def inner(*__args: PyObject, __fn: Callable = fn) -> PyObject:
        new_kvs: list[object] = []
        eval_str = "__fn("
        for i, arg in enumerate(__args):
            new_kvs.append(f"__arg_{i}")
            new_kvs.append(arg)
            eval_str += f"__arg_{i}, "
        eval_str += ")"
        return py_eval(eval_str, PyObject({"__fn": __fn}).dict_update(*new_kvs), __fn.__globals__)

    return inner


@BUILTINS.function(egg_fn="py-exec")
def py_exec(code: StringLike, globals: object = PyObject.dict(), locals: object = PyObject.dict()) -> PyObject:
    """
    Copies the locals, execs the Python code, and returns the locals with any updates.
    """
    ...
