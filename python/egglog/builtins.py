"""
Builtin sorts and function to egg.
"""

from __future__ import annotations

from typing import Generic, TypeVar, Union

from .egraph import BUILTINS, BaseExpr, Unit

__all__ = [
    "BUILTINS",
    "i64",
    "i64Like",
    "f64",
    "f64Like",
    "String",
    "StringLike",
    "Map",
    "Rational",
    "Set",
    "Vec",
    "join",
]

# The types which can be converted into an i64
i64Like = Union[int, "i64"]


@BUILTINS.class_(egg_sort="i64")
class i64(BaseExpr):

  def __init__(self, value: int):
    ...

  @BUILTINS.method(egg_fn="+")
  def __add__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="-")
  def __sub__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="*")
  def __mul__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="/")
  def __truediv__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="%")
  def __mod__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="&")
  def __and__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="|")
  def __or__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="^")
  def __xor__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="<<")
  def __lshift__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn=">>")
  def __rshift__(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="not-64")
  def __invert__(self) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="<")
  def __lt__(self, other: i64Like) -> Unit:  # type: ignore[empty-body,has-type]
    ...

  @BUILTINS.method(egg_fn=">")
  def __gt__(self, other: i64Like) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="min")
  def min(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="max")
  def max(self, other: i64Like) -> i64:  # type: ignore[empty-body]
    ...


f64Like = Union[float, "f64"]


@BUILTINS.class_(egg_sort="f64")
class f64(BaseExpr):

  def __init__(self, value: float):
    ...

  @BUILTINS.method(egg_fn="neg")
  def __neg__(self) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="+")
  def __add__(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="-")
  def __sub__(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="*")
  def __mul__(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="/")
  def __truediv__(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="%")
  def __mod__(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="<")
  def __lt__(self, other: f64Like) -> Unit:  # type: ignore[empty-body,has-type]
    ...

  @BUILTINS.method(egg_fn=">")
  def __gt__(self, other: f64Like) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="<=")
  def __le__(self, other: f64Like) -> Unit:  # type: ignore[empty-body,has-type]
    ...

  @BUILTINS.method(egg_fn=">=")
  def __ge__(self, other: f64Like) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="min")
  def min(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="max")
  def max(self, other: f64Like) -> f64:  # type: ignore[empty-body]
    ...


StringLike = Union[str, "String"]


@BUILTINS.class_
class String(BaseExpr):

  def __init__(self, value: str):
    ...


@BUILTINS.function(egg_fn="+")
def join(*strings: StringLike) -> String:  # type: ignore[empty-body]
  ...


T = TypeVar("T", bound=BaseExpr)
V = TypeVar("V", bound=BaseExpr)


@BUILTINS.class_(egg_sort="Map")
class Map(BaseExpr, Generic[T, V]):

  @BUILTINS.method(egg_fn="map-empty")
  @classmethod
  def empty(cls) -> Map[T, V]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="map-insert")
  def insert(self, key: T, value: V) -> Map[T, V]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="map-get")
  def __getitem__(self, key: T) -> V:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="map-not-contains")
  def not_contains(self, key: T) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="map-contains")
  def contains(self, key: T) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="map-remove")
  def remove(self, key: T) -> Map[T, V]:  # type: ignore[empty-body]
    ...


@BUILTINS.class_(egg_sort="Set")
class Set(BaseExpr, Generic[T]):

  @BUILTINS.method(egg_fn="set-of")
  def __init__(self, *args: T) -> None:
    ...

  @BUILTINS.method(egg_fn="set-empty")
  @classmethod
  def empty(cls) -> Set[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-insert")
  def insert(self, value: T) -> Set[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-not-contains")
  def not_contains(self, value: T) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-contains")
  def contains(self, value: T) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-remove")
  def remove(self, value: T) -> Set[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-union")
  def __or__(self, other: Set[T]) -> Set[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-diff")
  def __sub__(self, other: Set[T]) -> Set[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="set-intersect")
  def __and__(self, other: Set[T]) -> Set[T]:  # type: ignore[empty-body]
    ...


@BUILTINS.class_(egg_sort="Rational")
class Rational(BaseExpr):

  @BUILTINS.method(egg_fn="rational")
  def __init__(self, num: i64Like, den: i64Like):
    ...

  @BUILTINS.method(egg_fn="to-f64")
  def to_f64(self) -> f64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="+")
  def __add__(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="-")
  def __sub__(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="*")
  def __mul__(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="/")
  def __truediv__(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="min")
  def min(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="max")
  def max(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="neg")
  def __neg__(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="abs")
  def __abs__(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="floor")
  def floor(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="ceil")
  def ceil(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="round")
  def round(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="pow")
  def __pow__(self, other: Rational) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="log")
  def log(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="sqrt")
  def sqrt(self) -> Rational:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="cbrt")
  def cbrt(self) -> Rational:  # type: ignore[empty-body]
    ...


@BUILTINS.class_(egg_sort="Vec")
class Vec(BaseExpr, Generic[T]):

  @BUILTINS.method(egg_fn="vec-of")
  def __init__(self, *args: T) -> None:
    ...

  @BUILTINS.method(egg_fn="vec-empty")
  @classmethod
  def empty(cls) -> Vec[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-append")
  def append(self, *others: Vec[T]) -> Vec[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-push")
  def push(self, value: T) -> Vec[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-pop")
  def pop(self) -> Vec[T]:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-not-contains")
  def not_contains(self, value: T) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-contains")
  def contains(self, value: T) -> Unit:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-length")
  def length(self) -> i64:  # type: ignore[empty-body]
    ...

  @BUILTINS.method(egg_fn="vec-get")
  def __getitem__(self, index: i64Like) -> T:  # type: ignore[empty-body]
    ...
