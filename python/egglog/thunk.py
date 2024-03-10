from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar, TypeVarTuple, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["Thunk"]

T = TypeVar("T")
P = ParamSpec("P")
TS = TypeVarTuple("TS")


@dataclass
class Thunk(Generic[T, Unpack[TS]]):
    """
    Cached delayed function call.
    """

    state: Resolved[T] | Unresolved[T, Unpack[TS]]

    @classmethod
    def fn(cls, fn: Callable[[Unpack[TS]], T], *args: *tuple[Unpack[TS]]) -> Thunk[T, Unpack[TS]]:
        return cls(Unresolved(fn, args))

    @classmethod
    def value(cls, value: T) -> Thunk[T]:
        return Thunk(Resolved(value))

    def __call__(self) -> T:
        match self.state:
            case Resolved(value):
                return value
            case Unresolved(fn, args):
                res = fn(*args)
                self.state = Resolved(res)
                return res


@dataclass
class Resolved(Generic[T]):
    value: T


@dataclass
class Unresolved(Generic[T, Unpack[TS]]):
    fn: Callable[[Unpack[TS]], T]
    args: tuple[Unpack[TS]]
