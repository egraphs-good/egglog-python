from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import TypeVarTuple, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["Thunk", "split_thunk"]

T = TypeVar("T")
TS = TypeVarTuple("TS")
V = TypeVar("V")


def split_thunk(fn: Callable[[], tuple[T, V]]) -> tuple[Callable[[], T], Callable[[], V]]:
    s = _Split(fn)
    return s.left, s.right


@dataclass
class _Split(Generic[T, V]):
    fn: Callable[[], tuple[T, V]]

    def left(self) -> T:
        return self.fn()[0]

    def right(self) -> V:
        return self.fn()[1]


@dataclass
class Thunk(Generic[T, Unpack[TS]]):
    """
    Cached delayed function call.
    """

    state: Resolved[T] | Unresolved[T, Unpack[TS]] | Resolving[T] | Error

    @classmethod
    def fn(
        cls, fn: Callable[[Unpack[TS]], T], *args: Unpack[TS], fallback: Callable[[], T] | None = None
    ) -> Thunk[T, Unpack[TS]]:
        """
        Create a thunk based on some functions and some partial args.

        If the function is called while it is being resolved recursively, will instead return the fallback, if provided.
        """
        return cls(Unresolved(fn, args, fallback))

    @classmethod
    def value(cls, value: T) -> Thunk[T]:
        return Thunk(Resolved(value))

    def __call__(self) -> T:
        match self.state:
            case Resolved(value):
                return value
            case Unresolved(fn, args, fallback):
                self.state = Resolving(fallback)
                try:
                    res = fn(*args)
                except Exception as e:
                    self.state = Error(e)
                    raise
                else:
                    self.state = Resolved(res)
                    return res
            case Resolving(fallback):
                if fallback is None:
                    msg = "Recursively resolving thunk without fallback"
                    raise ValueError(msg)
                return fallback()
            case Error(e):
                raise e


@dataclass
class Resolved(Generic[T]):
    value: T


@dataclass
class Unresolved(Generic[T, Unpack[TS]]):
    fn: Callable[[Unpack[TS]], T]
    args: tuple[Unpack[TS]]
    fallback: Callable[[], T] | None


@dataclass
class Resolving(Generic[T]):
    fallback: Callable[[], T] | None


@dataclass
class Error:
    e: Exception
