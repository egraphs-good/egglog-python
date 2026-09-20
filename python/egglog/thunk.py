from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, Unpack

from typing_extensions import TypeVarTuple

from ._threading import INITIALIZATION_ACTIVE, initialize

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
class Thunk(Generic[T, *TS]):
    """
    Cached delayed function call.
    """

    state: Resolved[T] | Unresolved[T, *TS] | Resolving[T] | Error

    @classmethod
    def fn(cls, fn: Callable[[Unpack[TS]], T], *args: *TS, context: str | None = None) -> Thunk[T, *TS]:
        """
        Create a thunk based on some functions and some partial args.

        Recursive calls raise an exception unless the resolver has supplied a partial value.
        """
        return cls(Unresolved(fn, args, context))

    @classmethod
    def value(cls, value: T) -> Thunk[T]:
        return Thunk(Resolved(value))

    def set_partial(self, value: T) -> None:
        """Allow recursive calls on the resolving thread to access an unfinished value."""
        with initialize():
            if not isinstance(self.state, Resolving):
                msg = "Cannot set a partial value outside thunk resolution"
                raise ValueError(msg)  # noqa: TRY004
            self.state = Resolving(Resolved(value))

    def __call__(self) -> T:
        if isinstance(state := self.state, Resolved) and not INITIALIZATION_ACTIVE.locked():
            return state.value
        # A resolved value can transitively contain a declaration owned by the
        # active initializer, so readers wait and recheck while it is partial.
        with initialize():
            match self.state:
                case Resolved(value):
                    return value
                case Unresolved(fn, args, context):
                    self.state = Resolving()
                    try:
                        res = fn(*args)
                    except BaseException as e:
                        self.state = Error(e, context)
                        raise
                    else:
                        self.state = Resolved(res)
                        return res
                case Resolving(Resolved(value)):
                    return value
                case Resolving():
                    msg = "Recursively resolving thunk"
                    raise ValueError(msg)
                case Error(e):
                    raise e


@dataclass
class Resolved(Generic[T]):
    value: T


@dataclass
class Unresolved(Generic[T, *TS]):
    fn: Callable[[Unpack[TS]], T]
    args: tuple[*TS]
    context: str | None


@dataclass
class Resolving(Generic[T]):
    partial: Resolved[T] | None = None


@dataclass
class Error:
    e: BaseException
    context: str | None
