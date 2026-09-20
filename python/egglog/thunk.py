from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Generic, TypeVar, Unpack

from typing_extensions import TypeVarTuple

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
    # Resolved value thunks are a hot path and never need synchronization.
    # Allocate a lock only for thunks which can transition out of Unresolved.
    _lock: RLock | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def fn(cls, fn: Callable[[Unpack[TS]], T], *args: *TS, context: str | None = None) -> Thunk[T, *TS]:
        """
        Create a thunk based on some functions and some partial args.

        Recursive calls raise an exception unless the resolver has supplied a partial value.
        """
        thunk = cls(Unresolved(fn, args, context))
        thunk._lock = RLock()
        return thunk

    @classmethod
    def value(cls, value: T) -> Thunk[T]:
        return Thunk(Resolved(value))

    def set_partial(self, value: T) -> None:
        """Allow recursive calls on the resolving thread to access an unfinished value."""
        lock = self._lock
        if lock is None:
            msg = "Cannot set a partial value outside thunk resolution"
            raise ValueError(msg)
        with lock:
            if not isinstance(self.state, Resolving):
                msg = "Cannot set a partial value outside thunk resolution"
                raise ValueError(msg)  # noqa: TRY004
            self.state = Resolving(Resolved(value))

    def __call__(self) -> T:
        # Resolved values never change, so cached calls do not need the lock.
        if isinstance(state := self.state, Resolved):
            return state.value
        # Other threads wait for resolution; recursive calls on this thread
        # can reenter the lock and still detect the Resolving state.
        lock = self._lock
        assert lock is not None
        with lock:
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
