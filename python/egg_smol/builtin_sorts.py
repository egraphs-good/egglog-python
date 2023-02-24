from __future__ import annotations
from typing import TypeVar, NoReturn


T = TypeVar("T")


class SupportsNotEqual:
    def __ne__(self: T, __o: T) -> Unit:  # type: ignore
        ...

    def __eq__(self, other: NoReturn) -> NoReturn:  # type: ignore
        raise NotImplementedError()


@sort(lit_type=type(None))
class Unit(SupportsNotEqual):
    def __init__(self) -> None:
        ...


@sort(lit_type=int)
class i64(SupportsNotEqual):
    def __init__(self, value: int) -> None:
        ...
1
    @builtins.function("+")
    def __add__(self, other: i64) -> i64:
        ...

    @builtins.function("-")
    def __sub__(self, other: i64) -> i64:
        ...

    @builtins.function("*")
    def __mul__(self, other: i64) -> i64:
        ...

    @builtins.function("/")
    def __truediv__(self, other: i64) -> i64:
        ...

    @builtins.function("%")
    def __mod__(self, other: i64) -> i64:
        ...

    @builtins.function("&")
    def __and__(self, other: i64) -> i64:
        ...

    @builtins.function("|")
    def __or__(self, other: i64) -> i64:
        ...

    @builtins.function("^")
    def __xor__(self, other: i64) -> i64:
        ...

    @builtins.function("<<")
    def __lshift__(self, other: i64) -> i64:
        ...

    @builtins.function(">>")
    def __rshift__(self, other: i64) -> i64:
        ...

    @builtins.function("not-64")
    def __invert__(self) -> i64:
        ...

    @builtins.function("<")
    def __lt__(self, other: i64) -> Unit:  # type: ignore
        ...

    @builtins.function(">")
    def __gt__(self, other: i64) -> Unit:
        ...

    @builtins.function("min")
    def min(self, other: i64) -> i64:
        ...

    @builtins.function("max")
    def max(self, other: i64) -> i64:
        ...
