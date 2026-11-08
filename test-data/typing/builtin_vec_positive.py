from collections.abc import Iterator
from typing import TypeVar, assert_type

from egglog import BaseExpr, Expr, Pair, String, Unit, Vec, VecLike, convert, f64, function, i64

owned = Vec[i64](i64(1), i64(2))
inferred = Vec(i64(1) + 2, i64(3))
assert_type(owned, Vec[i64])
assert_type(inferred, Vec[i64])
assert_type(Vec[i64](), Vec[i64])
assert_type(Vec[i64].empty(), Vec[i64])
assert_type(owned[0], i64)
assert_type(owned[i64(0)], i64)
assert_type(owned.append(), Vec[i64])
assert_type(owned.append(owned, [i64(3)], (i64(4),)), Vec[i64])
assert_type(owned.push(i64(3)), Vec[i64])
assert_type(owned.pop(), Vec[i64])
assert_type(owned.length(), i64)
assert_type(owned.set(0, i64(3)), Vec[i64])
assert_type(owned.remove(0), Vec[i64])
assert_type(owned.contains(i64(3)), Unit)
assert_type(owned.not_contains(i64(3)), Unit)


class Atom(Expr): ...


assert_type(Vec[Atom].empty() | Vec[Atom].empty(), Vec[Atom])
T = TypeVar("T", bound=BaseExpr)


def same_type_union(left: Vec[T], right: Vec[T]) -> Vec[T]:
    return left | right


assert_type(owned.value, tuple[i64, ...])
assert_type(iter(owned), Iterator[i64])
assert_type(len(owned), int)
assert_type(i64(1) in owned, bool)
assert_type(convert([1, 2], Vec[i64]), Vec[i64])
assert_type(convert((), Vec[f64]), Vec[f64])
assert_type(Vec(Pair(i64(1), f64(2.0))), Vec[Pair[i64, f64]])
assert_type(i64(5).range(), Vec[i64])
assert_type(i64(5).range()[0], i64)
assert_type(i64(5).range().value, tuple[i64, ...])

match owned:
    case Vec(children):
        assert_type(children, tuple[i64, ...])


@function
def vector() -> Vec[i64]: ...


@function
def size(value: VecLike[i64, int]) -> i64: ...


assert_type(vector(), Vec[i64])
assert_type(size((1, 2)), i64)
assert_type(size([1, 2]), i64)
assert_type(size(owned), i64)


def stringify(value: i64) -> String: ...


def atomize(value: i64) -> Atom: ...


assert_type(owned.map(stringify), Vec[String])
assert_type(owned.map(atomize), Vec[Atom])
assert_type(owned.map(lambda value: value + 1), Vec[i64])
