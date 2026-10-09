from collections.abc import Iterator
from typing import assert_type

from egglog import Pair, Vec, VecLike, convert, f64, function, i64

owned = Vec[i64](i64(1), i64(2))
inferred = Vec(i64(1) + 2, i64(3))
assert_type(owned, Vec[i64])
assert_type(inferred, Vec[i64])
assert_type(Vec[i64](), Vec[i64])
assert_type(Vec[i64].empty(), Vec[i64])
assert_type(owned[0], i64)
assert_type(owned[i64(0)], i64)
assert_type(owned.value, tuple[i64, ...])
assert_type(iter(owned), Iterator[i64])
assert_type(len(owned), int)
assert_type(i64(1) in owned, bool)
assert_type(convert([1, 2], Vec[i64]), Vec[i64])
assert_type(convert((), Vec[f64]), Vec[f64])
assert_type(Vec(Pair(i64(1), f64(2.0))), Vec[Pair[i64, f64]])

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
