from collections.abc import Iterator
from typing import assert_type

from egglog import Pair, Set, SetLike, String, Unit, convert, function, i64

owned = Set[i64](i64(1), i64(2))
assert_type(owned, Set[i64])
assert_type(Set(i64(1) + 1, i64(2)), Set[i64])
assert_type(Set[i64](), Set[i64])
assert_type(Set[i64].empty(), Set[i64])
assert_type(owned.value, set[i64])
assert_type(iter(owned), Iterator[i64])
assert_type(len(owned), int)
assert_type(i64(1) in owned, bool)
assert_type(owned.insert(i64(3)), Set[i64])
assert_type(owned.remove(i64(1)), Set[i64])
assert_type(owned.contains(i64(1)), Unit)
assert_type(owned.not_contains(i64(3)), Unit)
assert_type(owned.length(), i64)
assert_type(owned | Set(i64(3)), Set[i64])
assert_type(owned - Set(i64(1)), Set[i64])
assert_type(owned & Set(i64(1)), Set[i64])
assert_type(convert({1, 2}, Set[i64]), Set[i64])
assert_type(convert(set(), Set[i64]), Set[i64])
assert_type(Set(Pair(i64(1), String("a"))), Set[Pair[i64, String]])

match owned:
    case Set(children):
        assert_type(children, set[i64])


@function
def size(value: SetLike[i64, int]) -> i64: ...


assert_type(size(owned), i64)
assert_type(size({1, 2}), i64)
assert_type(size(set()), i64)
