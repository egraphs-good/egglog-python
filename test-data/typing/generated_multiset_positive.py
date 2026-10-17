from collections.abc import Iterator
from typing import assert_type

from egglog import (
    Expr,
    MultiSet,
    MultiSetLike,
    Unit,
    convert,
    function,
    i64,
    multiset_contains_swapped,
    multiset_not_contains_swapped,
    multiset_remove_swapped,
    multiset_subtract_swapped,
)

owned = MultiSet[i64](i64(2), i64(1), i64(2))
assert_type(owned, MultiSet[i64])
assert_type(MultiSet(i64(1) + 1, i64(2)), MultiSet[i64])
assert_type(MultiSet[i64](), MultiSet[i64])
assert_type(owned.value, list[i64])
assert_type(owned.eval(), list[i64])
assert_type(iter(owned), Iterator[i64])
assert_type(len(owned), int)
assert_type(i64(1) in owned, bool)
assert_type(MultiSet[i64].single(i64(1), 2), MultiSet[i64])
assert_type(owned.pick(), i64)
assert_type(owned.pick_max(), i64)
assert_type(owned.insert(i64(3)), MultiSet[i64])
assert_type(owned.remove(i64(1)), MultiSet[i64])
assert_type(owned - owned, MultiSet[i64])
assert_type(owned & owned, MultiSet[i64])
assert_type(owned + owned, MultiSet[i64])
assert_type(owned.reset_counts(), MultiSet[i64])
assert_type(owned.length(), i64)
assert_type(owned.count(i64(2)), i64)
assert_type(owned.contains(i64(1)), Unit)
assert_type(owned.not_contains(i64(3)), Unit)
assert_type(multiset_remove_swapped(i64(2), owned), MultiSet[i64])
assert_type(multiset_subtract_swapped(MultiSet(i64(1)), owned), MultiSet[i64])
assert_type(multiset_contains_swapped(i64(1), owned), Unit)
assert_type(multiset_not_contains_swapped(i64(3), owned), Unit)
assert_type(MultiSet.sum_multisets(MultiSet(owned, owned)), MultiSet[i64])
assert_type(convert((), MultiSet[i64]), MultiSet[i64])
assert_type(convert((1, 2, 1), MultiSet[i64]), MultiSet[i64])

match owned:
    case MultiSet(children):
        assert_type(children, list[i64])


class Atom(Expr):
    def __init__(self, value: i64) -> None: ...


assert_type(MultiSet(Atom(i64(1))).union_values(), Atom)


@function
def size(value: MultiSetLike[i64, int]) -> i64: ...


assert_type(size(owned), i64)
assert_type(size((1, 2, 1)), i64)
assert_type(size(()), i64)
