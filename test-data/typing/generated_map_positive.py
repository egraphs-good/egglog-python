from collections.abc import Iterator
from typing import assert_type

from egglog import Map, MapLike, Pair, Unit, Vec, convert, f64, function, i64, map_fold_kv

empty = Map[i64, f64].empty()
owned = empty.insert(i64(1), f64(2.0))
entries = Vec(Pair(i64(1), f64(2.0)))
assert_type(empty, Map[i64, f64])
assert_type(owned, Map[i64, f64])
assert_type(Map.from_pairs(entries), Map[i64, f64])
assert_type(Map[i64, f64].from_pairs(entries), Map[i64, f64])
assert_type(Map[i64, f64].from_pairs(Vec[Pair[i64, f64]].empty()), Map[i64, f64])
assert_type(owned[i64(1)], f64)
assert_type(owned.remove(i64(1)), Map[i64, f64])
assert_type(owned.length(), i64)
assert_type(owned.contains(i64(1)), Unit)
assert_type(owned.not_contains(i64(1)), Unit)
assert_type(owned.value, dict[i64, f64])
assert_type(iter(owned), Iterator[i64])
assert_type(len(owned), int)
assert_type(i64(1) in owned, bool)
assert_type(convert({1: 2.0}, Map[i64, f64]), Map[i64, f64])
assert_type(convert({}, Map[i64, f64]), Map[i64, f64])
assert_type(Map.from_pairs(Vec(Pair(i64(1), entries))), Map[i64, Vec[Pair[i64, f64]]])

match owned:
    case Map(children):
        assert_type(children, dict[i64, f64])


@function
def vector() -> Vec[Pair[i64, f64]]: ...


@function
def mapping() -> Map[i64, f64]: ...


@function
def size(value: MapLike[i64, f64, int, float]) -> i64: ...


assert_type(Map.from_pairs(vector()), Map[i64, f64])
assert_type(mapping(), Map[i64, f64])
assert_type(size({1: 2.0}), i64)
assert_type(size(owned), i64)


@function
def summarize(total: Pair[i64, f64], key: i64, item: f64) -> Pair[i64, f64]: ...


assert_type(map_fold_kv(f=summarize, initial=Pair(i64(0), f64(0.0)), xs=owned), Pair[i64, f64])
assert_type(map_fold_kv(lambda total, _key, item: total + item, f64(0.0), owned), f64)
