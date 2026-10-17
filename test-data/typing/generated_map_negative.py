from egglog import Map, MapLike, Pair, Vec, f64, function, i64, map_fold_kv

Map[i64, f64]()  # error: call-arg
Map[i64, f64](i64(1))  # error: arg-type
Map[int, f64].empty()  # error: type-var
Map[i64].empty()  # error: misc
Map[i64, f64].empty(i64(1))  # error: call-arg
value = Map[i64, f64].empty()
value.insert(f64(1.0), f64(2.0))  # error: arg-type
value.insert(i64(1), i64(2))  # error: arg-type
value.insert(1, f64(2.0))  # error: arg-type
value.insert(i64(1))  # error: call-arg
value.insert(key=i64(1), item=f64(2.0))  # error: call-arg
Map[i64, f64].from_pairs()  # error: call-arg
Map.from_pairs(Pair(i64(1), f64(2.0)))  # error: arg-type
wrong_entries = Vec(Pair(f64(1.0), i64(2)))
Map[i64, f64].from_pairs(wrong_entries)  # error: arg-type
Map[i64, f64].from_pairs(items=Vec(Pair(i64(1), f64(2.0))))  # error: call-arg
value[f64(1.0)]  # error: index
value[1]  # error: index
wrong_result: Map[f64, i64] = value  # error: assignment
wrong_item: i64 = value[i64(1)]  # error: assignment
wrong_value: dict[int, float] = value.value  # error: assignment
Map.of(i64(1), f64(2.0))  # error: attr-defined
value.remove(f64(1.0))  # error: arg-type
value.remove()  # error: call-arg
value.contains(f64(1.0))  # error: arg-type
value.not_contains(f64(1.0))  # error: arg-type
value.length(i64(1))  # error: call-arg


@function
def size(value: MapLike[i64, f64, int, float]) -> i64: ...


size({"wrong": 2.0})  # error: dict-item
size({1: "wrong"})  # error: dict-item


@function
def wrong_fold(total: i64, key: f64, item: i64) -> i64: ...


map_fold_kv(wrong_fold, i64(0), value)  # error: arg-type
map_fold_kv(lambda total, _key, _item: total, i64(0))  # error: call-arg
folded = map_fold_kv(lambda total, _key, item: total + item, f64(0.0), value)
wrong_folded: i64 = folded  # error: assignment
