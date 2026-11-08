from egglog import (
    MultiSet,
    MultiSetLike,
    String,
    f64,
    function,
    i64,
    multiset_contains_swapped,
    multiset_flat_map,
    multiset_fold,
    multiset_remove_swapped,
    multiset_subtract_swapped,
)

MultiSet[i64](f64(1.0))  # error: arg-type
MultiSet[i64](1)  # error: arg-type
MultiSet[i64](args=(i64(1),))  # error: call-arg
MultiSet[int](1)  # error: type-var
MultiSet[i64, f64](i64(1))  # error: misc
MultiSet[i64].single(i64(1))  # error: call-arg
MultiSet[i64].single(i64(1), String("bad"))  # error: arg-type
owned = MultiSet[i64](i64(1))
other = MultiSet[f64](f64(1.0))
owned.insert(f64(1.0))  # error: arg-type
owned.remove(f64(1.0))  # error: arg-type
owned.count(f64(1.0))  # error: arg-type
owned.contains(f64(1.0))  # error: arg-type
owned.not_contains(f64(1.0))  # error: arg-type
owned + other  # error: operator
owned - other  # error: operator
owned & other  # error: operator
owned[0]  # error: index
MultiSet[MultiSet[i64]](other)  # error: arg-type
MultiSet[i64].sum_multisets(owned)  # error: arg-type
multiset_remove_swapped(i64(1))  # error: call-arg
multiset_contains_swapped(i64(1))  # error: call-arg
multiset_subtract_swapped(i64(1), owned)  # error: arg-type
removed = multiset_remove_swapped(i64(1), owned)
wrong_result: MultiSet[f64] = removed  # error: assignment
wrong_value: tuple[i64, ...] = owned.value  # error: assignment
wrong_set: set[i64] = owned.value  # error: assignment
wrong_length: int = owned.length()  # error: assignment


@function
def size(value: MultiSetLike[i64, int]) -> i64: ...


size(("bad",))  # error: arg-type
size(other)  # error: arg-type
size([1])  # error: arg-type
size({1})  # error: arg-type


@function
def stringify(value: i64) -> String: ...


@function
def wrong_input(value: f64) -> i64: ...


@function
def wrong_index(values: MultiSet[i64], value: i64) -> String: ...


owned.map(wrong_input)  # error: arg-type
mapped = owned.map(stringify)
wrong_mapped: MultiSet[i64] = mapped  # error: assignment
owned.filter(stringify)  # error: arg-type
owned.filter_not(stringify)  # error: arg-type
multiset_flat_map(stringify, owned)  # error: arg-type
multiset_fold(i64.__add__, f64(0.0), owned)  # error: misc
owned.fill_index(wrong_index)  # error: arg-type
owned.clear_index(wrong_index)  # error: arg-type
