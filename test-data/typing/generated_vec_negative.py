from egglog import Pair, Vec, VecLike, f64, function, i64

Vec[i64](f64(1.0))  # error: arg-type
Vec[i64](i64(1), f64(2.0))  # error: arg-type
Vec[i64](1)  # error: arg-type
Vec[i64](args=(i64(1),))  # error: call-arg
Vec[int](1)  # error: type-var
Vec[i64, f64](i64(1))  # error: misc
Vec[i64].empty(i64(1))  # error: call-arg
value = Vec[i64](i64(1))
value[f64(0.0)]  # error: index
value["0"]  # error: index
value[0, 1]  # error: index
wrong_result: Vec[f64] = value  # error: assignment
wrong_item: f64 = value[0]  # error: assignment
wrong_range: Vec[f64] = i64(5).range()  # error: assignment
i64(5).range(1)  # error: call-arg
f64(5.0).range()  # error: attr-defined
i64.range(f64(5.0))  # error: arg-type
wrong_value: tuple[int, ...] = value.value  # error: assignment
wrong_pair = Pair(f64(1.0), i64(2))
Vec[Pair[i64, f64]](wrong_pair)  # error: arg-type


@function
def size(value: VecLike[i64, int]) -> i64: ...


size(["wrong"])  # error: list-item
size(Vec(f64(1.0)))  # error: arg-type
