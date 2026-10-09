from egglog import Pair, f64, i64

Pair[i64, f64](f64(1.0), f64(2.0))  # error: arg-type
Pair[i64, f64](i64(1), i64(2))  # error: arg-type
Pair[i64, f64](i64(1))  # error: call-arg
Pair[i64, f64](i64(1), f64(2.0), i64(3))  # error: call-arg
Pair[i64, f64](left=i64(1), right=f64(2.0), unknown=i64(3))  # error: call-arg
Pair[i64, f64](1, f64(2.0))  # error: arg-type
Pair[int, f64](1, f64(2.0))  # error: type-var
Pair[i64](i64(1), i64(2))  # error: misc
value = Pair[i64, f64](i64(1), f64(2.0))
wrong_result: Pair[f64, i64] = value  # error: assignment
wrong_left: f64 = value.left  # error: assignment
wrong_right: i64 = value.right  # error: assignment
wrong_value: tuple[int, float] = value.value  # error: assignment
value.left()  # error: operator
