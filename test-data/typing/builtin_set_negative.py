from egglog import Set, SetLike, f64, function, i64

Set[i64](f64(1.0))  # error: arg-type
Set[i64](1)  # error: arg-type
Set[i64](args=(i64(1),))  # error: call-arg
Set[int](1)  # error: type-var
Set[i64, f64](i64(1))  # error: misc
Set[i64].empty(i64(1))  # error: call-arg
owned = Set[i64](i64(1))
wrong_operand = Set(f64(1.0))
owned.insert(f64(1.0))  # error: arg-type
owned.remove("bad")  # error: arg-type
owned.contains(f64(1.0))  # error: arg-type
owned.not_contains(f64(1.0))  # error: arg-type
owned | wrong_operand  # error: operator
owned - wrong_operand  # error: operator
owned & wrong_operand  # error: operator
owned.get(0)  # error: attr-defined
owned[0]  # error: index
wrong_result: Set[f64] = owned  # error: assignment
wrong_value: set[int] = owned.value  # error: assignment
wrong_length: int = owned.length()  # error: assignment


@function
def size(value: SetLike[i64, int]) -> i64: ...


size({"bad"})  # error: arg-type
size(wrong_operand)  # error: arg-type
size([1])  # error: arg-type
size(frozenset({1}))  # error: arg-type
