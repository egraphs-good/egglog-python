from egglog import Maybe, String, catch, f64, function, i64

Maybe[int].none()  # error: type-var
Maybe[i64].none(i64(1))  # error: call-arg
Maybe[i64].some()  # error: call-arg
Maybe[i64].some(f64(1.0))  # error: arg-type
Maybe[i64].some(1)  # error: arg-type
Maybe[i64].some(i64(1), i64(2))  # error: call-arg
some = Maybe[i64].some(i64(1))
some.unwrap(i64(1))  # error: call-arg
some.unwrap_or()  # error: call-arg
some.unwrap_or(f64(1.0))  # error: arg-type
some.unwrap_or(1)  # error: arg-type
wrong_result: f64 = some.unwrap()  # error: assignment
wrong_value: i64 = some.value  # error: assignment
wrong_nested: Maybe[Maybe[i64]] = some  # error: assignment
catch(i64(1))  # error: arg-type
some.match()  # error: call-arg
some.match(lambda item: item, i64(0), i64(1))  # error: call-arg


@function
def wrong_match_input(item: f64) -> i64: ...


some.match(wrong_match_input, i64(0))  # error: arg-type
matched = some.match(lambda item: item.to_string(), String("none"))
wrong_matched: i64 = matched  # error: assignment
