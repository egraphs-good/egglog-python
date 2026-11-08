from typing import assert_type

from egglog import Pair, convert, f64, i64

owned = Pair[i64, f64](left=i64(1), right=f64(2.0))
inferred = Pair(i64(1) + 2, f64(3.0))
assert_type(owned, Pair[i64, f64])
assert_type(inferred, Pair[i64, f64])
assert_type(owned.left, i64)
assert_type(owned.right, f64)
assert_type(owned.value, tuple[i64, f64])
assert_type(owned.value[0].value, int)
assert_type(owned.value[1].value, float)
assert_type(convert((1, 2.0), Pair[i64, f64]), Pair[i64, f64])
assert_type(Pair(owned, i64(3)).value, tuple[Pair[i64, f64], i64])

match owned:
    case Pair((left, right)):
        assert_type(left, i64)
        assert_type(right, f64)
