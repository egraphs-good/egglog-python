from typing import assert_type

from egglog import Expr, Maybe, convert, f64, function, i64


class Atom(Expr):
    def __init__(self, value: i64) -> None: ...


none = Maybe[i64].none()
some = Maybe[i64].some(i64(1))
assert_type(none, Maybe[i64])
assert_type(some, Maybe[i64])
assert_type(Maybe.some(i64(1)), Maybe[i64])
assert_type(some.value, i64 | None)
assert_type(none.unwrap(), i64)
assert_type(some.unwrap_or(i64(2)), i64)
assert_type(Maybe[f64].some(f64(1.0)), Maybe[f64])
assert_type(Maybe[Atom].some(Atom(i64(1))).unwrap(), Atom)
assert_type(Maybe[Maybe[i64]].some(none).value, Maybe[i64] | None)
assert_type(convert(None, Maybe[i64]), Maybe[i64])

match some:
    case Maybe(child):
        assert_type(child, i64 | None)


@function
def identity(value: Maybe[i64]) -> Maybe[i64]: ...


assert_type(identity(some), Maybe[i64])
