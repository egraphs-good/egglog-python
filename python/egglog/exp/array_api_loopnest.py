"""
In progress module

https://gist.github.com/sklam/5e5737137d48d6e5b816d14a90076f1d

"""

# %%
# mypy: disable-error-code="empty-body"
from __future__ import annotations

from egglog import *
from egglog.exp.array_api import *

__all__ = ["LoopNestAPI", "OptionalLoopNestAPI", "ShapeAPI"]


class ShapeAPI(Expr):
    def __init__(self, dims: TupleIntLike) -> None: ...

    def deselect(self, axis: TupleIntLike) -> ShapeAPI: ...

    def select(self, axis: TupleIntLike) -> ShapeAPI: ...

    def to_tuple(self) -> TupleInt: ...


@array_api_ruleset.register
def shape_api_ruleset(dims: TupleInt, axis: TupleInt):
    s = ShapeAPI(dims)
    yield rewrite(s.deselect(axis), subsume=True).to(
        ShapeAPI(TupleInt.range(dims.length()).filter(lambda i: ~axis.contains(i)).map(lambda i: dims[i]))
    )
    yield rewrite(s.select(axis), subsume=True).to(
        ShapeAPI(TupleInt.range(dims.length()).filter(lambda i: axis.contains(i)).map(lambda i: dims[i]))
    )
    yield rewrite(s.to_tuple(), subsume=True).to(dims)


class OptionalLoopNestAPI(Expr):
    def __init__(self, value: LoopNestAPI) -> None: ...

    NONE: ClassVar[OptionalLoopNestAPI]

    def unwrap(self) -> LoopNestAPI: ...


class LoopNestAPI(Expr):
    def __init__(self, dim: Int, inner: OptionalLoopNestAPI) -> None: ...

    @classmethod
    def from_tuple(cls, args: TupleInt) -> OptionalLoopNestAPI: ...

    @method(preserve=True)
    def __iter__(self) -> Iterator[TupleInt]:
        return iter(self.indices)

    @property
    def indices(self) -> TupleTupleInt: ...

    def get_dims(self) -> TupleInt: ...

    def fold(self, fn: Callable[[NDArray, TupleInt], NDArray], init: NDArrayLike) -> NDArray: ...


@function
def tuple_tuple_int_reduce_ndarray(
    xs: TupleTupleInt, fn: Callable[[NDArray, TupleInt], NDArray], init: NDArray
) -> NDArray: ...


@function
def tuple_int_map_tuple_int(xs: TupleInt, fn: Callable[[Int], TupleInt]) -> TupleTupleInt: ...


@function
def tuple_tuple_int_product(xs: TupleTupleInt) -> TupleTupleInt: ...


@array_api_ruleset.register
def _loopnest_api_ruleset(
    head: Int,
    tail: TupleInt,
    lna: LoopNestAPI,
    fn: Callable[[NDArray, TupleInt], NDArray],
    init: NDArray,
    dim: Int,
    idx_fn: Callable[[Int], Int],
    i: i64,
):
    # from_tuple
    yield rewrite(LoopNestAPI.from_tuple(TupleInt(0, idx_fn)), subsume=True).to(OptionalLoopNestAPI.NONE)
    yield rewrite(LoopNestAPI.from_tuple(TupleInt(Int(i), idx_fn)), subsume=True).to(
        OptionalLoopNestAPI(
            LoopNestAPI(idx_fn(Int(0)), LoopNestAPI.from_tuple(TupleInt(Int(i - 1), lambda i: idx_fn(i + 1))))
        ),
        ne(i).to(i64(0)),
    )
    # reduce
    yield rewrite(lna.fold(fn, init), subsume=True).to(tuple_tuple_int_reduce_ndarray(lna.indices, fn, init))
    # get_dims
    yield rewrite(LoopNestAPI(dim, OptionalLoopNestAPI.NONE).get_dims(), subsume=True).to(TupleInt.single(dim))
    yield rewrite(LoopNestAPI(dim, OptionalLoopNestAPI(lna)).get_dims(), subsume=True).to(
        TupleInt.single(dim) + lna.get_dims()
    )
    # indices
    yield rewrite(lna.indices, subsume=True).to(
        tuple_tuple_int_product(tuple_int_map_tuple_int(lna.get_dims(), TupleInt.range))
    )
