# mypy: disable-error-code="empty-body"
from __future__ import annotations

from egglog import *
from egglog.exp.array_api import *


class ShapeAPI(Expr):
    def __init__(self, dims: TupleInt) -> None: ...

    def deselect(self, axis: TupleInt) -> ShapeAPI: ...

    def select(self, axis: TupleInt) -> ShapeAPI: ...

    def to_tuple(self) -> TupleInt: ...


@array_api_ruleset.register
def shape_api_ruleset(dims: TupleInt, axis: TupleInt):  # noqa: ANN201
    s = ShapeAPI(dims)
    yield rewrite(s.deselect(axis)).to(
        ShapeAPI(range_(dims.length()).filter(lambda i: ~axis.contains(i)).map(lambda i: dims[i]))
    )
    yield rewrite(s.select(axis)).to(
        ShapeAPI(range_(dims.length()).filter(lambda i: axis.contains(i)).map(lambda i: dims[i]))
    )
    yield rewrite(s.to_tuple()).to(dims)


class OptionalLoopNestAPI(Expr):
    def __init__(self, value: LoopNestAPI) -> None: ...

    NONE: ClassVar[OptionalLoopNestAPI]

    def unwrap(self) -> LoopNestAPI: ...


class LoopNestAPI(Expr):
    def __init__(self, dim: Int, inner: OptionalLoopNestAPI = OptionalLoopNestAPI.NONE) -> None: ...

    @classmethod
    def from_tuple(cls, args: TupleInt) -> OptionalLoopNestAPI: ...

    def __iter__(self) -> Iterator[TupleInt]:
        return iter(self.indices)

    @property
    def indices(self) -> TupleTupleInt: ...

    def get_dims(self) -> TupleInt: ...

    def reduce(self, fn: Callable[[NDArray, TupleInt], NDArray], init: NDArrayLike) -> NDArray: ...


@array_api_ruleset.register
def _loopnest_api_ruleset(
    head: enp.Int,
    tail: enp.TupleInt,
    lna: LoopNestAPI,
    fn: Callable[[enp.NDArray, enp.TupleInt], enp.NDArray],
    init: enp.NDArray,
    dim: enp.Int,
):
    # from_tuple
    yield rewrite(LoopNestAPI.from_tuple(enp.TupleInt.EMPTY)).to(OptionalLoopNestAPI.NONE)
    yield rewrite(
        LoopNestAPI.from_tuple(enp.TupleInt.some(head, tail)),
    ).to(
        OptionalLoopNestAPI(LoopNestAPI(head, LoopNestAPI.from_tuple(tail))),
    )
    # reduce
    yield rewrite(lna.reduce(fn, init)).to(lna.indices.reduce_ndarray(fn, init))
    # get_dims
    yield rewrite(LoopNestAPI(dim, OptionalLoopNestAPI.NONE).get_dims()).to(enp.TupleInt(dim))
    yield rewrite(LoopNestAPI(dim, OptionalLoopNestAPI(lna)).get_dims()).to(enp.TupleInt(dim) + lna.get_dims())
    # indices
    yield rewrite(lna.indices).to(lna.get_dims().map_tuple_int(enp.range_).product())


@function(ruleset=array_api_ruleset)
def linalg_norm(X: NDArray, axis: TupleInt) -> NDArray:  # noqa: N803
    # peel off the outer shape for result array
    outshape = ShapeAPI(X.shape).deselect(axis).to_tuple()
    # get only the inner shape for reduction
    reduce_axis = ShapeAPI(X.shape).select(axis).to_tuple()

    return NDArray(
        outshape,
        X.dtype,
        lambda k: sqrt(
            LoopNestAPI.from_tuple(reduce_axis)
            .unwrap()
            .reduce(lambda carry, i: carry + real(conj(x := X[i + k]) * x), init=0.0)
        ).to_value(),
    )
