"""
Example module to replicate behavior expressed in

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


class LoopNestAPI(Expr, ruleset=array_api_ruleset):
    def __init__(self, last: Int, before: OptionalLoopNestAPI) -> None: ...

    @classmethod
    def from_tuple(cls, args: TupleIntLike) -> OptionalLoopNestAPI: ...

    @method(preserve=True)
    def __iter__(self) -> Iterator[TupleInt]:
        return iter(self.indices())

    def indices(self) -> TupleTupleInt:
        return self.get_dims().map_tuple_int(TupleInt.range).product()

    def get_dims(self) -> TupleInt: ...


@array_api_ruleset.register
def _loopnest_api_ruleset(lna: LoopNestAPI, dim: Int, ti: TupleInt, idx_fn: Callable[[Int], Int], i: i64):
    # from_tuple
    yield rewrite(LoopNestAPI.from_tuple(TupleInt.EMPTY), subsume=True).to(OptionalLoopNestAPI.NONE)
    yield rewrite(LoopNestAPI.from_tuple(ti.append(dim)), subsume=True).to(
        OptionalLoopNestAPI(LoopNestAPI(dim, LoopNestAPI.from_tuple(ti)))
    )
    # get_dims
    yield rewrite(LoopNestAPI(dim, OptionalLoopNestAPI.NONE).get_dims(), subsume=True).to(TupleInt.single(dim))
    yield rewrite(LoopNestAPI(dim, OptionalLoopNestAPI(lna)).get_dims(), subsume=True).to(lna.get_dims().append(dim))
    # unwrap
    yield rewrite(OptionalLoopNestAPI(lna).unwrap()).to(lna)
