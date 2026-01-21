# # Container Primitive Sorts
#
# In egglog, we not only have primitive sorts like `i64`, `f64` and `String`, but also container
# sorts like `Vec`, `Set` and `Map`. Similar to other primitives, they have a number of built-in
# functions that are defined on them. They are implemented in Rust and egglog can also be extended
# with extra user defined functions on these container sorts as well as entirely new sorts.
#
# For example, the vector sort `Vec` represents a variable-length ordered collection of elements.
# It supports indexing, length retrieval, appending elements, and more:

# +
# mypy: disable-error-code="empty-body"

from __future__ import annotations
from typing import TypeAlias
from egglog import *


egraph = EGraph(save_egglog_string=True)
# vectors support indexing
egraph.extract(Vec(i64(0), i64(1), i64(2))[0])

# -

# This is translated into egglog primitives which execute "eagerly", i.e. they don't have to wait for a rule to replace
# their execution with concrete values. This also means they can also execute on concrete values directly, not on
# uninterpreted functions.

print(egraph.as_egglog_string)

# As an example, let's look at implementing polynomial expressions in egglog:


# +
class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @method(cost=2)
    def __add__(self, other: NumLike) -> Num: ...
    @method(cost=10)
    def __mul__(self, other: NumLike) -> Num: ...

    # These will be translated to non-reversed ones
    def __radd__(self, other: NumLike) -> Num: ...
    def __rmul__(self, other: NumLike) -> Num: ...


NumLike: TypeAlias = Num | StringLike | i64Like
converter(i64, Num, Num)

(x, y, z) = vars_("x y z", Num)
(p := x * (y * x + z) + 100 * z)
# -

# Now let's say we want to find the lowest cost way to compute this by potentially factoring it.
