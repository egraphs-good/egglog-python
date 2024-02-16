# mypy: disable-error-code="empty-body"

"""
N-Dimensional Arrays
====================

Example of building NDarray in the vein of Mathemetics of Arrays.
"""

from __future__ import annotations

from egglog import *

egraph = EGraph()


class Value(Expr):
    def __init__(self, v: i64Like) -> None: ...

    def __mul__(self, other: Value) -> Value: ...

    def __add__(self, other: Value) -> Value: ...


i, j = vars_("i j", i64)
egraph.register(
    rewrite(Value(i) * Value(j)).to(Value(i * j)),
    rewrite(Value(i) + Value(j)).to(Value(i + j)),
)


class Values(Expr):
    def __init__(self, v: Vec[Value]) -> None: ...

    def __getitem__(self, idx: Value) -> Value: ...

    def length(self) -> Value: ...

    def concat(self, other: Values) -> Values: ...


@egraph.register
def _values(vs: Vec[Value], other: Vec[Value]):
    yield rewrite(Values(vs)[Value(i)]).to(vs[i])
    yield rewrite(Values(vs).length()).to(Value(vs.length()))
    yield rewrite(Values(vs).concat(Values(other))).to(Values(vs.append(other)))
    # yield rewrite(l.concat(r).length()).to(l.length() + r.length())
    # yield rewrite(l.concat(r)[idx])


class NDArray(Expr):
    """
    An n-dimensional array.
    """

    def __getitem__(self, idx: Values) -> Value: ...

    def shape(self) -> Values: ...


@function
def arange(n: Value) -> NDArray: ...


@egraph.register
def _ndarray_arange(n: Value, idx: Values):
    yield rewrite(arange(n).shape()).to(Values(Vec(n)))
    yield rewrite(arange(n)[idx]).to(idx[Value(0)])


def assert_simplifies(left: Expr, right: Expr) -> None:
    """
    Simplify and print
    """
    egraph.register(left)
    egraph.run(30)
    res = egraph.extract(left)
    print(f"{left} == {right} ➡  {res}")
    egraph.check(eq(left).to(right))


assert_simplifies(arange(Value(10)).shape(), Values(Vec(Value(10))))
assert_simplifies(arange(Value(10))[Values(Vec(Value(0)))], Value(0))
assert_simplifies(arange(Value(10))[Values(Vec(Value(1)))], Value(1))


@function
def py_value(s: StringLike) -> Value: ...


@egraph.register
def _py_value(l: String, r: String):
    yield rewrite(py_value(l) + py_value(r)).to(py_value(join(l, " + ", r)))
    yield rewrite(py_value(l) * py_value(r)).to(py_value(join(l, " * ", r)))


@function
def py_values(s: StringLike) -> Values: ...


@egraph.register
def _py_values(l: String, r: String):
    yield rewrite(py_values(l)[py_value(r)]).to(py_value(join(l, "[", r, "]")))
    yield rewrite(py_values(l).length()).to(py_value(join("len(", l, ")")))
    yield rewrite(py_values(l).concat(py_values(r))).to(py_values(join(l, " + ", r)))


@function
def py_ndarray(s: StringLike) -> NDArray: ...


@egraph.register
def _py_ndarray(l: String, r: String):
    yield rewrite(py_ndarray(l)[py_values(r)]).to(py_value(join(l, "[", r, "]")))
    yield rewrite(py_ndarray(l).shape()).to(py_values(join(l, ".shape")))
    yield rewrite(arange(py_value(l))).to(py_ndarray(join("np.arange(", l, ")")))


assert_simplifies(py_ndarray("x").shape(), py_values("x.shape"))
assert_simplifies(arange(py_value("x"))[py_values("y")], py_value("np.arange(x)[y]"))
# assert_simplifies(arange(py_value("x"))[py_values("y")], py_value("y[0]"))


@function
def cross(l: NDArray, r: NDArray) -> NDArray: ...


@egraph.register
def _cross(l: NDArray, r: NDArray, idx: Values):
    yield rewrite(cross(l, r).shape()).to(l.shape().concat(r.shape()))
    yield rewrite(cross(l, r)[idx]).to(l[idx] * r[idx])


assert_simplifies(cross(arange(Value(10)), arange(Value(11))).shape(), Values(Vec(Value(10), Value(11))))
assert_simplifies(cross(py_ndarray("x"), py_ndarray("y")).shape(), py_values("x.shape + y.shape"))
assert_simplifies(cross(py_ndarray("x"), py_ndarray("y"))[py_values("idx")], py_value("x[idx] * y[idx]"))


@egraph.register
def _cross_py(l: String, r: String):
    yield rewrite(cross(py_ndarray(l), py_ndarray(r))).to(py_ndarray(join("np.multiply.outer(", l, ", ", r, ")")))


assert_simplifies(cross(py_ndarray("x"), py_ndarray("y"))[py_values("idx")], py_value("np.multiply.outer(x, y)[idx]"))
