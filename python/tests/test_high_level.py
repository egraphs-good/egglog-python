from __future__ import annotations

import importlib
import pathlib
from typing import ClassVar

import pytest
from egglog import *

EXAMPLE_FILES = list((pathlib.Path(__file__).parent / "../egglog/examples").glob("*.py"))


# Test all files in the `examples` directory by importing them in this parametrized test
@pytest.mark.parametrize("name", [f.stem for f in EXAMPLE_FILES if f.stem != "__init__"])
def test_example(name):
    importlib.import_module(f"egglog.examples.{name}")


class TestExprStr:
    def test_unwrap_lit(self):
        assert str(i64(1) + 1) == "i64(1) + 1"
        assert str(i64(1).max(2)) == "i64(1).max(2)"


def test_eqsat_basic():
    egraph = EGraph()

    @egraph.class_
    class Math(BaseExpr):
        def __init__(self, value: i64Like) -> None:
            ...

        @classmethod
        def var(cls, v: StringLike) -> Math:  # type: ignore[empty-body]
            ...

        def __add__(self, other: Math) -> Math:  # type: ignore[empty-body]
            ...

        def __mul__(self, other: Math) -> Math:  # type: ignore[empty-body]
            ...

    # expr1 = 2 * (x + 3)
    expr1 = egraph.define("expr1", Math(2) * (Math.var("x") + Math(3)))

    # expr2 = 6 + 2 * x
    expr2 = egraph.define("expr2", Math(6) + Math(2) * Math.var("x"))

    a, b, c = vars_("a b c", Math)
    x, y = vars_("x y", i64)

    egraph.register(
        rewrite(a + b).to(b + a),
        rewrite(a * (b + c)).to((a * b) + (a * c)),
        rewrite(Math(x) + Math(y)).to(Math(x + y)),
        rewrite(Math(x) * Math(y)).to(Math(x * y)),
    )

    egraph.run(10)

    egraph.check(eq(expr1).to(expr2))


def test_fib():
    egraph = EGraph()

    @egraph.function
    def fib(x: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    f0, f1, x = vars_("f0 f1 x", i64)
    egraph.register(
        set_(fib(0)).to(i64(1)),
        set_(fib(1)).to(i64(1)),
        rule(
            eq(f0).to(fib(x)),
            eq(f1).to(fib(x + 1)),
        ).then(set_(fib(x + 2)).to(f0 + f1)),
    )
    egraph.run(7)
    egraph.check(eq(fib(i64(7))).to(i64(21)))


@pytest.mark.xfail
def test_fib_demand():
    egraph = EGraph()

    @egraph.class_
    class Num(BaseExpr):
        def __init__(self, i: i64Like) -> None:
            ...

        def __add__(self, other: Num) -> Num:  # type: ignore[empty-body]
            ...

    @egraph.function
    def fib(x: i64Like) -> Num:  # type: ignore[empty-body]
        ...

    a, b, x = vars_("a b x", i64)
    f = var("f", Num)
    egraph.register(
        rewrite(Num(a) + Num(b)).to(Num(a + b)),
        rule(eq(f).to(fib(x)), x > 1).then(set_(fib(x)).to(fib(x - 1) + fib(x - 2))),
        set_(fib(0)).to(Num(0)),
        set_(fib(1)).to(Num(1)),
    )
    f7 = egraph.define("f7", fib(7))
    egraph.run(14)
    egraph.check(eq(f7).to(Num(13)))
    res = egraph.extract(f7)
    assert expr_parts(res) == expr_parts(Num(13))


def test_push_pop():
    egraph = EGraph()

    @egraph.function(merge=lambda old, new: old.max(new))
    def foo() -> i64:  # type: ignore[empty-body]
        ...

    egraph.register(set_(foo()).to(i64(1)))
    egraph.check(eq(foo()).to(i64(1)))

    with egraph:
        egraph.register(set_(foo()).to(i64(2)))
        egraph.check(eq(foo()).to(i64(2)))

    egraph.check(eq(foo()).to(i64(1)))


def test_constants():
    egraph = EGraph()

    one = egraph.constant("one", i64)
    egraph.register(set_(one).to(i64(1)))
    egraph.check(eq(one).to(i64(1)))


def test_class_vars():
    egraph = EGraph()

    @egraph.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[i64]

    egraph.register(set_(Numeric.ONE).to(i64(1)))
    egraph.check(eq(Numeric.ONE).to(i64(1)))


def test_simplify_constant():
    egraph = EGraph()

    @egraph.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[Numeric]

        def __init__(self, v: i64) -> None:
            pass

    assert expr_parts(egraph.simplify(Numeric.ONE, 10)) == expr_parts(Numeric.ONE)

    egraph.register(union(Numeric.ONE).with_(Numeric(i64(1))))
    egraph.run(10)
    egraph.check(eq(Numeric.ONE).to(Numeric(i64(1))))


def test_extract_constant_twice():
    # Sometimes extrcting a constant twice will give an error
    egraph = EGraph()

    @egraph.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[Numeric]

    egraph.extract(Numeric.ONE)
    egraph.extract(Numeric.ONE)


def test_relation():
    egraph = EGraph()

    test_relation = egraph.relation("test_relation", i64, i64)
    egraph.register(test_relation(i64(1), i64(1)))


def test_variable_args():
    egraph = EGraph()
    # Create dummy function with type so its registered
    egraph.relation("_", Set[i64])

    egraph.check(Set(i64(1), i64(2)).contains(i64(1)))


@pytest.mark.xfail(reason="We have to manually register sorts before using them")
def test_generic_sort():
    egraph = EGraph()
    egraph.check(Set(i64(1), i64(2)).contains(i64(1)))


def test_keyword_args():
    egraph = EGraph()

    @egraph.function
    def foo(x: i64Like, y: i64Like) -> i64:  # type: ignore[empty-body]
        ...

    pos = expr_parts(foo(i64(1), i64(2)))
    assert expr_parts(foo(i64(1), y=i64(2))) == pos
    assert expr_parts(foo(y=i64(2), x=i64(1))) == pos


def test_modules() -> None:
    m = Module()

    @m.class_
    class Numeric(BaseExpr):
        ONE: ClassVar[Numeric]

    m2 = Module()

    @m2.class_
    class OtherNumeric(BaseExpr):
        @m2.method(cost=10)
        def __init__(self, v: i64Like) -> None:
            ...

    egraph = EGraph(deps=[m, m2])

    @egraph.function
    def from_numeric(n: Numeric) -> OtherNumeric:  # type: ignore[empty-body]
        ...

    egraph.register(rewrite(OtherNumeric(1)).to(from_numeric(Numeric.ONE)))
    assert expr_parts(egraph.simplify(OtherNumeric(i64(1)), 10)) == expr_parts(from_numeric(Numeric.ONE))


class TestPyObject:
    def test_from_string(self):
        egraph = EGraph()
        res = egraph.simplify(PyObject.from_string("foo"), 1)
        assert egraph.load_object(res) == "foo"

    def test_to_string(self):
        egraph = EGraph()
        s = egraph.save_object("foo")
        res = egraph.simplify(s.to_string(), 1)
        assert expr_parts(res) == expr_parts(String("foo"))

    def test_dict_update(self):
        egraphs = EGraph()
        original_d = {"foo": "bar"}
        obj = egraphs.save_object(original_d)
        res = obj.dict_update(PyObject.from_string("foo"), PyObject.from_string("baz"))
        simplified = egraphs.simplify(res, 1)
        assert egraphs.load_object(simplified) == {"foo": "baz"}
        assert original_d == {"foo": "bar"}

    def test_eval(self):
        egraph = EGraph()
        x, y = 10, 20
        res = py_eval("x + y", egraph.save_object({"x": x, "y": y}), egraph.save_object({}))
        res_simpl = egraph.simplify(res, 1)
        assert egraph.load_object(res_simpl) == 30

    def test_eval_local(self):
        egraph = EGraph()
        x = "hi"
        res = py_eval(
            "my_add(x, y)",
            egraph.save_object(locals()).dict_update(PyObject.from_string("y"), PyObject.from_string("there")),
            egraph.save_object(globals()),
        )
        res_simpl = egraph.simplify(res, 1)
        assert egraph.load_object(res_simpl) == "hithere"


def my_add(a, b):
    return a + b
