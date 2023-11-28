# mypy: disable-error-code="empty-body"
from __future__ import annotations

import importlib
from multiprocessing import Value
import pathlib
from copy import copy
from typing import ClassVar, Union

import pytest
from egglog import *
from egglog.declarations import (
    CallDecl,
    FunctionRef,
    JustTypeRef,
    MethodRef,
    TypedExprDecl,
)

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
    class Math(Expr):
        def __init__(self, value: i64Like) -> None:
            ...

        @classmethod
        def var(cls, v: StringLike) -> Math:
            ...

        def __add__(self, other: Math) -> Math:
            ...

        def __mul__(self, other: Math) -> Math:
            ...

    # expr1 = 2 * (x + 3)
    expr1 = egraph.let("expr1", Math(2) * (Math.var("x") + Math(3)))

    # expr2 = 6 + 2 * x
    expr2 = egraph.let("expr2", Math(6) + Math(2) * Math.var("x"))

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
    def fib(x: i64Like) -> i64:
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


def test_fib_demand():
    egraph = EGraph()

    @egraph.class_
    class Num(Expr):
        def __init__(self, i: i64Like) -> None:
            ...

        def __add__(self, other: Num) -> Num:
            ...

    @egraph.function(cost=20)
    def fib(x: i64Like) -> Num:
        ...

    @egraph.register
    def _fib(a: i64, b: i64):
        yield rewrite(
            Num(a) + Num(b)
        ).to(
            Num(a + b)
        )
        yield rewrite(
            fib(a)
        ).to(
            fib(a - 1) + fib(a - 2),
            a > 1
        )
        yield rewrite(
            fib(a)
        ).to(
            Num(a),
            a <= 1
        )

    f7 = egraph.let("f7", fib(7))
    egraph.run(14)
    egraph.check(eq(f7).to(Num(13)))
    res = egraph.extract(f7)
    assert expr_parts(res) == expr_parts(Num(13))


def test_push_pop():
    egraph = EGraph()

    @egraph.function(merge=lambda old, new: old.max(new))
    def foo() -> i64:
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
    class Numeric(Expr):
        ONE: ClassVar[i64]

    egraph.register(set_(Numeric.ONE).to(i64(1)))
    egraph.check(eq(Numeric.ONE).to(i64(1)))


def test_simplify_constant():
    egraph = EGraph()

    @egraph.class_
    class Numeric(Expr):
        ONE: ClassVar[Numeric]

        def __init__(self, v: i64) -> None:
            pass

    assert expr_parts(egraph.simplify(Numeric.ONE, 10)) == expr_parts(Numeric.ONE)

    egraph.register(union(Numeric.ONE).with_(Numeric(i64(1))))
    egraph.run(10)
    egraph.check(eq(Numeric.ONE).to(Numeric(i64(1))))


def test_extract_constant_twice():
    # Sometimes extracting a constant twice will give an error
    egraph = EGraph()

    @egraph.class_
    class Numeric(Expr):
        ONE: ClassVar[Numeric]

    egraph.extract(Numeric.ONE)
    egraph.extract(Numeric.ONE)

def test_extract_include_cost():
    _, cost = EGraph().extract(i64(0), include_cost=True)
    assert cost == 1

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
    def foo(x: i64Like, y: i64Like) -> i64:
        ...

    pos = expr_parts(foo(i64(1), i64(2)))
    assert expr_parts(foo(i64(1), y=i64(2))) == pos
    assert expr_parts(foo(y=i64(2), x=i64(1))) == pos


def test_modules() -> None:
    m = Module()

    @m.class_
    class Numeric(Expr):
        ONE: ClassVar[Numeric]

    m2 = Module()

    @m2.class_
    class OtherNumeric(Expr):
        @m2.method(cost=10)
        def __init__(self, v: i64Like) -> None:
            ...

    egraph = EGraph([m, m2])

    @egraph.function
    def from_numeric(n: Numeric) -> OtherNumeric:
        ...

    egraph.register(rewrite(OtherNumeric(1)).to(from_numeric(Numeric.ONE)))
    assert expr_parts(egraph.simplify(OtherNumeric(i64(1)), 10)) == expr_parts(from_numeric(Numeric.ONE))


def test_property():
    egraph = EGraph()

    @egraph.class_
    class Foo(Expr):
        def __init__(self) -> None:
            ...

        @property
        def bar(self) -> i64:
            ...

    egraph.register(set_(Foo().bar).to(i64(1)))
    egraph.check(eq(Foo().bar).to(i64(1)))


def test_default_args():
    egraph = EGraph()

    @egraph.function
    def foo(x: i64Like, y: i64Like = i64(1)) -> i64:
        ...

    assert expr_parts(foo(i64(1))) == expr_parts(foo(i64(1), i64(1)))

    assert str(foo(i64(1), i64(2))) == "foo(1, 2)"
    assert str(foo(i64(1), i64(1))) == "foo(1)"


class TestPyObject:
    def test_from_string(self):
        assert EGraph().eval(PyObject.from_string("foo")) == "foo"

    def test_to_string(self):
        assert EGraph().eval(PyObject("foo").to_string()) == "foo"

    def test_dict_update(self):
        original_d = {"foo": "bar"}
        res = EGraph().eval(PyObject(original_d).dict_update("foo", "baz"))
        assert res == {"foo": "baz"}
        assert original_d == {"foo": "bar"}

    def test_eval(self):
        assert EGraph().eval(py_eval("x + y", {"x": 10, "y": 20}, {})) == 30

    def test_eval_local(self):
        x = "hi"
        res = py_eval(
            "my_add(x, y)",
            PyObject(locals()).dict_update("y", "there"),
            globals()
        )
        assert EGraph().eval(res) == "hithere"

    def test_exec(self):
        assert EGraph().eval(py_exec("x = 10")) == {"x": 10}

    def test_exec_globals(self):
        assert EGraph().eval(py_exec("x = y + 1", {"y": 10})) == {"x": 11}


def my_add(a, b):
    return a + b


def test_convert_int_float():
    egraph = EGraph()
    egraph.check(eq(i64(1)).to(f64(1.0).to_i64()))
    egraph.check(eq(f64(1.0)).to(f64.from_i64(i64(1))))


def test_f64_negation() -> None:
    egraph = EGraph()
    # expr1 = -2.0
    expr1 = egraph.let("expr1", -f64(2.0))

    # expr2 = 2.0
    expr2 = egraph.let("expr2", f64(2.0))

    # expr3 = -(-2.0)
    expr3 = egraph.let("expr3", -(-f64(2.0)))
    egraph.check(eq(expr1).to(-expr2))
    egraph.check(eq(expr3).to(expr2))


def test_not_equals():
    egraph = EGraph()
    egraph.check(i64(10) != i64(2))


class TestMutate:
    def test_setitem_defaults(self):
        egraph = EGraph()

        @egraph.class_
        class Foo(Expr):
            def __init__(self) -> None:
                ...

            def __setitem__(self, key: i64Like, value: i64Like) -> None:
                ...

        foo = Foo()
        foo[10] = 20
        assert str(foo) == "_Foo_1 = Foo()\n_Foo_1[10] = 20\n_Foo_1"
        assert expr_parts(foo) == TypedExprDecl(
            JustTypeRef("Foo"),
            CallDecl(MethodRef("Foo", "__setitem__"), (expr_parts(Foo()), expr_parts(i64(10)), expr_parts(i64(20)))),
        )

    def test_function(self):
        egraph = EGraph()

        @egraph.class_
        class Math(Expr):
            def __init__(self, i: i64Like) -> None:
                ...

            def __add__(self, other: Math) -> Math:
                ...

        @egraph.function(mutates_first_arg=True)
        def incr(x: Math) -> None:
            ...

        x = Math(i64(10))
        x_copied = copy(x)
        incr(x)
        assert expr_parts(x_copied) == expr_parts(Math(i64(10)))
        assert expr_parts(x) == TypedExprDecl(
            JustTypeRef("Math"),
            CallDecl(FunctionRef("incr"), (expr_parts(x_copied),)),
        )
        assert str(x) == "_Math_1 = Math(10)\nincr(_Math_1)\n_Math_1"
        assert str(x + Math(10)) == "_Math_1 = Math(10)\nincr(_Math_1)\n_Math_1 + Math(10)"

        i, j = vars_("i j", Math)
        incr_i = copy(i)
        incr(incr_i)
        egraph.register(rewrite(incr_i).to(i + Math(1)), x)
        egraph.run(10)
        egraph.check(eq(x).to(Math(10) + Math(1)))

        x_incr_copy = copy(x)
        incr(x)
        assert (
            str(x_incr_copy + x)
            == "_Math_1 = Math(10)\nincr(_Math_1)\n_Math_2 = copy(_Math_1)\nincr(_Math_2)\n_Math_1 + _Math_2"
        ), "only copy when re-used later"


def test_builtin_reflected():
    assert expr_parts(5 + i64(10)) == expr_parts(i64(5) + i64(10))


def test_reflected_binary_method():
    # If we have a reflected binary method, it should be converted into the non-reflected version
    egraph = EGraph()

    @egraph.class_
    class Math(Expr):
        def __init__(self, value: i64Like) -> None:
            ...

        def __add__(self, other: Math) -> Math:
            ...

        def __radd__(self, other: Math) -> Math:
            ...

    converter(i64, Math, Math)

    expr = 10 + Math(5)  # type: ignore[operator]
    assert str(expr) == "Math(10) + Math(5)"
    assert expr_parts(expr) == TypedExprDecl(
        JustTypeRef("Math"),
        CallDecl(MethodRef("Math", "__add__"), (expr_parts(Math(i64(10))), expr_parts(Math(i64(5))))),
    )


def test_upcast_args():
    # -0.1 + Int(x) -> -0.1 + Float(x)
    egraph = EGraph()

    @egraph.class_
    class Int(Expr):
        def __init__(self, value: i64Like) -> None:
            ...

        def __add__(self, other: Int) -> Int:
            ...

    @egraph.class_
    class Float(Expr):
        def __init__(self, value: f64Like) -> None:
            ...

        def __add__(self, other: Float) -> Float:
            ...

        @classmethod
        def from_int(cls, other: Int) -> Float:
            ...

    converter(i64, Int, Int)
    converter(f64, Float, Float)
    converter(Int, Float, Float.from_int)

    res: Expr = -0.1 + Int(10)  # type: ignore
    assert expr_parts(res) == expr_parts(Float(-0.1) + Float.from_int(Int(10)))

    res: Expr = Int(10) + -0.1  # type: ignore
    assert expr_parts(res) == expr_parts(Float.from_int(Int(10)) + Float(-0.1))

def test_rewrite_upcasts():
    rewrite(i64(1)).to(0) # type: ignore


def test_upcast_self_lower_cost():
    # Verifies that self will be upcasted, if that upcast has a lower cast than converting the other arg
    # i.e. Int(x) + NDArray(y) -> NDArray(Int(x)) + NDArray(y) instead of Int(x) + NDArray(y).to_int()
    egraph = EGraph()

    @egraph.class_
    class Int(Expr):
        def __init__(self, name: StringLike) -> None:
            ...

        def __add__(self, other: Int) -> Int:
            ...

    NDArrayLike = Union[Int, "NDArray"]

    @egraph.class_
    class NDArray(Expr):
        def __init__(self, name: StringLike) -> None:
            ...

        def __add__(self, other: NDArrayLike) -> NDArray:
            ...

        def __radd__(self, other: NDArrayLike) -> NDArray:
            ...

        def to_int(self) -> Int:
            ...

        @classmethod
        def from_int(cls, other: Int) -> NDArray:
            ...

    converter(Int, NDArray, NDArray.from_int)
    converter(NDArray, Int, lambda a: a.to_int(), 100)

    r = Int("x") + NDArray("y")
    assert expr_parts(r) == expr_parts(NDArray.from_int(Int("x")) + NDArray("y"))


def test_eval():
    egraph = EGraph()
    assert egraph.eval(String("hi")) == "hi"
    assert egraph.eval(i64(10)) == 10
    assert egraph.eval(f64(10.0)) == 10.0
    assert egraph.eval(Bool(True)) is True
    assert egraph.eval(PyObject((1, 2))) == (1, 2)


def test_egglog_string():
    egraph = EGraph(save_egglog_string=True)
    egraph.register((i64(1)))
    assert egraph.as_egglog_string

def test_no_egglog_string():
    egraph = EGraph()
    egraph.register((i64(1)))
    with pytest.raises(ValueError):
        egraph.as_egglog_string



def test_eval_fn():
    egraph = EGraph()

    assert egraph.eval(py_eval_fn(lambda x: (x,))(PyObject.from_int(1))) == (1,)


def _global_make_tuple(x):
    return (x,)

def test_eval_fn_globals():
    egraph = EGraph()

    assert egraph.eval(py_eval_fn(lambda x: _global_make_tuple(x))(PyObject.from_int(1))) == (1,)

def test_eval_fn_locals():
    egraph = EGraph()


    def _locals_make_tuple(x):
        return (x,)

    assert egraph.eval(py_eval_fn(lambda x: _locals_make_tuple(x))(PyObject.from_int(1))) == (1,)
