# mypy: disable-error-code="empty-body"
from __future__ import annotations

import importlib
import pathlib
from copy import copy
from fractions import Fraction
from functools import partial
from typing import ClassVar, TypeAlias, TypeVar
from unittest.mock import MagicMock

import pytest

from egglog import *
from egglog.declarations import (
    CallDecl,
    FunctionRef,
    JustTypeRef,
    MethodRef,
    TypedExprDecl,
)
from egglog.version_compat import BEFORE_3_11


class TestExprStr:
    def test_unwrap_lit(self):
        assert str(i64(1) + 1) == "i64(1) + 1"
        assert str(i64(1).max(2)) == "i64(1).max(2)"

    def test_ne(self):
        assert str(ne(i64(1)).to(i64(2))) == "ne(i64(1)).to(i64(2))"


def test_eqsat_basic():
    egraph = EGraph()

    class Math(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @classmethod
        def var(cls, v: StringLike) -> Math: ...

        def __add__(self, other: Math) -> Math: ...

        def __mul__(self, other: Math) -> Math: ...

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

    @function
    def fib(x: i64Like) -> i64: ...

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

    class Num(Expr):
        def __init__(self, i: i64Like) -> None: ...

        def __add__(self, other: Num) -> Num: ...

    @function(cost=20)
    def fib(x: i64Like) -> Num: ...

    @egraph.register
    def _fib(a: i64, b: i64):
        yield rewrite(Num(a) + Num(b)).to(Num(a + b))
        yield rewrite(fib(a)).to(fib(a - 1) + fib(a - 2), a > 1)
        yield rewrite(fib(a)).to(Num(a), a <= 1)

    f7 = egraph.let("f7", fib(7))
    egraph.run(14)
    egraph.check(eq(f7).to(Num(13)))
    res = egraph.extract(f7)
    assert expr_parts(res) == expr_parts(Num(13))


def test_push_pop():
    egraph = EGraph()

    @function(merge=lambda old, new: old.max(new))
    def foo() -> i64: ...

    egraph.register(set_(foo()).to(i64(1)))
    egraph.check(eq(foo()).to(i64(1)))

    with egraph:
        egraph.register(set_(foo()).to(i64(2)))
        egraph.check(eq(foo()).to(i64(2)))

    egraph.check(eq(foo()).to(i64(1)))


def test_constants():
    egraph = EGraph()

    class A(Expr):
        pass

    one = constant("one", A)
    two = constant("two", A)

    egraph.register(union(one).with_(two))
    egraph.check(eq(one).to(two))


def test_class_vars():
    egraph = EGraph()

    class B(Expr):
        ONE: ClassVar[B]

    two = constant("two", B)

    egraph.register(union(B.ONE).with_(two))
    egraph.check(eq(B.ONE).to(two))


def test_extract_constant_twice():
    # Sometimes extracting a constant twice will give an error
    egraph = EGraph()

    class Numeric(Expr):
        ONE: ClassVar[Numeric]

    egraph.extract(Numeric.ONE)
    egraph.extract(Numeric.ONE)


def test_extract_include_cost():
    _, cost = EGraph().extract(i64(0), include_cost=True)
    assert cost == 1


def test_relation():
    egraph = EGraph()

    test_relation = relation("test_relation", i64, i64)
    egraph.register(test_relation(i64(1), i64(1)))


def test_variable_args():
    egraph = EGraph()
    egraph.check(Set(i64(1), i64(2)).contains(i64(1)))


def test_generic_sort():
    egraph = EGraph()
    egraph.check(Set(i64(1), i64(2)).contains(i64(1)))


def test_keyword_args():
    EGraph()

    @function
    def foo(x: i64Like, y: i64Like) -> i64: ...

    pos = expr_parts(foo(i64(1), i64(2)))
    assert expr_parts(foo(i64(1), y=i64(2))) == pos
    assert expr_parts(foo(y=i64(2), x=i64(1))) == pos


def test_keyword_args_init():
    EGraph()

    class Foo(Expr):
        def __init__(self, x: i64Like) -> None: ...

    assert expr_parts(Foo(1)) == expr_parts(Foo(x=1))


def test_property():
    egraph = EGraph()

    class Foo(Expr):
        def __init__(self) -> None: ...

        @property
        def bar(self) -> i64: ...

    egraph.register(set_(Foo().bar).to(i64(1)))
    egraph.check(eq(Foo().bar).to(i64(1)))


def test_default_args():
    EGraph()

    @function
    def foo(x: i64Like, y: i64Like = i64(1)) -> i64: ...

    assert expr_parts(foo(i64(1))) == expr_parts(foo(i64(1), i64(1)))

    assert str(foo(i64(1), i64(2))) == "foo(1, 2)"
    assert str(foo(i64(1), i64(1))) == "foo(1)"


class TestPyObject:
    def test_from_string(self):
        assert EGraph().extract(PyObject.from_string("foo")).value == "foo"

    def test_to_string(self):
        EGraph().check(PyObject("foo").to_string() == String("foo"))

    def test_dict_update(self):
        original_d = {"foo": "bar"}
        res = EGraph().extract(PyObject(original_d).dict_update("foo", "baz")).value
        assert res == {"foo": "baz"}
        assert original_d == {"foo": "bar"}

    def test_eval(self):
        assert EGraph().extract(py_eval("x + y", {"x": 10, "y": 20}, {})).value == 30

    def test_eval_local(self):
        x = "hi"
        res = py_eval("my_add(x, y)", PyObject(locals()).dict_update("y", "there"), globals())
        assert EGraph().extract(res).value == "hithere"

    def test_exec(self):
        assert EGraph().extract(py_exec("x = 10")).value == {"x": 10}

    def test_exec_globals(self):
        assert EGraph().extract(py_exec("x = y + 1", {"y": 10})).value == {"x": 11}


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
    expr3 = egraph.let("expr3", -(-f64(2.0)))  # noqa: B002
    egraph.check(eq(expr1).to(-expr2))
    egraph.check(eq(expr3).to(expr2))


def test_not_equals():
    egraph = EGraph()
    egraph.check(ne(i64(10)).to(i64(2)))


def test_custom_equality():
    egraph = EGraph()

    class Boolean(Expr):
        def __init__(self, value: BoolLike) -> None: ...

        def __eq__(self, other: Boolean) -> Boolean:  # type: ignore[override]
            ...

        def __ne__(self, other: Boolean) -> Boolean:  # type: ignore[override]
            ...

    egraph.register(rewrite(Boolean(True) == Boolean(True)).to(Boolean(False)))
    egraph.register(rewrite(Boolean(True) != Boolean(True)).to(Boolean(True)))

    should_be_true = Boolean(True) == Boolean(True)
    should_be_false = Boolean(True) != Boolean(True)
    egraph.register(should_be_true, should_be_false)
    egraph.run(10)
    egraph.check(eq(should_be_true).to(Boolean(False)))
    egraph.check(eq(should_be_false).to(Boolean(True)))


class TestMutate:
    def test_setitem_defaults(self):
        EGraph()

        class Foo(Expr):
            def __init__(self) -> None: ...
            def __setitem__(self, key: i64Like, value: i64Like) -> None: ...

        foo = Foo()
        foo[10] = 20
        assert str(foo) == "_Foo_1 = Foo()\n_Foo_1[10] = 20\n_Foo_1"
        assert expr_parts(foo) == TypedExprDecl(
            JustTypeRef("Foo"),
            CallDecl(MethodRef("Foo", "__setitem__"), (expr_parts(Foo()), expr_parts(i64(10)), expr_parts(i64(20)))),
        )

    def test_function(self):
        egraph = EGraph()

        class Math(Expr):
            def __init__(self, i: i64Like) -> None: ...

            def __add__(self, other: Math) -> Math: ...

        @function(mutates_first_arg=True)
        def incr(x: Math) -> None: ...

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

        i, _j = vars_("i j", Math)
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
    EGraph()

    class Math(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def __add__(self, other: Math) -> Math: ...

        def __radd__(self, other: Math) -> Math: ...

    converter(i64, Math, Math)

    expr = 10 + Math(5)  # type: ignore[operator]
    assert str(expr) == "Math(10) + Math(5)"
    assert expr_parts(expr) == TypedExprDecl(
        JustTypeRef("Math"),
        CallDecl(MethodRef("Math", "__add__"), (expr_parts(Math(i64(10))), expr_parts(Math(i64(5))))),
    )


def test_rewrite_upcasts():
    class X(Expr):
        def __init__(self, value: i64Like) -> None: ...

    converter(i64, X, X)
    rewrite(X(1)).to(0)  # type: ignore[arg-type]


def test_function_default_upcasts():
    @function
    def f(x: i64Like) -> i64: ...

    assert expr_parts(f(1)) == expr_parts(f(i64(1)))


def test_upcast_self_lower_cost():
    # Verifies that self will be upcasted, if that upcast has a lower cast than converting the other arg
    # i.e. Int(x) + NDArray(y) -> NDArray(Int(x)) + NDArray(y) instead of Int(x) + NDArray(y).to_int()

    class Int(Expr):
        def __init__(self, name: StringLike) -> None: ...

        def __add__(self, other: Int) -> Int: ...

    class NDArray(Expr):
        def __init__(self, name: StringLike) -> None: ...

        def __add__(self, other: NDArrayLike) -> NDArray: ...

        def __radd__(self, other: NDArrayLike) -> NDArray: ...

        def to_int(self) -> Int: ...

        @classmethod
        def from_int(cls, other: Int) -> NDArray: ...

    NDArrayLike: TypeAlias = NDArray | Int

    converter(Int, NDArray, NDArray.from_int)
    converter(NDArray, Int, lambda a: a.to_int(), 100)

    r = Int("x") + NDArray("y")
    assert expr_parts(r) == expr_parts(NDArray.from_int(Int("x")) + NDArray("y"))


class TestEval:
    def test_string(self):
        assert String("hi").value == "hi"

    def test_bool(self):
        assert Bool(True).value is True
        assert bool(Bool(True)) is True

    def test_i64(self):
        assert i64(10).value == 10
        assert int(i64(10)) == 10
        assert [10][i64(0)] == 10

    def test_f64(self):
        assert f64(10.0).value == 10.0
        assert int(f64(10.0)) == 10
        assert float(f64(10.0)) == 10.0

    def test_map(self):
        assert Map[String, i64].empty().value == {}
        m = Map[String, i64].empty().insert(String("a"), i64(1)).insert(String("b"), i64(2))
        assert m.value == {String("a"): i64(1), String("b"): i64(2)}

        assert set(m) == {String("a"), String("b")}
        assert len(m) == 2
        assert String("a") in m
        assert String("c") not in m

    def test_set(self):
        assert EGraph().extract(Set[i64].empty()).value == set()
        s = Set(i64(1), i64(2))
        assert s.value == {i64(1), i64(2)}

        assert set(s) == {i64(1), i64(2)}
        assert len(s) == 2
        assert i64(1) in s
        assert i64(3) not in s

    def test_rational(self):
        assert Rational(1, 2).value == Fraction(1, 2)
        assert float(Rational(1, 2)) == 0.5
        assert int(Rational(1, 1)) == 1

    def test_vec(self):
        assert Vec[i64].empty().value == ()
        s = Vec(i64(1), i64(2))
        assert s.value == (i64(1), i64(2))

        assert list(s) == [i64(1), i64(2)]
        assert len(s) == 2
        assert i64(1) in s
        assert i64(3) not in s

    def test_py_object(self):
        assert PyObject(10).value == 10
        o = object()
        assert PyObject(o).value is o

    def test_big_int(self):
        assert int(EGraph().extract(BigInt(10))) == 10

    def test_big_rat(self):
        br = EGraph().extract(BigRat(1, 2))
        assert float(br) == 1 / 2
        assert br.value == Fraction(1, 2)

    def test_multiset(self):
        assert list(MultiSet(i64(1), i64(1))) == [i64(1), i64(1)]

    def test_unstable_fn(self):
        class Math(Expr):
            def __init__(self) -> None: ...

        @function
        def f(x: Math) -> Math: ...

        u_f = UnstableFn(f)
        assert u_f.value == f
        p_u_f = UnstableFn(f, Math())
        value = p_u_f.value
        assert isinstance(value, partial)
        assert value.func == f
        assert value.args == (Math(),)


# def test_egglog_string():
#     egraph = EGraph(save_egglog_string=True)
#     egraph.register((i64(1)))
#     assert egraph.as_egglog_string

# def test_no_egglog_string():
#     egraph = EGraph()
#     egraph.register((i64(1)))
#     with pytest.raises(ValueError):
#         egraph.as_egglog_string


def test_eval_fn():
    assert EGraph().extract(py_eval_fn(lambda x: (x,))(PyObject.from_int(1))).value == (1,)


def _global_make_tuple(x):
    return (x,)


def test_eval_fn_globals():
    assert EGraph().extract(py_eval_fn(lambda x: _global_make_tuple(x))(PyObject.from_int(1))).value == (1,)


def test_eval_fn_locals():
    def _locals_make_tuple(x):
        return (x,)

    assert EGraph().extract(py_eval_fn(lambda x: _locals_make_tuple(x))(PyObject.from_int(1))).value == (1,)


def test_lazy_types():
    class A(Expr):
        def __init__(self) -> None: ...

        def b(self) -> B: ...

    class B(Expr): ...

    EGraph().register(A().b())


# https://github.com/egraphs-good/egglog-python/issues/100
def test_functions_seperate_pop():
    egraph = EGraph()

    class T(Expr):
        def __init__(self, x: i64Like) -> None: ...

    with egraph:

        @function
        def f(x: T) -> T: ...

        egraph.register(f(T(1)))

    with egraph:

        @function
        def f(x: T, y: T) -> T: ...  # type: ignore[misc]

        egraph.register(f(T(1), T(2)))  # type: ignore[call-arg]


# https://github.com/egraphs-good/egglog/issues/113
def test_multiple_generics():
    @function
    def f() -> Vec[i64]: ...

    @function
    def g() -> Vec[String]: ...

    egraph = EGraph()

    egraph.register(
        set_(f()).to(Vec[i64]()),
        set_(g()).to(Vec[String]()),
    )

    assert str(egraph.extract(f())) == "Vec[i64].empty()"
    assert str(egraph.extract(g())) == "Vec[String].empty()"


def test_deferred_ruleset():
    @ruleset
    def rules(x: AA):
        yield rewrite(first(x)).to(second(x))

    class AA(Expr):
        def __init__(self) -> None: ...

    @function
    def first(x: AA) -> AA: ...

    @function
    def second(x: AA) -> AA: ...

    check(
        eq(first(AA())).to(second(AA())),
        rules,
        first(AA()),
    )


def test_access_method_on_class():
    class A(Expr):
        def __init__(self) -> None: ...

        def b(self, x: i64Like) -> A: ...

    assert expr_parts(A.b(A(), 1)) == expr_parts(A().b(1))


def test_access_property_on_class():
    class A(Expr):
        def __init__(self) -> None: ...

        @property
        def b(self) -> i64: ...

    assert expr_parts(A.b(A())) == expr_parts(A().b)


class A(Expr):
    def __init__(self) -> None: ...


class TestDefaultReplacements:
    def test_function(self):
        @function
        def f() -> A:
            return A()

        check_eq(f(), A(), run())

    def test_function_ruleset(self):
        r = ruleset()

        @function(ruleset=r)
        def f() -> A:
            return A()

        check_eq(f(), A(), r)

    def test_constant(self):
        a = constant("a", A, A())
        check_eq(a, A(), run())

    def test_constant_ruleset(self):
        r = ruleset()
        a = constant("a", A, A(), ruleset=r)

        check_eq(a, A(), r)

    def test_method(self):
        class B(Expr):
            def __init__(self) -> None: ...
            def f(self) -> A:
                return A()

        check_eq(B().f(), A(), run())

    def test_method_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            def __init__(self) -> None: ...
            def f(self) -> A:
                return A()

        check_eq(B().f(), A(), r)

    def test_classmethod(self):
        class B(Expr):
            @classmethod
            def f(cls) -> A:
                return A()

        check_eq(B.f(), A(), run())

    def test_classmethod_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            @classmethod
            def f(cls) -> A:
                return A()

        check_eq(B.f(), A(), r)

    def test_classvar(self):
        class B(Expr):
            a: ClassVar[A] = A()

        check_eq(B.a, A(), run())

    def test_classvar_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            a: ClassVar[A] = A()

        check_eq(B.a, A(), r)

    def test_method_refer_to_later(self):
        """
        Verify that an earlier method body can refer to values defined in later ones
        """

        class B(Expr):
            def __init__(self) -> None: ...
            def f(self) -> A:
                return self.g()

            def g(self) -> A: ...

        B()
        left = B().f()
        right = B().g()
        check_eq(left, right, run())

    def test_classmethod_own_class(self):
        class B(Expr):
            def __init__(self) -> None: ...
            @classmethod
            def f(cls) -> B:
                return B()

        check_eq(B.f(), B(), run())


class TestIssue166:
    """
    Raised by @cgyurgyik in https://github.com/egraphs-good/egglog-python/issues/166
    """

    def test_inserting_map(self):
        egraph = EGraph()
        m = egraph.let("map", Map[String, i64].empty().insert(String("a"), i64(42)))
        egraph.run(5)
        egraph.extract(m)

    def test_creating_map(self):
        m = Map[String, i64].empty()
        egraph = EGraph()
        egraph.register(m)
        egraph.extract(m)


def test_helpful_error_function_class():
    class E(Expr):
        @function(cost=10)
        def __init__(self) -> None: ...

    match = "Inside of classes, wrap methods with the `method` decorator, not `function`"
    # If we are after 3 11 we have context included
    if not BEFORE_3_11:
        match += "\nError processing E.__init__"
    with pytest.raises(ValueError, match=match):
        E()


def test_vec_like_conversion():
    """
    Test that we can use a generic type alias for conversion
    """

    @function
    def my_fn(xs: VecLike[i64, i64Like]) -> Unit: ...

    assert expr_parts(my_fn((1, 2))) == expr_parts(my_fn(Vec[i64](i64(1), i64(2))))
    assert expr_parts(my_fn([])) == expr_parts(my_fn(Vec[i64]()))


def test_set_like_conversion():
    @function
    def my_fn(xs: SetLike[i64, i64Like]) -> Unit: ...

    assert expr_parts(my_fn({1, 2})) == expr_parts(my_fn(Set[i64](i64(1), i64(2))))
    assert expr_parts(my_fn(set())) == expr_parts(my_fn(Set[i64]()))


def test_map_like_conversion():
    @function
    def my_fn(xs: MapLike[i64, String, i64Like, StringLike]) -> Unit: ...

    assert expr_parts(my_fn({1: "hi"})) == expr_parts(my_fn(Map[i64, String].empty().insert(i64(1), String("hi"))))
    assert expr_parts(my_fn({})) == expr_parts(my_fn(Map[i64, String].empty()))


class TestEqNE:
    def test_eq(self):
        assert i64(3) == i64(3)

    def test_ne(self):
        EGraph().check(i64(3) != i64(4))

    def test_eq_false(self):
        assert not (i64(3) == 4)  # noqa: SIM201


def test_no_upcast_eq():
    """
    Verifies that if two items can be upcast to something, calling == on them won't use
    equality
    """

    class A(Expr):
        def __init__(self) -> None: ...

    class B(Expr):
        def __init__(self) -> None: ...
        def __eq__(self, other: B) -> B: ...  # type: ignore[override]

    converter(A, B, lambda a: B())

    assert isinstance(A() == A(), Fact)
    assert not isinstance(B() == B(), Fact)


def test_isinstance_expr():
    """
    Verifies that isinstance() works on Exprs, and returns a Fact
    """

    class A(Expr):
        def __init__(self) -> None: ...

    class B(Expr):
        def __init__(self) -> None: ...

    assert isinstance(A(), A)
    assert not isinstance(A(), B)


class TestMatch:
    def test_class(self):
        """
        Verify that we can pattern match on expressions
        """

        class A(Expr):
            def __init__(self) -> None: ...

        class B(Expr):
            def __init__(self) -> None: ...

        a = A()
        match a:
            case B():
                msg = "Should not have matched B"
                raise ValueError(msg)
            case A():
                pass
            case _:
                msg = "Should have matched A"
                raise ValueError(msg)

    def test_literal(self):
        match i64(10):
            case i64(i):
                assert i == 10
            case _:
                msg = "Should have matched i64(10)"
                raise ValueError(msg)

    def test_literal_fail(self):
        """
        Verify that matching on a literal that does not match raises an error
        """
        match i64(10) + i64(10):
            case i64(_i):
                msg = "Should not have matched i64(20)"
                raise ValueError(msg)

    def test_custom_args(self):
        class A(Expr):
            def __init__(self) -> None: ...

            __match_args__ = ("a", "b")

            @method(preserve=True)  # type: ignore[prop-decorator]
            @property
            def a(self) -> int:
                return 1

            @method(preserve=True)  # type: ignore[prop-decorator]
            @property
            def b(self) -> str:
                return "hi"

        match A():
            case A(a, b):
                assert a == 1
                assert b == "hi"
            case _:
                msg = "Should have matched A"
                raise ValueError(msg)

    def test_custom_args_fail(self):
        """
        Verify that matching on a custom match that does not match raises an error
        """

        class A(Expr):
            def __init__(self) -> None: ...

            __match_args__ = ("a",)

            @method(preserve=True)  # type: ignore[prop-decorator]
            @property
            def a(self) -> int:
                raise AttributeError

        match A():
            case A(_a):
                msg = "Should not have matched A"
                raise ValueError(msg)


T = TypeVar("T")


def test_type_param_sub():
    """
    Verify that type substituion works properly, by comparing string version.

    Comparing actual versions is always false if they are no the same object for unions
    """
    V = Vec[T] | int
    assert str(V[Unit]) == str(Vec[Unit] | int)  # type: ignore[misc]


def test_override_hash():
    class A(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __hash__(self) -> int:
            return 42

    assert hash(A()) == 42


def test_serialize_warning_max_functions():
    class A(Expr):
        def __init__(self) -> None: ...

    egraph = EGraph()
    egraph.register(A())
    with pytest.warns(UserWarning, match="A"):
        egraph._serialize(max_functions=0)


def test_serialize_warning_max_calls():
    class A(Expr): ...

    @function
    def f(x: StringLike) -> A: ...

    egraph = EGraph()
    egraph.register(f("a"), f("b"))
    with pytest.warns(UserWarning, match="f"):
        egraph._serialize(max_calls_per_function=1)


EXAMPLE_FILES = list((pathlib.Path(__file__).parent / "../egglog/examples").glob("*.py"))


# Test all files in the `examples` directory by importing them in this parametrized test
@pytest.mark.parametrize("name", [f.stem for f in EXAMPLE_FILES if f.stem != "__init__"])
def test_example(name):
    importlib.import_module(f"egglog.examples.{name}")


@function
def f() -> i64: ...


class E(Expr):
    X: ClassVar[i64]

    def __init__(self) -> None: ...
    def m(self) -> i64: ...

    @property
    def p(self) -> i64: ...

    @classmethod
    def cm(cls) -> i64: ...


egraph = EGraph()

C = constant("C", i64)

zero = i64(0)
egraph.register(
    set_(f()).to(zero),
    set_(E().m()).to(zero),
    set_(E.X).to(zero),
    set_(E().p).to(zero),
    set_(C).to(zero),
    set_(E.cm()).to(zero),
)


@pytest.mark.parametrize(
    "c",
    [
        pytest.param(E, id="init"),
        pytest.param(f, id="function"),
        pytest.param(E.m, id="method"),
        pytest.param(E.X, id="class var"),
        pytest.param(E.p, id="property"),
        pytest.param(C, id="constant"),
        pytest.param(E.cm, id="class method"),
    ],
)
def test_function_size(c):
    assert egraph.function_size(c) == 1


def test_all_function_size():
    res = egraph.all_function_sizes()
    assert set(res) == {
        (E, 1),
        (f, 1),
        (E.m, 1),
        (E.X, 1),
        (E.p, 1),
        (C, 1),
        (E.cm, 1),
    }


def test_overall_run_report():
    assert EGraph().stats()


def test_function_values():
    egraph = EGraph()

    @function
    def f(x: i64Like) -> i64: ...

    egraph.register(set_(f(i64(1))).to(i64(2)))
    values = egraph.function_values(f)
    assert values == {f(i64(1)): i64(2)}


def test_dynamic_cost():
    """
    https://github.com/egraphs-good/egglog-experimental/blob/6d07a34ac76deec751f86f70d9b9358cd3e236ca/tests/integration_test.rs#L5-L35
    """

    class E(Expr):
        def __init__(self, x: i64Like) -> None: ...
        def __add__(self, other: E) -> E: ...
        @method(cost=200)
        def __sub__(self, other: E) -> E: ...

    egraph = EGraph()
    egraph.register(
        union(E(2)).with_(E(1) + E(1)),
        set_cost(E(2), 1000),
        set_cost(E(1), 100),
    )
    assert egraph.extract(E(2), include_cost=True) == (E(1) + E(1), 203)
    with egraph:
        egraph.register(set_cost(E(1) + E(1), 800))
        assert egraph.extract(E(2), include_cost=True) == (E(2), 1001)
    with egraph:
        egraph.register(set_cost(E(1) + E(1), 798))
        assert egraph.extract(E(2), include_cost=True) == (E(1) + E(1), 1000)
    egraph.register(union(E(2)).with_(E(5) - E(3)))
    assert egraph.extract(E(2), include_cost=True) == (E(1) + E(1), 203)
    egraph.register(set_cost(E(5) - E(3), 198))
    assert egraph.extract(E(2), include_cost=True) == (E(5) - E(3), 202)


class TestScheduler:
    def test_sequence_repeat_saturate(self):
        """
        Mirrors the scheduling example: alternate step-right and step-left,
        saturating each, repeated 10 times. Verifies final facts.
        """
        egraph = EGraph()

        left = relation("left", i64)
        right = relation("right", i64)

        x, y = vars_("x y", i64)

        # Name rulesets to make schedule translation stable and explicit
        step_left = ruleset(
            rule(
                left(x),
                right(x),
            ).then(left(x + 1)),
            name="step-left",
        )
        step_right = ruleset(
            rule(
                left(x),
                right(y),
                eq(x).to(y + 1),
            ).then(right(x)),
            name="step-right",
        )

        # Initial facts
        egraph.register(left(i64(0)), right(i64(0)))

        # (repeat 10 (seq (run step-right) (saturate step-left)))
        egraph.run(seq(step_right, step_left) * 10)

        # We took 10 left steps, but only 9 right steps (first can't move)
        egraph.check(left(i64(10)), right(i64(9)))
        egraph.check_fail(left(i64(11)), right(i64(10)))

    def test_backoff_scheduler(self):
        """
        Passing `scheduler=...` to run(...) hoists the scheduler to the
        outer scope. This is equivalent to an explicit outer `bo.scope(...)`.

        https://egraphs.zulipchat.com/#narrow/channel/375765-egg.2Fegglog/topic/.E2.9C.94.20Backoff.20Scheduler.20Example/with/538745863
        """
        includes = relation("includes", i64)
        x = var("x", i64)
        grow = ruleset(rule(includes(x)).then(includes(x + 1)))
        shrink = ruleset(rule(includes(x)).then(includes(x - 1)))

        e1 = EGraph()
        e1.register(includes(i64(0)))
        # default scheduler
        with e1:
            e1.run((grow + shrink) * 3)
            e1.check(includes(i64(3)), includes(i64(-3)))
        # back-off implicit outer hoisting
        bo = back_off(match_limit=1)
        with e1:
            e1.run((run(grow, scheduler=bo) + shrink) * 3)
            e1.check(includes(i64(2)), includes(i64(-3)))
            e1.check_fail(includes(i64(3)))
        # back off inner hoisting
        with e1:
            e1.run(bo.scope(run(grow, scheduler=bo) + shrink) * 3)
            e1.check(includes(i64(1)), includes(i64(-3)))
            e1.check_fail(includes(i64(2)))

    def test_custom_scheduler_invalid_until(self):
        """
        Custom schedulers do not support equality facts in :until,
        and only allow a single non-equality fact.
        """
        egraph = EGraph()

        rel = relation("rel", i64)
        x = var("x", i64)
        r = ruleset(name="r")
        bo = back_off(match_limit=1)

        # Equality in until should error via high-level run
        with pytest.raises(ValueError, match="Cannot use equality fact with custom scheduler"):
            egraph.run(run(r, eq(x).to(i64(1)), scheduler=bo))

        # Multiple until facts should error via high-level run
        with pytest.raises(ValueError, match="Can only have one until fact with custom scheduler"):
            egraph.run(run(r, rel(i64(0)), rel(i64(1)), scheduler=bo))


@function
def ff(x: i64Like, y: i64Like) -> E: ...


@function
def gg() -> E: ...


class TestCustomExtract:
    @pytest.mark.parametrize(
        "expr",
        [
            pytest.param(i64(10), id="i64"),
            pytest.param(f64(10.0), id="f64"),
            pytest.param(String("hi"), id="String"),
            pytest.param(Bool(True), id="Bool"),
            pytest.param(Rational(1, 2), id="Rational"),
            pytest.param(BigInt(10), id="BigInt"),
            pytest.param(BigRat(1, 2), id="BigRat"),
            pytest.param(PyObject("hi"), id="PyObject"),
            pytest.param(Vec(i64(1), i64(2)), id="Vec"),
            pytest.param(Set(i64(1), i64(2)), id="Set"),
            pytest.param(Map[i64, String].empty().insert(i64(1), String("hi")), id="Map"),
            pytest.param(MultiSet(i64(1), i64(1)), id="MultiSet"),
            pytest.param(Unit(), id="Unit"),
            pytest.param(UnstableFn[E, i64, i64](ff), id="fn"),
            pytest.param(UnstableFn[E, i64](ff, i64(1)), id="fn partial"),
        ],
    )
    def test_to_from_value(self, expr):
        egraph = EGraph()
        expr = egraph.extract(expr)
        assert expr == self._to_from_value(egraph, expr)

    def _to_from_value(self, egraph, expr):
        typed_expr = expr.__egg_typed_expr__
        value = egraph._state.typed_expr_to_value(typed_expr)
        res_val = egraph._state.value_to_expr(typed_expr.tp, value)
        return expr.__with_expr__(TypedExprDecl(typed_expr.tp, res_val))

    def test_compare_values(self):
        egraph = EGraph()
        egraph.register(E(), gg())
        e_value = self._to_from_value(egraph, E())
        gg_value = self._to_from_value(egraph, gg())
        assert e_value != gg_value
        assert hash(e_value) != hash(gg_value)
        assert str(e_value) != str(gg_value)

    def test_no_changes(self):
        egraph = EGraph()
        assert egraph.extract(E(), include_cost=True) == egraph.extract(
            E(), include_cost=True, cost_model=default_cost_model
        )

    def test_calls_methods(self):
        @function
        def my_f(xs: Vec[i64]) -> E: ...

        # cost = 2
        x = i64(10)
        # cost = 3 + 2 = 5
        xs = Vec[i64](x)
        # cost = 100
        res = E()
        # cost = 1 + 5  = 6
        called = my_f(xs)
        egraph = EGraph()
        egraph.register(union(called).with_(res))

        def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if get_callable_fn(expr) == E:
                return 100
            match expr:
                case i64():
                    return 2
                case Vec():
                    return 3 + sum(children_costs)
            return default_cost_model(egraph, expr, children_costs)

        my_cost_model = MagicMock(side_effect=my_cost_model)
        assert egraph.extract(called, include_cost=True, cost_model=my_cost_model) == (called, 6)

        my_cost_model.assert_any_call(egraph, res, [])
        my_cost_model.assert_any_call(egraph, xs, [2])
        my_cost_model.assert_any_call(egraph, x, [])
        my_cost_model.assert_any_call(egraph, called, [5])

    @pytest.mark.xfail(reason="Errors dont bubble, just panic")
    def test_errors_bubble(self):
        def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            msg = "bad"
            raise ValueError(msg)

        egraph = EGraph()

        with pytest.raises(ValueError, match="bad"):
            egraph.extract(i64(10), cost_model=my_cost_model)

    def test_dag_cost_model(self):
        egraph = EGraph()
        expr = ff(1, 2)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=greedy_dag_cost_model())
        assert cost.total == 3
        assert expr == res

        expr = ff(1, 1)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=greedy_dag_cost_model())
        assert cost.total == 2
        assert expr == res

        @function
        def bin(l: E, r: E) -> E: ...

        x = constant("x", E)
        y = constant("y", E)
        expr = bin(x, bin(x, y))
        egraph.register(expr)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=greedy_dag_cost_model())
        assert cost.total == 4
        assert expr == res
