# mypy: disable-error-code="empty-body"
from __future__ import annotations

import ast
import math
from collections.abc import Callable
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import pytest

from egglog import *

if TYPE_CHECKING:
    from egglog.runtime import RuntimeExpr


class A(Expr):
    V: ClassVar[A]

    def __init__(self) -> None: ...
    @classmethod
    def cls_method(cls) -> A: ...
    def method(self) -> A: ...
    def __neg__(self) -> A: ...
    def __add__(self, other: A) -> A: ...
    def __getitem__(self, key: A) -> A: ...
    def __call__(self) -> A: ...
    def __delitem__(self, key: A) -> None: ...
    def __setitem__(self, key: A, value: A) -> None: ...
    @property
    def prop(self) -> A: ...

    def __sub__(self, other: A) -> A: ...
    def __mul__(self, other: A) -> A: ...
    def __truediv__(self, other: A) -> A: ...
    def __floordiv__(self, other: A) -> A: ...
    def __mod__(self, other: A) -> A: ...
    def __pow__(self, other: A) -> A: ...
    def __lshift__(self, other: A) -> A: ...
    def __rshift__(self, other: A) -> A: ...

    def __and__(self, other: A) -> A: ...
    def __xor__(self, other: A) -> A: ...
    def __lt__(self, other: A) -> A: ...
    def __le__(self, other: A) -> A: ...
    def __gt__(self, other: A) -> A: ...
    def __ge__(self, other: A) -> A: ...
    def __eq__(self, other: A) -> A: ...  # type: ignore[override]
    def __ne__(self, other: A) -> A: ...  # type: ignore[override]
    def __getattr__(self, name: StringLike) -> A: ...

    def __pos__(self) -> A: ...
    def __invert__(self) -> A: ...
    def __round__(self) -> A: ...
    def __abs__(self) -> A: ...
    def __trunc__(self) -> A: ...
    def __floor__(self) -> A: ...
    def __ceil__(self) -> A: ...


class Box(Expr):
    def __init__(self, vec: Vec[A] = Vec[A].empty()) -> None: ...


class Wrapper(Expr):
    def __init__(self, box: Box) -> None: ...


@function
def f(x: A) -> A: ...


@function
def g() -> A: ...


@function
def h() -> A: ...


@function
def p() -> i64: ...


@function
def has_default(x: A = A()) -> A: ...


@function
def higher_order(x: Callable[[A], A]) -> A: ...


@function
def binary(x: A, y: A) -> A: ...


del_a = A()
del del_a[g()]

del_del_a = copy(del_a)
del del_del_a[h()]

del_del_a_two = copy(del_a)
del del_del_a_two[A()]

setitem_a = A()
setitem_a[g()] = h()

b = constant("b", A)
c_i64 = constant("c_i64", i64)
rel = relation("rel", A)


@function
def my_very_long_function_name() -> A: ...


long_line = (
    my_very_long_function_name()
    + my_very_long_function_name()
    + my_very_long_function_name()
    + my_very_long_function_name()
)

r = ruleset(name="r")

bo = back_off(ban_length=5)
bo_persistent = back_off(ban_length=5).persistent()


class BadRepr:
    def __repr__(self) -> str:
        return "&&&"


PARAMS = [
    # expression function calls
    pytest.param(A(), "A()", id="init"),
    pytest.param(f(A()), "f(A())", id="call0"),
    pytest.param(A.cls_method(), "A.cls_method()", id="class method"),
    pytest.param(A().method(), "A().method()", id="instance method"),
    pytest.param(-A(), "-A()", id="unary operator"),
    pytest.param(A() + g(), "A() + g()", id="binary operator"),
    pytest.param(A()[g()], "A()[g()]", id="getitem"),
    pytest.param(del_a, "_A_1 = A()\ndel _A_1[g()]\n_A_1", id="delitem"),
    pytest.param(setitem_a, "_A_1 = A()\n_A_1[g()] = h()\n_A_1", id="setitem"),
    pytest.param(del_a + del_a, "_A_1 = A()\ndel _A_1[g()]\n_A_1 + _A_1", id="existing de-duplicate"),
    pytest.param(del_del_a, "_A_1 = A()\ndel _A_1[g()]\ndel _A_1[h()]\n_A_1", id="re-use variable"),
    pytest.param(
        del_del_a + del_del_a_two,
        """_A_1 = A()
del _A_1[g()]
_A_2 = copy(_A_1)
del _A_2[h()]
_A_3 = copy(_A_1)
del _A_3[A()]
_A_2 + _A_3""",
        id="copy name",
    ),
    pytest.param(b, "b", id="constant"),
    pytest.param(A.V, "A.V", id="class variable"),
    pytest.param(A().prop, "A().prop", id="property"),
    pytest.param(ne(A()).to(g()), "ne(A()).to(g())", id="ne"),
    pytest.param(has_default(A()), "has_default()", id="has default"),
    pytest.param(A() - A(), "A() - A()", id="subtraction"),
    pytest.param(A() * A(), "A() * A()", id="multiplication"),
    pytest.param(A() / A(), "A() / A()", id="division"),
    pytest.param(A() // A(), "A() // A()", id="floor division"),
    pytest.param(A() % A(), "A() % A()", id="modulus"),
    pytest.param(A() ** A(), "A() ** A()", id="exponentiation"),
    pytest.param(A() << A(), "A() << A()", id="left shift"),
    pytest.param(A() >> A(), "A() >> A()", id="right shift"),
    pytest.param(A() & A(), "A() & A()", id="and"),
    pytest.param(A() ^ A(), "A() ^ A()", id="xor"),
    pytest.param(A() < A(), "A() < A()", id="less than"),
    pytest.param(A() <= A(), "A() <= A()", id="less than or equal"),
    pytest.param(A() > A(), "A() > A()", id="greater than"),
    pytest.param(A() >= A(), "A() >= A()", id="greater than or equal"),
    pytest.param(A() == A(), "A() == A()", id="equal"),
    pytest.param(A() != A(), "A() != A()", id="not equal"),
    pytest.param(A().__getattr__(String("a") + "b"), 'A().__getattr__(join("a", "b"))', id="getattr expr"),
    pytest.param(+A(), "+A()", id="pos"),
    pytest.param(~A(), "~A()", id="invert"),
    # primitives
    pytest.param(Unit(), "Unit()", id="unit"),
    pytest.param(Bool(True), "Bool(True)", id="bool"),
    pytest.param(i64(42), "i64(42)", id="i64"),
    pytest.param(f64(42.1), "f64(42.1)", id="f64"),
    pytest.param(String("hello"), 'String("hello")', id="string"),
    pytest.param(PyObject("hi"), 'PyObject("hi")', id="pyobject"),
    pytest.param(PyObject(BadRepr()), 'PyObject(eval("&&&"))', id="pyobject-invalid-str"),
    pytest.param(var("x", A), "x", id="variable"),
    # commands
    pytest.param(rewrite(g()).to(h(), A()), "rewrite(g()).to(h(), A())", id="rewrite"),
    pytest.param(rule(g()).then(h()), "rule(g()).then(h())", id="rule"),
    # Actions
    pytest.param(expr_action(A()), "A()", id="action"),
    pytest.param(set_(p()).to(i64(1)), "set_(p()).to(i64(1))", id="set"),
    pytest.param(union(g()).with_(h()), "union(g()).with_(h())", id="union"),
    pytest.param(let("x", A()), "set_(x).to(A())", id="let"),
    pytest.param(expr_action(A()), "A()", id="expr action"),
    pytest.param(delete(p()), "delete(p())", id="delete"),
    pytest.param(panic("oh no"), 'panic("oh no")', id="panic"),
    # Fact
    pytest.param(expr_fact(A()), "A()", id="expr fact"),
    pytest.param(eq(g()).to(h()), "eq(g()).to(h())", id="eq"),
    # Ruleset
    pytest.param(ruleset(rewrite(g()).to(h())), "ruleset(rewrite(g()).to(h()))", id="ruleset"),
    # Functions
    pytest.param(f, "f", id="function"),
    pytest.param(A().method, "A().method", id="method"),
    # lambda
    pytest.param(higher_order(lambda x: A() + x), "higher_order(lambda x: A() + x)", id="lambda"),
    # partial
    pytest.param(higher_order(partial(binary, A())), "higher_order(partial(binary, A()))", id="partial"),
    pytest.param(higher_order(lambda x: f(b)), "higher_order(lambda x: f(b))", id="partial-lambda"),
]


@pytest.mark.parametrize(("x", "s"), PARAMS)
def test_str(x: RuntimeExpr, s: str) -> None:
    assert ast.dump(ast.parse(str(x))) == ast.dump(ast.parse(s))


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(A()(), id="call1"),
        pytest.param(A().foo, id="getattr"),
        pytest.param(round(A()), id="round"),
        pytest.param(abs(A()), id="abs"),
        pytest.param(math.trunc(A()), id="trunc"),
        pytest.param(math.floor(A()), id="floor"),
        pytest.param(math.ceil(A()), id="ceil"),
    ],
)
def test_expr_pretty_round_trip(expr: BaseExpr) -> None:
    assert expr_parts(eval(str(expr), globals())) == expr_parts(expr)


@pytest.mark.parametrize("original", [pytest.param(rewrite(long_line).to(long_line), id="wrap long line")])
def test_rewrite_pretty_round_trip(original: RewriteOrRule) -> None:
    assert eval(str(original), globals()) == original


@pytest.mark.parametrize(
    ("schedule", "rendered"),
    [
        pytest.param(r, 'ruleset(name="r")', id="ruleset with name"),
        pytest.param(r.saturate(), 'ruleset(name="r").saturate()', id="saturate"),
        pytest.param(r * 10, 'ruleset(name="r") * 10', id="repeat"),
        pytest.param(r + r, 'seq(ruleset(name="r"), ruleset(name="r"))', id="sequence"),
        pytest.param(seq(r, r, r), 'seq(ruleset(name="r"), ruleset(name="r"), ruleset(name="r"))', id="seq"),
        pytest.param(run(r, h()), 'repeat(times=1, body=seq(ruleset(name="r")), until=[h()])', id="run"),
        pytest.param(run(None, h()), "repeat(times=1, body=seq(run()), until=[h()])", id="default run with until"),
        pytest.param(
            run(None, h(), scheduler=bo),
            "seq(bind_scheduler({scheduler_name}, back_off(ban_length=5)), "
            "repeat(times=1, body=seq(run(scheduler={scheduler_name})), until=[h()]))",
            id="default run with scheduler",
        ),
        pytest.param(
            run(r, h(), scheduler=bo),
            "seq(bind_scheduler({scheduler_name}, back_off(ban_length=5)), "
            'repeat(times=1, body=seq(run(ruleset(name="r"), scheduler={scheduler_name})), until=[h()]))',
            id="run with scheduler",
        ),
        pytest.param(
            bo.scope(run(r, scheduler=bo)),
            "seq(bind_scheduler({scheduler_name}, back_off(ban_length=5)), "
            'run(ruleset(name="r"), scheduler={scheduler_name}))',
            id="scoped scheduler",
        ),
        pytest.param(
            bo_persistent,
            "bind_scheduler({scheduler_name}, back_off(ban_length=5)).persistent()",
            id="persistent scheduler",
        ),
    ],
)
def test_schedule_str(schedule: Schedule | BackOff, rendered: str) -> None:
    actual = ast.parse(str(schedule))
    bindings = [
        node
        for node in ast.walk(actual)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bind_scheduler"
    ]
    assert len(bindings) <= 1
    scheduler_name = None
    if bindings:
        name = bindings[0].args[0]
        assert isinstance(name, ast.Constant)
        scheduler_name = name.value
        assert isinstance(scheduler_name, str)
        assert scheduler_name
    expected = ast.parse(rendered.format(scheduler_name=repr(scheduler_name)))
    assert ast.dump(actual) == ast.dump(expected)

    if schedule is r:
        for selected, expected_source in [
            (ruleset(), "ruleset()"),
            (ruleset(name=""), 'ruleset(name="")'),
            (
                unstable_combine_rulesets(r, name="joined"),
                'unstable_combine_rulesets(ruleset(name="r"), name="joined")',
            ),
            (unstable_combine_rulesets(r, name=""), 'unstable_combine_rulesets(ruleset(name="r"))'),
        ]:
            assert ast.dump(ast.parse(str(selected))) == ast.dump(ast.parse(expected_source))


@pytest.mark.parametrize(
    ("eval_mode", "option"),
    [
        ("seminaive", ""),
        ("naive", ', eval_mode="naive"'),
        ("unsafe-seminaive", ', eval_mode="unsafe-seminaive"'),
    ],
)
def test_rule_eval_mode_pretty_round_trip(eval_mode: RuleEvalMode, option: str) -> None:
    original = rule(rel(g()), name="mode rule", eval_mode=eval_mode).then(rel(h()))
    rendered = f'rule(rel(g()), name="mode rule"{option}).then(rel(h()))'

    assert ast.dump(ast.parse(str(original))) == ast.dump(ast.parse(rendered))
    assert eval(str(original), globals()) == original


def test_rule_no_decomp_pretty_round_trip() -> None:
    original = rule(rel(g()), name="no decomp rule", no_decomp=True).then(rel(h()))
    rendered = 'rule(rel(g()), name="no decomp rule", no_decomp=True).then(rel(h()))'

    assert ast.dump(ast.parse(str(original))) == ast.dump(ast.parse(rendered))
    assert eval(str(original), globals()) == original


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(BigRat(2, 1), id="integer BigRat"),
        pytest.param(BigRat(1, 2), id="fractional BigRat"),
        pytest.param(Map[String, BigRat].empty().insert(String("x"), BigRat(1, 2)), id="Map"),
        pytest.param(Maybe[String].some(String("value")), id="Maybe some"),
        pytest.param(Maybe[String].none(), id="Maybe none"),
        pytest.param(
            Pair[Map[String, BigRat], Maybe[String]](
                Map[String, BigRat].empty().insert(String("x"), BigRat(1, 2)),
                Maybe[String].some(String("value")),
            ),
            id="Pair",
        ),
    ],
)
def test_extracted_typed_pretty_is_executable(expr: BaseExpr) -> None:
    extracted = EGraph().extract(expr)

    rebuilt = eval(str(extracted), globals())

    check_eq(extracted, rebuilt)


def test_frozen_typed_container_bytes_round_trip() -> None:
    expr = Pair[Map[String, BigRat], Maybe[String]](
        Map[String, BigRat].empty().insert(String("x"), BigRat(1, 2)), Maybe[String].some(String("value"))
    )
    frozen = EGraph(expr).freeze()

    rebuilt = type(frozen).from_binary(frozen.to_binary())

    assert rebuilt == frozen
    assert rebuilt.to_binary() == frozen.to_binary()
