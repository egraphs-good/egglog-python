# mypy: disable-error-code="empty-body"
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

from egglog import *
from egglog.exp.program_gen import *

if TYPE_CHECKING:
    from types import FunctionType


class Math(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, v: StringLike) -> Math: ...

    def __add__(self, other: Math) -> Math: ...

    def __mul__(self, other: Math) -> Math: ...

    def __neg__(self) -> Math: ...

    @method(cost=1000)  # type: ignore[misc]
    @property
    def program(self) -> Program: ...


@function
def assume_pos(x: Math) -> Math: ...


@ruleset
def to_program_ruleset(
    s: String,
    y_expr: String,
    z_expr: String,
    old_statements: String,
    x: Math,
    i: i64,
    y: Math,
    z: Math,
    old_gensym: i64,
):
    yield rewrite(Math.var(s).program).to(Program(s))
    yield rewrite(Math(i).program).to(Program(i.to_string()))
    yield rewrite((y + z).program).to((y.program + " + " + z.program).assign())
    yield rewrite((y * z).program).to((y.program + " * " + z.program).assign())
    yield rewrite((-y).program).to(Program("-") + y.program)
    assigned_x = x.program.assign()
    yield rewrite(assume_pos(x).program).to(assigned_x.statement(Program("assert ") + assigned_x + " > 0"))


def test_to_string(snapshot_py) -> None:
    first = assume_pos(-Math.var("x")) + Math.var("y")
    fn = (first + Math(2) + first).program.function_two(Math.var("x").program, Math.var("y").program, "my_fn")
    egraph = EGraph()
    egraph.register(fn.compile())
    egraph.run((to_program_ruleset | program_gen_ruleset).saturate())
    egraph.check(fn.expr == String("my_fn"))
    assert egraph.extract(fn.statements).eval() == snapshot_py


def test_to_string_function_three(snapshot_py) -> None:
    first = assume_pos(-Math.var("x")) + Math.var("y") + Math.var("z")
    fn = (first + Math(2) + first).program.function_three(
        Math.var("x").program, Math.var("y").program, Math.var("z").program, "my_fn"
    )
    egraph = EGraph()
    egraph.register(fn.compile())
    egraph.run((to_program_ruleset | program_gen_ruleset).saturate())
    assert egraph.extract(fn.expr).eval() == "my_fn"
    assert egraph.extract(fn.statements).eval() == snapshot_py


def test_py_object():
    x = Math.var("x")
    y = Math.var("y")
    z = Math.var("z")
    fn = (x + y + z).program.function_two(x.program, y.program)
    evalled = EvalProgram(fn, {"z": 10})
    egraph = EGraph()
    egraph.register(evalled)
    egraph.run((to_program_ruleset | eval_program_rulseset | program_gen_ruleset).saturate())
    res = cast("FunctionType", egraph.extract(evalled.as_py_object).eval())
    assert res(1, 2) == 13
    assert inspect.getsource(res)
