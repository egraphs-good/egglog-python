# mypy: disable-error-code="empty-body"
from __future__ import annotations

from egglog import *
from egglog.exp.program_gen import *


def test_to_string(snapshot_py) -> None:
    egraph = EGraph([program_gen_module])

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

        def __neg__(self) -> Math:
            ...

        @property
        def program(self) -> Program:
            ...

    @egraph.function
    def assume_pos(x: Math) -> Math:
        ...

    @egraph.register
    def _rules(
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
        yield rewrite((-y).program).to(Program("-(") + y.program + ")")
        assigned_x = x.program.assign()
        yield rewrite(assume_pos(x).program).to(assigned_x.statement(Program("assert ") + assigned_x + " > 0"))

    first = assume_pos(-Math.var("x")) + Math(3)
    y = egraph.let("y", Math(2) * first + Math(0) + first)
    compiled = Compiler().compile(y.program)
    egraph.register(compiled)
    egraph.run(100)
    egraph.display(max_calls_per_function=40, n_inline_leaves=2)
    assert egraph.load_object(egraph.extract(PyObject.from_string(compiled.string))) == snapshot_py

    # egraph.run(10)
    # egraph.check(eq(y.expr).to(String("_1")))
    # egraph.check(eq(y.statements).to(String("_0 = x + -3\n_1 = 2 * _0\n")))
