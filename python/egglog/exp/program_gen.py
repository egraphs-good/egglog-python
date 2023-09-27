# mypy: disable-error-code="empty-body"
"""
Builds up imperative string expressions from a functional expression.
"""
from __future__ import annotations

from typing import Union

from egglog import *

program_gen_module = Module()

ProgramLike = Union["Program", StringLike]


@program_gen_module.class_
class Program(Expr):
    """
    Semanticallly represents an expression with a number of ordered statements that it depends on to run.

    The expression and statements are all represented as strings.
    """

    def __init__(self, expr: StringLike) -> None:
        """
        Create a program based on a string expression.
        """
        ...

    def __add__(self, other: ProgramLike) -> Program:
        """
        Concats the strings of the two expressions and also the statements.
        """
        ...

    def statement(self, statement: ProgramLike) -> Program:
        """
        Uses the expression of the statement and adds that as a statement to the program.
        """
        ...

    def assign(self) -> Program:
        """
        Returns a new program with the expression assigned to a gensym.
        """
        ...

    @property
    def expr(self) -> String:
        """
        Returns the expression of the program, if it's been compiled
        """
        ...

    @property
    def statements(self) -> String:
        """
        Returns the statements of the program, if it's been compiled
        """
        ...

    @property
    def next_sym(self) -> i64:
        """
        Returns the next gensym to use.
        """
        ...

    @program_gen_module.method(default=Unit())
    def compile(self, next_sym: i64 = i64(0)) -> Unit:
        """
        Triggers compilation of the program.
        """


converter(String, Program, Program)


@program_gen_module.register
def _compile(
    s: String,
    s1: String,
    s2: String,
    s3: String,
    s4: String,
    p: Program,
    p1: Program,
    p2: Program,
    # c: Compiler,
    statements: Program,
    expr: Program,
    i: i64,
    m: Map[Program, Program],
):
    # Combining two strings is just joining them
    yield rewrite(Program(s1) + Program(s2)).to(Program(join(s1, s2)))

    # Compiling a string just gives that string
    program_expr = Program(s)
    yield rule(program_expr.compile(i)).then(
        set_(program_expr.expr).to(s),
        set_(program_expr.statements).to(String("")),
        set_(program_expr.next_sym).to(i),
    )
    # Compiling a statement means that we should use the expression of the statement as a statement and use the expression
    # of the underlying program
    program_statement = p.statement(p1)
    # First compile the expression
    yield rule(program_statement.compile(i)).then(p.compile(i))
    # Then, when the expression is compiled, compile the statement, and set the expr of the whole statement
    yield rule(
        eq(p2).to(program_statement),
        eq(i).to(p.next_sym),
        eq(s).to(p.expr),
    ).then(p1.compile(i), set_(p2.expr).to(s))
    # When both are compiled, add the statements of both + the expr of p1 to the statements of p
    yield rule(
        eq(p2).to(program_statement),
        eq(s1).to(p.statements),
        eq(s2).to(p1.statements),
        eq(s).to(p1.expr),
        eq(i).to(p1.next_sym),
    ).then(
        set_(p2.statements).to(join(s1, s2, s, "\n")),
        set_(p2.next_sym).to(i),
    )

    # Compiling an addition is the same as compiling one, then the other, then setting the expression as the addition
    # of the two
    program_add = p1 + p2
    # Compile the first
    yield rule(program_add.compile(i)).then(p1.compile(i))
    # Once the first is finished, do the second
    yield rule(program_add, eq(i).to(p1.next_sym)).then(p2.compile(i))
    # Once the second is finished, set the the addition to the addition of the two expressions
    yield rule(
        eq(p).to(program_add),
        eq(s1).to(p1.expr),
        eq(s2).to(p2.expr),
        eq(s3).to(p1.statements),
        eq(s4).to(p2.statements),
        eq(i).to(p2.next_sym),
    ).then(
        set_(p.expr).to(join(s1, s2)),
        set_(p.statements).to(join(s3, s4)),
        set_(p.next_sym).to(i),
    )

    # Compiling an assign is the same as compiling the expression, adding an assign statement, then setting the
    # expression as the gensym
    program_assign = p.assign()
    # Compile the expression
    yield rule(program_assign.compile(i)).then(p.compile(i))
    # Once the expression is compiled, add the assign statement to the statements and set the expr

    symbol = join(String("_"), i.to_string())
    yield rule(
        eq(p1).to(program_assign),
        eq(s1).to(p.statements),
        eq(s2).to(p.expr),
        eq(i).to(p.next_sym),
    ).then(
        set_(p1.statements).to(join(s1, symbol, " = ", s2, "\n")),
        set_(p1.expr).to(symbol),
        set_(p1.next_sym).to(i + 1),
    )
