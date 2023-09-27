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


converter(String, Program, Program)


@program_gen_module.class_
class Compiler(Expr):
    def __init__(
        self,
        # The mapping from programs to their compiled of the expressions
        compiled_programs: Map[Program, Program] = Map[Program, Program].empty(),
        # The next gensym counter
        sym_counter: i64Like = i64(0),
        # The cumulative list of statements seperated by newlines, all stored as the expression of a program
        compiled_statements: Program = Program(""),
        # The compiled expression from the last `compile` call
        compiled_expr: Program = Program(""),
    ) -> None:
        ...

    def compile(self, program: ProgramLike) -> Compiler:
        ...

    @property
    def expr(self) -> Program:
        ...

    def set_expr(self, expr: Program) -> Compiler:
        ...

    def add_statement(self, statements: Program) -> Compiler:
        ...

    @property
    def string(self) -> String:
        ...

    def added_sym(self) -> Compiler:
        ...

    @property
    def next_sym(self) -> Program:
        ...


@program_gen_module.register
def _compile(
    s: String,
    s1: String,
    s2: String,
    p: Program,
    p1: Program,
    p2: Program,
    c: Compiler,
    statements: Program,
    expr: Program,
    i: i64,
    m: Map[Program, Program],
):
    # Combining two strings is just joining them
    yield rewrite(Program(s1) + Program(s2)).to(Program(join(s1, s2)))

    compiler = Compiler(m, i, statements, expr)
    # Compiling a program that is already in the compiled programs is a no-op, but the expression is updated
    yield rewrite(compiler.compile(p)).to(compiler.set_expr(m[p]), m.contains(p))
    # Compiling a string just gives that string
    program_expr = Program(s)
    yield rewrite(compiler.compile(program_expr)).to(
        Compiler(m.insert(program_expr, program_expr), i, statements, program_expr), m.not_contains(program_expr)
    )
    # Compiling a statement means that we should use the expression of the statement as a statement and use the expression
    # of the underlying program
    program_statement = p.statement(p1)
    p_compiled = compiler.compile(p)
    p1_compiled = p_compiled.compile(p1)
    yield rewrite(compiler.compile(program_statement)).to(
        p1_compiled.add_statement(p1_compiled.expr).set_expr(p_compiled.expr), m.not_contains(program_statement)
    )

    # Compiling an addition is the same as compiling one, then the other, then setting the expression as the addition
    # of the two
    program_add = p1 + p2
    p1_compiled = compiler.compile(p1)
    p2_compiled = p1_compiled.compile(p2)
    yield rewrite(compiler.compile(program_add)).to(
        p2_compiled.set_expr(p1_compiled.expr + p2_compiled.expr), m.not_contains(program_add)
    )

    # Compiling an assign is the same as compiling the expression, adding an assign statement, then setting the
    # expression as the gensym
    program_assign = p.assign()
    p_compiled = compiler.compile(p)
    yield rewrite(compiler.compile(program_assign)).to(
        p_compiled.add_statement(p_compiled.next_sym + " = " + p_compiled.expr)
        .set_expr(p_compiled.next_sym)
        .added_sym(),
        m.not_contains(program_assign),
    )

    yield rewrite(compiler.set_expr(p)).to(Compiler(m, i, statements, p))
    yield rewrite(compiler.add_statement(p)).to(Compiler(m, i, statements + "\n" + p, expr), m.not_contains(p))
    yield rewrite(compiler.add_statement(p)).to(compiler, m.contains(p))
    yield rewrite(compiler.expr).to(expr)
    yield rewrite(compiler.added_sym()).to(Compiler(m, i + 1, statements, expr))
    yield rewrite(compiler.next_sym).to(Program(join("_", i.to_string())))

    # Set `to_string` to the compiled statements added to the compiled expression
    yield rule(
        eq(c).to(Compiler(m, i, Program(s1), Program(s2))),
    ).then(set_(c.string).to(join(s1, "\n", s2, "\n")))
