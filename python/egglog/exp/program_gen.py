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

    def expr_to_statement(self) -> Program:
        """
        Returns a new program with the expression as a statement and the new expression empty.
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
        Returns the next gensym to use. This is set after calling `compile(i)` on a program.
        """
        ...

    @program_gen_module.method(default=Unit())
    def compile(self, next_sym: i64 = i64(0)) -> Unit:
        """
        Triggers compilation of the program.
        """

    @program_gen_module.method(merge=lambda old, new: old)  # type: ignore[misc]
    @property
    def parent(self) -> Program:
        """
        Returns the parent of the program, if it's been compiled into the parent.

        Only keeps the original parent, not any additional ones, so that each set of statements is only added once.
        """
        ...


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

    ##
    # Statement
    ##
    # Compiling a statement means that we should use the expression of the statement as a statement and use the expression of the first
    yield rewrite(p1.statement(p2)).to(p1 + p2.expr_to_statement())

    ##
    # Expr to statement
    ##
    stmt = p1.expr_to_statement()
    # 1. Set parent
    yield rule(eq(p).to(stmt), p.compile(i)).then(set_(p1.parent).to(p))
    # 2. Compile p1 if parent set
    yield rule(eq(p).to(stmt), p.compile(i), eq(p1.parent).to(stmt)).then(p1.compile(i))
    # 3.a. If parent not set, set statements to expr
    yield rule(
        eq(p).to(stmt),
        p.compile(i),
        p1.parent != p,
        eq(s1).to(p1.expr),
    ).then(
        set_(p.statements).to(join(s1, "\n")),
        set_(p.next_sym).to(i),
        set_(p.expr).to(String("")),
    )
    # 3.b. If parent set, set statements to expr + statements
    yield rule(
        eq(p).to(stmt),
        eq(p1.parent).to(stmt),
        eq(s1).to(p1.expr),
        eq(s2).to(p1.statements),
        eq(i).to(p1.next_sym),
    ).then(
        set_(p.statements).to(join(s2, s1, "\n")),
        set_(p.next_sym).to(i),
        set_(p.expr).to(String("")),
    )

    ##
    # Addition
    ##

    # Compiling an addition is the same as compiling one, then the other, then setting the expression as the addition
    # of the two
    program_add = p1 + p2

    # Set parent of p1
    yield rule(eq(p).to(program_add), p.compile(i)).then(set_(p1.parent).to(p))

    # Compile p1, if p1 parent equal
    yield rule(eq(p).to(program_add), p.compile(i), eq(p1.parent).to(program_add)).then(p1.compile(i))

    # Set parent of p2, once p1 compiled
    yield rule(eq(p).to(program_add), p1.next_sym).then(set_(p2.parent).to(p))

    # Compile p2, if p1 parent not equal
    yield rule(eq(p).to(program_add), p.compile(i), p1.parent != p).then(p2.compile(i))

    # Compile p2, if p1 parent eqal
    yield rule(eq(p).to(program_add), eq(p1.parent).to(program_add), eq(i).to(p1.next_sym)).then(p2.compile(i))

    # Set p expr to join of p1 and p2
    yield rule(
        eq(p).to(program_add),
        eq(s1).to(p1.expr),
        eq(s2).to(p2.expr),
    ).then(
        set_(p.expr).to(join(s1, s2)),
    )
    # Set p statements to join and next sym to p2 if both parents set
    yield rule(
        eq(p).to(program_add),
        eq(p1.parent).to(p),
        eq(p2.parent).to(p),
        eq(s1).to(p1.statements),
        eq(s2).to(p2.statements),
        eq(i).to(p2.next_sym),
    ).then(
        set_(p.statements).to(join(s1, s2)),
        set_(p.next_sym).to(i),
    )
    # Set p statements to empty and next sym to i if neither parents set
    yield rule(
        eq(p).to(program_add),
        p.compile(i),
        p1.parent != p,
        p2.parent != p,
    ).then(
        set_(p.statements).to(String("")),
        set_(p.next_sym).to(i),
    )
    # Set p statements to p1 and next sym to p1 if p1 parent set and p2 parent not set
    yield rule(
        eq(p).to(program_add),
        eq(p1.parent).to(p),
        p2.parent != p,
        eq(s1).to(p1.statements),
        eq(i).to(p1.next_sym),
    ).then(
        set_(p.statements).to(s1),
        set_(p.next_sym).to(i),
    )
    # Set p statements to p2 and next sym to p2 if p2 parent set and p1 parent not set
    yield rule(
        eq(p).to(program_add),
        eq(p2.parent).to(p),
        p1.parent != p,
        eq(s2).to(p2.statements),
        eq(i).to(p2.next_sym),
    ).then(
        set_(p.statements).to(s2),
        set_(p.next_sym).to(i),
    )

    ##
    # Assign
    ##

    # Compiling an assign is the same as compiling the expression, adding an assign statement, then setting the
    # expression as the gensym
    program_assign = p1.assign()
    # Set parent
    yield rule(eq(p).to(program_assign), p.compile(i)).then(set_(p1.parent).to(p))
    # If parent set, compile the expression
    yield rule(eq(p).to(program_assign), p.compile(i), eq(p1.parent).to(program_assign)).then(p1.compile(i))

    # If p1 parent is p, then use statements of p, next sym of p
    symbol = join(String("_"), i.to_string())
    yield rule(
        eq(p).to(program_assign),
        eq(p1.parent).to(p),
        eq(s1).to(p1.statements),
        eq(i).to(p1.next_sym),
        eq(s2).to(p1.expr),
    ).then(
        set_(p.statements).to(join(s1, symbol, " = ", s2, "\n")),
        set_(p.expr).to(symbol),
        set_(p.next_sym).to(i + 1),
    )
    # If p1 parent is not p, then just use assign as statement, next sym of i
    yield rule(
        eq(p).to(program_assign),
        p1.parent != p,
        p.compile(i),
        eq(s2).to(p1.expr),
    ).then(
        set_(p.statements).to(join(symbol, " = ", s2, "\n")),
        set_(p.expr).to(symbol),
        set_(p.next_sym).to(i + 1),
    )
