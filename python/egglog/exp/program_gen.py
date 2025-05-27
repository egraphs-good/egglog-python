# mypy: disable-error-code="empty-body"
"""
Builds up imperative string expressions from a functional expression.
"""

from __future__ import annotations

from typing import TypeAlias

from egglog import *


class Program(Expr):
    """
    Semanticallly represents an expression with a number of ordered statements that it depends on to run.

    The expression and statements are all represented as strings.
    """

    def __init__(self, expr: StringLike, is_identifier: BoolLike = Bool(False)) -> None:
        """
        Create a program based on a string expression.
        """

    def __add__(self, other: ProgramLike) -> Program:
        """
        Concats the strings of the two expressions and also the statements.
        """

    @method(unextractable=True)
    def statement(self, statement: ProgramLike) -> Program:
        """
        Uses the expression of the statement and adds that as a statement to the program.
        """

    def assign(self) -> Program:
        """
        Returns a new program with the expression assigned to a gensym.
        """

    def function_two(self, arg1: ProgramLike, arg2: ProgramLike, name: StringLike = String("__fn")) -> Program:
        """
        Returns a new program defining a function with two arguments.
        """

    def function_three(
        self, arg1: ProgramLike, arg2: ProgramLike, arg3: ProgramLike, name: StringLike = String("__fn")
    ) -> Program:
        """
        Returns a new program defining a function with three arguments.
        """

    def expr_to_statement(self) -> Program:
        """
        Returns a new program with the expression as a statement and the new expression empty.
        """

    @property
    def expr(self) -> String:
        """
        Returns the expression of the program, if it's been compiled
        """

    @property
    def statements(self) -> String:
        """
        Returns the statements of the program, if it's been compiled
        """

    @property
    def next_sym(self) -> i64:
        """
        Returns the next gensym to use. This is set after calling `compile(i)` on a program.
        """

    # TODO: Replace w/ def next_sym(self) -> i64: ... ?
    def compile(self, next_sym: i64 = i64(0)) -> Unit:
        """
        Triggers compilation of the program.
        """

    @method(merge=lambda old, _new: old)  # type: ignore[misc]
    @property
    def parent(self) -> Program:
        """
        Returns the parent of the program, if it's been compiled into the parent.

        Only keeps the original parent, not any additional ones, so that each set of statements is only added once.
        """

    @property
    def is_identifer(self) -> Bool:
        """
        Returns whether the expression is an identifier. Used so that we don't re-assign any identifiers.
        """


ProgramLike: TypeAlias = Program | StringLike


converter(String, Program, Program)


class EvalProgram(Expr):
    def __init__(self, program: Program, globals: object) -> None:
        """
        Evaluates the program and saves as the py_object
        """

    # Only allow it to be set once, b/c hash of functions not stable
    @method(merge=lambda old, _new: old)  # type: ignore[misc]
    @property
    def as_py_object(self) -> PyObject:
        """
        Returns the python object of the program, if it's been evaluated.
        """


@ruleset
def eval_program_rulseset(ep: EvalProgram, p: Program, expr: String, statements: String, g: PyObject):
    # When we evaluate a program, we first want to compile to a string
    yield rule(EvalProgram(p, g)).then(p.compile())
    # Then we want to evaluate the statements/expr
    yield rule(
        eq(ep).to(EvalProgram(p, g)),
        eq(p.statements).to(statements),
        eq(p.expr).to(expr),
    ).then(
        set_(ep.as_py_object).to(
            py_eval(
                "l['___res']",
                PyObject.dict(PyObject.from_string("l"), py_exec(join(statements, "\n", "___res = ", expr), g)),
            )
        ),
    )


@ruleset
def program_gen_ruleset(
    s: String,
    s1: String,
    s2: String,
    s3: String,
    s4: String,
    s5: String,
    s6: String,
    p: Program,
    p1: Program,
    p2: Program,
    p3: Program,
    p4: Program,
    i: i64,
    i2: i64,
    b: Bool,
):
    ##
    # Expression
    ##

    # Compiling a string just gives that string
    yield rule(eq(p).to(Program(s, b)), p.compile(i)).then(
        set_(p.expr).to(s),
        set_(p.statements).to(String("")),
        set_(p.next_sym).to(i),
        set_(p.is_identifer).to(b),
    )

    ##
    # Statement
    ##
    # Compiling a statement means that we should use the expression of the statement as a statement and use the expression of the first
    yield rewrite(p1.statement(p2)).to(p1 + p2.expr_to_statement())

    ##
    # Expr to statement
    ##
    stmt = eq(p).to(p1.expr_to_statement())
    # 1. Set parent and is_identifier to false, since its empty
    yield rule(stmt, p.compile(i)).then(set_(p1.parent).to(p), set_(p.is_identifer).to(Bool(False)))
    # 2. Compile p1 if parent set
    yield rule(stmt, p.compile(i), eq(p1.parent).to(p)).then(p1.compile(i))
    # 3.a. If parent not set, set statements to expr
    yield rule(
        stmt,
        p.compile(i),
        ne(p1.parent).to(p),
        eq(s1).to(p1.expr),
    ).then(
        set_(p.statements).to(join(s1, "\n")),
        set_(p.next_sym).to(i),
        set_(p.expr).to(String("")),
    )
    # 3.b. If parent set, set statements to expr + statements
    yield rule(
        stmt,
        eq(p1.parent).to(p),
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
    program_add = eq(p).to(p1 + p2)

    # If the resulting expression is either of the inputs, then its an identifer if those are
    # Otherwise, if its not equal to either input, its not an identifier
    yield rule(program_add, eq(p.expr).to(p1.expr), eq(b).to(p1.is_identifer)).then(set_(p.is_identifer).to(b))
    yield rule(program_add, eq(p.expr).to(p2.expr), eq(b).to(p2.is_identifer)).then(set_(p.is_identifer).to(b))
    yield rule(program_add, ne(p.expr).to(p1.expr), ne(p.expr).to(p2.expr)).then(set_(p.is_identifer).to(Bool(False)))

    # Set parent of p1
    yield rule(program_add, p.compile(i)).then(
        set_(p1.parent).to(p),
    )

    # Compile p1, if p1 parent equal
    yield rule(program_add, p.compile(i), eq(p1.parent).to(p)).then(p1.compile(i))

    # Set parent of p2, once p1 compiled
    yield rule(program_add, p.compile(i), p1.next_sym).then(set_(p2.parent).to(p))

    # Compile p2, if p1 parent not equal, but p2 parent equal
    yield rule(program_add, p.compile(i), ne(p1.parent).to(p), eq(p2.parent).to(p)).then(p2.compile(i))

    # Compile p2, if p1 parent eqal
    yield rule(program_add, p.compile(i2), eq(p1.parent).to(p), eq(i).to(p1.next_sym), eq(p2.parent).to(p)).then(
        p2.compile(i)
    )

    # Set p expr to join of p1 and p2
    yield rule(
        program_add,
        eq(s1).to(p1.expr),
        eq(s2).to(p2.expr),
    ).then(
        set_(p.expr).to(join(s1, s2)),
    )
    # Set p statements to join and next sym to p2 if both parents set
    yield rule(
        program_add,
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
        program_add,
        p.compile(i),
        ne(p1.parent).to(p),
        ne(p2.parent).to(p),
    ).then(
        set_(p.statements).to(String("")),
        set_(p.next_sym).to(i),
    )
    # Set p statements to p1 and next sym to p1 if p1 parent set and p2 parent not set
    yield rule(
        program_add,
        eq(p1.parent).to(p),
        ne(p2.parent).to(p),
        eq(s1).to(p1.statements),
        eq(i).to(p1.next_sym),
    ).then(
        set_(p.statements).to(s1),
        set_(p.next_sym).to(i),
    )
    # Set p statements to p2 and next sym to p2 if p2 parent set and p1 parent not set
    yield rule(
        program_add,
        eq(p2.parent).to(p),
        ne(p1.parent).to(p),
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
    # expression as the gensym, and setting is_identifier to true
    program_assign = eq(p).to(p1.assign())
    # Set parent
    yield rule(program_assign, p.compile(i)).then(set_(p1.parent).to(p), set_(p.is_identifer).to(Bool(True)))
    # If parent set, compile the expression
    yield rule(program_assign, p.compile(i), eq(p1.parent).to(p)).then(p1.compile(i))

    # 1. If p1 is not an identifier, then we must create a new one

    # 1. a. If p1 parent is p, then use statements of p, next sym of p
    symbol = join(String("_"), i.to_string())
    yield rule(
        program_assign,
        eq(p1.parent).to(p),
        eq(s1).to(p1.statements),
        eq(i).to(p1.next_sym),
        eq(s2).to(p1.expr),
        eq(p1.is_identifer).to(Bool(False)),
    ).then(
        set_(p.statements).to(join(s1, symbol, " = ", s2, "\n")),
        set_(p.expr).to(symbol),
        set_(p.next_sym).to(i + 1),
    )
    # 1. b. If p1 parent is not p, then just use assign as statement, next sym of i
    yield rule(
        program_assign,
        ne(p1.parent).to(p),
        p.compile(i),
        eq(s2).to(p1.expr),
        eq(p1.is_identifer).to(Bool(False)),
    ).then(
        set_(p.statements).to(join(symbol, " = ", s2, "\n")),
        set_(p.expr).to(symbol),
        set_(p.next_sym).to(i + 1),
    )

    # 2. If p1 is an identifier, then program assign is a no-op

    # 1. a. If p1 parent is p, then use statements of p, next sym of p
    yield rule(
        program_assign,
        eq(p1.parent).to(p),
        eq(s1).to(p1.statements),
        eq(i).to(p1.next_sym),
        eq(s2).to(p1.expr),
        eq(p1.is_identifer).to(Bool(True)),
    ).then(
        set_(p.statements).to(s1),
        set_(p.expr).to(s2),
        set_(p.next_sym).to(i),
    )
    # 1. b. If p1 parent is not p, then just use assign as statement, next sym of i
    yield rule(
        program_assign,
        ne(p1.parent).to(p),
        p.compile(i),
        eq(s2).to(p1.expr),
        eq(p1.is_identifer).to(Bool(True)),
    ).then(
        set_(p.statements).to(String("")),
        set_(p.expr).to(s2),
        set_(p.next_sym).to(i),
    )

    ##
    # Function two
    ##

    # When compiling a function, the two args, p2 and p3, should get compiled when we compile p1, and should just be vars.
    fn_two = eq(p).to(p1.function_two(p2, p3, s1))
    # 1. Set parents of both args to p and compile them
    # Assumes that this if the first thing to compile, so no need to check, and assumes that compiling args doesn't result in any
    # change in the next sym
    yield rule(fn_two, p.compile(i)).then(
        set_(p2.parent).to(p),
        set_(p3.parent).to(p),
        set_(p1.parent).to(p),
        p2.compile(i),
        p3.compile(i),
        p1.compile(i),
        set_(p.is_identifer).to(Bool(True)),
    )
    # 2. Set statements to function body and the next sym to i
    yield rule(
        fn_two,
        p.compile(i),
        eq(s2).to(p1.expr),
        eq(s3).to(p1.statements),
        eq(s4).to(p2.expr),
        eq(s5).to(p3.expr),
    ).then(
        set_(p.statements).to(
            join("def ", s1, "(", s4, ", ", s5, "):\n    ", s3.replace("\n", "\n    "), "return ", s2, "\n")
        ),
        set_(p.next_sym).to(i),
        set_(p.expr).to(s1),
    )

    ##
    # Function three
    ##

    fn_three = eq(p).to(p1.function_three(p2, p3, p4, s1))
    yield rule(fn_three, p.compile(i)).then(
        set_(p2.parent).to(p),
        set_(p3.parent).to(p),
        set_(p1.parent).to(p),
        set_(p4.parent).to(p),
        p2.compile(i),
        p3.compile(i),
        p1.compile(i),
        p4.compile(i),
        set_(p.is_identifer).to(Bool(True)),
    )
    yield rule(
        fn_three,
        p.compile(i),
        eq(s2).to(p1.expr),
        eq(s3).to(p1.statements),
        eq(s4).to(p2.expr),
        eq(s5).to(p3.expr),
        eq(s6).to(p4.expr),
    ).then(
        set_(p.statements).to(
            join("def ", s1, "(", s4, ", ", s5, ", ", s6, "):\n    ", s3.replace("\n", "\n    "), "return ", s2, "\n")
        ),
        set_(p.next_sym).to(i),
        set_(p.expr).to(s1),
    )
