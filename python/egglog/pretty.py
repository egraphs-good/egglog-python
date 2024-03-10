"""
Pretty printing for declerations
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import assert_never

import black

from .declarations import *

__all__ = [
    "pretty_command",
    "pretty_action",
    "pretty_fact",
    "pretty_ruleset",
    "pretty_expr",
    "pretty_callable",
    "pretty_type",
    "pretty_type_or_var",
    "BINARY_METHODS",
    "UNARY_METHODS",
]
MAX_LINE_LENGTH = 110
LINE_DIFFERENCE = 10
BLACK_MODE = black.Mode(line_length=180)

# Use this special character in place of the args, so that if the args are inlined
# in the viz, they will replace it
ARG_STR = "·"

# Special methods which we might want to use as functions
# Mapping to the operator they represent for pretty printing them
# https://docs.python.org/3/reference/datamodel.html
BINARY_METHODS = {
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
    # Numeric
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__matmul__": "@",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    # TODO: Support divmod, with tuple return value
    # "__divmod__": "divmod",
    # TODO: Three arg power
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
}


UNARY_METHODS = {
    "__pos__": "+",
    "__neg__": "-",
    "__invert__": "~",
}


def pretty_command(decls: Declarations, command: CommandDecl) -> str:
    """
    Pretty print a command.
    """


def pretty_action(decls: Declarations, action: ActionDecl) -> str:
    """
    Pretty print an action.
    """


def pretty_fact(decls: Declarations, fact: FactDecl) -> str:
    """
    Pretty print a fact.
    """


def pretty_ruleset(decls: Declarations, ruleset: RulesetDecl) -> str:
    """
    Pretty print a ruleset.
    """


def pretty_expr(decls: Declarations, expr: ExprDecl, wrapping_fn: str | None) -> str:
    """
    Pretty print an expression.
    """
    context = PrettyContext(decls)
    context.traverse_for_parents(expr)
    pretty_expr = context(expr, parens=False)
    pretty_statements = context.render(pretty_expr)
    try:
        # TODO: Try replacing with ruff for speed
        # https://github.com/amyreese/ruff-api
        return black.format_str(pretty_statements, mode=BLACK_MODE).strip()
    except black.parsing.InvalidInput:
        return pretty_statements


def pretty_callable(decls: Declarations, ref: CallableRef) -> str:
    """
    Pretty print a callable reference, using a dummy value for
    the args if the function is not in the form `f(x, ...)`.

    To be used in the visualization.
    """
    res = PrettyContext(decls)._call_inner(
        ref,
        # Pass in three dummy args, which are the max used for any operation that
        # is not a generic function call
        [LitDecl(ARG_STR)] * 3,
        None,
        False,
    )
    return res[0] if isinstance(res, tuple) else res


def pretty_type(type_ref: JustTypeRef) -> str:
    """
    Pretty print a type reference.
    """
    if type_ref.args:
        return f"{type_ref.name}[{', '.join(map(pretty_type, type_ref.args))}]"
    return type_ref.name


def pretty_type_or_var(type_ref: TypeOrVarRef) -> str:
    match type_ref:
        case ClassTypeVarRef(name):
            return name
        case TypeRefWithVars(name, args):
            if args:
                return f"{name}[{', '.join(map(pretty_type_or_var, args))}]"
            return name
        case _:
            assert_never(type_ref)


@dataclass
class PrettyContext:
    decls: Declarations
    # List of statements of "context" setting variable for the expr
    statements: list[str] = field(default_factory=list)

    names: dict[ExprDecl, str] = field(default_factory=dict)
    parents: dict[ExprDecl, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    _traversed_exprs: set[ExprDecl] = field(default_factory=set)

    # Mapping of type to the number of times we have generated a name for that type, used to generate unique names
    _gen_name_types: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def __call__(self, expr: ExprDecl, *, unwrap_lit: bool = True, parens: bool = False) -> str:  # noqa: PLR0911
        if expr in self.names:
            return self.names[expr]
        match expr:
            case LitDecl(value):
                match value:
                    case None:
                        return "Unit()"
                    case bool(b):
                        return str(b) if unwrap_lit else f"Bool({b})"
                    case int(i):
                        return str(i) if unwrap_lit else f"i64({i})"
                    case float(f):
                        return str(f) if unwrap_lit else f"f64({f})"
                    case str(s):
                        return repr(s) if unwrap_lit else f"String({s!r})"
                    case _:
                        assert_never(value)
                        return None
            case VarDecl(name):
                return name
            case CallDecl(ref, args, bound_tp_params):
                s, saved = self._call(ref, [a.expr for a in args], bound_tp_params, parens, self.parents[expr])
                if saved:
                    self.names[expr] = s
                return s
            case PyObjectDecl(value):
                return repr(value)
            case _:
                assert_never(expr)

    def _call(
        self,
        ref: CallableRef,
        args: list[ExprDecl],
        bound_tp_params: tuple[JustTypeRef, ...] | None,
        parens: bool,
        n_parents: int,
    ) -> tuple[str, bool]:
        """
        Pretty print the call. Also returns if it was saved as a name.

        :param parens: If true, wrap the call in parens if it is a binary method call.
        """
        # Special case !=
        if ref == FunctionRef("!="):
            return f"ne({self(args[0], unwrap_lit=False)}).to({self(args[1], unwrap_lit=False)})", False
        function_decl = self.decls.get_callable_decl(ref).to_function_decl()
        # Determine how many of the last arguments are defaults, by iterating from the end and comparing the arg with the default
        n_defaults = 0
        for arg, default in zip(
            reversed(args), reversed(function_decl.arg_defaults), strict=not function_decl.var_arg_type
        ):
            if arg != default:
                break
            n_defaults += 1
        if n_defaults:
            args = args[:-n_defaults]
        if function_decl.mutates:
            first_arg = args[0]
            expr_str = self(first_arg)
            # copy an identifer expression iff it has multiple parents (b/c then we can't mutate it directly)
            has_multiple_parents = self.parents[first_arg] > 1
            expr_name = self.name_expr(
                function_decl.semantic_return_type, expr_str, copy_identifier=has_multiple_parents
            )
            # Set the first arg to be the name of the mutated arg and return the name
            args[0] = VarDecl(expr_name)
        else:
            expr_name = None
        res = self._call_inner(ref, args, bound_tp_params, parens)
        expr = f"{res[0]}({', '.join(self(a, parens=False) for a in res[1])})" if isinstance(res, tuple) else res
        del res
        # If we have a name, then we mutated
        if expr_name:
            self.statements.append(expr)
            return expr_name, True

        # We use a heuristic to decide whether to name this sub-expression as a variable
        # The rough goal is to reduce the number of newlines, given our line length of ~180
        # We determine it's worth making a new line for this expression if the total characters
        # it would take up is > than some constant (~ line length).
        line_diff: int = len(expr) - LINE_DIFFERENCE
        if n_parents > 1 and n_parents * line_diff > MAX_LINE_LENGTH:
            return self.name_expr(function_decl.semantic_return_type, expr, copy_identifier=False), True
        return expr, False

    def _call_inner(  # noqa: PLR0911
        self, ref: CallableRef, args: list[ExprDecl], bound_tp_params: tuple[JustTypeRef, ...] | None, parens: bool
    ) -> tuple[str, list[ExprDecl]] | str:
        """
        Pretty print the call, returning either the full function call or a tuple of the function and the args.
        """
        match ref:
            case FunctionRef(name):
                return name, args
            case ClassMethodRef(class_name, method_name):
                fn_str = pretty_type(JustTypeRef(class_name, bound_tp_params or ()))
                if method_name != "__init__":
                    fn_str += f".{method_name}"
                return fn_str, args
            case MethodRef(_class_name, method_name):
                slf, *args = args
                slf = self(slf, unwrap_lit=False, parens=True)
                match method_name:
                    case _ if method_name in UNARY_METHODS:
                        expr = f"{UNARY_METHODS[method_name]}{slf}"
                        return f"({expr})" if parens else expr
                    case _ if method_name in BINARY_METHODS:
                        expr = f"{slf} {BINARY_METHODS[method_name]} {self(args[0], parens=True)}"
                        return f"({expr})" if parens else expr
                    case "__getitem__":
                        return f"{slf}[{self(args[0])}]"
                    case "__call__":
                        return slf, args
                    case "__delitem__":
                        return f"del {slf}[{self(args[0])}]"
                    case "__setitem__":
                        return f"{slf}[{self(args[0])}] = {self(args[1])}"
                    case _:
                        return f"{slf}.{method_name}", args
            case ConstantRef(name):
                return name
            case ClassVariableRef(class_name, variable_name):
                return f"{class_name}.{variable_name}"
            case PropertyRef(_class_name, property_name):
                return f"{self(args[0], parens=True, unwrap_lit=False)}.{property_name}"
            case _:
                assert_never(ref)

    def generate_name(self, typ: str) -> str:
        self._gen_name_types[typ] += 1
        return f"_{typ}_{self._gen_name_types[typ]}"

    def name_expr(self, expr_type: TypeOrVarRef, expr_str: str, copy_identifier: bool) -> str:
        tp_name = expr_type.to_just().name
        # If the thing we are naming is already a variable, we don't need to name it
        if expr_str.isidentifier():
            if copy_identifier:
                name = self.generate_name(tp_name)
                self.statements.append(f"{name} = copy({expr_str})")
            else:
                name = expr_str
        else:
            name = self.generate_name(tp_name)
            self.statements.append(f"{name} = {expr_str}")
        return name

    def render(self, expr: str) -> str:
        return "\n".join([*self.statements, expr])

    def traverse_for_parents(self, expr: ExprDecl) -> None:
        if expr in self._traversed_exprs:
            return
        self._traversed_exprs.add(expr)
        if isinstance(expr, CallDecl):
            for arg in set(expr.args):
                self.parents[arg.expr] += 1
                self.traverse_for_parents(arg.expr)


def _plot_line_length(expr: object):
    """
    Plots the number of line lengths based on different max lengths
    """
    global MAX_LINE_LENGTH, LINE_DIFFERENCE
    import altair as alt
    import pandas as pd

    sizes = []
    for line_length in range(40, 180, 10):
        MAX_LINE_LENGTH = line_length
        for diff in range(0, 40, 5):
            LINE_DIFFERENCE = diff
            new_l = len(str(expr).split())
            sizes.append((line_length, diff, new_l))

    df = pd.DataFrame(sizes, columns=["MAX_LINE_LENGTH", "LENGTH_DIFFERENCE", "n"])  # noqa: PD901

    return alt.Chart(df).mark_rect().encode(x="MAX_LINE_LENGTH:O", y="LENGTH_DIFFERENCE:O", color="n:Q")
