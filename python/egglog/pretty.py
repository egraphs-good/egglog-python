"""
Pretty printing for declerations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias, assert_never

import black

from .declarations import *

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "pretty_decl",
    "pretty_callable_ref",
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

AllDecls: TypeAlias = RulesetDecl | CommandDecl | ActionDecl | FactDecl | ExprDecl | ScheduleDecl


def pretty_decl(
    decls: Declarations, decl: AllDecls, *, wrapping_fn: str | None = None, ruleset_name: str | None = None
) -> str:
    """
    Pretty print a decleration.

    This will use re-format the result and put the expression on the last line, preceeded by the statements.
    """
    traverse = TraverseContext()
    traverse(decl, toplevel=True)
    pretty = traverse.pretty(decls)
    expr = pretty(decl, ruleset_name=ruleset_name)
    if wrapping_fn:
        expr = f"{wrapping_fn}({expr})"
    program = "\n".join([*pretty.statements, expr])
    try:
        # TODO: Try replacing with ruff for speed
        # https://github.com/amyreese/ruff-api
        return black.format_str(program, mode=BLACK_MODE).strip()
    except black.parsing.InvalidInput:
        return program


def pretty_callable_ref(
    decls: Declarations,
    ref: CallableRef,
    first_arg: ExprDecl | None = None,
    bound_tp_params: tuple[JustTypeRef, ...] | None = None,
) -> str:
    """
    Pretty print a callable reference, using a dummy value for
    the args if the function is not in the form `f(x, ...)`.

    To be used in the visualization.
    """
    # Pass in three dummy args, which are the max used for any operation that
    # is not a generic function call
    args: list[ExprDecl] = [LitDecl(ARG_STR)] * 3
    if first_arg:
        args.insert(0, first_arg)
    res = PrettyContext(decls, defaultdict(lambda: 0))._call_inner(
        ref, args, bound_tp_params=bound_tp_params, parens=False
    )
    # Either returns a function or a function with args. If args are provided, they would just be called,
    # on the function, so return them, because they are dummies
    return res[0] if isinstance(res, tuple) else res


@dataclass
class TraverseContext:
    """
    State for traversing expressions (or declerations that contain expressions), so we can know how many parents each
    expression has.
    """

    # All expressions we have seen (incremented the parent counts of all children)
    _seen: set[AllDecls] = field(default_factory=set)
    # The number of parents for each expressions
    parents: Counter[AllDecls] = field(default_factory=Counter)

    def pretty(self, decls: Declarations) -> PrettyContext:
        """
        Create a pretty context from the state of this traverse context.
        """
        return PrettyContext(decls, self.parents)

    def __call__(self, decl: AllDecls, toplevel: bool = False) -> None:  # noqa: C901
        if not toplevel:
            self.parents[decl] += 1
        if decl in self._seen:
            return
        match decl:
            case RewriteDecl(lhs, rhs, conditions) | BiRewriteDecl(lhs, rhs, conditions):
                self(lhs)
                self(rhs)
                for cond in conditions:
                    self(cond)
            case RuleDecl(head, body, _):
                for action in head:
                    self(action)
                for fact in body:
                    self(fact)
            case SetDecl(lhs, rhs) | UnionDecl(lhs, rhs):
                self(lhs)
                self(rhs)
            case (
                LetDecl(_, d)
                | ExprActionDecl(d)
                | DeleteDecl(d)
                | ExprFactDecl(d)
                | SaturateDecl(d)
                | RepeatDecl(d, _)
                | ActionCommandDecl(d)
            ):
                self(d)
            case PanicDecl(_) | VarDecl(_) | LitDecl(_) | PyObjectDecl(_):
                pass
            case EqDecl(decls) | SequenceDecl(decls) | RulesetDecl(decls):
                for de in decls:
                    self(de)
            case CallDecl(_, exprs, _):
                for e in exprs:
                    self(e.expr)
            case RunDecl(_, until):
                if until:
                    for f in until:
                        self(f)
            case _:
                assert_never(decl)

        self._seen.add(decl)


@dataclass
class PrettyContext:
    """

    We need to build up a list of all the expressions we are pretty printing, so that we can see who has parents and who is mutated
    and create temp variables for them.

    """

    decls: Declarations
    parents: Mapping[AllDecls, int]

    # All the expressions we have saved as names
    names: dict[AllDecls, str] = field(default_factory=dict)
    # A list of statements assigning variables or calling destructive ops
    statements: list[str] = field(default_factory=list)
    # Mapping of type to the number of times we have generated a name for that type, used to generate unique names
    _gen_name_types: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def __call__(
        self, decl: AllDecls, *, unwrap_lit: bool = False, parens: bool = False, ruleset_name: str | None = None
    ) -> str:
        if decl in self.names:
            return self.names[decl]
        expr, tp_name = self.uncached(decl, unwrap_lit=unwrap_lit, parens=parens, ruleset_name=ruleset_name)
        # We use a heuristic to decide whether to name this sub-expression as a variable
        # The rough goal is to reduce the number of newlines, given our line length of ~180
        # We determine it's worth making a new line for this expression if the total characters
        # it would take up is > than some constant (~ line length).
        line_diff: int = len(expr) - LINE_DIFFERENCE
        n_parents = self.parents[decl]
        if n_parents > 1 and n_parents * line_diff > MAX_LINE_LENGTH:
            self.names[decl] = expr_name = self._name_expr(tp_name, expr, copy_identifier=False)
            return expr_name
        return expr

    def uncached(self, decl: AllDecls, *, unwrap_lit: bool, parens: bool, ruleset_name: str | None) -> tuple[str, str]:  # noqa: PLR0911
        match decl:
            case LitDecl(value):
                match value:
                    case None:
                        return "Unit()", "Unit"
                    case bool(b):
                        return str(b) if unwrap_lit else f"Bool({b})", "Bool"
                    case int(i):
                        return str(i) if unwrap_lit else f"i64({i})", "i64"
                    case float(f):
                        return str(f) if unwrap_lit else f"f64({f})", "f64"
                    case str(s):
                        return repr(s) if unwrap_lit else f"String({s!r})", "String"
                assert_never(value)
            case VarDecl(name):
                return name, name
            case CallDecl(_, _, _):
                return self._call(decl, parens)
            case PyObjectDecl(value):
                return repr(value) if unwrap_lit else f"PyObject({value!r})", "PyObject"
            case ActionCommandDecl(action):
                return self(action), "action"
            case RewriteDecl(lhs, rhs, conditions) | BiRewriteDecl(lhs, rhs, conditions):
                args = ", ".join(map(self, (rhs, *conditions)))
                fn = "rewrite" if isinstance(decl, RewriteDecl) else "birewrite"
                return f"{fn}({self(lhs)}).to({args})", "rewrite"
            case RuleDecl(head, body, name):
                l = ", ".join(map(self, body))
                if name:
                    l += f", name={name}"
                r = ", ".join(map(self, head))
                return f"rule({l}).then({r})", "rule"
            case SetDecl(lhs, rhs):
                return f"set_({self(lhs)}).to({self(rhs)})", "action"
            case UnionDecl(lhs, rhs):
                return f"union({self(lhs)}).with_({self(rhs)})", "action"
            case LetDecl(name, expr):
                return f"let({name!r}, {self(expr)})", "action"
            case ExprActionDecl(expr):
                return self(expr), "action"
            case ExprFactDecl(expr):
                return self(expr), "fact"
            case DeleteDecl(expr):
                return f"delete({self(expr)})", "action"
            case PanicDecl(s):
                return f"panic({s!r})", "action"
            case EqDecl(exprs):
                first, *rest = exprs
                return f"eq({self(first)}).to({', '.join(map(self, rest))})", "fact"
            case RulesetDecl(rules):
                if ruleset_name:
                    return f"ruleset(name={ruleset_name!r})", f"ruleset_{ruleset_name}"
                args = ", ".join(map(self, rules))
                return f"ruleset({args})", "ruleset"
            case SaturateDecl(schedule):
                return f"{self(schedule, parens=True)}.saturate()", "schedule"
            case RepeatDecl(schedule, times):
                return f"{self(schedule, parens=True)} * {times}", "schedule"
            case SequenceDecl(schedules):
                if len(schedules) == 2:
                    return f"{self(schedules[0], parens=True)} + {self(schedules[1], parens=True)}", "schedule"
                args = ", ".join(map(self, schedules))
                return f"seq({args})", "schedule"
            case RunDecl(ruleset_name, until):
                ruleset = self.decls._rulesets[ruleset_name]
                ruleset_str = self(ruleset, ruleset_name=ruleset_name)
                if not until:
                    return ruleset_str, "schedule"
                args = ", ".join(map(self, until))
                return f"run({ruleset_str}, {args})", "schedule"
        assert_never(decl)

    def _call(
        self,
        decl: CallDecl,
        parens: bool,
    ) -> tuple[str, str]:
        """
        Pretty print the call. Also returns if it was saved as a name.

        :param parens: If true, wrap the call in parens if it is a binary method call.
        """
        args = [a.expr for a in decl.args]
        ref = decl.callable
        # Special case !=
        if decl.callable == FunctionRef("!="):
            l, r = self(args[0]), self(args[1])
            return f"ne({l}).to({r})", "Unit"
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

        tp_name = function_decl.semantic_return_type.name
        if function_decl.mutates:
            first_arg = args[0]
            expr_str = self(first_arg)
            # copy an identifier expression iff it has multiple parents (b/c then we can't mutate it directly)
            has_multiple_parents = self.parents[first_arg] > 1
            self.names[decl] = expr_name = self._name_expr(tp_name, expr_str, copy_identifier=has_multiple_parents)
            # Set the first arg to be the name of the mutated arg and return the name
            args[0] = VarDecl(expr_name)
        else:
            expr_name = None
        res = self._call_inner(ref, args, decl.bound_tp_params, parens)
        expr = (
            f"{res[0]}({', '.join(self(a, parens=False, unwrap_lit=True) for a in res[1])})"
            if isinstance(res, tuple)
            else res
        )
        # If we have a name, then we mutated
        if expr_name:
            self.statements.append(expr)
            return expr_name, tp_name
        return expr, tp_name

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
                fn_str = str(JustTypeRef(class_name, bound_tp_params or ()))
                if method_name != "__init__":
                    fn_str += f".{method_name}"
                return fn_str, args
            case MethodRef(_class_name, method_name):
                slf, *args = args
                slf = self(slf, parens=True)
                match method_name:
                    case _ if method_name in UNARY_METHODS:
                        expr = f"{UNARY_METHODS[method_name]}{slf}"
                        return f"({expr})" if parens else expr
                    case _ if method_name in BINARY_METHODS:
                        expr = f"{slf} {BINARY_METHODS[method_name]} {self(args[0], parens=True, unwrap_lit=True)}"
                        return f"({expr})" if parens else expr
                    case "__getitem__":
                        return f"{slf}[{self(args[0], unwrap_lit=True)}]"
                    case "__call__":
                        return slf, args
                    case "__delitem__":
                        return f"del {slf}[{self(args[0], unwrap_lit=True)}]"
                    case "__setitem__":
                        return f"{slf}[{self(args[0], unwrap_lit=True)}] = {self(args[1], unwrap_lit=True)}"
                    case _:
                        return f"{slf}.{method_name}", args
            case ConstantRef(name):
                return name
            case ClassVariableRef(class_name, variable_name):
                return f"{class_name}.{variable_name}"
            case PropertyRef(_class_name, property_name):
                return f"{self(args[0], parens=True)}.{property_name}"
        assert_never(ref)

    def _generate_name(self, typ: str) -> str:
        self._gen_name_types[typ] += 1
        return f"_{typ}_{self._gen_name_types[typ]}"

    def _name_expr(self, tp_name: str, expr_str: str, copy_identifier: bool) -> str:
        # tp_name =
        # If the thing we are naming is already a variable, we don't need to name it
        if expr_str.isidentifier():
            if copy_identifier:
                name = self._generate_name(tp_name)
                self.statements.append(f"{name} = copy({expr_str})")
            else:
                name = expr_str
        else:
            name = self._generate_name(tp_name)
            self.statements.append(f"{name} = {expr_str}")
        return name


def _plot_line_length(expr: object):  # pragma: no cover
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
