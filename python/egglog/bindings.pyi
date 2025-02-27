from datetime import timedelta
from pathlib import Path
from typing import TypeAlias

from typing_extensions import final

__all__ = [
    "ActionCommand",
    "AddRuleset",
    "Best",
    "BiRewriteCommand",
    "Bool",
    "Call",
    "Change",
    "Check",
    "Constructor",
    "Datatype",
    "Datatypes",
    "Delete",
    "EGraph",
    "EggSmolError",
    "EgglogSpan",
    "Eq",
    "Expr_",
    "Extract",
    "Fact",
    "Fail",
    "Float",
    "Function",
    "IdentSort",
    "Include",
    "Input",
    "Int",
    "Let",
    "Lit",
    "NewSort",
    "Output",
    "Panic",
    "PanicSpan",
    "Pop",
    "PrintFunction",
    "PrintOverallStatistics",
    "PrintSize",
    "Push",
    "PyObjectSort",
    "QueryExtract",
    "Relation",
    "Repeat",
    "Rewrite",
    "RewriteCommand",
    "Rule",
    "RuleCommand",
    "Run",
    "RunConfig",
    "RunReport",
    "RunSchedule",
    "RustSpan",
    "Saturate",
    "Schema",
    "Sequence",
    "SerializedEGraph",
    "Set",
    "SetOption",
    "Simplify",
    "Sort",
    "SrcFile",
    "String",
    "SubVariants",
    "Subsume",
    "TermApp",
    "TermDag",
    "TermLit",
    "TermVar",
    "Union",
    "Unit",
    "UnstableCombinedRuleset",
    "Var",
    "Variant",
    "Variants",
]

@final
class SerializedEGraph:
    def inline_leaves(self) -> None: ...
    def saturate_inline_leaves(self) -> None: ...
    def to_dot(self) -> str: ...
    def to_json(self) -> str: ...
    def map_ops(self, map: dict[str, str]) -> None: ...
    def split_classes(self, egraph: EGraph, ops: set[str]) -> None: ...

@final
class PyObjectSort:
    def __init__(self) -> None: ...
    def store(self, __o: object, /) -> _Expr: ...
    def load(self, __e: _Expr, /) -> object: ...

@final
class EGraph:
    def __init__(
        self,
        py_object_sort: PyObjectSort | None = None,
        /,
        *,
        fact_directory: str | Path | None = None,
        seminaive: bool = True,
        record: bool = False,
    ) -> None: ...
    def parse_program(self, __input: str, /, filename: str | None = None) -> list[_Command]: ...
    def commands(self) -> str | None: ...
    def run_program(self, *commands: _Command) -> list[str]: ...
    def extract_report(self) -> _ExtractReport | None: ...
    def run_report(self) -> RunReport | None: ...
    def serialize(
        self,
        root_eclasses: list[_Expr],
        *,
        max_functions: int | None = None,
        max_calls_per_function: int | None = None,
        include_temporary_functions: bool = False,
    ) -> SerializedEGraph: ...

@final
class EggSmolError(Exception):
    context: str

##
# Spans
##

@final
class PanicSpan:
    def __init__(self) -> None: ...

@final
class SrcFile:
    name: str | None
    contents: str
    def __init__(self, name: str | None, contents: str) -> None: ...

@final
class EgglogSpan:
    file: SrcFile
    i: int
    j: int
    def __init__(self, file: SrcFile, i: int, j: int) -> None: ...

@final
class RustSpan:
    file: str
    line: int
    column: int
    def __init__(self, file: str, line: int, column: int) -> None: ...

_Span: TypeAlias = PanicSpan | EgglogSpan | RustSpan

##
# Literals
##

@final
class Int:
    def __init__(self, value: int) -> None: ...
    value: int

@final
class Float:
    value: float
    def __init__(self, value: float) -> None: ...

@final
class String:
    def __init__(self, value: str) -> None: ...
    value: str

@final
class Unit:
    def __init__(self) -> None: ...

@final
class Bool:
    def __init__(self, b: bool) -> None: ...
    value: bool

_Literal: TypeAlias = Int | Float | String | Bool | Unit

##
# Expressions
##

@final
class Lit:
    def __init__(self, span: _Span, value: _Literal) -> None: ...
    span: _Span
    value: _Literal

@final
class Var:
    def __init__(self, span: _Span, name: str) -> None: ...
    span: _Span
    name: str

@final
class Call:
    def __init__(self, span: _Span, name: str, args: list[_Expr]) -> None: ...
    span: _Span
    name: str
    args: list[_Expr]

# Unions must be private becuase it is not actually exposed by the runtime library.
_Expr: TypeAlias = Lit | Var | Call

##
# Terms
##

@final
class TermLit:
    def __init__(self, value: _Literal) -> None: ...
    value: _Literal

@final
class TermVar:
    def __init__(self, name: str) -> None: ...
    name: str

@final
class TermApp:
    def __init__(self, name: str, args: list[int]) -> None: ...
    name: str
    args: list[int]

_Term: TypeAlias = TermLit | TermVar | TermApp

##
# Facts
##

@final
class Eq:
    def __init__(self, span: _Span, left: _Expr, right: _Expr) -> None: ...
    span: _Span
    left: _Expr
    right: _Expr

@final
class Fact:
    def __init__(self, expr: _Expr) -> None: ...
    expr: _Expr

_Fact: TypeAlias = Fact | Eq

##
# Change
##

@final
class Delete:
    def __init__(self) -> None: ...

@final
class Subsume:
    def __init__(self) -> None: ...

_Change: TypeAlias = Delete | Subsume

##
# Actions
##

@final
class Let:
    def __init__(self, span: _Span, lhs: str, rhs: _Expr) -> None: ...
    span: _Span
    lhs: str
    rhs: _Expr

@final
class Set:
    def __init__(self, span: _Span, lhs: str, args: list[_Expr], rhs: _Expr) -> None: ...
    span: _Span
    lhs: str
    args: list[_Expr]
    rhs: _Expr

@final
class Change:
    span: _Span
    change: _Change
    sym: str
    args: list[_Expr]
    def __init__(self, span: _Span, change: _Change, sym: str, args: list[_Expr]) -> None: ...

@final
class Union:
    def __init__(self, span: _Span, lhs: _Expr, rhs: _Expr) -> None: ...
    span: _Span
    lhs: _Expr
    rhs: _Expr

@final
class Panic:
    def __init__(self, span: _Span, msg: str) -> None: ...
    span: _Span
    msg: str

@final
class Expr_:  # noqa: N801
    def __init__(self, span: _Span, expr: _Expr) -> None: ...
    span: _Span
    expr: _Expr

@final
class Extract:
    def __init__(self, span: _Span, expr: _Expr, variants: _Expr) -> None: ...
    span: _Span
    expr: _Expr
    variants: _Expr

_Action: TypeAlias = Let | Set | Change | Union | Panic | Expr_ | Extract

##
# Other Structs
##

@final
class Variant:
    def __init__(self, span: _Span, name: str, types: list[str], cost: int | None = None) -> None: ...
    span: _Span
    name: str
    types: list[str]
    cost: int | None

@final
class Schema:
    input: list[str]
    output: str
    def __init__(self, input: list[str], output: str) -> None: ...

@final
class Rule:
    span: _Span
    head: list[_Action]
    body: list[_Fact]
    def __init__(self, span: _Span, head: list[_Action], body: list[_Fact]) -> None: ...

@final
class Rewrite:
    span: _Span
    lhs: _Expr
    rhs: _Expr
    conditions: list[_Fact]

    def __init__(self, span: _Span, lhs: _Expr, rhs: _Expr, conditions: list[_Fact] = []) -> None: ...

@final
class RunConfig:
    ruleset: str
    until: list[_Fact] | None
    def __init__(self, ruleset: str, until: list[_Fact] | None = None) -> None: ...

@final
class IdentSort:
    ident: str
    sort: str
    def __init__(self, ident: str, sort: str) -> None: ...

@final
class RunReport:
    updated: bool
    search_time_per_rule: dict[str, timedelta]
    apply_time_per_rule: dict[str, timedelta]
    search_time_per_ruleset: dict[str, timedelta]
    apply_time_per_ruleset: dict[str, timedelta]
    rebuild_time_per_ruleset: dict[str, timedelta]
    num_matches_per_rule: dict[str, int]

    def __init__(
        self,
        updated: bool,
        search_time_per_rule: dict[str, timedelta],
        apply_time_per_rule: dict[str, timedelta],
        search_time_per_ruleset: dict[str, timedelta],
        apply_time_per_ruleset: dict[str, timedelta],
        rebuild_time_per_ruleset: dict[str, timedelta],
        num_matches_per_rule: dict[str, int],
    ) -> None: ...

@final
class Variants:
    termdag: TermDag
    terms: list[_Term]
    def __init__(self, termdag: TermDag, terms: list[_Term]) -> None: ...

@final
class Best:
    termdag: TermDag
    cost: int
    term: _Term
    def __init__(self, termdag: TermDag, cost: int, term: _Term) -> None: ...

_ExtractReport: TypeAlias = Variants | Best

##
# Schedules
##

@final
class Saturate:
    span: _Span
    schedule: _Schedule
    def __init__(self, span: _Span, schedule: _Schedule) -> None: ...

@final
class Repeat:
    span: _Span
    length: int
    schedule: _Schedule
    def __init__(self, span: _Span, length: int, schedule: _Schedule) -> None: ...

@final
class Run:
    span: _Span
    config: RunConfig
    def __init__(self, span: _Span, config: RunConfig) -> None: ...

@final
class Sequence:
    span: _Span
    schedules: list[_Schedule]
    def __init__(self, span: _Span, schedules: list[_Schedule]) -> None: ...

_Schedule: TypeAlias = Saturate | Repeat | Run | Sequence

##
# Subdatatypes
##

@final
class SubVariants:
    def __init__(self, variants: list[Variant]) -> None: ...
    variants: list[Variant]

@final
class NewSort:
    def __init__(self, name: str, args: list[_Expr]) -> None: ...
    name: str
    args: list[_Expr]

_Subdatatypes: TypeAlias = SubVariants | NewSort

##
# Commands
##

@final
class SetOption:
    name: str
    value: _Expr
    def __init__(self, name: str, value: _Expr) -> None: ...

@final
class Datatype:
    span: _Span
    name: str
    variants: list[Variant]
    def __init__(self, span: _Span, name: str, variants: list[Variant]) -> None: ...

@final
class Datatypes:
    span: _Span
    datatypes: list[tuple[_Span, str, _Subdatatypes]]
    def __init__(self, span: _Span, datatypes: list[tuple[_Span, str, _Subdatatypes]]) -> None: ...

@final
class Sort:
    span: _Span
    name: str
    presort_and_args: tuple[str, list[_Expr]] | None
    def __init__(self, span: _Span, name: str, presort_and_args: tuple[str, list[_Expr]] | None = None) -> None: ...

@final
class Function:
    span: _Span
    name: str
    schema: Schema
    merge: _Expr | None
    def __init__(self, span: _Span, name: str, schema: Schema, merge: _Expr | None) -> None: ...

@final
class AddRuleset:
    name: str
    def __init__(self, name: str) -> None: ...

@final
class RuleCommand:
    name: str
    ruleset: str
    rule: Rule
    def __init__(self, name: str, ruleset: str, rule: Rule) -> None: ...

@final
class RewriteCommand:
    # TODO: Rename to ruleset
    name: str
    rewrite: Rewrite
    subsume: bool
    def __init__(self, name: str, rewrite: Rewrite, subsume: bool) -> None: ...

@final
class BiRewriteCommand:
    # TODO: Rename to ruleset
    name: str
    rewrite: Rewrite
    def __init__(self, name: str, rewrite: Rewrite) -> None: ...

@final
class ActionCommand:
    action: _Action
    def __init__(self, action: _Action) -> None: ...

@final
class RunSchedule:
    schedule: _Schedule
    def __init__(self, schedule: _Schedule) -> None: ...

@final
class Simplify:
    span: _Span
    expr: _Expr
    schedule: _Schedule
    def __init__(self, span: _Span, expr: _Expr, schedule: _Schedule) -> None: ...

@final
class QueryExtract:
    span: _Span
    variants: int
    expr: _Expr
    def __init__(self, span: _Span, variants: int, expr: _Expr) -> None: ...

@final
class Check:
    span: _Span
    facts: list[_Fact]
    def __init__(self, span: _Span, facts: list[_Fact]) -> None: ...

@final
class PrintFunction:
    span: _Span
    name: str
    length: int
    def __init__(self, span: _Span, name: str, length: int) -> None: ...

@final
class PrintSize:
    span: _Span
    name: str | None
    def __init__(self, span: _Span, name: str | None) -> None: ...

@final
class Output:
    span: _Span
    file: str
    exprs: list[_Expr]
    def __init__(self, span: _Span, file: str, exprs: list[_Expr]) -> None: ...

@final
class Input:
    span: _Span
    name: str
    file: str
    def __init__(self, span: _Span, name: str, file: str) -> None: ...

@final
class Push:
    length: int
    def __init__(self, length: int) -> None: ...

@final
class Pop:
    span: _Span
    length: int
    def __init__(self, span: _Span, length: int) -> None: ...

@final
class Fail:
    span: _Span
    command: _Command
    def __init__(self, span: _Span, command: _Command) -> None: ...

@final
class Include:
    span: _Span
    path: str
    def __init__(self, span: _Span, path: str) -> None: ...

@final
class Relation:
    span: _Span
    name: str
    inputs: list[str]

    def __init__(self, span: _Span, name: str, inputs: list[str]) -> None: ...

@final
class Constructor:
    span: _Span
    name: str
    schema: Schema
    cost: int | None
    unextractable: bool
    def __init__(self, span: _Span, name: str, schema: Schema, cost: int | None, unextractable: bool) -> None: ...

@final
class PrintOverallStatistics:
    def __init__(self) -> None: ...

@final
class UnstableCombinedRuleset:
    name: str
    rulesets: list[str]
    def __init__(self, name: str, rulesets: list[str]) -> None: ...

_Command: TypeAlias = (
    SetOption
    | Datatype
    | Datatypes
    | Sort
    | Function
    | AddRuleset
    | RuleCommand
    | RewriteCommand
    | BiRewriteCommand
    | ActionCommand
    | RunSchedule
    | Simplify
    | QueryExtract
    | Check
    | PrintFunction
    | PrintSize
    | Output
    | Input
    | Push
    | Pop
    | Fail
    | Include
    | Relation
    | PrintOverallStatistics
    | UnstableCombinedRuleset
    | Constructor
)

##
# TermDag
##

@final
class TermDag:
    def __init__(self) -> None: ...
    def size(self) -> int: ...
    def lookup(self, node: _Term) -> int: ...
    def get(self, id: int) -> _Term: ...
    def app(self, sym: str, children: list[int]) -> _Term: ...
    def lit(self, lit: _Literal) -> _Term: ...
    def var(self, sym: str) -> _Term: ...
    def expr_to_term(self, expr: _Expr) -> _Term: ...
    def term_to_expr(self, term: _Term, span: _Span) -> _Expr: ...
    def to_string(self, term: _Term) -> str: ...
