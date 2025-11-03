from collections.abc import Callable
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, final

__all__ = [
    "ActionCommand",
    "AddRuleset",
    "BiRewriteCommand",
    "Bool",
    "CSVPrintFunctionMode",
    "Call",
    "Change",
    "Check",
    "Constructor",
    "CostModel",
    "Datatype",
    "Datatypes",
    "DefaultPrintFunctionMode",
    "Delete",
    "EGraph",
    "EggSmolError",
    "EgglogSpan",
    "Eq",
    "Expr_",
    "Extract",
    "ExtractBest",
    "ExtractVariants",
    "Extractor",
    "Fact",
    "Fail",
    "Float",
    "Function",
    "FunctionCommand",
    "IdentSort",
    "Include",
    "Input",
    "Int",
    "Let",
    "Lit",
    "NewSort",
    "Output",
    "OverallStatistics",
    "Panic",
    "PanicSpan",
    "Pop",
    "PrintAllFunctionsSize",
    "PrintFunction",
    "PrintFunctionOutput",
    "PrintFunctionSize",
    "PrintOverallStatistics",
    "PrintSize",
    "Push",
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
    "RunScheduleOutput",
    "RustSpan",
    "Saturate",
    "Schema",
    "Sequence",
    "SerializedEGraph",
    "Set",
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
    "UserDefined",
    "UserDefinedCommandOutput",
    "UserDefinedOutput",
    "Value",
    "Var",
    "Variant",
]

@final
class SerializedEGraph:
    @property
    def truncated_functions(self) -> list[str]: ...
    @property
    def discarded_functions(self) -> list[str]: ...
    def inline_leaves(self) -> None: ...
    def saturate_inline_leaves(self) -> None: ...
    def to_dot(self) -> str: ...
    def to_json(self) -> str: ...
    def map_ops(self, map: dict[str, str]) -> None: ...
    def split_classes(self, egraph: EGraph, ops: set[str]) -> None: ...

@final
class EGraph:
    def __new__(
        cls, *, fact_directory: str | Path | None = None, seminaive: bool = True, record: bool = False
    ) -> EGraph: ...
    def parse_program(self, __input: str, /, filename: str | None = None) -> list[_Command]: ...
    def commands(self) -> str | None: ...
    def run_program(self, *commands: _Command) -> list[_CommandOutput]: ...
    def serialize(
        self,
        root_eclasses: list[_Expr],
        *,
        max_functions: int | None = None,
        max_calls_per_function: int | None = None,
        include_temporary_functions: bool = False,
    ) -> SerializedEGraph: ...
    def lookup_function(self, name: str, key: list[Value]) -> Value | None: ...
    def eval_expr(self, expr: _Expr) -> tuple[str, Value]: ...
    def value_to_i64(self, v: Value) -> int: ...
    def value_to_f64(self, v: Value) -> float: ...
    def value_to_string(self, v: Value) -> str: ...
    def value_to_bool(self, v: Value) -> bool: ...
    def value_to_rational(self, v: Value) -> Fraction: ...
    def value_to_bigint(self, v: Value) -> int: ...
    def value_to_bigrat(self, v: Value) -> Fraction: ...
    def value_to_pyobject(self, v: Value) -> object: ...
    def value_to_map(self, v: Value) -> dict[Value, Value]: ...
    def value_to_multiset(self, v: Value) -> list[Value]: ...
    def value_to_vec(self, v: Value) -> list[Value]: ...
    def value_to_function(self, v: Value) -> tuple[str, list[Value]]: ...
    def value_to_set(self, v: Value) -> set[Value]: ...
    # def dynamic_cost_model_enode_cost(self, func: str, args: list[Value]) -> int: ...

@final
class Value:
    def __hash__(self) -> int: ...
    def __eq__(self, value: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...

@final
class EggSmolError(Exception):
    context: str
    def __new__(cls, context: str) -> EggSmolError: ...
    def __init__(self, /, *args: Any, **kwargs: Any) -> None: ...

##
# Spans
##

@final
class PanicSpan: ...

@final
class SrcFile:
    name: str | None
    contents: str
    def __new__(cls, name: str | None, contents: str) -> SrcFile: ...

@final
class EgglogSpan:
    file: SrcFile
    i: int
    j: int
    def __new__(cls, file: SrcFile, i: int, j: int) -> EgglogSpan: ...

@final
class RustSpan:
    file: str
    line: int
    column: int
    def __new__(cls, file: str, line: int, column: int) -> RustSpan: ...

_Span: TypeAlias = PanicSpan | EgglogSpan | RustSpan

##
# Literals
##

@final
class Int:
    value: int
    def __new__(cls, value: int) -> Int: ...

@final
class Float:
    value: float
    def __new__(cls, value: float) -> Float: ...

@final
class String:
    value: str
    def __new__(cls, value: str) -> String: ...

@final
class Unit: ...

@final
class Bool:
    value: bool
    def __new__(cls, value: bool) -> Bool: ...

_Literal: TypeAlias = Int | Float | String | Bool | Unit

##
# Expressions
##

@final
class Lit:
    span: _Span
    value: _Literal
    def __new__(cls, span: _Span, value: _Literal) -> Lit: ...

@final
class Var:
    span: _Span
    name: str
    def __new__(cls, span: _Span, name: str) -> Var: ...

@final
class Call:
    span: _Span
    name: str
    args: list[_Expr]
    def __new__(cls, span: _Span, name: str, args: list[_Expr]) -> Call: ...

# Unions must be private becuase it is not actually exposed by the runtime library.
_Expr: TypeAlias = Lit | Var | Call

##
# Terms
##

@final
class TermLit:
    value: _Literal
    def __new__(cls, value: _Literal) -> TermLit: ...

@final
class TermVar:
    name: str
    def __new__(cls, name: str) -> TermVar: ...

@final
class TermApp:
    name: str
    args: list[int]
    def __new__(cls, name: str, args: list[int]) -> TermApp: ...

_Term: TypeAlias = TermLit | TermVar | TermApp

##
# Facts
##

@final
class Eq:
    span: _Span
    left: _Expr
    right: _Expr
    def __new__(cls, span: _Span, left: _Expr, right: _Expr) -> Eq: ...

@final
class Fact:
    expr: _Expr
    def __new__(cls, expr: _Expr) -> Fact: ...

_Fact: TypeAlias = Fact | Eq

##
# Change
##

@final
class Delete: ...

@final
class Subsume: ...

_Change: TypeAlias = Delete | Subsume

##
# Actions
##

@final
class Let:
    span: _Span
    lhs: str
    rhs: _Expr
    def __new__(cls, span: _Span, lhs: str, rhs: _Expr) -> Let: ...

@final
class Set:
    span: _Span
    lhs: str
    args: list[_Expr]
    rhs: _Expr
    def __new__(cls, span: _Span, lhs: str, args: list[_Expr], rhs: _Expr) -> Set: ...

@final
class Change:
    span: _Span
    change: _Change
    sym: str
    args: list[_Expr]
    def __new__(cls, span: _Span, change: _Change, sym: str, args: list[_Expr]) -> Change: ...

@final
class Union:
    span: _Span
    lhs: _Expr
    rhs: _Expr
    def __new__(cls, span: _Span, lhs: _Expr, rhs: _Expr) -> Union: ...

@final
class Panic:
    span: _Span
    msg: str
    def __new__(cls, span: _Span, msg: str) -> Panic: ...

@final
class Expr_:  # noqa: N801
    span: _Span
    expr: _Expr
    def __new__(cls, span: _Span, expr: _Expr) -> Expr_: ...

_Action: TypeAlias = Let | Set | Change | Union | Panic | Expr_

##
# Other Structs
##

@final
class Variant:
    def __new__(
        cls, span: _Span, name: str, types: list[str], cost: int | None = ..., unextractable: bool = ...
    ) -> Variant: ...

    span: _Span
    name: str
    types: list[str]
    cost: int | None
    unextractable: bool

@final
class Schema:
    input: list[str]
    output: str
    def __new__(cls, input: list[str], output: str) -> Schema: ...

@final
class Rule:
    span: _Span
    head: list[_Action]
    body: list[_Fact]
    name: str
    ruleset: str
    def __new__(cls, span: _Span, head: list[_Action], body: list[_Fact], name: str, ruleset: str) -> Rule: ...

@final
class Rewrite:
    span: _Span
    lhs: _Expr
    rhs: _Expr
    conditions: list[_Fact]

    def __new__(cls, span: _Span, lhs: _Expr, rhs: _Expr, conditions: list[_Fact] = ...) -> Rewrite: ...

@final
class RunConfig:
    ruleset: str
    until: list[_Fact] | None
    def __new__(cls, ruleset: str, until: list[_Fact] | None = ...) -> RunConfig: ...

@final
class IdentSort:
    ident: str
    sort: str
    def __new__(cls, ident: str, sort: str) -> IdentSort: ...

@final
class UserDefinedCommandOutput: ...

@final
class Function:
    name: str

@final
class RunReport:
    updated: bool
    search_and_apply_time_per_rule: dict[str, timedelta]
    num_matches_per_rule: dict[str, int]
    search_and_apply_time_per_ruleset: dict[str, timedelta]
    merge_time_per_ruleset: dict[str, timedelta]
    rebuild_time_per_ruleset: dict[str, timedelta]

    def __new__(
        cls,
        updated: bool,
        search_and_apply_time_per_rule: dict[str, timedelta],
        num_matches_per_rule: dict[str, int],
        search_and_apply_time_per_ruleset: dict[str, timedelta],
        merge_time_per_ruleset: dict[str, timedelta],
        rebuild_time_per_ruleset: dict[str, timedelta],
    ) -> RunReport: ...

##
# Command Outputs
##

@final
class PrintFunctionSize:
    size: int
    def __new__(cls, size: int) -> PrintFunctionSize: ...

@final
class PrintAllFunctionsSize:
    sizes: list[tuple[str, int]]
    def __new__(cls, sizes: list[tuple[str, int]]) -> PrintAllFunctionsSize: ...

@final
class ExtractVariants:
    termdag: TermDag
    terms: list[_Term]
    def __new__(cls, termdag: TermDag, terms: list[_Term]) -> ExtractVariants: ...

@final
class ExtractBest:
    termdag: TermDag
    cost: int
    term: _Term
    def __new__(cls, termdag: TermDag, cost: int, term: _Term) -> ExtractBest: ...

@final
class OverallStatistics:
    report: RunReport
    def __new__(cls, report: RunReport) -> OverallStatistics: ...

@final
class RunScheduleOutput:
    report: RunReport
    def __new__(cls, report: RunReport) -> RunScheduleOutput: ...

@final
class PrintFunctionOutput:
    function: Function
    termdag: TermDag
    terms: list[tuple[_Term, _Term]]
    mode: _PrintFunctionMode
    def __new__(
        cls, function: Function, termdag: TermDag, terms: list[tuple[_Term, _Term]], mode: _PrintFunctionMode
    ) -> PrintFunctionOutput: ...

@final
class UserDefinedOutput:
    output: UserDefinedCommandOutput
    def __new__(cls, output: UserDefinedCommandOutput) -> UserDefinedOutput: ...

_CommandOutput: TypeAlias = (
    PrintFunctionSize
    | PrintAllFunctionsSize
    | ExtractVariants
    | ExtractBest
    | OverallStatistics
    | RunScheduleOutput
    | PrintFunctionOutput
    | UserDefinedOutput
)

##
# Print Function Modes
##

@final
class DefaultPrintFunctionMode: ...

@final
class CSVPrintFunctionMode: ...

_PrintFunctionMode: TypeAlias = DefaultPrintFunctionMode | CSVPrintFunctionMode

##
# Schedules
##

@final
class Saturate:
    span: _Span
    schedule: _Schedule
    def __new__(cls, span: _Span, schedule: _Schedule) -> Saturate: ...

@final
class Repeat:
    span: _Span
    length: int
    schedule: _Schedule
    def __new__(cls, span: _Span, length: int, schedule: _Schedule) -> Repeat: ...

@final
class Run:
    span: _Span
    config: RunConfig
    def __new__(cls, span: _Span, config: RunConfig) -> Run: ...

@final
class Sequence:
    span: _Span
    schedules: list[_Schedule]
    def __new__(cls, span: _Span, schedules: list[_Schedule]) -> Sequence: ...

_Schedule: TypeAlias = Saturate | Repeat | Run | Sequence

##
# Subdatatypes
##

@final
class SubVariants:
    variants: list[Variant]
    def __new__(cls, variants: list[Variant]) -> SubVariants: ...

@final
class NewSort:
    name: str
    args: list[_Expr]
    def __new__(cls, name: str, args: list[_Expr]) -> NewSort: ...

_Subdatatypes: TypeAlias = SubVariants | NewSort

##
# Commands
##

@final
class Datatype:
    span: _Span
    name: str
    variants: list[Variant]
    def __new__(cls, span: _Span, name: str, variants: list[Variant]) -> Datatype: ...

@final
class Datatypes:
    span: _Span
    datatypes: list[tuple[_Span, str, _Subdatatypes]]
    def __new__(cls, span: _Span, datatypes: list[tuple[_Span, str, _Subdatatypes]]) -> Datatypes: ...

@final
class Sort:
    span: _Span
    name: str
    presort_and_args: tuple[str, list[_Expr]] | None
    def __new__(cls, span: _Span, name: str, presort_and_args: tuple[str, list[_Expr]] | None) -> Sort: ...

@final
class FunctionCommand:
    span: _Span
    name: str
    schema: Schema
    merge: _Expr | None
    def __new__(cls, span: _Span, name: str, schema: Schema, merge: _Expr | None) -> FunctionCommand: ...

@final
class AddRuleset:
    span: _Span
    name: str
    def __new__(cls, span: _Span, name: str) -> AddRuleset: ...

@final
class RuleCommand:
    rule: Rule
    def __new__(cls, rule: Rule) -> RuleCommand: ...

@final
class RewriteCommand:
    # TODO: Rename to ruleset
    name: str
    rewrite: Rewrite
    subsume: bool
    def __new__(cls, name: str, rewrite: Rewrite, subsume: bool) -> RewriteCommand: ...

@final
class BiRewriteCommand:
    # TODO: Rename to ruleset
    name: str
    rewrite: Rewrite
    def __new__(cls, name: str, rewrite: Rewrite) -> BiRewriteCommand: ...

@final
class ActionCommand:
    action: _Action
    def __new__(cls, action: _Action) -> ActionCommand: ...

@final
class RunSchedule:
    schedule: _Schedule
    def __new__(cls, schedule: _Schedule) -> RunSchedule: ...

@final
class Extract:
    span: _Span
    expr: _Expr
    variants: _Expr
    def __new__(cls, span: _Span, expr: _Expr, variants: _Expr) -> Extract: ...

@final
class Check:
    span: _Span
    facts: list[_Fact]
    def __new__(cls, span: _Span, facts: list[_Fact]) -> Check: ...

@final
class PrintFunction:
    span: _Span
    name: str
    length: int | None
    filename: str | None
    mode: _PrintFunctionMode
    def __new__(
        cls, span: _Span, name: str, length: int | None, filename: str | None, mode: _PrintFunctionMode
    ) -> PrintFunction: ...

@final
class PrintSize:
    span: _Span
    name: str | None
    def __new__(cls, span: _Span, name: str | None) -> PrintSize: ...

@final
class Output:
    span: _Span
    file: str
    exprs: list[_Expr]
    def __new__(cls, span: _Span, file: str, exprs: list[_Expr]) -> Output: ...

@final
class Input:
    span: _Span
    name: str
    file: str
    def __new__(cls, span: _Span, name: str, file: str) -> Input: ...

@final
class Push:
    length: int
    def __new__(cls, length: int) -> Push: ...

@final
class Pop:
    span: _Span
    length: int
    def __new__(cls, span: _Span, length: int) -> Pop: ...

@final
class Fail:
    span: _Span
    command: _Command
    def __new__(cls, span: _Span, command: _Command) -> Fail: ...

@final
class Include:
    span: _Span
    path: str
    def __new__(cls, span: _Span, path: str) -> Include: ...

@final
class Relation:
    span: _Span
    name: str
    inputs: list[str]

    def __new__(cls, span: _Span, name: str, inputs: list[str]) -> Relation: ...

@final
class Constructor:
    span: _Span
    name: str
    schema: Schema
    cost: int | None
    unextractable: bool
    def __new__(cls, span: _Span, name: str, schema: Schema, cost: int | None, unextractable: bool) -> Constructor: ...

@final
class PrintOverallStatistics: ...

@final
class UserDefined:
    span: _Span
    name: str
    args: list[_Expr]
    def __new__(cls, span: _Span, name: str, args: list[_Expr]) -> UserDefined: ...

@final
class UnstableCombinedRuleset:
    span: _Span
    name: str
    rulesets: list[str]
    def __new__(cls, span: _Span, name: str, rulesets: list[str]) -> UnstableCombinedRuleset: ...

_Command: TypeAlias = (
    Datatype
    | Datatypes
    | Sort
    | FunctionCommand
    | AddRuleset
    | RuleCommand
    | RewriteCommand
    | BiRewriteCommand
    | ActionCommand
    | RunSchedule
    | Extract
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
    | UserDefined
)

##
# TermDag
##

@final
class TermDag:
    def size(self) -> int: ...
    def lookup(self, node: _Term) -> int: ...
    def get(self, id: int) -> _Term: ...
    def app(self, sym: str, children: list[int]) -> _Term: ...
    def lit(self, lit: _Literal) -> _Term: ...
    def var(self, sym: str) -> _Term: ...
    def expr_to_term(self, expr: _Expr) -> _Term: ...
    def term_to_expr(self, term: _Term, span: _Span) -> _Expr: ...
    def to_string(self, term: _Term) -> str: ...

##
# Extraction
##
class _Cost(Protocol):
    def __lt__(self, other: _Cost) -> bool: ...
    def __le__(self, other: _Cost) -> bool: ...
    def __gt__(self, other: _Cost) -> bool: ...
    def __ge__(self, other: _Cost) -> bool: ...

_COST = TypeVar("_COST", bound=_Cost)

_ENODE_COST = TypeVar("_ENODE_COST")

@final
class CostModel(Generic[_COST, _ENODE_COST]):
    def __new__(
        cls,
        fold: Callable[[str, _ENODE_COST, list[_COST]], _COST],
        enode_cost: Callable[[str, list[Value]], _ENODE_COST],
        container_cost: Callable[[str, Value, list[_COST]], _COST],
        base_value_cost: Callable[[str, Value], _COST],
    ) -> CostModel[_COST, _ENODE_COST]: ...

@final
class Extractor(Generic[_COST]):
    def __new__(
        cls, rootsorts: list[str] | None, egraph: EGraph, cost_model: CostModel[_COST, Any]
    ) -> Extractor[_COST]: ...
    def extract_best(self, egraph: EGraph, termdag: TermDag, value: Value, sort: str) -> tuple[_COST, _Term]: ...
    def extract_variants(
        self, egraph: EGraph, termdag: TermDag, value: Value, nvariants: int, sort: str
    ) -> list[tuple[_COST, _Term]]: ...
