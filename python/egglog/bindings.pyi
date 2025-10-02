from collections.abc import Callable
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from typing_extensions import final

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
    "PyObjectSort",
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
    def value_to_pyobject(self, py_object_sort: PyObjectSort, v: Value) -> object: ...
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

_Action: TypeAlias = Let | Set | Change | Union | Panic | Expr_

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

    def __init__(
        self,
        updated: bool,
        search_and_apply_time_per_rule: dict[str, timedelta],
        num_matches_per_rule: dict[str, int],
        search_and_apply_time_per_ruleset: dict[str, timedelta],
        merge_time_per_ruleset: dict[str, timedelta],
        rebuild_time_per_ruleset: dict[str, timedelta],
    ) -> None: ...

##
# Command Outputs
##

@final
class PrintFunctionSize:
    size: int
    def __init__(self, size: int) -> None: ...

@final
class PrintAllFunctionsSize:
    sizes: list[tuple[str, int]]
    def __init__(self, sizes: list[tuple[str, int]]) -> None: ...

@final
class ExtractVariants:
    termdag: TermDag
    terms: list[_Term]
    def __init__(self, termdag: TermDag, terms: list[_Term]) -> None: ...

@final
class ExtractBest:
    termdag: TermDag
    cost: int
    term: _Term
    def __init__(self, termdag: TermDag, cost: int, term: _Term) -> None: ...

@final
class OverallStatistics:
    report: RunReport
    def __init__(self, report: RunReport) -> None: ...

@final
class RunScheduleOutput:
    report: RunReport
    def __init__(self, report: RunReport) -> None: ...

@final
class PrintFunctionOutput:
    function: Function
    termdag: TermDag
    terms: list[tuple[_Term, _Term]]
    mode: _PrintFunctionMode
    def __init__(
        self, function: Function, termdag: TermDag, terms: list[tuple[_Term, _Term]], mode: _PrintFunctionMode
    ) -> None: ...

@final
class UserDefinedOutput:
    output: UserDefinedCommandOutput
    def __init__(self, output: UserDefinedCommandOutput) -> None: ...

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
class FunctionCommand:
    span: _Span
    name: str
    schema: Schema
    merge: _Expr | None
    def __init__(self, span: _Span, name: str, schema: Schema, merge: _Expr | None) -> None: ...

@final
class AddRuleset:
    span: _Span
    name: str
    def __init__(self, span: _Span, name: str) -> None: ...

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
class Extract:
    span: _Span
    expr: _Expr
    variants: _Expr
    def __init__(self, span: _Span, expr: _Expr, variants: _Expr) -> None: ...

@final
class Check:
    span: _Span
    facts: list[_Fact]
    def __init__(self, span: _Span, facts: list[_Fact]) -> None: ...

@final
class PrintFunction:
    span: _Span
    name: str
    length: int | None
    filename: str | None
    mode: _PrintFunctionMode
    def __init__(
        self, span: _Span, name: str, length: int | None, filename: str | None, mode: _PrintFunctionMode
    ) -> None: ...

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
class UserDefined:
    span: _Span
    name: str
    args: list[_Expr]
    def __init__(self, span: _Span, name: str, args: list[_Expr]) -> None: ...

@final
class UnstableCombinedRuleset:
    span: _Span
    name: str
    rulesets: list[str]
    def __init__(self, span: _Span, name: str, rulesets: list[str]) -> None: ...

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
    def __init__(
        self,
        fold: Callable[[str, _ENODE_COST, list[_COST]], _COST],
        enode_cost: Callable[[str, list[Value]], _ENODE_COST],
        container_cost: Callable[[str, Value, list[_COST]], _COST],
        base_value_cost: Callable[[str, Value], _COST],
    ) -> None: ...

@final
class Extractor(Generic[_COST]):
    def __init__(self, rootsorts: list[str] | None, egraph: EGraph, cost_model: CostModel[_COST, Any]) -> None: ...
    def extract_best(self, egraph: EGraph, termdag: TermDag, value: Value, sort: str) -> tuple[_COST, _Term]: ...
    def extract_variants(
        self, egraph: EGraph, termdag: TermDag, value: Value, nvariants: int, sort: str
    ) -> list[tuple[_COST, _Term]]: ...
