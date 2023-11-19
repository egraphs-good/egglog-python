from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

from typing_extensions import final

HIGH_COST: int

@final
class SerializedEGraph:
    def inline_leaves(self) -> None: ...
    def saturate_inline_leaves(self) -> None: ...
    def to_dot(self) -> str: ...
    def to_json(self) -> str: ...
    def map_ops(self, map: dict[str, str]) -> None: ...

@final
class PyObjectSort:
    def __init__(self) -> None: ...
    def store(self, __o: object, /) -> _Expr: ...

@final
class EGraph:
    def __init__(
        self,
        __py_object_sort: PyObjectSort | None = None,
        *,
        fact_directory: str | Path | None = None,
        seminaive: bool = True,
        terms_encoding: bool = False,
    ) -> None: ...
    def parse_program(self, __input: str, /) -> list[_Command]: ...
    def run_program(self, *commands: _Command) -> list[str]: ...
    def extract_report(self) -> _ExtractReport | None: ...
    def run_report(self) -> RunReport | None: ...
    def serialize(
        self,
        *,
        max_functions: int | None = None,
        max_calls_per_function: int | None = None,
        include_temporary_functions: bool = False,
        split_primitive_outputs: bool = False,
    ) -> SerializedEGraph: ...
    def eval_py_object(self, __expr: _Expr) -> object: ...
    def eval_i64(self, __expr: _Expr) -> int: ...
    def eval_f64(self, __expr: _Expr) -> float: ...
    def eval_string(self, __expr: _Expr) -> str: ...
    def eval_bool(self, __expr: _Expr) -> bool: ...
    def eval_rational(self, __expr: _Expr) -> Fraction: ...

@final
class EggSmolError(Exception):
    context: str

##
# Literals
##

@final
class Int:
    def __init__(self, value: int) -> None: ...
    value: int

@final
class F64:
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

_Literal: TypeAlias = Int | F64 | String | Bool | Unit

##
# Expressions
##

@final
class Lit:
    def __init__(self, value: _Literal) -> None: ...
    value: _Literal

@final
class Var:
    def __init__(self, name: str) -> None: ...
    name: str

@final
class Call:
    def __init__(self, name: str, args: list[_Expr]) -> None: ...
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

@final
class TermDag:
    nodes: list[_Term]
    hashcons: dict[_Term, int]

##
# Facts
##

@final
class Eq:
    def __init__(self, exprs: list[_Expr]) -> None: ...
    exprs: list[_Expr]

@final
class Fact:
    def __init__(self, expr: _Expr) -> None: ...
    expr: _Expr

_Fact: TypeAlias = Fact | Eq

##
# Actions
##

@final
class Let:
    def __init__(self, lhs: str, rhs: _Expr) -> None: ...
    lhs: str
    rhs: _Expr

@final
class Set:
    def __init__(self, lhs: str, args: list[_Expr], rhs: _Expr) -> None: ...
    lhs: str
    args: list[_Expr]
    rhs: _Expr

@final
class Delete:
    sym: str
    args: list[_Expr]
    def __init__(self, sym: str, args: list[_Expr]) -> None: ...

@final
class Union:
    def __init__(self, lhs: _Expr, rhs: _Expr) -> None: ...
    lhs: _Expr
    rhs: _Expr

@final
class Panic:
    def __init__(self, msg: str) -> None: ...
    msg: str

@final
class Expr_:  # noqa: N801
    def __init__(self, expr: _Expr) -> None: ...
    expr: _Expr

@final
class Extract:
    def __init__(self, expr: _Expr, variants: _Expr) -> None: ...
    expr: _Expr
    variants: _Expr

_Action: TypeAlias = Let | Set | Delete | Union | Panic | Expr_ | Extract

##
# Other Structs
##

@final
class FunctionDecl:
    name: str
    schema: Schema
    default: _Expr | None
    merge: _Expr | None
    merge_action: list[_Action]
    cost: int | None
    unextractable: bool

    def __init__(
        self,
        name: str,
        schema: Schema,
        default: _Expr | None = None,
        merge: _Expr | None = None,
        merge_action: list[_Action] = [],  # noqa: B006
        cost: int | None = None,
        unextractable: bool = False,
    ) -> None: ...

@final
class Variant:
    def __init__(self, name: str, types: list[str], cost: int | None = None) -> None: ...
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
    head: list[_Action]
    body: list[_Fact]
    def __init__(self, head: list[_Action], body: list[_Fact]) -> None: ...

@final
class Rewrite:
    lhs: _Expr
    rhs: _Expr
    conditions: list[_Fact]

    def __init__(self, lhs: _Expr, rhs: _Expr, conditions: list[_Fact] = []) -> None: ...  # noqa: B006

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

    def __init__(
        self,
        updated: bool,
        search_time_per_rule: dict[str, timedelta],
        apply_time_per_rule: dict[str, timedelta],
        search_time_per_ruleset: dict[str, timedelta],
        apply_time_per_ruleset: dict[str, timedelta],
        rebuild_time_per_ruleset: dict[str, timedelta],
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
    schedule: _Schedule
    def __init__(self, schedule: _Schedule) -> None: ...

@final
class Repeat:
    length: int
    schedule: _Schedule
    def __init__(self, length: int, schedule: _Schedule) -> None: ...

@final
class Run:
    config: RunConfig
    def __init__(self, config: RunConfig) -> None: ...

@final
class Sequence:
    schedules: list[_Schedule]
    def __init__(self, schedules: list[_Schedule]) -> None: ...

_Schedule: TypeAlias = Saturate | Repeat | Run | Sequence

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
    name: str
    variants: list[Variant]
    def __init__(self, name: str, variants: list[Variant]) -> None: ...

@final
class Declare:
    name: str
    sort: str
    def __init__(self, name: str, sort: str) -> None: ...

@final
class Sort:
    name: str
    presort_and_args: tuple[str, list[_Expr]] | None
    def __init__(self, name: str, presort_and_args: tuple[str, list[_Expr]] | None = None) -> None: ...

@final
class Function:
    decl: FunctionDecl
    def __init__(self, decl: FunctionDecl) -> None: ...

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
    def __init__(self, name: str, rewrite: Rewrite) -> None: ...

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
    expr: _Expr
    schedule: _Schedule
    def __init__(self, expr: _Expr, schedule: _Schedule) -> None: ...

@final
class Calc:
    identifiers: list[IdentSort]
    exprs: list[_Expr]
    def __init__(self, identifiers: list[IdentSort], exprs: list[_Expr]) -> None: ...

@final
class QueryExtract:
    variants: int
    expr: _Expr
    def __init__(self, variants: int, expr: _Expr) -> None: ...

@final
class Check:
    facts: list[_Fact]
    def __init__(self, facts: list[_Fact]) -> None: ...

@final
class PrintFunction:
    name: str
    length: int
    def __init__(self, name: str, length: int) -> None: ...

@final
class PrintSize:
    name: str | None
    def __init__(self, name: str | None) -> None: ...

@final
class Output:
    file: str
    exprs: list[_Expr]
    def __init__(self, file: str, exprs: list[_Expr]) -> None: ...

@final
class Input:
    name: str
    file: str
    def __init__(self, name: str, file: str) -> None: ...

@final
class Push:
    length: int
    def __init__(self, length: int) -> None: ...

@final
class Pop:
    length: int
    def __init__(self, length: int) -> None: ...

@final
class Fail:
    command: _Command
    def __init__(self, command: _Command) -> None: ...

@final
class Include:
    path: str
    def __init__(self, path: str) -> None: ...

@final
class CheckProof:
    def __init__(self) -> None: ...

@final
class Relation:
    constructor: str
    inputs: list[str]

    def __init__(self, constructor: str, inputs: list[str]) -> None: ...

@final
class PrintOverallStatistics:
    def __init__(self) -> None: ...

_Command: TypeAlias = (
    SetOption
    | Datatype
    | Declare
    | Sort
    | Function
    | AddRuleset
    | RuleCommand
    | RewriteCommand
    | BiRewriteCommand
    | ActionCommand
    | RunSchedule
    | Calc
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
    | CheckProof
    | Relation
    | PrintOverallStatistics
)

def termdag_term_to_expr(termdag: TermDag, term: _Term) -> _Expr: ...
