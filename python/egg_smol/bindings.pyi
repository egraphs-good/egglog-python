from datetime import timedelta
from pathlib import Path
from typing import Optional

from typing_extensions import final

HIGH_COST: int

@final
class EGraph:
    def __init__(self, fact_directory: str | Path | None = None, seminaive=True) -> None: ...
    def parse_program(self, __input: str, /) -> list[_Command]: ...
    def run_program(self, *commands: _Command) -> list[str]: ...
    def extract_report(self) -> Optional[ExtractReport]: ...
    def run_report(self) -> Optional[RunReport]: ...

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

_Literal = Int | F64 | String | Unit

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
_Expr = Lit | Var | Call

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

_Fact = Fact | Eq

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
class Expr_:
    def __init__(self, expr: _Expr) -> None: ...
    expr: _Expr

_Action = Let | Set | Delete | Union | Panic | Expr_

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
    def __init__(
        self,
        name: str,
        schema: Schema,
        default: _Expr | None = None,
        merge: _Expr | None = None,
        merge_action: list[_Action] = [],
        cost: int | None = None,
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

    def __init__(self, lhs: _Expr, rhs: _Expr, conditions: list[_Fact] = []) -> None: ...

@final
class RunConfig:
    ruleset: str
    limit: int
    until: Optional[list[_Fact]]
    def __init__(self, ruleset: str, limit: int, until: Optional[list[_Fact]] = None) -> None: ...

@final
class IdentSort:
    ident: str
    sort: str
    def __init__(self, ident: str, sort: str) -> None: ...

@final
class RunReport:
    updated: bool
    search_time: timedelta
    apply_time: timedelta
    rebuild_time: timedelta

    def __init__(
        self,
        updated: bool,
        search_time: timedelta,
        apply_time: timedelta,
        rebuild_time: timedelta,
    ) -> None: ...

@final
class ExtractReport:
    cost: int
    expr: _Expr
    variants: list[_Expr]

    def __init__(self, cost: int, expr: _Expr, variants: list[_Expr]) -> None: ...

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

_Schedule = Saturate | Repeat | Run | Sequence

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
    presort_and_args: Optional[tuple[str, list[_Expr]]]
    def __init__(self, name: str, presort_and_args: Optional[tuple[str, list[_Expr]]] = None) -> None: ...

@final
class Function:
    decl: FunctionDecl
    def __init__(self, decl: FunctionDecl) -> None: ...

@final
class Define:
    name: str
    expr: _Expr
    cost: int | None
    def __init__(self, name: str, expr: _Expr, cost: int | None) -> None: ...

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
class RunCommand:
    config: RunConfig
    def __init__(self, config: RunConfig) -> None: ...

@final
class RunScheduleCommand:
    schedule: _Schedule
    def __init__(self, schedule: _Schedule) -> None: ...

@final
class Simplify:
    expr: _Expr
    config: RunConfig
    def __init__(self, expr: _Expr, config: RunConfig) -> None: ...

@final
class Calc:
    identifiers: list[IdentSort]
    exprs: list[_Expr]
    def __init__(self, identifiers: list[IdentSort], exprs: list[_Expr]) -> None: ...

@final
class Extract:
    variants: int
    expr: _Expr
    def __init__(self, variants: int, expr: _Expr) -> None: ...

@final
class Check:
    facts: list[_Fact]
    def __init__(self, facts: list[_Fact]) -> None: ...

@final
class Print:
    name: str
    length: int
    def __init__(self, name: str, length: int) -> None: ...

@final
class PrintSize:
    name: str
    def __init__(self, name: str) -> None: ...

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

_Command = (
    SetOption
    | Datatype
    | Declare
    | Sort
    | Function
    | Define
    | AddRuleset
    | RuleCommand
    | RewriteCommand
    | BiRewriteCommand
    | ActionCommand
    | RunCommand
    | RunScheduleCommand
    | Calc
    | Simplify
    | Extract
    | Check
    | Print
    | PrintSize
    | Output
    | Input
    | Push
    | Pop
    | Fail
    | Include
)
