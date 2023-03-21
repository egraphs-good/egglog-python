from datetime import timedelta

from typing_extensions import final

def parse(input: str) -> list[_Command]: ...
@final
class EGraph:
    def parse_and_run_program(self, input: str) -> list[str]: ...
    def declare_constructor(self, variant: Variant, sort: str) -> None: ...
    def declare_sort(
        self, name: str, presort_and_args: tuple[str, list[_Expr]] | None = None
    ) -> None: ...
    def declare_function(self, decl: FunctionDecl) -> None: ...
    def define(self, name: str, expr: _Expr, cost: int | None = None) -> None: ...
    def add_rewrite(self, rewrite: Rewrite) -> str: ...
    def run_rules(self, limit: int) -> tuple[timedelta, timedelta, timedelta]: ...
    def check_fact(self, fact: _Fact) -> None: ...
    def extract_expr(
        self, expr: _Expr, variants: int = 0
    ) -> tuple[int, _Expr, list[_Expr]]: ...
    def add_rule(self, rule: Rule) -> str: ...
    def eval_actions(self, *actions: _Action) -> None: ...
    def push(self) -> None: ...
    def pop(self) -> None: ...
    def clear(self) -> None: ...
    def clear_rules(self) -> None: ...
    def print_size(self, name: str) -> str: ...
    def print_function(self, name: str, n: int) -> str: ...

@final
class EggSmolError(Exception):
    context: str

@final
class Int:
    def __init__(self, value: int) -> None: ...
    value: int

@final
class String:
    def __init__(self, value: str) -> None: ...
    value: str

@final
class Unit:
    def __init__(self) -> None: ...

_Literal = Int | String | Unit

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

# This must be private becuase it is not actually exposed by the runtime library.
_Expr = Lit | Var | Call

@final
class Eq:
    def __init__(self, exprs: list[_Expr]) -> None: ...
    exprs: list[_Expr]

@final
class Fact:
    def __init__(self, expr: _Expr) -> None: ...
    expr: _Expr

_Fact = Fact | Eq

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

@final
class FunctionDecl:
    name: str
    schema: Schema
    default: _Expr | None
    merge: _Expr | None
    cost: int | None
    def __init__(
        self,
        name: str,
        schema: Schema,
        default: _Expr | None = None,
        merge: _Expr | None = None,
        cost: int | None = None,
    ) -> None: ...

@final
class Variant:
    def __init__(
        self, name: str, types: list[str], cost: int | None = None
    ) -> None: ...
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

    def __init__(
        self, lhs: _Expr, rhs: _Expr, conditions: list[_Fact] = []
    ) -> None: ...

@final
class Datatype:
    name: str
    variants: list[Variant]
    def __init__(self, name: str, variants: list[Variant]) -> None: ...

@final
class Sort:
    name: str
    presort: str
    args: list[_Expr]
    def __init__(self, name: str, presort: str, args: list[_Expr]) -> None: ...

@final
class Function:
    decl: FunctionDecl
    def __init__(self, decl: FunctionDecl) -> None: ...

@final
class Define:
    name: str
    expr: _Expr
    cost: int | None
    def __init__(self, name: str, expr: _Expr, cost: int | None = None) -> None: ...

@final
class RuleCommand:
    rule: Rule
    def __init__(self, rule: Rule) -> None: ...

@final
class RewriteCommand:
    rewrite: Rewrite
    def __init__(self, rewrite: Rewrite) -> None: ...

@final
class ActionCommand:
    action: _Action
    def __init__(self, action: _Action) -> None: ...

@final
class Run:
    length: int
    def __init__(self, length: int) -> None: ...

@final
class Extract:
    variants: int
    expr: _Expr
    def __init__(self, variants: int, expr: _Expr) -> None: ...

@final
class Check:
    fact: _Fact
    def __init__(self, fact: _Fact) -> None: ...

@final
class ClearRules:
    def __init__(self) -> None: ...

@final
class Clear:
    def __init__(self) -> None: ...

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
class Input:
    name: str
    file: str
    def __init__(self, name: str, file: str) -> None: ...

@final
class Query:
    facts: list[_Fact]
    def __init__(self, facts: list[_Fact]) -> None: ...

@final
class Push:
    length: int
    def __init__(self, length: int) -> None: ...

@final
class Pop:
    length: int
    def __init__(self, length: int) -> None: ...

_Command = (
    Datatype
    | Sort
    | Function
    | Define
    | RuleCommand
    | RewriteCommand
    | ActionCommand
    | Run
    | Extract
    | Check
    | ClearRules
    | Clear
    | Print
    | PrintSize
    | Input
    | Query
    | Push
    | Pop
)
