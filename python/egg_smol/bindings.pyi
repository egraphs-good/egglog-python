from datetime import timedelta
from typing import Optional

from typing_extensions import final

@final
class EGraph:
    def parse_and_run_program(self, input: str) -> list[str]: ...
    def declare_constructor(self, variant: Variant, sort: str) -> None: ...
    def declare_sort(self, name: str) -> None: ...
    def declare_function(self, decl: FunctionDecl) -> None: ...
    def define(self, name: str, expr: _Expr, cost: Optional[int] = None) -> None: ...
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
class Define:
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

_Action = Define | Set | Delete | Union | Panic | Expr_

@final
class FunctionDecl:
    name: str
    schema: Schema
    default: Optional[_Expr]
    merge: Optional[_Expr]
    cost: Optional[int]
    def __init__(
        self,
        name: str,
        schema: Schema,
        default: Optional[_Expr] = None,
        merge: Optional[_Expr] = None,
        cost: Optional[int] = None,
    ) -> None: ...

@final
class Variant:
    def __init__(
        self, name: str, types: list[str], cost: Optional[int] = None
    ) -> None: ...
    name: str
    types: list[str]
    cost: Optional[int]

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
