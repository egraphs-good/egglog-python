# TODO: Figure out what these modules should be called
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Variant:
    name: str
    types: list[str]
    cost: Optional[int] = None


@dataclass(frozen=True)
class FunctionDecl:
    name: str
    schema: Schema
    default: Optional[Expr] = None
    merge: Optional[Expr] = None
    cost: Optional[int] = None


@dataclass(frozen=True)
class Schema:
    input: list[str]
    output: str


@dataclass(frozen=True)
class Lit:
    value: Literal


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Call:
    name: str
    args: list[Expr]


Expr = Union[Lit, Var, Call]


@dataclass(frozen=True)
class Int:
    value: int


@dataclass(frozen=True)
class String:
    value: str


@dataclass(frozen=True)
class Unit:
    pass


Literal = Union[Int, String, Unit]


@dataclass(frozen=True)
class Rewrite:
    lhs: Expr
    rhs: Expr
    conditions: list[Fact_] = field(default_factory=list)


@dataclass(frozen=True)
class Fact:
    expr: Expr


@dataclass
class Eq:
    exprs: list[Expr]


Fact_ = Union[Fact, Eq]
