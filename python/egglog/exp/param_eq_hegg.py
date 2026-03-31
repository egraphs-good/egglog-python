# mypy: disable-error-code="empty-body,arg-type,return-value,call-overload,assignment"

"""
Helpers for reproducing the paper-era param-eq EqSat pipeline in egglog.

The archived sources expose two closely related Haskell implementations:

- `param-eq-haskell/src/FixTree.hs`, which is the actual experiment harness
  used to generate the paper tables shipped with the archive.
- `pandoc-symreg/src/Data/SRTree/EqSat.hs`, which contains a later hegg-based
  implementation together with a depth-limited matcher in `Data.Equality`.

This module mirrors the experiment harness as the baseline, then adds an
approximate "height guard" mode that tries to mimic the matcher pruning from
the second archive by constraining the highest-growth rewrite patterns.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias, cast

import egglog
from egglog import *
from egglog.deconstruct import get_callable_args

__all__ = [
    "HEIGHT_LIMIT",
    "Mode",
    "Num",
    "NumLike",
    "PaperPipelineReport",
    "count_nodes",
    "count_params",
    "parse_expression",
    "render_num",
    "run_paper_pipeline",
]


HEIGHT_LIMIT = 8
MAX_PASSES = 2
BACKOFF_MATCH_LIMIT = 2500
BACKOFF_BAN_LENGTH = 30

Mode = Literal["egglog-baseline", "egglog-height-guard"]


language_rules = ruleset(name="param_eq_hegg_lang")
const_analysis_rules = ruleset(name="param_eq_hegg_const_analysis")
height_analysis_rules = ruleset(name="param_eq_hegg_height")
basic_rules = ruleset(name="param_eq_hegg_basic")
basic_rules_height_guard = ruleset(name="param_eq_hegg_basic_height_guard")
fun_rules = ruleset(name="param_eq_hegg_fun")
fun_rules_height_guard = ruleset(name="param_eq_hegg_fun_height_guard")


class OptionalF64(Expr, ruleset=language_rules):
    none: ClassVar[OptionalF64]

    @classmethod
    def some(cls, value: f64Like) -> OptionalF64: ...


class Num(Expr, ruleset=language_rules):
    @method(cost=5)
    def __init__(self, value: f64Like) -> None: ...

    @method(cost=1)
    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    @method(cost=1)
    def __add__(self, other: NumLike) -> Num: ...

    @method(cost=1)
    def __sub__(self, other: NumLike) -> Num: ...

    @method(cost=1)
    def __mul__(self, other: NumLike) -> Num: ...

    @method(cost=1)
    def __truediv__(self, other: NumLike) -> Num: ...

    @method(cost=1)
    def __pow__(self, other: NumLike) -> Num: ...

    @method(cost=1)
    def exp(self) -> Num: ...

    @method(cost=1)
    def log(self) -> Num: ...

    @method(cost=1)
    def sqrt(self) -> Num: ...

    @method(cost=1)
    def __abs__(self) -> Num: ...

    def __radd__(self, other: NumLike) -> Num: ...

    def __rsub__(self, other: NumLike) -> Num: ...

    def __rmul__(self, other: NumLike) -> Num: ...

    def __rtruediv__(self, other: NumLike) -> Num: ...

    def __rpow__(self, other: NumLike) -> Num: ...


NumLike: TypeAlias = Num | StringLike | float | int

converter(float, Num, Num)
converter(int, Num, lambda value: Num(float(value)))
converter(String, Num, Num.var)
converter(str, Num, Num.var)


@function(
    ruleset=const_analysis_rules,
    merge=lambda old, new: old if old != OptionalF64.none else new,
)
def const_value(num: Num) -> OptionalF64: ...


@function(ruleset=height_analysis_rules, merge=lambda old, new: old.max(new))
def expr_height(num: Num) -> i64: ...


def exp(x: NumLike) -> Num:
    return convert(x, Num).exp()


def log(x: NumLike) -> Num:
    return convert(x, Num).log()


def sqrt(x: NumLike) -> Num:
    return convert(x, Num).sqrt()


def neg(x: NumLike) -> Num:
    return Num(-1.0) * convert(x, Num)


def square(x: NumLike) -> Num:
    value = convert(x, Num)
    return value * value


def cube(x: NumLike) -> Num:
    value = convert(x, Num)
    return value * value * value


def plog(x: NumLike) -> Num:
    return log(abs(convert(x, Num)))


def _render_float(value: float) -> str:
    if value == 0.0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    return repr(value)


def _safe_pow(base: float, exponent: float) -> float | None:
    if base == 0.0 and exponent < 0.0:
        return None
    if base < 0.0 and not exponent.is_integer():
        return None
    try:
        value = base**exponent
    except (OverflowError, ValueError):
        return None
    if isinstance(value, complex):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _combine_binary(lhs: float, rhs: float, op: Literal["add", "sub", "mul", "div", "pow"]) -> float | None:
    if op == "add":
        value = lhs + rhs
    elif op == "sub":
        value = lhs - rhs
    elif op == "mul":
        value = lhs * rhs
    elif op == "div":
        if rhs == 0.0:
            return None
        value = lhs / rhs
    else:
        return _safe_pow(lhs, rhs)
    return value if math.isfinite(value) else None


@const_analysis_rules.register
def _const_analysis(
    num: Num,
    x: Num,
    y: Num,
    a: f64,
    b: f64,
    s: String,
) -> Iterable[RewriteOrRule]:
    yield rule(eq(num).to(Num(a))).then(set_(const_value(num)).to(OptionalF64.some(a)))
    yield rule(eq(num).to(Num.var(s))).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x + y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a + b)),
        union(num).with_(Num(a + b)),
    )
    yield rule(eq(num).to(x - y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a - b)),
        union(num).with_(Num(a - b)),
    )
    yield rule(eq(num).to(x * y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a * b)),
        union(num).with_(Num(a * b)),
    )
    yield rule(
        eq(num).to(x / y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        b != 0.0,
    ).then(
        set_(const_value(num)).to(OptionalF64.some(a / b)),
        union(num).with_(Num(a / b)),
    )
    yield rule(eq(num).to(x**y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b), a >= 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a**b)),
        union(num).with_(Num(a**b)),
    )
    yield rule(eq(num).to(Num(-1.0) * x), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(-a)),
        union(num).with_(Num(-a)),
    )
    yield rule(eq(num).to(exp(x)), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(a.exp())),
        union(num).with_(Num(a.exp())),
    )
    yield rule(eq(num).to(log(x)), const_value(x) == OptionalF64.some(a), a > 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a.log())),
        union(num).with_(Num(a.log())),
    )
    yield rule(eq(num).to(sqrt(x)), const_value(x) == OptionalF64.some(a), a >= 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a.sqrt())),
        union(num).with_(Num(a.sqrt())),
    )
    yield rule(eq(num).to(abs(x)), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(abs(a))),
        union(num).with_(Num(abs(a))),
    )


@height_analysis_rules.register
def _height_analysis(
    num: Num,
    x: Num,
    y: Num,
    a: f64,
    i: i64,
    j: i64,
    s: String,
) -> Iterable[RewriteOrRule]:
    yield rule(eq(num).to(Num.var(s))).then(set_(expr_height(num)).to(i64(1)))
    yield rule(eq(num).to(Num(a))).then(set_(expr_height(num)).to(i64(1)))
    yield rule(eq(num).to(x + y), expr_height(x) == i, expr_height(y) == j).then(
        set_(expr_height(num)).to(i.max(j) + 1)
    )
    yield rule(eq(num).to(x - y), expr_height(x) == i, expr_height(y) == j).then(
        set_(expr_height(num)).to(i.max(j) + 1)
    )
    yield rule(eq(num).to(x * y), expr_height(x) == i, expr_height(y) == j).then(
        set_(expr_height(num)).to(i.max(j) + 1)
    )
    yield rule(eq(num).to(x / y), expr_height(x) == i, expr_height(y) == j).then(
        set_(expr_height(num)).to(i.max(j) + 1)
    )
    yield rule(eq(num).to(x**y), expr_height(x) == i, expr_height(y) == j).then(
        set_(expr_height(num)).to(i.max(j) + 1)
    )
    yield rule(eq(num).to(Num(-1.0) * x), expr_height(x) == i).then(set_(expr_height(num)).to(i + 1))
    yield rule(eq(num).to(exp(x)), expr_height(x) == i).then(set_(expr_height(num)).to(i + 1))
    yield rule(eq(num).to(log(x)), expr_height(x) == i).then(set_(expr_height(num)).to(i + 1))
    yield rule(eq(num).to(sqrt(x)), expr_height(x) == i).then(set_(expr_height(num)).to(i + 1))
    yield rule(eq(num).to(abs(x)), expr_height(x) == i).then(set_(expr_height(num)).to(i + 1))


def _guard(guarded: bool, *terms: Num) -> tuple[Unit, ...]:
    if not guarded:
        return ()
    return tuple(expr_height(term) <= i64(HEIGHT_LIMIT) for term in terms)


def _basic_rules_impl(guarded: bool, x: Num, y: Num, z: Num, a: Num, b: Num, c: Num, d: Num, e: f64) -> Iterable[RewriteOrRule]:
    yield rewrite(x + y).to(y + x, *_guard(guarded, x, y))
    yield rewrite(x * y).to(y * x, *_guard(guarded, x, y))
    yield rewrite(x + (y + z)).to((x + y) + z, *_guard(guarded, x, y, z))
    yield rewrite(x * (y * z)).to((x * y) * z, *_guard(guarded, x, y, z))
    yield rewrite(x * (y / z)).to((x * y) / z, *_guard(guarded, x, y, z))
    yield rewrite((x * y) / z).to(x * (y / z), *_guard(guarded, x, y, z))
    yield rewrite((a * x) * (b * y)).to((a * b) * (x * y), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite(a * x + b).to(a * (x + (b / a)), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite(a * x - b).to(a * (x - (b / a)), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite(b - (a * x)).to(a * ((b / a) - x), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite(a * x + (b * y)).to(a * (x + ((b / a) * y)), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite(a * x - (b * y)).to(a * (x - ((b / a) * y)), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite(a * x + (b / y)).to(a * (x + ((b / a) / y)), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite(a * x - (b / y)).to(a * (x - ((b / a) / y)), const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))

    yield rewrite(a / (b * x)).to((a / b) / x, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite(x / (b * y)).to((1.0 / b) * x / y, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, b, x, y))
    yield rewrite((x / a) + b).to((x + (b * a)) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite((x / a) - b).to((x - (b * a)) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite(b - (x / a)).to(((b * a) - x) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, a, b, x))
    yield rewrite((x / a) + (b * y)).to((x + ((b * a) * y)) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite((x / a) + (y / b)).to((x + (y / (b * a))) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite((x / a) - (b * y)).to((x - ((b * a) * y)) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite((x / a) - (b / y)).to((x - (y / (b * a))) / a, const_value(a) != OptionalF64.none, const_value(b) != OptionalF64.none, const_value(x) == OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, a, b, x, y))
    yield rewrite((b + (a * x)) / (c + (d * y))).to(
        (a / d) * (((b / a) + x) / ((c / d) + y)),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(c) != OptionalF64.none,
        const_value(d) != OptionalF64.none,
        *_guard(guarded, a, b, c, d, x, y),
    )
    yield rewrite((b + x) / (c + (d * y))).to(
        (1.0 / d) * ((b + x) / ((c / d) + y)),
        const_value(b) != OptionalF64.none,
        const_value(c) != OptionalF64.none,
        const_value(d) != OptionalF64.none,
        *_guard(guarded, b, c, d, x, y),
    )

    yield rewrite(0.0 + x).to(x)
    yield rewrite(x - 0.0).to(x)
    yield rewrite(1.0 * x).to(x)
    yield rewrite(0.0 * x).to(0.0)
    yield rewrite(0.0 / x).to(0.0)
    yield rewrite(x - x).to(0.0)
    yield rewrite(x / x).to(1.0, const_value(x) == OptionalF64.none, *_guard(guarded, x))
    yield rewrite(x / x).to(1.0, const_value(x) == OptionalF64.some(e), e > 0.0, *_guard(guarded, x))
    yield rewrite(x / x).to(1.0, const_value(x) == OptionalF64.some(e), e < 0.0, *_guard(guarded, x))
    yield rewrite((x * y) + (x * z)).to(x * (y + z), *_guard(guarded, x, y, z))
    yield rewrite(x - (y + z)).to((x - y) - z, *_guard(guarded, x, y, z))
    yield rewrite(x - (y - z)).to((x - y) + z, *_guard(guarded, x, y, z))
    yield rewrite(Num(-1.0) * (x + y)).to((Num(-1.0) * x) - y, *_guard(guarded, x, y))
    yield rewrite(x - a).to(x + (Num(-1.0) * a), const_value(a) != OptionalF64.none, const_value(x) == OptionalF64.none, *_guard(guarded, x, a))
    yield rewrite(x - (a * y)).to(x + ((Num(-1.0) * a) * y), const_value(a) != OptionalF64.none, const_value(y) == OptionalF64.none, *_guard(guarded, x, a, y))
    yield rewrite((1.0 / x) * (1.0 / y)).to(1.0 / (x * y), *_guard(guarded, x, y))
    yield rewrite(x * (1.0 / x)).to(1.0, const_value(x) == OptionalF64.none, *_guard(guarded, x))
    yield rewrite(x * (1.0 / x)).to(1.0, const_value(x) == OptionalF64.some(e), e > 0.0, *_guard(guarded, x))
    yield rewrite(x * (1.0 / x)).to(1.0, const_value(x) == OptionalF64.some(e), e < 0.0, *_guard(guarded, x))
    yield rewrite(x - ((-1.0) * y)).to(x + y, const_value(y) == OptionalF64.none, *_guard(guarded, x, y))
    yield rewrite(x + (Num(-1.0) * y)).to(x - y, const_value(y) == OptionalF64.none, *_guard(guarded, x, y))
    yield rewrite(0.0 - x).to(Num(-1.0) * x, const_value(x) == OptionalF64.none, *_guard(guarded, x))


def _fun_rules_impl(guarded: bool, x: Num, y: Num, a: Num, b: Num, c: f64, d: f64) -> Iterable[RewriteOrRule]:
    yield rewrite(log(x * y)).to(log(x) + log(y), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0, *_guard(guarded, x, y))
    yield rewrite(log(x * y)).to(log(x) + log(y), const_value(y) == OptionalF64.some(d), d >= 0.0, d != 0.0, *_guard(guarded, x, y))
    yield rewrite(log(x / y)).to(log(x) - log(y), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0, *_guard(guarded, x, y))
    yield rewrite(log(x / y)).to(log(x) - log(y), const_value(y) == OptionalF64.some(d), d >= 0.0, d != 0.0, *_guard(guarded, x, y))
    yield rewrite(log(x**y)).to(y * log(x), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0, *_guard(guarded, x, y))
    yield rewrite(log(1.0)).to(0.0)
    yield rewrite(log(sqrt(x))).to(0.5 * log(x), const_value(x) == OptionalF64.none, *_guard(guarded, x))
    yield rewrite(log(exp(x))).to(x, const_value(x) == OptionalF64.none, *_guard(guarded, x))
    yield rewrite(exp(log(x))).to(x, const_value(x) == OptionalF64.none, *_guard(guarded, x))
    yield rewrite(x ** 0.5).to(sqrt(x), *_guard(guarded, x))
    yield rewrite(sqrt(a * x)).to(sqrt(a) * sqrt(x), const_value(a) == OptionalF64.some(c), c >= 0.0, *_guard(guarded, a, x))
    yield rewrite(sqrt(a * (x - y))).to(sqrt(Num(-1.0) * a) * sqrt(y - x), const_value(a) == OptionalF64.some(c), c < 0.0, *_guard(guarded, a, x, y))
    yield rewrite(sqrt(a * (b + y))).to(sqrt(Num(-1.0) * a) * sqrt(b - y), const_value(a) == OptionalF64.some(c), c < 0.0, const_value(b) == OptionalF64.some(d), d < 0.0, *_guard(guarded, a, b, y))
    yield rewrite(sqrt(a / x)).to(sqrt(a) / sqrt(x), const_value(a) == OptionalF64.some(c), c >= 0.0, *_guard(guarded, a, x))
    yield rewrite(abs(x * y)).to(abs(x) * abs(y), *_guard(guarded, x, y))


@basic_rules.register
def _basic_rewrites(x: Num, y: Num, z: Num, a: Num, b: Num, c: Num, d: Num, e: f64) -> Iterable[RewriteOrRule]:
    yield from _basic_rules_impl(False, x, y, z, a, b, c, d, e)


@basic_rules_height_guard.register
def _basic_rewrites_height_guard(x: Num, y: Num, z: Num, a: Num, b: Num, c: Num, d: Num, e: f64) -> Iterable[RewriteOrRule]:
    yield from _basic_rules_impl(True, x, y, z, a, b, c, d, e)


@fun_rules.register
def _fun_rewrites(x: Num, y: Num, a: Num, b: Num, c: f64, d: f64) -> Iterable[RewriteOrRule]:
    yield from _fun_rules_impl(False, x, y, a, b, c, d)


@fun_rules_height_guard.register
def _fun_rewrites_height_guard(x: Num, y: Num, a: Num, b: Num, c: f64, d: f64) -> Iterable[RewriteOrRule]:
    yield from _fun_rules_impl(True, x, y, a, b, c, d)


@dataclass(frozen=True)
class PaperPipelineReport:
    mode: Mode
    status: str
    passes: int
    total_sec: float
    total_size: int
    node_count: int
    eclass_count: int
    before_nodes: int
    before_params: int
    after_nodes: int
    after_params: int
    rendered: str
    extracted: Num
    extracted_cost: int


def _normalize_expression(source: str) -> str:
    return source.strip().replace("^", "**")


def _from_ast(node: ast.AST) -> Num:  # noqa: C901, PLR0911, PLR0912
    if isinstance(node, ast.Expression):
        return _from_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return Num(float(node.value))
        msg = f"Unsupported constant: {node.value!r}"
        raise ValueError(msg)
    if isinstance(node, ast.Name):
        return Num.var(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int | float):
            return Num(float(-node.operand.value))
        return neg(_from_ast(node.operand))
    if isinstance(node, ast.BinOp):
        lhs = _from_ast(node.left)
        rhs = _from_ast(node.right)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.Div):
            return lhs / rhs
        if isinstance(node.op, ast.Pow):
            return lhs**rhs
        msg = f"Unsupported binary operator: {ast.dump(node.op)}"
        raise TypeError(msg)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            msg = f"Unsupported call target: {ast.dump(node.func)}"
            raise TypeError(msg)
        name = node.func.id
        args = [_from_ast(arg) for arg in node.args]
        if name == "exp":
            return exp(args[0])
        if name == "log":
            return log(args[0])
        if name == "sqrt":
            return sqrt(args[0])
        if name == "abs":
            return abs(args[0])
        if name == "plog":
            return plog(args[0])
        if name == "square":
            return square(args[0])
        if name == "cube":
            return cube(args[0])
        msg = f"Unsupported function call: {name}"
        raise ValueError(msg)
    msg = f"Unsupported AST node: {ast.dump(node)}"
    raise TypeError(msg)


def parse_expression(source: str) -> Num:
    return _from_ast(ast.parse(_normalize_expression(source), mode="eval"))


def render_num(num: Num) -> str:  # noqa: C901, PLR0911, PLR0912
    match get_callable_args(num, Num):
        case (value_expr,):
            return _render_float(float(cast("f64", value_expr).value))
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (name_expr,):
            return cast("String", name_expr).value
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs))} + {render_num(cast('Num', rhs))})"
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs))} - {render_num(cast('Num', rhs))})"
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs))} * {render_num(cast('Num', rhs))})"
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs))} / {render_num(cast('Num', rhs))})"
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs))} ** {render_num(cast('Num', rhs))})"
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,):
            return f"exp({render_num(cast('Num', inner))})"
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,):
            return f"log({render_num(cast('Num', inner))})"
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,):
            return f"sqrt({render_num(cast('Num', inner))})"
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            return f"abs({render_num(cast('Num', inner))})"
        case _:
            pass
    msg = f"Unsupported Num node for rendering: {num!r}"
    raise TypeError(msg)


def _eval_num(num: Num, env: Mapping[str, float]) -> float | None:  # noqa: C901, PLR0911, PLR0912
    match get_callable_args(num, Num):
        case (value_expr,):
            value = float(cast("f64", value_expr).value)
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (name_expr,):
            value = env.get(cast("String", name_expr).value)
            if value is None or not math.isfinite(value):
                return None
            return float(value)
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            left = _eval_num(cast("Num", lhs), env)
            right = _eval_num(cast("Num", rhs), env)
            return None if left is None or right is None else _combine_binary(left, right, "add")
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            left = _eval_num(cast("Num", lhs), env)
            right = _eval_num(cast("Num", rhs), env)
            return None if left is None or right is None else _combine_binary(left, right, "sub")
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            left = _eval_num(cast("Num", lhs), env)
            right = _eval_num(cast("Num", rhs), env)
            return None if left is None or right is None else _combine_binary(left, right, "mul")
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            left = _eval_num(cast("Num", lhs), env)
            right = _eval_num(cast("Num", rhs), env)
            return None if left is None or right is None else _combine_binary(left, right, "div")
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            left = _eval_num(cast("Num", lhs), env)
            right = _eval_num(cast("Num", rhs), env)
            return None if left is None or right is None else _combine_binary(left, right, "pow")
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,):
            value = _eval_num(cast("Num", inner), env)
            if value is None:
                return None
            try:
                result = math.exp(value)
            except OverflowError:
                return None
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,):
            value = _eval_num(cast("Num", inner), env)
            if value is None or value <= 0.0:
                return None
            result = math.log(value)
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,):
            value = _eval_num(cast("Num", inner), env)
            if value is None or value < 0.0:
                return None
            result = math.sqrt(value)
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            value = _eval_num(cast("Num", inner), env)
            if value is None:
                return None
            result = abs(value)
            return result if math.isfinite(result) else None
        case _:
            pass
    msg = f"Unsupported Num node for evaluation: {num!r}"
    raise TypeError(msg)


def count_params(num: Num) -> int:  # noqa: C901, PLR0911, PLR0912
    match get_callable_args(num, Num):
        case (_value_expr,):
            return 1
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (_name_expr,):
            return 0
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            return count_params(cast("Num", lhs)) + count_params(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            return count_params(cast("Num", lhs)) + count_params(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            return count_params(cast("Num", lhs)) + count_params(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            return count_params(cast("Num", lhs)) + count_params(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            return count_params(cast("Num", lhs))
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,):
            return count_params(cast("Num", inner))
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,):
            return count_params(cast("Num", inner))
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,):
            return count_params(cast("Num", inner))
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            return count_params(cast("Num", inner))
        case _:
            pass
    msg = f"Unsupported Num node while counting parameters: {num!r}"
    raise TypeError(msg)


def count_nodes(num: Num) -> int:  # noqa: C901, PLR0911, PLR0912
    match get_callable_args(num, Num):
        case (_value_expr,):
            return 1
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (_name_expr,):
            return 1
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            return 1 + count_nodes(cast("Num", lhs)) + count_nodes(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            return 1 + count_nodes(cast("Num", lhs)) + count_nodes(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            return 1 + count_nodes(cast("Num", lhs)) + count_nodes(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            return 1 + count_nodes(cast("Num", lhs)) + count_nodes(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            return 1 + count_nodes(cast("Num", lhs)) + count_nodes(cast("Num", rhs))
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,):
            return 1 + count_nodes(cast("Num", inner))
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,):
            return 1 + count_nodes(cast("Num", inner))
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,):
            return 1 + count_nodes(cast("Num", inner))
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            return 1 + count_nodes(cast("Num", inner))
        case _:
            pass
    msg = f"Unsupported Num node while counting nodes: {num!r}"
    raise TypeError(msg)


def _serialized_counts(egraph: egglog.EGraph) -> tuple[int, int]:
    payload = json.loads(egraph._serialize().to_json())
    return len(payload.get("nodes", {})), len(payload.get("class_data", {}))


def _rules_for_mode(mode: Mode) -> egglog.Ruleset:
    if mode == "egglog-baseline":
        return const_analysis_rules | basic_rules | fun_rules
    return const_analysis_rules | height_analysis_rules | basic_rules_height_guard | fun_rules_height_guard


def _run_single_pass(num: Num, mode: Mode) -> tuple[Num, int, int, int, int, float]:
    egraph = egglog.EGraph()
    egraph.register(num)
    scheduler = back_off(match_limit=BACKOFF_MATCH_LIMIT, ban_length=BACKOFF_BAN_LENGTH)
    start = time.perf_counter()
    egraph.run(run(_rules_for_mode(mode), scheduler=scheduler).saturate())
    elapsed = time.perf_counter() - start
    extracted, cost = egraph.extract(num, include_cost=True)
    total_size = sum(size for _, size in egraph.all_function_sizes())
    node_count, eclass_count = _serialized_counts(egraph)
    return extracted, int(cost), total_size, node_count, eclass_count, elapsed


def run_paper_pipeline(num: Num, *, mode: Mode) -> PaperPipelineReport:
    before_nodes = count_nodes(num)
    before_params = count_params(num)
    current = num
    last_cost = 0
    total_size = 0
    node_count = 0
    eclass_count = 0
    total_sec = 0.0
    passes = 0
    status = "saturated"
    for pass_index in range(1, MAX_PASSES + 1):
        extracted, last_cost, total_size, node_count, eclass_count, elapsed = _run_single_pass(current, mode)
        total_sec += elapsed
        passes = pass_index
        if render_num(extracted) == render_num(current):
            current = extracted
            break
        current = extracted
    after_nodes = count_nodes(current)
    after_params = count_params(current)
    return PaperPipelineReport(
        mode=mode,
        status=status,
        passes=passes,
        total_sec=total_sec,
        total_size=total_size,
        node_count=node_count,
        eclass_count=eclass_count,
        before_nodes=before_nodes,
        before_params=before_params,
        after_nodes=after_nodes,
        after_params=after_params,
        rendered=render_num(current),
        extracted=current,
        extracted_cost=last_cost,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("egglog-baseline", "egglog-height-guard"), required=True)
    parser.add_argument("--expr", required=True)
    args = parser.parse_args()
    report = run_paper_pipeline(parse_expression(args.expr), mode=cast("Mode", args.mode))
    payload = {
        "mode": report.mode,
        "status": report.status,
        "passes": report.passes,
        "total_sec": report.total_sec,
        "total_size": report.total_size,
        "node_count": report.node_count,
        "eclass_count": report.eclass_count,
        "before_nodes": report.before_nodes,
        "before_params": report.before_params,
        "after_nodes": report.after_nodes,
        "after_params": report.after_params,
        "rendered": report.rendered,
        "extracted_cost": report.extracted_cost,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    _cli()
