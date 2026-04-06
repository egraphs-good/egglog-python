# mypy: disable-error-code="empty-body"

"""
Helpers for reproducing the paper-era param-eq EqSat pipeline in egglog.

The archived source of truth is `param-eq-haskell/src/FixTree.hs`, together
with the reporting code in `param-eq-haskell/src/Main.hs`.

This module mirrors that paper experiment harness only.

Design notes for future agents:

- The paper-facing Haskell source is `param-eq-haskell/src/FixTree.hs`.
- That Haskell implementation keeps analysis inside hegg rebuild; this Python
  file has to approximate that with explicit Egglog rulesets plus a schedule.
- Numeric literals stay as constants in the EqSat language, matching
  `FixTree.toFixTree (Const x) = ConstF x`. The paper only turns constants into
  fitted parameters later for reporting, via
  `recountParams . replaceConstsWithParams` in `Main.hs`.
- Mixed classes are intentional and paper-faithful. A class may contain both a
  constant representative and a non-constant representative while its analysis
  still says "not constant". Only classes whose analysis is definitively
  constant get pruned to leaf nodes.

Quick symbol map back to the Haskell file:

- `joinA` -> `join_const_value`
- `evalConstant` -> `const_seed_rules | const_propagation_rules`
- `modifyA` -> `const_prune_rules`
- `rewritesBasic` -> `_basic_rewrites`
- `rewritesFun` -> `_fun_rewrites`
- `rewriteTree` -> `_run_single_pass_*`
- `simplifyE` -> `run_paper_pipeline`
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
from egglog.egraph import FactLike

__all__ = [
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


MAX_PASSES = 2
HASKELL_INNER_ITERATION_LIMIT = 30
BACKOFF_MATCH_LIMIT = 2500
BACKOFF_BAN_LENGTH = 30
CONST_MERGE_TOLERANCE = 1e-6

Mode = Literal[
    "egglog-baseline",
    "egglog-haskell-literal",
    "no-haskell-backoff",
    "no-graph-size-stop",
    "no-bound-scheduler",
    "no-fresh-rematch",
]


# Language and ruleset declarations
language_rules = ruleset(name="param_eq_hegg_lang")
const_merge_rules = ruleset(name="param_eq_hegg_const_merge")
const_seed_rules = ruleset(name="param_eq_hegg_const_seed")
const_propagation_rules = ruleset(name="param_eq_hegg_const_propagation")
const_prune_rules = ruleset(name="param_eq_hegg_const_prune")
basic_add_comm_rules = ruleset(name="param_eq_hegg_basic_add_comm")
basic_mul_comm_rules = ruleset(name="param_eq_hegg_basic_mul_comm")
basic_add_assoc_rules = ruleset(name="param_eq_hegg_basic_add_assoc")
basic_mul_assoc_rules = ruleset(name="param_eq_hegg_basic_mul_assoc")
basic_mul_div_rules = ruleset(name="param_eq_hegg_basic_mul_div")
basic_product_regroup_rules = ruleset(name="param_eq_hegg_basic_product_regroup")
basic_other_rules = ruleset(name="param_eq_hegg_basic_other")
fun_rules = ruleset(name="param_eq_hegg_fun")


class OptionalF64(Expr, ruleset=language_rules):
    """
    Explicit stand-in for Haskell's `Maybe Double` analysis domain.

    `none` corresponds to `Nothing`, `some(x)` corresponds to `Just x`.
    """

    none: ClassVar[OptionalF64]

    @classmethod
    def some(cls, value: f64Like) -> OptionalF64: ...


class Num(Expr, ruleset=language_rules):
    """
    Paper EqSat language subset.

    This is deliberately closer to `FixTree`'s `SRTreeF` than to the broader
    experimental translations that were removed during cleanup. The paper
    corpus only needs constants,
    variables, arithmetic, and a small unary-function set.
    """

    @method(cost=5)
    def __init__(self, value: f64Like) -> None: ...

    __match_args__ = ("value",)

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> float:
        match get_callable_args(self, Num):
            case (f64(value),):
                return value
        raise ExprValueError(self, "Num")

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


NumLike: TypeAlias = Num | StringLike | f64Like | i64Like

converter(f64, Num, Num)
converter(i64, Num, lambda value: Num(f64.from_i64(value)))
converter(String, Num, Num.var)


# Analysis domain and merge
@function
def join_const_value(old: OptionalF64, new: OptionalF64) -> OptionalF64: ...


@function(
    merge=lambda old, new: join_const_value(old, new),
)
def const_value(num: Num) -> OptionalF64: ...


# Surface syntax helpers
# Convenience wrappers keep the parser and rewrites close to the Haskell
# surface syntax while still using the operator-overloaded Python DSL.


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
    return value**2


def cube(x: NumLike) -> Num:
    value = convert(x, Num)
    return value**3


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


# Analysis algebra
@const_merge_rules.register
def _const_merge(
    merged: OptionalF64,
    other: OptionalF64,
    a: f64,
    b: f64,
) -> Iterable[RewriteOrRule]:
    """
    Mirror `FixTree.joinA` on `Maybe Double`.

    The Haskell behavior is:
    - `Nothing` dominates any merge
    - `Just a` with `Just b` is allowed only when the constants agree up to the
      tolerance used in the paper code
    - differing constants are an invariant violation

    This is the key reason mixed classes remain possible: if one representative
    analyzes to `none` and another to `some(a)`, the class analysis stays
    `none`.
    """
    yield rewrite(join_const_value(OptionalF64.none, other)).to(OptionalF64.none)
    yield rewrite(join_const_value(other, OptionalF64.none)).to(OptionalF64.none)
    yield rewrite(join_const_value(OptionalF64.some(a), OptionalF64.some(a))).to(OptionalF64.some(a))
    yield rewrite(join_const_value(OptionalF64.some(a), OptionalF64.some(b))).to(
        OptionalF64.some(a),
        eq(a).to(b),
    )
    yield rewrite(join_const_value(OptionalF64.some(a), OptionalF64.some(b))).to(
        OptionalF64.some(a),
        abs(a - b) <= CONST_MERGE_TOLERANCE,
    )
    yield rule(
        eq(merged).to(join_const_value(OptionalF64.some(a), OptionalF64.some(b))),
        abs(a - b) > CONST_MERGE_TOLERANCE,
    ).then(panic("Merged different constant values"))


@const_seed_rules.register
def _const_seed(
    num: Num,
    a: f64,
    s: String,
) -> Iterable[RewriteOrRule]:
    """
    Seed the explicit analysis with leaf facts.

    This corresponds to the leaf cases in `FixTree.evalConstant`:
    - `ConstF x -> Just x`
    - `VarF _ -> Nothing`

    The paper harness's benchmark expressions do not materialize `ParamF`
    before EqSat, so there is no separate parameter leaf here.
    """
    yield rule(eq(num).to(Num(a))).then(set_(const_value(num)).to(OptionalF64.some(a)))
    yield rule(eq(num).to(Num.var(s))).then(set_(const_value(num)).to(OptionalF64.none))


@const_propagation_rules.register
def _const_propagation(
    num: Num,
    x: Num,
    y: Num,
    a: f64,
    b: f64,
) -> Iterable[RewriteOrRule]:
    """
    Explicit approximation of `FixTree.makeA` + `evalConstant`.

    In Haskell this logic is one algebra over `SRTreeF (Maybe Double)` and runs
    as part of rebuild. In Egglog we spell it as rules:
    - if any required child is `none`, the parent becomes `none`
    - if all required children are `some`, compute the folded constant
    - when a folded constant exists, union the class with `Num(constant)`
    """
    yield rule(eq(num).to(x + y), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x + y), const_value(y) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x + y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a + b)),
        union(num).with_(Num(a + b)),
    )
    yield rule(eq(num).to(x - y), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x - y), const_value(y) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x - y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a - b)),
        union(num).with_(Num(a - b)),
    )
    yield rule(eq(num).to(x * y), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x * y), const_value(y) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x * y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a * b)),
        union(num).with_(Num(a * b)),
    )
    yield rule(eq(num).to(x / y), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x / y), const_value(y) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(
        eq(num).to(x / y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        eq(b).to(f64(0.0)),
    ).then(
        set_(const_value(num)).to(OptionalF64.none),
    )
    yield rule(
        eq(num).to(x / y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        b > 0.0,
    ).then(
        set_(const_value(num)).to(OptionalF64.some(a / b)),
        union(num).with_(Num(a / b)),
    )
    yield rule(
        eq(num).to(x / y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        b < 0.0,
    ).then(
        set_(const_value(num)).to(OptionalF64.some(a / b)),
        union(num).with_(Num(a / b)),
    )
    yield rule(eq(num).to(x**y), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(x**y), const_value(y) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(
        eq(num).to(x**y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        a < 0.0,
        eq(f64.from_i64(b.to_i64())).to(b),
    ).then(
        set_(const_value(num)).to(OptionalF64.some(a**b)),
        union(num).with_(Num(a**b)),
    )
    yield rule(
        eq(num).to(x**y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        a < 0.0,
        ne(f64.from_i64(b.to_i64())).to(b),
    ).then(
        set_(const_value(num)).to(OptionalF64.none),
    )
    yield rule(
        eq(num).to(x**y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        eq(a).to(f64(0.0)),
        b < 0.0,
    ).then(
        set_(const_value(num)).to(OptionalF64.none),
    )
    yield rule(
        eq(num).to(x**y),
        const_value(x) == OptionalF64.some(a),
        const_value(y) == OptionalF64.some(b),
        eq(a).to(f64(0.0)),
        b >= 0.0,
    ).then(
        set_(const_value(num)).to(OptionalF64.some(a**b)),
        union(num).with_(Num(a**b)),
    )
    yield rule(
        eq(num).to(x**y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b), a > 0.0
    ).then(
        set_(const_value(num)).to(OptionalF64.some(a**b)),
        union(num).with_(Num(a**b)),
    )
    yield rule(eq(num).to(exp(x)), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(exp(x)), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(a.exp())),
        union(num).with_(Num(a.exp())),
    )
    yield rule(eq(num).to(log(x)), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(log(x)), const_value(x) == OptionalF64.some(a), a <= 0.0).then(
        set_(const_value(num)).to(OptionalF64.none),
    )
    yield rule(eq(num).to(log(x)), const_value(x) == OptionalF64.some(a), a > 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a.log())),
        union(num).with_(Num(a.log())),
    )
    yield rule(eq(num).to(sqrt(x)), const_value(x) == OptionalF64.none).then(
        set_(const_value(num)).to(OptionalF64.none)
    )
    yield rule(eq(num).to(sqrt(x)), const_value(x) == OptionalF64.some(a), a < 0.0).then(
        set_(const_value(num)).to(OptionalF64.none),
    )
    yield rule(eq(num).to(sqrt(x)), const_value(x) == OptionalF64.some(a), a >= 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a.sqrt())),
        union(num).with_(Num(a.sqrt())),
    )
    yield rule(eq(num).to(abs(x)), const_value(x) == OptionalF64.none).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(abs(x)), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(abs(a))),
        union(num).with_(Num(abs(a))),
    )


# Guard helpers
GuardConditions: TypeAlias = tuple[FactLike, ...]
GuardCases: TypeAlias = tuple[GuardConditions, ...]


def is_const(num: Num, value: f64) -> GuardConditions:
    return (const_value(num) == OptionalF64.some(value),)


def is_not_const(num: Num) -> GuardConditions:
    return (const_value(num) == OptionalF64.none,)


def _is_nonnegative_const(num: Num, value: f64) -> GuardConditions:
    return (*is_const(num, value), value >= 0.0)


def _is_positive_const(num: Num, value: f64) -> GuardConditions:
    return (*is_const(num, value), value > 0.0)


def is_negative(num: Num, value: f64) -> GuardConditions:
    return (*is_const(num, value), value < 0.0)


def is_not_zero(num: Num, value: f64) -> GuardCases:
    return (
        is_not_const(num),
        _is_positive_const(num, value),
        is_negative(num, value),
    )


def is_not_neg_consts(left: Num, right: Num, left_value: f64, right_value: f64) -> GuardCases:
    return (
        _is_nonnegative_const(left, left_value),
        _is_nonnegative_const(right, right_value),
    )


# `rewritesBasic`
@const_prune_rules.register
def _const_prune(
    num: Num,
    x: Num,
    y: Num,
    a: f64,
) -> Iterable[RewriteOrRule]:
    """
    Approximate `FixTree.modifyA`.

    Haskell does two things once an e-class analysis becomes `Just d`:
    - inserts the constant representative `ConstF d`
    - prunes the class down to leaf e-nodes

    Egglog cannot filter a class's node set directly, so we approximate that by
    deleting composite representatives that are already known constant. This is
    intentionally weaker than the hegg implementation, but it serves the same
    purpose: keep truly constant classes from carrying around redundant call
    nodes.
    """
    yield rule(eq(num).to(x + y), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(x + y),
    )
    yield rule(eq(num).to(x - y), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(x - y),
    )
    yield rule(eq(num).to(x * y), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(x * y),
    )
    yield rule(eq(num).to(x / y), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(x / y),
    )
    yield rule(eq(num).to(x**y), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(x**y),
    )
    yield rule(eq(num).to(exp(x)), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(exp(x)),
    )
    yield rule(eq(num).to(log(x)), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(log(x)),
    )
    yield rule(eq(num).to(sqrt(x)), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(sqrt(x)),
    )
    yield rule(eq(num).to(abs(x)), const_value(num) == OptionalF64.some(a)).then(
        union(num).with_(Num(a)),
        delete(abs(x)),
    )


@basic_add_comm_rules.register
def _basic_add_comm(x: Num, y: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x + y).to(y + x)


@basic_mul_comm_rules.register
def _basic_mul_comm(x: Num, y: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x * y).to(y * x)


@basic_add_assoc_rules.register
def _basic_add_assoc(x: Num, y: Num, z: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x + (y + z)).to((x + y) + z)


@basic_mul_assoc_rules.register
def _basic_mul_assoc(x: Num, y: Num, z: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x * (y * z)).to((x * y) * z)


@basic_mul_div_rules.register
def _basic_mul_div(x: Num, y: Num, z: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x * (y / z)).to((x * y) / z)
    yield rewrite((x * y) / z).to(x * (y / z))


@basic_product_regroup_rules.register
def _basic_product_regroup(a: Num, b: Num, x: Num, y: Num, ca: f64, cb: f64) -> Iterable[RewriteOrRule]:
    yield rewrite((a * x) * (b * y)).to(
        (a * b) * (x * y),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )


@basic_other_rules.register
def _basic_rewrites(
    x: Num,
    y: Num,
    z: Num,
    a: Num,
    b: Num,
    c: Num,
    d: Num,
    ca: f64,
    cb: f64,
    cc: f64,
    cd: f64,
    e: f64,
) -> Iterable[RewriteOrRule]:
    """
    Translation of `FixTree.rewritesBasic`.

    Guard style differs from Haskell only where Egglog soundness requires it:
    we avoid `!= none`-style class tests and instead match explicit
    `const_value(...) == some(...)` or `== none` cases.
    """
    yield rewrite(a * x + b).to(
        a * (x + (b / a)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite(a * x - b).to(
        a * (x - (b / a)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite(b - (a * x)).to(
        a * ((b / a) - x),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite(a * x + (b * y)).to(
        a * (x + ((b / a) * y)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite(a * x - (b * y)).to(
        a * (x - ((b / a) * y)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite(a * x + (b / y)).to(
        a * (x + ((b / a) / y)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite(a * x - (b / y)).to(
        a * (x - ((b / a) / y)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite(a / (b * x)).to(
        (a / b) / x,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite(x / (b * y)).to(
        (Num(1.0) / b) * x / y,
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite((x / a) + b).to(
        (x + (b * a)) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite((x / a) - b).to(
        (x - (b * a)) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite(b - (x / a)).to(
        ((b * a) - x) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
    )
    yield rewrite((x / a) + (b * y)).to(
        (x + ((b * a) * y)) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite((x / a) + (y / b)).to(
        (x + (y / (b * a))) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite((x / a) - (b * y)).to(
        (x - ((b * a) * y)) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite((x / a) - (b / y)).to(
        (x - (y / (b * a))) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
    yield rewrite((b + (a * x)) / (c + (d * y))).to(
        (a / d) * (((b / a) + x) / ((c / d) + y)),
        *is_const(a, ca),
        *is_const(b, cb),
        *is_const(c, cc),
        *is_const(d, cd),
    )
    yield rewrite((b + x) / (c + (d * y))).to(
        (Num(1.0) / d) * ((b + x) / ((c / d) + y)),
        *is_const(b, cb),
        *is_const(c, cc),
        *is_const(d, cd),
    )

    yield rewrite(Num(0.0) + x).to(x)
    yield rewrite(x - Num(0.0)).to(x)
    yield rewrite(Num(1.0) * x).to(x)
    yield rewrite(Num(0.0) * x).to(Num(0.0))
    yield rewrite(Num(0.0) / x).to(Num(0.0))
    yield rewrite(x - x).to(Num(0.0))
    for guard in is_not_zero(x, ca):
        yield rewrite(x / x).to(Num(1.0), *guard)
    yield rewrite((x * y) + (x * z)).to(x * (y + z))
    yield rewrite(x - (y + z)).to((x - y) - z)
    yield rewrite(x - (y - z)).to((x - y) + z)
    yield rewrite(Num(-1.0) * (x + y)).to((Num(-1.0) * x) - y)
    yield rewrite(x - a).to(
        x + (Num(-1.0) * a),
        *is_const(a, ca),
        *is_not_const(x),
    )
    yield rewrite(x - (a * y)).to(
        x + ((Num(-1.0) * a) * y),
        *is_const(a, ca),
        *is_not_const(y),
    )
    yield rewrite((Num(1.0) / x) * (Num(1.0) / y)).to(Num(1.0) / (x * y))
    for guard in is_not_zero(x, ca):
        yield rewrite(x * (Num(1.0) / x)).to(Num(1.0), *guard)
    yield rewrite(x - (Num(-1.0) * y)).to(x + y, *is_not_const(y))
    yield rewrite(x + (Num(-1.0) * y)).to(x - y, *is_not_const(y))
    yield rewrite(Num(0.0) - x).to(Num(-1.0) * x, *is_not_const(x))


# `rewritesFun`
@fun_rules.register
def _fun_rewrites(x: Num, y: Num, a: Num, b: Num, c: f64, d: f64) -> Iterable[RewriteOrRule]:
    """
    Translation of `FixTree.rewritesFun`.

    The unary function subset here is the subset exercised by the archived
    paper corpora.
    """
    yield rewrite(log(x * y)).to(
        log(x) + log(y),
        const_value(x) == OptionalF64.some(c),
        c > 0.0,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite(log(x * y)).to(
        log(x) + log(y),
        const_value(x) == OptionalF64.some(c),
        c > 0.0,
        const_value(y) == OptionalF64.some(d),
        d > 0.0,
    )
    yield rewrite(log(x * y)).to(
        log(x) + log(y),
        const_value(y) == OptionalF64.some(c),
        c > 0.0,
        const_value(x) == OptionalF64.none,
    )
    yield rewrite(log(x * y)).to(
        log(x) + log(y),
        const_value(y) == OptionalF64.some(c),
        c > 0.0,
        const_value(x) == OptionalF64.some(d),
        d > 0.0,
    )
    yield rewrite(log(x * y)).to(
        log(x) + log(y),
        const_value(x) == OptionalF64.some(c),
        c > 0.0,
        const_value(y) == OptionalF64.some(d),
        d < 0.0,
    )
    yield rewrite(log(x * y)).to(
        log(x) + log(y),
        const_value(y) == OptionalF64.some(c),
        c > 0.0,
        const_value(x) == OptionalF64.some(d),
        d < 0.0,
    )
    yield rewrite(log(x / y)).to(
        log(x) - log(y),
        const_value(x) == OptionalF64.some(c),
        c > 0.0,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite(log(x / y)).to(
        log(x) - log(y),
        const_value(x) == OptionalF64.some(c),
        c > 0.0,
        const_value(y) == OptionalF64.some(d),
        d > 0.0,
    )
    yield rewrite(log(x / y)).to(
        log(x) - log(y),
        const_value(y) == OptionalF64.some(c),
        c > 0.0,
        const_value(x) == OptionalF64.none,
    )
    yield rewrite(log(x / y)).to(
        log(x) - log(y),
        const_value(y) == OptionalF64.some(c),
        c > 0.0,
        const_value(x) == OptionalF64.some(d),
        d > 0.0,
    )
    yield rewrite(log(x / y)).to(
        log(x) - log(y),
        const_value(x) == OptionalF64.some(c),
        c > 0.0,
        const_value(y) == OptionalF64.some(d),
        d < 0.0,
    )
    yield rewrite(log(x / y)).to(
        log(x) - log(y),
        const_value(y) == OptionalF64.some(c),
        c > 0.0,
        const_value(x) == OptionalF64.some(d),
        d < 0.0,
    )
    yield rewrite(log(x**y)).to(y * log(x), const_value(x) == OptionalF64.some(c), c > 0.0)
    yield rewrite(log(Num(1.0))).to(Num(0.0))
    yield rewrite(log(sqrt(x))).to(Num(0.5) * log(x), *is_not_const(x))
    yield rewrite(log(exp(x))).to(x, *is_not_const(x))
    yield rewrite(exp(log(x))).to(x, *is_not_const(x))
    yield rewrite(x ** Num(0.5)).to(sqrt(x))
    for guard in is_not_neg_consts(a, x, c, d):
        yield rewrite(sqrt(a * x)).to(sqrt(a) * sqrt(x), *guard)
    yield rewrite(sqrt(a * (x - y))).to(sqrt(Num(-1.0) * a) * sqrt(y - x), *is_negative(a, c))
    yield rewrite(sqrt(a * (b + y))).to(sqrt(Num(-1.0) * a) * sqrt(b - y), *is_negative(a, c), *is_negative(b, d))
    for guard in is_not_neg_consts(a, x, c, d):
        yield rewrite(sqrt(a / x)).to(sqrt(a) / sqrt(x), *guard)
    yield rewrite(abs(x * y)).to(abs(x) * abs(y))


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


@dataclass(frozen=True)
class ScheduleModeConfig:
    """Low-level EqSat controls for one `rewriteTree`-like pass."""

    persistent_scheduler: bool
    fresh_rematch: bool
    haskell_backoff: bool
    graph_size_stop: bool


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
    """
    Parse the archived paper expression syntax into the Egglog DSL.

    The input strings come from the normalized copies of the Haskell benchmark
    corpora. Numeric literals are kept as constants here; the paper's
    parameter-count reporting projects them to fitted parameters later.
    """
    return _from_ast(ast.parse(_normalize_expression(source), mode="eval"))


def render_num(num: Num) -> str:  # noqa: C901, PLR0911, PLR0912
    """Render a `Num` back into a Python-like surface syntax for reports."""
    match get_callable_args(num, Num):
        case (value_expr,) if isinstance(value_expr, f64):
            return _render_float(float(value_expr.value))
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (name_expr,) if isinstance(name_expr, String):
            return name_expr.value
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return f"({render_num(lhs)} + {render_num(rhs)})"
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return f"({render_num(lhs)} - {render_num(rhs)})"
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return f"({render_num(lhs)} * {render_num(rhs)})"
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return f"({render_num(lhs)} / {render_num(rhs)})"
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return f"({render_num(lhs)} ** {render_num(rhs)})"
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,) if isinstance(inner, Num):
            return f"exp({render_num(inner)})"
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,) if isinstance(inner, Num):
            return f"log({render_num(inner)})"
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,) if isinstance(inner, Num):
            return f"sqrt({render_num(inner)})"
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,) if isinstance(inner, Num):
            return f"abs({render_num(inner)})"
        case _:
            pass
    msg = f"Unsupported Num node for rendering: {num!r}"
    raise TypeError(msg)


def _eval_num(num: Num, env: Mapping[str, float]) -> float | None:  # noqa: C901, PLR0911, PLR0912
    """
    Structural evaluator for notebook/debugging use.

    This is not part of the EqSat algorithm. It is only used for artifact
    checks and mirrors the real-domain restrictions used elsewhere in the
    replication notebooks.
    """
    match get_callable_args(num, Num):
        case (value_expr,) if isinstance(value_expr, f64):
            value = float(value_expr.value)
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (name_expr,) if isinstance(name_expr, String):
            env_value = env.get(name_expr.value)
            if env_value is None or not math.isfinite(env_value):
                return None
            return float(env_value)
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            left = _eval_num(lhs, env)
            right = _eval_num(rhs, env)
            return None if left is None or right is None else _combine_binary(left, right, "add")
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            left = _eval_num(lhs, env)
            right = _eval_num(rhs, env)
            return None if left is None or right is None else _combine_binary(left, right, "sub")
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            left = _eval_num(lhs, env)
            right = _eval_num(rhs, env)
            return None if left is None or right is None else _combine_binary(left, right, "mul")
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            left = _eval_num(lhs, env)
            right = _eval_num(rhs, env)
            return None if left is None or right is None else _combine_binary(left, right, "div")
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            left = _eval_num(lhs, env)
            right = _eval_num(rhs, env)
            return None if left is None or right is None else _combine_binary(left, right, "pow")
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,) if isinstance(inner, Num):
            inner_value = _eval_num(inner, env)
            if inner_value is None:
                return None
            try:
                result = math.exp(inner_value)
            except OverflowError:
                return None
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,) if isinstance(inner, Num):
            inner_value = _eval_num(inner, env)
            if inner_value is None or inner_value <= 0.0:
                return None
            result = math.log(inner_value)
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,) if isinstance(inner, Num):
            inner_value = _eval_num(inner, env)
            if inner_value is None or inner_value < 0.0:
                return None
            result = math.sqrt(inner_value)
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,) if isinstance(inner, Num):
            inner_value = _eval_num(inner, env)
            if inner_value is None:
                return None
            result = abs(inner_value)
            return result if math.isfinite(result) else None
        case _:
            pass
    msg = f"Unsupported Num node for evaluation: {num!r}"
    raise TypeError(msg)


def count_params(num: Num) -> int:  # noqa: C901, PLR0911, PLR0912
    """
    Mirror the paper harness's parameter counting:

        recountParams . replaceConstsWithParams

    from ``param-eq-haskell/src/Main.hs``.

    That means numeric leaves are counted as parameters for reporting, but the
    EqSat language itself still sees them as constants. Power exponents follow
    the Haskell projection too: constant exponents are not counted as
    parameters, but non-constant exponent subexpressions are still traversed.
    """
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
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return count_params(lhs) + count_params(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return count_params(lhs) + count_params(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return count_params(lhs) + count_params(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return count_params(lhs) + count_params(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            match get_callable_args(rhs, Num):
                case (_value_expr,):
                    return count_params(lhs)
                case _:
                    return count_params(lhs) + count_params(rhs)
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,) if isinstance(inner, Num):
            return count_params(inner)
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,) if isinstance(inner, Num):
            return count_params(inner)
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,) if isinstance(inner, Num):
            return count_params(inner)
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,) if isinstance(inner, Num):
            return count_params(inner)
        case _:
            pass
    msg = f"Unsupported Num node while counting parameters: {num!r}"
    raise TypeError(msg)


def count_nodes(num: Num) -> int:  # noqa: C901, PLR0911, PLR0912
    """Count AST nodes in the rendered tree, matching the paper tables' style."""
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
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return 1 + count_nodes(lhs) + count_nodes(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return 1 + count_nodes(lhs) + count_nodes(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return 1 + count_nodes(lhs) + count_nodes(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return 1 + count_nodes(lhs) + count_nodes(rhs)
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs) if isinstance(lhs, Num) and isinstance(rhs, Num):
            return 1 + count_nodes(lhs) + count_nodes(rhs)
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,) if isinstance(inner, Num):
            return 1 + count_nodes(inner)
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,) if isinstance(inner, Num):
            return 1 + count_nodes(inner)
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,) if isinstance(inner, Num):
            return 1 + count_nodes(inner)
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,) if isinstance(inner, Num):
            return 1 + count_nodes(inner)
        case _:
            pass
    msg = f"Unsupported Num node while counting nodes: {num!r}"
    raise TypeError(msg)


def _serialized_counts(egraph: egglog.EGraph) -> tuple[int, int]:
    payload = json.loads(egraph._serialize().to_json())
    return len(payload.get("nodes", {})), len(payload.get("class_data", {}))


MODE_CONFIGS: dict[Mode, ScheduleModeConfig] = {
    "egglog-baseline": ScheduleModeConfig(
        persistent_scheduler=True,
        fresh_rematch=True,
        haskell_backoff=True,
        graph_size_stop=True,
    ),
    # Historical alias kept so old commands and notes still resolve to the
    # retained baseline.
    "egglog-haskell-literal": ScheduleModeConfig(
        persistent_scheduler=True,
        fresh_rematch=True,
        haskell_backoff=True,
        graph_size_stop=True,
    ),
    "no-haskell-backoff": ScheduleModeConfig(
        persistent_scheduler=True,
        fresh_rematch=True,
        haskell_backoff=False,
        graph_size_stop=True,
    ),
    "no-graph-size-stop": ScheduleModeConfig(
        persistent_scheduler=True,
        fresh_rematch=True,
        haskell_backoff=True,
        graph_size_stop=False,
    ),
    "no-bound-scheduler": ScheduleModeConfig(
        persistent_scheduler=False,
        fresh_rematch=True,
        haskell_backoff=True,
        graph_size_stop=True,
    ),
    "no-fresh-rematch": ScheduleModeConfig(
        persistent_scheduler=True,
        fresh_rematch=False,
        haskell_backoff=True,
        graph_size_stop=True,
    ),
}


# Haskell runs one `equalitySaturation' (BackoffScheduler 2500 30)` per outer
# pass, and `simplifyE` repeats that extracted result up to twice. The retained
# pipeline now mirrors that control flow directly through the low-level bound
# scheduler path in `_run_single_pass_haskell_literal`.
#
# The schedule objects below remain as a small bounded helper for local e-graph
# checks and historical experiments, but they are no longer the retained
# `run_paper_pipeline` baseline.
analysis_schedule = const_merge_rules | const_seed_rules | const_propagation_rules | const_prune_rules
basic_rules = (
    basic_add_comm_rules
    | basic_mul_comm_rules
    | basic_add_assoc_rules
    | basic_mul_assoc_rules
    | basic_mul_div_rules
    | basic_product_regroup_rules
    | basic_other_rules
)
# The retained baseline should keep the Haskell rewrite set intact. If the
# current schedule still diverges from Haskell, that should be diagnosed as a
# schedule or engine issue rather than by silently dropping `add_comm`.
baseline_basic_rules = basic_rules
scheduler = back_off(match_limit=BACKOFF_MATCH_LIMIT, ban_length=BACKOFF_BAN_LENGTH, egg_like=True)
rewrite_schedule = run(baseline_basic_rules | fun_rules, scheduler=scheduler)
analysis_rewrite_round = analysis_schedule.saturate() + rewrite_schedule
total_ruleset = scheduler.scope(
    analysis_rewrite_round
    + analysis_rewrite_round
    + analysis_rewrite_round
    + analysis_rewrite_round
    + analysis_schedule.saturate()
)
baseline_rewrite_ruleset = baseline_basic_rules | fun_rules
literal_rewrite_ruleset = basic_rules | fun_rules


def _run_single_pass_baseline(num: Num) -> tuple[Num, int, int, int, int, float]:
    """
    One retained paper-style EqSat pass.

    The retained baseline now matches the Haskell-style inner loop directly:
    one reused backoff scheduler, up to 30 rewrite iterations, and explicit
    analysis saturation after each rewrite step.
    """
    return _run_single_pass_for_mode(num, "egglog-baseline")


def _add_iteration_scheduler(
    egraph: egglog.EGraph,
    *,
    fresh_rematch: bool,
    haskell_backoff: bool,
) -> egglog.bindings.SchedulerHandle:
    """Create one scheduler instance for the current rewrite iteration."""
    return egraph._add_backoff_scheduler(
        match_limit=BACKOFF_MATCH_LIMIT,
        ban_length=BACKOFF_BAN_LENGTH,
        egg_like=fresh_rematch,
        haskell_backoff=haskell_backoff,
    )


def _run_single_pass_with_config_egraph(
    num: Num,
    config: ScheduleModeConfig,
) -> tuple[egglog.EGraph, float]:
    """
    Run one `rewriteTree`-like pass and return the populated e-graph.

    The baseline uses one persistent fresh-rematch scheduler with Haskell-style
    backoff accounting and graph-size stability stopping. Ablation modes toggle
    one of those controls at a time while keeping the rewrite set and analysis
    structure fixed.
    """
    egraph = egglog.EGraph()
    egraph.register(num)
    scheduler_handle = (
        _add_iteration_scheduler(
            egraph,
            fresh_rematch=config.fresh_rematch,
            haskell_backoff=config.haskell_backoff,
        )
        if config.persistent_scheduler
        else None
    )

    start = time.perf_counter()
    previous_counts = _serialized_counts(egraph)
    for _ in range(HASKELL_INNER_ITERATION_LIMIT):
        rewrite_scheduler = scheduler_handle or _add_iteration_scheduler(
            egraph,
            fresh_rematch=config.fresh_rematch,
            haskell_backoff=config.haskell_backoff,
        )
        rewrite_report = egraph._run_ruleset_with_scheduler(literal_rewrite_ruleset, rewrite_scheduler)
        analysis_report = egraph.run(analysis_schedule.saturate())
        if config.graph_size_stop:
            current_counts = _serialized_counts(egraph)
            if current_counts == previous_counts:
                break
            previous_counts = current_counts
        elif not (rewrite_report.updated or analysis_report.updated):
            break
    elapsed = time.perf_counter() - start
    return egraph, elapsed


def _run_single_pass_haskell_literal_egraph(num: Num) -> tuple[egglog.EGraph, float]:
    """Historical alias for the retained baseline's one-pass e-graph trace."""
    return _run_single_pass_with_config_egraph(num, MODE_CONFIGS["egglog-baseline"])


def _run_single_pass_haskell_literal(num: Num) -> tuple[Num, int, int, int, int, float]:
    """
    Mirror one Haskell `rewriteTree` pass as directly as Egglog allows.

    `FixTree.rewriteTree` runs one `equalitySaturation' (BackoffScheduler 2500
    30)`. Here we keep one fresh-rematch backoff scheduler bound to the same
    e-graph across up to 30 rewrite iterations, and we interleave explicit
    analysis saturation after each rewrite step because Egglog does not embed
    that analysis inside rebuild. The scheduler also uses Haskell-style
    backoff accounting based on substitution width instead of raw match count.
    """
    egraph, elapsed = _run_single_pass_haskell_literal_egraph(num)
    extracted, cost = egraph.extract(num, include_cost=True)
    total_size = sum(size for _, size in egraph.all_function_sizes())
    node_count, eclass_count = _serialized_counts(egraph)
    return extracted, int(cost), total_size, node_count, eclass_count, elapsed


def _run_single_pass_for_mode(num: Num, mode: Mode) -> tuple[Num, int, int, int, int, float]:
    """Run one pass using the low-level schedule controls for `mode`."""
    config = MODE_CONFIGS[mode]
    egraph, elapsed = _run_single_pass_with_config_egraph(num, config)
    extracted, cost = egraph.extract(num, include_cost=True)
    total_size = sum(size for _, size in egraph.all_function_sizes())
    node_count, eclass_count = _serialized_counts(egraph)
    return extracted, int(cost), total_size, node_count, eclass_count, elapsed


def _run_single_pass(num: Num, mode: Mode = "egglog-baseline") -> tuple[Num, int, int, int, int, float]:
    if mode in MODE_CONFIGS:
        return _run_single_pass_for_mode(num, mode)
    msg = f"Unsupported param-eq mode: {mode}"
    raise ValueError(msg)


def run_paper_pipeline(num: Num, *, mode: Mode) -> PaperPipelineReport:
    """
    Approximate `simplifyE` from `FixTree.hs`.

    Haskell does:
        relabelParams . toSRTree . rewriteUntilNoChange [rewriteTree] 2 . toFixTree

    Here we mirror that as:
    - run `_run_single_pass` on the current extracted term
    - extract
    - stop early if the rendered term is unchanged
    - otherwise rebuild from the extracted term and try again
    """
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
    parser.add_argument("--mode", choices=tuple(MODE_CONFIGS), required=True)
    parser.add_argument("--expr", required=True)
    args = parser.parse_args()
    report = run_paper_pipeline(parse_expression(args.expr), mode=cast(Mode, args.mode))
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
