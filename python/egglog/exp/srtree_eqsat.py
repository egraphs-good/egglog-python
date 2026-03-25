"""
Helpers for reproducing the srtree-eqsat pipeline in egglog.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias, cast

import numpy as np

import egglog
from egglog import *
from egglog.deconstruct import get_callable_args
from egglog.exp.program_gen import Program, program_gen_ruleset

__all__ = [
    "HASKELL_REFERENCE_ROWS",
    "Comparison",
    "Example",
    "Expr",
    "HaskellRow",
    "IterationTrace",
    "MetricReport",
    "PipelineReport",
    "StageReport",
    "compare_to_haskell",
    "compile_expr_callable",
    "core_examples",
    "count_paper_params",
    "default_hl_eval_env",
    "estimate_corpus_runtime",
    "expr_to_python_source",
    "load_example_hl_row",
    "load_example_hl_rows",
    "parse_hl_expr",
    "run_baseline_pipeline",
    "run_multiset_pipeline",
    "sample_runtime_probe",
]


SOURCE_REPO = Path("/Users/saul/p/srtree-eqsat")
EXAMPLE_HL_PATH = SOURCE_REPO / "test/example_hl"

ROW_1_SOURCE = "sqr(-9.29438919215253 + 2.93547417364396 * theta)"
ROW_50_SOURCE = (
    "(exp(0.743694003014863 * alpha) * (-0.0121179632900701 * theta + 0.00904122619609017 * alpha) * "
    "(-3.05659895630567 * theta + 8.63005732191704) + -0.557193153898209 * alpha - log(0.782997897866162 * theta) + "
    "sqr(exp(-0.144728813168975 * theta)) * (-1.54770141702422 + -3.31046821812388 * theta) + "
    "6.34043434659957 * beta * 0.643712432648199) * -0.0413897531650583 + 0.530747148732844"
)


language_rules = ruleset(name="srtree_eqsat_lang")
const_analysis_rules = ruleset(name="srtree_eqsat_const_analysis")
basic_rules = ruleset(name="srtree_eqsat_basic")
fun_rules = ruleset(name="srtree_eqsat_fun")
const_reduction_rules = ruleset(name="srtree_eqsat_const_reduction")
const_fusion_rules = ruleset(name="srtree_eqsat_const_fusion")
program_rules = ruleset(name="srtree_eqsat_program")
multiset_lower_rules = ruleset(name="srtree_eqsat_multiset_lower")
multiset_simplify_rules = ruleset(name="srtree_eqsat_multiset_simplify")
multiset_reify_rules = ruleset(name="srtree_eqsat_multiset_reify")


class OptionalF64(Expr, ruleset=language_rules):
    none: ClassVar[OptionalF64]

    @classmethod
    def some(cls, value: f64Like) -> OptionalF64: ...


class Expr(Expr, ruleset=language_rules):
    @method(cost=5)
    def __init__(self, value: f64Like) -> None: ...

    @method(cost=1)
    @classmethod
    def var(cls, name: StringLike) -> Expr: ...

    @method(cost=5)
    @classmethod
    def param(cls, index: i64Like) -> Expr: ...

    @method(cost=1)
    def __add__(self, other: ExprLike) -> Expr: ...

    @method(cost=1)
    def __sub__(self, other: ExprLike) -> Expr: ...

    @method(cost=1)
    def __mul__(self, other: ExprLike) -> Expr: ...

    @method(cost=1)
    def __truediv__(self, other: ExprLike) -> Expr: ...

    @method(cost=1)
    def __pow__(self, other: ExprLike) -> Expr: ...

    @method(cost=1)
    def __neg__(self) -> Expr: ...

    @method(cost=1)
    def exp(self) -> Expr: ...

    @method(cost=1)
    def log(self) -> Expr: ...

    @method(cost=1)
    def sqrt(self) -> Expr: ...

    @method(cost=1)
    def __abs__(self) -> Expr: ...

    def __radd__(self, other: ExprLike) -> Expr: ...

    def __rsub__(self, other: ExprLike) -> Expr: ...

    def __rmul__(self, other: ExprLike) -> Expr: ...

    def __rtruediv__(self, other: ExprLike) -> Expr: ...

    def __rpow__(self, other: ExprLike) -> Expr: ...


ExprLike: TypeAlias = Expr | StringLike | float | int

converter(float, Expr, Expr)
converter(int, Expr, lambda value: Expr(float(value)))
converter(String, Expr, Expr.var)
converter(str, Expr, Expr.var)


@function(ruleset=program_rules)
def expr_program(expr: Expr) -> Program: ...


@function(ruleset=multiset_lower_rules | multiset_simplify_rules | multiset_reify_rules)
def sum_(xs: MultiSetLike[Expr, ExprLike]) -> Expr: ...


@function(ruleset=multiset_lower_rules | multiset_simplify_rules | multiset_reify_rules)
def product_(xs: MultiSetLike[Expr, ExprLike]) -> Expr: ...


@function(ruleset=const_analysis_rules, merge=lambda old, _new: old)
def const_value(expr: Expr) -> OptionalF64: ...


def exp(x: ExprLike) -> Expr:
    return convert(x, Expr).exp()


def log(x: ExprLike) -> Expr:
    return convert(x, Expr).log()


def sqrt(x: ExprLike) -> Expr:
    return convert(x, Expr).sqrt()


def cbrt(x: ExprLike) -> Expr:
    return convert(x, Expr) ** (1.0 / 3.0)


def sqr(x: ExprLike) -> Expr:
    return convert(x, Expr) ** 2.0


def cube(x: ExprLike) -> Expr:
    return convert(x, Expr) ** 3.0


def _zero() -> Expr:
    return Expr(0.0)


def _one() -> Expr:
    return Expr(1.0)


@const_analysis_rules.register
def _const_analysis(
    expr: Expr,
    x: Expr,
    y: Expr,
    a: f64,
    b: f64,
    i: i64,
    s: String,
) -> Iterable[RewriteOrRule]:
    yield rule(eq(expr).to(Expr(a))).then(set_(const_value(expr)).to(OptionalF64.some(a)))
    yield rule(eq(expr).to(Expr.var(s))).then(set_(const_value(expr)).to(OptionalF64.none))
    yield rule(eq(expr).to(Expr.param(i))).then(set_(const_value(expr)).to(OptionalF64.none))

    yield rule(eq(expr).to(x + y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(expr)).to(OptionalF64.some(a + b))
    )
    yield rule(eq(expr).to(x - y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(expr)).to(OptionalF64.some(a - b))
    )
    yield rule(eq(expr).to(x * y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(expr)).to(OptionalF64.some(a * b))
    )
    yield rule(eq(expr).to(x / y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(expr)).to(OptionalF64.some(a / b))
    )
    yield rule(eq(expr).to(x**y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(expr)).to(OptionalF64.some(a**b))
    )
    yield rule(eq(expr).to(-x), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(expr)).to(OptionalF64.some(-a))
    )
    yield rule(eq(expr).to(x), const_value(x) == OptionalF64.some(a)).then(union(expr).with_(Expr(a)))


@basic_rules.register
def _basic_rewrites(x: Expr, y: Expr, z: Expr, a: Expr, b: Expr) -> Iterable[RewriteOrRule]:
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)
    yield rewrite(x * x).to(x**2.0)
    yield rewrite((x**a) * x).to(x ** (a + 1.0))
    yield rewrite((x**a) * (x**b)).to(x ** (a + b))
    yield rewrite((x + y) + z).to(x + (y + z))
    yield rewrite((x * y) * z).to(x * (y * z))
    yield rewrite((x * y) / z).to(x * (y / z))
    yield rewrite(x - (y + z)).to((x - y) - z)
    yield rewrite(x - (y - z)).to((x - y) + z)
    yield rewrite(-(x + y)).to((-x) - y)
    yield rewrite(x - a).to(x + (-a), const_value(a) != OptionalF64.none, const_value(x) == OptionalF64.none)
    yield rewrite(x - (a * y)).to(
        x + ((-a) * y),
        const_value(a) != OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((_one() / x) * (_one() / y)).to(_one() / (x * y))


@fun_rules.register
def _fun_rewrites(x: Expr, y: Expr, a: Expr, b: Expr, c: f64, d: f64) -> Iterable[RewriteOrRule]:
    half = Expr(0.5)
    yield rewrite(log(x * y)).to(log(x) + log(y), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0)
    yield rewrite((x**a) * (x**b)).to(x ** (a + b))
    yield rewrite(log(x / y)).to(log(x) - log(y), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0)
    yield rewrite(log(x**y)).to(y * log(x), const_value(y) == OptionalF64.some(c), c >= 0.0, c != 0.0)
    yield rewrite(log(sqrt(x))).to(half * log(x), const_value(x) == OptionalF64.none)
    yield rewrite(log(exp(x))).to(x, const_value(x) == OptionalF64.none)
    yield rewrite(exp(log(x))).to(x, const_value(x) == OptionalF64.none)
    yield rewrite(x**0.5).to(sqrt(x))
    yield rewrite(sqrt(a * x)).to(sqrt(a) * sqrt(x), const_value(a) == OptionalF64.some(c), c >= 0.0)
    yield rewrite(sqrt(a * x)).to(sqrt(a) * sqrt(x), const_value(x) == OptionalF64.some(c), c >= 0.0)
    yield rewrite(sqrt(a * (x - y))).to(sqrt(-a) * sqrt(y - x), const_value(a) == OptionalF64.some(c), c < 0.0)
    yield rewrite(sqrt(a * (b + y))).to(
        sqrt(-a) * sqrt(b - y),
        const_value(a) == OptionalF64.some(c),
        c < 0.0,
        const_value(b) == OptionalF64.some(d),
        d < 0.0,
    )
    yield rewrite(sqrt(a / x)).to(sqrt(a) / sqrt(x), const_value(a) == OptionalF64.some(c), c >= 0.0)
    yield rewrite(sqrt(a / x)).to(sqrt(a) / sqrt(x), const_value(x) == OptionalF64.some(c), c >= 0.0)
    yield rewrite(abs(x * y)).to(abs(x) * abs(y))


@const_reduction_rules.register
def _const_reduction(x: Expr, y: Expr, z: Expr, a: Expr, b: Expr, c: f64) -> Iterable[RewriteOrRule]:
    zero = _zero()
    one = _one()
    yield rewrite(zero + x).to(x)
    yield rewrite(x + zero).to(x)
    yield rewrite(x - zero).to(x)
    yield rewrite(one * x).to(x)
    yield rewrite(x * one).to(x)
    yield rewrite(zero * x).to(zero)
    yield rewrite(x * zero).to(zero)
    yield rewrite(zero / x).to(zero)
    yield rewrite(x - x).to(zero)
    yield rewrite(x / x).to(one, const_value(x) != OptionalF64.some(0.0))
    yield rewrite(x**1.0).to(x)
    yield rewrite(zero**x).to(zero)
    yield rewrite(one**x).to(one)
    yield rewrite(x * (one / x)).to(one, const_value(x) != OptionalF64.some(0.0))
    yield rewrite((x * y) + (x * z)).to(x * (y + z))
    yield rewrite(x - ((-1.0) * y)).to(x + y, const_value(y) == OptionalF64.none)
    yield rewrite(x + (-y)).to(x - y, const_value(y) == OptionalF64.none)
    yield rewrite(zero - x).to(-x, const_value(x) == OptionalF64.none)
    yield rewrite((a * x) * (b * y)).to(
        (a * b) * (x * y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite(a / (b * x)).to(
        (a / b) / x,
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
    )


@const_fusion_rules.register
def _const_fusion(x: Expr, y: Expr, a: Expr, b: Expr) -> Iterable[RewriteOrRule]:
    yield rewrite((a * x) + b).to(
        a * (x + (b / a)),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
    )
    yield rewrite((a * x) + (b / y)).to(
        a * (x + (b / a) / y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((a * x) - (b / y)).to(
        a * (x - (b / a) / y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite(x / (b * y)).to(
        (_one() / b) * x / y,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((x / a) + b).to(
        (_one() / a) * (x + (b * a)),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
    )
    yield rewrite((x / a) - b).to(
        (_one() / a) * (x - (b * a)),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
    )
    yield rewrite(b - (x / a)).to(
        (_one() / a) * ((b * a) - x),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
    )
    yield rewrite((x / a) + (b * y)).to(
        (_one() / a) * (x + (b * a) * y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((x / a) + (y / b)).to(
        (_one() / a) * (x + y / (b * a)),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((x / a) - (b * y)).to(
        (_one() / a) * (x - (b * a) * y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((x / a) - (b / y)).to(
        (_one() / a) * (x - y / (b * a)),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )


@multiset_lower_rules.register
def _multiset_lower(x: Expr, y: Expr, xs: MultiSet[Expr], ys: MultiSet[Expr]) -> Iterable[RewriteOrRule]:
    yield rewrite(x + y, subsume=True).to(sum_(MultiSet(x, y)))
    yield rewrite(x * y, subsume=True).to(product_(MultiSet(x, y)))
    yield rule(eq(x).to(sum_(xs)), eq(y).to(sum_(ys))).then(union(x + y).with_(sum_(xs + ys)))
    yield rule(eq(x).to(product_(xs)), eq(y).to(product_(ys))).then(union(x * y).with_(product_(xs + ys)))


@multiset_simplify_rules.register
def _multiset_simplify(
    x: Expr,
    y: Expr,
    z: Expr,
    xs: MultiSet[Expr],
    ys: MultiSet[Expr],
    zs: MultiSet[Expr],
    common: MultiSet[Expr],
    i: f64,
    j: f64,
) -> Iterable[RewriteOrRule]:
    yield rewrite(sum_(MultiSet[Expr]())).to(_zero())
    yield rewrite(product_(MultiSet[Expr]())).to(_one())
    yield rule(eq(x).to(sum_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(product_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(sum_(xs)), xs.contains(_zero()), xs.length() > 1).then(union(x).with_(sum_(xs.remove(_zero()))))
    yield rule(eq(x).to(product_(xs)), xs.contains(_one()), xs.length() > 1).then(
        union(x).with_(product_(xs.remove(_one())))
    )
    yield rule(eq(x).to(product_(xs)), xs.contains(_zero())).then(union(x).with_(_zero()))

    yield rule(
        eq(x).to(sum_(xs)),
        eq(y).to(Expr(i)),
        xs.contains(y),
        eq(z).to(Expr(j)),
        xs.remove(y).contains(z),
    ).then(union(x).with_(sum_(xs.remove(y).remove(z).insert(Expr(i + j)))))
    yield rule(
        eq(x).to(product_(xs)),
        eq(y).to(Expr(i)),
        xs.contains(y),
        eq(z).to(Expr(j)),
        xs.remove(y).contains(z),
    ).then(union(x).with_(product_(xs.remove(y).remove(z).insert(Expr(i * j)))))

    yield rule(
        eq(x).to(sum_(xs)),
        eq(y).to(product_(ys)),
        eq(z).to(product_(zs)),
        xs.contains(y),
        xs.remove(y).contains(z),
        eq(common).to(ys & zs),
        common.length() > 0,
    ).then(
        union(x).with_(
            sum_(
                xs.remove(y)
                .remove(z)
                .insert(product_(common.insert(sum_(MultiSet(product_(ys - common), product_(zs - common))))))
            )
        )
    )


@multiset_reify_rules.register
def _multiset_reify(x: Expr, y: Expr, xs: MultiSet[Expr], ys: MultiSet[Expr]) -> Iterable[RewriteOrRule]:
    yield rewrite(sum_(MultiSet[Expr]())).to(_zero())
    yield rewrite(product_(MultiSet[Expr]())).to(_one())
    yield rule(eq(x).to(sum_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(product_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(sum_(xs)), xs.length() > 1, eq(y).to(xs.pick()), eq(ys).to(xs.remove(y))).then(
        union(x).with_(y + sum_(ys))
    )
    yield rule(eq(x).to(product_(xs)), xs.length() > 1, eq(y).to(xs.pick()), eq(ys).to(xs.remove(y))).then(
        union(x).with_(y * product_(ys))
    )


@program_rules.register
def _expr_program(x: Expr, y: Expr, value: f64, name: String, index: i64) -> Iterable[RewriteOrRule]:
    yield rewrite(expr_program(Expr(value)), subsume=True).to(Program(value.to_string()))
    yield rewrite(expr_program(Expr.var(name)), subsume=True).to(Program(name, True))
    yield rewrite(expr_program(Expr.param(index)), subsume=True).to(Program("params[") + index.to_string() + "]")
    yield rewrite(expr_program(x + y), subsume=True).to(Program("(") + expr_program(x) + " + " + expr_program(y) + ")")
    yield rewrite(expr_program(x - y), subsume=True).to(Program("(") + expr_program(x) + " - " + expr_program(y) + ")")
    yield rewrite(expr_program(x * y), subsume=True).to(Program("(") + expr_program(x) + " * " + expr_program(y) + ")")
    yield rewrite(expr_program(x / y), subsume=True).to(Program("(") + expr_program(x) + " / " + expr_program(y) + ")")
    yield rewrite(expr_program(x**y), subsume=True).to(Program("(") + expr_program(x) + " ** " + expr_program(y) + ")")
    yield rewrite(expr_program(-x), subsume=True).to(Program("(-") + expr_program(x) + ")")
    yield rewrite(expr_program(exp(x)), subsume=True).to(Program("np.exp(") + expr_program(x) + ")")
    yield rewrite(expr_program(log(x)), subsume=True).to(Program("np.log(") + expr_program(x) + ")")
    yield rewrite(expr_program(sqrt(x)), subsume=True).to(Program("np.sqrt(") + expr_program(x) + ")")
    yield rewrite(expr_program(abs(x)), subsume=True).to(Program("np.abs(") + expr_program(x) + ")")
    yield rewrite(expr_program(sum_(MultiSet[Expr]())), subsume=True).to(Program("0.0"))
    yield rewrite(expr_program(product_(MultiSet[Expr]())), subsume=True).to(Program("1.0"))


@dataclass(frozen=True)
class Example:
    name: str
    row: int
    source: str
    description: str
    input_names: tuple[str, ...]
    sample_points: tuple[tuple[float, ...], ...]

    @property
    def expr(self) -> Expr:
        return parse_hl_expr(self.source)


@dataclass(frozen=True)
class IterationTrace:
    iteration: int
    updated: bool
    runtime_sec: float
    total_size: int
    node_count: int
    eclass_count: int
    matches_per_rule: Mapping[str, int]


@dataclass(frozen=True)
class StageReport:
    name: str
    extracted: Expr
    cost: int
    register_sec: float
    run_sec: float
    extract_sec: float
    total_size: int
    node_count: int
    eclass_count: int
    stop_reason: str
    traces: tuple[IterationTrace, ...]
    matches_per_rule: Mapping[str, int]
    search_and_apply_time_per_rule: Mapping[str, float]
    op_counts: Mapping[str, int]

    @property
    def total_sec(self) -> float:
        return self.register_sec + self.run_sec + self.extract_sec


@dataclass(frozen=True)
class MetricReport:
    before_parameter_count: int
    after_parameter_count: int
    reduction_ratio: float
    jacobian_rank: int
    jacobian_rank_gap: int


@dataclass(frozen=True)
class PipelineReport:
    mode: str
    stages: tuple[StageReport, ...]
    extracted: Expr
    cost: int
    total_size: int
    node_count: int
    eclass_count: int
    stop_reason: str
    python_source: str
    metric_report: MetricReport
    numeric_max_abs_error: float
    notes: tuple[str, ...] = ()

    @property
    def total_sec(self) -> float:
        return sum(stage.total_sec for stage in self.stages)


@dataclass(frozen=True)
class HaskellRow:
    row: int
    runtime_sec: float
    memo_size: int
    eclass_count: int
    before_parameter_count: int
    after_parameter_count: int
    before_node_count: int
    after_node_count: int
    simplified_python: str
    notes: str = ""


@dataclass(frozen=True)
class Comparison:
    name: str
    egglog_mode: str
    egglog_runtime_sec: float
    haskell_runtime_sec: float
    egglog_after_parameter_count: int
    haskell_after_parameter_count: int
    egglog_total_size: int
    haskell_memo_size: int
    egglog_stop_reason: str
    notes: tuple[str, ...]


HASKELL_REFERENCE_ROWS: dict[int, HaskellRow] = {
    1: HaskellRow(
        row=1,
        runtime_sec=0.001351,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=2,
        after_parameter_count=2,
        before_node_count=7,
        after_node_count=7,
        simplified_python="(-9.29438919215253 + (2.93547417364396 * x[:, 2])) ** 2.0",
        notes="Source helper uses exported simplifyEqSat only; final e-graph sizes are not exposed.",
    ),
    50: HaskellRow(
        row=50,
        runtime_sec=0.547641,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=14,
        after_parameter_count=12,
        before_node_count=60,
        after_node_count=46,
        simplified_python=(
            "(-4.13897531650583e-2 * ((np.exp((x[:, 2] * -0.144728813168975)) ** 2.0) * "
            "(-1.54770141702422 - (x[:, 2] * 3.31046821812388)) + ((x[:, 1] * 4.081416417295803) + "
            "(((((np.exp((0.743694003014863 * x[:, 0])) * ((x[:, 0] * 9.04122619609017e-3) - "
            "(x[:, 2] * 1.21179632900701e-2))) * (8.63005732191704 - (x[:, 2] * 3.05659895630567))) - "
            "(x[:, 0] * 0.557193153898209)) - np.log(x[:, 2])) + -12.578528004457953))))"
        ),
        notes="Source helper uses exported simplifyEqSat only; final e-graph sizes are not exposed.",
    ),
}


def _default_sample_points(
    input_names: Sequence[str], *, seed: int = 0, count: int = 64
) -> tuple[tuple[float, ...], ...]:
    rng = np.random.default_rng(seed)
    return tuple(tuple(float(x) for x in row) for row in rng.uniform(0.25, 2.0, size=(count, len(input_names))))


def core_examples() -> tuple[Example, Example]:
    input_names = ("alpha", "beta", "theta")
    return (
        Example(
            name="row_1",
            row=1,
            source=ROW_1_SOURCE,
            description="Small sanity case from test/example_hl row 1.",
            input_names=input_names,
            sample_points=_default_sample_points(input_names, seed=1),
        ),
        Example(
            name="row_50",
            row=50,
            source=ROW_50_SOURCE,
            description="Function-heavy representative case from test/example_hl row 50.",
            input_names=input_names,
            sample_points=_default_sample_points(input_names, seed=50),
        ),
    )


def default_hl_eval_env() -> dict[str, object]:
    alpha = Expr.var("alpha")
    beta = Expr.var("beta")
    theta = Expr.var("theta")
    return {
        "__builtins__": {},
        "alpha": alpha,
        "beta": beta,
        "theta": theta,
        "exp": exp,
        "log": log,
        "sqrt": sqrt,
        "cbrt": cbrt,
        "sqr": sqr,
        "cube": cube,
        "abs": abs,
    }


def parse_hl_expr(source: str) -> Expr:
    expr = eval(source, default_hl_eval_env(), {})
    if not isinstance(expr, Expr):
        msg = f"HL expression did not produce an Expr: {source!r}"
        raise TypeError(msg)
    return expr


def load_example_hl_rows() -> tuple[str, ...]:
    return tuple(line.strip() for line in EXAMPLE_HL_PATH.read_text().splitlines() if line.strip())


def load_example_hl_row(row: int) -> str:
    rows = load_example_hl_rows()
    if row < 1 or row > len(rows):
        msg = f"row must be between 1 and {len(rows)}, got {row}"
        raise ValueError(msg)
    return rows[row - 1]


def _expr_to_program(expr: Expr) -> tuple[str, str]:
    egraph = egglog.EGraph()
    program = expr_program(expr)
    egraph.register(program)
    egraph.register(program.compile())
    egraph.run((program_rules | program_gen_ruleset).saturate())
    statements = cast("String", egraph.extract(program.statements)).value
    body = cast("String", egraph.extract(program.expr)).value
    return statements, body


def _multiset_items(ms: object) -> tuple[Expr, ...]:
    if args := get_callable_args(ms, MultiSet):
        return tuple(cast("Expr", arg) for arg in args)
    if args := get_callable_args(ms, MultiSet.single):
        value, count = args
        return tuple(cast("Expr", value) for _ in range(int(cast("i64", count).value)))
    msg = f"Unsupported multiset node: {ms!r}"
    raise TypeError(msg)


def _contains_multiset(expr: Expr) -> bool:
    if get_callable_args(expr, sum_) or get_callable_args(expr, product_):
        return True
    args = get_callable_args(expr)
    if not args:
        return False
    return any(_contains_multiset(cast("Expr", arg)) for arg in args if isinstance(arg, Expr))


def _manual_expr_to_python(expr: Expr) -> str:
    if args := get_callable_args(expr, Expr):
        return repr(float(cast("f64", args[0]).value))
    if args := get_callable_args(expr, Expr.var):
        return cast("String", args[0]).value
    if args := get_callable_args(expr, Expr.param):
        return f"params[{int(cast('i64', args[0]).value)}]"
    if args := get_callable_args(expr, Expr.__add__):
        return f"({_manual_expr_to_python(cast('Expr', args[0]))} + {_manual_expr_to_python(cast('Expr', args[1]))})"
    if args := get_callable_args(expr, Expr.__sub__):
        return f"({_manual_expr_to_python(cast('Expr', args[0]))} - {_manual_expr_to_python(cast('Expr', args[1]))})"
    if args := get_callable_args(expr, Expr.__mul__):
        return f"({_manual_expr_to_python(cast('Expr', args[0]))} * {_manual_expr_to_python(cast('Expr', args[1]))})"
    if args := get_callable_args(expr, Expr.__truediv__):
        return f"({_manual_expr_to_python(cast('Expr', args[0]))} / {_manual_expr_to_python(cast('Expr', args[1]))})"
    if args := get_callable_args(expr, Expr.__pow__):
        return f"({_manual_expr_to_python(cast('Expr', args[0]))} ** {_manual_expr_to_python(cast('Expr', args[1]))})"
    if args := get_callable_args(expr, Expr.__neg__):
        return f"(-{_manual_expr_to_python(cast('Expr', args[0]))})"
    if args := get_callable_args(expr, Expr.exp):
        return f"np.exp({_manual_expr_to_python(cast('Expr', args[0]))})"
    if args := get_callable_args(expr, Expr.log):
        return f"np.log({_manual_expr_to_python(cast('Expr', args[0]))})"
    if args := get_callable_args(expr, Expr.sqrt):
        return f"np.sqrt({_manual_expr_to_python(cast('Expr', args[0]))})"
    if args := get_callable_args(expr, Expr.__abs__):
        return f"np.abs({_manual_expr_to_python(cast('Expr', args[0]))})"
    if args := get_callable_args(expr, sum_):
        (ms,) = args
        terms = [_manual_expr_to_python(term) for term in _multiset_items(ms)]
        return "(" + " + ".join(terms) + ")" if terms else "0.0"
    if args := get_callable_args(expr, product_):
        (ms,) = args
        terms = [_manual_expr_to_python(term) for term in _multiset_items(ms)]
        return "(" + " * ".join(terms) + ")" if terms else "1.0"
    msg = f"Unsupported Expr node for Python rendering: {expr!r}"
    raise TypeError(msg)


def expr_to_python_source(expr: Expr, *, parameterize: bool = False) -> tuple[str, tuple[float, ...]]:
    if parameterize:
        expr, values = _parameterize_expr(expr)
    else:
        values = ()
    if _contains_multiset(expr):
        statements = ""
        body = _manual_expr_to_python(expr)
    else:
        statements, body = _expr_to_program(expr)
    lines = [line for line in statements.splitlines() if line.strip()]
    return ("\n".join((*lines, f"return {body}")) if lines else f"return {body}", values)


def _compile_function_source(source: str, input_names: Sequence[str], *, with_params: bool) -> Callable[..., float]:
    args = list(input_names)
    if with_params:
        args.append("params")
    namespace = {"np": np}
    exec(f"def __fn({', '.join(args)}):\n" + "\n".join(f"    {line}" for line in source.splitlines()), namespace)
    return cast("Callable[..., float]", namespace["__fn"])


def compile_expr_callable(
    expr: Expr, *, input_names: Sequence[str], parameterize: bool = False
) -> Callable[..., float]:
    source, _ = expr_to_python_source(expr, parameterize=parameterize)
    return _compile_function_source(source, input_names, with_params=parameterize)


def _count_expr_nodes(expr: Expr) -> int:
    args = get_callable_args(expr)
    if not args:
        return 1
    return 1 + sum(_count_expr_nodes(cast("Expr", arg)) for arg in args if isinstance(arg, Expr))


def _walk_expr(expr: Expr) -> Iterable[Expr]:
    yield expr
    args = get_callable_args(expr)
    if not args:
        return
    for arg in args:
        if isinstance(arg, Expr):
            yield from _walk_expr(cast("Expr", arg))


def _parameterize_expr(expr: Expr) -> tuple[Expr, tuple[float, ...]]:
    params: list[float] = []

    def go(node: Expr) -> Expr:
        if args := get_callable_args(node, Expr):
            (value_expr,) = args
            value = float(cast("f64", value_expr).value)
            if value.is_integer():
                return Expr(value)
            index = len(params)
            params.append(value)
            return Expr.param(index)
        if args := get_callable_args(node, Expr.var):
            return Expr.var(cast("String", args[0]).value)
        if args := get_callable_args(node, Expr.param):
            return Expr.param(cast("i64", args[0]).value)
        if args := get_callable_args(node, Expr.__add__):
            return go(cast("Expr", args[0])) + go(cast("Expr", args[1]))
        if args := get_callable_args(node, Expr.__sub__):
            return go(cast("Expr", args[0])) - go(cast("Expr", args[1]))
        if args := get_callable_args(node, Expr.__mul__):
            return go(cast("Expr", args[0])) * go(cast("Expr", args[1]))
        if args := get_callable_args(node, Expr.__truediv__):
            return go(cast("Expr", args[0])) / go(cast("Expr", args[1]))
        if args := get_callable_args(node, Expr.__pow__):
            return go(cast("Expr", args[0])) ** go(cast("Expr", args[1]))
        if args := get_callable_args(node, Expr.__neg__):
            return -go(cast("Expr", args[0]))
        if args := get_callable_args(node, Expr.exp):
            return exp(go(cast("Expr", args[0])))
        if args := get_callable_args(node, Expr.log):
            return log(go(cast("Expr", args[0])))
        if args := get_callable_args(node, Expr.sqrt):
            return sqrt(go(cast("Expr", args[0])))
        if args := get_callable_args(node, Expr.__abs__):
            return abs(go(cast("Expr", args[0])))
        if args := get_callable_args(node, sum_):
            (ms,) = args
            return sum_(MultiSet(*(go(term) for term in _multiset_items(ms))))
        if args := get_callable_args(node, product_):
            (ms,) = args
            return product_(MultiSet(*(go(term) for term in _multiset_items(ms))))
        msg = f"Unsupported Expr node while parameterizing: {node!r}"
        raise TypeError(msg)

    return go(expr), tuple(params)


def count_paper_params(expr: Expr) -> int:
    _, values = _parameterize_expr(expr)
    return len(values)


def jacobian_rank(expr: Expr, *, input_names: Sequence[str], sample_points: Sequence[Sequence[float]]) -> int:
    parameterized_expr, params = _parameterize_expr(expr)
    if not params:
        return 0
    source, _ = expr_to_python_source(parameterized_expr, parameterize=False)
    fn = _compile_function_source(source, input_names, with_params=True)
    params_np = np.array(params, dtype=float)
    eps = 1e-6
    rows: list[np.ndarray] = []
    for point in sample_points:
        base = float(fn(*point, params_np))
        if not np.isfinite(base):
            continue
        grads = np.zeros(params_np.size, dtype=float)
        for index in range(params_np.size):
            perturbed = params_np.copy()
            perturbed[index] += eps
            value = float(fn(*point, perturbed))
            grads[index] = (value - base) / eps
        if np.isfinite(grads).all():
            rows.append(grads)
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.vstack(rows)))


def _numeric_max_abs_error(
    original: Expr,
    simplified: Expr,
    *,
    input_names: Sequence[str],
    sample_points: Sequence[Sequence[float]],
) -> float:
    original_fn = compile_expr_callable(original, input_names=input_names)
    simplified_fn = compile_expr_callable(simplified, input_names=input_names)
    errors = []
    for point in sample_points:
        original_value = float(original_fn(*point))
        simplified_value = float(simplified_fn(*point))
        if np.isfinite(original_value) and np.isfinite(simplified_value):
            errors.append(abs(original_value - simplified_value))
    return float(max(errors, default=0.0))


def _serialized_counts(egraph: egglog.EGraph) -> tuple[int, int, dict[str, int]]:
    payload = json.loads(egraph._serialize().to_json())
    nodes = payload.get("nodes", {})
    class_data = payload.get("class_data", {})
    ops = Counter(node["op"] for node in nodes.values())
    return len(nodes), len(class_data), dict(ops)


def _aggregate_rule_times(run_report: object) -> tuple[dict[str, int], dict[str, float]]:
    matches = {name: int(count) for name, count in getattr(run_report, "num_matches_per_rule", {}).items()}
    times = {
        name: float(delta.total_seconds())
        for name, delta in getattr(run_report, "search_and_apply_time_per_rule", {}).items()
    }
    return matches, times


def _run_stage(
    name: str,
    expr: Expr,
    rules: egglog.Ruleset | egglog.Schedule,
    *,
    node_cutoff: int | None,
    iteration_limit: int,
    scheduler: egglog.BackOff | None = None,
) -> StageReport:
    egraph = egglog.EGraph()
    start = time.perf_counter()
    egraph.register(expr)
    register_sec = time.perf_counter() - start

    traces: list[IterationTrace] = []
    total_run_sec = 0.0
    aggregate_matches: Counter[str] = Counter()
    aggregate_times: defaultdict[str, float] = defaultdict(float)
    stop_reason = "budget_hit"

    for iteration in range(1, iteration_limit + 1):
        start = time.perf_counter()
        step_schedule = run(rules, scheduler=scheduler) if isinstance(rules, egglog.Ruleset) else rules
        run_report = egraph.run(step_schedule)
        elapsed = time.perf_counter() - start
        total_run_sec += elapsed
        matches, times = _aggregate_rule_times(run_report)
        aggregate_matches.update(matches)
        for key, value in times.items():
            aggregate_times[key] += value
        node_count, eclass_count, _ = _serialized_counts(egraph)
        total_size = sum(size for _, size in egraph.all_function_sizes())
        updated = bool(run_report.updated)
        traces.append(
            IterationTrace(
                iteration=iteration,
                updated=updated,
                runtime_sec=elapsed,
                total_size=total_size,
                node_count=node_count,
                eclass_count=eclass_count,
                matches_per_rule=matches,
            )
        )
        if node_cutoff is not None and total_size > node_cutoff:
            stop_reason = "cutoff_hit"
            break
        if not updated:
            stop_reason = "saturated"
            break

    start = time.perf_counter()
    extracted, cost = egraph.extract(expr, include_cost=True)
    extract_sec = time.perf_counter() - start
    total_size = sum(size for _, size in egraph.all_function_sizes())
    node_count, eclass_count, op_counts = _serialized_counts(egraph)
    return StageReport(
        name=name,
        extracted=extracted,
        cost=cost,
        register_sec=register_sec,
        run_sec=total_run_sec,
        extract_sec=extract_sec,
        total_size=total_size,
        node_count=node_count,
        eclass_count=eclass_count,
        stop_reason=stop_reason,
        traces=tuple(traces),
        matches_per_rule=dict(aggregate_matches),
        search_and_apply_time_per_rule=dict(aggregate_times),
        op_counts=op_counts,
    )


def _baseline_rulesets() -> tuple[egglog.Ruleset, egglog.Ruleset]:
    rewrite_const = const_analysis_rules | const_reduction_rules
    rewrite_all = const_analysis_rules | basic_rules | const_reduction_rules | const_fusion_rules | fun_rules
    return rewrite_const, rewrite_all


def _multiset_cleanup_rules() -> egglog.Ruleset:
    return const_analysis_rules | const_reduction_rules | const_fusion_rules | fun_rules


def _pipeline_metrics(
    original: Expr,
    extracted: Expr,
    *,
    input_names: Sequence[str],
    sample_points: Sequence[Sequence[float]],
) -> MetricReport:
    before = count_paper_params(original)
    after = count_paper_params(extracted)
    rank = jacobian_rank(extracted, input_names=input_names, sample_points=sample_points)
    return MetricReport(
        before_parameter_count=before,
        after_parameter_count=after,
        reduction_ratio=((before - after) / before) if before else 0.0,
        jacobian_rank=rank,
        jacobian_rank_gap=after - rank,
    )


def _baseline_notes(stage: StageReport) -> tuple[str, ...]:
    if stage.stop_reason == "cutoff_hit":
        return ("Baseline stopped at the user-imposed node cutoff before reaching a fixpoint.",)
    if stage.stop_reason == "budget_hit":
        return ("Baseline hit the iteration budget before proving saturation.",)
    return ()


def run_baseline_pipeline(
    expr: Expr,
    *,
    node_cutoff: int,
    iteration_limit: int,
    input_names: Sequence[str] = ("alpha", "beta", "theta"),
    sample_points: Sequence[Sequence[float]] | None = None,
) -> PipelineReport:
    rewrite_const, rewrite_all = _baseline_rulesets()
    const_scheduler = back_off(match_limit=100, ban_length=10)
    all_scheduler = back_off(match_limit=2500, ban_length=30)
    const_stage = _run_stage(
        "rewrite_const",
        expr,
        rewrite_const,
        node_cutoff=node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=const_scheduler,
    )
    current = const_stage.extracted
    stages = [const_stage]
    previous_source = expr_to_python_source(current)[0]
    stop_reason = const_stage.stop_reason
    for pass_index in range(1, 3):
        stage = _run_stage(
            f"rewrite_all_pass_{pass_index}",
            current,
            rewrite_all,
            node_cutoff=node_cutoff,
            iteration_limit=iteration_limit,
            scheduler=all_scheduler,
        )
        stages.append(stage)
        current = stage.extracted
        stop_reason = stage.stop_reason
        current_source = expr_to_python_source(current)[0]
        if current_source == previous_source:
            break
        previous_source = current_source
        if stop_reason != "saturated":
            break
    sample_points = sample_points or _default_sample_points(input_names)
    metrics = _pipeline_metrics(expr, current, input_names=input_names, sample_points=sample_points)
    return PipelineReport(
        mode="baseline",
        stages=tuple(stages),
        extracted=current,
        cost=stages[-1].cost,
        total_size=stages[-1].total_size,
        node_count=stages[-1].node_count,
        eclass_count=stages[-1].eclass_count,
        stop_reason=stop_reason,
        python_source=expr_to_python_source(current)[0],
        metric_report=metrics,
        numeric_max_abs_error=_numeric_max_abs_error(
            expr, current, input_names=input_names, sample_points=sample_points
        ),
        notes=_baseline_notes(stages[-1]),
    )


def _multiset_notes(stages: Sequence[StageReport]) -> tuple[str, ...]:
    notes: list[str] = []
    first = stages[0]
    if first.stop_reason == "saturated":
        notes.append("Multiset lowering saturated without backoff or node limits.")
    else:
        notes.append(f"Multiset lowering stopped with {first.stop_reason}.")
    for stage in stages:
        hot_rules = sorted(stage.matches_per_rule.items(), key=lambda item: item[1], reverse=True)[:3]
        if hot_rules:
            notes.append(f"{stage.name} hottest rules: {', '.join(f'{name}={count}' for name, count in hot_rules)}")
    return tuple(notes)


def run_multiset_pipeline(
    expr: Expr,
    *,
    saturate_without_limits: bool = True,
    node_cutoff: int | None = None,
    iteration_limit: int | None = None,
    input_names: Sequence[str] = ("alpha", "beta", "theta"),
    sample_points: Sequence[Sequence[float]] | None = None,
) -> PipelineReport:
    iteration_limit = iteration_limit or 80
    stage1 = _run_stage(
        "multiset_lower",
        expr,
        const_analysis_rules | multiset_lower_rules,
        node_cutoff=None if saturate_without_limits else node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=None,
    )
    stage2 = _run_stage(
        "multiset_simplify",
        stage1.extracted,
        const_analysis_rules | multiset_simplify_rules | fun_rules,
        node_cutoff=None if saturate_without_limits else node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=None,
    )
    stage3 = _run_stage(
        "multiset_reify_cleanup",
        stage2.extracted,
        multiset_reify_rules | _multiset_cleanup_rules(),
        node_cutoff=None if saturate_without_limits else node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=None,
    )
    stages = (stage1, stage2, stage3)
    sample_points = sample_points or _default_sample_points(input_names)
    metrics = _pipeline_metrics(expr, stage3.extracted, input_names=input_names, sample_points=sample_points)
    return PipelineReport(
        mode="multiset",
        stages=stages,
        extracted=stage3.extracted,
        cost=stage3.cost,
        total_size=stage3.total_size,
        node_count=stage3.node_count,
        eclass_count=stage3.eclass_count,
        stop_reason=stage3.stop_reason,
        python_source=expr_to_python_source(stage3.extracted)[0],
        metric_report=metrics,
        numeric_max_abs_error=_numeric_max_abs_error(
            expr, stage3.extracted, input_names=input_names, sample_points=sample_points
        ),
        notes=_multiset_notes(stages),
    )


def compare_to_haskell(expr_name: str, egglog_report: PipelineReport, haskell_row: HaskellRow) -> Comparison:
    notes: list[str] = []
    if egglog_report.metric_report.after_parameter_count != haskell_row.after_parameter_count:
        notes.append("Parameter counts differ between Egglog and Haskell.")
    if egglog_report.stop_reason != "saturated":
        notes.append(f"Egglog stopped with {egglog_report.stop_reason}.")
    return Comparison(
        name=expr_name,
        egglog_mode=egglog_report.mode,
        egglog_runtime_sec=egglog_report.total_sec,
        haskell_runtime_sec=haskell_row.runtime_sec,
        egglog_after_parameter_count=egglog_report.metric_report.after_parameter_count,
        haskell_after_parameter_count=haskell_row.after_parameter_count,
        egglog_total_size=egglog_report.total_size,
        haskell_memo_size=haskell_row.memo_size,
        egglog_stop_reason=egglog_report.stop_reason,
        notes=tuple(notes),
    )


def sample_runtime_probe(
    rows: Sequence[int] = (1, 20, 50, 100, 200, 400, 657),
    *,
    node_cutoff: int = 50_000,
    iteration_limit: int = 20,
) -> dict[str, float]:
    baseline_total = 0.0
    multiset_total = 0.0
    for row in rows:
        source = load_example_hl_row(row)
        expr = parse_hl_expr(source)
        baseline_total += run_baseline_pipeline(
            expr, node_cutoff=node_cutoff, iteration_limit=iteration_limit
        ).total_sec
        multiset_total += run_multiset_pipeline(
            expr,
            saturate_without_limits=False,
            node_cutoff=node_cutoff,
            iteration_limit=iteration_limit,
        ).total_sec
    scale = len(load_example_hl_rows()) / len(rows)
    return {
        "sampled_rows": float(len(rows)),
        "baseline_sample_sec": baseline_total,
        "multiset_sample_sec": multiset_total,
        "baseline_projected_sec": baseline_total * scale,
        "multiset_projected_sec": multiset_total * scale,
    }


def estimate_corpus_runtime() -> dict[str, float]:
    return sample_runtime_probe()
