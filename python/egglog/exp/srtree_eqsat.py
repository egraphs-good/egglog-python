"""
Helpers for reproducing the baseline srtree-eqsat pipeline in egglog.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias, cast

import numpy as np

import egglog
from egglog import *
from egglog.deconstruct import get_callable_args

__all__ = [
    "HASKELL_REFERENCE_ROWS",
    "Comparison",
    "Example",
    "HaskellRow",
    "IterationTrace",
    "MetricReport",
    "Num",
    "NumLike",
    "PipelineReport",
    "StageReport",
    "compare_to_haskell",
    "core_examples",
    "count_paper_params",
    "default_hl_eval_env",
    "estimate_corpus_runtime",
    "eval_num",
    "jacobian_rank",
    "load_example_hl_row",
    "load_example_hl_rows",
    "parse_hl_expr",
    "render_num",
    "run_baseline_pipeline",
    "sample_runtime_probe",
]


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REPO = REPO_ROOT.parent / "srtree-eqsat"
EXAMPLE_HL_PATH = SOURCE_REPO / "test/example_hl"

SELECTED_BATCH_ROWS = (1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200)
SELECTED_ROW_SOURCES: dict[int, str] = {
    1: "sqr(-9.29438919215253 + 2.93547417364396 * theta)",
    2: "(((2.91970418475328 + -0.773583225885789 * theta) * 2.87230609775115 * theta + (-1.22356797479824 + 0.377885426397629 * theta) * (10.3917281470804 - -1.20862070725184 * alpha)) * (0.114247677264604 * alpha + 0.127799885276295 * beta) * (3.60250727355285 * beta + 1.35381696253641 * theta - 9.60356246300082 * alpha) - (sqr(-9.29438919215253 + 2.93547417364396 * theta) + (-1.31893797108115 * beta + 2.10814063839763 * alpha) - 5.16995214427275 * alpha) * (1.25099225339683 * alpha + -1.0075335593301 * beta)) * 0.227993040492379 + 1.72695871415228",
    3: "(((2.78948361720301 + -0.603311011990701 * theta) * 2.66295898187985 * theta + (-1.70053796438431 + 0.354410284952137 * theta) * (2.30920195902924 * alpha + -0.684514001199842 * beta)) * (-0.423113013561051 * alpha + -0.820293590903207 * beta) * exp(0.648433111594964 * alpha) - (4.84782777418965 * alpha + -10.2013161038227 * beta + (4.49181352322002 + 3.34338756313595 * theta) - 1.52463076043089 * alpha) * sqr(-4.40371470183112 + 1.239832234323 * theta)) * 0.0555952399242293 + 2.3974618452108",
    4: "(((2.84721301451649 + -1.0832466055762 * theta) * 2.87288386010019 * theta + (-0.960822430613027 + 0.352212452215726 * theta) * (9.49502848514098 - -4.43457932618305 * alpha)) * (0.0231254136991372 * alpha + -0.201859225439622 * beta) * exp(0.533684665488698 * alpha) - (4.3709278998084 * alpha + 0.0676271161563555 * beta + (-10.1636621343434 + 2.20402481737344 * theta) - 2.00153063481211 * alpha) * (-0.173514070270718 * alpha + 1.4309773332239 * beta)) * 0.418278900765285 + -0.250852384709946",
    5: "(2.98279150112649 - 2.50335104760053 * alpha + exp(0.472180751947384 * alpha) + (-5.46511983933291 * alpha + 4.7308388959051 * beta + sqr(1.9820027542127 * alpha + -0.507993116540432 * beta) + -0.902691894725103 * theta) * (sqr(-9.71025706279119 + 3.06939441276146 * theta) + -12.1940251737044) * (-0.00113611973003391 * alpha + 0.0069224336526928 * beta)) * 1.38844130555412 + 0.330982870226231",
    10: "(-4.0257291259403 * alpha + exp(0.496521530470785 * alpha) * (sqr(-9.90385501742711 + 3.09891553397754 * theta) + -12.0313313266527) * (-0.0209790347331461 * alpha + 0.15542475900314 * beta)) * 0.218016003919151 + 2.93075744602469",
    20: "sqr(cube(-0.453946451098545 * theta) + 3.0472620183122 * theta) * -0.342063915063894 + 10.6692145111104",
    30: "(exp(-1.47422568429434 * alpha) - (cube(-0.58923877562757 * theta) + sqr(2.39236882178653 * theta - 6.26125635041365)) * -0.00792212718106198 * beta) * (exp(1.19822491431359 * alpha) + 19.3608807319069 * beta) * 0.313237693440143 + -0.504509880559088",
    40: "sqr(0.309853513321887 * theta - exp(-0.00478572314032874 * alpha)) * 32.3824199084528 + -4.32199869886884",
    50: "(exp(0.743694003014863 * alpha) * (-0.0121179632900701 * theta + 0.00904122619609017 * alpha) * (-3.05659895630567 * theta + 8.63005732191704) + -0.557193153898209 * alpha - log(0.782997897866162 * theta) + sqr(exp(-0.144728813168975 * theta)) * (-1.54770141702422 + -3.31046821812388 * theta) + 6.34281372899835) * 1.70035591779884 * beta * 1.03068155805492 + 0.404362453868565",
    75: "(0.27658499902994 * beta + (-3.09568685791928 * theta - 1.06543349470034 * beta + 4.80847033528277 * alpha) * (0.00840515601044034 * theta - exp(-5.89563245316599 * theta - -2.2128445010022 * beta))) * (sqr(-1.73329369062705 * theta + 5.01701830214846) - 3.04752460994615 * alpha) * 0.830662588380978 + 2.77768079307633",
    100: "(-0.817246761082895 * theta + 4.7987094782841 - 1.18369793731582 * alpha * (3.27838657131292 * theta + -15.2760394159237) * (-0.0953755961420584 * theta + 0.0270703912792823 * alpha)) * 1.11985975437354 * beta * 1.01334538331164 + 0.201404917055389",
    150: "((5.53327625349294 * beta - 9.01964233670581 * theta - (-17.4546811325902 * alpha - 2.50583781801447 * alpha)) * (-2.49821516165647 - -0.402138004942828 * theta) * (log(0.347653612616773 * alpha) - -0.512350969799209 * theta) - (1.40058250223625 * alpha - -0.313213248196185 * beta - -0.244937726279955 * beta) * exp(0.569380315263419 * alpha) * (-0.841465262470357 * theta + exp(0.242251869587583 * alpha) - sqr(0.466508892559209 * theta + -2.80204970690573))) * 0.0632264979538498 * beta * 1.00057660932339 + -0.455923813201509",
    200: "(sqr(-7.01398906858873 * theta - -20.9157503480256 + 0.40965685006749 * beta) - 18.7131049568053 * alpha) / (-2.89176232552097 * beta + 15.8269215595989 + sqr(-1.76570724282939 * theta - -5.46414193899484)) * 0.689375955248431 + 0.570693365253855",
}


language_rules = ruleset(name="srtree_eqsat_lang")
const_analysis_rules = ruleset(name="srtree_eqsat_const_analysis")
basic_rules = ruleset(name="srtree_eqsat_basic")
fun_rules = ruleset(name="srtree_eqsat_fun")
const_reduction_rules = ruleset(name="srtree_eqsat_const_reduction")
const_fusion_rules = ruleset(name="srtree_eqsat_const_fusion")


class OptionalF64(Expr, ruleset=language_rules):
    none: ClassVar[OptionalF64]

    @classmethod
    def some(cls, value: f64Like) -> OptionalF64: ...  # type: ignore[empty-body]


class Num(Expr, ruleset=language_rules):
    @method(cost=5)
    def __init__(self, value: f64Like) -> None: ...

    @method(cost=1)
    @classmethod
    def var(cls, name: StringLike) -> Num: ...  # type: ignore[empty-body]

    @method(cost=5)
    @classmethod
    def param(cls, index: i64Like) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __add__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __sub__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __mul__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __truediv__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __pow__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __neg__(self) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def exp(self) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def log(self) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def sqrt(self) -> Num: ...  # type: ignore[empty-body]

    @method(cost=1)
    def __abs__(self) -> Num: ...  # type: ignore[empty-body]

    def __radd__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    def __rsub__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    def __rmul__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    def __rtruediv__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]

    def __rpow__(self, other: NumLike) -> Num: ...  # type: ignore[empty-body]


NumLike: TypeAlias = Num | StringLike | float | int

converter(float, Num, Num)
converter(int, Num, lambda value: Num(float(value)))
converter(String, Num, Num.var)
converter(str, Num, Num.var)


@function(ruleset=const_analysis_rules, merge=lambda old, _new: old)  # type: ignore[call-overload]
def const_value(num: Num) -> OptionalF64: ...  # type: ignore[empty-body]


def exp(x: NumLike) -> Num:
    return convert(x, Num).exp()


def log(x: NumLike) -> Num:
    return convert(x, Num).log()


def sqrt(x: NumLike) -> Num:
    return convert(x, Num).sqrt()


def cbrt(x: NumLike) -> Num:
    return convert(x, Num) ** (1.0 / 3.0)


def sqr(x: NumLike) -> Num:
    return convert(x, Num) ** 2.0


def cube(x: NumLike) -> Num:
    return convert(x, Num) ** 3.0


def _zero() -> Num:
    return Num(0.0)


def _one() -> Num:
    return Num(1.0)


@const_analysis_rules.register
def _const_analysis(
    num: Num,
    x: Num,
    y: Num,
    a: f64,
    b: f64,
    i: i64,
    s: String,
) -> Iterable[RewriteOrRule]:
    yield rule(eq(num).to(Num(a))).then(set_(const_value(num)).to(OptionalF64.some(a)))
    yield rule(eq(num).to(Num.var(s))).then(set_(const_value(num)).to(OptionalF64.none))
    yield rule(eq(num).to(Num.param(i))).then(set_(const_value(num)).to(OptionalF64.none))

    yield rule(eq(num).to(x + y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a + b))
    )
    yield rule(eq(num).to(x - y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a - b))
    )
    yield rule(eq(num).to(x * y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a * b))
    )
    yield rule(eq(num).to(x / y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a / b))
    )
    yield rule(eq(num).to(x**y), const_value(x) == OptionalF64.some(a), const_value(y) == OptionalF64.some(b)).then(
        set_(const_value(num)).to(OptionalF64.some(a**b))
    )
    yield rule(eq(num).to(-x), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(-a))
    )
    yield rule(eq(num).to(exp(x)), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(a.exp()))
    )
    yield rule(eq(num).to(log(x)), const_value(x) == OptionalF64.some(a), a > 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a.log()))
    )
    yield rule(eq(num).to(sqrt(x)), const_value(x) == OptionalF64.some(a), a >= 0.0).then(
        set_(const_value(num)).to(OptionalF64.some(a.sqrt()))
    )
    yield rule(eq(num).to(abs(x)), const_value(x) == OptionalF64.some(a)).then(
        set_(const_value(num)).to(OptionalF64.some(abs(a)))
    )
    yield rule(eq(num).to(x), const_value(x) == OptionalF64.some(a)).then(union(num).with_(Num(a)))


@basic_rules.register
def _basic_rewrites(x: Num, y: Num, z: Num, a: Num, b: Num) -> Iterable[RewriteOrRule]:
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
def _fun_rewrites(x: Num, y: Num, a: Num, b: Num, c: f64, d: f64) -> Iterable[RewriteOrRule]:
    half = Num(0.5)
    yield rewrite(log(x * y)).to(log(x) + log(y), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0)  # type: ignore[arg-type]
    yield rewrite((x**a) * (x**b)).to(x ** (a + b))
    yield rewrite(log(x / y)).to(log(x) - log(y), const_value(x) == OptionalF64.some(c), c >= 0.0, c != 0.0)  # type: ignore[arg-type]
    yield rewrite(log(x**y)).to(y * log(x), const_value(y) == OptionalF64.some(c), c >= 0.0, c != 0.0)  # type: ignore[arg-type]
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
def _const_reduction(x: Num, y: Num, z: Num, a: Num, b: Num, c: f64) -> Iterable[RewriteOrRule]:
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
def _const_fusion(x: Num, y: Num, a: Num, b: Num) -> Iterable[RewriteOrRule]:
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


@dataclass(frozen=True)
class Example:
    name: str
    row: int
    source: str
    description: str
    input_names: tuple[str, ...]
    sample_points: tuple[tuple[float, ...], ...]

    @property
    def expr(self) -> Num:
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
    extracted: Num
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
    optimal_parameter_count: int | None
    gap_to_optimal: int | None
    optimal_status: str


@dataclass(frozen=True)
class PipelineReport:
    mode: str
    stages: tuple[StageReport, ...]
    extracted: Num
    cost: int
    total_size: int
    node_count: int
    eclass_count: int
    stop_reason: str
    rendered: str
    metric_report: MetricReport
    numeric_max_abs_error: float | None
    numeric_status: str
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
        runtime_sec=0.001157,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=2,
        after_parameter_count=2,
        before_node_count=7,
        after_node_count=7,
        simplified_python="(-9.29438919215253 + (2.93547417364396 * x[:, 2])) ** 2.0",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    2: HaskellRow(
        row=2,
        runtime_sec=0.045097,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=21,
        after_parameter_count=18,
        before_node_count=73,
        after_node_count=65,
        simplified_python="((((((x[:, 2] * ((x[:, 2] + -3.7742599465107096) * -2.221967816829757)) + ((x[:, 0] + 8.598006045013985) * ((x[:, 2] + -3.237933747438949) * 0.45672015131286553))) * (((x[:, 0] * 0.114247677264604) + (0.127799885276295 * x[:, 1])) * (((x[:, 1] * 3.60250727355285) + (x[:, 2] * 1.35381696253641)) - (x[:, 0] * 9.60356246300082)))) - ((((-9.29438919215253 + (x[:, 2] * 2.93547417364396)) ** 2.0 + (x[:, 1] * -1.31893797108115)) + (x[:, 0] * -3.0618115058751196)) * ((x[:, 0] * 1.25099225339683) + (x[:, 1] * -1.0075335593301)))) * 0.227993040492379) + 1.72695871415228)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    3: HaskellRow(
        row=3,
        runtime_sec=0.221145,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=19,
        after_parameter_count=17,
        before_node_count=66,
        after_node_count=60,
        simplified_python="((((((x[:, 2] * ((x[:, 2] + -4.623624568029607) * -1.6065924782476593)) + ((-1.70053796438431 + (x[:, 2] * 0.354410284952137)) * ((2.30920195902924 * x[:, 0]) + (-0.684514001199842 * x[:, 1])))) * (((x[:, 0] * -0.423113013561051) + (x[:, 1] * -0.820293590903207)) * np.exp((x[:, 0] * 0.648433111594964)))) - (((x[:, 1] * -10.2013161038227) + ((4.49181352322002 + (x[:, 2] * 3.34338756313595)) + (x[:, 0] * 3.32319701375876))) * (-4.40371470183112 + (x[:, 2] * 1.239832234323)) ** 2.0)) * 5.55952399242293e-2) + 2.3974618452108)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    4: HaskellRow(
        row=4,
        runtime_sec=0.058744,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=19,
        after_parameter_count=16,
        before_node_count=64,
        after_node_count=56,
        simplified_python="((((((x[:, 2] * ((x[:, 2] + -2.628407049567445) * -3.1120416896681817)) + ((x[:, 0] + 2.1411339806415373) * ((x[:, 2] + -2.7279626957213154) * 1.5619140590200937))) * (((x[:, 0] * 2.31254136991372e-2) + (-0.201859225439622 * x[:, 1])) * np.exp((x[:, 0] * 0.533684665488698)))) - (((x[:, 1] * 6.76271161563555e-2) + ((-10.1636621343434 + (x[:, 2] * 2.20402481737344)) + (x[:, 0] * 2.3693972649962904))) * ((x[:, 0] * -0.173514070270718) + (x[:, 1] * 1.4309773332239)))) * 0.418278900765285) + -0.250852384709946)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    5: HaskellRow(
        row=5,
        runtime_sec=0.243354,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=15,
        after_parameter_count=14,
        before_node_count=54,
        after_node_count=52,
        simplified_python="(1.38844130555412 * ((np.exp((x[:, 0] * 0.472180751947384)) + (x[:, 0] * -2.50335104760053)) + (((((((x[:, 0] * -5.46511983933291) + (4.7308388959051 * x[:, 1])) + ((x[:, 0] * 1.9820027542127) + (x[:, 1] * -0.507993116540432)) ** 2.0) + (-0.902691894725103 * x[:, 2])) * ((-9.71025706279119 + (x[:, 2] * 3.06939441276146)) ** 2.0 + -12.1940251737044)) * ((x[:, 0] * -1.13611973003391e-3) + (x[:, 1] * 6.9224336526928e-3))) + 3.2211759894748377)))",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    10: HaskellRow(
        row=10,
        runtime_sec=0.004942,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=9,
        after_parameter_count=9,
        before_node_count=30,
        after_node_count=30,
        simplified_python="((((-4.0257291259403 * x[:, 0]) + ((np.exp((x[:, 0] * 0.496521530470785)) * ((-9.90385501742711 + (3.09891553397754 * x[:, 2])) ** 2.0 + -12.0313313266527)) * ((x[:, 0] * -2.09790347331461e-2) + (0.15542475900314 * x[:, 1])))) * 0.218016003919151) + 2.93075744602469)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    20: HaskellRow(
        row=20,
        runtime_sec=0.001421,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=4,
        after_parameter_count=4,
        before_node_count=15,
        after_node_count=15,
        simplified_python="((((-0.453946451098545 * x[:, 2]) ** 3.0 + (x[:, 2] * 3.0472620183122)) ** 2.0 * -0.342063915063894) + 10.6692145111104)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    30: HaskellRow(
        row=30,
        runtime_sec=0.01223,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=9,
        after_parameter_count=9,
        before_node_count=35,
        after_node_count=35,
        simplified_python="((((np.exp((-1.47422568429434 * x[:, 0])) - (((-0.58923877562757 * x[:, 2]) ** 3.0 + ((x[:, 2] * 2.39236882178653) - 6.26125635041365) ** 2.0) * (-7.92212718106198e-3 * x[:, 1]))) * (np.exp((x[:, 0] * 1.19822491431359)) + (x[:, 1] * 19.3608807319069))) * 0.313237693440143) + -0.504509880559088)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    40: HaskellRow(
        row=40,
        runtime_sec=0.00085,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=4,
        after_parameter_count=4,
        before_node_count=14,
        after_node_count=14,
        simplified_python="((((0.309853513321887 * x[:, 2]) - np.exp((-4.78572314032874e-3 * x[:, 0]))) ** 2.0 * 32.3824199084528) + -4.32199869886884)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    50: HaskellRow(
        row=50,
        runtime_sec=0.495965,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=14,
        after_parameter_count=12,
        before_node_count=60,
        after_node_count=46,
        simplified_python="((((np.exp((x[:, 2] * -0.144728813168975)) ** 2.0 * (-1.54770141702422 - (x[:, 2] * 3.31046821812388))) + (((((np.exp((0.743694003014863 * x[:, 0])) * ((x[:, 0] * 9.04122619609017e-3) - (x[:, 2] * 1.21179632900701e-2))) * (8.63005732191704 - (x[:, 2] * 3.05659895630567))) - (x[:, 0] * 0.557193153898209)) - np.log(x[:, 2])) + 6.5874389967108335)) * (x[:, 1] * 1.752525486604812)) + 0.404362453868565)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    75: HaskellRow(
        row=75,
        runtime_sec=0.007188,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=12,
        after_parameter_count=12,
        before_node_count=44,
        after_node_count=44,
        simplified_python="(((((0.27658499902994 * x[:, 1]) + ((((-3.09568685791928 * x[:, 2]) - (x[:, 1] * 1.06543349470034)) + (4.80847033528277 * x[:, 0])) * ((x[:, 2] * 8.40515601044034e-3) - np.exp(((x[:, 2] * -5.89563245316599) - (x[:, 1] * -2.2128445010022)))))) * (((x[:, 2] * -1.73329369062705) + 5.01701830214846) ** 2.0 - (x[:, 0] * 3.04752460994615))) * 0.830662588380978) + 2.77768079307633)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    100: HaskellRow(
        row=100,
        runtime_sec=0.0546,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=10,
        after_parameter_count=8,
        before_node_count=31,
        after_node_count=27,
        simplified_python="((((-0.817246761082895 * x[:, 2]) + (-3.880619422186987 * ((x[:, 0] * (((x[:, 2] * -9.53755961420584e-2) + (x[:, 0] * 2.70703912792823e-2)) * (x[:, 2] + -4.659621153159492))) + -1.2365833791502565))) * (x[:, 1] * 1.134804712050934)) + 0.201404917055389)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    150: HaskellRow(
        row=150,
        runtime_sec=0.554163,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=19,
        after_parameter_count=16,
        before_node_count=72,
        after_node_count=62,
        simplified_python="(((((((5.53327625349294 * x[:, 1]) - (9.01964233670581 * x[:, 2])) - (x[:, 0] * -19.96051895060467)) * ((-2.49821516165647 - (x[:, 2] * -0.402138004942828)) * (np.log((x[:, 0] * 0.347653612616773)) - (x[:, 2] * -0.512350969799209)))) - (((x[:, 0] * 1.40058250223625) + (x[:, 1] * 0.55815097447614)) * (np.exp((x[:, 0] * 0.569380315263419)) * (((x[:, 2] * -0.841465262470357) + np.exp((x[:, 0] * 0.242251869587583))) - ((x[:, 2] * 0.466508892559209) + -2.80204970690573) ** 2.0)))) * (x[:, 1] * 6.326295494205529e-2)) + -0.455923813201509)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
    200: HaskellRow(
        row=200,
        runtime_sec=0.009585,
        memo_size=-1,
        eclass_count=-1,
        before_parameter_count=10,
        after_parameter_count=10,
        before_node_count=33,
        after_node_count=33,
        simplified_python="(((((((-7.01398906858873 * x[:, 2]) - -20.9157503480256) + (0.40965685006749 * x[:, 1])) ** 2.0 - (18.7131049568053 * x[:, 0])) / (((x[:, 1] * -2.89176232552097) + 15.8269215595989) + ((x[:, 2] * -1.76570724282939) - -5.46414193899484) ** 2.0)) * 0.689375955248431) + 0.570693365253855)",
        notes="This helper uses the source repo's exported simplifyEqSat. The public API does not expose intermediate e-graph sizes or stop reasons.",
    ),
}


@dataclass(frozen=True)
class SampleSelection:
    points: tuple[tuple[float, ...], ...]
    status: str
    attempts: int


def _real_cube_root(value: float) -> float:
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def _is_integer_like(value: float, *, tol: float = 1e-9) -> bool:
    return abs(value - round(value)) <= tol


def _render_float(value: float) -> str:
    return repr(float(value))


def _eval_power(base: float, exponent: float) -> float | None:  # noqa: PLR0911
    if not math.isfinite(base) or not math.isfinite(exponent):
        return None
    if base == 0.0 and exponent < 0.0:
        return None
    if base < 0.0 and not _is_integer_like(exponent):
        if math.isclose(exponent, 1.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9):
            return _real_cube_root(base)
        if math.isclose(exponent, -1.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9):
            cube_root = _real_cube_root(base)
            if cube_root == 0.0:
                return None
            return 1.0 / cube_root
        return None
    try:
        value = base**exponent
    except (OverflowError, ValueError):
        return None
    if isinstance(value, complex):
        return None
    return float(value) if math.isfinite(float(value)) else None


def render_num(num: Num, *, param_name: str = "params") -> str:  # noqa: C901, PLR0911, PLR0912
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
    match get_callable_args(num, Num.param):
        case (index_expr,):
            return f"{param_name}[{int(cast('i64', index_expr).value)}]"
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs), param_name=param_name)} + {render_num(cast('Num', rhs), param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs), param_name=param_name)} - {render_num(cast('Num', rhs), param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs), param_name=param_name)} * {render_num(cast('Num', rhs), param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs), param_name=param_name)} / {render_num(cast('Num', rhs), param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            return f"({render_num(cast('Num', lhs), param_name=param_name)} ** {render_num(cast('Num', rhs), param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.__neg__):
        case (inner,):
            return f"(-{render_num(inner, param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,):
            return f"exp({render_num(inner, param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,):
            return f"log({render_num(inner, param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,):
            return f"sqrt({render_num(inner, param_name=param_name)})"
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            return f"abs({render_num(inner, param_name=param_name)})"
        case _:
            pass
    msg = f"Unsupported Num node for rendering: {num!r}"
    raise TypeError(msg)


def eval_num(num: Num, env: Mapping[str, float], params: Sequence[float] | None = None) -> float | None:  # noqa: C901, PLR0911, PLR0912
    match get_callable_args(num, Num):
        case (value_expr,):
            value = float(cast("f64", value_expr).value)
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.var):
        case (name_expr,):
            env_value = env.get(cast("String", name_expr).value)
            if env_value is None or not math.isfinite(env_value):
                return None
            return float(env_value)
        case _:
            pass
    match get_callable_args(num, Num.param):
        case (index_expr,):
            if params is None:
                return None
            index = int(cast("i64", index_expr).value)
            if index < 0 or index >= len(params):
                return None
            value = float(params[index])
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            left = eval_num(cast("Num", lhs), env, params)
            right = eval_num(cast("Num", rhs), env, params)
            if left is None or right is None:
                return None
            value = left + right
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            left = eval_num(cast("Num", lhs), env, params)
            right = eval_num(cast("Num", rhs), env, params)
            if left is None or right is None:
                return None
            value = left - right
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            left = eval_num(cast("Num", lhs), env, params)
            right = eval_num(cast("Num", rhs), env, params)
            if left is None or right is None:
                return None
            value = left * right
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            left = eval_num(cast("Num", lhs), env, params)
            right = eval_num(cast("Num", rhs), env, params)
            if left is None or right is None or right == 0.0:
                return None
            value = left / right
            return value if math.isfinite(value) else None
        case _:
            pass
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            left = eval_num(cast("Num", lhs), env, params)
            right = eval_num(cast("Num", rhs), env, params)
            if left is None or right is None:
                return None
            return _eval_power(left, right)
        case _:
            pass
    match get_callable_args(num, Num.__neg__):
        case (inner,):
            inner_value = eval_num(inner, env, params)
            if inner_value is None:
                return None
            neg_value = -inner_value
            return neg_value if math.isfinite(neg_value) else None
        case _:
            pass
    match get_callable_args(num, Num.exp):
        case (inner,):
            exp_input = eval_num(inner, env, params)
            if exp_input is None:
                return None
            try:
                result = math.exp(exp_input)
            except OverflowError:
                return None
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.log):
        case (inner,):
            log_input = eval_num(inner, env, params)
            if log_input is None or log_input <= 0.0:
                return None
            result = math.log(log_input)
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.sqrt):
        case (inner,):
            sqrt_input = eval_num(inner, env, params)
            if sqrt_input is None or sqrt_input < 0.0:
                return None
            result = math.sqrt(sqrt_input)
            return result if math.isfinite(result) else None
        case _:
            pass
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            abs_input = eval_num(inner, env, params)
            if abs_input is None:
                return None
            result = abs(abs_input)
            return result if math.isfinite(result) else None
        case _:
            pass
    msg = f"Unsupported Num node for evaluation: {num!r}"
    raise TypeError(msg)


def _parameterize_num(num: Num) -> tuple[Num, tuple[float, ...]]:  # noqa: C901
    params: list[float] = []

    def go(node: Num) -> Num:  # noqa: C901, PLR0911, PLR0912
        match get_callable_args(node, Num):
            case (value_expr,):
                value = float(cast("f64", value_expr).value)
                if value.is_integer():
                    return Num(value)
                index = len(params)
                params.append(value)
                return Num.param(index)
            case _:
                pass
        match get_callable_args(node, Num.var):
            case (name_expr,):
                return Num.var(cast("String", name_expr).value)
            case _:
                pass
        match get_callable_args(node, Num.param):
            case (index_expr,):
                return Num.param(cast("i64", index_expr).value)
            case _:
                pass
        match get_callable_args(node, Num.__add__):
            case (lhs, rhs):
                return go(cast("Num", lhs)) + go(cast("Num", rhs))
            case _:
                pass
        match get_callable_args(node, Num.__sub__):
            case (lhs, rhs):
                return go(cast("Num", lhs)) - go(cast("Num", rhs))
            case _:
                pass
        match get_callable_args(node, Num.__mul__):
            case (lhs, rhs):
                return go(cast("Num", lhs)) * go(cast("Num", rhs))
            case _:
                pass
        match get_callable_args(node, Num.__truediv__):
            case (lhs, rhs):
                return go(cast("Num", lhs)) / go(cast("Num", rhs))
            case _:
                pass
        match get_callable_args(node, Num.__pow__):
            case (lhs, rhs):
                return go(cast("Num", lhs)) ** go(cast("Num", rhs))
            case _:
                pass
        match get_callable_args(node, Num.__neg__):
            case (inner,):
                return -go(inner)
            case _:
                pass
        match get_callable_args(node, Num.exp):
            case (inner,):
                return exp(go(inner))
            case _:
                pass
        match get_callable_args(node, Num.log):
            case (inner,):
                return log(go(inner))
            case _:
                pass
        match get_callable_args(node, Num.sqrt):
            case (inner,):
                return sqrt(go(inner))
            case _:
                pass
        match get_callable_args(node, Num.__abs__):
            case (inner,):
                return abs(go(inner))
            case _:
                pass
        msg = f"Unsupported Num node while parameterizing: {node!r}"
        raise TypeError(msg)

    return go(num), tuple(params)


def count_paper_params(num: Num) -> int:
    _, values = _parameterize_num(num)
    return len(values)


def _choose_domain_safe_sample_points(
    num: Num,
    input_names: Sequence[str],
    *,
    seed: int = 0,
    count: int = 64,
    max_attempts: int = 8192,
) -> SampleSelection:
    rng = np.random.default_rng(seed)
    accepted: list[tuple[float, ...]] = []
    attempts = 0
    while len(accepted) < count and attempts < max_attempts:
        attempts += 1
        candidate = tuple(float(x) for x in rng.uniform(0.25, 2.0, size=len(input_names)))
        env = dict(zip(input_names, candidate, strict=True))
        if eval_num(num, env) is None:
            continue
        accepted.append(candidate)
    status = "ok" if len(accepted) == count else "domain_limited"
    return SampleSelection(points=tuple(accepted), status=status, attempts=attempts)


def core_examples() -> tuple[Example, ...]:
    input_names = ("alpha", "beta", "theta")
    descriptions = {
        1: "Small sanity case from test/example_hl row 1.",
        50: "Function-heavy representative case from test/example_hl row 50.",
    }
    return tuple(
        Example(
            name=f"row_{row}",
            row=row,
            source=SELECTED_ROW_SOURCES[row],
            description=descriptions.get(row, f"Selected batch case from test/example_hl row {row}."),
            input_names=input_names,
            sample_points=_choose_domain_safe_sample_points(
                parse_hl_expr(SELECTED_ROW_SOURCES[row]), input_names, seed=row
            ).points,
        )
        for row in SELECTED_BATCH_ROWS
    )


def default_hl_eval_env() -> dict[str, object]:
    alpha = Num.var("alpha")
    beta = Num.var("beta")
    theta = Num.var("theta")
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


def parse_hl_expr(source: str) -> Num:
    num = eval(source, default_hl_eval_env(), {})
    if not isinstance(num, Num):
        msg = f"HL expression did not produce a Num: {source!r}"
        raise TypeError(msg)
    return num


def load_example_hl_rows() -> tuple[str, ...]:
    return tuple(line.strip() for line in EXAMPLE_HL_PATH.read_text().splitlines() if line.strip())


def load_example_hl_row(row: int) -> str:
    if row in SELECTED_ROW_SOURCES:
        return SELECTED_ROW_SOURCES[row]
    rows = load_example_hl_rows()
    if row < 1 or row > len(rows):
        msg = f"row must be between 1 and {len(rows)}, got {row}"
        raise ValueError(msg)
    return rows[row - 1]


def jacobian_rank(num: Num, *, input_names: Sequence[str], sample_points: Sequence[Sequence[float]]) -> int | None:
    parameterized_num, params = _parameterize_num(num)
    if not params:
        return 0
    params_np = np.array(params, dtype=float)
    eps = 1e-6
    rows: list[np.ndarray] = []
    for point in sample_points:
        env = dict(zip(input_names, point, strict=True))
        base = eval_num(parameterized_num, env, tuple(float(value) for value in params_np))
        if base is None:
            continue
        grads = np.zeros(params_np.size, dtype=float)
        valid = True
        for index in range(params_np.size):
            perturbed = params_np.copy()
            perturbed[index] += eps
            value = eval_num(parameterized_num, env, tuple(float(entry) for entry in perturbed))
            if value is None:
                valid = False
                break
            grads[index] = (value - base) / eps
        if valid and np.isfinite(grads).all():
            rows.append(grads)
    if not rows:
        return None
    return int(np.linalg.matrix_rank(np.vstack(rows)))


def _numeric_max_abs_error(
    original: Num,
    simplified: Num,
    *,
    input_names: Sequence[str],
    sample_points: Sequence[Sequence[float]],
) -> tuple[float | None, str]:
    if not sample_points:
        return None, "domain_limited"
    errors: list[float] = []
    for point in sample_points:
        env = dict(zip(input_names, point, strict=True))
        original_value = eval_num(original, env)
        simplified_value = eval_num(simplified, env)
        if original_value is None:
            continue
        if simplified_value is None:
            return None, "simplified_domain_error"
        errors.append(abs(original_value - simplified_value))
    if not errors:
        return None, "domain_limited"
    return float(max(errors)), "ok"


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
    num: Num,
    rules: egglog.Ruleset | egglog.Schedule,
    *,
    node_cutoff: int | None,
    iteration_limit: int,
    scheduler: egglog.BackOff | None = None,
) -> StageReport:
    egraph = egglog.EGraph()
    start = time.perf_counter()
    egraph.register(num)
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
    extracted, cost = egraph.extract(num, include_cost=True)
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


def _baseline_rulesets() -> tuple[egglog.Schedule, egglog.Schedule]:
    rewrite_const = const_analysis_rules | const_reduction_rules
    rewrite_all = const_analysis_rules | basic_rules | const_reduction_rules | const_fusion_rules | fun_rules
    return rewrite_const, rewrite_all


def _pipeline_metrics(
    original: Num,
    extracted: Num,
    *,
    input_names: Sequence[str],
    sample_points: Sequence[Sequence[float]],
) -> MetricReport:
    before = count_paper_params(original)
    after = count_paper_params(extracted)
    optimal = jacobian_rank(original, input_names=input_names, sample_points=sample_points)
    return MetricReport(
        before_parameter_count=before,
        after_parameter_count=after,
        reduction_ratio=((before - after) / before) if before else 0.0,
        optimal_parameter_count=optimal,
        gap_to_optimal=(after - optimal) if optimal is not None else None,
        optimal_status="ok" if optimal is not None else "domain_limited",
    )


def _baseline_notes(stage: StageReport) -> tuple[str, ...]:
    if stage.stop_reason == "cutoff_hit":
        return ("Baseline stopped at the user-imposed node cutoff before reaching a fixpoint.",)
    if stage.stop_reason == "budget_hit":
        return ("Baseline hit the iteration budget before proving saturation.",)
    return ()


def run_baseline_pipeline(
    num: Num,
    *,
    node_cutoff: int,
    iteration_limit: int,
    input_names: Sequence[str] = ("alpha", "beta", "theta"),
    sample_points: Sequence[Sequence[float]] | None = None,
) -> PipelineReport:
    rewrite_const, rewrite_all = _baseline_rulesets()
    const_scheduler = back_off(match_limit=100, ban_length=10)
    all_scheduler = back_off(match_limit=2500, ban_length=30)
    sample_selection = (
        SampleSelection(tuple(tuple(float(value) for value in row) for row in sample_points), "ok", len(sample_points))
        if sample_points is not None
        else _choose_domain_safe_sample_points(num, input_names)
    )
    const_stage = _run_stage(
        "rewrite_const",
        num,
        rewrite_const,
        node_cutoff=node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=const_scheduler,
    )
    current = const_stage.extracted
    stages = [const_stage]
    previous_rendered = render_num(current)
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
        current_rendered = render_num(current)
        if current_rendered == previous_rendered:
            break
        previous_rendered = current_rendered
        if stop_reason != "saturated":
            break
    metrics = _pipeline_metrics(num, current, input_names=input_names, sample_points=sample_selection.points)
    numeric_max_abs_error, numeric_status = _numeric_max_abs_error(
        num,
        current,
        input_names=input_names,
        sample_points=sample_selection.points,
    )
    notes = list(_baseline_notes(stages[-1]))
    if sample_selection.status != "ok":
        notes.append(
            "Numeric validation used a reduced domain-safe sample because not enough valid points were available."
        )
    return PipelineReport(
        mode="baseline",
        stages=tuple(stages),
        extracted=current,
        cost=stages[-1].cost,
        total_size=stages[-1].total_size,
        node_count=stages[-1].node_count,
        eclass_count=stages[-1].eclass_count,
        stop_reason=stop_reason,
        rendered=render_num(current),
        metric_report=metrics,
        numeric_max_abs_error=numeric_max_abs_error,
        numeric_status=numeric_status if sample_selection.status == "ok" else sample_selection.status,
        notes=tuple(notes),
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
    for row in rows:
        source = load_example_hl_row(row)
        num = parse_hl_expr(source)
        baseline_total += run_baseline_pipeline(num, node_cutoff=node_cutoff, iteration_limit=iteration_limit).total_sec
    scale = len(load_example_hl_rows()) / len(rows)
    return {
        "sampled_rows": float(len(rows)),
        "baseline_sample_sec": baseline_total,
        "baseline_projected_sec": baseline_total * scale,
    }


def estimate_corpus_runtime() -> dict[str, float]:
    return sample_runtime_probe()
