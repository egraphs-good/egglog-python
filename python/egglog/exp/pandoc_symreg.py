"""
Helpers for reproducing the pandoc-symreg EqSat examples in egglog.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, TypeAlias, cast

import numpy as np

import egglog
from egglog import *
from egglog.deconstruct import get_callable_args
from egglog.exp.program_gen import EvalProgram, Program, eval_program_ruleset, program_gen_ruleset

__all__ = [
    "Comparison",
    "MetricReport",
    "PipelineReport",
    "Witness",
    "build_dramatic_witness",
    "build_pysr_stress_witness",
    "build_readable_witness",
    "build_sanity_witnesses",
    "compare_witness",
    "compile_term_callable",
    "count_float_params",
    "run_binary_pipeline",
    "run_multiset_pipeline",
    "selected_witnesses",
]


pandoc_language = ruleset()
const_analysis_rules = ruleset(name="pandoc_const_analysis")
basic_rules = ruleset(name="pandoc_basic")
fun_rules = ruleset(name="pandoc_fun")
const_reduction_rules = ruleset(name="pandoc_const_reduction")
const_fusion_rules = ruleset(name="pandoc_const_fusion")
pandoc_program_rules = ruleset(name="pandoc_program")
pandoc_multiset_rules = ruleset(name="pandoc_multiset")


class OptionalF64(Expr, ruleset=pandoc_language):
    none: ClassVar[OptionalF64]

    @classmethod
    def some(cls, value: f64Like) -> OptionalF64: ...


class Term(Expr, ruleset=pandoc_language):
    @method(cost=5)
    def __init__(self, value: f64Like) -> None: ...

    @method(cost=1)
    @classmethod
    def var(cls, name: StringLike) -> Term: ...

    @method(cost=1)
    def __add__(self, other: TermLike) -> Term: ...

    @method(cost=3)
    def __sub__(self, other: TermLike) -> Term: ...

    @method(cost=1)
    def __mul__(self, other: TermLike) -> Term: ...

    @method(cost=3)
    def __truediv__(self, other: TermLike) -> Term: ...

    @method(cost=3)
    def __pow__(self, other: TermLike) -> Term: ...

    @method(cost=1)
    def __neg__(self) -> Term: ...

    @method(cost=1)
    def exp(self) -> Term: ...

    @method(cost=1)
    def log(self) -> Term: ...

    @method(cost=1)
    def sqrt(self) -> Term: ...

    @method(cost=1)
    def __abs__(self) -> Term: ...

    def __radd__(self, other: TermLike) -> Term: ...

    def __rsub__(self, other: TermLike) -> Term: ...

    def __rmul__(self, other: TermLike) -> Term: ...

    def __rtruediv__(self, other: TermLike) -> Term: ...

    def __rpow__(self, other: TermLike) -> Term: ...


TermLike: TypeAlias = Term | StringLike | float
converter(float, Term, Term)
converter(String, Term, Term.var)
converter(str, Term, Term.var)


@function(ruleset=const_analysis_rules, merge=lambda old, _new: old)
def const_value(expr: Term) -> OptionalF64: ...


@function(ruleset=pandoc_program_rules)
def term_program(expr: Term) -> Program: ...


@function(ruleset=pandoc_multiset_rules)
def sum_(xs: MultiSetLike[Term, TermLike]) -> Term: ...


@function(ruleset=pandoc_multiset_rules)
def product_(xs: MultiSetLike[Term, TermLike]) -> Term: ...


def _zero() -> Term:
    return Term(0.0)


def _one() -> Term:
    return Term(1.0)


def _is_some(opt: OptionalF64, value: f64) -> Unit:
    return opt == OptionalF64.some(value)


@const_analysis_rules.register
def _const_analysis(
    expr: Term,
    x: Term,
    y: Term,
    a: f64,
    b: f64,
    s: String,
) -> Iterable[RewriteOrRule]:
    yield rule(eq(expr).to(Term(a))).then(set_(const_value(expr)).to(OptionalF64.some(a)))
    yield rule(eq(expr).to(Term.var(s))).then(set_(const_value(expr)).to(OptionalF64.none))

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
    yield rule(eq(expr).to(x), const_value(x) == OptionalF64.some(a)).then(union(expr).with_(Term(a)))


@basic_rules.register
def _basic_rewrites(x: Term, y: Term, z: Term, a: Term, b: Term, c: f64) -> Iterable[RewriteOrRule]:
    one = _one()
    _zero()
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)
    yield rewrite(x * x).to(x**2.0)
    yield rewrite((x**a) * x).to(x ** (a + 1.0))
    yield rewrite((x**a) * (x**b)).to(x ** (a + b))
    yield rewrite((x + y) + z).to(x + (y + z))
    yield rewrite((x * y) * z).to(x * (y * z))
    yield rewrite(x - (y + z)).to((x - y) - z)
    yield rewrite(x - (y - z)).to((x - y) + z)
    yield rewrite(-(x + y)).to((-x) - y)
    yield rewrite(x - a).to(x + (-a), const_value(a) != OptionalF64.none, const_value(x) == OptionalF64.none)
    yield rewrite(x - (a * y)).to(
        x + ((-a) * y),
        const_value(a) != OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
    yield rewrite((one / x) * (one / y)).to(one / (x * y))


@fun_rules.register
def _fun_rewrites(x: Term, y: Term, a: Term, b: Term, c: f64) -> Iterable[RewriteOrRule]:
    half = Term(0.5)
    yield rewrite((x * y).log()).to(x.log() + y.log(), const_value(x) == OptionalF64.some(c), c >= 0.0, c > 0.0)
    yield rewrite((x**a) * (x**b)).to(x ** (a + b))
    yield rewrite((x / y).log()).to(x.log() - y.log(), const_value(x) == OptionalF64.some(c), c >= 0.0, c > 0.0)
    yield rewrite((x**y).log()).to(y * x.log(), const_value(y) == OptionalF64.some(c), c >= 0.0, c > 0.0)
    yield rewrite((x.sqrt()).log()).to(half * x.log(), const_value(x) == OptionalF64.none)
    yield rewrite((x.exp()).log()).to(x, const_value(x) == OptionalF64.none)
    yield rewrite((x.log()).exp()).to(x, const_value(x) == OptionalF64.none)
    yield rewrite(x**0.5).to(x.sqrt())
    yield rewrite((a * x).sqrt()).to(a.sqrt() * x.sqrt(), const_value(a) == OptionalF64.some(c), c >= 0.0)
    yield rewrite((a * (x - y)).sqrt()).to((-a).sqrt() * (y - x).sqrt(), const_value(a) == OptionalF64.some(c), c < 0.0)
    yield rewrite((a * (b + y)).sqrt()).to(
        (-a).sqrt() * (b - y).sqrt(),
        const_value(a) == OptionalF64.some(c),
        c < 0.0,
        const_value(b) == OptionalF64.some(c),
        c < 0.0,
    )
    yield rewrite((a / x).sqrt()).to(a.sqrt() / x.sqrt(), const_value(a) == OptionalF64.some(c), c >= 0.0)
    yield rewrite(abs(x * y)).to(abs(x) * abs(y))


@const_reduction_rules.register
def _const_reduction(x: Term, y: Term, z: Term, a: Term, b: Term, i: f64, j: f64) -> Iterable[RewriteOrRule]:
    zero = _zero()
    one = _one()
    yield rewrite(zero + x).to(x)
    yield rewrite(x + zero).to(x)
    yield rewrite(x - zero).to(x)
    yield rewrite(zero * x).to(zero)
    yield rewrite(x * zero).to(zero)
    yield rewrite(zero / x).to(zero)
    yield rewrite(x - x).to(zero)
    yield rewrite(x / x).to(one, const_value(x) == OptionalF64.some(i), i > 0.0)
    yield rewrite(x / x).to(one, const_value(x) == OptionalF64.some(i), i < 0.0)
    yield rewrite(x**1.0).to(x)
    yield rewrite(zero**x).to(zero)
    yield rewrite(one**x).to(one)
    yield rewrite(x * (one / x)).to(one, const_value(x) == OptionalF64.some(i), i > 0.0)
    yield rewrite(x * (one / x)).to(one, const_value(x) == OptionalF64.some(i), i < 0.0)
    yield rewrite((x * y) + (x * z)).to(x * (y + z))
    yield rewrite(x - ((-1.0) * y)).to(x + y, const_value(y) == OptionalF64.none)
    yield rewrite(x + (-y)).to(x - y, const_value(y) == OptionalF64.none)
    yield rewrite(_zero() - x).to(-x, const_value(x) == OptionalF64.none)
    yield rewrite((a * x) + (b * y)).to(
        a * (x + (b / a) * y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )
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
    yield rewrite((a * x) / y).to(
        a * (x / y),
        const_value(a) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )


@const_fusion_rules.register
def _const_fusion(x: Term, y: Term, a: Term, b: Term) -> Iterable[RewriteOrRule]:
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
    yield rewrite((a * x) / (b * y)).to(
        (a / b) * (x / y),
        const_value(a) != OptionalF64.none,
        const_value(b) != OptionalF64.none,
        const_value(x) == OptionalF64.none,
        const_value(y) == OptionalF64.none,
    )


@pandoc_multiset_rules.register
def _multiset_rewrites(
    a: Term,
    b: Term,
    c: Term,
    d: Term,
    xs: MultiSet[Term],
    ys: MultiSet[Term],
    zs: MultiSet[Term],
    i: f64,
    j: f64,
) -> Iterable[RewriteOrRule]:
    yield rewrite(a + b, subsume=True).to(sum_(MultiSet(a, b)))
    yield rewrite(a * b, subsume=True).to(product_(MultiSet(a, b)))
    yield rule(a == sum_(xs), b == sum_(ys)).then(union(a + b).with_(sum_(xs + ys)))
    yield rule(a == product_(xs), b == product_(ys)).then(union(a * b).with_(product_(xs + ys)))
    yield rule(a == sum_(xs), xs.length() == i64(1)).then(union(a).with_(xs.pick()))
    yield rule(a == product_(xs), xs.length() == i64(1)).then(union(a).with_(xs.pick()))
    yield rewrite(sum_(MultiSet[Term]())).to(_zero())
    yield rewrite(product_(MultiSet[Term]())).to(_one())
    yield rule(
        a == sum_(xs),
        b == Term(i),
        xs.contains(b),
        ys == xs.remove(b),
        c == Term(j),
        ys.contains(c),
    ).then(union(a).with_(sum_(ys.remove(c).insert(Term(i + j)))))
    yield rule(
        a == product_(xs),
        b == Term(i),
        xs.contains(b),
        ys == xs.remove(b),
        c == Term(j),
        ys.contains(c),
    ).then(union(a).with_(product_(ys.remove(c).insert(Term(i * j)))))
    yield rule(
        a == product_(xs),
        b == sum_(ys),
        xs.contains(b),
        xs.length() > 1,
        zs == xs.remove(b),
    ).then(union(a).with_(sum_(ys.map(lambda t: product_(zs.insert(t))))))


@pandoc_program_rules.register
def _term_program(x: Term, y: Term, value: f64, name: String) -> Iterable[RewriteOrRule]:
    yield rewrite(term_program(Term(value)), subsume=True).to(Program(value.to_string()))
    yield rewrite(term_program(Term.var(name)), subsume=True).to(Program(name, True))
    yield rewrite(term_program(x + y), subsume=True).to(Program("(") + term_program(x) + " + " + term_program(y) + ")")
    yield rewrite(term_program(x - y), subsume=True).to(Program("(") + term_program(x) + " - " + term_program(y) + ")")
    yield rewrite(term_program(x * y), subsume=True).to(Program("(") + term_program(x) + " * " + term_program(y) + ")")
    yield rewrite(term_program(x / y), subsume=True).to(Program("(") + term_program(x) + " / " + term_program(y) + ")")
    yield rewrite(term_program(x**y), subsume=True).to(Program("(") + term_program(x) + " ** " + term_program(y) + ")")
    yield rewrite(term_program(-x), subsume=True).to(Program("(-") + term_program(x) + ")")
    yield rewrite(term_program(x.exp()), subsume=True).to(Program("np.exp(") + term_program(x) + ")")
    yield rewrite(term_program(x.log()), subsume=True).to(Program("np.log(") + term_program(x) + ")")
    yield rewrite(term_program(x.sqrt()), subsume=True).to(Program("np.sqrt(") + term_program(x) + ")")
    yield rewrite(term_program(abs(x)), subsume=True).to(Program("np.abs(") + term_program(x) + ")")


@dataclass(frozen=True)
class Witness:
    name: str
    source_path: str
    row: int
    description: str
    expr: Term
    input_names: tuple[str, ...]
    sample_points: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class MetricReport:
    parameter_count: int
    unique_parameter_count: int
    parameter_reduction_ratio: float
    jacobian_rank: int


@dataclass(frozen=True)
class PipelineReport:
    mode: str
    extracted: Term
    cost: int
    total_size: int
    register_sec: float
    run_sec: float
    extract_sec: float
    function_sizes: list[tuple[egglog.ExprCallable, int]]
    python_source: str
    metric_report: MetricReport
    numeric_max_abs_error: float
    notes: tuple[str, ...] = ()

    @property
    def total_sec(self) -> float:
        return self.register_sec + self.run_sec + self.extract_sec


@dataclass(frozen=True)
class Comparison:
    witness: Witness
    original_source: str
    binary: PipelineReport
    multiset: PipelineReport | None


def build_readable_witness() -> Witness:
    x1 = Term.var("x1")
    x2 = Term.var("x2")
    expr = -(abs((-1.3 * x1) + (1.56 * x2)).log().exp())
    return Witness(
        name="readable",
        source_path="/Users/saul/p/pandoc-symreg/problems",
        row=4,
        description="Small nonlinear example from problems:4 with additive structure inside abs/log/exp.",
        expr=expr,
        input_names=("x1", "x2"),
        sample_points=((-2.0, -1.0), (-1.0, 0.5), (0.5, 1.5), (1.0, -1.5), (2.0, 0.25)),
    )


def build_dramatic_witness() -> Witness:
    theta = Term.var("theta")
    sigma = Term.var("sigma")
    expr = (
        theta * -0.464405298
        + theta * theta * 0.102369047
        + sigma * theta * -0.175176339
        + sigma * 0.736577255
        + (sigma * theta * 2.55279029).exp() * 2.86662658e-11
        + (959.586017 / ((sigma * sigma * (theta * -0.891744723).exp() * 83.1663039) + 317.624683))
        + -2.69989657
    )
    return Witness(
        name="dramatic",
        source_path="/Users/saul/p/pandoc-symreg/examples/feynman_I_6_2.hl",
        row=11,
        description="A/C-heavy Feynman example with additive linear terms and nonlinear atoms.",
        expr=expr,
        input_names=("theta", "sigma"),
        sample_points=((0.2, 0.3), (0.5, 1.0), (1.0, 0.5), (1.5, 1.2), (2.0, 0.8)),
    )


def build_pysr_stress_witness() -> Witness:
    x1 = Term.var("x1")
    x2 = Term.var("x2")
    expr = (
        (
            (((x2 / -1.1526895432904412) * x2) - ((0.1560929128293564 + 0.04873876471741229) + -0.10767615949898787))
            * -0.7499061083076463
        ).exp()
        - -1.1526895432904412
    ) * (
        (((x2 * x2).log() - 78.41442075024227 + 1.0254454456547013).exp())
        + (((x1 * (x1 + 0.0034420466463509335)).log() - 0.6739116903430199) * 0.1560929128293564)
        + 1.4227637491659069
    )
    return Witness(
        name="pysr-stress",
        source_path="/Users/saul/p/pandoc-symreg/examples/example.pysr",
        row=3,
        description="Optional nested PySR stress case with deeper nonlinear structure.",
        expr=expr,
        input_names=("x1", "x2"),
        sample_points=((0.5, 0.5), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 1.25)),
    )


def build_sanity_witnesses() -> tuple[Witness, Witness]:
    x1 = Term.var("X1")
    x2 = Term.var("X2")
    x3 = Term.var("X3")
    erro1 = Witness(
        name="erro-1",
        source_path="/Users/saul/p/pandoc-symreg/erro",
        row=1,
        description="Small factorization sanity case from erro:1. This one does reduce parameter count.",
        expr=9.453222 + (-33847.042969) * ((0.000032 * x3) + (0.000032 * x2)),
        input_names=("X2", "X3"),
        sample_points=((0.5, 1.0), (1.0, 2.0), (2.0, 3.0)),
    )
    erro2 = Witness(
        name="erro-2",
        source_path="/Users/saul/p/pandoc-symreg/erro",
        row=2,
        description="Division cancellation sanity case from erro:2.",
        expr=5.0 + ((3.0 * x2) / x2 - (3.0 * x1) / x1),
        input_names=("X1", "X2"),
        sample_points=((0.5, 1.0), (1.0, 2.0), (2.0, 3.0)),
    )
    return erro1, erro2


def selected_witnesses() -> tuple[Witness, Witness, Witness]:
    return build_readable_witness(), build_dramatic_witness(), build_pysr_stress_witness()


def count_float_params(expr: Term) -> int:
    return len(_term_to_python(expr, parameterize_floats=True)[1])


def _raw_python_source(expr: Term) -> str:
    return _term_to_python(expr)[0]


def _term_to_python(expr: Term, *, parameterize_floats: bool = False) -> tuple[str, list[float]]:
    params: list[float] = []

    def multiset_items(ms: Expr) -> tuple[Term, ...]:
        if args := get_callable_args(ms, MultiSet):
            return tuple(cast("Term", arg) for arg in args)
        if args := get_callable_args(ms, MultiSet.single):
            value, count = args
            return tuple(cast("Term", value) for _ in range(cast("i64", count).value))
        msg = f"Unsupported multiset node: {ms!r}"
        raise TypeError(msg)

    def rec(node: Term) -> str:
        if args := get_callable_args(node, Term):
            (value_expr,) = args
            value = float(cast("f64", value_expr).value)
            if parameterize_floats and not value.is_integer():
                idx = len(params)
                params.append(value)
                return f"params[{idx}]"
            return repr(value)
        if args := get_callable_args(node, Term.var):
            (name_expr,) = args
            return cast("String", name_expr).value
        if args := get_callable_args(node, Term.__add__):
            return f"({rec(cast('Term', args[0]))} + {rec(cast('Term', args[1]))})"
        if args := get_callable_args(node, Term.__sub__):
            return f"({rec(cast('Term', args[0]))} - {rec(cast('Term', args[1]))})"
        if args := get_callable_args(node, Term.__mul__):
            return f"({rec(cast('Term', args[0]))} * {rec(cast('Term', args[1]))})"
        if args := get_callable_args(node, Term.__truediv__):
            return f"({rec(cast('Term', args[0]))} / {rec(cast('Term', args[1]))})"
        if args := get_callable_args(node, Term.__pow__):
            return f"({rec(cast('Term', args[0]))} ** {rec(cast('Term', args[1]))})"
        if args := get_callable_args(node, Term.__neg__):
            return f"(-{rec(cast('Term', args[0]))})"
        if args := get_callable_args(node, Term.exp):
            return f"np.exp({rec(cast('Term', args[0]))})"
        if args := get_callable_args(node, Term.log):
            return f"np.log({rec(cast('Term', args[0]))})"
        if args := get_callable_args(node, Term.sqrt):
            return f"np.sqrt({rec(cast('Term', args[0]))})"
        if args := get_callable_args(node, Term.__abs__):
            return f"np.abs({rec(cast('Term', args[0]))})"
        if args := get_callable_args(node, sum_):
            (xs,) = args
            rendered = [rec(term) for term in multiset_items(cast("Expr", xs))]
            return "(" + " + ".join(rendered) + ")" if rendered else "0.0"
        if args := get_callable_args(node, product_):
            (xs,) = args
            rendered = [rec(term) for term in multiset_items(cast("Expr", xs))]
            return "(" + " * ".join(rendered) + ")" if rendered else "1.0"
        msg = f"Unsupported term node: {node!r}"
        raise TypeError(msg)

    return rec(expr), params


def _compile_python_source(source: str, input_names: tuple[str, ...], with_params: bool) -> object:
    if len(input_names) == 2 and not with_params:
        args = ", ".join(input_names)
    elif len(input_names) == 2 and with_params:
        args = ", ".join((*input_names, "params"))
    elif len(input_names) == 1 and with_params:
        args = f"{input_names[0]}, params"
    elif len(input_names) == 1 and not with_params:
        args = input_names[0]
    else:
        args = ", ".join((*input_names, "params")) if with_params else ", ".join(input_names)
    namespace = {"np": np}
    exec(f"def __fn({args}):\n    return {source}\n", namespace)
    return namespace["__fn"]


def compile_term_callable(expr: Term, input_names: tuple[str, ...]) -> object:
    if len(input_names) not in (1, 2, 3):
        msg = f"Only 1-3 input names are supported, got {input_names!r}"
        raise ValueError(msg)
    if _contains_multiset(expr):
        return _compile_python_source(_raw_python_source(expr), input_names, with_params=False)
    egraph = egglog.EGraph()
    body = term_program(expr)
    if len(input_names) == 1:
        fn_program = body.function_two(Program(input_names[0], True), Program("_unused", True), "__fn")
    elif len(input_names) == 2:
        fn_program = body.function_two(Program(input_names[0], True), Program(input_names[1], True), "__fn")
    else:
        fn_program = body.function_three(
            Program(input_names[0], True),
            Program(input_names[1], True),
            Program(input_names[2], True),
            "__fn",
        )
    evaluated = EvalProgram(fn_program, {"np": np})
    egraph.register(evaluated)
    egraph.run((pandoc_program_rules | program_gen_ruleset | eval_program_ruleset).saturate())
    compiled = egraph.extract(evaluated.as_py_object).value
    if len(input_names) == 1:
        return lambda x: compiled(x, None)
    return compiled


def _contains_multiset(expr: Term) -> bool:
    if get_callable_args(expr, sum_) or get_callable_args(expr, product_):
        return True
    args = get_callable_args(expr)
    if not args:
        return False
    return any(_contains_multiset(cast("Term", arg)) for arg in args if isinstance(arg, Term))


def _parameter_metrics(expr: Term, witness: Witness) -> MetricReport:
    source, params = _term_to_python(expr, parameterize_floats=True)
    parameter_count = len(params)
    fn = _compile_python_source(source, witness.input_names, with_params=True)
    jacobian_rank = _jacobian_rank(fn, witness.sample_points, np.array(params, dtype=float))
    return MetricReport(
        parameter_count=parameter_count,
        unique_parameter_count=parameter_count,
        parameter_reduction_ratio=0.0,
        jacobian_rank=jacobian_rank,
    )


def _jacobian_rank(fn: object, sample_points: tuple[tuple[float, ...], ...], params: np.ndarray) -> int:
    if params.size == 0:
        return 0
    jac_rows: list[np.ndarray] = []
    eps = 1e-6
    for point in sample_points:
        row = np.zeros(params.size, dtype=float)
        base = float(fn(*point, params))  # type: ignore[misc]
        for i in range(params.size):
            perturbed = params.copy()
            perturbed[i] += eps
            row[i] = (float(fn(*point, perturbed)) - base) / eps  # type: ignore[misc]
        jac_rows.append(row)
    return int(np.linalg.matrix_rank(np.vstack(jac_rows)))


def _max_abs_error(original: object, extracted: object, sample_points: tuple[tuple[float, ...], ...]) -> float:
    errors = [abs(float(original(*point)) - float(extracted(*point))) for point in sample_points]  # type: ignore[misc]
    return float(max(errors, default=0.0))


def _run_schedule(
    schedule: egglog.Schedule | egglog.Ruleset, expr: Term
) -> tuple[Term, int, list[tuple[egglog.ExprCallable, int]], float, float, float]:
    egraph = egglog.EGraph()
    start = time.perf_counter()
    egraph.register(expr)
    register_sec = time.perf_counter() - start

    start = time.perf_counter()
    egraph.run(schedule)
    run_sec = time.perf_counter() - start

    start = time.perf_counter()
    extracted, cost = egraph.extract(expr, include_cost=True)
    extract_sec = time.perf_counter() - start
    return extracted, cost, egraph.all_function_sizes(), register_sec, run_sec, extract_sec


def _binary_schedule() -> tuple[egglog.Schedule, egglog.Schedule]:
    rewrite_const_rules = const_analysis_rules | basic_rules | const_reduction_rules
    rewrite_all_rules = rewrite_const_rules | const_fusion_rules | fun_rules
    const_schedule = rewrite_const_rules.saturate()
    all_schedule = rewrite_all_rules.saturate()
    return const_schedule, all_schedule


def _run_binary_impl(expr: Term) -> tuple[Term, int, list[tuple[egglog.ExprCallable, int]], float, float, float]:
    const_schedule, all_schedule = _binary_schedule()
    current, cost, sizes, register_sec, run_sec, extract_sec = _run_schedule(const_schedule, expr)
    previous_source = _raw_python_source(current)
    for _ in range(2):
        nxt, cost, sizes, r_i, t_i, e_i = _run_schedule(all_schedule, current)
        register_sec += r_i
        run_sec += t_i
        extract_sec += e_i
        next_source = _raw_python_source(nxt)
        current = nxt
        if next_source == previous_source:
            break
        previous_source = next_source
    return current, cost, sizes, register_sec, run_sec, extract_sec


def _run_multiset_impl(
    expr: Term,
) -> tuple[Term, int, list[tuple[egglog.ExprCallable, int]], float, float, float, tuple[str, ...]]:
    rewrite_const_schedule, rewrite_all_schedule = _binary_schedule()
    current, cost, sizes, register_sec, run_sec, extract_sec = _run_schedule(rewrite_const_schedule, expr)
    current, cost, sizes, r_i, t_i, e_i = _run_schedule(
        run(pandoc_multiset_rules, scheduler=back_off(match_limit=512, ban_length=3)) * 4,
        current,
    )
    register_sec += r_i
    run_sec += t_i
    extract_sec += e_i
    previous_source = _raw_python_source(current)
    for _ in range(2):
        nxt, cost, sizes, r_i, t_i, e_i = _run_schedule(rewrite_all_schedule, current)
        register_sec += r_i
        run_sec += t_i
        extract_sec += e_i
        next_source = _raw_python_source(nxt)
        current = nxt
        if next_source == previous_source:
            break
        previous_source = next_source
    return (
        current,
        cost,
        sizes,
        register_sec,
        run_sec,
        extract_sec,
        (
            "Multiset path currently ports A/C flattening, distributive expansion, and constant combining.",
            "Binary pandoc rules are rerun after reifying the multiset result to keep the rest of the EqSat pipeline active.",
        ),
    )


def run_binary_pipeline(witness: Witness) -> PipelineReport:
    extracted, cost, function_sizes, register_sec, run_sec, extract_sec = _run_binary_impl(witness.expr)
    original_fn = compile_term_callable(witness.expr, witness.input_names)
    extracted_fn = compile_term_callable(extracted, witness.input_names)
    metrics = _parameter_metrics(extracted, witness)
    before_params = len(_term_to_python(witness.expr, parameterize_floats=True)[1])
    after_params = metrics.parameter_count
    metrics = MetricReport(
        parameter_count=after_params,
        unique_parameter_count=metrics.unique_parameter_count,
        parameter_reduction_ratio=((before_params - after_params) / before_params) if before_params else 0.0,
        jacobian_rank=metrics.jacobian_rank,
    )
    return PipelineReport(
        mode="binary",
        extracted=extracted,
        cost=cost,
        total_size=sum(size for _, size in function_sizes),
        register_sec=register_sec,
        run_sec=run_sec,
        extract_sec=extract_sec,
        function_sizes=function_sizes,
        python_source=_raw_python_source(extracted),
        metric_report=metrics,
        numeric_max_abs_error=_max_abs_error(original_fn, extracted_fn, witness.sample_points),
    )


def run_multiset_pipeline(witness: Witness) -> PipelineReport:
    extracted, cost, function_sizes, register_sec, run_sec, extract_sec, notes = _run_multiset_impl(witness.expr)
    original_fn = compile_term_callable(witness.expr, witness.input_names)
    extracted_fn = compile_term_callable(extracted, witness.input_names)
    metrics = _parameter_metrics(extracted, witness)
    before_params = len(_term_to_python(witness.expr, parameterize_floats=True)[1])
    after_params = metrics.parameter_count
    metrics = MetricReport(
        parameter_count=after_params,
        unique_parameter_count=metrics.unique_parameter_count,
        parameter_reduction_ratio=((before_params - after_params) / before_params) if before_params else 0.0,
        jacobian_rank=metrics.jacobian_rank,
    )
    return PipelineReport(
        mode="multiset",
        extracted=extracted,
        cost=cost,
        total_size=sum(size for _, size in function_sizes),
        register_sec=register_sec,
        run_sec=run_sec,
        extract_sec=extract_sec,
        function_sizes=function_sizes,
        python_source=_raw_python_source(extracted),
        metric_report=metrics,
        numeric_max_abs_error=_max_abs_error(original_fn, extracted_fn, witness.sample_points),
        notes=notes,
    )


def compare_witness(name: str) -> Comparison:
    witness_map = {w.name: w for w in selected_witnesses()}
    witness = witness_map[name]
    binary = run_binary_pipeline(witness)
    try:
        multiset = run_multiset_pipeline(witness)
    except Exception as exc:  # pragma: no cover - surfaced in tutorial output
        multiset = None
        binary = PipelineReport(
            mode=binary.mode,
            extracted=binary.extracted,
            cost=binary.cost,
            total_size=binary.total_size,
            register_sec=binary.register_sec,
            run_sec=binary.run_sec,
            extract_sec=binary.extract_sec,
            function_sizes=binary.function_sizes,
            python_source=binary.python_source,
            metric_report=binary.metric_report,
            numeric_max_abs_error=binary.numeric_max_abs_error,
            notes=(*binary.notes, f"Multiset pipeline failed: {exc}"),
        )
    return Comparison(witness, _raw_python_source(witness.expr), binary, multiset)
