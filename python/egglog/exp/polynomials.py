"""
Helpers for the polynomial container examples in the containers docs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import egglog
import egglog.exp.array_api as enp

__all__ = [
    "Report",
    "TotalReport",
    "bending_function",
    "distribute",
    "factoring",
    "remove_subtraction",
    "run_example",
    "symbolic_bending_examples",
    "symbolic_bending_inputs",
    "try_example",
]


def bending_function(Q, Bp, Bpp):
    xp = Q.__array_namespace__()
    QM = xp.reshape(Q, (4, 3)).T

    yip = xp.vecdot(QM, Bp)
    yipp = xp.vecdot(QM, Bpp)
    num = xp.linalg.vector_norm(xp.cross(yip, yipp))
    den = xp.linalg.vector_norm(yip) ** 3
    return (num / den) ** 2


def symbolic_bending_inputs() -> tuple[enp.NDArray, enp.NDArray, enp.NDArray]:
    bp = enp.NDArray([enp.Value.var(f"bp{i}") for i in range(1, 5)])
    bpp = enp.NDArray([enp.Value.var(f"bpp{i}") for i in range(1, 5)])
    q = enp.NDArray([enp.Value.var(f"q{i}") for i in range(1, 13)])
    return bp, bpp, q


def symbolic_bending_examples() -> tuple[enp.NDArray, enp.NDArray]:
    bp, bpp, q = symbolic_bending_inputs()
    function_bending = enp.NDArray(bending_function(q, bp, bpp).eval())
    gradient_bending = enp.NDArray(function_bending.diff(q).eval())
    return function_bending, gradient_bending


@egglog.ruleset
def remove_subtraction(a: enp.Value, b: enp.Value):
    yield egglog.rewrite(a - b, subsume=True).to(a + enp.Value.from_int(-1) * b)


@egglog.ruleset
def distribute(a: enp.Value, b: enp.Value, c: enp.Value):
    yield egglog.rewrite((a + b) * c, subsume=True).to(a * c + b * c)
    yield egglog.rewrite(c * (a + b), subsume=True).to(c * a + c * b)


@egglog.ruleset
def factoring(a: enp.Value, b: enp.Value, c: enp.Value):
    yield egglog.birewrite((a + b) * c).to(a * c + b * c)
    yield egglog.rewrite(a * b).to(b * a)
    yield egglog.rewrite(a + b).to(b + a)
    yield egglog.birewrite(a * (b * c)).to((a * b) * c)


@dataclass(frozen=True)
class Report:
    register_sec: float
    run_sec: float
    extract_sec: float
    extracted: enp.NDArray
    cost: int
    function_sizes: list[tuple[egglog.ExprCallable, int]]
    updated: bool

    @property
    def total_sec(self) -> float:
        return self.register_sec + self.run_sec + self.extract_sec

    @property
    def total_size(self) -> int:
        return sum(size for _, size in self.function_sizes)


def run_example(
    ruleset: egglog.Schedule | egglog.Ruleset, input: enp.NDArray, egraph: egglog.EGraph | None = None
) -> Report:
    if egraph is None:
        egraph = egglog.EGraph()

    start = time.perf_counter()
    egraph.register(input)
    register_sec = time.perf_counter() - start

    start = time.perf_counter()
    run_report = egraph.run(ruleset)
    run_sec = time.perf_counter() - start

    start = time.perf_counter()
    extracted, cost = egraph.extract(input, include_cost=True)
    extract_sec = time.perf_counter() - start

    return Report(register_sec, run_sec, extract_sec, extracted, cost, egraph.all_function_sizes(), run_report.updated)


@dataclass(frozen=True)
class TotalReport:
    original: Report
    distributed: Report
    factored: list[Report]
    polynomial_multisets: Report
    polynomial_multisets_factored: Report
    polynomial: Report

    @property
    def combined_factored(self) -> Report:
        if not self.factored:
            return self.distributed
        return Report(
            register_sec=self.factored[0].register_sec,
            run_sec=sum(r.run_sec for r in self.factored),
            extract_sec=self.factored[-1].extract_sec,
            extracted=self.factored[-1].extracted,
            cost=self.factored[-1].cost,
            function_sizes=self.factored[-1].function_sizes,
            updated=self.factored[-1].updated,
        )

    @property
    def combined_polynomial(self) -> Report:
        return Report(
            register_sec=self.polynomial_multisets.register_sec,
            run_sec=self.polynomial_multisets.run_sec
            + self.polynomial_multisets_factored.run_sec
            + self.polynomial.run_sec,
            extract_sec=self.polynomial.extract_sec,
            extracted=self.polynomial.extracted,
            cost=self.polynomial.cost,
            function_sizes=self.polynomial.function_sizes,
            updated=self.polynomial.updated,
        )

    def __str__(self) -> str:
        return f"""Costs:
* original: {self.original.cost:,}
* distributed: {self.distributed.cost:,}
* factored: {self.combined_factored.cost:,}
* horner multisets: {self.combined_polynomial.cost:,}


Number of nodes:
* original: {self.original.total_size:,}
* distributed: {self.distributed.total_size:,}
* factored: {self.combined_factored.total_size:,}
* horner multisets: {self.combined_polynomial.total_size:,}

Time:
* original: {self.original.total_sec:.2f}s
* distributed: {self.distributed.total_sec:.2f}s
* factored: {self.combined_factored.total_sec:.2f}s
* horner multisets: {self.combined_polynomial.total_sec:.2f}s
"""


def try_example(
    expr: enp.NDArray,
    *,
    max_factoring_iters: int = 20,
    max_factoring_sec: float = 10.0,
) -> TotalReport:
    original_report = run_example(remove_subtraction, expr)
    print(f"original cost: {original_report.cost:,}")
    distributed_report = run_example(distribute.saturate(), original_report.extracted)
    print(f"distributed cost: {distributed_report.cost:,}")

    egraph = egglog.EGraph()
    polynomial_multisets_report = run_example(
        enp.to_polynomial_ruleset.saturate(), distributed_report.extracted, egraph
    )
    polynomial_multisets_factored_report = run_example(
        enp.factor_ruleset.saturate(), polynomial_multisets_report.extracted, egraph
    )
    polynomial_report = run_example(
        enp.from_polynomial_ruleset.saturate(), polynomial_multisets_factored_report.extracted, egraph
    )
    print(f"polynomial cost: {polynomial_report.cost:,}")

    egraph = egglog.EGraph()
    factored_reports: list[Report] = []
    for i in range(max_factoring_iters):
        res = run_example(factoring, distributed_report.extracted, egraph)
        if not res.updated or res.run_sec > max_factoring_sec:
            break
        print(f"factoring iteration {i}, cost: {res.cost:,}")
        factored_reports.append(res)
    print("Finished\n")

    return TotalReport(
        original_report,
        distributed_report,
        factored_reports,
        polynomial_multisets_report,
        polynomial_multisets_factored_report,
        polynomial_report,
    )


def main() -> None:
    rng = np.random.default_rng(0)
    q = rng.random(12)
    bp = rng.random(4)
    bpp = rng.random(4)

    qm = np.reshape(q, (4, 3)).T
    yip = qm @ bp
    yipp = qm @ bpp
    expected = (np.linalg.norm(np.cross(yip, yipp)) / np.linalg.norm(yip) ** 3) ** 2
    result = bending_function(q, bp, bpp)
    assert np.isclose(result, expected)

    function_bending, gradient_bending = symbolic_bending_examples()

    function_report = try_example(function_bending, max_factoring_iters=0)
    assert function_report.original.cost > 0
    assert function_report.polynomial.cost > 0

    gradient_report = run_example(remove_subtraction, gradient_bending)
    assert gradient_report.cost > 0
    assert gradient_report.total_sec >= 0.0
    assert gradient_report.total_size > 0

    print(function_report)
    print("gradient remove_subtraction cost:", gradient_report.cost)


if __name__ == "__main__":
    main()
