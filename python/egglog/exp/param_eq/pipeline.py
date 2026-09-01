# mypy: disable-error-code="empty-body"

"""Retained paper-era `param_eq` pipeline plus the experimental map variant."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from egglog import *

from .domain import *

MAX_PASSES = 2
HASKELL_INNER_ITERATION_LIMIT = 30
BACKOFF_MATCH_LIMIT = 1000
BACKOFF_BAN_LENGTH = 30


# Keep derived map operations as explicitly typed folds in this research
# module; only map_fold_kv is a backend primitive and public builtin.
# Store discovered constants in a global map so semi-naive analysis can join
# them with polynomial terms. If two singleton updates collide, keep the first
# representative; tolerant float canonicalization is not part of this paused
# research slice.
CONSTS = constant(
    "CONSTS",
    Map[Num, f64],
    merge=lambda left, right: map_fold_kv(
        lambda result, key, value: catch(lambda: result[key]).match(
            lambda old_value: result.insert(key, old_value), result.insert(key, value)
        ),
        left,
        right,
    ),
)

# Map a monomial of the form `{polynomial(P): 1}` to one representative `P`.
# This acts as an index from concrete monomial keys back to nested polynomial
# bodies. Directly matching `polynomial(P) == n` and then constructing
# `{n: 1}` is semantically equivalent, but it can make matching enumerate many
# unrelated polynomial e-classes before proving the singleton monomial exists.
#
# Concrete example:
#   a*polynomial(P) + R -> a*P + R
# should start from monomial keys already present in the outer polynomial, not
# from every `polynomial(P) == n` relation in the e-graph.
POLYNOMIAL_MONOMIALS = constant(
    "POLYNOMIAL_MONOMIALS",
    Map[ContainerMonomial, ContainerPolynomial],
    merge=lambda left, right: map_fold_kv(
        lambda result, key, value: catch(lambda: result[key]).match(
            lambda old_value: result.insert(key, old_value), result.insert(key, value)
        ),
        left,
        right,
    ),
)


@ruleset
def shared_analysis_rules(a: f64) -> Iterable[RewriteOrRule]:
    yield rewrite(exp(Num(a)), subsume=True).to(Num(a.exp()), a.exp().is_finite())
    yield rewrite(log(Num(a)), subsume=True).to(Num(a.log()), a > 0.0, a.log().is_finite())
    yield rule(log(Num(a)), a <= 0.0).then(panic("Log of non-positive number"))
    yield rewrite(abs(Num(a)), subsume=True).to(Num(abs(a)), abs(a).is_finite())


@ruleset
def binary_analysis_rules(x: Num, a: f64, b: f64) -> Iterable[RewriteOrRule]:
    yield rewrite(Num(a) / Num(b), subsume=True).to(Num(a / b), b != f64(0.0), (a / b).is_finite())
    yield rule(x / Num(0.0)).then(panic("Division by zero"))
    yield rewrite(Num(a) - Num(b), subsume=True).to(Num(a - b), (a - b).is_finite())
    yield rewrite(Num(a) * Num(b), subsume=True).to(Num(a * b), (a * b).is_finite())
    yield rewrite(Num(a) + Num(b), subsume=True).to(Num(a + b), (a + b).is_finite())

    yield rewrite(Num(a) ** Num(b), subsume=True).to(Num(a**b), (a**b).is_finite())
    yield rewrite(sqrt(Num(a)), subsume=True).to(Num(a.sqrt()), a >= 0.0, a.sqrt().is_finite())
    yield rule(sqrt(Num(a)), a < 0.0).then(panic("Sqrt of negative number"))

    # cancellations
    yield rewrite(x - x, subsume=True).to(Num(0.0))
    yield rewrite(x / x, subsume=True).to(Num(1.0), x != Num(0.0))

    # multiplicative of inverse
    yield rewrite(x * (1 / x), subsume=True).to(Num(1.0), x != Num(0.0))

    yield rewrite(0 * x, subsume=True).to(Num(0.0))
    yield rewrite(0 / x, subsume=True).to(Num(0.0), x != Num(0.0))


@ruleset
def container_analysis_rules(
    n: Num,
    a: f64,
    poly: ContainerPolynomial,
    poly1: ContainerPolynomial,
    consts: Map[Num, f64],
) -> Iterable[RewriteOrRule]:
    yield rule(n == Num(a)).then(set_(CONSTS).to(Map[Num, f64].empty().insert(n, a)))
    yield rule(polynomial(poly) == n).then(
        set_(POLYNOMIAL_MONOMIALS).to(
            Map[ContainerMonomial, ContainerPolynomial]
            .empty()
            .insert(ContainerMonomial.empty().insert(n, BigRat(1, 1)), poly)
        )
    )
    # Constant fold polynomials so that in each monomial,
    # all constants terms are pulled into a co-efficient
    # and all empty terms are combined. Also drops terms with zero exponents.
    # like: {{}: 3.14, {x: 2}: 2.71}}

    yield rewrite(polynomial(poly), subsume=True).to(
        polynomial(poly1),
        # pull in this so it gets joined in semi-naive
        consts == CONSTS,
        poly1
        == map_fold_kv(
            lambda res_poly, mono, coef: res_poly.insert(
                # split monomial into non constants and constants (which are combined into the coefficient):
                (
                    mono_and_coef := map_fold_kv(
                        lambda res_mono_and_coef, term, exp: catch(lambda: exp != BigRat(0, 1)).match(
                            # if the exponent is not zero, process it
                            lambda _: catch(lambda: consts[term]).match(
                                # if it is a constant, multiply it into the coefficient and drop it from the monomial:
                                lambda v: catch(lambda: exp != BigRat(-1, 1)).match(
                                    lambda _: Pair(
                                        res_mono_and_coef.left,
                                        res_mono_and_coef.right * (v ** exp.to_f64()),
                                    ),
                                    catch(lambda: v != f64(0.0)).match(
                                        lambda _: Pair(
                                            res_mono_and_coef.left,
                                            res_mono_and_coef.right / v,
                                        ),
                                        Pair(
                                            res_mono_and_coef.left.insert(term, exp),
                                            res_mono_and_coef.right,
                                        ),
                                    ),
                                ),
                                # if it is not a constant, keep it in the monomial
                                Pair(
                                    res_mono_and_coef.left.insert(term, exp),
                                    res_mono_and_coef.right,
                                ),
                            ),
                            # if the exponent is zero, the term is just 1 and can be dropped from the monomial, so keep the monomial as is
                            res_mono_and_coef,
                        ),
                        Pair(ContainerMonomial.empty(), coef),
                        mono,
                    )
                ).left,
                mono_and_coef.right + catch(lambda: res_poly[mono_and_coef.left]).unwrap_or(f64(0.0)),
            ),
            ContainerPolynomial.empty(),
            poly,
        ),
        poly != poly1,
    )

    # Turn polynomials that are actually just constant factors into constants, so they can be used in more rewrites.
    yield rewrite(polynomial(poly), subsume=True).to(
        Num(poly[ContainerMonomial.empty()]),
        poly.length() == i64(1),
        # The only key is an empty monomial, so the polynomial is just a constant term:
        ContainerMonomial.empty()
        == map_fold_kv(
            lambda picked, key, _value: picked.match(lambda _: picked, Maybe[ContainerMonomial].some(key)),
            Maybe[ContainerMonomial].none(),
            poly,
        ).unwrap(),
    )

    # remove monomials with zero coefficients
    yield rewrite(polynomial(poly), subsume=True).to(
        polynomial(poly1),
        poly1
        == map_fold_kv(
            lambda result, key, value: catch(lambda: value != f64(0.0)).match(
                lambda _: result.insert(key, value), result
            ),
            ContainerPolynomial.empty(),
            poly,
        ),
        poly != poly1,
    )


@ruleset
def binary_basic_rules(x: Num, y: Num, z: Num, af: f64, bf: f64, cf: f64, df: f64) -> Iterable[RewriteOrRule]:
    a = Num(af)
    b = Num(bf)
    c = Num(cf)
    d = Num(df)

    # commutativity
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)

    # associativity
    yield rewrite(x + (y + z)).to((x + y) + z)  # no-op
    yield rewrite(x * (y * z)).to((x * y) * z)  # no-op
    yield rewrite(x * (y / z)).to((x * y) / z)  # no-op
    yield rewrite((x * y) / z).to(x * (y / z))  # no-op
    yield rewrite((a * x) * (b * y)).to((a * b) * (x * y))  # no-op
    yield rewrite(a * x + b).to(a * (x + b / a))  # no-op
    yield rewrite(a * x - b).to(a * (x - b / a))  # no-op
    yield rewrite(b - (a * x)).to(a * ((b / a) - x))  # no-op
    yield rewrite(a * x + b * y).to(
        a * (x + (b / a) * y)
    )  # factoring out one constant from one term, and dividing the others who have constant terms to compensate
    yield rewrite(a * x - b * y).to(a * (x - (b / a) * y))  # same as above
    yield rewrite(a * x + b / y).to(a * (x + (b / a) / y))  # same as above
    yield rewrite(a * x - b / y).to(a * (x - (b / a) / y))  # same as above

    yield rewrite(a / (b * x)).to((a / b) / x)  # no-op
    yield rewrite(x / (b * y)).to((1 / b) * x / y)  # no-op
    yield rewrite(x / a + b).to((x + b * a) / a)  # same as above
    yield rewrite(x / a - b).to((x - b * a) / a)  # same as above
    yield rewrite(b - x / a).to(((b * a) - x) / a)  # same as above
    yield rewrite(x / a + b * y).to((x + (b * a) * y) / a)  # same as above
    yield rewrite(x / a - b * y).to((x - (b * a) * y) / a)  # same as above
    yield rewrite((b + a * x) / (c + d * y)).to((a / d) * (b / a + x) / (c / d + y))
    yield rewrite((b + x) / (c + d * y)).to((1 / d) * (b + x) / (c / d + y))

    # identities
    yield rewrite(0 + x).to(x)
    yield rewrite(x - 0).to(x)
    yield rewrite(1 * x).to(x)

    # distributive and factorization
    yield rewrite((x * y) + (x * z)).to(x * (y + z))
    yield rewrite(x - (y + z)).to((x - y) - z)
    yield rewrite(x - (y - z)).to((x - y) + z)
    yield rewrite(-(x + y)).to(-x - y)
    yield rewrite(x - a).to(x + -a)
    yield rewrite(x - (a * y)).to(x + -a * y)
    yield rewrite((1 / x) * (1 / y)).to(1 / (x * y))

    # negate
    yield rewrite(x - -y).to(x + y)
    yield rewrite(x + -y).to(x - y)
    yield rewrite(0 - x).to(-x)


@ruleset
def container_basic_rules(
    poly: ContainerPolynomial,
    poly1: ContainerPolynomial,
    poly2: ContainerPolynomial,
    nonconst_poly: ContainerPolynomial,
    coef: f64,
    polynomial_monomials: Map[ContainerMonomial, ContainerPolynomial],
    counts: MultiSet[Num],
    n: Num,
    poly_pair: Pair[ContainerPolynomial, ContainerPolynomial],
    exp: BigRat,
    mono: ContainerMonomial,
) -> Iterable[RewriteOrRule]:
    # Factor one representative non-unit coefficient from a small polynomial:
    #
    # a*x + b -> a*(x + b/a)
    yield rewrite(polynomial(poly)).to(
        polynomial(
            ContainerPolynomial.empty().insert(ContainerMonomial.empty().insert(polynomial(poly1), BigRat(1, 1)), coef)
        ),
        poly.length() > i64(1),
        poly.length() <= i64(4),
        nonconst_poly
        == map_fold_kv(
            lambda result, key, value: catch(lambda: key != ContainerMonomial.empty()).match(
                lambda _: result.insert(key, value), result
            ),
            ContainerPolynomial.empty(),
            poly,
        ),
        poly2
        == map_fold_kv(
            lambda result, key, value: catch(lambda: value != f64(1.0)).match(
                lambda _: result.insert(key, value), result
            ),
            ContainerPolynomial.empty(),
            nonconst_poly,
        ),
        coef
        == poly2[
            map_fold_kv(
                lambda picked, key, _value: picked.match(lambda _: picked, Maybe[ContainerMonomial].some(key)),
                Maybe[ContainerMonomial].none(),
                poly2,
            ).unwrap()
        ],
        poly2.length() == nonconst_poly.length(),
        poly1
        == map_fold_kv(
            lambda result, key, value: result.insert(key, value / coef),
            ContainerPolynomial.empty(),
            poly,
        ),
    )

    # Greedy multivariate Horner factorization for rational exponents. Choose
    # the term present in the most monomials, then factor out its minimum
    # exponent:
    #
    # x*y + x*z -> x*(y + z)
    yield rewrite(polynomial(poly)).to(
        polynomial(
            poly_pair.right.insert(
                ContainerMonomial.empty().insert(n, exp).insert(polynomial(poly_pair.left), BigRat(1, 1)),
                f64(1.0),
            )
        ),
        counts
        == map_fold_kv(
            lambda counts, mono, _coef: map_fold_kv(
                lambda updated_counts, term, _exp: updated_counts.insert(term),
                counts,
                mono,
            ),
            MultiSet[Num](),
            poly,
        ),
        n == counts.pick_max(),
        counts.count(n) > i64(1),
        exp
        == map_fold_kv(
            lambda min_exp, mono, _coef: catch(lambda: mono[n] < min_exp).match(lambda _: mono[n], min_exp),
            BigRat(2**63 - 1, 1),
            poly,
        ),
        poly_pair
        == map_fold_kv(
            lambda divided_and_remainder, mono, coef: catch(lambda: mono[n]).match(
                lambda current_exp: Pair(
                    divided_and_remainder.left.insert(mono.insert(n, current_exp - exp), coef),
                    divided_and_remainder.right,
                ),
                Pair(
                    divided_and_remainder.left,
                    divided_and_remainder.right.insert(mono, coef),
                ),
            ),
            Pair(ContainerPolynomial.empty(), ContainerPolynomial.empty()),
            poly,
        ),
    )

    # Flatten an exact nested polynomial term inside a larger polynomial:
    #
    # a*polynomial(P) + R -> a*P + R
    #
    # The whole monomial must be the nested polynomial at exponent one, which
    # avoids distributing arbitrary products.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            map_fold_kv(
                lambda result, key, value: catch(lambda: result[key]).match(
                    lambda old_value: result.insert(key, old_value + value), result.insert(key, value)
                ),
                poly.remove(mono),
                map_fold_kv(
                    lambda result, nested_mono, nested_coef: result.insert(nested_mono, nested_coef * poly[mono]),
                    ContainerPolynomial.empty(),
                    poly1,
                ),
            )
        ),
        polynomial_monomials == POLYNOMIAL_MONOMIALS,
        poly.length() > i64(1),
        mono
        == map_fold_kv(
            lambda selected, candidate_mono, _candidate_coef: catch(lambda: polynomial_monomials[candidate_mono]).match(
                lambda _nested_poly: candidate_mono,
                selected,
            ),
            ContainerMonomial.empty(),
            poly,
        ),
        mono.length() > i64(0),
        poly[mono] != f64(0.0),
        poly1 == polynomial_monomials[mono],
        poly1.length() > i64(1),
    )


@ruleset
def shared_fun_rules(x: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(log(exp(x))).to(x)
    yield rewrite(log(abs(exp(x)))).to(x)


@ruleset
def binary_fun_rules(x: Num, y: Num, af: f64) -> Iterable[RewriteOrRule]:
    a = Num(af)

    yield rewrite(log(a * y)).to(log(a) + log(y), af > 0.0, y != Num(0.0))
    yield rewrite(log(y * a)).to(log(y) + log(a), af > 0.0, y != Num(0.0))
    yield rewrite(log(a / y)).to(log(a) - log(y), af > 0.0, y != Num(0.0))
    yield rewrite(log(y / a)).to(log(y) - log(a), af > 0.0, y != Num(0.0))

    yield rewrite(log(a**y)).to(y * log(a), af > 0.0)
    yield rewrite(log(sqrt(x))).to(0.5 * log(x))
    yield rewrite(x**0.5).to(sqrt(x))


@ruleset
def container_fun_rules(poly: ContainerPolynomial, m: ContainerMonomial, term: Num) -> Iterable[RewriteOrRule]:
    # Preserve the one-factor identity needed by the public log(2*x0) case.
    # Expanding multiple factors or an even power would be unsound when
    # negative factors combine into a positive product.
    yield rewrite(log(polynomial(poly))).to(
        polynomial(
            map_fold_kv(
                lambda res_poly, term, exp: map_fold_kv(
                    lambda result, mono, coef: catch(lambda: result[mono]).match(
                        lambda old_coef: result.insert(mono, old_coef + coef), result.insert(mono, coef)
                    ),
                    res_poly,
                    ContainerPolynomial.empty().insert(
                        ContainerMonomial.empty().insert(log(term), BigRat(1, 1)),
                        exp.to_f64(),
                    ),
                ),
                ContainerPolynomial.empty().insert(ContainerMonomial.empty(), poly[m].log()),
                m,
            )
        ),
        poly.length() == i64(1),
        m
        == map_fold_kv(
            lambda picked, key, _value: picked.match(lambda _: picked, Maybe[ContainerMonomial].some(key)),
            Maybe[ContainerMonomial].none(),
            poly,
        ).unwrap(),
        poly[m] > f64(0.0),
        m.length() == i64(1),
        term
        == map_fold_kv(
            lambda picked, key, _value: picked.match(lambda _: picked, Maybe[Num].some(key)),
            Maybe[Num].none(),
            m,
        ).unwrap(),
        m[term] == BigRat(1, 1),
    )


@dataclass(frozen=True)
class PaperPipelineReport:
    """Bounded result; `saturated` describes inner schedules, not an outer fixed point."""

    status: Literal["saturated", "iteration_limit"]
    passes: int
    total_sec: float
    total_size: int
    before_nodes: int
    before_params: int
    extracted: str
    extracted_nodes: int
    extracted_params: int


# Reporting and pipeline loop


binary_analysis_ruleset = shared_analysis_rules | binary_analysis_rules
container_analysis_ruleset = shared_analysis_rules | container_analysis_rules
binary_analysis_schedule = binary_analysis_ruleset.saturate()
containers_analysis_schedule = container_analysis_ruleset.saturate()
shared_rewrite_ruleset = shared_fun_rules
binary_rewrite_ruleset = shared_rewrite_ruleset | binary_basic_rules | binary_fun_rules
container_rewrite_ruleset = shared_rewrite_ruleset | container_basic_rules | container_fun_rules


def _graph_size(egraph: EGraph) -> int:
    return sum(size for _, size in egraph.all_function_sizes())


binary_schedule = run(
    binary_rewrite_ruleset,
    scheduler=back_off(
        match_limit=BACKOFF_MATCH_LIMIT,
        ban_length=BACKOFF_BAN_LENGTH,
    ).persistent(),
)
container_schedule = run(
    container_rewrite_ruleset,
    scheduler=back_off(
        # This lower container budget preserves corpus parameter parity while
        # avoiding synthetic appended-expression slowdowns from over-searching
        # equivalent container factorizations.
        match_limit=5,
        ban_length=10,
    ).persistent(),
)


def _run_single_pass(
    egraph: EGraph,
    num: Num,
    cost_model: CostModel[ParamCost],
    analysis_schedule: Schedule,
    schedule: Schedule,
) -> tuple[Num, ParamCost, int, bool]:
    """
    Run one `rewriteTree`-like pass and return the populated e-graph.

    This mirrors Haskell at the control-flow level while using the ordinary
    persistent backoff scheduler available upstream:
    - up to 30 inner rewrite rounds
    - one saturated analysis round after each rewrite round
    - stop when both schedules report no changes or deferred work
    """
    n = egraph.let("n", num)
    current_size = _graph_size(egraph)
    saturated = False
    for _ in range(HASKELL_INNER_ITERATION_LIMIT):
        analysis_report = egraph.run(analysis_schedule)
        rewrite_report = egraph.run(schedule)
        current_size = _graph_size(egraph)
        if analysis_report.can_stop and rewrite_report.can_stop:
            saturated = True
            break
    extracted, cost = egraph.extract(n, include_cost=True, cost_model=cost_model)
    return extracted, cost, current_size, saturated


def run_paper_pipeline(
    initial: Num,
    decode: Callable[[Num], Num] = lambda x: x,
    cost_model: CostModel[ParamCost] = param_cost_model,
    schedule: Schedule = binary_schedule,
    analysis_schedule: Schedule = binary_analysis_schedule,
) -> PaperPipelineReport:
    current, before_cost = EGraph(save_egglog_string=False).extract(initial, include_cost=True, cost_model=cost_model)
    # get schedule decls so that it's pre-cached
    schedule.__egg_decls__
    analysis_schedule.__egg_decls__
    start = time.perf_counter()
    # Add constants to the egraph so that they can be used in rules without needing to be registered each pass
    egraph = EGraph(Num(0.0), save_egglog_string=False)
    # pre-run rulesets so that we don't have to register them each pipeline pass
    egraph.run(analysis_schedule)
    last_cost = before_cost
    max_size = 0
    passes = 0
    status: Literal["saturated", "iteration_limit"] = "saturated"
    for pass_index in range(1, MAX_PASSES + 1):
        with egraph:
            extracted, last_cost, total_size, saturated = _run_single_pass(
                egraph,
                current,
                cost_model=cost_model,
                schedule=schedule,
                analysis_schedule=analysis_schedule,
            )
        max_size = max(max_size, total_size)
        passes = pass_index
        unchanged = extracted == current
        current = extracted
        if not saturated:
            status = "iteration_limit"
            break
        if unchanged:
            break
    return PaperPipelineReport(
        status=status,
        passes=passes,
        total_sec=time.perf_counter() - start,
        total_size=max_size,
        before_nodes=before_cost.node_count,
        before_params=before_cost.floats,
        extracted_nodes=last_cost.node_count,
        extracted_params=last_cost.floats,
        extracted=render_num(decode(current)),
    )


def run_paper_pipeline_container(initial: Num) -> PaperPipelineReport:
    try:
        return run_paper_pipeline(
            initial,
            decode=containers_to_binary,
            cost_model=container_cost_model,
            schedule=container_schedule,
            analysis_schedule=containers_analysis_schedule,
        )
    except ValueError as error:
        if "non-finite" not in str(error):
            raise
        msg = "The Param-Eq container pipeline requires every coefficient normalization to remain finite"
        raise ValueError(msg) from error
