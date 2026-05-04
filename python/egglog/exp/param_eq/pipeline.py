# mypy: disable-error-code="empty-body"

"""Retained paper-era `param_eq` pipeline plus the experimental map variant."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from egglog import *

from .domain import *

MAX_PASSES = 2
HASKELL_INNER_ITERATION_LIMIT = 30
BACKOFF_MATCH_LIMIT = 1000
BACKOFF_BAN_LENGTH = 30
CONST_MERGE_TOLERANCE = 1e-6
SOLE_MONOMIAL_GENERIC_TERM_PENALTY = 1000

T = TypeVar("T", bound=BaseExpr)
V = TypeVar("V", bound=BaseExpr)


# Active invariants live here; larger parked design notes are in
# `PIPELINE_DESIGN_NOTES.md`. The live baseline analysis path now uses a
# partial `f64` lookup, and `bigrat_pow_const_value` stays a partial helper for
# antecedent-style use.

# Store constants in a global mapping so they work with semi-naive
CONSTS = constant(
    "CONSTS",
    Map[Num, f64],
    merge=partial(
        map_merge_with,
        lambda old_value, new_value: collapse_floats_with_tol(old_value, new_value, CONST_MERGE_TOLERANCE),
    ),
)

def _choose_lower_scored_monomial(
    old_value: Pair[i64, ContainerMonomial], new_value: Pair[i64, ContainerMonomial]
) -> Pair[i64, ContainerMonomial]:
    return catch(lambda: old_value.left <= new_value.left).match(lambda _: old_value, new_value)


# Store a mapping from e-classes that are polynomials with a sole monomial
# to one representative sole monomial, paired with a local float-count score.
# Used for folding nested polynomials like turning (xy)^2 into x^2 * y^2, or polynomial({ M: c })^z into c^z * M^z.
SOLE_MONOMIALS = constant(
    "SOLE_MONOMIALS",
    Map[Num, Pair[i64, ContainerMonomial]],
    merge=partial(map_merge_with, _choose_lower_scored_monomial),
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
    merge=partial(map_merge_with, lambda old_value, new_value: old_value),
)

# Map polynomial e-classes to their current representative bodies. This is a
# join-shaping cache for rules that first discover candidate `Num` terms from
# monomial keys and then need the corresponding polynomial body.
#
# Concrete example:
#   P + c*M*P + R -> P * (1 + c*M) + R
# scans terms already present in `P + c*M*P + R`. Looking up those terms in
# this map keeps the join keyed by present terms; directly matching
# `polynomial(body) == n` made the same rule much slower on repeated-product
# log examples.
POLYNOMIAL_BODIES = constant(
    "POLYNOMIAL_BODIES",
    Map[Num, ContainerPolynomial],
    merge=partial(map_merge_with, lambda old_value, new_value: new_value),
)

def if_defined(cond: Unit, then: T, otherwise: T) -> T:
    return catch(lambda: cond).match(lambda _: then, otherwise)


def try_match(expr: T, on_some: Callable[[T], V], default: V) -> V:
    return catch(lambda: expr).match(on_some, default)


def try_or(expr: T, default: T) -> T:
    return catch(lambda: expr).unwrap_or(default)


def f64_param_score(value: f64) -> i64:
    return if_defined(value != f64.from_i64(value.to_i64()), i64(1), i64(0))


def bigrat_param_score(value: BigRat) -> i64:
    return if_defined(value != BigRat(value.to_i64(), 1), i64(1), i64(0))


def shallow_polynomial_param_score(poly: ContainerPolynomial) -> i64:
    return map_fold_kv(
        lambda score, mono, coef: score
        + f64_param_score(coef)
        + map_fold_kv(lambda mono_score, _term, exp: mono_score + bigrat_param_score(exp), i64(0), mono),
        i64(0),
        poly,
    )


def generic_sole_monomial_score(poly: ContainerPolynomial) -> i64:
    # Generic aliases cannot see through polynomial-valued terms, so give each
    # term a large penalty. A more specific one-level polynomial alias, when it
    # exists, should beat this fallback using its decoded coefficient score.
    return map_fold_kv(
        lambda score, mono, coef: score
        + f64_param_score(coef)
        + map_fold_kv(
            lambda mono_score, _term, exp: mono_score
            + bigrat_param_score(exp)
            + i64(SOLE_MONOMIAL_GENERIC_TERM_PENALTY),
            i64(0),
            mono,
        ),
        i64(0),
        poly,
    )


@ruleset
def shared_analysis_rules(a: f64, b: f64) -> Iterable[RewriteOrRule]:
    yield rewrite(exp(Num(a)), subsume=True).to(Num(a.exp()))
    yield rewrite(log(Num(a)), subsume=True).to(Num(a.log()), a > 0.0)
    yield rule(log(Num(a)), a <= 0.0).then(panic("Log of non-positive number"))
    yield rewrite(abs(Num(a)), subsume=True).to(Num(abs(a)))
    # merge constants that are close to each other,
    # yield rule(Num(a), Num(b), a != b, abs(a - b) <= f64(CONST_MERGE_TOLERANCE)).then(union(Num(a)).with_(Num(b)))


@ruleset
def binary_analysis_rules(x: Num, a: f64, b: f64) -> Iterable[RewriteOrRule]:
    yield rewrite(Num(a) / Num(b), subsume=True).to(Num(a / b), b != f64(0.0))
    yield rule(x / Num(0.0)).then(panic("Division by zero"))
    yield rewrite(Num(a) - Num(b), subsume=True).to(Num(a - b))
    yield rewrite(Num(a) * Num(b), subsume=True).to(Num(a * b))
    yield rewrite(Num(a) + Num(b), subsume=True).to(Num(a + b))

    yield rewrite(Num(a) ** Num(b), subsume=True).to(Num(a**b))
    yield rewrite(sqrt(Num(a)), subsume=True).to(Num(a.sqrt()), a >= 0.0)
    yield rule(sqrt(Num(a)), a < 0.0).then(panic("Sqrt of negative number"))

    # cancellations
    yield rewrite(x - x, subsume=True).to(Num(0.0))
    yield rewrite(x / x, subsume=True).to(Num(1.0), x != Num(0.0))

    # multiplicative of inverse
    yield rewrite(x * (1 / x), subsume=True).to(Num(1.0))

    yield rewrite(0 * x, subsume=True).to(Num(0.0))
    yield rewrite(0 / x, subsume=True).to(Num(0.0))


@ruleset
def container_analysis_rules(
    n: Num,
    a: f64,
    coef: f64,
    poly: ContainerPolynomial,
    poly1: ContainerPolynomial,
    consts: Map[Num, f64],
    key: ContainerMonomial,
    mono: ContainerMonomial,
    score: i64,
    term: Num,
) -> Iterable[RewriteOrRule]:
    # default values
    yield rule().then(
        set_(CONSTS).to(Map[Num, f64].empty()),
        set_(SOLE_MONOMIALS).to(Map[Num, Pair[i64, ContainerMonomial]].empty()),
        set_(POLYNOMIAL_MONOMIALS).to(Map[ContainerMonomial, ContainerPolynomial].empty()),
        set_(POLYNOMIAL_BODIES).to(Map[Num, ContainerPolynomial].empty()),
    )
    yield rule(n == Num(a)).then(set_(CONSTS).to(Map[Num, f64].empty().insert(n, a)))
    yield rule(
        polynomial(poly) == n,
        poly.length() == i64(1),
        key == poly.pick_key(),
        key.not_contains(n),
        score == generic_sole_monomial_score(poly),
    ).then(
        set_(SOLE_MONOMIALS).to(
            Map[Num, Pair[i64, ContainerMonomial]]
            .empty()
            # add constant term into monomial
            .insert(n, Pair(score, key.insert(Num(poly[key]), BigRat(1, 1))))
        )
    )
    # When the sole monomial is itself a polynomial, score the decoded one-level
    # shape. This chooses aliases like `a * polynomial(P)` based on the floats in
    # `P`, instead of whichever scalar factor happened to be discovered first.
    yield rule(
        polynomial(poly) == n,
        poly.length() == i64(1),
        key == poly.pick_key(),
        key.not_contains(n),
        key.length() == i64(1),
        term == key.pick_key(),
        key[term] == BigRat(1, 1),
        polynomial(poly1) == term,
        score == f64_param_score(poly[key]) + shallow_polynomial_param_score(poly1),
    ).then(
        set_(SOLE_MONOMIALS).to(
            Map[Num, Pair[i64, ContainerMonomial]]
            .empty()
            # add constant term into monomial
            .insert(n, Pair(score, key.insert(Num(poly[key]), BigRat(1, 1))))
        )
    )
    yield rule(polynomial(poly) == n).then(
        set_(POLYNOMIAL_MONOMIALS).to(
            Map[ContainerMonomial, ContainerPolynomial]
            .empty()
            .insert(ContainerMonomial.empty().insert(n, BigRat(1, 1)), poly)
        )
    )
    yield rule(polynomial(poly) == n).then(
        set_(POLYNOMIAL_BODIES).to(Map[Num, ContainerPolynomial].empty().insert(n, poly))
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
            lambda res_poly, mono, coef: (
                # split monomial into non constants and constants (which are combined into the coefficient):
                map_fold_kv(
                    lambda res_mono_and_coef, term, exp: if_defined(
                        exp != BigRat(0, 1),
                        # if the exponent is not zero, process it
                        try_match(
                            consts[term],
                            # if it is a constant, multiply it into the coefficient and drop it from the monomial:
                            lambda v: if_defined(
                                exp != BigRat(-1, 1),
                                res_mono_and_coef.map_right(lambda prev_coef: prev_coef * v.pow_bigrat(exp)),
                                if_defined(
                                    v != f64(0.0),
                                    res_mono_and_coef.map_right(lambda prev_coef: prev_coef / v),
                                    res_mono_and_coef.map_left(lambda mono: mono.insert(term, exp)),
                                ),
                            ),
                            # if it is not a constant, keep it in the monomial
                            res_mono_and_coef.map_left(lambda mono: mono.insert(term, exp)),
                        ),
                        # if the exponent is zero, the term is just 1 and can be dropped from the monomial, so keep the monomial as is
                        res_mono_and_coef,
                    ),
                    Pair(ContainerMonomial.empty(), coef),
                    mono,
                ).match(lambda mono, coef: res_poly.insert(mono, coef + try_or(res_poly[mono], f64(0.0))))
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
        ContainerMonomial.empty() == poly.pick_key(),
    )

    # remove monomials with zero coefficients
    yield rewrite(polynomial(poly), subsume=True).to(
        polynomial(poly1),
        poly1 == map_drop_zero_values(poly),
        poly != poly1,
    )
    # zero polynomial is zero
    yield rewrite(polynomial(ContainerPolynomial.empty()), subsume=True).to(Num(0.0))

    # Turn polynomials with only one monomial that is a single variable, with 1.0 coefficient and exponent into just that variable
    yield rewrite(polynomial(poly), subsume=True).to(
        n,
        poly.length() == i64(1),
        key == poly.pick_key(),
        poly[key] == f64(1.0),
        key.length() == i64(1),
        n == key.pick_key(),
        key[n] == BigRat(1, 1),
    )

    # # union polynomials that are close to each other in coefficients, subsuming one of them as well
    # yield rule(
    #     polynomial(poly),
    #     polynomial(poly1),
    #     poly != poly1,
    #     poly1.keys() == poly.keys(),
    #     # try merging them to see if they are close. If they aren't, this won't match
    #     map_merge_with(lambda x, y: collapse_floats_with_tol(x, y, CONST_MERGE_TOLERANCE), poly, poly1),
    # ).then(
    #     union(polynomial(poly)).with_(polynomial(poly1)),
    #     subsume(polynomial(poly1)),
    # )


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
    # yield rewrite(x / a + y / b).to((x + y / (b * a)) / a) unsound
    yield rewrite(x / a - b * y).to((x - (b * a) * y) / a)  # same as above
    # yield rewrite(x / a - b / y).to((x - y / (b * a)) / a, bf != f64(0.0)) unsound
    yield rewrite((b + a * x) / (c + d * y)).to((a / d) * (b / a + x) / (c / d + y))
    yield rewrite((b + x) / (c + d * y)).to((1 / d) * (b + x) / (c / d + y))

    # identities
    yield rewrite(0 + x).to(x)
    yield rewrite(x - 0).to(x)
    yield rewrite(1 * x).to(x)
    # yield rewrite(0 * x).to(Num(0.0))
    # yield rewrite(0 / x).to(Num(0.0))

    # # cancellations
    # yield rewrite(x - x).to(Num(0.0))
    # yield rewrite(x / x).to(Num(1), x != Num(0.0))

    # distributive and factorization
    yield rewrite((x * y) + (x * z)).to(x * (y + z))
    yield rewrite(x - (y + z)).to((x - y) - z)
    yield rewrite(x - (y - z)).to((x - y) + z)
    yield rewrite(-(x + y)).to(-x - y)
    yield rewrite(x - a).to(x + -a)
    yield rewrite(x - (a * y)).to(x + -a * y)
    yield rewrite((1 / x) * (1 / y)).to(1 / (x * y))

    # # multiplicative of inverse
    # yield rewrite(x * (1 / x)).to(Num(1), x != Num(0.0))

    # negate
    yield rewrite(x - -y).to(x + y)
    yield rewrite(x + -y).to(x - y)
    yield rewrite(0 - x).to(-x)


@ruleset
def container_basic_rules(
    poly: ContainerPolynomial,
    poly1: ContainerPolynomial,
    poly2: ContainerPolynomial,
    poly3: ContainerPolynomial,
    residual_poly: ContainerPolynomial,
    nonconst_poly: ContainerPolynomial,
    coef: f64,
    inner_coef: f64,
    coef_counts: MultiSet[f64],
    sole_monomials: Map[Num, Pair[i64, ContainerMonomial]],
    polynomial_monomials: Map[ContainerMonomial, ContainerPolynomial],
    polynomial_bodies: Map[Num, ContainerPolynomial],
    matching_polynomial_bodies: Map[Num, ContainerPolynomial],
    counts: MultiSet[Num],
    n: Num,
    poly_pair: Pair[ContainerPolynomial, ContainerPolynomial],
    polynomial_factor: Pair[Num, ContainerPolynomial],
    monomial_factor: Pair[ContainerMonomial, ContainerPolynomial],
    exp: BigRat,
    mono: ContainerMonomial,
) -> Iterable[RewriteOrRule]:
    # If a non-one coefficient exists in the polynomial, factor out one
    # representative coefficient. In a binary form this covers things like:
    #
    # a * x + b  -> a * (x + b / a)
    #
    # P = sum_i c_i * M_i
    # P -> s * polynomial(sum_i (c_i / s) * M_i)
    yield rewrite(polynomial(poly)).to(
        polynomial(
            ContainerPolynomial.empty().insert(ContainerMonomial.empty().insert(polynomial(poly1), BigRat(1, 1)), coef)
        ),
        # only apply to polynomials with more than one monomial
        poly.length() > i64(1),
        # filter to monomials with non one coefficients and with keys that are non empty (so we don't pull out constant factors)
        nonconst_poly == map_filter_kv(lambda k, v: k != ContainerMonomial.empty(), poly),
        poly2 == map_filter_kv(lambda k, v: v != f64(1.0), nonconst_poly),
        # Pick one representative coefficient, preserving the older rule's
        # reachability for binary-style target expressions.
        coef == poly2[poly2.pick_key()],
        # If there is no non-constant term with coefficient 1, the polynomial is not already scaled.
        poly2.length() == nonconst_poly.length(),  # divide the terms by that coefficient
        poly1 == map_map_values(lambda _, c: c / coef, poly),
    )

    # Add one alternate factor choice when a coefficient exposes integer ratios
    # with other terms. This is the minimal container analogue of binary
    # pairwise coefficient factoring needed for cases like
    # `0.02889 * x - 0.00963 * z -> -0.00963 * (-3 * x + z)`.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            ContainerPolynomial.empty().insert(ContainerMonomial.empty().insert(polynomial(poly1), BigRat(1, 1)), coef)
        ),
        poly.length() > i64(1),
        nonconst_poly == map_filter_kv(lambda k, v: k != ContainerMonomial.empty(), poly),
        poly2 == map_filter_kv(lambda k, v: v != f64(1.0), nonconst_poly),
        coef_counts
        == map_fold_kv(
            lambda counts, _candidate_mono, candidate_coef: if_defined(
                candidate_coef != f64(0.0),
                map_fold_kv(
                    lambda counts, _other_mono, other_coef: if_defined(
                        (other_coef / candidate_coef) != f64.from_i64((other_coef / candidate_coef).to_i64()),
                        counts,
                        counts.insert(candidate_coef),
                    ),
                    counts,
                    poly2,
                ),
                counts,
            ),
            MultiSet[f64](),
            poly2,
        ),
        coef == coef_counts.pick_max(),
        coef_counts.count(coef) > i64(1),
        # If there is no non-constant term with coefficient 1, the polynomial is not already scaled.
        poly2.length() == nonconst_poly.length(),  # divide the terms by that coefficient
        poly1 == map_map_values(lambda _, c: c / coef, poly),
    )

    # Add one factor choice when dividing by a coefficient exposes a later
    # integer-residual split. This keeps the outer scale needed for cases where
    # `b/a` is not itself an integer, but `b/a - k` matches another divided
    # coefficient and `k` is free.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            ContainerPolynomial.empty().insert(ContainerMonomial.empty().insert(polynomial(poly1), BigRat(1, 1)), coef)
        ),
        poly.length() > i64(1),
        poly.length() <= i64(6),
        nonconst_poly == map_filter_kv(lambda k, _v: k != ContainerMonomial.empty(), poly),
        poly2 == map_filter_kv(lambda k, v: v != f64(1.0), nonconst_poly),
        coef == map_factor_coef_for_integer_residual_split(poly),
        poly2.length() == nonconst_poly.length(),
        poly1 == map_map_values(lambda _, c: c / coef, poly),
    )

    # Recover binary-style factoring after container lowering has already
    # merged like monomials:
    #
    # a*M + b*N + R -> a*(M + N) + (b - a)*N + R
    #
    # This handles `-0.3*x0 + 0.7*x1 -> -0.3*(x0 + x1) + x1`
    # without preserving the pre-normalized `-0.3*x1 + x1` shape.
    yield rewrite(polynomial(poly)).to(
        polynomial(poly3.insert(ContainerMonomial.empty().insert(polynomial(poly1), BigRat(1, 1)), coef)),
        poly.length() <= i64(6),
        monomial_factor == map_integer_residual_split_candidate(poly),
        mono == monomial_factor.left,
        poly1 == monomial_factor.right,
        polynomial_bodies == POLYNOMIAL_BODIES,
        counts
        == map_fold_kv(lambda counts, candidate_mono, _coef: counts + candidate_mono.keys(), MultiSet[Num](), poly),
        map_restrict_keys(counts, polynomial_bodies).length() == i64(0),
        coef == poly[mono],
        residual_poly == map_drop_zero_values(map_intersect_with(lambda original, _one: original - coef, poly, poly1)),
        poly2 == map_remove_keys(map_keys(poly1), poly),
        poly3 == map_merge_with(lambda left, right: left + right, poly2, residual_poly),
    )

    # Factor a repeated scalar coefficient magnitude from a subset while
    # preserving the remaining terms:
    #
    # c*M1 + c*M2 + R -> c*(M1 + M2) + R
    # c*M1 - c*M2 + R -> c*(M1 - M2) + R
    #
    # Binary sees this through `(x * y) + (x * z) -> x * (y + z)`. In the
    # container form numeric factors live as polynomial coefficients, so the
    # ordinary monomial-key Horner rule cannot see them.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            poly_pair.right.insert(
                ContainerMonomial.empty().insert(polynomial(poly_pair.left), BigRat(1, 1)),
                coef,
            )
        ),
        nonconst_poly == map_filter_kv(lambda k, _v: k != ContainerMonomial.empty(), poly),
        coef_counts
        == map_fold_kv(
            lambda counts, _mono, candidate_coef: if_defined(
                abs(candidate_coef) != f64(1.0),
                counts.insert(abs(candidate_coef)),
                counts,
            ),
            MultiSet[f64](),
            nonconst_poly,
        ),
        coef == coef_counts.pick_max(),
        coef != f64(0.0),
        coef_counts.count(coef) > i64(1),
        poly_pair
        == map_fold_kv(
            lambda selected_and_remainder, mono, candidate_coef: if_defined(
                abs(candidate_coef) != coef,
                selected_and_remainder.map_right(lambda remainder: remainder.insert(mono, candidate_coef)),
                selected_and_remainder.map_left(lambda selected: selected.insert(mono, candidate_coef / coef)),
            ),
            Pair(ContainerPolynomial.empty(), ContainerPolynomial.empty()),
            poly,
        ),
        poly_pair.left.length() > i64(1),
        poly_pair.right.length() > i64(0),
    )

    # Factor a constant term and one equal-and-opposite non-constant term out
    # of a larger polynomial:
    #
    # c - c*M + R -> c * (1 - M) + R
    #
    # `map_filter_kv` cannot express equality because its callback must return
    # `Unit`, not a `Fact`, so equality is encoded as failed inequality.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            poly2.remove(mono).insert(
                ContainerMonomial.empty().insert(
                    polynomial(
                        ContainerPolynomial.empty().insert(ContainerMonomial.empty(), f64(1.0)).insert(mono, f64(-1.0))
                    ),
                    BigRat(1, 1),
                ),
                coef,
            )
        ),
        poly.length() > i64(2),
        poly.contains(ContainerMonomial.empty()),
        coef == poly[ContainerMonomial.empty()],
        coef != f64(0.0),
        poly2 == poly.remove(ContainerMonomial.empty()),
        poly1
        == map_fold_kv(
            lambda selected, candidate_mono, candidate_coef: if_defined(
                candidate_coef != -coef,
                selected,
                selected.insert(candidate_mono, candidate_coef),
            ),
            ContainerPolynomial.empty(),
            poly2,
        ),
        poly1.length() > i64(0),
        mono == poly1.pick_key(),
    )

    # If a polynomial contains both `P` and `c*M*P`, preserve the shared
    # product instead of only keeping the expanded container presentation:
    #
    # P + c*M*P + R -> P * (1 + c*M) + R
    #
    # This is the container analogue of the binary path that keeps repeated
    # subexpressions available for later reciprocal/log factoring.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            poly2.insert(
                ContainerMonomial
                .empty()
                .insert(polynomial(poly1), BigRat(1, 1))
                .insert(
                    polynomial(
                        ContainerPolynomial
                        .empty()
                        .insert(ContainerMonomial.empty(), f64(1.0))
                        .insert(mono.remove(n), coef)
                    ),
                    BigRat(1, 1),
                ),
                f64(1.0),
            )
        ),
        polynomial_bodies == POLYNOMIAL_BODIES,
        counts
        == map_fold_kv(lambda counts, candidate_mono, _coef: counts + candidate_mono.keys(), MultiSet[Num](), poly),
        matching_polynomial_bodies == map_restrict_keys(counts, polynomial_bodies),
        matching_polynomial_bodies.length() > i64(0),
        n == matching_polynomial_bodies.pick_key(),
        poly1 == matching_polynomial_bodies[n],
        mono
        == map_fold_kv(
            lambda selected, candidate_mono, _candidate_coef: try_match(
                candidate_mono[n],
                lambda exp: if_defined(exp != BigRat(1, 1), selected, candidate_mono),
                selected,
            ),
            ContainerMonomial.empty(),
            poly,
        ),
        mono.length() > i64(0),
        map_restrict_keys(poly1.keys(), poly.remove(mono)) == poly1,
        poly2 == map_remove_keys(poly1.keys(), poly.remove(mono)),
        coef == poly[mono],
    )

    # If an outer `M * P - c` appears, choose the outer constant as the scale.
    # This is more targeted than trying every coefficient of P and covers the
    # binary path:
    #
    # (c*x + d*y + k) * M - c -> c * (M * (x + (d/c)*y + k/c) - 1)
    yield rewrite(polynomial(poly)).to(
        polynomial(
            ContainerPolynomial.empty().insert(
                ContainerMonomial.empty().insert(
                    polynomial(
                        ContainerPolynomial
                        .empty()
                        .insert(ContainerMonomial.empty(), f64(-1.0))
                        .insert(
                            mono.remove(n).insert(
                                polynomial(map_divide_all_values_by_f64(inner_coef, poly1)),
                                BigRat(1, 1),
                            ),
                            f64(1.0),
                        )
                    ),
                    BigRat(1, 1),
                ),
                inner_coef,
            )
        ),
        polynomial_bodies == POLYNOMIAL_BODIES,
        poly.length() == i64(2),
        poly.contains(ContainerMonomial.empty()),
        coef == poly[ContainerMonomial.empty()],
        coef != f64(0.0),
        inner_coef == -coef,
        nonconst_poly == poly.remove(ContainerMonomial.empty()),
        nonconst_poly.length() == i64(1),
        mono == nonconst_poly.pick_key(),
        poly[mono] == f64(1.0),
        matching_polynomial_bodies == map_restrict_keys(mono.keys(), polynomial_bodies),
        matching_polynomial_bodies.length() == i64(1),
        n == matching_polynomial_bodies.pick_key(),
        mono[n] == BigRat(1, 1),
        poly1 == matching_polynomial_bodies[n],
        map_nonconst_nonunit_f64_values(poly1).contains(inner_coef),
    )
    # Flatten nested polynomials, where a a monomial term is itself a polynomial, but just with a single monomial that has an integer exponent, like:
    #
    # (xy)^2 -> x^2 * y^2
    # polynomial({ M: c })^z -> c^z * M^z

    yield rewrite(polynomial(poly)).to(
        polynomial(poly1),
        # Keep the flattening bounded to avoid fanning out every large equivalent
        # presentation, but allow common scaled factors hidden in modest sums.
        # poly.length() <= i64(5),
        # Add this as a direct dependency, to work around current semi-naive bug where it won't know what's pulled in by higher order functions.
        sole_monomials == SOLE_MONOMIALS,
        counts == map_fold_kv(lambda counts, mono, _coef: counts + mono.keys(), MultiSet[Num](), poly),
        map_restrict_keys(counts, sole_monomials).length() > i64(0),
        poly1
        == map_fold_kv(
            lambda res_poly, mono, coef: res_poly.insert(
                map_fold_kv(
                    lambda res_mono, term, exp: (
                        # Only collapse monomials with integer exponents, so we don't end up with (-xy)^0.5 taking the sqrt of a negative number
                        # Also can fail if this term has no sole_monomials present for it
                        try_match(
                            map_map_values(lambda _, c: c * exp.to_i64(), sole_monomials[term].right),
                            lambda m: map_merge_with(lambda l, r: l + r, m, res_mono),
                            res_mono.insert(term, exp),
                        )
                    ),
                    ContainerMonomial.empty(),
                    mono,
                ),
                coef,
            ),
            ContainerPolynomial.empty(),
            poly,
        ),
        poly != poly1,
    )

    # Container analogue of the binary `-(x + y) -> -x - y` rule.
    # It only pushes an outer `-1` into one nested two-term polynomial factor,
    # instead of expanding arbitrary scaled polynomial factors.
    yield rewrite(polynomial(poly)).to(
        polynomial(poly1),
        polynomial_monomials == POLYNOMIAL_MONOMIALS,
        poly.length() == i64(1),
        mono == poly.pick_key(),
        poly[mono] == f64(-1.0),
        polynomial_factor
        == map_fold_kv(
            lambda selected, term, _exp: try_match(
                polynomial_monomials[ContainerMonomial.empty().insert(term, BigRat(1, 1))],
                lambda nested_poly: Pair(term, nested_poly),
                selected,
            ),
            Pair(Num(0.0), ContainerPolynomial.empty()),
            mono,
        ),
        n == polynomial_factor.left,
        mono[n] == BigRat(1, 1),
        poly2 == polynomial_factor.right,
        poly2.length() == i64(2),
        poly1
        == map_fold_kv(
            lambda res_poly, inner_mono, inner_coef: map_merge_with(
                lambda l, r: l + r,
                res_poly,
                ContainerPolynomial.empty().insert(
                    map_merge_with(lambda l, r: l + r, mono.remove(n), inner_mono),
                    -inner_coef,
                ),
            ),
            ContainerPolynomial.empty(),
            poly2,
        ),
        poly != poly1,
    )

    # a variant of greedy multivariate horner factorization for rational exponents.
    # chooses just one term to extract at a time (for now), based on the term that shows up in the most monomials
    # then factors out the minimum exponent of that term in all monomials it is in.
    # to support rules like
    # (x*y) + (x*z) -> x * (y + z)
    yield rewrite(polynomial(poly)).to(
        polynomial(
            poly_pair.right.insert(
                ContainerMonomial.empty().insert(n, exp).insert(polynomial(poly_pair.left), BigRat(1, 1)),
                f64(1.0),
            )
        ),
        # Find term in most monomials
        counts == map_fold_kv(lambda counts, mono, _coef: counts + mono.keys(), MultiSet[Num](), poly),
        n == counts.pick_max(),
        # Make sure it shows up in more than one term
        counts.count(n) > i64(1),
        # Find minimum exponent of that term across all monomials it is in
        exp
        == map_fold_kv(
            # Skips monomials that don't contain the term or where the term is greater than the current minimum exponent
            lambda min_exp, mono, _coef: try_match(mono[n] < min_exp, lambda _: mono[n], min_exp),
            BigRat(2**63 - 1, 1),  # start with a very large exponent, so any real exponent will be smaller
            poly,
        ),
        # Now we go through each term, and either add it to the remainder or keep it in the main factored part
        poly_pair
        == map_fold_kv(
            lambda divided_and_remainder, mono, coef: try_match(
                mono[n],
                lambda current_exp: divided_and_remainder.map_left(
                    lambda divided: divided.insert(
                        mono.insert(n, current_exp - exp),
                        coef,
                    )
                ),
                divided_and_remainder.map_right(lambda remainder: remainder.insert(mono, coef)),
            ),
            Pair(ContainerPolynomial.empty(), ContainerPolynomial.empty()),
            poly,
        ),
    )

    # Flatten an exact nested polynomial term inside a larger polynomial:
    #
    # a*polynomial(P) + R -> a*P + R
    #
    # This is intentionally narrower than full product distribution. It only
    # applies when the whole monomial is the nested polynomial at exponent 1,
    # which is enough to expose constants from rules like `log(a*P)` while
    # avoiding the earlier blowup from distributing arbitrary products.
    yield rewrite(polynomial(poly)).to(
        polynomial(
            map_merge_with(
                lambda x, y: x + y,
                poly.remove(mono),
                map_map_values(lambda _nested_mono, nested_coef: nested_coef * poly[mono], poly1),
            )
        ),
        polynomial_monomials == POLYNOMIAL_MONOMIALS,
        poly.length() > i64(1),
        mono
        == map_fold_kv(
            lambda selected, candidate_mono, _candidate_coef: try_match(
                polynomial_monomials[candidate_mono],
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

    # a * polynomial(P) + b + R -> a * polynomial(P + b/a) + R
    yield rewrite(polynomial(poly)).to(
        polynomial(
            poly2.remove(mono).insert(
                ContainerMonomial.empty().insert(
                    polynomial(
                        map_merge_with(
                            lambda x, y: x + y,
                            poly1,
                            ContainerPolynomial.empty().insert(ContainerMonomial.empty(), (coef / poly[mono])),
                        ),
                    ),
                    BigRat(1, 1),
                ),
                poly[mono],
            )
        ),
        polynomial_monomials == POLYNOMIAL_MONOMIALS,
        poly.contains(ContainerMonomial.empty()),
        coef == poly[ContainerMonomial.empty()],
        poly2 == poly.remove(ContainerMonomial.empty()),
        mono
        == map_fold_kv(
            lambda selected, candidate_mono, _candidate_coef: try_match(
                polynomial_monomials[candidate_mono],
                lambda _nested_poly: candidate_mono,
                selected,
            ),
            ContainerMonomial.empty(),
            poly2,
        ),
        mono.length() > i64(0),
        mono.length() == i64(1),
        mono.pick_key() == n,
        mono[n] == BigRat(1, 1),
        poly[mono] != f64(0.0),
        poly1 == polynomial_monomials[mono],
    )


@ruleset
def shared_fun_rules(x: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(log(exp(x))).to(x)
    yield rewrite(log(log(x))).to(x)


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
def container_fun_rules(
    poly: ContainerPolynomial, poly1: ContainerPolynomial, m: ContainerMonomial
) -> Iterable[RewriteOrRule]:
    # log(a^n b^k) = n log(a) + k log(b) where a > 0 and b > 0
    yield rewrite(log(polynomial(poly))).to(
        polynomial(poly1),
        poly.length() == i64(1),
        m == poly.pick_key(),
        poly[m] > f64(0.0),
        poly1
        == map_fold_kv(
            lambda res_poly, term, exp: res_poly.insert(
                ContainerMonomial.empty().insert(log(term), BigRat(1, 1)),
                exp.to_f64(),
            ),
            ContainerPolynomial.empty().insert(ContainerMonomial.empty(), poly[m].log()),
            m,
        ),
    )


@ruleset
def shared_extra_rules(x: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(log(abs(exp(x)))).to(x)


@ruleset
def binary_extra_rules(
    x: Num,
    y: Num,
    z: Num,
    w: Num,
    u: Num,
    v: Num,
    t: Num,
    af: f64,
    bf: f64,
    cf: f64,
    df: f64,
) -> Iterable[RewriteOrRule]:
    yield from ()
    # a = Num(af)
    # b = Num(bf)
    # c = Num(cf)
    # d = Num(df)

    # # Numeric constants inside exp can often be absorbed by surrounding
    # # coefficient rules after they are exposed as multiplicative factors.
    # yield rewrite(exp(a + x)).to(exp(a) * exp(x))
    # yield rewrite(exp(x + a)).to(exp(a) * exp(x))
    # yield rewrite(exp(x - a)).to(exp(-a) * exp(x))
    # yield rewrite(exp(a - x)).to(exp(a) / exp(x))
    # yield rewrite(exp(x * (y + a))).to(exp(x * y) * exp(a * x))
    # yield rewrite(exp(x * (y - a))).to(exp(x * y) * exp(-a * x))
    # yield rewrite(exp((x + a) - y)).to(exp(a) * exp(x - y))
    # yield rewrite(exp(((x + a) - y) - z)).to(exp(a) * exp((x - y) - z))
    # yield rewrite(exp((x + a) / b) / c).to(
    #     (exp(a / b) / c) * exp(x / b),
    #     bf != f64(0.0),
    #     cf != f64(0.0),
    # )
    # yield rewrite(exp((x - a) / b) / c).to(
    #     (exp(-a / b) / c) * exp(x / b),
    #     bf != f64(0.0),
    #     cf != f64(0.0),
    # )
    # yield rewrite(exp((a * x - x) * (x / b))).to(
    #     exp(((a - Num(1.0)) / b) * (x * x)),
    #     bf != f64(0.0),
    # )

    # # Coefficient exposure for quotients. These keep the algebra local and let
    # # the existing constant-folding and affine rules combine the exposed terms.
    # yield rewrite(x / z + y / z).to((x + y) / z, z != Num(0.0))
    # yield rewrite(x / z - y / z).to((x - y) / z, z != Num(0.0))
    # yield rewrite(x / (y / a)).to((a * x) / y, af != f64(0.0), y != Num(0.0))
    # yield rewrite(x / (a / y)).to((x / a) * y, af != f64(0.0), y != Num(0.0))
    # yield rewrite(a / (x / y)).to((a * y) / x, x != Num(0.0), y != Num(0.0))
    # yield rewrite(x / a).to((Num(1.0) / a) * x, af != f64(0.0))
    # yield rewrite(a / (b / x + c)).to(
    #     (a / b) / (Num(1.0) / x + c / b),
    #     bf != f64(0.0),
    #     x != Num(0.0),
    # )
    # yield rewrite(a / (c + b / x)).to(
    #     (a / b) / (c / b + Num(1.0) / x),
    #     bf != f64(0.0),
    #     x != Num(0.0),
    # )
    # yield rewrite((a * x) / (b * y + z)).to(
    #     x / (y / (a / b) + z / a),
    #     af != f64(0.0),
    #     bf != f64(0.0),
    #     b * y + z != Num(0.0),
    # )
    # yield rewrite((a * x) / (z + b * y)).to(
    #     x / (z / a + y / (a / b)),
    #     af != f64(0.0),
    #     bf != f64(0.0),
    #     z + b * y != Num(0.0),
    # )
    # yield rewrite((x * y + z * w) / (y * w)).to(x / w + z / y, y != Num(0.0), w != Num(0.0))
    # yield rewrite((z * w + x * y) / (y * w)).to(z / y + x / w, y != Num(0.0), w != Num(0.0))
    # yield rewrite(a * (((x / b) / c) - y)).to(
    #     (a / (b * c)) * x - a * y,
    #     bf != f64(0.0),
    #     cf != f64(0.0),
    # )
    # yield rewrite((w - x / z) + y / z).to(w + ((y - x) / z), z != Num(0.0))
    # yield rewrite((x + y * z) / z).to(x / z + y, z != Num(0.0))
    # yield rewrite((z * x) / (z * y + w)).to(x / (y + w / z), z != Num(0.0))
    # yield rewrite((z * x) / (w + z * y)).to(x / (w / z + y), z != Num(0.0))
    # yield rewrite((x / a) / x).to(Num(1.0) / a, af != f64(0.0), x != Num(0.0))
    # yield rewrite((x + y) * (a / x)).to(a + y * (a / x), x != Num(0.0))
    # yield rewrite((b * (c * x - z) + y) / (c * x - z)).to(
    #     b + y / (c * x - z),
    #     c * x - z != Num(0.0),
    # )
    # yield rewrite((x * (y * z - w)) / (z * u)).to(
    #     x * (y - w / z) / u,
    #     z != Num(0.0),
    #     u != Num(0.0),
    # )
    # yield rewrite((a * x + y - b * z) / (c * x - z)).to(
    #     b + y / (c * x - z),
    #     abs(af - bf * cf) <= f64(CONST_MERGE_TOLERANCE),
    #     c * x - z != Num(0.0),
    # )
    # yield rewrite(x / exp(log(abs(a + b / exp(y))) - y)).to(
    #     (x / abs(a)) * exp(y) / abs(Num(1.0) + (b / a) / exp(y)),
    #     af != f64(0.0),
    # )
    # yield rewrite(x / exp(log(abs(a - b / exp(y))) - y)).to(
    #     (x / abs(a)) * exp(y) / abs(Num(1.0) - (b / a) / exp(y)),
    #     af != f64(0.0),
    # )

    # # Affine constant collection that requires arithmetic across a scalar
    # # multiplication boundary, not just associativity/commutativity.
    # yield rewrite(a * (x + b) + y).to(a * (x + (b + y / a)), af != f64(0.0))
    # yield rewrite(a * (x - b) + y).to(a * (x + (y / a - b)), af != f64(0.0))
    # yield rewrite(a * x + y).to(a * (x + y / a), af != f64(0.0))
    # yield rewrite((a - x) + b).to((a + b) - x)
    # yield rewrite(a + (b - x)).to((a + b) - x)
    # yield rewrite(((x + a) - y) + b).to((x - y) + (a + b))
    # yield rewrite(c + a * ((x + b) - y)).to(
    #     a * ((x - y) + (b + c / a)),
    #     af != f64(0.0),
    # )
    # yield rewrite(a * ((x + b) - y) + c).to(
    #     a * ((x - y) + (b + c / a)),
    #     af != f64(0.0),
    # )
    # yield rewrite(y + a * ((x + (b - z)) - w)).to(
    #     a * (((x + (b + y / a)) - z) - w),
    #     af != f64(0.0),
    # )
    # yield rewrite(((x + a) * (x + b)) + c).to(x * (x + (a + b)) + (a * b + c))
    # yield rewrite(((x + a) * (b + x)) + c).to(x * (x + (a + b)) + (a * b + c))
    # yield rewrite(c + ((x + a) * (x + b))).to(x * (x + (a + b)) + (a * b + c))
    # yield rewrite(((a - x) * (x + b)) + c).to((c + a * b) - x * (x + (b - a)))
    # yield rewrite(((a - x) * (x + b)) + (c + y)).to((c + a * b + y) - x * (x + (b - a)))
    # yield rewrite(x * x).to(x ** Num(2.0))
    # yield rewrite(x / a + x).to((Num(1.0) / a + Num(1.0)) * x, af != f64(0.0))
    # yield rewrite((a * x) ** Num(2.0)).to((a ** Num(2.0)) * (x ** Num(2.0)), af != f64(0.0))
    # yield rewrite((a * x) ** Num(3.0)).to((a ** Num(3.0)) * (x ** Num(3.0)), af != f64(0.0))
    # yield rewrite((a * x) ** Num(4.0)).to((a ** Num(4.0)) * (x ** Num(4.0)), af != f64(0.0))
    # yield rewrite((a * x) ** Num(6.0)).to((a ** Num(6.0)) * (x ** Num(6.0)), af != f64(0.0))
    # yield rewrite(((a * x) ** Num(3.0)) ** Num(2.0)).to(
    #     (a ** Num(6.0)) * (x ** Num(6.0)),
    #     af != f64(0.0),
    # )
    # yield rewrite(a * ((x * (b * y + z)) ** Num(-1.0))).to(
    #     (x * (y / (a / b) + z / a)) ** Num(-1.0),
    #     af != f64(0.0),
    #     bf != f64(0.0),
    #     x != Num(0.0),
    #     b * y + z != Num(0.0),
    # )
    # yield rewrite(x + a * x).to((Num(1.0) + a) * x)
    # yield rewrite(a * x + x).to((a + Num(1.0)) * x)
    # yield rewrite(a * x + b * x).to((a + b) * x)
    # yield rewrite(a * x - b * x).to((a - b) * x)
    # yield rewrite(a * (b + c / x + d / y)).to(
    #     (a * c) * (b / c + Num(1.0) / x + (d / c) / y),
    #     cf != f64(0.0),
    #     x != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(a * (b - c / x - d / y)).to(
    #     (a * c) * (b / c - Num(1.0) / x - (d / c) / y),
    #     cf != f64(0.0),
    #     x != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(a * (b + c / x - d / y)).to(
    #     (a * c) * (b / c + Num(1.0) / x - (d / c) / y),
    #     cf != f64(0.0),
    #     x != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(a * (b - c / x + d / y)).to(
    #     (a * c) * (b / c - Num(1.0) / x + (d / c) / y),
    #     cf != f64(0.0),
    #     x != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(a * (b + c / x + d / y)).to(
    #     a * b + (a * c) / x + (a * d) / y,
    #     x != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(a * (b - c / x - d / y)).to(
    #     a * b - (a * c) / x - (a * d) / y,
    #     x != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(x * (a * y + z) - w).to(a * (x * (y + z / a) - w / a), af != f64(0.0))
    # yield rewrite(x * (z + a * y) - w).to(a * (x * (z / a + y) - w / a), af != f64(0.0))
    # yield rewrite(a * x + b * y).to(
    #     a * (x + y),
    #     abs(af - bf) <= f64(CONST_MERGE_TOLERANCE),
    #     af != f64(0.0),
    # )

    # # Scale extraction through log(abs(...)). These mirror the existing log
    # # scale rules but account for an intervening abs node.
    # yield rewrite(a + log(abs(b / x + (x / c) ** Num(3.0)))).to(
    #     log(abs((exp(a) * b) / x + (exp(a) / (c ** Num(3.0))) * (x ** Num(3.0)))),
    #     x != Num(0.0),
    #     cf != f64(0.0),
    # )
    # yield rewrite(log(abs(b / x + (x / c) ** Num(3.0))) + a).to(
    #     log(abs((exp(a) * b) / x + (exp(a) / (c ** Num(3.0))) * (x ** Num(3.0)))),
    #     x != Num(0.0),
    #     cf != f64(0.0),
    # )
    # yield rewrite(log(abs((x / a) / y + b))).to(
    #     log(abs(x / y + a * b)) - log(abs(a)),
    #     af != f64(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(c * abs(a + b / x)).to(
    #     (c * abs(b)) * abs(a / b + Num(1.0) / x),
    #     bf != f64(0.0),
    #     x != Num(0.0),
    # )

    # # Exponential product/quotient normalization. These are intentionally small;
    # # coefficient absorption is left to existing arithmetic rules.
    # yield rewrite((x * exp(z) + y) * exp(-z)).to(x + y * exp(-z))
    # yield rewrite((x * exp(a * z) + y) * exp(Num(-af) * z)).to(
    #     x + y * exp(Num(-af) * z),
    #     af != f64(0.0),
    # )
    # yield rewrite(x * exp(u) + y * exp(u + v)).to(exp(u) * (x + y * exp(v)))
    # yield rewrite((x * exp(z) + y) * exp(w - z)).to((x + y * exp(-z)) * exp(w))
    # yield rewrite(((x * exp(z) + y) * u) * exp(w - z)).to(u * (x + y * exp(-z)) * exp(w))
    # yield rewrite((u * (x * exp(z) + y)) * exp(w - z)).to(u * (x + y * exp(-z)) * exp(w))
    # yield rewrite((x * exp(a * z) + y) * exp(w + Num(-af) * z)).to(
    #     (x + y * exp(Num(-af) * z)) * exp(w),
    #     af != f64(0.0),
    # )
    # yield rewrite((u * (x * exp(a * z) + y)) * exp(w + Num(-af) * z)).to(
    #     u * (x + y * exp(Num(-af) * z)) * exp(w),
    #     af != f64(0.0),
    # )
    # yield rewrite(a / exp(y)).to(exp(log(a) - y), af > f64(0.0))
    # yield rewrite(a / (exp(y) ** Num(2.0))).to(exp(log(a) - Num(2.0) * y), af > f64(0.0))

    # # Move numeric scales out of squared denominators.
    # yield rewrite(x / ((a * y) ** Num(2.0))).to(
    #     (x / (a ** Num(2.0))) / (y ** Num(2.0)),
    #     af != f64(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite(x / (((a / y) ** Num(2.0)) / (z ** Num(2.0)))).to(
    #     (x / (a ** Num(2.0))) * (y ** Num(2.0)) * (z ** Num(2.0)),
    #     af != f64(0.0),
    #     y != Num(0.0),
    #     z != Num(0.0),
    # )
    # yield rewrite((x * (t ** Num(4.0))) / (y * (z + t ** Num(2.0)) ** Num(2.0))).to(
    #     x / (y * (Num(1.0) + z / (t ** Num(2.0))) ** Num(2.0)),
    #     t != Num(0.0),
    #     y != Num(0.0),
    # )
    # yield rewrite((x ** Num(4.0)) / ((y * (x ** Num(2.0)) + z) ** Num(2.0))).to(
    #     Num(1.0) / ((y + z / (x ** Num(2.0))) ** Num(2.0)),
    #     x != Num(0.0),
    # )
    # yield rewrite(log(((x * z + y) ** Num(2.0)) / ((a * z) ** Num(2.0)))).to(
    #     log((x + y / z) ** Num(2.0)) - log(a ** Num(2.0)),
    #     af != f64(0.0),
    #     z != Num(0.0),
    # )
    # yield rewrite(log(((x * z + y) ** Num(2.0)) / (z ** Num(2.0)))).to(
    #     log((x + y / z) ** Num(2.0)),
    #     z != Num(0.0),
    # )

    # # Conservative tolerance canonicalizations. These mirror the existing
    # # constant merge tolerance and avoid relation-based coefficient guessing.
    # yield rewrite(x + a).to(x, abs(af) <= f64(CONST_MERGE_TOLERANCE))
    # yield rewrite(a + x).to(x, abs(af) <= f64(CONST_MERGE_TOLERANCE))
    # yield rewrite(a * x).to(x, abs(af - f64(1.0)) <= f64(CONST_MERGE_TOLERANCE))
    # yield rewrite(a * x).to(-x, abs(af + f64(1.0)) <= f64(CONST_MERGE_TOLERANCE))
    # yield rewrite(x * a).to(x, abs(af - f64(1.0)) <= f64(CONST_MERGE_TOLERANCE))
    # yield rewrite(x * a).to(-x, abs(af + f64(1.0)) <= f64(CONST_MERGE_TOLERANCE))


@ruleset
def container_extra_rules() -> Iterable[RewriteOrRule]:
    yield from ()


@dataclass(frozen=True)
class PaperPipelineReport:
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
analysis_rules = binary_analysis_ruleset
analysis_schedule = binary_analysis_schedule
shared_rewrite_ruleset = shared_fun_rules | shared_extra_rules
binary_rewrite_ruleset = shared_rewrite_ruleset | binary_basic_rules | binary_fun_rules | binary_extra_rules
container_rewrite_ruleset = shared_rewrite_ruleset | container_basic_rules | container_fun_rules | container_extra_rules


def _graph_size(egraph: EGraph) -> int:
    return sum(size for _, size in egraph.all_function_sizes())


def parse_expression_container(source: str) -> Num:
    return binary_to_containers(parse_expression(source))


rewrite_scheduler: BackOff = back_off(
    match_limit=BACKOFF_MATCH_LIMIT,
    ban_length=BACKOFF_BAN_LENGTH,
    fresh_rematch=True,
).persistent()

binary_schedule = run(binary_rewrite_ruleset, scheduler=rewrite_scheduler)
container_schedule = run(container_rewrite_ruleset, scheduler=rewrite_scheduler)



def _run_single_pass(
    egraph: EGraph,
    num: Num,
    cost_model: CostModel[ParamCost],
    analysis_schedule: Schedule,
    schedule: Schedule,
) -> tuple[Num, ParamCost, int]:
    """
    Run one `rewriteTree`-like pass and return the populated e-graph.

    This mirrors Haskell at the control-flow level:
    - one fresh-rematch backoff scheduler per outer pass
    - up to 30 inner rewrite rounds
    - one saturated analysis round after each rewrite round
    - stop when total egraph size stops changing
    """
    n = egraph.let("n", num)
    previous_size = _graph_size(egraph)
    for _ in range(HASKELL_INNER_ITERATION_LIMIT):
        egraph.run(analysis_schedule)
        egraph.run(schedule)
        current_size = _graph_size(egraph)
        if current_size == previous_size:
            break
        previous_size = current_size
    extracted, cost = egraph.extract(n, include_cost=True, cost_model=cost_model)
    # print(egraph._state.egglog_file_state)
    return extracted, cost, current_size


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
    for pass_index in range(1, MAX_PASSES + 1):
        with egraph:
            extracted, last_cost, total_size = _run_single_pass(
                egraph,
                current,
                cost_model=cost_model,
                schedule=schedule,
                analysis_schedule=analysis_schedule,
            )
        max_size = max(max_size, total_size)
        passes = pass_index
        if extracted == current:
            break
        current = extracted
    return PaperPipelineReport(
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
    return run_paper_pipeline(
        initial,
        decode=containers_to_binary,
        cost_model=container_cost_model,
        schedule=container_schedule,
        analysis_schedule=containers_analysis_schedule,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expr", required=True)
    parser.add_argument("--variant", choices=("baseline", "container"), default="baseline")
    args = parser.parse_args()
    if args.variant == "baseline":
        report = run_paper_pipeline(parse_expression(args.expr))
    else:
        report = run_paper_pipeline_container(parse_expression_container(args.expr))
    payload = {
        "passes": report.passes,
        "total_sec": report.total_sec,
        "total_size": report.total_size,
        "before_nodes": report.before_nodes,
        "before_params": report.before_params,
        "after_nodes": report.extracted_nodes,
        "after_params": report.extracted_params,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    _cli()
