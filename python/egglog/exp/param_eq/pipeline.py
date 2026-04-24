# mypy: disable-error-code="empty-body"

"""Retained paper-era `param_eq` pipeline plus the experimental map variant."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

from egglog import *
from egglog.egraph import UnstableCombinedRuleset

from .domain import *

MAX_PASSES = 2
HASKELL_INNER_ITERATION_LIMIT = 30
BACKOFF_MATCH_LIMIT = 1000
BACKOFF_BAN_LENGTH = 30
CONST_MERGE_TOLERANCE = 1e-6


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

# Store a mapping from e-classes that are polynomials with a sole monomial
# to one representative sole monomial. Don't store all of them at this time.
# Used for folding nested polynomials like turning (xy)^2 into x^2 * y^2, or polynomial({ M: c })^z into c^z * M^z.
SOLE_MONOMIALS = constant(
    "SOLE_MONOMIALS", Map[Num, ContainerMonomial], merge=partial(map_merge_with, lambda old_value, new_value: old_value)
)

# Mapping of e-classes to polynomials, so that we can use this for distributing constants inside of polynomials.
# POLYNOMIALS = constant(
#     "POLYNOMIALS", Map[Num, ContainerPolynomial], merge=partial(map_merge_with, lambda old_value, new_value: old_value)
# )


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
    n: Num, a: f64, poly: ContainerPolynomial, poly1: ContainerPolynomial, consts: Map[Num, f64], key: ContainerMonomial
) -> Iterable[RewriteOrRule]:
    # default values
    yield rule().then(
        set_(CONSTS).to(Map[Num, f64].empty()),
        set_(SOLE_MONOMIALS).to(Map[Num, ContainerMonomial].empty()),
        # set_(POLYNOMIALS).to(Map[Num, ContainerPolynomial].empty()),
    )
    yield rule(n == Num(a)).then(set_(CONSTS).to(Map[Num, f64].empty().insert(n, a)))
    yield rule(
        polynomial(poly) == n,
        poly.length() == i64(1),
        key == poly.pick_key(),
    ).then(
        set_(SOLE_MONOMIALS).to(
            Map[Num, ContainerMonomial]
            .empty()
            # add constant term into monomial
            .insert(n, key.insert(Num(poly[key]), BigRat(1, 1)))
        )
    )
    # yield rule(polynomial(poly) == n).then(set_(POLYNOMIALS).to(Map[Num, ContainerPolynomial].empty().insert(n, poly)))

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
                    lambda res_mono_and_coef, term, exp: catch(lambda: exp != BigRat(0, 1)).match(
                        # if the exponent is not zero, process it
                        lambda _: catch(lambda: consts[term]).match(
                            # if it is a constant, multiply it into the coefficient and drop it from the monomial:
                            lambda v: res_mono_and_coef.map_right(lambda prev_coef: prev_coef * v.pow_bigrat(exp)),
                            # if it is not a constant, keep it in the monomial
                            res_mono_and_coef.map_left(lambda mono: mono.insert(term, exp)),
                        ),
                        # if the exponent is zero, the term is just 1 and can be dropped from the monomial, so keep the monomial as is
                        res_mono_and_coef,
                    ),
                    Pair(ContainerMonomial.empty(), coef),
                    mono,
                ).match(
                    lambda mono, coef: res_poly.insert(mono, coef + catch(lambda: res_poly[mono]).unwrap_or(f64(0.0)))
                )
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
    coef: f64,
    sole_monomials: Map[Num, ContainerMonomial],
    counts: MultiSet[Num],
    n: Num,
    poly_pair: Pair[ContainerPolynomial, ContainerPolynomial],
    exp: BigRat,
    # polynomials: Map[Num, ContainerPolynomial],
    mono: ContainerMonomial,
) -> Iterable[RewriteOrRule]:
    # If a non one coefficient exists in the polynomial it has more than one monomial
    # factor it out, as long as its not a constant. In a binary form this covers things like:
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
        poly2
        == map_filter_kv(lambda k, v: v != f64(1.0), map_filter_kv(lambda k, v: k != ContainerMonomial.empty(), poly)),
        # Pick the first one (stopping of course if there are none)
        coef == poly2[poly2.pick_key()],
        # divide the terms by that coefficient
        poly1 == map_map_values(lambda _, c: c / coef, poly),
    )
    # Flatten nested polynomials, where a a monomial term is itself a polynomial, but just with a single monomial that has an integer exponent, like:
    #
    # (xy)^2 -> x^2 * y^2
    # polynomial({ M: c })^z -> c^z * M^z
    yield rewrite(polynomial(poly)).to(
        polynomial(poly1),
        sole_monomials == SOLE_MONOMIALS,
        poly1
        == map_fold_kv(
            lambda res_poly, mono, coef: res_poly.insert(
                map_fold_kv(
                    lambda res_mono, term, exp: (
                        # Only collapse monomials with integer exponents, so we don't end up with (-xy)^0.5 taking the sqrt of a negative number
                        # Also can fail if this term has no sole_monomials present for it
                        catch(lambda: map_map_values(lambda _, c: c * exp.to_i64(), sole_monomials[term])).match(
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
            lambda min_exp, mono, _coef: catch(lambda: mono[n] < min_exp).match(lambda _: mono[n], min_exp),
            BigRat(2**63 - 1, 1),  # start with a very large exponent, so any real exponent will be smaller
            poly,
        ),
        # Now we go through each term, and either add it to the remainder or keep it in the main factored part
        poly_pair
        == map_fold_kv(
            lambda divided_and_remainder, mono, coef: catch(lambda: mono[n]).match(
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

    # DISABLE THIS FOR NOW BECAUSE IT'S TOO BROAD AND REPLACE WITH SPECIAL CASE BELOW
    # Distribute constants inside of polynomials. If we have a monomial that is just another polynomial,
    # then we can distribute the coefficient of that monomial into the inner polynomial, multiplying it by each of the coefficients in the inner polynomial,
    # and merge it into the outer one
    # c * polynomial({ M: c2 })^1 -> polynomial({ M: c * c2 })
    # yield rewrite(polynomial(poly)).to(
    #     polynomial(poly1),
    #     polynomials == POLYNOMIALS,
    #     poly1
    #     == map_fold_kv(
    #         lambda res_poly, mono, coef: assume_not(
    #             ContainerPolynomial,
    #             catch(lambda: polynomials[mono.pick_key()]),
    #             mono.length() != i64(1),
    #             mono[mono.pick_key()] != BigRat(1, 1),
    #         ).match(
    #             lambda inner_poly: map_merge_with(
    #                 lambda x, y: x + y, res_poly, map_map_values(lambda _, v: v * coef, inner_poly)
    #             ),
    #             res_poly.insert(mono, coef),
    #         ),
    #         ContainerPolynomial.empty(),
    #         poly,
    #     ),
    # )

    # a * polynomial(P) + b -> a * polynomial(P + b/a)
    # special case this for only polynomials of size two, where one is a constant and the other is a polynomial with a coefficient
    yield rewrite(polynomial(poly)).to(
        polynomial(
            ContainerPolynomial.empty().insert(
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
        poly.length() == i64(2),
        coef == poly[ContainerMonomial.empty()],
        poly.remove(ContainerMonomial.empty()).pick_key() == mono,
        mono.length() == i64(1),
        mono.pick_key() == n,
        mono[n] == BigRat(1, 1),
        n == polynomial(poly1),
    )


T = TypeVar("T", bound=BaseExpr)


def assume_not(t: type[T], value: Maybe[T], *conds: Unit) -> Maybe[T]:
    """
    Assume that the condition is not true. If the condition is true, this will return Maybe.none() and block the rule from firing. If the condition is false, this will return the value as a Maybe.some and allow the rule to fire with that value.
    """
    for cond in conds:
        value = catch(lambda: cond).match(lambda _: Maybe[t].none(), value)
    return value


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


rewrite_scheduler: BackOff | None = back_off(
    match_limit=BACKOFF_MATCH_LIMIT,
    ban_length=BACKOFF_BAN_LENGTH,
    fresh_rematch=True,
).persistent()


def _new_rewrite_scheduler() -> BackOff | None:
    if rewrite_scheduler is None:
        return None
    scheduler = rewrite_scheduler.scheduler
    return back_off(
        match_limit=scheduler.match_limit,
        ban_length=scheduler.ban_length,
        fresh_rematch=scheduler.fresh_rematch,
    ).persistent()


def _run_single_pass(
    num: Num,
    cost_model: CostModel[ParamCost],
    analysis_schedule: Schedule,
    active_rewrite_ruleset: Ruleset | UnstableCombinedRuleset,
) -> tuple[Num, ParamCost, int, float]:
    """
    Run one `rewriteTree`-like pass and return the populated e-graph.

    This mirrors Haskell at the control-flow level:
    - one fresh-rematch backoff scheduler per outer pass
    - up to 30 inner rewrite rounds
    - one saturated analysis round after each rewrite round
    - stop when total egraph size stops changing
    """
    egraph = EGraph(save_egglog_string=True)
    egraph.register(
        num,
        # so we can match against != 0
        Num(0.0),
    )

    start = time.perf_counter()
    previous_size = _graph_size(egraph)
    egraph.run(analysis_schedule)
    scheduler = _new_rewrite_scheduler()
    for _ in range(HASKELL_INNER_ITERATION_LIMIT):
        egraph.run(run(active_rewrite_ruleset, scheduler=scheduler))
        egraph.run(analysis_schedule)
        current_size = _graph_size(egraph)
        if current_size == previous_size:
            break
        previous_size = current_size
    egraph.extract(num)
    extracted, cost = egraph.extract(num, include_cost=True, cost_model=cost_model)
    elapsed = time.perf_counter() - start
    print(egraph._state.egglog_file_state)
    return extracted, cost, current_size, elapsed


def run_paper_pipeline(
    initial: Num,
    decode: Callable[[Num], Num] = lambda x: x,
    cost_model: CostModel[ParamCost] = param_cost_model,
    analysis_schedule: Schedule = binary_analysis_schedule,
    active_rewrite_ruleset: Ruleset | UnstableCombinedRuleset = binary_rewrite_ruleset,
) -> PaperPipelineReport:
    current, before_cost = EGraph().extract(initial, include_cost=True, cost_model=cost_model)
    decoded_current = decode(current)
    last_cost = ParamCost()
    total_size = 0
    total_sec = 0.0
    passes = 0
    for pass_index in range(1, MAX_PASSES + 1):
        extracted, last_cost, total_size, elapsed = _run_single_pass(
            current,
            cost_model=cost_model,
            analysis_schedule=analysis_schedule,
            active_rewrite_ruleset=active_rewrite_ruleset,
        )
        total_sec += elapsed
        passes = pass_index
        decoded_extracted = decode(extracted)
        if render_num(decoded_extracted) == render_num(decoded_current):
            current = extracted
            decoded_current = decoded_extracted
            break
        current = extracted
        decoded_current = decoded_extracted
    return PaperPipelineReport(
        passes=passes,
        total_sec=total_sec,
        total_size=total_size,
        before_nodes=before_cost.node_count,
        before_params=before_cost.floats,
        extracted_nodes=last_cost.node_count,
        extracted_params=last_cost.floats,
        extracted=render_num(decoded_current),
    )


def run_paper_pipeline_container(initial: Num) -> PaperPipelineReport:
    return run_paper_pipeline(
        initial,
        decode=containers_to_binary,
        cost_model=container_cost_model,
        analysis_schedule=containers_analysis_schedule,
        active_rewrite_ruleset=container_rewrite_ruleset,
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
