# mypy: disable-error-code="empty-body"
from __future__ import annotations

from functools import partial
from typing import TypeAlias

from egglog import *


class Number(Expr):
    def __init__(self, value: i64Like) -> None: ...
    def __add__(self, other: NumberLike) -> Number: ...
    def __radd__(self, other: NumberLike) -> Number: ...
    def __sub__(self, other: NumberLike) -> Number: ...
    def __rsub__(self, other: NumberLike) -> Number: ...
    def __mul__(self, other: NumberLike) -> Number: ...
    def __rmul__(self, other: NumberLike) -> Number: ...
    def __truediv__(self, other: NumberLike) -> Number: ...
    def __rtruediv__(self, other: NumberLike) -> Number: ...
    def __pow__(self, power: NumberLike) -> Number: ...
    def __neg__(self) -> Number: ...


NumberLike: TypeAlias = Number | i64Like
converter(i64, Number, Number)

bpp1 = constant("bpp1", Number)
bpp2 = constant("bpp2", Number)
bpp3 = constant("bpp3", Number)
bpp4 = constant("bpp4", Number)
bp1 = constant("bp1", Number)
bp2 = constant("bp2", Number)
bp3 = constant("bp3", Number)
bp4 = constant("bp4", Number)
q1 = constant("q1", Number)
q2 = constant("q2", Number)
q3 = constant("q3", Number)
q4 = constant("q4", Number)
q5 = constant("q5", Number)
q6 = constant("q6", Number)
q7 = constant("q7", Number)
q8 = constant("q8", Number)
q9 = constant("q9", Number)
q10 = constant("q10", Number)
q11 = constant("q11", Number)
q12 = constant("q12", Number)

res = (
    (
        -bp4 * bpp1 * q1 * q11
        + bp1 * bpp4 * q1 * q11
        + bp4 * bpp1 * q10 * q2
        - bp1 * bpp4 * q10 * q2
        - bp4 * bpp2 * q11 * q4
        + bp2 * bpp4 * q11 * q4
        + bp2 * bpp1 * q2 * q4
        - bp1 * bpp2 * q2 * q4
        - bp2 * bpp1 * q1 * q5
        + bp1 * bpp2 * q1 * q5
        + bp4 * bpp2 * q10 * q5
        - bp2 * bpp4 * q10 * q5
        - bp4 * bpp3 * q11 * q7
        + bp3 * bpp4 * q11 * q7
        + bp3 * bpp1 * q2 * q7
        - bp1 * bpp3 * q2 * q7
        + bp3 * bpp2 * q5 * q7
        - bp2 * bpp3 * q5 * q7
        - bp3 * bpp1 * q1 * q8
        + bp1 * bpp3 * q1 * q8
        + bp4 * bpp3 * q10 * q8
        - bp3 * bpp4 * q10 * q8
        - bp3 * bpp2 * q4 * q8
        + bp2 * bpp3 * q4 * q8
    )
    ** 2
    + (
        bp4 * bpp1 * q1 * q12
        - bp1 * bpp4 * q1 * q12
        - bp4 * bpp1 * q10 * q3
        + bp1 * bpp4 * q10 * q3
        + bp4 * bpp2 * q12 * q4
        - bp2 * bpp4 * q12 * q4
        - bp2 * bpp1 * q3 * q4
        + bp1 * bpp2 * q3 * q4
        + bp2 * bpp1 * q1 * q6
        - bp1 * bpp2 * q1 * q6
        - bp4 * bpp2 * q10 * q6
        + bp2 * bpp4 * q10 * q6
        + bp4 * bpp3 * q12 * q7
        - bp3 * bpp4 * q12 * q7
        - bp3 * bpp1 * q3 * q7
        + bp1 * bpp3 * q3 * q7
        - bp3 * bpp2 * q6 * q7
        + bp2 * bpp3 * q6 * q7
        + bp3 * bpp1 * q1 * q9
        - bp1 * bpp3 * q1 * q9
        - bp4 * bpp3 * q10 * q9
        + bp3 * bpp4 * q10 * q9
        + bp3 * bpp2 * q4 * q9
        - bp2 * bpp3 * q4 * q9
    )
    ** 2
    + (
        -bp4 * bpp1 * q12 * q2
        + bp1 * bpp4 * q12 * q2
        + bp4 * bpp1 * q11 * q3
        - bp1 * bpp4 * q11 * q3
        - bp4 * bpp2 * q12 * q5
        + bp2 * bpp4 * q12 * q5
        + bp2 * bpp1 * q3 * q5
        - bp1 * bpp2 * q3 * q5
        + bp4 * bpp2 * q11 * q6
        - bp2 * bpp4 * q11 * q6
        - bp2 * bpp1 * q2 * q6
        + bp1 * bpp2 * q2 * q6
        - bp4 * bpp3 * q12 * q8
        + bp3 * bpp4 * q12 * q8
        + bp3 * bpp1 * q3 * q8
        - bp1 * bpp3 * q3 * q8
        + bp3 * bpp2 * q6 * q8
        - bp2 * bpp3 * q6 * q8
        + bp4 * bpp3 * q11 * q9
        - bp3 * bpp4 * q11 * q9
        - bp3 * bpp1 * q2 * q9
        + bp1 * bpp3 * q2 * q9
        - bp3 * bpp2 * q5 * q9
        + bp2 * bpp3 * q5 * q9
    )
    ** 2
) / (
    (bp1 * q1 + bp4 * q10 + bp2 * q4 + bp3 * q7) ** 2
    + (bp4 * q11 + bp1 * q2 + bp2 * q5 + bp3 * q8) ** 2
    + (bp4 * q12 + bp1 * q3 + bp2 * q6 + bp3 * q9) ** 2
) ** 3
# print(res)


# 1. Convert to polynomial(MultiSet[MultiSet[Number]])


n1 = constant("n1", Number)
n2 = constant("n2", Number)
n3 = constant("n3", Number)


@function
def monomial(x: MultiSetLike[Number, NumberLike]) -> Number: ...


@function(merge=lambda old, new: new)
def get_monomial(x: Number) -> MultiSet[Number]:
    """
    Only defined on monomials:

        get_monomial(monomial(xs)) => xs
    """


@function(merge=lambda old, new: new)
def get_sole_polynomial(xs: MultiSet[Number]) -> MultiSet[MultiSet[Number]]:
    """
    Only defined on monomials that contain a single polynomial:

        get_sole_polynomial(MultiSet(polynomial(xs))) => xs
    """


@function
def polynomial(x: MultiSetLike[MultiSet[Number], MultiSetLike[Number, NumberLike]]) -> Number: ...


# TODO: simplify MultiSet(bp3, polynomial(MultiSet(MultiSet(q7, _Number_6))))
# @function(merge=i64.__add__)
# def ms_index(xs: MultiSet[Number], x: Number) -> i64: ...


# @function(merge=i64.__add__)
# def mss_index(xss: MultiSet[MultiSet[Number]], xs: MultiSet[Number]) -> i64: ...


@ruleset
def to_polynomial_ruleset(
    n1: Number,
    n2: Number,
    n3: Number,
    # ms: MultiSet[Number],
    # ms1: MultiSet[Number],
    # ms2: MultiSet[Number],
    mss: MultiSet[MultiSet[Number]],
    mss1: MultiSet[MultiSet[Number]],
    # mss2: MultiSet[MultiSet[Number]],
):
    yield rewrite(-n1, subsume=True).to(-1 * n1)
    yield rewrite(n1 - n2, subsume=True).to(n1 + (-1 * n2))
    yield rule(
        n3 == n1 + n2,
        name="add",
    ).then(
        union(n3).with_(polynomial(MultiSet(MultiSet(n1), MultiSet(n2)))),
        set_(get_sole_polynomial(MultiSet(n3))).to(MultiSet(MultiSet(n1), MultiSet(n2))),
        delete(n1 + n2),
        # MultiSet(MultiSet(n1), MultiSet(n2)).fill_index(mss_index),
        # MultiSet(n1).fill_index(ms_index),
        # MultiSet(n2).fill_index(ms_index),
    )
    yield rule(
        n3 == n1 * n2,
        name="mul",
    ).then(
        union(n3).with_(monomial(MultiSet(n1, n2))),
        set_(get_monomial(n3)).to(MultiSet(n1, n2)),
        delete(n1 * n2),
        # MultiSet(n1, n2).fill_index(ms_index),
    )
    yield rule(
        n1 == polynomial(mss),
        mss1 == mss.map(partial(multiset_flat_map, get_monomial)),
        mss != mss1,
        name="unwrap monomial",
    ).then(
        union(n1).with_(polynomial(mss1)),
        delete(polynomial(mss)),
        set_(get_sole_polynomial(MultiSet(n1))).to(mss1),
        # mss.clear_index(mss_index),
        # mss1.fill_index(mss_index),
    )
    yield rule(
        n1 == polynomial(mss),
        mss1 == multiset_flat_map(UnstableFn(get_sole_polynomial), mss),
        mss != mss1,
        name="unwrap polynomial",
    ).then(
        union(n1).with_(polynomial(mss1)),
        delete(polynomial(mss)),
        set_(get_sole_polynomial(MultiSet(n1))).to(mss1),
        # mss.clear_index(mss_index),
        # ms.clear_index(ms_index),
        # ms2.fill_index(ms_index),
        # mss2.fill_index(mss_index),
    )


@ruleset
def factor_ruleset(
    n: Number,
    mss: MultiSet[MultiSet[Number]],
    counts: MultiSet[Number],
    factor: Number,
    divided: MultiSet[MultiSet[Number]],
    remainder: MultiSet[MultiSet[Number]],
):
    yield rule(
        n == polynomial(mss),
        # Find factor that shows up in most monomials, at least two of them
        counts == MultiSet.sum_multisets(mss.map(MultiSet[Number].reset_counts)),
        factor == counts.pick_max(),
        # Only factor out if it term appears in more than one monomial
        counts.count(factor) > 1,
        # map, including only those factor that contain the factor, removing them from that
        # other items are omitted from the map
        divided == mss.map(partial(multiset_remove_swapped, factor)),
        # remainder is those monomials that do not contain the factor
        remainder == mss.filter(partial(multiset_not_contains_swapped, factor)),
        name="factor",
    ).then(
        union(n).with_(polynomial(MultiSet(MultiSet(factor, polynomial(divided))) + remainder)),
        delete(polynomial(mss)),
    )


@function(merge=lambda old, new: new)
def get_polynomial_sole_monomial(x: Number) -> MultiSet[Number]:
    """
    Only defined on polynomials that contain a single monomial:

        get_polynomial_sole_monomial(polynomial(MultiSet(xs))) => xs
    """


@ruleset
def simplified_factor_ruleset(
    n: Number,
    mss: MultiSet[MultiSet[Number]],
    mss1: MultiSet[MultiSet[Number]],
    ms: MultiSet[Number],
):
    # replace all monomial with polyomials with one monomial with just monomials

    # a = polynomial(
    #     MultiSet(
    #         MultiSet(bp1, bpp4, q12, q1),
    #         MultiSet(q7, polynomial(MultiSet(MultiSet(bp3, _Number_10)))),
    #         MultiSet(q9, polynomial(MultiSet(MultiSet(bpp3, _Number_8)))),
    #     )
    # )
    # replace_with = polynomial(
    #     MultiSet(
    #         MultiSet(bp1, bpp4, q12, q1),
    #         MultiSet(q7, bp3, _Number_10),
    #         MultiSet(q9, bpp3, _Number_8),
    #     )
    # )

    yield rule(
        n == polynomial(mss),
        mss.length() == i64(1),
        ms == mss.pick(),
        name="set polynomial sole monomial",
    ).then(
        set_(get_polynomial_sole_monomial(n)).to(ms),
    )
    yield rule(
        n == polynomial(mss),
        mss1 == mss.map(partial(multiset_flat_map, get_polynomial_sole_monomial)),
        mss != mss1,
        name="unwrap polynomial sole monomial",
    ).then(
        union(n).with_(polynomial(mss1)),
        subsume(polynomial(mss)),
    )


# _Number_8 = polynomial(MultiSet(MultiSet(bp1, q1), MultiSet(bp4, q10), MultiSet(bp2, q4)))
# _Number_10 = polynomial(MultiSet(MultiSet(bpp4, q12), MultiSet(bpp1, q3), MultiSet(bpp2, q6)))

# a = polynomial(
#     MultiSet(
#         MultiSet(bp1, bpp4, q12, q1),
#         MultiSet(q7, polynomial(MultiSet(MultiSet(bp3, _Number_10)))),
#         MultiSet(q9, polynomial(MultiSet(MultiSet(bpp3, _Number_8)))),
#     )
# )
# replace_with = polynomial(
#     MultiSet(
#         MultiSet(bp1, bpp4, q12, q1),
#         MultiSet(q7, bp3, _Number_10),
#         MultiSet(q9, bpp3, _Number_8),
#     )
# )
# map then flatmap of get polynomial


# _MultiSet_1 = MultiSet[Number]()
# res = polynomial(
#     MultiSet(_MultiSet_1, MultiSet(-bp4, bpp1, q12, q2), MultiSet(bp1, bpp4, q12, q2), MultiSet(bp4, bpp1, q11, q3))
# ) - monomial(MultiSet(q3, monomial(MultiSet(q11, monomial(MultiSet(bp1, bpp4))))))

# res = (
#     -bp4 * bpp1 * q12 * q2
#     + bp1 * bpp4 * q12 * q2
#     + bp4 * bpp1 * q11 * q3
#     - bp1 * bpp4 * q11 * q3
#     - bp4 * bpp2 * q12 * q5
#     + bp2 * bpp4 * q12 * q5
#     + bp2 * bpp1 * q3 * q5
#     - bp1 * bpp2 * q3 * q5
#     + bp4 * bpp2 * q11 * q6
#     - bp2 * bpp4 * q11 * q6
#         - bp2 * bpp1 * q2 * q6
#         + bp1 * bpp2 * q2 * q6
#         - bp4 * bpp3 * q12 * q8
#         + bp3 * bpp4 * q12 * q8
#         + bp3 * bpp1 * q3 * q8
#         - bp1 * bpp3 * q3 * q8
#         + bp3 * bpp2 * q6 * q8
#         - bp2 * bpp3 * q6 * q8
#         + bp4 * bpp3 * q11 * q9
#         - bp3 * bpp4 * q11 * q9
#         - bp3 * bpp1 * q2 * q9
#         + bp1 * bpp3 * q2 * q9
#         - bp3 * bpp2 * q5 * q9
#         + bp2 * bpp3 * q5 * q9
# )

from pathlib import Path

INITIAL_STR = "from egglog.exp.polynomials import *\n\n"

print("saving initial expression to initial.py")
Path("initial.py").write_text(INITIAL_STR + str(res))

egraph = EGraph(save_egglog_string=True)
res = egraph.let("res", res)
print("Running")
egraph.run(to_polynomial_ruleset.saturate())
print("Extracting")
finished = egraph.extract(res)
print("saving polynomial expression to polynomial.py")
Path("polynomial.py").write_text(INITIAL_STR + str(finished))


print("factoring")
egraph.run(factor_ruleset.saturate())
print("extracting factored")
finished_factored = egraph.extract(res)
print("saving factored expression to factored.py")
Path("factored.py").write_text(INITIAL_STR + str(finished_factored))

# print("simplifying factored")
# egraph.run(simplified_factor_ruleset.saturate())
# print("extracting simplified factored")
# finished_simplified = egraph.extract(res)
# print("saving simplified factored expression to simplified_factored.py")
# Path("simplified_factored.py").write_text(INITIAL_STR + str(finished_simplified))

Path("polynomials.egg").write_text(egraph.get_egglog_string)
