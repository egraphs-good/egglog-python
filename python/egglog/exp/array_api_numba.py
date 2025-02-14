# mypy: disable-error-code="empty-body"
"""
Module for generating array api code that works with Numba.
"""

from __future__ import annotations

from egglog import *
from egglog.exp.array_api import *

array_api_numba_ruleset = ruleset()
array_api_numba_schedule = (array_api_combined_ruleset | array_api_numba_ruleset).saturate()
# For these rules, we not only wanna rewrite, we also want to subsume the original expression,
# so that the rewritten one is used, even if the original one is simpler.


# Rewrite mean(x, <int>, <expand dims>) to use sum b/c numba cant do mean with axis
# https://github.com/numba/numba/issues/1269
@array_api_numba_ruleset.register
def _mean(y: NDArray, x: NDArray, i: Int):
    axis = OptionalIntOrTuple.some(IntOrTuple.int(i))
    res = sum(x, axis) / NDArray.scalar(Value.int(x.shape[i]))

    yield rewrite(mean(x, axis, FALSE), subsume=True).to(res)
    yield rewrite(mean(x, axis, TRUE), subsume=True).to(expand_dims(res, i))


# Rewrite std(x, <int>) to use mean and sum b/c numba cant do std with axis
@array_api_numba_ruleset.register
def _std(y: NDArray, x: NDArray, i: Int):
    axis = OptionalIntOrTuple.some(IntOrTuple.int(i))
    # https://numpy.org/doc/stable/reference/generated/numpy.std.html
    # "std = sqrt(mean(x)), where x = abs(a - a.mean())**2."
    yield rewrite(
        std(x, axis),
        subsume=True,
    ).to(
        sqrt(mean(square(x - mean(x, axis, keepdims=TRUE)), axis)),
    )


# rewrite unique_counts to count each value one by one, since numba doesn't support np.unique(..., return_counts=True)
@function(ruleset=array_api_numba_ruleset, subsume=True)
def count_values(x: NDArrayLike, values: TupleValueLike) -> TupleValue:
    """
    Returns a tuple of the count of each of the values in the array as values
    """
    x = cast(NDArray, x)
    values = cast(TupleValue, values)
    return TupleValue(values.length(), lambda i: sum(x == values[i]).to_value())


@array_api_numba_ruleset.register
def _unique_counts(x: NDArray, c: NDArray, tv: TupleValue, v: Value):
    return [
        # The unique counts are the count of all the unique values
        rewrite(unique_counts(x)[1], subsume=True).to(NDArray.vector(count_values(x, unique_values(x).to_values()))),
    ]


# do the same for unique_inverse
@array_api_numba_ruleset.register
def _unique_inverse(x: NDArray, i: Int):
    return [
        # Creating a mask array of when the unique inverse is a value is the same as a mask array for when the value is that index of the unique values
        rewrite(unique_inverse(x)[Int(1)] == NDArray.scalar(Value.int(i)), subsume=True).to(
            x == NDArray.scalar(unique_values(x).index((i,)))
        ),
    ]
