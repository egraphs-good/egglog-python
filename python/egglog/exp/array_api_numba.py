# mypy: disable-error-code="empty-body"
"""
Module for generating array api code that works with Numba.
"""

from __future__ import annotations

import operator

from egglog import *
from egglog.exp.array_api import *

array_api_numba_module = Module([array_api_module])

# For these rules, we not only wanna rewrite, we also want to delete the original expression,
# so that the rewritten one is used, even if the original one is simpler.

# TODO: Try deleting instead if we support that in the future, and remove high cost
# https://egraphs.zulipchat.com/#narrow/stream/375765-egglog/topic/replacing.20an.20expression.20with.20delete


# Rewrite mean(x, <int>, <expand dims>) to use sum b/c numba cant do mean with axis
# https://github.com/numba/numba/issues/1269
@array_api_numba_module.register
def _mean(y: NDArray, x: NDArray, i: Int):
    axis = OptionalIntOrTuple.some(IntOrTuple.int(i))
    res = sum(x, axis) / NDArray.scalar(Value.int(x.shape[i]))

    yield rewrite(mean(x, axis, FALSE)).to(res)
    yield rewrite(mean(x, axis, TRUE)).to(expand_dims(res, i))


# Rewrite std(x, <int>) to use mean and sum b/c numba cant do std with axis
@array_api_numba_module.register
def _std(y: NDArray, x: NDArray, i: Int):
    axis = OptionalIntOrTuple.some(IntOrTuple.int(i))
    # https://numpy.org/doc/stable/reference/generated/numpy.std.html
    # "std = sqrt(mean(x)), where x = abs(a - a.mean())**2."
    yield rewrite(std(x, axis)).to(sqrt(mean(square(x - mean(x, axis, keepdims=TRUE)), axis)))


# rewrite unique_counts to count each value one by one, since numba doesn't support np.unique(..., return_counts=True)
@array_api_numba_module.function(unextractable=True)
def count_values(x: NDArray, values: NDArray) -> TupleValue:
    """
    Returns a tuple of the count of each of the values in the array.
    """
    ...


@array_api_numba_module.register
def _unique_counts(x: NDArray, c: NDArray, tv: TupleValue, v: Value):
    return [
        # The unique counts are the count of all the unique values
        rewrite(unique_counts(x)[Int(1)]).to(NDArray.vector(count_values(x, unique_values(x)))),
        rewrite(count_values(x, NDArray.vector(TupleValue(v) + tv))).to(
            TupleValue(sum(x == NDArray.scalar(v)).to_value()) + count_values(x, NDArray.vector(tv))
        ),
        rewrite(count_values(x, NDArray.vector(TupleValue(v)))).to(
            TupleValue(sum(x == NDArray.scalar(v)).to_value()),
        ),
    ]


# do the same for unique_inverse
@array_api_numba_module.register
def _unique_inverse(x: NDArray, i: Int):
    return [
        # Creating a mask array of when the unique inverse is a value is the same as a mask array for when the value is that index of the unique values
        rewrite(unique_inverse(x)[Int(1)] == NDArray.scalar(Value.int(i))).to(
            x == NDArray.scalar(unique_values(x).index(TupleInt(i)))
        ),
    ]


# Inline these changes until this PR is released to add suport for checking dtypes equal
# https://github.com/numba/numba/pull/9249
try:
    from llvmlite import ir
    from numba.core import types
    from numba.core.imputils import impl_ret_untracked, lower_builtin
    from numba.core.typing.templates import AbstractTemplate, infer_global, signature
except ImportError:
    pass
else:

    @infer_global(operator.eq)
    class DtypeEq(AbstractTemplate):
        def generic(self, args, kws):  # noqa: ANN201, ANN001
            [lhs, rhs] = args
            if isinstance(lhs, types.DType) and isinstance(rhs, types.DType):
                return signature(types.boolean, lhs, rhs)
            return None

    @lower_builtin(operator.eq, types.DType, types.DType)
    def const_eq_impl(context, builder, sig, args):  # noqa: ANN201, ANN001
        arg1, arg2 = sig.args
        val = 1 if arg1 == arg2 else 0
        res = ir.Constant(ir.IntType(1), val)
        return impl_ret_untracked(context, builder, sig.return_type, res)
