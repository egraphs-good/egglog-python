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
        def generic(self, args, kws):
            [lhs, rhs] = args
            if isinstance(lhs, types.DType) and isinstance(rhs, types.DType):
                return signature(types.boolean, lhs, rhs)

    @lower_builtin(operator.eq, types.DType, types.DType)
    def const_eq_impl(context, builder, sig, args):
        arg1, arg2 = sig.args
        val = 1 if arg1 == arg2 else 0
        res = ir.Constant(ir.IntType(1), val)
        return impl_ret_untracked(context, builder, sig.return_type, res)
