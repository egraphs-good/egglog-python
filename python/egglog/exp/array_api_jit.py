import inspect
from collections.abc import Callable
from typing import TypeVar, cast

from egglog import EGraph, try_evaling
from egglog.exp.array_api import NDArray
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import array_api_program_gen_schedule, ndarray_function_two

X = TypeVar("X", bound=Callable)


def jit(fn: X) -> X:
    """
    Jit compiles a function
    """
    sig = inspect.signature(fn)
    arg1, arg2 = sig.parameters.keys()
    egraph = EGraph()
    with egraph.set_current():
        res = fn(NDArray.var(arg1), NDArray.var(arg2))
    res_optimized = egraph.simplify(res, array_api_numba_schedule)

    fn_program = ndarray_function_two(res_optimized, NDArray.var(arg1), NDArray.var(arg2))
    fn = try_evaling(array_api_program_gen_schedule, fn_program, fn_program.as_py_object)
    fn.initial_expr = res  # type: ignore[attr-defined]
    fn.expr = res_optimized  # type: ignore[attr-defined]
    return cast(X, fn)
