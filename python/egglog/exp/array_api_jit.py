import inspect
from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np

from egglog import EGraph, try_evaling
from egglog.exp.array_api import NDArray
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import EvalProgram, array_api_program_gen_schedule, ndarray_function_two_program

from .program_gen import Program

X = TypeVar("X", bound=Callable)


def jit(fn: X) -> X:
    """
    Jit compiles a function
    """
    egraph, res, res_optimized, program = function_to_program(fn, save_egglog_string=False)
    fn_program = EvalProgram(program, {"np": np})
    with egraph.set_current():
        fn = cast("X", try_evaling(array_api_program_gen_schedule, fn_program, fn_program.as_py_object))
    fn.initial_expr = res  # type: ignore[attr-defined]
    fn.expr = res_optimized  # type: ignore[attr-defined]
    return fn


def function_to_program(fn: Callable, save_egglog_string: bool) -> tuple[EGraph, NDArray, NDArray, Program]:
    sig = inspect.signature(fn)
    arg1, arg2 = sig.parameters.keys()
    egraph = EGraph(save_egglog_string=save_egglog_string)
    with egraph:
        with egraph.set_current():
            res = fn(NDArray.var(arg1), NDArray.var(arg2))
        res_optimized = egraph.simplify(res, array_api_numba_schedule)

    return (
        egraph,
        res,
        res_optimized,
        ndarray_function_two_program(res_optimized, NDArray.var(arg1), NDArray.var(arg2)),
    )
