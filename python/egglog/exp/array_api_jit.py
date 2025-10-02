import inspect
from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np

from egglog import EGraph, greedy_dag_cost_model
from egglog.exp.array_api import NDArray, set_array_api_egraph, try_evaling
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import EvalProgram, array_api_program_gen_schedule, ndarray_function_two_program

from .program_gen import Program

X = TypeVar("X", bound=Callable)


def jit(
    fn: X,
    *,
    handle_expr: Callable[[NDArray], None] | None = None,
    handle_optimized_expr: Callable[[NDArray], None] | None = None,
) -> X:
    """
    Jit compiles a function
    """
    egraph, res, res_optimized, program = function_to_program(fn, save_egglog_string=False)
    if handle_expr:
        handle_expr(res)
    if handle_optimized_expr:
        handle_optimized_expr(res_optimized)
    fn_program = EvalProgram(program, {"np": np})
    return cast("X", try_evaling(egraph, array_api_program_gen_schedule, fn_program, fn_program.as_py_object))


def function_to_program(fn: Callable, save_egglog_string: bool) -> tuple[EGraph, NDArray, NDArray, Program]:
    sig = inspect.signature(fn)
    arg1, arg2 = sig.parameters.keys()
    egraph = EGraph(save_egglog_string=save_egglog_string)
    with egraph:
        with set_array_api_egraph(egraph):
            res = fn(NDArray.var(arg1), NDArray.var(arg2))
        egraph.register(res)
        egraph.run(array_api_numba_schedule)
        res_optimized = egraph.extract(res, cost_model=greedy_dag_cost_model())

    return (
        egraph,
        res,
        res_optimized,
        ndarray_function_two_program(res_optimized, NDArray.var(arg1), NDArray.var(arg2)),
    )
