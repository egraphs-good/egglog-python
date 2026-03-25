import inspect
from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
from opentelemetry import trace

from egglog import EGraph, bindings, greedy_dag_cost_model
from egglog.exp.array_api import NDArray, set_array_api_egraph
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import EvalProgram, array_api_program_gen_schedule, ndarray_function_two_program

from .program_gen import Program

X = TypeVar("X", bound=Callable)
_TRACER = trace.get_tracer(__name__)


@_TRACER.start_as_current_span("jit")
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
    egraph = EGraph()
    if handle_expr:
        handle_expr(res)
    if handle_optimized_expr:
        handle_optimized_expr(res_optimized)
    fn_program = EvalProgram(program, {"np": np})
    egraph.register(fn_program)

    egraph.run(array_api_program_gen_schedule)

    try:
        return cast("X", egraph.extract(fn_program.as_py_object).value)
    except bindings.EggSmolError as e:
        try:
            debug_program = egraph.extract(fn_program)
        except bindings.EggSmolError:
            debug_program = fn_program
        e.add_note(f"Failed to get py object from {debug_program}")
        raise


@_TRACER.start_as_current_span("function_to_program")
def function_to_program(fn: Callable, save_egglog_string: bool) -> tuple[EGraph, NDArray, NDArray, Program]:
    sig = inspect.signature(fn)
    arg1, arg2 = sig.parameters.keys()
    egraph = EGraph(save_egglog_string=save_egglog_string)
    with egraph:
        with _TRACER.start_as_current_span("call_function"), set_array_api_egraph(egraph):
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
