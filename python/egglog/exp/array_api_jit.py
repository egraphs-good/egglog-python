import inspect
from collections.abc import Callable
from typing import TypeVar, cast

from egglog import EGraph
from egglog.exp.array_api import NDArray
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import array_api_program_gen_schedule, ndarray_function_two

X = TypeVar("X", bound=Callable)


def jit(fn: X) -> X:
    """
    Jit compiles a function
    """
    # 1. Create variables for each of the two args in the functions
    sig = inspect.signature(fn)
    arg1, arg2 = sig.parameters.keys()
    egraph = EGraph()
    with egraph:
        res = fn(NDArray.var(arg1), NDArray.var(arg2))
        egraph.register(res)
        egraph.run(array_api_numba_schedule)
        res_optimized = egraph.extract(res)
        # egraph.display(split_primitive_outputs=True, n_inline_leaves=3)

    fn_program = ndarray_function_two(res_optimized, NDArray.var(arg1), NDArray.var(arg2))
    egraph.register(fn_program)
    egraph.run(array_api_program_gen_schedule)
    # egraph.display(split_primitive_outputs=True, n_inline_leaves=3)
    try:
        fn = cast(X, egraph.eval(egraph.extract(fn_program.py_object)))
    except Exception as err:
        err.add_note(f"Failed to compile the program into a string: \n\n{egraph.extract(fn_program)}")
        egraph.display()
        raise
    fn.expr = res_optimized  # type: ignore[attr-defined]
    fn.statements = egraph.eval(fn_program.statements)  # type: ignore[attr-defined]
    return fn
