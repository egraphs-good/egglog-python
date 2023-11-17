import inspect
from collections.abc import Callable
from typing import TypeVar, cast

from egglog import EGraph
from egglog.exp.array_api import NDArray
from egglog.exp.array_api_numba import array_api_numba_module
from egglog.exp.array_api_program_gen import array_api_module_string, ndarray_function_two

X = TypeVar("X", bound=Callable)


def jit(fn: X) -> X:
    """
    Jit compiles a function
    """
    from IPython.display import SVG

    # 1. Create variables for each of the two args in the functions
    sig = inspect.signature(fn)
    arg1, arg2 = sig.parameters.keys()

    with EGraph([array_api_numba_module]) as egraph:
        res = fn(NDArray.var(arg1), NDArray.var(arg2))
        egraph.register(res)
        egraph.run(10000)
        res_optimized = egraph.extract(res)
        svg = SVG(egraph.graphviz_svg(split_primitive_outputs=True, n_inline_leaves=3))

    egraph = EGraph([array_api_module_string])
    fn_program = ndarray_function_two(res_optimized, NDArray.var(arg1), NDArray.var(arg2))
    egraph.register(fn_program)
    egraph.run(10000)
    fn = cast(X, egraph.eval(fn_program.py_object))
    fn.egraph = svg  # type: ignore[attr-defined]
    fn.expr = res_optimized  # type: ignore[attr-defined]
    return fn
