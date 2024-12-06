import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numba
import pytest
from sklearn import config_context, datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from egglog.egraph import set_current_ruleset
from egglog.exp.array_api import *
from egglog.exp.array_api_jit import jit
from egglog.exp.array_api_loopnest import *
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import *


def test_simplify_any_unique():
    X = NDArray.var("X")
    res = (
        any(
            (astype(unique_counts(X)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0))))
            < NDArray.scalar(Value.int(Int(0)))
        )
        .to_value()
        .to_bool
    )

    egraph = EGraph()
    egraph.register(res)
    egraph.run(array_api_schedule)
    egraph.check(eq(res).to(FALSE))


def test_tuple_value_includes():
    x = TupleValue(Value.bool(FALSE))
    should_be_true = x.includes(Value.bool(FALSE))
    should_be_false = x.includes(Value.bool(TRUE))
    egraph = EGraph()
    egraph.register(should_be_true)
    egraph.register(should_be_false)
    egraph.run(array_api_schedule)
    egraph.check(eq(should_be_true).to(TRUE))
    egraph.check(eq(should_be_false).to(FALSE))


def test_reshape_index():
    # Verify that it doesn't expand forever
    x = NDArray.var("x")
    new_shape = TupleInt.single(Int(-1))
    res = reshape(x, new_shape).index(TupleInt.single(Int(1)) + TupleInt.single(Int(2)))
    egraph = EGraph()
    egraph.register(res)
    egraph.run(array_api_schedule)
    equiv_expr = egraph.extract_multiple(res, 10)
    assert len(equiv_expr) < 10


def test_reshape_vec_noop():
    x = NDArray.var("x")
    assume_shape(x, TupleInt.single(Int(5)))
    res = reshape(x, TupleInt.single(Int(-1)))
    egraph = EGraph()
    egraph.register(res)
    egraph.run(array_api_schedule)
    equiv_expr = egraph.extract_multiple(res, 10)

    assert len(equiv_expr) == 2
    egraph.check(eq(res).to(x))


def test_filter():
    with set_current_ruleset(array_api_ruleset):
        x = TupleInt.range(5).filter(lambda i: i < 2).length()
    check_eq(x, Int(2), array_api_schedule)


@function(ruleset=array_api_ruleset, subsume=True)
def linalg_norm(X: NDArray, axis: TupleIntLike) -> NDArray:
    # peel off the outer shape for result array
    outshape = ShapeAPI(X.shape).deselect(axis).to_tuple()
    # get only the inner shape for reduction
    reduce_axis = ShapeAPI(X.shape).select(axis).to_tuple()

    return NDArray(
        outshape,
        X.dtype,
        lambda k: sqrt(
            LoopNestAPI.from_tuple(reduce_axis)
            .unwrap()
            .fold(lambda carry, i: carry + real(conj(x := X[i + k]) * x), init=0.0)
        ).to_value(),
    )


class TestLoopNest:
    def test_shape(self):
        X = NDArray.var("X")
        assume_shape(X, (3, 2, 3, 4))
        val = linalg_norm(X, (0, 1))

        check_eq(val.shape.length(), Int(2), array_api_schedule)
        check_eq(val.shape[0], Int(3), array_api_schedule)
        check_eq(val.shape[1], Int(4), array_api_schedule)


# This test happens in different steps. Each will be benchmarked and saved as a snapshot.
# The next step will load the old snapshot and run their test on it.


def run_lda(x, y):
    with config_context(array_api_dispatch=True):
        lda = LinearDiscriminantAnalysis(n_components=2)
        return lda.fit(x, y).transform(x)


iris = datasets.load_iris()
X_np, y_np = (iris.data, iris.target)


def _load_py_snapshot(fn: Callable, var: str | None = None) -> Any:
    """
    Load a python snapshot, evaling the code, and returning the `var` defined in it.

    If no var is provided, then return the last expression.
    """
    path = Path(__file__).parent / "__snapshots__" / "test_array_api" / f"TestLDA.{fn.__name__}.py"
    contents = path.read_text()

    contents = "import numpy as np\nfrom egglog.exp.array_api import *\n" + contents
    globals: dict[str, Any] = {}
    if var is None:
        # exec once as a full statement
        exec(contents, globals)
        # Eval the last statement
        last_expr = ast.unparse(ast.parse(contents).body[-1])
        return eval(last_expr, globals)
    exec(contents, globals)
    return globals[var]


def load_source(fn_program: EvalProgram, egraph: EGraph):
    egraph.register(fn_program)
    egraph.run(array_api_program_gen_schedule)
    # dp the needed pieces in here for benchmarking
    return egraph.eval(egraph.extract(fn_program.py_object))


def lda(X, y):
    assume_dtype(X, X_np.dtype)
    assume_shape(X, X_np.shape)
    assume_isfinite(X)

    assume_dtype(y, y_np.dtype)
    assume_shape(y, y_np.shape)
    assume_value_one_of(y, tuple(map(int, np.unique(y_np))))  # type: ignore[arg-type]
    return run_lda(X, y)


def simplify_lda(egraph: EGraph, expr: NDArray) -> NDArray:
    egraph.register(expr)
    egraph.run(array_api_numba_schedule)
    return egraph.extract(expr)


@pytest.mark.benchmark(min_rounds=3)
class TestLDA:
    """
    Incrementally benchmark each part of the LDA to see how long it takes to run.
    """

    def test_trace(self, snapshot_py, benchmark):
        X = NDArray.var("X")
        y = NDArray.var("y")
        with EGraph():
            X_r2 = benchmark(lda, X, y)
        assert str(X_r2) == snapshot_py

    def test_optimize(self, snapshot_py, benchmark):
        egraph = EGraph()
        X = NDArray.var("X")
        y = NDArray.var("y")
        with egraph:
            expr = lda(X, y)
            simplified = benchmark(simplify_lda, egraph, expr)
        assert str(simplified) == snapshot_py

    # @pytest.mark.xfail(reason="Original source is not working")
    # def test_source(self, snapshot_py, benchmark):
    #     egraph = EGraph()
    #     expr = trace_lda(egraph)
    #     assert benchmark(load_source, expr, egraph) == snapshot_py

    def test_source_optimized(self, snapshot_py, benchmark):
        egraph = EGraph()
        X = NDArray.var("X")
        y = NDArray.var("y")
        with egraph:
            expr = lda(X, y)
            optimized_expr = simplify_lda(egraph, expr)
        fn_program = ndarray_function_two(optimized_expr, NDArray.var("X"), NDArray.var("y"))
        py_object = benchmark(load_source, fn_program, egraph)
        assert np.allclose(py_object(X_np, y_np), run_lda(X_np, y_np))
        assert egraph.eval(fn_program.statements) == snapshot_py

    @pytest.mark.parametrize(
        "fn",
        [
            pytest.param(LinearDiscriminantAnalysis(n_components=2).fit_transform, id="base"),
            pytest.param(run_lda, id="array_api"),
            pytest.param(_load_py_snapshot(test_source_optimized, "__fn"), id="array_api-optimized"),
            pytest.param(numba.njit(_load_py_snapshot(test_source_optimized, "__fn")), id="array_api-optimized-numba"),
            pytest.param(jit(lda), id="array_api-jit"),
        ],
    )
    def test_execution(self, fn, benchmark):
        # warmup once for numba
        assert np.allclose(run_lda(X_np, y_np), fn(X_np, y_np))
        benchmark(fn, X_np, y_np)


# if calling as script, print out egglog source for test
# similar to jit, but don't include pyobject parts so it works in vanilla egglog
if __name__ == "__main__":
    print("Generating egglog source for test")
    egraph = EGraph(save_egglog_string=True)
    X_ = NDArray.var("X")
    y_ = NDArray.var("y")
    with egraph:
        expr = lda(X_, y_)
    optimized_expr = egraph.simplify(expr, array_api_numba_schedule)
    fn_program = ndarray_function_two_program(optimized_expr, X_, y_)
    egraph.register(fn_program.compile())
    egraph.run(array_api_program_gen_ruleset.saturate() + program_gen_ruleset.saturate())
    egraph.extract(fn_program.statements)
    name = "python.egg"
    print("Saving to", name)
    Path(name).write_text(egraph.as_egglog_string)
