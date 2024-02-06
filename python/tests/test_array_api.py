import pytest

import numba
from pathlib import Path
from typing import Any, Callable, cast
import ast

from egglog.exp.array_api import *
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import *
from sklearn import config_context, datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    new_shape = TupleInt(Int(-1))
    res = reshape(x, new_shape).index(TupleInt(Int(1)) + TupleInt(Int(2)))
    egraph = EGraph()
    egraph.register(res)
    egraph.run(array_api_schedule)
    equiv_expr = egraph.extract_multiple(res, 10)
    assert len(equiv_expr) < 10


def test_reshape_vec_noop():
    x = NDArray.var("x")
    assume_shape(x, TupleInt(Int(5)))
    res = reshape(x, TupleInt(Int(-1)))
    egraph = EGraph()
    egraph.register(res)
    egraph.run(array_api_schedule)
    equiv_expr = egraph.extract_multiple(res, 10)

    assert len(equiv_expr) == 2
    egraph.check(eq(res).to(x))


# This test happens in different steps. Each will be benchmarked and saved as a snapshot.
# The next step will load the old snapshot and run their test on it.


def run_lda(x, y):
    with config_context(array_api_dispatch=True):
        lda = LinearDiscriminantAnalysis(n_components=2)
        return lda.fit(x, y).transform(x)


iris = datasets.load_iris()
X_np, y_np = (iris.data, iris.target)
res = run_lda(X_np, y_np)


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
    else:
        exec(contents, globals)
        return globals[var]


def load_source(expr):
    egraph = EGraph()
    fn_program = egraph.let("fn_program", ndarray_function_two(expr, NDArray.var("X"), NDArray.var("y")))
    egraph.run(array_api_program_gen_schedule)
    # cast b/c issue with it not recognizing py_object as property
    fn = cast(Any, egraph.eval(fn_program.py_object))
    assert np.allclose(res, run_lda(X_np, y_np))
    return egraph.eval(fn_program.statements)


@pytest.mark.benchmark(min_rounds=3)
class TestLDA:
    def test_trace(self, snapshot_py, benchmark):
        @benchmark
        def X_r2():
            X_arr = NDArray.var("X")
            assume_dtype(X_arr, X_np.dtype)
            assume_shape(X_arr, X_np.shape)  # type: ignore
            assume_isfinite(X_arr)

            y_arr = NDArray.var("y")
            assume_dtype(y_arr, y_np.dtype)
            assume_shape(y_arr, y_np.shape)  # type: ignore
            assume_value_one_of(y_arr, tuple(map(int, np.unique(y_np))))  # type: ignore

            with EGraph():
                return run_lda(X_arr, y_arr)

        assert str(X_r2) == snapshot_py

    def test_optimize(self, snapshot_py, benchmark):
        expr = _load_py_snapshot(self.test_trace)
        simplified = benchmark(lambda: EGraph().simplify(expr, array_api_numba_schedule))
        assert str(simplified) == snapshot_py

    @pytest.mark.xfail(reason="Original source is not working")
    def test_source(self, snapshot_py, benchmark):
        expr = _load_py_snapshot(self.test_trace)
        assert benchmark(load_source, expr) == snapshot_py

    def test_source_optimized(self, snapshot_py, benchmark):
        expr = _load_py_snapshot(self.test_optimize)
        assert benchmark(load_source, expr) == snapshot_py

    @pytest.mark.parametrize(
        ("fn",),
        [
            pytest.param(LinearDiscriminantAnalysis(n_components=2).fit_transform, id="base"),
            pytest.param(run_lda, id="array_api"),
            pytest.param(_load_py_snapshot(test_source_optimized, "__fn"), id="array_api-optimized"),
            pytest.param(
                numba.njit(_load_py_snapshot(test_source_optimized, "__fn")), id="array_api-optimized-numba"
            ),
        ],
    )
    def test_execution(self, fn, benchmark):
        # warmup once for numba
        assert np.allclose(res, fn(X_np, y_np))
        benchmark(fn, X_np, y_np)
