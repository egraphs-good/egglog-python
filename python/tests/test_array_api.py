# mypy: disable-error-code="empty-body"
import inspect
from collections.abc import Callable
from itertools import product
from pathlib import Path
from types import FunctionType

import numba
import pytest
from sklearn import config_context, datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from egglog.egraph import set_current_ruleset
from egglog.exp.array_api import *
from egglog.exp.array_api import NDArray
from egglog.exp.array_api_jit import function_to_program, jit
from egglog.exp.array_api_loopnest import *
from egglog.exp.array_api_numba import array_api_numba_schedule
from egglog.exp.array_api_program_gen import *
from egglog.exp.program_gen import EvalProgram, Program

some_shape = constant("some_shape", TupleInt)
some_dtype = constant("some_dtype", DType)
some_index = constant("some_index", TupleInt)
some_length = constant("some_length", Int)
some_value = constant("some_value", Value)
some_int_index = constant("some_int_index", Int)


@function(ruleset=array_api_ruleset)
def is_even(x: Int) -> Boolean:
    return x % 2 == 0


class TestTupleValue:
    def test_includes(self):
        x = TupleValue.EMPTY.append(Value.bool(FALSE))
        check_eq(x.contains(Value.bool(FALSE)), TRUE, array_api_schedule)
        check_eq(x.contains(Value.bool(TRUE)), FALSE, array_api_schedule)


class TestTupleInt:
    def test_conversion(self):
        @function
        def f(x: TupleIntLike) -> TupleInt: ...

        assert expr_parts(f((1, 2))) == expr_parts(f(TupleInt.from_vec(Vec[Int](Int(1), Int(2)))))

    def test_cons_to_vec(self):
        check_eq(
            TupleInt.EMPTY.append(2),
            TupleInt.from_vec(Vec(Int(2))),
            array_api_schedule,
            add_second=False,
        )

    def test_vec_to_cons(self):
        check_eq(
            TupleInt.from_vec(Vec(Int(1), Int(2))),
            TupleInt.EMPTY.append(1).append(2),
            array_api_schedule,
            add_second=False,
        )

    def test_indexing_cons(self):
        check_eq(TupleInt.EMPTY.append(1).append(2)[Int(0)], Int(1), array_api_schedule)
        check_eq(TupleInt.EMPTY.append(1).append(2)[Int(1)], Int(2), array_api_schedule)

    def test_length_cons(self):
        check_eq(TupleInt.EMPTY.append(1).append(2).length(), Int(2), array_api_schedule)

    def test_fn_to_cons(self):
        check_eq(TupleInt(2, lambda i: i), TupleInt.EMPTY.append(0).append(1), array_api_schedule, add_second=False)

    def test_range_length(self):
        check_eq(TupleInt.range(some_length).length(), some_length, array_api_schedule)

    def test_range_index(self):
        check_eq(
            TupleInt.range(some_length)[some_int_index], check_index(some_length, some_int_index), array_api_schedule
        )

    def test_not_contains_example(self):
        check_eq(TupleInt.from_vec(Vec(Int(0), Int(1))).contains(Int(3)), FALSE, array_api_schedule)

    def test_contains_example(self):
        check_eq(TupleInt.from_vec(Vec(Int(0), Int(3))).contains(Int(3)), TRUE, array_api_schedule)

    def test_filter_append(self):
        check_eq(
            TupleInt.EMPTY.append(1).append(2).filter(is_even),
            TupleInt.EMPTY.append(2),
            array_api_schedule,
            add_second=False,
        )

    def test_filter_range(self):
        check_eq(TupleInt.range(4).filter(is_even), TupleInt.from_vec(Vec(Int(0), Int(2))), array_api_schedule)

    def test_filter_lambda_length(self):
        with set_current_ruleset(array_api_ruleset):
            x = TupleInt.range(5).filter(lambda i: i < 2).length()
        check_eq(x, Int(2), array_api_schedule)


@function
def some_array_idx_fn(x: TupleInt) -> Value: ...


class TestNDArray:
    def test_index(self):
        x = NDArray(some_shape, some_dtype, some_array_idx_fn)
        check_eq(x.index(some_index), some_array_idx_fn(some_index), array_api_schedule)

    def test_shape(self):
        x = NDArray(some_shape, some_dtype, some_array_idx_fn)
        check_eq(x.shape, some_shape, array_api_schedule)

    def test_simplify_any_unique(self):
        res = (
            any(
                (
                    astype(unique_counts(NDArray.var("X"))[Int(1)], DType.float64)
                    / NDArray.scalar(Value.float(Float(150.0)))
                )
                < NDArray.scalar(Value.int(Int(0)))
            )
            .to_value()
            .to_bool
        )
        check_eq(res, FALSE, array_api_schedule)

    def test_reshape_index(self):
        # Verify that it doesn't expand forever
        x = NDArray.var("x")
        new_shape = TupleInt.single(Int(-1))
        res = reshape(x, new_shape).index(TupleInt.single(Int(1)) + TupleInt.single(Int(2)))
        egraph = EGraph()
        egraph.register(res)
        egraph.run(array_api_schedule)
        equiv_expr = egraph.extract_multiple(res, 10)
        assert len(equiv_expr) < 10

    def test_reshape_vec_noop(self):
        x = NDArray.var("x")
        assume_shape(x, TupleInt.single(Int(5)))
        res = reshape(x, TupleInt.single(Int(-1)))
        egraph = EGraph()
        egraph.register(res)
        egraph.run(array_api_schedule)
        equiv_expr = egraph.extract_multiple(res, 10)

        assert len(equiv_expr) == 2
        egraph.check(eq(res).to(x))


@function
def some_tuple_tuple_int_idx_fn(x: Int) -> TupleInt: ...


@function
def some_tuple_tuple_int_reduce_value_fn(carry: Value, x: TupleInt) -> Value: ...


class TestTupleTupleInt:
    def test_reduce_value_zero(self):
        x = TupleTupleInt(0, some_tuple_tuple_int_idx_fn)
        check_eq(x.foldl_value(some_tuple_tuple_int_reduce_value_fn, some_value), some_value, array_api_schedule)

    def test_reduce_value_one(self):
        x = TupleTupleInt(1, some_tuple_tuple_int_idx_fn)
        check_eq(
            x.foldl_value(some_tuple_tuple_int_reduce_value_fn, some_value),
            some_tuple_tuple_int_reduce_value_fn(some_value, some_tuple_tuple_int_idx_fn(Int(0))),
            array_api_schedule,
        )

    def test_product_example(self):
        """
        From Python docs:

        product('ABCD', 'xy') → Ax Ay Bx By Cx Cy Dx Dy

        aka product((0, 1, 2, 3), (4, 5)) ==
        """
        # TODO: Increase size, but for now check doesnt terminate at larger sizes for some reason
        # input = ((0, 1, 2, 3), (4, 5))
        input = ((0, 1), (4, 5))
        expected_output = tuple(product(*input))
        check_eq(
            convert(input, TupleTupleInt).product(),
            convert(expected_output, TupleTupleInt),
            add_second=False,
            schedule=array_api_schedule,
        )


@function(ruleset=array_api_ruleset, subsume=True)
def linalg_norm(X: NDArray, axis: TupleIntLike) -> NDArray:
    # peel off the outer shape for result array
    outshape = ShapeAPI(X.shape).deselect(axis).to_tuple()
    # get only the inner shape for reduction
    reduce_axis = ShapeAPI(X.shape).select(axis).to_tuple()

    return NDArray(
        outshape,
        X.dtype,
        lambda k: LoopNestAPI.from_tuple(reduce_axis)
        .unwrap()
        .indices()
        .foldl_value(lambda carry, i: carry + ((x := X.index(i + k)).conj() * x).real(), init=0.0)
        .sqrt(),
    )


@function(ruleset=array_api_ruleset, subsume=True)
def linalg_norm_v2(X: NDArrayLike, axis: TupleIntLike) -> NDArray:
    X = cast(NDArray, X)
    return NDArray(
        X.shape.deselect(axis),
        X.dtype,
        lambda k: ndindex(X.shape.select(axis))
        .foldl_value(lambda carry, i: carry + ((x := X.index(i + k)).conj() * x).real(), init=0.0)
        .sqrt(),
    )


def linalg_val(X: NDArray, linalg_fn: Callable[[NDArray, TupleIntLike], NDArray]) -> NDArray:
    assume_shape(X, (3, 2, 3, 4))
    return linalg_fn(X, (0, 1))


class TestLoopNest:
    @pytest.mark.parametrize("linalg_fn", [linalg_norm, linalg_norm_v2])
    def test_shape(self, linalg_fn):
        X = np.random.random((3, 2, 3, 4))
        expect = np.linalg.norm(X, axis=(0, 1))
        assert expect.shape == (3, 4)

        check_eq(linalg_val(constant("X", NDArray), linalg_fn).shape, TupleInt.from_vec((3, 4)), array_api_schedule)

    @pytest.mark.parametrize("linalg_fn", [linalg_norm, linalg_norm_v2])
    def test_abstract_index(self, linalg_fn):
        i = constant("i", Int)
        j = constant("j", Int)
        X = constant("X", NDArray)
        idxed = linalg_val(X, linalg_fn).index((i, j))

        _Value_1 = X.index(TupleInt.from_vec(Vec[Int](Int(0), Int(0), i, j)))
        _Value_2 = X.index(TupleInt.from_vec(Vec[Int](Int(0), Int(1), i, j)))
        _Value_3 = X.index(TupleInt.from_vec(Vec[Int](Int(1), Int(0), i, j)))
        _Value_4 = X.index(TupleInt.from_vec(Vec[Int](Int(1), Int(1), i, j)))
        _Value_5 = X.index(TupleInt.from_vec(Vec[Int](Int(2), Int(0), i, j)))
        _Value_6 = X.index(TupleInt.from_vec(Vec[Int](Int(2), Int(1), i, j)))
        res = (
            (
                (
                    (
                        ((_Value_1.conj() * _Value_1).real() + (_Value_2.conj() * _Value_2).real())
                        + (_Value_3.conj() * _Value_3).real()
                    )
                    + (_Value_4.conj() * _Value_4).real()
                )
                + (_Value_5.conj() * _Value_5).real()
            )
            + (_Value_6.conj() * _Value_6).real()
        ).sqrt()
        check_eq(idxed, res, array_api_schedule)

    def test_index_codegen(self, snapshot_py):
        X = NDArray.var("X")
        i = Int.var("i")
        j = Int.var("j")
        idxed = linalg_val(X, linalg_norm_v2).index((i, j))
        simplified_index = simplify(idxed, array_api_schedule)
        assert str(simplified_index) == snapshot_py(name="expr")

        res = EvalProgram(
            value_program(simplified_index).function_three(ndarray_program(X), int_program(i), int_program(j)),
            {"np": np},
        )
        fn = cast(FunctionType, try_evaling(EGraph(), array_api_program_gen_schedule, res, res.as_py_object))

        assert inspect.getsource(fn) == snapshot_py(name="code")

        X = np.random.random((3, 2, 3, 4))
        expect = np.linalg.norm(X, axis=(0, 1))

        for idxs in np.ndindex(*expect.shape):
            assert np.allclose(fn(X, *idxs), expect[idxs], rtol=1e-03)


# This test happens in different steps. Each will be benchmarked and saved as a snapshot.
# The next step will load the old snapshot and run their test on it.


def run_lda(x, y):
    with config_context(array_api_dispatch=True):
        lda = LinearDiscriminantAnalysis(n_components=2)
        return lda.fit_transform(x, y)


iris = datasets.load_iris()
X_np, y_np = (iris.data, iris.target)


@pytest.mark.parametrize(
    "program",
    [
        pytest.param(tuple_value_program((1, 2)), id="tuple"),
    ],
)
def test_program_compile(program: Program, snapshot_py):
    # simplify first to do any pre-conversion
    egraph = EGraph()
    egraph.register(program)
    egraph.run(array_api_numba_schedule)
    simplified_program = egraph.extract(program)
    assert str(simplified_program) == snapshot_py(name="expr")
    egraph = EGraph()
    egraph.register(simplified_program.compile())
    egraph.run(array_api_program_gen_schedule)
    statements = egraph.extract(simplified_program.statements).eval()
    expr = egraph.extract(simplified_program.expr).eval()
    assert "\n".join([*statements.split("\n"), expr]) == snapshot_py(name="code")


def lda(X: NDArray, y: NDArray):
    assume_dtype(X, X_np.dtype)
    assume_shape(X, X_np.shape)
    assume_isfinite(X)

    assume_dtype(y, y_np.dtype)
    assume_shape(y, y_np.shape)
    assume_value_one_of(y, tuple(map(int, np.unique(y_np))))
    return run_lda(X, y)


@pytest.mark.parametrize(
    "program",
    [
        pytest.param(lambda x, y: x + y, id="add"),
        pytest.param(lambda x, y: x[(x.shape + TupleInt.from_vec((1, 2)))[100]], id="tuple"),
        pytest.param(lda, id="lda"),
    ],
)
def test_jit(program, snapshot_py, benchmark):
    jitted = benchmark(jit, program)
    assert str(jitted.initial_expr) == snapshot_py(name="initial_expr")
    assert str(jitted.expr) == snapshot_py(name="expr")
    assert inspect.getsource(jitted) == snapshot_py(name="code")


@pytest.mark.parametrize(
    "fn_thunk",
    [
        pytest.param(lambda: LinearDiscriminantAnalysis(n_components=2).fit_transform, id="base"),
        pytest.param(lambda: run_lda, id="array_api"),
        pytest.param(lambda: jit(lda), id="array_api-optimized"),
        pytest.param(lambda: numba.njit(jit(lda)), id="array_api-optimized-numba"),
    ],
)
def test_run_lda(fn_thunk, benchmark):
    fn = fn_thunk()
    # warmup once for numba
    assert np.allclose(run_lda(X_np, y_np), fn(X_np, y_np), rtol=1e-03)
    benchmark(fn, X_np, y_np)


# if calling as script, print out egglog source for test
# similar to jit, but don't include pyobject parts so it works in vanilla egglog
if __name__ == "__main__":
    print("Generating egglog source for test")
    egraph, _, _, program = function_to_program(lda, True)
    egraph.register(program.compile())
    try_evaling(egraph, array_api_program_gen_combined_ruleset.saturate(), program, program.statements)
    name = "python.egg"
    print("Saving to", name)
    Path(name).write_text(egraph.as_egglog_string)
