# import pytest
from egglog.exp.array_api import *
from egglog.exp.array_api_numba import array_api_numba_module
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

    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run((run() * 20).saturate())
    egraph.check(eq(res).to(FALSE))


def test_tuple_value_includes():
    x = TupleValue(Value.bool(FALSE))
    should_be_true = x.includes(Value.bool(FALSE))
    should_be_false = x.includes(Value.bool(TRUE))
    egraph = EGraph([array_api_module])
    egraph.register(should_be_true)
    egraph.register(should_be_false)
    egraph.run((run() * 10).saturate())
    egraph.check(eq(should_be_true).to(TRUE))
    egraph.check(eq(should_be_false).to(FALSE))


def test_to_source(snapshot_py):
    _NDArray_1 = NDArray.var("X")
    X_orig = copy(_NDArray_1)
    assume_dtype(_NDArray_1, DType.float64)
    assume_shape(_NDArray_1, TupleInt(Int(150)) + TupleInt(Int(4)))
    assume_isfinite(_NDArray_1)

    _NDArray_2 = NDArray.var("y")
    Y_orig = copy(_NDArray_2)

    assume_dtype(_NDArray_2, int64)
    assume_shape(_NDArray_2, TupleInt(Int(150)))
    assume_value_one_of(
        _NDArray_2, TupleValue(Value.int(Int(0))) + (TupleValue(Value.int(Int(1))) + TupleValue(Value.int(Int(2))))
    )
    _NDArray_3 = astype(
        NDArray.vector(
            TupleValue(sum(_NDArray_2 == NDArray.scalar(Value.int(Int(0)))).to_value())
            + (
                TupleValue(sum(_NDArray_2 == NDArray.scalar(Value.int(Int(1)))).to_value())
                + TupleValue(sum(_NDArray_2 == NDArray.scalar(Value.int(Int(2)))).to_value())
            )
        ),
        DType.float64,
    ) / NDArray.scalar(Value.float(Float(150.0)))
    _NDArray_4 = zeros(
        TupleInt(Int(3)) + TupleInt(Int(4)), OptionalDType.some(DType.float64), OptionalDevice.some(_NDArray_1.device)
    )
    _MultiAxisIndexKey_1 = MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice()))
    _IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(0))) + _MultiAxisIndexKey_1)
    _NDArray_5 = _NDArray_1[ndarray_index(_NDArray_2 == NDArray.scalar(Value.int(Int(0))))]
    _OptionalIntOrTuple_1 = OptionalIntOrTuple.some(IntOrTuple.int(Int(0)))
    _NDArray_4[_IndexKey_1] = sum(_NDArray_5, _OptionalIntOrTuple_1) / NDArray.scalar(
        Value.int(_NDArray_5.shape[Int(0)])
    )
    _IndexKey_2 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(1))) + _MultiAxisIndexKey_1)
    _NDArray_6 = _NDArray_1[ndarray_index(_NDArray_2 == NDArray.scalar(Value.int(Int(1))))]
    _NDArray_4[_IndexKey_2] = sum(_NDArray_6, _OptionalIntOrTuple_1) / NDArray.scalar(
        Value.int(_NDArray_6.shape[Int(0)])
    )
    _IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(2))) + _MultiAxisIndexKey_1)
    _NDArray_7 = _NDArray_1[ndarray_index(_NDArray_2 == NDArray.scalar(Value.int(Int(2))))]
    _NDArray_4[_IndexKey_3] = sum(_NDArray_7, _OptionalIntOrTuple_1) / NDArray.scalar(
        Value.int(_NDArray_7.shape[Int(0)])
    )
    _NDArray_8 = concat(
        TupleNDArray(_NDArray_5 - _NDArray_4[_IndexKey_1])
        + (TupleNDArray(_NDArray_6 - _NDArray_4[_IndexKey_2]) + TupleNDArray(_NDArray_7 - _NDArray_4[_IndexKey_3])),
        OptionalInt.some(Int(0)),
    )
    _NDArray_9 = square(
        _NDArray_8
        - expand_dims(sum(_NDArray_8, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_8.shape[Int(0)])))
    )
    _NDArray_10 = sqrt(sum(_NDArray_9, _OptionalIntOrTuple_1) / NDArray.scalar(Value.int(_NDArray_9.shape[Int(0)])))
    _NDArray_11 = copy(_NDArray_10)
    _NDArray_11[ndarray_index(_NDArray_10 == NDArray.scalar(Value.int(Int(0))))] = NDArray.scalar(
        Value.float(Float(1.0))
    )
    _TupleNDArray_1 = svd(
        sqrt(NDArray.scalar(Value.float(Float.rational(Rational(1, 147))))) * (_NDArray_8 / _NDArray_11), FALSE
    )
    _Slice_1 = Slice(
        OptionalInt.none,
        OptionalInt.some(
            astype(sum(_TupleNDArray_1[Int(1)] > NDArray.scalar(Value.float(Float(0.0001)))), DType.int32)
            .to_value()
            .to_int
        ),
    )
    _NDArray_12 = (
        _TupleNDArray_1[Int(2)][
            IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(_Slice_1)) + _MultiAxisIndexKey_1)
        ]
        / _NDArray_11
    ).T / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
    _TupleNDArray_2 = svd(
        (
            sqrt(
                (NDArray.scalar(Value.int(Int(150))) * _NDArray_3)
                * NDArray.scalar(Value.float(Float.rational(Rational(1, 2))))
            )
            * (_NDArray_4 - (_NDArray_3 @ _NDArray_4)).T
        ).T
        @ _NDArray_12,
        FALSE,
    )
    res = (
        (_NDArray_1 - (_NDArray_3 @ _NDArray_4))
        @ (
            _NDArray_12
            @ _TupleNDArray_2[Int(2)].T[
                IndexKey.multi_axis(
                    _MultiAxisIndexKey_1
                    + MultiAxisIndexKey(
                        MultiAxisIndexKeyItem.slice(
                            Slice(
                                OptionalInt.none,
                                OptionalInt.some(
                                    astype(
                                        sum(
                                            _TupleNDArray_2[Int(1)]
                                            > (
                                                NDArray.scalar(Value.float(Float(0.0001)))
                                                * _TupleNDArray_2[Int(1)][IndexKey.int(Int(0))]
                                            )
                                        ),
                                        DType.int32,
                                    )
                                    .to_value()
                                    .to_int
                                ),
                            )
                        )
                    )
                )
            ]
        )
    )[
        IndexKey.multi_axis(
            _MultiAxisIndexKey_1
            + MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice(OptionalInt.none, OptionalInt.some(Int(2)))))
        )
    ]
    egraph = EGraph([array_api_numba_module])
    egraph.register(res)
    egraph.run(100000)
    res = egraph.extract(res)

    egraph = EGraph([array_api_module_string])
    fn = ndarray_function_two(res, X_orig, Y_orig)
    # with egraph:
    egraph.register(fn)
    egraph.run(1000000)
    # new_fn = egraph.extract(fn)
    # egraph.register(new_fn)
    # egraph.display(n_inline_leaves=2, split_primitive_outputs=True)
    fn_source = egraph.eval(fn.statements)
    assert fn_source == snapshot_py
    fn = egraph.eval(fn.py_object)


def run_lda(x, y):
    with config_context(array_api_dispatch=True):
        lda = LinearDiscriminantAnalysis(n_components=2)
        return lda.fit(x, y).transform(x)


# @pytest.mark.xfail(raises=AssertionError)
def test_sklearn_lda(snapshot_py):
    X_arr = NDArray.var("X")
    assume_dtype(X_arr, float64)
    assume_shape(X_arr, TupleInt(150) + TupleInt(4))  # type: ignore
    assume_isfinite(X_arr)

    y_arr = NDArray.var("y")
    assume_dtype(y_arr, int64)
    assume_shape(y_arr, TupleInt(150))  # type: ignore
    assume_value_one_of(y_arr, (0, 1, 2))  # type: ignore

    with EGraph([array_api_module]):
        X_r2 = run_lda(X_arr, y_arr)
    assert str(X_r2) == snapshot_py(name="original")

    with EGraph([array_api_numba_module]) as egraph:
        egraph.register(X_r2)
        egraph.run(100000)
        # egraph.graphviz(n_inline_leaves=3).render("3", view=True)

        res = egraph.extract(X_r2)
    assert str(res) == snapshot_py(name="optimized")
    # egraph.display()


def test_sklearn_lda_runs():
    X_arr = NDArray.var("X")
    X_orig = copy(X_arr)

    assume_dtype(X_arr, float64)
    assume_shape(X_arr, TupleInt(150) + TupleInt(4))  # type: ignore
    assume_isfinite(X_arr)

    y_arr = NDArray.var("y")
    y_orig = copy(y_arr)

    assume_dtype(y_arr, int64)
    assume_shape(y_arr, TupleInt(150))  # type: ignore
    assume_value_one_of(y_arr, (0, 1, 2))  # type: ignore

    with EGraph([array_api_module]) as egraph:
        X_r2 = run_lda(X_arr, y_arr)

    egraph = EGraph([array_api_numba_module])
    egraph.register(X_r2)
    egraph.run((run() * 10).saturate())
    X_r2 = egraph.extract(X_r2)

    egraph = EGraph([array_api_module_string])

    fn_program = ndarray_function_two(X_r2, X_orig, y_orig)
    egraph.register(fn_program)
    egraph.run(10000)
    fn = egraph.eval(fn_program.py_object)
    iris = datasets.load_iris()

    X_np, y_np = (iris.data, iris.target)
    real_res = run_lda(X_np, y_np)
    optimized_res = fn(X_np, y_np)  # type: ignore
    assert np.allclose(real_res, optimized_res)

    # Numba isn't supported on all platforms, so only test this if we can import
    try:
        import numba
    except ImportError:
        pass
    else:
        numba_res = numba.njit(fn)(X_np, y_np)
        assert np.allclose(real_res, numba_res)


def test_reshape_index():
    # Verify that it doesn't expand forever
    x = NDArray.var("x")
    new_shape = TupleInt(Int(-1))
    res = reshape(x, new_shape).index(TupleInt(Int(1)) + TupleInt(Int(2)))
    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run(run() * 10)
    equiv_expr = egraph.extract_multiple(res, 10)
    assert len(equiv_expr) < 10


def test_reshape_vec_noop():
    x = NDArray.var("x")
    assume_shape(x, TupleInt(Int(5)))
    res = reshape(x, TupleInt(Int(-1)))
    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run(run() * 10)
    equiv_expr = egraph.extract_multiple(res, 10)

    assert len(equiv_expr) == 2
    egraph.check(eq(res).to(x))


# def test_reshape_transform_index():
#     x = NDArray.var("x")
#     assume_shape(x, TupleInt(Int(5)))
#     res: Value = reshape(x, TupleInt(Int(-1))).index(ALL_INDICES)
#     egraph = EGraph([array_api_module])
#     egraph.register(res)
#     egraph.run(20)
#     egraph.check(eq(res).to(x.index(reshape_transform_index(TupleInt(Int(5)), TupleInt(Int(-1)), ALL_INDICES))))
#     # egraph.display()
#     # Verify that this doesn't blow up
#     assert len(egraph.extract_multiple(res, 100)) < 10
