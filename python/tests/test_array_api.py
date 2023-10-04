from egglog.exp.array_api import *
from egglog.exp.array_api_program_gen import *


def test_simplify_any_unique():
    X = NDArray.var("X")
    res = any(
        (astype(unique_counts(X)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0))))
        < NDArray.scalar(Value.int(Int(0)))
    ).to_bool()

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
    import numpy

    _NDArray_1 = NDArray.var("X")
    X_orig = copy(_NDArray_1)
    assume_dtype(_NDArray_1, DType.float64)
    assume_shape(_NDArray_1, TupleInt(Int(150)) + TupleInt(Int(4)))

    _NDArray_2 = NDArray.var("y")
    Y_orig = copy(_NDArray_2)

    assume_dtype(_NDArray_2, int64)
    assume_shape(_NDArray_2, TupleInt(150))  # type: ignore
    assume_value_one_of(_NDArray_2, (0, 1, 2))  # type: ignore

    _NDArray_3 = reshape(_NDArray_2, TupleInt(Int(-1)))
    _NDArray_4 = astype(unique_counts(_NDArray_3)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))
    _NDArray_5 = zeros(
        unique_values(_NDArray_3).shape + TupleInt(Int(4)),
        OptionalDType.some(DType.float64),
        OptionalDevice.some(_NDArray_1.device),
    )
    res = _NDArray_3 + _NDArray_5 + _NDArray_4
    egraph = EGraph([array_api_module_string])
    with egraph:
        egraph.register(res)
        egraph.run(10000)
        res = egraph.extract(res)
    fn = ndarray_program(res).function_two(ndarray_program(X_orig), ndarray_program(Y_orig))
    with egraph:
        egraph.register(fn.eval_py_object(egraph.save_object({"np": numpy})))
        egraph.run(10000)
        fn = egraph.extract(fn)
        # egraph.display(n_inline_leaves=0, split_primitive_outputs=True)
        fn_source = egraph.load_object(egraph.extract(PyObject.from_string(fn.statements)))
    assert fn_source == snapshot_py


# @pytest.mark.xfail(raises=TODO)
def test_sklearn_lda(snapshot_py):
    from sklearn import config_context
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X_arr = NDArray.var("X")
    assume_dtype(X_arr, float64)
    assume_shape(X_arr, TupleInt(150) + TupleInt(4))  # type: ignore
    assume_isfinite(X_arr)

    y_arr = NDArray.var("y") d
    assume_dtype(y_arr, int64)
    assume_shape(y_arr, TupleInt(150))  # type: ignore
    assume_value_one_of(y_arr, (0, 1, 2))  # type: ignore

    with EGraph([array_api_module]):
        with config_context(array_api_dispatch=True):
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_r2 = lda.fit(X_arr, y_arr).transform(X_arr)

    with EGraph([array_api_module]) as egraph:
        egraph.register(X_r2)
        egraph.run((run() * 10).saturate())
        # egraph.graphviz(n_inline_leaves=3).render("3", view=True)

        res = egraph.extract(X_r2)
        assert str(res) == snapshot_py
        # egraph.display()


def test_reshape_index():
    # Verify that it doesn't expand forever
    x = NDArray.var("x")
    new_shape = TupleInt(Int(-1))
    res = reshape(x, new_shape).index(TupleInt(Int(1)) + TupleInt(Int(2)))
    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run(run() * 10)
    equiv_expr = egraph.extract_multiple(res, 10)
    assert len(equiv_expr) == 2
