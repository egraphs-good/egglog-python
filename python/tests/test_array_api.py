import pytest
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
    assume_isfinite(_NDArray_1)

    _NDArray_2 = NDArray.var("y")
    Y_orig = copy(_NDArray_2)

    assume_dtype(_NDArray_2, int64)
    assume_shape(_NDArray_2, TupleInt(Int(150)))
    _TupleValue_1 = TupleValue(Value.int(Int(0))) + (TupleValue(Value.int(Int(1))) + TupleValue(Value.int(Int(2))))
    assume_value_one_of(_NDArray_2, _TupleValue_1)

    _NDArray_3 = reshape(_NDArray_2, TupleInt(Int(-1)))
    _NDArray_4 = astype(unique_counts(_NDArray_3)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))
    _NDArray_5 = zeros(
        TupleInt(Int(3)) + TupleInt(Int(4)), OptionalDType.some(DType.float64), OptionalDevice.some(_NDArray_1.device)
    )
    _MultiAxisIndexKey_1 = MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(Slice()))
    _IndexKey_1 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(0))) + _MultiAxisIndexKey_1)
    _NDArray_5[_IndexKey_1] = mean(
        _NDArray_1[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(0))))],
        OptionalIntOrTuple.int(Int(0)),
    )
    _IndexKey_2 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(1))) + _MultiAxisIndexKey_1)
    _NDArray_5[_IndexKey_2] = mean(
        _NDArray_1[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(1))))],
        OptionalIntOrTuple.int(Int(0)),
    )
    _IndexKey_3 = IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.int(Int(2))) + _MultiAxisIndexKey_1)
    _NDArray_5[_IndexKey_3] = mean(
        _NDArray_1[ndarray_index(unique_inverse(_NDArray_3)[Int(1)] == NDArray.scalar(Value.int(Int(2))))],
        OptionalIntOrTuple.int(Int(0)),
    )
    _NDArray_6 = concat(
        TupleNDArray(
            _NDArray_1[ndarray_index(_NDArray_3 == NDArray.vector(_TupleValue_1)[IndexKey.int(Int(0))])]
            - _NDArray_5[_IndexKey_1]
        )
        + (
            TupleNDArray(
                _NDArray_1[ndarray_index(_NDArray_3 == NDArray.vector(_TupleValue_1)[IndexKey.int(Int(1))])]
                - _NDArray_5[_IndexKey_2]
            )
            + TupleNDArray(
                _NDArray_1[ndarray_index(_NDArray_3 == NDArray.vector(_TupleValue_1)[IndexKey.int(Int(2))])]
                - _NDArray_5[_IndexKey_3]
            )
        ),
        OptionalInt.some(Int(0)),
    )
    _NDArray_7 = std(_NDArray_6, OptionalIntOrTuple.int(Int(0)))
    _NDArray_7[
        ndarray_index(std(_NDArray_6, OptionalIntOrTuple.int(Int(0))) == NDArray.scalar(Value.int(Int(0))))
    ] = NDArray.scalar(Value.float(Float(1.0)))
    _TupleNDArray_1 = svd(
        sqrt(NDArray.scalar(Value.int(NDArray.scalar(Value.float(Float(1.0))).to_int() / Int(147))))
        * (_NDArray_6 / _NDArray_7),
        FALSE,
    )
    _Slice_1 = Slice(
        OptionalInt.none,
        OptionalInt.some(
            astype(sum(_TupleNDArray_1[Int(1)] > NDArray.scalar(Value.float(Float(0.0001)))), DType.int32).to_int()
        ),
    )
    _NDArray_8 = (
        _TupleNDArray_1[Int(2)][
            IndexKey.multi_axis(MultiAxisIndexKey(MultiAxisIndexKeyItem.slice(_Slice_1)) + _MultiAxisIndexKey_1)
        ]
        / _NDArray_7
    ).T / _TupleNDArray_1[Int(1)][IndexKey.slice(_Slice_1)]
    _TupleNDArray_2 = svd(
        (
            sqrt(
                NDArray.scalar(
                    Value.int(
                        (Int(150) * _NDArray_4.to_int()) * (NDArray.scalar(Value.float(Float(1.0))).to_int() / Int(2))
                    )
                )
            )
            * (_NDArray_5 - (_NDArray_4 @ _NDArray_5)).T
        ).T
        @ _NDArray_8,
        FALSE,
    )

    res = (
        (_NDArray_1 - (_NDArray_4 @ _NDArray_5))
        @ (
            _NDArray_8
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
                                    ).to_int()
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
    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run(1000)
    res = egraph.extract(res)

    egraph = EGraph([array_api_module_string])
    fn = ndarray_program(res).function_two(ndarray_program(X_orig), ndarray_program(Y_orig))
    egraph.register(fn.eval_py_object(egraph.save_object({"np": numpy})))
    egraph.run(10000)
    fn_source = egraph.load_object(egraph.extract(PyObject.from_string(fn.statements)))
    assert fn_source == snapshot_py
    fn = egraph.load_object(egraph.extract(fn.py_object))


@pytest.mark.xfail(raises=AssertionError)
def test_sklearn_lda(snapshot_py):
    from sklearn import config_context
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X_arr = NDArray.var("X")
    assume_dtype(X_arr, float64)
    assume_shape(X_arr, TupleInt(150) + TupleInt(4))  # type: ignore
    assume_isfinite(X_arr)

    y_arr = NDArray.var("y")
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
