from egglog.exp.array_api import *


def test_to_source():
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

    res = _NDArray_4 + _NDArray_1
    fn = FunctionExprTwo("my_fn", res, X_orig, Y_orig)

    egraph = EGraph([array_api_module_string])
    egraph.register(fn)

    egraph.run((run() * 20).saturate())
    # while egraph.run((run())).updated:
    #     print(egraph.load_object(egraph.extract(PyObject.from_string(statements()))))
    # egraph.graphviz.render(view=True)

    egraph.run(run(fn_ruleset))

    fn_source = egraph.load_object(egraph.extract(PyObject.from_string(fn.source)))

    exec(fn_source, {"np": np})  # type: ignore
    fn = locals["my_fn"]
    print(fn_source)
