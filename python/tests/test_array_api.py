from egglog.exp.array_api import *


def test_simplify_any_unique():
    X = NDArray.var("X")
    res = any(
        (astype(unique_counts(X)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0))))
        < NDArray.scalar(Value.int(Int(0)))
    ).to_bool()

    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run((run() * 20).saturate())
    # egraph.graphviz(inline_leaves=True).render(view=True)
    # print(egraph.extract(res))
    egraph.check(eq(res).to(FALSE))


def test_tuple_value_includes():
    x = TupleValue(Value.bool(FALSE))
    should_be_true = x.includes(Value.bool(FALSE))
    should_be_false = x.includes(Value.bool(TRUE))
    egraph = EGraph([array_api_module])
    egraph.register(should_be_true)
    egraph.register(should_be_false)
    egraph.run((run() * 10).saturate())
    egraph.graphviz(inline_leaves=True).render(view=True)
    # print(egraph.extract(should_be_true))
    # print(egraph.extract(should_be_false))
    egraph.check(eq(should_be_true).to(TRUE))
    egraph.check(eq(should_be_false).to(FALSE))


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
    # _NDArray_4 = astype(unique_counts(_NDArray_3)[Int(1)], DType.float64) / NDArray.scalar(Value.float(Float(150.0)))
    _NDArray_5 = zeros(
        unique_values(_NDArray_3).shape + TupleInt(Int(4)),
        OptionalDType.some(DType.float64),
        OptionalDevice.some(_NDArray_1.device),
    )
    res = _NDArray_1 + _NDArray_5
    fn = FunctionExprTwo("my_fn", res, X_orig, Y_orig)

    egraph = EGraph([array_api_module_string])
    egraph.register(fn)

    egraph.run(run() * 20)
    # while egraph.run((run())).updated:
    #     print(egraph.load_object(egraph.extract(PyObject.from_string(statements()))))
    egraph.graphviz().render(view=True)
    egraph.graphviz(n_inline_leaves=3).render("inlined", view=True)

    egraph.run(run(fn_ruleset))
    fn_source = egraph.load_object(egraph.extract(PyObject.from_string(fn.source)))
    print(fn_source)
    locals_: dict[str, object] = {}
    exec(fn_source, {"np": np}, locals_)  # type: ignore
    fn: object = locals_["my_fn"]


def test_reshape_index():
    # Verify that it doesn't expand forever
    x = NDArray.var("x")
    new_shape = TupleInt(Int(-1))
    res = reshape(x, new_shape).index(TupleInt(Int(1)) + TupleInt(Int(2)))
    egraph = EGraph([array_api_module])
    egraph.register(res)
    egraph.run(run() * 10)
    # egraph.graphviz().render(view=True)
    equiv_expr = egraph.extract_multiple(res, 10)
    print(equiv_expr)
    assert len(equiv_expr) == 2
