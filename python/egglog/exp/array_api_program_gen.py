# mypy: disable-error-code="empty-body"
from __future__ import annotations

from egglog import *

from .array_api import *
from .program_gen import *

##
# Functionality to compile expression to strings of NumPy code.
# Depends on `np` as a global variable.
##

array_api_module_string = Module([array_api_module, program_gen_module])


@array_api_module_string.function()
def ndarray_program(x: NDArray) -> Program:
    ...


@array_api_module_string.function()
def dtype_program(x: DType) -> Program:
    ...


@array_api_module_string.function()
def tuple_int_program(x: TupleInt) -> Program:
    ...


@array_api_module_string.function()
def int_program(x: Int) -> Program:
    ...


@array_api_module_string.function()
def tuple_value_program(x: TupleValue) -> Program:
    ...


@array_api_module_string.function()
def value_program(x: Value) -> Program:
    ...


array_api_module_string.register(
    union(dtype_program(DType.float64)).with_(Program("np.float64")),
    union(dtype_program(DType.int64)).with_(Program("np.int64")),
)


@array_api_module_string.function
def bool_program(x: Bool) -> Program:
    ...


array_api_module_string.register(
    union(bool_program(TRUE)).with_(Program("True")),
    union(bool_program(FALSE)).with_(Program("False")),
)


@array_api_module_string.function
def float_program(x: Float) -> Program:
    ...


@array_api_module_string.function
def tuple_ndarray_program(x: TupleNDArray) -> Program:
    ...


@array_api_module_string.function
def optional_dtype_program(x: OptionalDType) -> Program:
    ...


@array_api_module_string.register
def _py_expr(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    s: String,
    y_str: String,
    z_str: String,
    dtype_str: String,
    dtype: DType,
    ti: TupleInt,
    ti1: TupleInt,
    ti2: TupleInt,
    ti_str: String,
    ti_str1: String,
    ti_str2: String,
    tv_str: String,
    tv1_str: String,
    tv2_str: String,
    i: Int,
    i_str: String,
    i64_: i64,
    tv: TupleValue,
    tv1: TupleValue,
    tv2: TupleValue,
    v: Value,
    v_str: String,
    b: Bool,
    f: Float,
    f_str: String,
    b_str: String,
    f64_: f64,
    ob: OptionalBool,
    tnd: TupleNDArray,
    tnd_str: String,
    device_: Device,
    optional_device_: OptionalDevice,
    optional_dtype_: OptionalDType,
):
    # Var
    yield rewrite(ndarray_program(NDArray.var(s))).to(Program(s))

    # Asssume dtype
    z_assumed_dtype = copy(z)
    assume_dtype(z_assumed_dtype, dtype)
    z_program = ndarray_program(z)
    yield rewrite(ndarray_program(z_assumed_dtype)).to(
        z_program.statement(Program("assert ") + z_program + ".dtype == " + dtype_program(dtype))
    )
    # assume shape
    z_assumed_shape = copy(z)
    assume_shape(z_assumed_shape, ti)
    yield rewrite(ndarray_program(z_assumed_shape)).to(
        z_program.statement(Program("assert ") + z_program + ".shape == " + tuple_int_program(ti))
    )
    # tuple int
    yield rewrite(tuple_int_program(ti1 + ti2)).to(tuple_int_program(ti1) + " + " + tuple_int_program(ti2))
    yield rewrite(tuple_int_program(TupleInt(i))).to(Program("(") + int_program(i) + ",)")
    # Int
    yield rewrite(int_program(Int(i64_))).to(Program(i64_.to_string()))

    # assume isfinite
    z_assumed_isfinite = copy(z)
    assume_isfinite(z_assumed_isfinite)
    yield rewrite(ndarray_program(z_assumed_isfinite)).to(
        z_program.statement(Program("assert np.all(np.isfinite(") + z_program + "))")
    )

    # Assume value_one_of
    z_assumed_value_one_of = copy(z)
    assume_value_one_of(z_assumed_value_one_of, tv)
    yield rewrite(ndarray_program(z_assumed_value_one_of)).to(
        z_program.statement(Program("assert set(") + z_program + ".flatten()) == set(" + tuple_value_program(tv) + ")")
    )

    # tuple values
    yield rewrite(tuple_value_program(tv1 + tv2)).to(tuple_value_program(tv1) + " + " + tuple_value_program(tv2))
    yield rewrite(tuple_value_program(TupleValue(v))).to(Program("(") + value_program(v) + ",)")

    # Value
    yield rewrite(value_program(Value.int(i))).to(int_program(i))
    yield rewrite(value_program(Value.bool(b))).to(bool_program(b))
    yield rewrite(value_program(Value.float(f))).to(float_program(f))

    # Float
    yield rewrite(float_program(Float(f64_))).to(Program(f64_.to_string()))

    # reshape (don't include copy, since not present in numpy)
    yield rewrite(ndarray_program(reshape(y, ti, ob))).to(
        (ndarray_program(y) + ".reshape(" + tuple_int_program(ti) + ")").assign()
    )

    # astype
    yield rewrite(ndarray_program(astype(y, dtype))).to(
        (ndarray_program(y) + ".astype(" + dtype_program(dtype) + ")").assign()
    )

    # unique_counts(x) => unique(x, return_counts=True)
    yield rewrite(tuple_ndarray_program(unique_counts(x))).to(
        (Program("np.unique(") + ndarray_program(x) + ", return_counts=True)").assign()
    )

    # Tuple ndarray indexing
    yield rewrite(ndarray_program(tnd[i])).to(tuple_ndarray_program(tnd) + "[" + int_program(i) + "]")

    # ndarray scalar
    # TODO: Use dtype and shape and indexing instead?
    # TODO: SPecify dtype?
    yield rewrite(ndarray_program(NDArray.scalar(v))).to(Program("np.array(") + value_program(v) + ")")

    # zeros
    yield rewrite(ndarray_program(zeros(ti, optional_dtype_, optional_device_))).to(
        (
            Program("np.zeros(") + tuple_int_program(ti) + ", dtype=" + optional_dtype_program(optional_dtype_) + ")"
        ).assign()
    )

    # Optional dtype
    yield rewrite(optional_dtype_program(OptionalDType.none)).to(Program("None"))
    yield rewrite(optional_dtype_program(OptionalDType.some(dtype))).to(dtype_program(dtype))

    # unique_values
    yield rewrite(ndarray_program(unique_values(x))).to((Program("np.unique(") + ndarray_program(x) + ")").assign())

    # reshape

    # NDARRAy ops
    yield rewrite(ndarray_program(x + y)).to((ndarray_program(x) + " + " + ndarray_program(y)).assign())
    yield rewrite(ndarray_program(x - y)).to((ndarray_program(x) + " - " + ndarray_program(y)).assign())
    yield rewrite(ndarray_program(x * y)).to((ndarray_program(x) + " * " + ndarray_program(y)).assign())
    yield rewrite(ndarray_program(x / y)).to((ndarray_program(x) + " / " + ndarray_program(y)).assign())
