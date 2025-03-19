# mypy: disable-error-code="empty-body"
from __future__ import annotations

from egglog import *

from .array_api import *
from .program_gen import *

##
# Functionality to compile expression to strings of NumPy code.
# Depends on `np` as a global variable.
##

array_api_program_gen_ruleset = ruleset(name="array_api_program_gen_ruleset")
array_api_program_gen_eval_ruleset = ruleset(name="array_api_program_gen_eval_ruleset")

array_api_program_gen_combined_ruleset = (
    array_api_program_gen_ruleset
    | program_gen_ruleset
    | array_api_program_gen_eval_ruleset
    | array_api_vec_to_cons_ruleset
)
array_api_program_gen_schedule = (array_api_program_gen_combined_ruleset | eval_program_rulseset).saturate()


@function
def bool_program(x: Boolean) -> Program: ...


@array_api_program_gen_ruleset.register
def _bool_program():
    yield rewrite(bool_program(TRUE)).to(Program("True"))
    yield rewrite(bool_program(FALSE)).to(Program("False"))


@function
def int_program(x: Int) -> Program: ...


@array_api_program_gen_ruleset.register
def _int_program(i64_: i64, i: Int, j: Int, s: String):
    yield rewrite(int_program(Int.var(s))).to(Program(s, True))
    yield rewrite(int_program(Int(i64_))).to(Program(i64_.to_string()))
    yield rewrite(int_program(~i)).to(Program("~") + int_program(i))
    yield rewrite(bool_program(i < j)).to(Program("(") + int_program(i) + " < " + int_program(j) + ")")
    yield rewrite(bool_program(i <= j)).to(Program("(") + int_program(i) + " <= " + int_program(j) + ")")
    yield rewrite(bool_program(i > j)).to(Program("(") + int_program(i) + " > " + int_program(j) + ")")
    yield rewrite(bool_program(i >= j)).to(Program("(") + int_program(i) + " >= " + int_program(j) + ")")
    yield rewrite(bool_program(i == j)).to(Program("(") + int_program(i) + " == " + int_program(j) + ")")
    yield rewrite(int_program(i + j)).to(Program("(") + int_program(i) + " + " + int_program(j) + ")")
    yield rewrite(int_program(i - j)).to(Program("(") + int_program(i) + " - " + int_program(j) + ")")
    yield rewrite(int_program(i * j)).to(Program("(") + int_program(i) + " * " + int_program(j) + ")")
    yield rewrite(int_program(i / j)).to(Program("(") + int_program(i) + " / " + int_program(j) + ")")
    yield rewrite(int_program(i % j)).to(Program("(") + int_program(i) + " % " + int_program(j) + ")")
    yield rewrite(int_program(i**j)).to(Program("(") + int_program(i) + " ** " + int_program(j) + ")")
    yield rewrite(int_program(i & j)).to(Program("(") + int_program(i) + " & " + int_program(j) + ")")
    yield rewrite(int_program(i | j)).to(Program("(") + int_program(i) + " | " + int_program(j) + ")")
    yield rewrite(int_program(i ^ j)).to(Program("(") + int_program(i) + " ^ " + int_program(j) + ")")
    yield rewrite(int_program(i << j)).to(Program("(") + int_program(i) + " << " + int_program(j) + ")")
    yield rewrite(int_program(i >> j)).to(Program("(") + int_program(i) + " >> " + int_program(j) + ")")
    yield rewrite(int_program(i // j)).to(Program("(") + int_program(i) + " // " + int_program(j) + ")")


@function
def tuple_int_foldl_program(xs: TupleIntLike, f: Callable[[Program, Int], Program], init: ProgramLike) -> Program: ...


@function(ruleset=array_api_program_gen_ruleset)
def tuple_int_program(x: TupleIntLike) -> Program:
    return tuple_int_foldl_program(x, lambda acc, i: acc + int_program(i) + ", ", "(") + ")"


@array_api_program_gen_ruleset.register
def _tuple_int_program(i: Int, ti: TupleInt, ti2: TupleInt, f: Callable[[Program, Int], Program], init: Program):
    yield rewrite(int_program(ti[i])).to(tuple_int_program(ti) + "[" + int_program(i) + "]")
    yield rewrite(int_program(ti.length())).to(Program("len(") + tuple_int_program(ti) + ")")

    yield rewrite(tuple_int_foldl_program(TupleInt.EMPTY, f, init)).to(init)
    yield rewrite(tuple_int_foldl_program(ti.append(i), f, init)).to(f(tuple_int_foldl_program(ti, f, init), i))

    yield rewrite(tuple_int_program(ti + ti2)).to(
        Program("(") + tuple_int_program(ti) + " + " + tuple_int_program(ti2) + ")"
    )


@function
def ndarray_program(x: NDArray) -> Program: ...


@function(ruleset=array_api_program_gen_ruleset)
def ndarray_function_two_program(res: NDArray, l: NDArray, r: NDArray) -> Program:
    return ndarray_program(res).function_two(ndarray_program(l), ndarray_program(r))


@function(ruleset=array_api_program_gen_eval_ruleset)
def ndarray_function_two(res: NDArray, l: NDArray, r: NDArray) -> EvalProgram:
    return EvalProgram(ndarray_function_two_program(res, l, r), {"np": np})


@function
def dtype_program(x: DType) -> Program: ...


@array_api_program_gen_ruleset.register
def _dtype_program():
    yield rewrite(dtype_program(DType.float64)).to(Program("np.dtype(np.float64)"))
    yield rewrite(dtype_program(DType.float32)).to(Program("np.dtype(np.float32)"))
    yield rewrite(dtype_program(DType.int64)).to(Program("np.dtype(np.int64)"))
    yield rewrite(dtype_program(DType.int32)).to(Program("np.dtype(np.int32)"))
    yield rewrite(dtype_program(DType.bool)).to(Program("np.dtype(np.bool)"))
    yield rewrite(dtype_program(DType.object)).to(Program("np.dtype(np.object_)"))


@function
def float_program(x: Float) -> Program: ...


@array_api_program_gen_ruleset.register
def _float_program(f: Float, g: Float, f64_: f64, i: Int, r: BigRat):
    yield rewrite(float_program(Float(f64_))).to(Program(f64_.to_string()))
    yield rewrite(float_program(f.abs())).to(Program("np.abs(") + float_program(f) + ")")
    yield rewrite(float_program(Float.from_int(i))).to(int_program(i))
    yield rewrite(float_program(f + g)).to(Program("(") + float_program(f) + " + " + float_program(g) + ")")
    yield rewrite(float_program(f - g)).to(Program("(") + float_program(f) + " - " + float_program(g) + ")")
    yield rewrite(float_program(f * g)).to(Program("(") + float_program(f) + " * " + float_program(g) + ")")
    yield rewrite(float_program(f / g)).to(Program("(") + float_program(f) + " / " + float_program(g) + ")")
    yield rewrite(float_program(Float.rational(r))).to(
        Program("float(") + Program(r.numer.to_string()) + " / " + Program(r.denom.to_string()) + ")",
        ne(r.denom).to(BigInt(1)),
    )
    yield rewrite(float_program(Float.rational(r))).to(
        Program("float(") + Program(r.numer.to_string()) + ")", eq(r.denom).to(BigInt(1))
    )


@function
def value_program(x: Value) -> Program: ...


@array_api_program_gen_ruleset.register
def _value_program(i: Int, b: Boolean, f: Float, x: NDArray, v1: Value, v2: Value, xs: NDArray, ti: TupleInt):
    yield rewrite(value_program(Value.int(i))).to(int_program(i))
    yield rewrite(value_program(Value.bool(b))).to(bool_program(b))
    yield rewrite(value_program(Value.float(f))).to(float_program(f))
    # Could add .item() but we usually dont need it.
    yield rewrite(value_program(x.to_value())).to(ndarray_program(x))
    yield rewrite(value_program(v1 < v2)).to(Program("(") + value_program(v1) + " < " + value_program(v2) + ")")
    yield rewrite(value_program(v1 / v2)).to(Program("(") + value_program(v1) + " / " + value_program(v2) + ")")
    yield rewrite(value_program(v1 + v2)).to(Program("(") + value_program(v1) + " + " + value_program(v2) + ")")
    yield rewrite(value_program(v1 * v2)).to(Program("(") + value_program(v1) + " * " + value_program(v2) + ")")
    yield rewrite(bool_program(v1.to_bool)).to(value_program(v1))
    yield rewrite(int_program(v1.to_int)).to(value_program(v1))
    yield rewrite(value_program(xs.index(ti))).to((ndarray_program(xs) + "[" + tuple_int_program(ti) + "]").assign())
    yield rewrite(value_program(v1.sqrt())).to(Program("np.sqrt(") + value_program(v1) + ")")
    yield rewrite(value_program(v1.real())).to(Program("np.real(") + value_program(v1) + ")")
    yield rewrite(value_program(v1.conj())).to(Program("np.conj(") + value_program(v1) + ")")


@function
def tuple_value_foldl_program(
    xs: TupleValueLike, f: Callable[[Program, Value], Program], init: ProgramLike
) -> Program: ...


@function(ruleset=array_api_program_gen_ruleset)
def tuple_value_program(x: TupleValueLike) -> Program:
    return tuple_value_foldl_program(x, lambda acc, i: acc + value_program(i) + ", ", "(") + ")"


@array_api_program_gen_ruleset.register
def _tuple_value_program(i: Int, ti: TupleValue, f: Callable[[Program, Value], Program], v: Value, init: Program):
    yield rewrite(value_program(ti[i])).to(tuple_value_program(ti) + "[" + int_program(i) + "]")
    yield rewrite(int_program(ti.length())).to(Program("len(") + tuple_value_program(ti) + ")")

    yield rewrite(tuple_value_foldl_program(TupleValue.EMPTY, f, init)).to(init)
    yield rewrite(tuple_value_foldl_program(ti.append(v), f, init)).to(f(tuple_value_foldl_program(ti, f, init), v))


@function
def tuple_ndarray_foldl_program(
    xs: TupleNDArrayLike, f: Callable[[Program, NDArray], Program], init: ProgramLike
) -> Program: ...


@function(ruleset=array_api_program_gen_ruleset)
def tuple_ndarray_program(x: TupleNDArrayLike) -> Program:
    return tuple_ndarray_foldl_program(x, lambda acc, i: acc + ndarray_program(i) + ", ", "(") + ")"


@array_api_program_gen_ruleset.register
def _tuple_ndarray_program(
    i: Int, ti: TupleNDArray, f: Callable[[Program, NDArray], Program], v: NDArray, init: Program
):
    yield rewrite(ndarray_program(ti[i])).to(tuple_ndarray_program(ti) + "[" + int_program(i) + "]")
    yield rewrite(int_program(ti.length())).to(Program("len(") + tuple_ndarray_program(ti) + ")")

    yield rewrite(tuple_ndarray_foldl_program(TupleNDArray.EMPTY, f, init)).to(init)
    yield rewrite(tuple_ndarray_foldl_program(ti.append(v), f, init)).to(f(tuple_ndarray_foldl_program(ti, f, init), v))


@function
def optional_dtype_program(x: OptionalDType) -> Program: ...


@array_api_program_gen_ruleset.register
def _optional_dtype_program(dtype: DType):
    yield rewrite(optional_dtype_program(OptionalDType.none)).to(Program("None"))
    yield rewrite(optional_dtype_program(OptionalDType.some(dtype))).to(dtype_program(dtype))


@function
def optional_int_program(x: OptionalInt) -> Program: ...


@array_api_program_gen_ruleset.register
def _optional_int_program(x: Int):
    yield rewrite(optional_int_program(OptionalInt.none)).to(Program("None"))
    yield rewrite(optional_int_program(OptionalInt.some(x))).to(int_program(x))


@function
def optional_int_slice_program(x: OptionalInt) -> Program:
    """
    Translates an optional int to a program, but translates None as "" instead of None
    """


@array_api_program_gen_ruleset.register
def _optional_int_slice_program(x: Int):
    yield rewrite(optional_int_slice_program(OptionalInt.none)).to(Program(""))
    yield rewrite(optional_int_slice_program(OptionalInt.some(x))).to(int_program(x))


@function
def slice_program(x: Slice) -> Program: ...


@array_api_program_gen_ruleset.register
def _slice_program(start: OptionalInt, stop: OptionalInt, i: Int):
    yield rewrite(slice_program(Slice(start, stop, OptionalInt.none))).to(
        optional_int_slice_program(start) + ":" + optional_int_slice_program(stop)
    )
    yield rewrite(slice_program(Slice(start, stop, OptionalInt.some(i)))).to(
        optional_int_slice_program(start) + ":" + optional_int_slice_program(stop) + ":" + int_program(i)
    )


@function
def multi_axis_index_key_item_program(x: MultiAxisIndexKeyItem) -> Program: ...


@array_api_program_gen_ruleset.register
def _multi_axis_index_key_item_program(i: Int, s: Slice):
    yield rewrite(multi_axis_index_key_item_program(MultiAxisIndexKeyItem.int(i))).to(int_program(i))
    yield rewrite(multi_axis_index_key_item_program(MultiAxisIndexKeyItem.slice(s))).to(slice_program(s))
    yield rewrite(multi_axis_index_key_item_program(MultiAxisIndexKeyItem.ELLIPSIS)).to(Program("..."))
    yield rewrite(multi_axis_index_key_item_program(MultiAxisIndexKeyItem.NONE)).to(Program("None"))


@function
def multi_axis_index_key_program(x: MultiAxisIndexKey) -> Program: ...


@array_api_program_gen_ruleset.register
def _multi_axis_index_key_program(
    idx_fn: Callable[[Int], MultiAxisIndexKeyItem], k: i64, vec: Vec[MultiAxisIndexKeyItem], i: MultiAxisIndexKeyItem
):
    yield rewrite(multi_axis_index_key_program(MultiAxisIndexKey(0, idx_fn))).to(Program(""))

    yield rewrite(multi_axis_index_key_program(MultiAxisIndexKey(Int(k), idx_fn))).to(
        multi_axis_index_key_item_program(idx_fn(Int(0)))
        + ", "
        + multi_axis_index_key_program(MultiAxisIndexKey(Int(k - 1), lambda i: idx_fn(i + 1))),
        ne(k).to(i64(0)),
    )

    yield rewrite(multi_axis_index_key_program(MultiAxisIndexKey.from_vec(Vec[MultiAxisIndexKeyItem]()))).to(
        Program("")
    )
    yield rewrite(multi_axis_index_key_program(MultiAxisIndexKey.from_vec(vec))).to(
        multi_axis_index_key_item_program(vec[0]) + ",",
        eq(vec.length()).to(i64(1)),
    )
    yield rewrite(multi_axis_index_key_program(MultiAxisIndexKey.from_vec(vec))).to(
        multi_axis_index_key_item_program(vec[0])
        + ", "
        + multi_axis_index_key_program(MultiAxisIndexKey.from_vec(vec.remove(0))),
        vec.length() > 1,
    )


@function
def index_key_program(x: IndexKey) -> Program: ...


@array_api_program_gen_ruleset.register
def _index_key_program(i: Int, s: Slice, key: MultiAxisIndexKey, a: NDArray):
    yield rewrite(index_key_program(IndexKey.ELLIPSIS)).to(Program("..."))
    yield rewrite(index_key_program(IndexKey.int(i))).to(int_program(i))
    yield rewrite(index_key_program(IndexKey.slice(s))).to(slice_program(s))
    yield rewrite(index_key_program(IndexKey.multi_axis(key))).to(multi_axis_index_key_program(key))
    yield rewrite(index_key_program(IndexKey.ndarray(a))).to(ndarray_program(a))


@function
def int_or_tuple_program(x: IntOrTuple) -> Program: ...


@array_api_program_gen_ruleset.register
def _int_or_tuple_program(x: Int, t: TupleInt):
    yield rewrite(int_or_tuple_program(IntOrTuple.int(x))).to(int_program(x))
    yield rewrite(int_or_tuple_program(IntOrTuple.tuple(t))).to(tuple_int_program(t))


@function
def optional_int_or_tuple_program(x: OptionalIntOrTuple) -> Program: ...


@array_api_program_gen_ruleset.register
def _optional_int_or_tuple_program(it: IntOrTuple):
    yield rewrite(optional_int_or_tuple_program(OptionalIntOrTuple.some(it))).to(int_or_tuple_program(it))
    yield rewrite(optional_int_or_tuple_program(OptionalIntOrTuple.none)).to(Program("None"))


@array_api_program_gen_ruleset.register
def _ndarray_program(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    s: String,
    dtype: DType,
    ti: TupleInt,
    i: Int,
    tv: TupleValue,
    v: Value,
    ob: OptionalBool,
    tnd: TupleNDArray,
    optional_device_: OptionalDevice,
    int_or_tuple_: IntOrTuple,
    idx: IndexKey,
    odtype: OptionalDType,
):
    # Var
    yield rewrite(ndarray_program(NDArray.var(s))).to(Program(s, True))

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
        z_program.statement(Program("assert set(np.unique(") + z_program + ")) == set(" + tuple_value_program(tv) + ")")
    )

    # Value

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
    # unique_inverse(x) => unique(x, return_inverse=True)
    yield rewrite(tuple_ndarray_program(unique_inverse(x))).to(
        (Program("np.unique(") + ndarray_program(x) + ", return_inverse=True)").assign()
    )

    # Tuple ndarray indexing
    yield rewrite(ndarray_program(tnd[i])).to(tuple_ndarray_program(tnd) + "[" + int_program(i) + "]")

    # ndarray scalar
    # TODO: Use dtype and shape and indexing instead?
    # TODO: SPecify dtype?
    yield rewrite(ndarray_program(NDArray.scalar(v))).to(Program("np.array(") + value_program(v) + ")")

    # zeros
    yield rewrite(ndarray_program(zeros(ti, OptionalDType.none, optional_device_))).to(
        (Program("np.zeros(") + tuple_int_program(ti) + ")").assign()
    )
    yield rewrite(ndarray_program(zeros(ti, OptionalDType.some(dtype), optional_device_))).to(
        (Program("np.zeros(") + tuple_int_program(ti) + ", dtype=" + dtype_program(dtype) + ")").assign(),
    )

    # unique_values
    yield rewrite(ndarray_program(unique_values(x))).to((Program("np.unique(") + ndarray_program(x) + ")").assign())

    # reshape

    def bin_op(res: NDArray, op: str) -> Command:
        return rewrite(ndarray_program(res)).to((ndarray_program(x) + f" {op} " + ndarray_program(y)).assign())

    # NDARRAy ops
    yield bin_op(x + y, "+")
    yield bin_op(x - y, "-")
    yield bin_op(x * y, "*")
    yield bin_op(x / y, "/")
    yield bin_op(x < y, "<")
    yield bin_op(x <= y, "<=")
    yield bin_op(x > y, ">")
    yield bin_op(x >= y, ">=")
    yield bin_op(x == y, "==")
    yield bin_op(x @ y, "@")
    yield bin_op(x % y, "%")
    yield bin_op(x & y, "&")
    yield bin_op(x | y, "|")
    yield bin_op(x ^ y, "^")
    yield bin_op(x << y, "<<")
    yield bin_op(x >> y, ">>")
    yield bin_op(x // y, "//")
    yield bin_op(x**y, "**")

    # setitem
    mod_x = copy(x)
    mod_x[idx] = y
    assigned_x = ndarray_program(x).assign()
    yield rewrite(ndarray_program(mod_x)).to(
        assigned_x.statement(assigned_x + "[" + index_key_program(idx) + "] = " + ndarray_program(y))
    )
    # getitem
    yield rewrite(ndarray_program(x[idx])).to(ndarray_program(x) + "[" + index_key_program(idx) + "]")

    # square
    yield rewrite(ndarray_program(square(x))).to((Program("np.square(") + ndarray_program(x) + ")").assign())

    # expand_dims(x, axis)
    yield rewrite(ndarray_program(expand_dims(x, i))).to(
        (Program("np.expand_dims(") + ndarray_program(x) + ", " + int_program(i) + ")").assign()
    )

    # mean(x, axis)
    yield rewrite(ndarray_program(mean(x))).to((Program("np.mean(") + ndarray_program(x) + ")").assign())
    yield rewrite(
        ndarray_program(mean(x, OptionalIntOrTuple.some(int_or_tuple_), FALSE)),
    ).to(
        (Program("np.mean(") + ndarray_program(x) + ", axis=" + int_or_tuple_program(int_or_tuple_) + ")").assign(),
    )
    yield rewrite(
        ndarray_program(mean(x, OptionalIntOrTuple.some(int_or_tuple_), TRUE)),
    ).to(
        (
            Program("np.mean(")
            + ndarray_program(x)
            + ", axis="
            + int_or_tuple_program(int_or_tuple_)
            + ", keepdims=True)"
        ).assign(),
    )

    # Concat
    yield rewrite(ndarray_program(concat(tnd, OptionalInt.none))).to(
        (Program("np.concatenate(") + tuple_ndarray_program(tnd) + ")").assign()
    )
    yield rewrite(ndarray_program(concat(tnd, OptionalInt.some(i)))).to(
        (Program("np.concatenate(") + tuple_ndarray_program(tnd) + ", axis=" + int_program(i) + ")").assign()
    )
    # Vector
    yield rewrite(ndarray_program(NDArray.vector(tv))).to(Program("np.array(") + tuple_value_program(tv) + ")")
    # std
    yield rewrite(ndarray_program(std(x))).to((Program("np.std(") + ndarray_program(x) + ")").assign())
    yield rewrite(ndarray_program(std(x, OptionalIntOrTuple.some(int_or_tuple_)))).to(
        (Program("np.std(") + ndarray_program(x) + ", axis=" + int_or_tuple_program(int_or_tuple_) + ")").assign(),
    )
    # svd
    yield rewrite(tuple_ndarray_program(svd(x))).to((Program("np.linalg.svd(") + ndarray_program(x) + ")").assign())
    yield rewrite(tuple_ndarray_program(svd(x, FALSE))).to(
        (Program("np.linalg.svd(") + ndarray_program(x) + ", full_matrices=False)").assign()
    )
    # sqrt
    yield rewrite(ndarray_program(sqrt(x))).to((Program("np.sqrt(") + ndarray_program(x) + ")").assign())
    # Transpose
    yield rewrite(ndarray_program(x.T)).to(ndarray_program(x) + ".T")
    # sum
    yield rewrite(ndarray_program(sum(x))).to((Program("np.sum(") + ndarray_program(x) + ")").assign())
    yield rewrite(ndarray_program(sum(x, OptionalIntOrTuple.some(int_or_tuple_)))).to(
        (Program("np.sum(") + ndarray_program(x) + ", axis=" + int_or_tuple_program(int_or_tuple_) + ")").assign()
    )
    yield rewrite(tuple_int_program(x.shape)).to(ndarray_program(x) + ".shape")
    yield rewrite(ndarray_program(abs(x))).to((Program("np.abs(") + ndarray_program(x) + ")").assign())

    # asarray
    yield rewrite(ndarray_program(asarray(x, odtype))).to(
        Program("np.asarray(") + ndarray_program(x) + ", " + optional_dtype_program(odtype) + ")"
    )
