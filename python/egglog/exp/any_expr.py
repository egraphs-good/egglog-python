"""
WIP

An `AnyExpr`, which can be used to trace arbitrary expressions.

Created from any Python object, it should forward any operations on it to the underlying Python object.

This will only happen when it needs to be "materialized" however, through operations like `__bool__` or `__iter__`.

Generally it will try to avoid materializing the underlying object, and instead just treat it as a black box.
"""
# mypy: disable-error-code="empty-body"

from __future__ import annotations

import contextlib
import math
import operator
from collections.abc import Iterator
from copy import copy
from functools import reduce
from typing import Any, TypeAlias

from egglog import *
from egglog.exp.program_gen import *


class AnyExpr(Expr):
    """
    Wraps an arbitrary Python object.

    Any operations on it will be forwarded to the underlying object when needed.

    Attempts to implement as many operations from https://docs.python.org/3/reference/datamodel.html as possible.

    Can be converted from any Python object:

    >>> AnyExpr(42) + 42
    AnyExpr(A(42) + A(42))

    Will also convert tuples and lists item by item:

    >>> AnyExpr((1, 2,)) + (5, 6)
    AnyExpr(append(append(A(()), A(1)), A(2)) + append(append(A(()), A(5)), A(6)))
    """

    def __init__(self, obj: ALike) -> None: ...

    __match_args__ = ("egglog_any_expr_value",)

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def egglog_any_expr_value(self) -> A:
        """
        Return the underlying Python object, if it was constructued with one.

        Long method name so it doesn't conflict with any user-defined properties.

        >>> AnyExpr(10).egglog_any_expr_value
        A(10)
        """
        match get_callable_args(self, AnyExpr):
            case (A() as any_expr,):
                return any_expr
        raise ExprValueError(self, "AnyExpr")

    @method(preserve=True)
    def __bytes__(self) -> bytes:
        """
        >>> bytes(AnyExpr(b"hello"))
        b'hello'
        """
        return any_eval(bytes_(self))

    @method(preserve=True)
    def __bool__(self) -> bool:
        """
        >>> bool(AnyExpr(True))
        True
        >>> bool(AnyExpr(False))
        False
        """
        return any_eval(bool_(self))

    @method(preserve=True)
    def __eq__(self, other: object) -> AnyExpr:  # type: ignore[override]
        """
        >>> bool(AnyExpr(1) == AnyExpr(1))
        True
        >>> bool(AnyExpr(1) == AnyExpr(2))
        False
        """
        return with_assert(self.egglog_any_expr_value == other)

    @method(preserve=True)
    def __ne__(self, other: object) -> AnyExpr:  # type: ignore[override]
        """
        >>> bool(AnyExpr(1) != AnyExpr(2))
        True
        >>> bool(AnyExpr(1) != AnyExpr(1))
        False
        """
        return with_assert(self.egglog_any_expr_value != other)

    @method(preserve=True)
    def __lt__(self, other: object) -> AnyExpr:
        """
        >>> bool(AnyExpr(1) < AnyExpr(2))
        True
        >>> bool(AnyExpr(2) < AnyExpr(1))
        False
        """
        return with_assert(self.egglog_any_expr_value < other)

    @method(preserve=True)
    def __le__(self, other: object) -> AnyExpr:
        """
        >>> bool(AnyExpr(2) <= AnyExpr(2))
        True
        >>> bool(AnyExpr(3) <= AnyExpr(2))
        False
        """
        return with_assert(self.egglog_any_expr_value <= other)

    @method(preserve=True)
    def __gt__(self, other: object) -> AnyExpr:
        """
        >>> bool(AnyExpr(3) > AnyExpr(2))
        True
        >>> bool(AnyExpr(2) > AnyExpr(3))
        False
        """
        return with_assert(self.egglog_any_expr_value > other)

    @method(preserve=True)
    def __ge__(self, other: object) -> AnyExpr:
        """
        >>> bool(AnyExpr(3) >= AnyExpr(3))
        True
        >>> bool(AnyExpr(2) >= AnyExpr(3))
        False
        """
        return with_assert(self.egglog_any_expr_value >= other)

    @method(preserve=True)
    def __hash__(self) -> int:
        """
        Turn the underlying object into a hash.

        >>> hash(AnyExpr("hello")) == hash("hello")
        True
        """
        return hash(any_eval(self.egglog_any_expr_value))

    @method(preserve=True)
    def __getattr__(self, name: StringLike) -> AnyExpr | Any:
        """
        Get an attribute of the underlying object.

        >>> int(AnyExpr([1, 2, 3]).index(2))
        1

        Also should work with hasattr:
        >>> hasattr(AnyExpr([1, 2, 3]), "index")
        True
        >>> hasattr(AnyExpr([1, 2, 3]), "nonexistent")
        False
        """
        inner = self.egglog_any_expr_value
        # Need to raise attribute error if it doesn't exist, since this is called for hasattr
        if not any_eval(hasattr_(inner, name)):
            raise AttributeError(f"{self} has no attribute {name}")
        egraph = _get_current_egraph()
        res = inner.__getattr__(name)
        egraph.register(res)
        egraph.run(any_expr_schedule)
        if egraph.check_bool(getattr_eager(inner, name)):
            return any_eval(res)
        return with_assert(res)

    # @method(mutates_self=True)
    # def __setattr__(self, name: StringLike, value: object) -> None:
    #     """
    #     Set an attribute of the underlying object.

    #     >>> x = lambda: None
    #     >>> expr = AnyExpr(x)
    #     >>> expr.attr = 42
    #     >>> int(expr.attr)
    #     42
    #     """

    # TODO: delattr
    # TODO: __get__/__set__?

    @method(preserve=True)
    def __len__(self) -> int:
        """
        Get the length of the underlying object.

        >>> len(AnyExpr([1, 2, 3]))
        3
        """
        return any_eval(len_(self))

    @method(preserve=True)
    def __call__(self, *args: object, **kwargs: object) -> AnyExpr:
        """
        Call the underlying object.

        >>> int(AnyExpr(int)(42))
        42
        >>> int(AnyExpr(lambda *x, **y: len(x) + len(y))(1, 2, a=3, b=4))
        4
        """
        args_expr = A(())
        for a in args:
            args_expr = append(args_expr, a)
        kwargs_expr = A({})
        for k, v in kwargs.items():
            kwargs_expr = set_kwarg(kwargs_expr, k, v)
        return with_assert(self.egglog_any_expr_value(args_expr, kwargs_expr))

    @method(preserve=True)
    def __getitem__(self, key: object) -> AnyExpr:
        """
        Get an item from the underlying object.

        >>> int(AnyExpr([1, 2, 3])[1])
        2
        """
        return with_assert(self.egglog_any_expr_value[key])

    @method(preserve=True)
    def __setitem__(self, key: object, value: object) -> None:
        """
        Set an item in the underlying object.

        >>> x = [1, 2, 3]
        >>> expr = AnyExpr(x)
        >>> expr[1] = 42
        >>> int(expr[1])
        42
        """
        any_expr_inner = self.egglog_any_expr_value
        any_expr_inner[key] = value
        self.__replace_expr__(AnyExpr(with_assert(any_expr_inner)))

    @method(preserve=True)
    def __delitem__(self, key: object) -> None:
        """
        Delete an item from the underlying object.

        >>> x = [1, 2, 3]
        >>> expr = AnyExpr(x)
        >>> del expr[1]
        >>> len(expr)
        2
        """
        any_expr_inner = self.egglog_any_expr_value
        del any_expr_inner[key]
        self.__replace_expr__(AnyExpr(with_assert(any_expr_inner)))

    # TODO: support real iterators
    @method(preserve=True)
    def __iter__(self) -> Iterator[AnyExpr]:
        """
        Iterate over the underlying object.

        >>> list(AnyExpr((1, 2)))
        [AnyExpr(append(append(A(()), A(1)), A(2))[A(0)]), AnyExpr(append(append(A(()), A(1)), A(2))[A(1)])]
        """
        return iter(self[i] for i in range(len(self)))

    # TODO: Not working for now
    # @method(preserve=True)
    # def __reversed__(self) -> Iterator[AnyExpr]:
    #     """
    #     Reverse iterate over the underlying object.

    #     >>> list(reversed(AnyExpr([1, 2, 3])))
    #     [AnyExpr(3), AnyExpr(2), AnyExpr(1)]
    #     """
    #     return map(AnyExpr, any_eval(reversed_op(self)))

    @method(preserve=True)
    def __contains__(self, item: object) -> bool:
        """
        Check if the underlying object contains an item.

        >>> class A:
        ...     def __contains__(self, item):
        ...         return item == 42
        >>> 42 in AnyExpr(A())
        True
        >>> 2 in AnyExpr(A())
        False
        """
        return any_eval(contains(self, item))

    ##
    # Emulating numeric types
    # https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    ##

    @method(preserve=True)
    def __add__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(1) + 2)
        3
        """
        return with_assert(self.egglog_any_expr_value + other)

    @method(preserve=True)
    def __sub__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(3) - 2)
        1
        """
        return with_assert(self.egglog_any_expr_value - other)

    @method(preserve=True)
    def __mul__(self, other: object) -> AnyExpr:
        """
        # >>> int(AnyExpr(3) * 2)
        # 6
        >>> 4 * AnyExpr(3)
        AnyExpr(A(4) * A(3))
        """
        return with_assert(self.egglog_any_expr_value * other)

    @method(preserve=True)
    def __matmul__(self, other: object) -> AnyExpr:
        """
        >>> class Matrix:
        ...     def __matmul__(self, other):
        ...         return 42
        >>> int(AnyExpr(Matrix()) @ Matrix())
        42
        """
        return with_assert(self.egglog_any_expr_value @ other)

    @method(preserve=True)
    def __truediv__(self, other: object) -> AnyExpr:
        """
        >>> float(AnyExpr(3) / 2)
        1.5
        """
        return with_assert(self.egglog_any_expr_value / other)

    @method(preserve=True)
    def __floordiv__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(3) // 2)
        1
        """
        return with_assert(self.egglog_any_expr_value // other)

    @method(preserve=True)
    def __mod__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(3) % 2)
        1
        """
        return with_assert(self.egglog_any_expr_value % other)

    @method(preserve=True)
    def __divmod__(self, other: object) -> AnyExpr:
        """
        >>> div, mod = divmod(AnyExpr(3), 2)
        >>> int(div)
        1
        >>> int(mod)
        1
        """
        return with_assert(divmod(self.egglog_any_expr_value, other))

    # TODO: Support modulo
    @method(preserve=True)
    def __pow__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(3) ** 2)
        9
        """
        return with_assert(self.egglog_any_expr_value**other)

    @method(preserve=True)
    def __lshift__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(1) << 2)
        4
        """
        return with_assert(self.egglog_any_expr_value << other)

    @method(preserve=True)
    def __rshift__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(4) >> 2)
        1
        """
        return with_assert(self.egglog_any_expr_value >> other)

    @method(preserve=True)
    def __and__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(6) & 3)
        2
        """
        return with_assert(self.egglog_any_expr_value & other)

    @method(preserve=True)
    def __xor__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(6) ^ 3)
        5
        """
        return with_assert(self.egglog_any_expr_value ^ other)

    @method(preserve=True)
    def __or__(self, other: object) -> AnyExpr:
        """
        >>> int(AnyExpr(6) | 3)
        7
        """
        return with_assert(self.egglog_any_expr_value | other)

    @method(preserve=True)
    def __neg__(self) -> AnyExpr:
        """
        >>> int(-AnyExpr(3))
        -3
        """
        return with_assert(-self.egglog_any_expr_value)

    @method(preserve=True)
    def __pos__(self) -> AnyExpr:
        """
        >>> int(+AnyExpr(3))
        3
        """
        return with_assert(+self.egglog_any_expr_value)

    @method(preserve=True)
    def __abs__(self) -> AnyExpr:
        """
        >>> int(abs(AnyExpr(-3)))
        3
        """
        return with_assert(abs(self.egglog_any_expr_value))

    @method(preserve=True)
    def __complex__(self) -> complex:
        """
        >>> complex(AnyExpr(3+4j))
        (3+4j)
        """
        return any_eval(complex_(self))

    @method(preserve=True)
    def __int__(self) -> int:
        """
        >>> int(AnyExpr(42))
        42
        """
        return any_eval(int_(self))

    @method(preserve=True)
    def __float__(self) -> float:
        """
        >>> float(AnyExpr(3.14))
        3.14
        """
        return any_eval(float_(self))

    @method(preserve=True)
    def __index__(self) -> int:
        """
        >>> import operator
        >>> operator.index(AnyExpr(42))
        42
        """
        return any_eval(index(self))

    # TODO: support ndigits with optional int
    @method(preserve=True)
    def __round__(self) -> AnyExpr:
        """
        >>> int(round(AnyExpr(3.6)))
        4
        """
        return with_assert(round(self.egglog_any_expr_value))

    @method(preserve=True)
    def __trunc__(self) -> AnyExpr:
        """
        >>> import math
        >>> int(math.trunc(AnyExpr(3.6)))
        3
        """
        return with_assert(math.trunc(self.egglog_any_expr_value))

    @method(preserve=True)
    def __floor__(self) -> AnyExpr:
        """
        >>> import math
        >>> int(math.floor(AnyExpr(3.6)))
        3
        """
        return with_assert(math.floor(self.egglog_any_expr_value))

    @method(preserve=True)
    def __ceil__(self) -> AnyExpr:
        """
        >>> import math
        >>> int(math.ceil(AnyExpr(3.4)))
        4
        """
        return with_assert(math.ceil(self.egglog_any_expr_value))

    # TODO: https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers


class A(Expr):
    def __init__(self, obj: object) -> None: ...

    __match_args__ = ("egglog_any_expr_value",)

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def egglog_any_expr_value(self) -> object:
        """
        Return the underlying Python object, if it was constructued with one.

        Long method name so it doesn't conflict with any user-defined properties.

        >>> A(10).egglog_any_expr_value
        10
        """
        match get_callable_args(self, A):
            case (PyObject(obj),):
                return obj
        raise ExprValueError(self, "A")

    def __eq__(self, other: ALike) -> A: ...  # type: ignore[override]
    def __ne__(self, other: ALike) -> A: ...  # type: ignore[override]
    def __lt__(self, other: ALike) -> A: ...
    def __le__(self, other: ALike) -> A: ...
    def __gt__(self, other: ALike) -> A: ...
    def __ge__(self, other: ALike) -> A: ...
    def __getattr__(self, name: StringLike) -> A: ...
    def __call__(self, args: ALike = (), kwargs: ALike = {}) -> A: ...  # noqa: B006
    def __getitem__(self, key: ALike) -> A: ...
    def __setitem__(self, key: ALike, value: ALike) -> None: ...
    def __delitem__(self, key: ALike) -> None: ...
    def __add__(self, other: ALike) -> A: ...
    def __sub__(self, other: ALike) -> A: ...
    def __mul__(self, other: ALike) -> A: ...
    def __matmul__(self, other: ALike) -> A: ...
    def __truediv__(self, other: ALike) -> A: ...
    def __floordiv__(self, other: ALike) -> A: ...
    def __mod__(self, other: ALike) -> A: ...
    def __divmod__(self, other: ALike) -> A: ...
    def __pow__(self, other: ALike) -> A: ...
    def __lshift__(self, other: ALike) -> A: ...
    def __rshift__(self, other: ALike) -> A: ...
    def __and__(self, other: ALike) -> A: ...
    def __xor__(self, other: ALike) -> A: ...
    def __or__(self, other: ALike) -> A: ...
    def __neg__(self) -> A: ...
    def __pos__(self) -> A: ...
    def __abs__(self) -> A: ...
    def __round__(self) -> A: ...
    def __trunc__(self) -> A: ...
    def __floor__(self) -> A: ...
    def __ceil__(self) -> A: ...


converter(A, AnyExpr, AnyExpr)
converter(object, AnyExpr, lambda x: AnyExpr(A(PyObject(x))))

converter(AnyExpr, A, lambda a: a.egglog_any_expr_value, cost=10)
converter(PyObject, A, A, cost=10)
converter(object, A, lambda x: A(PyObject(x)), cost=10)

ALike: TypeAlias = A | object


@function()
def given(expr: ALike, condition: ALike) -> A: ...
@function
def bytes_(expr: ALike) -> A: ...
@function
def bool_(expr: ALike) -> A: ...
@function
def hasattr_(expr: ALike, name: StringLike) -> A: ...
@function
def getattr_eager(expr: ALike, name: StringLike) -> Unit:
    """
    Set if we should eagerly get the attribute.
    """


@function
def len_(expr: ALike) -> A: ...
@function
def append(expr: ALike, item: ALike) -> A:
    """
    Appends an item to a tuple.
    """


@function
def set_kwarg(expr: ALike, key: StringLike, value: ALike) -> A:
    """
    Sets a value in a dict with a string key
    """


@function
def contains(expr: ALike, item: ALike) -> A: ...
@function
def complex_(expr: ALike) -> A: ...
@function
def int_(expr: ALike) -> A: ...
@function
def float_(expr: ALike) -> A: ...
@function
def index(expr: ALike) -> A: ...
@function
def slice_(start: ALike = None, stop: ALike = None, step: ALike = None) -> A: ...
@function
def list_(expr: ALike) -> A: ...
@function
def not_(expr: ALike) -> A: ...
@function
def and_(left: ALike, right: ALike) -> A: ...


# Special case containers so that Any expressions inside
converter(tuple, A, lambda x: reduce(append, x, A(())))
converter(list, A, lambda x: list_(tuple(x)))
converter(slice, A, lambda x: slice_(x.start, x.stop, x.step))


@ruleset
def any_expr_ruleset(x: PyObject, y: PyObject, z: PyObject, s: String, a: A):
    yield rewrite(bytes_(A(x))).to(A(PyObject(bytes)(x)))
    yield rewrite(bool_(A(x))).to(A(PyObject(bool)(x)))
    yield rewrite(A(x) == A(y)).to(A(PyObject(operator.eq)(x, y)))
    yield rewrite(A(x) != A(y)).to(A(PyObject(operator.ne)(x, y)))
    yield rewrite(A(x) < A(y)).to(A(PyObject(operator.lt)(x, y)))
    yield rewrite(A(x) <= A(y)).to(A(PyObject(operator.le)(x, y)))
    yield rewrite(A(x) > A(y)).to(A(PyObject(operator.gt)(x, y)))
    yield rewrite(A(x) >= A(y)).to(A(PyObject(operator.ge)(x, y)))
    yield rewrite(A(x).__getattr__(s)).to(A(PyObject(getattr)(x, PyObject.from_string(s))))
    yield rewrite(hasattr_(A(x), s)).to(A(PyObject(hasattr)(x, PyObject.from_string(s))))
    yield rewrite(len_(A(x))).to(A(PyObject(len)(x)))
    yield rewrite(A(x)(y, z)).to(A(x.call_extended(y, z)))
    yield rewrite(append(A(x), A(y))).to(A(PyObject(lambda t, v: (*t, v))(x, y)))
    yield rewrite(set_kwarg(A(x), s, A(y))).to(A(PyObject(lambda d, k, v: {**d, k: v})(x, PyObject.from_string(s), y)))
    yield rewrite(A(x)[A(y)]).to(A(PyObject(operator.getitem)(x, y)))
    setitem_any = A(x)
    setitem_any[A(y)] = A(z)
    yield rewrite(setitem_any).to(A(PyObject(lambda obj, k, v: operator.setitem(obj, k, v) or obj)(x, y, z)))
    delitem_any = A(x)
    del delitem_any[A(y)]
    yield rewrite(delitem_any).to(A(PyObject(lambda obj, k: operator.delitem(obj, k) or obj)(x, y)))
    yield rewrite(contains(A(x), A(y))).to(A(PyObject(operator.contains)(x, y)))
    yield rewrite(A(x) + A(y)).to(A(PyObject(operator.add)(x, y)))
    yield rewrite(A(x) - A(y)).to(A(PyObject(operator.sub)(x, y)))
    yield rewrite(A(x) * A(y)).to(A(PyObject(operator.mul)(x, y)))
    yield rewrite(A(x) @ A(y)).to(A(PyObject(operator.matmul)(x, y)))
    yield rewrite(A(x) / A(y)).to(A(PyObject(operator.truediv)(x, y)))
    yield rewrite(A(x) // A(y)).to(A(PyObject(operator.floordiv)(x, y)))
    yield rewrite(A(x) % A(y)).to(A(PyObject(operator.mod)(x, y)))
    yield rewrite(divmod(A(x), A(y))).to(A(PyObject(divmod)(x, y)))
    yield rewrite(A(x) ** A(y)).to(A(PyObject(operator.pow)(x, y)))
    yield rewrite(A(x) << A(y)).to(A(PyObject(operator.lshift)(x, y)))
    yield rewrite(A(x) >> A(y)).to(A(PyObject(operator.rshift)(x, y)))
    yield rewrite(A(x) & A(y)).to(A(PyObject(operator.and_)(x, y)))
    yield rewrite(A(x) ^ A(y)).to(A(PyObject(operator.xor)(x, y)))
    yield rewrite(A(x) | A(y)).to(A(PyObject(operator.or_)(x, y)))
    yield rewrite(-A(x)).to(A(PyObject(operator.neg)(x)))
    yield rewrite(+A(x)).to(A(PyObject(operator.pos)(x)))
    yield rewrite(abs(A(x))).to(A(PyObject(operator.abs)(x)))
    yield rewrite(complex_(A(x))).to(A(PyObject(complex)(x)))
    yield rewrite(int_(A(x))).to(A(PyObject(int)(x)))
    yield rewrite(float_(A(x))).to(A(PyObject(float)(x)))
    yield rewrite(index(A(x))).to(A(PyObject(operator.index)(x)))
    yield rewrite(round(A(x))).to(A(PyObject(round)(x)))
    yield rewrite(math.trunc(A(x))).to(A(PyObject(math.trunc)(x)))
    yield rewrite(math.floor(A(x))).to(A(PyObject(math.floor)(x)))
    yield rewrite(math.ceil(A(x))).to(A(PyObject(math.ceil)(x)))
    yield rewrite(list_(A(x))).to(A(PyObject(list)(x)))
    yield rewrite(slice_(A(x), A(y), A(z))).to(A(PyObject(slice)(x, y, z)))

    # Given
    yield rewrite(given(A(x), a)).to(A(x))


any_expr_schedule = any_expr_ruleset.saturate()


def any_eval(self: A) -> Any:
    """
    Evaluate the AnyExpr to get its underlying Python value.

    Runs rules if it's not already resolved
    """
    global _LAST_ASSERT
    egraph = _get_current_egraph()
    # 1. First see if it's already a primitive value
    try:
        return self.egglog_any_expr_value
    except ExprValueError:
        pass
    # 2. If not, try to extract it from the egraph
    expr = egraph.extract(self)
    try:
        res = expr.egglog_any_expr_value
    except ExprValueError:
        # 3. If that isn't one, then try running the schedule to extract it
        egraph.register(expr)
        egraph.run(any_expr_schedule)
        expr = egraph.extract(expr)
        res = expr.egglog_any_expr_value
    # Don't save hasattr asserts
    if get_callable_fn(self) != hasattr_:
        # If we are calling bool_ same as just asserting vlaues
        match get_callable_args(self, bool_):
            case (A() as inner,):
                self = inner
        if eq(expr).to(A(True)):
            asserted = self
            _LAST_ASSERT = with_assert(self).egglog_any_expr_value
        elif eq(expr).to(A(False)):
            match get_callable_args(self, A.__eq__):
                case (A() as left, A() as right):
                    asserted = left != right
                case _:
                    match get_callable_args(self, A.__ne__):
                        case (A() as left, A() as right):
                            asserted = left == right
                        case _:
                            asserted = not_(self)
        else:
            asserted = self == expr
        # _LAST_ASSERT = (
        #     asserted if _LAST_ASSERT is None or eq(_LAST_ASSERT).to(asserted) else and_(_LAST_ASSERT, asserted)
        # )
        _LAST_ASSERT = given(asserted, _LAST_ASSERT) if _LAST_ASSERT is not None else asserted
    return res


_CURRENT_EGRAPH: None | EGraph = None
_LAST_ASSERT: None | A = None


@contextlib.contextmanager
def set_any_expr_egraph(egraph: EGraph) -> Iterator[None]:
    """
    Context manager that will set the current egraph. It will be set back after.
    """
    global _CURRENT_EGRAPH, _LAST_ASSERT
    assert _CURRENT_EGRAPH is None
    assert _LAST_ASSERT is None
    _CURRENT_EGRAPH = egraph
    try:
        yield
    finally:
        _CURRENT_EGRAPH = None
        _LAST_ASSERT = None


def _get_current_egraph() -> EGraph:
    return _CURRENT_EGRAPH or EGraph()


def with_assert(expr: A) -> AnyExpr:
    """
    Add all current asserts to the given expression.

    This is used to make sure that any_evaled expressions are consistent with
    the current context.
    """
    if _CURRENT_EGRAPH and _LAST_ASSERT is not None:  # noqa: SIM108
        a = given(expr, _LAST_ASSERT)
        # match get_callable_args(expr, given):
        #     case (A() as inner, A() as condition):
        #         a = expr if eq(condition).to(_LAST_ASSERT) else given(inner, and_(condition, _LAST_ASSERT))
        #     case _:

    else:
        a = expr
    return AnyExpr(a)


@ruleset
def given_ruleset(x: A, y: A, z: A):
    yield rewrite(not_(given(x, y)), subsume=True).to(given(not_(x), y))
    yield rewrite(given(given(x, y), z), subsume=True).to(given(x, and_(y, z)))
    yield rewrite(and_(x, x), subsume=True).to(x)


@function
def any_expr_program(x: AnyExpr) -> Program:
    r"""
    Convert an AnyExpr to a Program.

    >>> any_expr_source(AnyExpr(42) == 10)
    '(42 == 10)\n'
    """


@function
def a_program(x: A) -> Program:
    """
    Convert an A to a Program.
    """


def w(p: Program) -> Program:
    return Program("(") + p + ")"


def ca(p: ProgramLike, *args: ProgramLike) -> Program:
    args_expr = Program("")
    for a in args[:-1]:
        args_expr += a + ", "
    args_expr += args[-1]
    return convert(p, Program) + Program("(") + args_expr + Program(")")


INLINE_TYPES = int, str, float, bytes, bool, type(None), tuple, dict


@ruleset
def any_program_ruleset(a: A, b: A, c: A, p: PyObject, s: String):
    yield rewrite(any_expr_program(AnyExpr(a)), subsume=True).to(a_program(a))

    yield rewrite(a_program(A(p)), subsume=True).to(
        Program(PyObject(repr)(p).to_string()),
        PyObject(lambda x: isinstance(x, INLINE_TYPES))(p).to_bool() == Bool(True),
    )
    yield rewrite(a_program(A(p)), subsume=True).to(
        Program(PyObject(repr)(p).to_string()).assign(),
        PyObject(lambda x: isinstance(x, INLINE_TYPES))(p).to_bool() == Bool(False),
    )
    yield rewrite(a_program(bytes_(a)), subsume=True).to(a_program(a) + ".bytes()")
    yield rewrite(a_program(bool_(a)), subsume=True).to(ca("bool", a_program(a)))
    yield rewrite(a_program(a == b), subsume=True).to(w(a_program(a) + " == " + a_program(b)))
    yield rewrite(a_program(a != b), subsume=True).to(w(a_program(a) + " != " + a_program(b)))
    yield rewrite(a_program(a < b), subsume=True).to(w(a_program(a) + " < " + a_program(b)))
    yield rewrite(a_program(a <= b), subsume=True).to(w(a_program(a) + " <= " + a_program(b)))
    yield rewrite(a_program(a > b), subsume=True).to(w(a_program(a) + " > " + a_program(b)))
    yield rewrite(a_program(a >= b), subsume=True).to(w(a_program(a) + " >= " + a_program(b)))
    yield rewrite(a_program(a.__getattr__(s)), subsume=True).to(a_program(a) + "." + s)
    yield rewrite(a_program(hasattr_(a, s)), subsume=True).to(
        ca("hasattr", a_program(a), PyObject(repr)(PyObject.from_string(s)).to_string())
    )
    yield rewrite(a_program(len_(a)), subsume=True).to(ca("len", a_program(a)))
    yield rewrite(a_program(a(b, c)), subsume=True).to(
        ca(a_program(a), "*" + a_program(b), "**" + a_program(c)).assign()
    )
    yield rewrite(a_program(append(a, b)), subsume=True).to(ca("", "*" + a_program(a), a_program(b)))
    yield rewrite(a_program(set_kwarg(a, s, b)), subsume=True).to(
        "{**" + a_program(a) + ", " + PyObject(repr)(PyObject.from_string(s)).to_string() + ": " + a_program(b) + "}"
    )
    yield rewrite(a_program(a[b]), subsume=True).to(a_program(a) + "[" + a_program(b) + "]")
    assigned_a = a_program(a).assign()
    setitem_a = copy(a)
    setitem_a[b] = c
    yield rewrite(a_program(setitem_a), subsume=True).to(
        assigned_a.statement(assigned_a + "[" + a_program(b) + "] = " + a_program(c))
    )
    delitem_a = copy(a)
    del delitem_a[b]
    yield rewrite(a_program(delitem_a), subsume=True).to(
        assigned_a.statement("del " + assigned_a + "[" + a_program(b) + "]")
    )
    yield rewrite(a_program(contains(a, b)), subsume=True).to(w(a_program(a) + " in " + a_program(b)))
    yield rewrite(a_program(a + b), subsume=True).to(w(a_program(a) + " + " + a_program(b)))
    yield rewrite(a_program(a - b), subsume=True).to(w(a_program(a) + " - " + a_program(b)))
    yield rewrite(a_program(a * b), subsume=True).to(w(a_program(a) + " * " + a_program(b)))
    yield rewrite(a_program(a @ b), subsume=True).to(w(a_program(a) + " @ " + a_program(b)))
    yield rewrite(a_program(a / b), subsume=True).to(w(a_program(a) + " / " + a_program(b)))
    yield rewrite(a_program(a // b), subsume=True).to(w(a_program(a) + " // " + a_program(b)))
    yield rewrite(a_program(a % b), subsume=True).to(w(a_program(a) + " % " + a_program(b)))
    yield rewrite(a_program(divmod(a, b)), subsume=True).to(ca("divmod", a_program(a), a_program(b)))
    yield rewrite(a_program(a**b), subsume=True).to(w(a_program(a) + " ** " + a_program(b)))
    yield rewrite(a_program(a << b), subsume=True).to(w(a_program(a) + " << " + a_program(b)))
    yield rewrite(a_program(a >> b), subsume=True).to(w(a_program(a) + " >> " + a_program(b)))
    yield rewrite(a_program(a & b), subsume=True).to(w(a_program(a) + " & " + a_program(b)))
    yield rewrite(a_program(a ^ b), subsume=True).to(w(a_program(a) + " ^ " + a_program(b)))
    yield rewrite(a_program(a | b), subsume=True).to(w(a_program(a) + " | " + a_program(b)))
    yield rewrite(a_program(-a), subsume=True).to("-" + a_program(a))
    yield rewrite(a_program(+a), subsume=True).to("+" + a_program(a))
    yield rewrite(a_program(abs(a)), subsume=True).to(ca("abs", a_program(a)))
    yield rewrite(a_program(complex_(a)), subsume=True).to(ca("complex", a_program(a)))
    yield rewrite(a_program(int_(a)), subsume=True).to(ca("int", a_program(a)))
    yield rewrite(a_program(float_(a)), subsume=True).to(ca("float", a_program(a)))
    yield rewrite(a_program(index(a)), subsume=True).to(ca("operator.index", a_program(a)))
    yield rewrite(a_program(round(a)), subsume=True).to(ca("round", a_program(a)))
    yield rewrite(a_program(math.trunc(a)), subsume=True).to(ca("math.trunc", a_program(a)))
    yield rewrite(a_program(math.floor(a)), subsume=True).to(ca("math.floor", a_program(a)))
    yield rewrite(a_program(math.ceil(a)), subsume=True).to(ca("math.ceil", a_program(a)))
    yield rewrite(a_program(list_(a)), subsume=True).to(ca("list", a_program(a)))
    yield rewrite(a_program(slice_(a, b, c)), subsume=True).to(ca("slice", a_program(a), a_program(b), a_program(c)))

    yield rewrite(a_program(not_(a)), subsume=True).to(w("not " + a_program(a)))
    yield rewrite(a_program(and_(a, b)), subsume=True).to(w(a_program(a) + " and " + a_program(b)))
    # # Given
    yield rewrite(a_program(given(a, b)), subsume=True).to(a_program(a).statement("assert " + a_program(b)))


any_program_schedule = any_program_ruleset.saturate() + program_gen_ruleset.saturate()


def any_expr_source(x: AnyExpr) -> str:
    x = x.egglog_any_expr_value
    # print(x)
    program = a_program(x)
    # print("program", program)
    egraph = EGraph()
    # program = egraph.let("program", program)
    egraph.register(program)
    egraph.run(any_program_ruleset.saturate())
    res_program = egraph.extract(program)
    egraph = EGraph()
    egraph.register(res_program.compile())
    egraph.run(program_gen_ruleset.saturate())
    # print(egraph.extract(program))
    # while egraph.run(any_program_ruleset).updated:
    #     print(egraph.extract(program))
    # print("extracted", egraph.extract(program))
    # egraph.run(program_gen_ruleset.saturate())
    res = join(res_program.statements, res_program.expr)
    return egraph.extract(res).value
    # egraph.display()
    # return black.format_str(str_res, mode=black.Mode()).strip()


x = AnyExpr([42])


print(x[0] + 10)
