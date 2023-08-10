import copy

import egglog.runtime
import pytest
from egglog import *


@pytest.fixture(autouse=True)
def reset_conversions():
    old_conversions = copy.copy(egglog.runtime.CONVERSIONS)
    yield
    egglog.runtime.CONVERSIONS = old_conversions


def test_conversion_custom_metaclass():
    class MyMeta(type):
        pass

    class MyType(metaclass=MyMeta):
        pass

    egraph = EGraph()

    @egraph.class_
    class MyTypeExpr(Expr):
        def __init__(self):
            ...

    converter(MyMeta, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion():
    egraph = EGraph()

    class MyType:
        pass

    @egraph.class_
    class MyTypeExpr(Expr):
        def __init__(self):
            ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion_transitive_forward():
    egraph = EGraph()

    class MyType:
        pass

    @egraph.class_
    class MyTypeExpr(Expr):
        def __init__(self):
            ...

    @egraph.class_
    class MyTypeExpr2(Expr):
        def __init__(self):
            ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())

    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())


def test_conversion_transitive_backward():
    egraph = EGraph()

    class MyType:
        pass

    @egraph.class_
    class MyTypeExpr(Expr):
        def __init__(self):
            ...

    @egraph.class_
    class MyTypeExpr2(Expr):
        def __init__(self):
            ...

    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())
    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())


def test_conversion_transitive_cycle():
    egraph = EGraph()

    class MyType:
        pass

    @egraph.class_
    class MyTypeExpr(Expr):
        def __init__(self):
            ...

    @egraph.class_
    class MyTypeExpr2(Expr):
        def __init__(self):
            ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())
    converter(MyTypeExpr2, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())
