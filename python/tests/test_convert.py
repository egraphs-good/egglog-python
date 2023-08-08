import numpy as np
from egglog import *
from egglog.runtime import CONVERSIONS


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
    print(CONVERSIONS)
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
