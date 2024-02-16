from egglog import *


def test_conversion_custom_metaclass():
    class MyMeta(type):
        pass

    class MyType(metaclass=MyMeta):
        pass

    EGraph()

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    converter(MyMeta, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion():
    EGraph()

    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion_transitive_forward():
    EGraph()

    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    class MyTypeExpr2(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())

    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())


def test_conversion_transitive_backward():
    EGraph()

    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    class MyTypeExpr2(Expr):
        def __init__(self) -> None: ...

    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())
    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())


def test_conversion_transitive_cycle():
    EGraph()

    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    class MyTypeExpr2(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())
    converter(MyTypeExpr2, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())
