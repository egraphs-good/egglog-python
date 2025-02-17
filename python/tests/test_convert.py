from typing import Generic, TypeVar

import pytest

from egglog import *

# TODO: Revert global conversion state after each test w/ fixture


def test_conversion_custom_metaclass():
    class MyMeta(type):
        pass

    class MyType(metaclass=MyMeta):
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    converter(MyMeta, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion():
    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion_transitive_forward():
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


T = TypeVar("T", bound=BaseExpr)


def test_convert_to_generic():
    """
    Tests that if you have a conversion from a python type to a generic type, it will work for a
    particular instance of that generic even if the general instance is registered
    """

    class G(BuiltinExpr, Generic[T]):
        def __init__(self, x: T) -> None: ...

    converter(i64, G[i64], lambda x: G(x))
    assert expr_parts(convert(10, G[i64])) == expr_parts(G(i64(10)))

    with pytest.raises(ConvertError):
        convert(10, G[String])

    with pytest.raises(ConvertError):
        convert("hi", G[i64])


def test_convert_to_unbound_generic():
    """
    Tests that if you have a conversion from a python type to a generic type, it will work for a
    particular instance of that generic even if the general instance is registered
    """

    class G(BuiltinExpr, Generic[T]):
        def __init__(self, x: i64) -> None: ...

    converter(i64, G, lambda x: G[get_type_args()[0]](x))  # type: ignore[misc, operator]
    assert expr_parts(convert(10, G[String])) == expr_parts(G[String](i64(10)))


def test_convert_generic_transitive():
    """
    If we have A -> B and B[C] -> D then we should
    have A -> D conversion
    """

    class A(Expr):
        def __init__(self) -> None: ...

    class B(BuiltinExpr, Generic[T]):
        def __init__(
            self,
        ) -> None: ...

    class C(Expr): ...

    class D(Expr):
        def __init__(self) -> None: ...

    converter(A, B, lambda _: B[get_type_args()[0]]())  # type: ignore[misc, operator]
    converter(B[C], D, lambda _: D())

    assert expr_parts(convert(A(), D)) == expr_parts(D())
