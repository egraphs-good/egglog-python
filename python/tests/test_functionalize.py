from collections.abc import Callable
from functools import partial
from inspect import signature
from typing import get_type_hints

from egglog.functionalize import functionalize


def get_annotation(x: object) -> type | None:
    return type(x) if x != 1 else None


x = "x"


def outer(y1: str, y2: int) -> Callable[[str, int], tuple[str, str, int, str, int]]:
    def inner(z1: str, z2: int) -> tuple[str, str, int, str, int]:
        return (x, y1, y2, z1, z2)

    return functionalize(inner, get_annotation)


res = outer("y1", 1)


def test_partial():
    assert isinstance(res, partial)
    assert res.args == ("x", "y1")


def test_signature():
    assert isinstance(res, partial)

    sig = signature(res.func)
    assert list(sig.parameters) == ["x", "y1", "z1", "z2"]


def test_annotations():
    assert isinstance(res, partial)

    annotations = get_type_hints(res.func)
    assert annotations == {
        "x": str,
        "y1": str,
        "z1": str,
        "z2": int,
        "return": tuple[str, str, int, str, int],
    }


def test_call():
    assert res("z1", 2) == ("x", "y1", 1, "z1", 2)


def test_call_again():
    assert res("z1_", 22) == ("x", "y1", 1, "z1_", 22)


def test_name():
    assert isinstance(res, partial)
    assert res.func.__name__ == "inner"
