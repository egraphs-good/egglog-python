import dataclasses
import gc
import weakref

import pytest
from egglog.bindings import *


@dataclasses.dataclass(frozen=True)
class MyObject:
    a: int = 10


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(10, id="hashable"),
        pytest.param([1, 2, 3], id="unhashable"),
        pytest.param(MyObject(), id="custom"),
    ],
)
def test_adding_retrieving_object(obj: object):
    egraph = EGraph()
    expr = egraph.save_object(obj)
    assert egraph.load_object(expr) == obj


def test_objects_cleaned_up():
    egraph = EGraph()
    my_object = MyObject()

    egraph.save_object(my_object)
    ref = weakref.ref(my_object)
    del my_object, egraph
    gc.collect()
    assert ref() is None


def test_object_keeps_ref():
    egraph = EGraph()
    my_object = MyObject()
    ref = weakref.ref(my_object)

    expr = egraph.save_object(my_object)
    del my_object
    gc.collect()
    assert ref() is not None
    assert egraph.load_object(expr) == MyObject()
