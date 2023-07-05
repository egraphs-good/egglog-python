import dataclasses
import gc
import weakref

import pytest
from egglog.bindings import *


@dataclasses.dataclass(frozen=True)
class MyObject:
    a: int = 10


class TestSaveLoad:
    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param(10, id="hashable"),
            pytest.param([1, 2, 3], id="unhashable"),
            pytest.param(MyObject(), id="custom"),
        ],
    )
    def test_adding_retrieving_object(self, obj: object):
        egraph = EGraph()
        expr = egraph.save_object(obj)
        assert egraph.load_object(expr) == obj

    def test_objects_cleaned_up(self):
        egraph = EGraph()
        my_object = MyObject()

        egraph.save_object(my_object)
        ref = weakref.ref(my_object)
        del my_object, egraph
        gc.collect()
        assert ref() is None

    def test_object_keeps_ref(self):
        egraph = EGraph()
        my_object = MyObject()
        ref = weakref.ref(my_object)

        expr = egraph.save_object(my_object)
        del my_object
        gc.collect()
        assert ref() is not None
        assert egraph.load_object(expr) == MyObject()


class TestDictUpdate:
    # Test that (py-dict-update dict key value key2 value2) works
    def test_dict_update(self):
        egraph = EGraph()

        initial_dict = {"a": 1}
        dict_expr = egraph.save_object(initial_dict)
        new_value_expr = egraph.save_object(2)
        egraph.run_program(
            Define(
                "new_dict",
                Call("py-dict-update", [dict_expr, Lit(String("a")), new_value_expr, Lit(String("b")), new_value_expr]),
                None,
            ),
            Extract(1, Var("new_dict")),
        )
        extract_report = egraph.extract_report()
        assert extract_report
        res = egraph.load_object(extract_report.expr)
        assert res == {"a": 2, "b": 2}

        # Verify that the original dict is unchanged
        assert initial_dict == {"a": 1}


def my_add(a, b):
    return a + b


class TestEval:
    # Test that (py-eval "my_add(x, y)" globals {**locals, "x": 1, "y": 2}) works
    def test_eval(self):
        egraph = EGraph()
        one = egraph.save_object(1)
        two = egraph.save_object(2)
        globals_ = egraph.save_object(globals())
        locals_ = egraph.save_object(locals())
        egraph.run_program(
            Define(
                "res",
                Call(
                    "py-eval",
                    [
                        Lit(String("my_add(x, y)")),
                        globals_,
                        Call("py-dict-update", [locals_, Lit(String("x")), one, Lit(String("y")), two]),
                    ],
                ),
                None,
            ),
            Extract(1, Var("res")),
        )
        extract_report = egraph.extract_report()
        assert extract_report
        res = egraph.load_object(extract_report.expr)
        assert res == 3
