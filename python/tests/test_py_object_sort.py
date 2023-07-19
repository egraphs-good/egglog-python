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
        a_expr = egraph.save_object("a")
        b_expr = egraph.save_object("b")
        egraph.run_program(
            Define(
                "new_dict",
                Call("py-dict-update", [dict_expr, a_expr, new_value_expr, b_expr, new_value_expr]),
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
        x_expr = egraph.save_object("x")
        y_expr = egraph.save_object("y")
        egraph.run_program(
            Define(
                "res",
                Call(
                    "py-eval",
                    [
                        Lit(String("my_add(x, y)")),
                        globals_,
                        Call("py-dict-update", [locals_, x_expr, one, y_expr, two]),
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


class TestConversion:
    def test_to_string(self):
        """
        Verify (py-to-string "hi")
        """
        egraph = EGraph()
        hi = egraph.save_object("hi")

        egraph.run_program(
            Define("res", Call("py-to-string", [hi]), None),
            Extract(1, Var("res")),
        )
        extract_report = egraph.extract_report()
        assert extract_report
        assert extract_report.expr == Lit(String("hi"))

    def test_from_string(self):
        """
        Verify (py-from-string "hi")
        """
        egraph = EGraph()

        egraph.run_program(
            Define("res", Call("py-from-string", [Lit(String("hi"))]), None),
            Extract(1, Var("res")),
        )
        extract_report = egraph.extract_report()
        assert extract_report
        res = egraph.load_object(extract_report.expr)
        assert res == "hi"