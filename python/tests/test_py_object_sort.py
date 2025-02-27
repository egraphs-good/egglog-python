import dataclasses
import gc
import weakref

import pytest

from egglog.bindings import *

DUMMY_SPAN = RustSpan(__name__, 0, 0)


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
        sort = PyObjectSort()
        expr = sort.store(obj)
        assert sort.load(expr) == obj

    def test_objects_cleaned_up(self):
        sort = PyObjectSort()
        my_object = MyObject()

        sort.store(my_object)
        ref = weakref.ref(my_object)
        del my_object, sort
        gc.collect()
        assert ref() is None

    def test_object_keeps_ref(self):
        sort = PyObjectSort()
        my_object = MyObject()
        ref = weakref.ref(my_object)

        expr = sort.store(my_object)
        del my_object
        gc.collect()
        assert ref() is not None
        assert sort.load(expr) == MyObject()


class TestDictUpdate:
    # Test that (py-dict-update dict key value key2 value2) works
    def test_dict_update(self):
        sort = PyObjectSort()
        egraph = EGraph(sort)

        initial_dict = {"a": 1}
        dict_expr = sort.store(initial_dict)
        new_value_expr = sort.store(2)
        a_expr = sort.store("a")
        b_expr = sort.store("b")
        egraph.run_program(
            ActionCommand(
                Let(
                    DUMMY_SPAN,
                    "new_dict",
                    Call(DUMMY_SPAN, "py-dict-update", [dict_expr, a_expr, new_value_expr, b_expr, new_value_expr]),
                )
            ),
            ActionCommand(Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "new_dict"), Lit(DUMMY_SPAN, Int(0)))),
        )
        report = egraph.extract_report()
        assert isinstance(report, Best)
        expr = report.termdag.term_to_expr(report.term, DUMMY_SPAN)
        assert sort.load(expr) == {"a": 2, "b": 2}

        # Verify that the original dict is unchanged
        assert initial_dict == {"a": 1}


def my_add(a, b):
    return a + b


class TestEval:
    # Test that (py-eval "my_add(x, y)" globals {**locals, "x": 1, "y": 2}) works
    def test_eval(self):
        sort = PyObjectSort()
        egraph = EGraph(sort)
        one = sort.store(1)
        two = sort.store(2)
        globals_ = sort.store(globals())
        locals_ = sort.store(locals())
        x_expr = sort.store("x")
        y_expr = sort.store("y")
        egraph.run_program(
            ActionCommand(
                Let(
                    DUMMY_SPAN,
                    "res",
                    Call(
                        DUMMY_SPAN,
                        "py-eval",
                        [
                            Lit(DUMMY_SPAN, String("my_add(x, y)")),
                            globals_,
                            Call(DUMMY_SPAN, "py-dict-update", [locals_, x_expr, one, y_expr, two]),
                        ],
                    ),
                )
            ),
            ActionCommand(Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), Lit(DUMMY_SPAN, Int(0)))),
        )
        report = egraph.extract_report()
        assert isinstance(report, Best)
        expr = report.termdag.term_to_expr(report.term, DUMMY_SPAN)
        assert sort.load(expr) == 3


class TestConversion:
    def test_to_string(self):
        """
        Verify (py-to-string "hi")
        """
        sort = PyObjectSort()
        egraph = EGraph(sort)

        egraph.run_program(
            ActionCommand(Let(DUMMY_SPAN, "res", Call(DUMMY_SPAN, "py-to-string", [sort.store("hi")]))),
            Check(DUMMY_SPAN, [Eq(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), Lit(DUMMY_SPAN, String("hi")))]),
        )

    def test_from_string(self):
        """
        Verify (py-from-string "hi")
        """
        sort = PyObjectSort()
        egraph = EGraph(sort)
        egraph.run_program(
            ActionCommand(Let(DUMMY_SPAN, "res", Call(DUMMY_SPAN, "py-from-string", [Lit(DUMMY_SPAN, String("hi"))]))),
            Check(DUMMY_SPAN, [Eq(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), sort.store("hi"))]),
        )
