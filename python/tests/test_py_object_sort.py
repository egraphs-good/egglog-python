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
        sort = PyObjectSort()
        egraph = EGraph(sort)
        expr = sort.store(obj)
        assert egraph.eval_py_object(expr) == obj

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
        assert EGraph(sort).eval_py_object(expr)== MyObject()


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
                    "new_dict",
                    Call("py-dict-update", [dict_expr, a_expr, new_value_expr, b_expr, new_value_expr]),
                )
            )
        )
        assert egraph.eval_py_object(Var("new_dict")) == {"a": 2, "b": 2}

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
                    "res",
                    Call(
                        "py-eval",
                        [
                            Lit(String("my_add(x, y)")),
                            globals_,
                            Call("py-dict-update", [locals_, x_expr, one, y_expr, two]),
                        ]
                    )
                )
            )
        )
        assert egraph.eval_py_object(Var("res")) == 3


class TestConversion:
    def test_to_string(self):
        """
        Verify (py-to-string "hi")
        """
        sort = PyObjectSort()
        egraph = EGraph(sort)

        egraph.run_program(
            ActionCommand(Let("res", Call("py-to-string", [sort.store("hi")])))
        )
        assert egraph.eval_string(Var("res")) == "hi"

    def test_from_string(self):
        """
        Verify (py-from-string "hi")
        """
        sort = PyObjectSort()
        egraph = EGraph(sort)
        egraph.run_program(
            ActionCommand(Let("res", Call("py-from-string", [Lit(String("hi"))]))),
        )
        assert egraph.eval_py_object(Var("res")) == "hi"

