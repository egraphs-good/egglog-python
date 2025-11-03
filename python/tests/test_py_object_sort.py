from __future__ import annotations

import dataclasses
import json
from base64 import standard_b64decode, standard_b64encode
from typing import TYPE_CHECKING

from cloudpickle import dumps, loads

from egglog.bindings import *

if TYPE_CHECKING:
    from egglog.bindings import _Expr

DUMMY_SPAN = RustSpan(__name__, 0, 0)


@dataclasses.dataclass(frozen=True)
class MyObject:
    a: int = 10


def py_object_to_expr(obj: object) -> Call:
    obj_bytes = dumps(obj)
    return Call(
        DUMMY_SPAN,
        "py-object",
        [Lit(DUMMY_SPAN, String(standard_b64encode(obj_bytes).decode("utf-8")))],
    )


def expr_to_py_object(expr: _Expr) -> object:
    assert isinstance(expr, Call)
    assert expr.name == "py-object"
    assert len(expr.args) == 1
    arg = expr.args[0]
    assert isinstance(arg, Lit)
    assert isinstance(arg.value, String)
    obj_bytes = standard_b64decode(arg.value.value.encode("utf-8"))
    return loads(obj_bytes)


class TestDictUpdate:
    # Test that (py-dict-update dict key value key2 value2) works
    def test_dict_update(self):
        initial_dict = {"a": 1}
        dict_expr = py_object_to_expr(initial_dict)
        new_value_expr = py_object_to_expr(2)
        a_expr = py_object_to_expr("a")
        b_expr = py_object_to_expr("b")
        egraph = EGraph()
        res = egraph.run_program(
            ActionCommand(
                Let(
                    DUMMY_SPAN,
                    "new_dict",
                    Call(DUMMY_SPAN, "py-dict-update", [dict_expr, a_expr, new_value_expr, b_expr, new_value_expr]),
                )
            ),
            Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "new_dict"), Lit(DUMMY_SPAN, Int(0))),
        )
        assert len(res) == 1
        report = res[0]
        assert isinstance(report, ExtractBest)
        expr = report.termdag.term_to_expr(report.term, DUMMY_SPAN)
        assert expr_to_py_object(expr) == {"a": 2, "b": 2}

        # Verify that the original dict is unchanged
        assert initial_dict == {"a": 1}


def my_add(a, b):
    return a + b


class TestEval:
    def test_eval(self):
        egraph = EGraph()
        one = py_object_to_expr(1)
        two = py_object_to_expr(2)
        globals_ = py_object_to_expr({"my_add": my_add})
        locals_ = py_object_to_expr({})
        x_expr = py_object_to_expr("x")
        y_expr = py_object_to_expr("y")
        res = egraph.run_program(
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
            Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), Lit(DUMMY_SPAN, Int(0))),
        )
        assert len(res) == 1
        report = res[0]
        assert isinstance(report, ExtractBest)
        expr = report.termdag.term_to_expr(report.term, DUMMY_SPAN)
        assert expr_to_py_object(expr) == 3


class TestConversion:
    def test_to_string(self):
        """
        Verify (py-to-string "hi")
        """
        egraph = EGraph()

        egraph.run_program(
            ActionCommand(Let(DUMMY_SPAN, "res", Call(DUMMY_SPAN, "py-to-string", [py_object_to_expr("hi")]))),
            Check(DUMMY_SPAN, [Eq(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), Lit(DUMMY_SPAN, String("hi")))]),
        )

    def test_from_string(self):
        """
        Verify (py-from-string "hi")
        """
        egraph = EGraph()
        egraph.run_program(
            ActionCommand(Let(DUMMY_SPAN, "res", Call(DUMMY_SPAN, "py-from-string", [Lit(DUMMY_SPAN, String("hi"))]))),
            Check(DUMMY_SPAN, [Eq(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), py_object_to_expr("hi"))]),
        )


def test_call():
    """
    Verify (py-call fn arg1 arg2 ...)
    """
    egraph = EGraph()
    fn_expr = py_object_to_expr(my_add)
    one_expr = py_object_to_expr(1)
    two_expr = py_object_to_expr(2)
    egraph.run_program(
        ActionCommand(
            Let(
                DUMMY_SPAN,
                "res",
                Call(DUMMY_SPAN, "py-call", [fn_expr, one_expr, two_expr]),
            )
        ),
        Check(DUMMY_SPAN, [Eq(DUMMY_SPAN, Var(DUMMY_SPAN, "res"), py_object_to_expr(3))]),
    )


def test_serialize_string():
    """
    Verify that when serializing the e-graph, PyObjects are turned into their repr

    Creates a relation that takes a py object, and add this to the e-graph
    """
    egraph = EGraph()
    obj = "my object"
    obj_expr = py_object_to_expr(obj)
    expr = Expr_(DUMMY_SPAN, Call(DUMMY_SPAN, "my_relation", [obj_expr]))
    egraph.run_program(Relation(DUMMY_SPAN, "my_relation", ["PyObject"]), ActionCommand(expr))
    serialized = egraph.serialize([])
    serialized_value = json.loads(serialized.to_json())
    # Verify that one of the s['node'][...]['op'] is the repr of obj
    ops = [node["op"] for node in serialized_value["nodes"].values()]
    assert repr(obj) in ops
