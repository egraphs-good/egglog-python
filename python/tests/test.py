import datetime

import pytest
from egg_smol.bindings import *


class TestEGraph:
    def test_parse_and_run_program(self):
        program = "(check (= (+ 2 2) 4))"
        egraph = EGraph()

        assert egraph.parse_and_run_program(program) == ["Checked."]

    def test_parse_and_run_program_exception(self):
        program = "(check (= 5 4))"
        egraph = EGraph()

        with pytest.raises(
            EggSmolError,
            match='Check failed: Value { tag: "i64", bits: 5 } != Value { tag: "i64", bits: 4 }',
        ):
            egraph.parse_and_run_program(program)

    # These examples are from eqsat-basic
    def test_datatype(self):
        egraph = EGraph()
        egraph.declare_sort("Math")
        egraph.declare_constructor(Variant("Num", ["i64"]), "Math")
        egraph.declare_constructor(Variant("Var", ["String"]), "Math")
        egraph.declare_constructor(Variant("Add", ["Math", "Math"]), "Math")
        egraph.declare_constructor(Variant("Mul", ["Math", "Math"]), "Math")

    def test_define(self):
        egraph = EGraph()
        egraph.declare_sort("Math")
        egraph.declare_constructor(Variant("Num", ["i64"]), "Math")
        egraph.declare_constructor(Variant("Var", ["String"]), "Math")
        egraph.declare_constructor(Variant("Add", ["Math", "Math"]), "Math")
        egraph.declare_constructor(Variant("Mul", ["Math", "Math"]), "Math")

        # (define expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
        egraph.define(
            "expr1",
            Call(
                "Mul",
                [
                    Call(
                        "Num",
                        [
                            Lit(Int(2)),
                        ],
                    ),
                    Call(
                        "Add",
                        [
                            Call(
                                "Var",
                                [
                                    Lit(String("x")),
                                ],
                            ),
                            Call(
                                "Num",
                                [
                                    Lit(Int(3)),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )

    def test_rewrite(self):
        egraph = EGraph()
        egraph.declare_sort("Math")
        egraph.declare_constructor(Variant("Add", ["Math", "Math"]), "Math")
        name = egraph.add_rewrite(
            Rewrite(
                Call(
                    "Add",
                    [
                        Var("a"),
                        Var("b"),
                    ],
                ),
                Call(
                    "Add",
                    [
                        Var("b"),
                        Var("a"),
                    ],
                ),
            )
        )
        assert isinstance(name, str)

    def test_run_rules(self):
        egraph = EGraph()
        egraph.declare_sort("Math")
        egraph.declare_constructor(Variant("Add", ["Math", "Math"]), "Math")
        egraph.add_rewrite(
            Rewrite(
                Call(
                    "Add",
                    [
                        Var("a"),
                        Var("b"),
                    ],
                ),
                Call(
                    "Add",
                    [
                        Var("b"),
                        Var("a"),
                    ],
                ),
            )
        )
        start_time = datetime.datetime.now()
        searching, applying, rebuilding = egraph.run_rules(10)
        end_time = datetime.datetime.now()
        assert isinstance(searching, datetime.timedelta)
        assert isinstance(applying, datetime.timedelta)
        assert isinstance(rebuilding, datetime.timedelta)
        total_measured_time = searching + applying + rebuilding
        # Verify  less than the total time (which includes time spent in Python).
        assert total_measured_time < (end_time - start_time)

    def test_check_fact(self):
        egraph = EGraph()
        egraph.declare_sort("Math")
        egraph.declare_constructor(Variant("Num", ["i64"]), "Math")
        egraph.declare_constructor(Variant("Var", ["String"]), "Math")
        egraph.declare_constructor(Variant("Add", ["Math", "Math"]), "Math")
        egraph.declare_constructor(Variant("Mul", ["Math", "Math"]), "Math")

        # (define expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
        egraph.define(
            "expr1",
            Call(
                "Mul",
                [
                    Call(
                        "Num",
                        [
                            Lit(Int(2)),
                        ],
                    ),
                    Call(
                        "Add",
                        [
                            Call(
                                "Var",
                                [
                                    Lit(String("x")),
                                ],
                            ),
                            Call(
                                "Num",
                                [
                                    Lit(Int(3)),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
        # (define expr2 (Add (Num 6) (Mul (Num 2) (Var "x"))))
        egraph.define(
            "expr2",
            Call(
                "Add",
                [
                    Call(
                        "Num",
                        [
                            Lit(Int(6)),
                        ],
                    ),
                    Call(
                        "Mul",
                        [
                            Call(
                                "Num",
                                [
                                    Lit(Int(2)),
                                ],
                            ),
                            Call(
                                "Var",
                                [
                                    Lit(String("x")),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
        # (rewrite (Add a b)
        #          (Add b a))
        egraph.add_rewrite(
            Rewrite(
                Call(
                    "Add",
                    [
                        Var("a"),
                        Var("b"),
                    ],
                ),
                Call(
                    "Add",
                    [
                        Var("b"),
                        Var("a"),
                    ],
                ),
            )
        )
        # (rewrite (Mul a (Add b c))
        #          (Add (Mul a b) (Mul a c)))
        egraph.add_rewrite(
            Rewrite(
                Call(
                    "Mul",
                    [
                        Var("a"),
                        Call(
                            "Add",
                            [
                                Var("b"),
                                Var("c"),
                            ],
                        ),
                    ],
                ),
                Call(
                    "Add",
                    [
                        Call(
                            "Mul",
                            [
                                Var("a"),
                                Var("b"),
                            ],
                        ),
                        Call(
                            "Mul",
                            [
                                Var("a"),
                                Var("c"),
                            ],
                        ),
                    ],
                ),
            )
        )

        # (rewrite (Add (Num a) (Num b))
        #          (Num (+ a b)))
        lhs = Call(
            "Add",
            [
                Call(
                    "Num",
                    [
                        Var("a"),
                    ],
                ),
                Call(
                    "Num",
                    [
                        Var("b"),
                    ],
                ),
            ],
        )
        rhs = Call(
            "Num",
            [
                Call(
                    "+",
                    [
                        Var("a"),
                        Var("b"),
                    ],
                )
            ],
        )
        egraph.add_rewrite(Rewrite(lhs, rhs))

        # (rewrite (Mul (Num a) (Num b))
        #          (Num (* a b)))
        lhs = Call(
            "Mul",
            [
                Call(
                    "Num",
                    [
                        Var("a"),
                    ],
                ),
                Call(
                    "Num",
                    [
                        Var("b"),
                    ],
                ),
            ],
        )
        rhs = Call(
            "Num",
            [
                Call(
                    "*",
                    [
                        Var("a"),
                        Var("b"),
                    ],
                )
            ],
        )
        egraph.add_rewrite(Rewrite(lhs, rhs))

        egraph.run_rules(10)
        egraph.check_fact(
            Eq(
                [
                    Var("expr1"),
                    Var("expr2"),
                ]
            )
        )

    def test_extract(self):
        # Example from extraction-cost
        egraph = EGraph()
        egraph.declare_sort("Expr")
        egraph.declare_constructor(Variant("Num", ["i64"], cost=5), "Expr")

        egraph.define("x", Call("Num", [Lit(Int(1))]), cost=10)
        egraph.define("y", Call("Num", [Lit(Int(2))]), cost=1)

        assert egraph.extract_expr(Var("x")) == (6, Call("Num", [Lit(Int(1))]), [])
        assert egraph.extract_expr(Var("y")) == (1, Call("y", []), [])

    def test_rule(self):
        # Example from fibonacci
        egraph = EGraph()
        egraph.declare_function(FunctionDecl("fib", Schema(["i64"], "i64")))
        egraph.eval_actions(Set("fib", [Lit(Int(0))], Lit(Int(0))))
        egraph.eval_actions(Set("fib", [Lit(Int(1))], Lit(Int(1))))
        egraph.add_rule(
            Rule(
                body=[
                    Eq([Var("f0"), Call("fib", [Var("x")])]),
                    Eq([Var("f1"), Call("fib", [Call("+", [Var("x"), Lit(Int(1))])])]),
                ],
                head=[
                    Set(
                        "fib",
                        [Call("+", [Var("x"), Lit(Int(2))])],
                        Call("+", [Var("f0"), Var("f1")]),
                    ),
                ],
            )
        )
        egraph.run_rules(7)
        egraph.check_fact(Eq([Call("fib", [Lit(Int(7))]), Lit(Int(13))]))

    def test_push_pop(self):
        egraph = EGraph()
        egraph.declare_function(
            FunctionDecl(
                "foo", Schema([], "i64"), merge=Call("max", [Var("old"), Var("new")])
            ),
        )
        egraph.eval_actions(Set("foo", [], Lit(Int(1))))
        egraph.check_fact(Eq([Call("foo", []), Lit(Int(1))]))

        egraph.push()
        egraph.eval_actions(Set("foo", [], Lit(Int(2))))
        egraph.check_fact(Eq([Call("foo", []), Lit(Int(2))]))
        egraph.pop()

        egraph.check_fact(Eq([Call("foo", []), Lit(Int(1))]))

    def test_clear(self):
        egraph = EGraph()
        egraph.define("x", Lit(Int(1)))
        egraph.check_fact(Eq([Call("x", []), Lit(Int(1))]))

        egraph.clear()
        with pytest.raises(EggSmolError):
            egraph.check_fact(Eq([Call("x", []), Lit(Int(1))]))

    def test_clear_rules(self):
        egraph = EGraph()
        egraph.declare_function(FunctionDecl("fib", Schema(["i64"], "i64")))
        egraph.eval_actions(Set("fib", [Lit(Int(0))], Lit(Int(0))))
        egraph.eval_actions(Set("fib", [Lit(Int(1))], Lit(Int(1))))
        egraph.add_rule(
            Rule(
                body=[
                    Eq([Var("f0"), Call("fib", [Var("x")])]),
                    Eq([Var("f1"), Call("fib", [Call("+", [Var("x"), Lit(Int(1))])])]),
                ],
                head=[
                    Set(
                        "fib",
                        [Call("+", [Var("x"), Lit(Int(2))])],
                        Call("+", [Var("f0"), Var("f1")]),
                    ),
                ],
            )
        )
        egraph.clear_rules()
        egraph.run_rules(7)
        with pytest.raises(EggSmolError):
            egraph.check_fact(Eq([Call("fib", [Lit(Int(7))]), Lit(Int(13))]))

    def test_print_size(self):
        egraph = EGraph()
        egraph.declare_function(FunctionDecl("fib", Schema(["i64"], "i64")))
        egraph.eval_actions(Set("fib", [Lit(Int(0))], Lit(Int(0))))
        egraph.eval_actions(Set("fib", [Lit(Int(1))], Lit(Int(1))))
        assert egraph.print_size("fib") == "Function fib has size 2"

    def test_print_function(self):
        egraph = EGraph()
        egraph.declare_function(FunctionDecl("fib", Schema(["i64"], "i64")))
        egraph.eval_actions(Set("fib", [Lit(Int(0))], Lit(Int(0))))
        egraph.eval_actions(Set("fib", [Lit(Int(1))], Lit(Int(1))))
        assert egraph.print_function("fib", 2) == "(fib 0) -> 0\n(fib 1) -> 1\n"


class TestVariant:
    def test_repr(self):
        assert repr(Variant("name", [])) == "Variant('name', [], None)"

    def test_name(self):
        assert Variant("name", []).name == "name"

    def test_types(self):
        assert Variant("name", ["a", "b"]).types == ["a", "b"]

    def test_cost(self):
        assert Variant("name", [], cost=1).cost == 1

    def test_compare(self):
        assert Variant("name", []) == Variant("name", [])
        assert Variant("name", []) != Variant("name", ["a"])
        assert Variant("name", []) != 10  # type: ignore
