import datetime
import json
import pathlib
import subprocess
from typing import Iterable

import pytest
from egg_smol.bindings import *
from egg_smol.commands import parse_and_run


def get_examples_files() -> Iterable[pathlib.Path]:
    """
    Return a list of egg examples files
    """
    metadata_process = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "-q"],
        capture_output=True,
        check=True,
    )
    metadata = json.loads(metadata_process.stdout)
    (egg_smol_package,) = [package for package in metadata["packages"] if package["name"] == "egg-smol"]
    egg_smol_folder = pathlib.Path(egg_smol_package["manifest_path"]).parent
    return (egg_smol_folder / "tests").glob("*.egg")


@pytest.mark.parametrize("example_file", [pytest.param(path, id=path.stem) for path in get_examples_files()])
def test_example(example_file: pathlib.Path):
    s = example_file.read_text()
    parse_and_run(s)


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

    def test_extract_string(self):
        egraph = EGraph()
        egraph.define("x", Lit(String("hello")))
        assert egraph.extract_expr(Var("x")) == (0, Lit(String("hello")), [])

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
            FunctionDecl("foo", Schema([], "i64"), merge=Call("max", [Var("old"), Var("new")])),
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

    def test_sort_alias(self):
        # From map example
        egraph = EGraph()
        egraph.declare_sort(
            "MyMap",
            ("Map", [Var("i64"), Var("String")]),
        )
        egraph.define(
            "my_map1",
            Call("insert", [Call("empty", []), Lit(Int(1)), Lit(String("one"))]),
        )
        egraph.define(
            "my_map2",
            Call("insert", [Var("my_map1"), Lit(Int(2)), Lit(String("two"))]),
        )

        egraph.check_fact(Eq([Lit(String("one")), Call("get", [Var("my_map1"), Lit(Int(1))])]))
        _, expr, _ = egraph.extract_expr(Var("my_map2"))
        assert expr == Call(
            "insert",
            [
                Call("insert", [Call("empty", []), Lit(Int(2)), Lit(String("two"))]),
                Lit(Int(1)),
                Lit(String("one")),
            ],
        )


def test_fib_demand():
    egraph = EGraph()
    # (datatype Expr
    #   (Num i64)
    #   (Add Expr Expr))
    egraph.declare_sort("Expr")
    egraph.declare_constructor(Variant("Num", ["i64"]), "Expr")
    egraph.declare_constructor(Variant("Add", ["Expr", "Expr"]), "Expr")
    # (function Fib (i64) Expr)
    egraph.declare_function(FunctionDecl("Fib", Schema(["i64"], "Expr")))
    # (rewrite (Add (Num a) (Num b)) (Num (+ a b)))
    egraph.add_rewrite(
        Rewrite(
            Call("Add", [Call("Num", [Var("a")]), Call("Num", [Var("b")])]),
            Call("Num", [Call("+", [Var("a"), Var("b")])]),
        )
    )
    # (rule ((= f (Fib x))
    #     (> x 1))
    #     ((set (Fib x) (Add (Fib (- x 1)) (Fib (- x 2))))))
    egraph.add_rule(
        Rule(
            [
                Set(
                    "Fib",
                    [Var("x")],
                    Call(
                        "Add",
                        [
                            Call("Fib", [Call("-", [Var("x"), Lit(Int(1))])]),
                            Call("Fib", [Call("-", [Var("x"), Lit(Int(2))])]),
                        ],
                    ),
                )
            ],
            [
                Eq([Var("f"), Call("Fib", [Var("x")])]),
                Fact(Call(">", [Var("x"), Lit(Int(1))])),
            ],
        )
    )
    # (set (Fib 0) (Num 0))
    egraph.eval_actions(Set("Fib", [Lit(Int(0))], Call("Num", [Lit(Int(0))])))
    # (set (Fib 1) (Num 1))
    egraph.eval_actions(Set("Fib", [Lit(Int(1))], Call("Num", [Lit(Int(1))])))
    # (define f7 (Fib 7))
    egraph.define("f7", Call("Fib", [Lit(Int(7))]))
    # (run 14)
    egraph.run_rules(14)
    # (extract f7)
    egraph.extract_expr(Var("f7"))
    # (check (= f7 (Num 13)))
    egraph.check_fact(Eq([Var("f7"), Call("Num", [Lit(Int(13))])]))


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


class TestParse:
    # TODO: Test all examples
    def test_parse_simple(self):
        res = parse(
            """(datatype Math
          (Num i64)
          (Var String)
          (Add Math Math)
          (Mul Math Math))

        ;; expr1 = 2 * (x + 3)
        (define expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))"""
        )

        assert res == [
            Datatype(
                "Math",
                [
                    Variant("Num", ["i64"]),
                    Variant("Var", ["String"]),
                    Variant("Add", ["Math", "Math"]),
                    Variant("Mul", ["Math", "Math"]),
                ],
            ),
            Define(
                "expr1",
                Call(
                    "Mul",
                    [
                        Call("Num", [Lit(Int(2))]),
                        Call(
                            "Add",
                            [
                                Call("Var", [Lit(String('"x"'))]),
                                Call("Num", [Lit(Int(3))]),
                            ],
                        ),
                    ],
                ),
                None,
            ),
        ]
