import _thread
import fractions
import json
import os
import pathlib
import subprocess

import pytest

from egglog.bindings import *


def get_egglog_folder() -> pathlib.Path:
    """
    Return the egglog source folder
    """
    metadata_process = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "-q"],  # noqa: S607
        capture_output=True,
        check=True,
    )
    metadata = json.loads(metadata_process.stdout)
    (egglog_package,) = (package for package in metadata["packages"] if package["name"] == "egglog")
    return pathlib.Path(egglog_package["manifest_path"]).parent


EGG_SMOL_FOLDER = get_egglog_folder()

SLOW_TESTS = ["repro-unsound"]


@pytest.mark.parametrize(
    "example_file",
    [
        pytest.param(path, id=path.stem, marks=pytest.mark.slow if path.stem in SLOW_TESTS else [])
        for path in (EGG_SMOL_FOLDER / "tests").glob("*.egg")
    ],
)
def test_example(example_file: pathlib.Path):
    egraph = EGraph(fact_directory=EGG_SMOL_FOLDER)
    commands = egraph.parse_program(example_file.read_text())
    # TODO: Include currently relies on the CWD instead of the fact directory. We should fix this upstream
    # and then remove this workaround.
    os.chdir(EGG_SMOL_FOLDER)
    egraph.run_program(*commands)


class TestEGraph:
    def test_parse_program(self, snapshot_py):
        res = EGraph().parse_program(
            """(datatype Math
          (Num i64)
          (Var String)
          (Add Math Math)
          (Mul Math Math))

        ;; expr1 = 2 * (x + 3)
        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))""",
            filename="test.egg",
        )
        assert res == snapshot_py

    def test_parse_and_run_program(self):
        program = "(check (= (+ 2 2) 4))"
        egraph = EGraph()

        assert egraph.run_program(*egraph.parse_program(program)) == []

    def test_parse_and_run_program_exception(self):
        program = "(check (= 1 1.0))"
        egraph = EGraph()

        with pytest.raises(
            EggSmolError,
            match="Type mismatch",
        ):
            egraph.run_program(*egraph.parse_program(program))

    def test_run_rules(self):
        egraph = EGraph()
        egraph.run_program(
            Datatype("Math", [Variant("Add", ["Math", "Math"])]),
            RewriteCommand(
                "",
                Rewrite(
                    DUMMY_SPAN,
                    Call(DUMMY_SPAN, "Add", [Var(DUMMY_SPAN, "a"), Var(DUMMY_SPAN, "b")]),
                    Call(DUMMY_SPAN, "Add", [Var(DUMMY_SPAN, "b"), Var(DUMMY_SPAN, "a")]),
                ),
                False,
            ),
            RunSchedule(Repeat(DUMMY_SPAN, 10, Run(DUMMY_SPAN, RunConfig("")))),
        )

        run_report = egraph.run_report()
        assert isinstance(run_report, RunReport)

    def test_extract(self):
        # Example from extraction-cost
        egraph = EGraph()
        egraph.run_program(
            Datatype("Expr", [Variant("Num", ["i64"], cost=5)]),
            ActionCommand(Let(DUMMY_SPAN, "x", Call(DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]))),
            QueryExtract(0, Var(DUMMY_SPAN, "x")),
        )

        extract_report = egraph.extract_report()
        assert isinstance(extract_report, Best)
        assert extract_report.cost == 6
        assert termdag_term_to_expr(extract_report.termdag, extract_report.term) == Call(
            DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]
        )

    def test_simplify(self):
        egraph = EGraph()
        egraph.run_program(
            Datatype("Expr", [Variant("Num", ["i64"], cost=5)]),
            ActionCommand(Let(DUMMY_SPAN, "x", Call(DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]))),
            Simplify(Var(DUMMY_SPAN, "x"), Run(DUMMY_SPAN, RunConfig(""))),
        )

        extract_report = egraph.extract_report()
        assert isinstance(extract_report, Best)
        assert extract_report.cost == 6
        assert termdag_term_to_expr(extract_report.termdag, extract_report.term) == Call(
            DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]
        )

    def test_sort_alias(self):
        # From map example
        egraph = EGraph()
        egraph.run_program(
            Sort(
                "MyMap",
                ("Map", [Var(DUMMY_SPAN, "i64"), Var(DUMMY_SPAN, "String")]),
            ),
            ActionCommand(
                Let(
                    DUMMY_SPAN,
                    "my_map1",
                    Call(
                        DUMMY_SPAN,
                        "map-insert",
                        [Call(DUMMY_SPAN, "map-empty", []), Lit(DUMMY_SPAN, Int(1)), Lit(DUMMY_SPAN, String("one"))],
                    ),
                )
            ),
            ActionCommand(
                Let(
                    DUMMY_SPAN,
                    "my_map2",
                    Call(
                        DUMMY_SPAN,
                        "map-insert",
                        [Var(DUMMY_SPAN, "my_map1"), Lit(DUMMY_SPAN, Int(2)), Lit(DUMMY_SPAN, String("two"))],
                    ),
                )
            ),
            Check(
                DUMMY_SPAN,
                [
                    Eq(
                        DUMMY_SPAN,
                        [
                            Lit(DUMMY_SPAN, String("one")),
                            Call(DUMMY_SPAN, "map-get", [Var(DUMMY_SPAN, "my_map1"), Lit(DUMMY_SPAN, Int(1))]),
                        ],
                    )
                ],
            ),
            ActionCommand(Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "my_map2"), Lit(DUMMY_SPAN, Int(0)))),
        )

        extract_report = egraph.extract_report()
        assert isinstance(extract_report, Best)
        assert termdag_term_to_expr(extract_report.termdag, extract_report.term) == Call(
            DUMMY_SPAN,
            "map-insert",
            [
                Call(
                    DUMMY_SPAN,
                    "map-insert",
                    [Call(DUMMY_SPAN, "map-empty", []), Lit(DUMMY_SPAN, Int(2)), Lit(DUMMY_SPAN, String("two"))],
                ),
                Lit(DUMMY_SPAN, Int(1)),
                Lit(DUMMY_SPAN, String("one")),
            ],
        )
        assert extract_report.cost == 4


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
        assert Variant("name", []) != 10  # type: ignore[comparison-overlap]


class TestEval:
    def test_i64(self):
        assert EGraph().eval_i64(Lit(DUMMY_SPAN, Int(1))) == 1

    def test_f64(self):
        assert EGraph().eval_f64(Lit(DUMMY_SPAN, F64(1.0))) == 1.0

    def test_string(self):
        assert EGraph().eval_string(Lit(DUMMY_SPAN, String("hi"))) == "hi"

    def test_bool(self):
        assert EGraph().eval_bool(Lit(DUMMY_SPAN, Bool(True))) is True

    @pytest.mark.xfail(reason="Depends on getting actual sort from egraph")
    def test_rational(self):
        egraph = EGraph()
        rational = Call(DUMMY_SPAN, "rational", [Lit(DUMMY_SPAN, Int(1)), Lit(DUMMY_SPAN, Int(2))])
        egraph.run_program(
            ActionCommand(
                Expr_(DUMMY_SPAN, Call(DUMMY_SPAN, "rational", [Lit(DUMMY_SPAN, Int(1)), Lit(DUMMY_SPAN, Int(2))]))
            )
        )
        assert egraph.eval_rational(rational) == fractions.Fraction(1, 2)


class TestThreads:
    """
    Verify that objects can be accessed from multiple threads at the same time.
    """

    def test_cmds(self):
        cmds = (
            Datatype("Math", [Variant("Add", ["Math", "Math"])]),
            RewriteCommand(
                "",
                Rewrite(
                    DUMMY_SPAN,
                    Call(DUMMY_SPAN, "Add", [Var(DUMMY_SPAN, "a"), Var(DUMMY_SPAN, "b")]),
                    Call(DUMMY_SPAN, "Add", [Var(DUMMY_SPAN, "b"), Var(DUMMY_SPAN, "a")]),
                ),
                False,
            ),
            RunSchedule(Repeat(DUMMY_SPAN, 10, Run(DUMMY_SPAN, RunConfig("")))),
        )

        _thread.start_new_thread(print, cmds)

    @pytest.mark.xfail(reason="egraphs are unsendable")
    def test_egraph(self):
        _thread.start_new_thread(EGraph().run_program, (Datatype("Math", [Variant("Add", ["Math", "Math"])]),))

    def test_serialized_egraph(self):
        egraph = EGraph()
        serialized = egraph.serialize([])
        _thread.start_new_thread(print, (serialized,))
