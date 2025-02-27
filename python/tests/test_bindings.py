import _thread
import json
import os
import pathlib
import subprocess

import black
import pytest

from egglog.bindings import *

DUMMY_SPAN = RustSpan(__name__, 0, 0)


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

# > 1 second
SLOW_TESTS = ["repro-unsound", "cykjson", "herbie", "lambda"]

# > 2 seconds
SKIP_TESTS = {"eggcc-extraction", "math-microbenchmark", "python_array_optimize", "typeinfer"}


@pytest.mark.parametrize(
    "example_file",
    [
        pytest.param(path, id=path.stem, marks=pytest.mark.slow if path.stem in SLOW_TESTS else [])
        for path in (EGG_SMOL_FOLDER / "tests").glob("*.egg")
        if path.stem not in SKIP_TESTS
    ],
)
def test_example(example_file: pathlib.Path):
    egraph = EGraph(fact_directory=EGG_SMOL_FOLDER)
    commands = egraph.parse_program(example_file.read_text())
    # TODO: Include currently relies on the CWD instead of the fact directory. We should fix this upstream
    # and then remove this workaround.
    os.chdir(EGG_SMOL_FOLDER)
    egraph.run_program(*commands)


BLACK_MODE = black.Mode(line_length=88)


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
        res_str = black.format_str(str(res), mode=BLACK_MODE).strip()
        assert res_str == snapshot_py

    def test_parse_and_run_program(self):
        program = "(check (= (+ 2 2) 4))"
        egraph = EGraph()

        assert egraph.run_program(*egraph.parse_program(program)) == []

    def test_parse_and_run_program_exception(self):
        program = "(check (= 1 1.0))"
        egraph = EGraph()

        with pytest.raises(
            EggSmolError,
            match="to have type",
        ):
            egraph.run_program(*egraph.parse_program(program))

    def test_run_rules(self):
        egraph = EGraph()
        egraph.run_program(
            Datatype(DUMMY_SPAN, "Math", [Variant(DUMMY_SPAN, "Add", ["Math", "Math"])]),
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
            Datatype(DUMMY_SPAN, "Expr", [Variant(DUMMY_SPAN, "Num", ["i64"], cost=5)]),
            ActionCommand(Let(DUMMY_SPAN, "x", Call(DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]))),
            QueryExtract(DUMMY_SPAN, 0, Var(DUMMY_SPAN, "x")),
        )

        extract_report = egraph.extract_report()
        assert isinstance(extract_report, Best)
        assert extract_report.cost == 6
        assert extract_report.termdag.term_to_expr(extract_report.term, DUMMY_SPAN) == Call(
            DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]
        )

    def test_simplify(self):
        egraph = EGraph()
        egraph.run_program(
            Datatype(DUMMY_SPAN, "Expr", [Variant(DUMMY_SPAN, "Num", ["i64"], cost=5)]),
            ActionCommand(Let(DUMMY_SPAN, "x", Call(DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]))),
            Simplify(DUMMY_SPAN, Var(DUMMY_SPAN, "x"), Run(DUMMY_SPAN, RunConfig(""))),
        )

        extract_report = egraph.extract_report()
        assert isinstance(extract_report, Best)
        assert extract_report.cost == 6
        assert extract_report.termdag.term_to_expr(extract_report.term, DUMMY_SPAN) == Call(
            DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]
        )

    def test_sort_alias(self):
        # From map example
        egraph = EGraph()
        egraph.run_program(
            Sort(
                DUMMY_SPAN,
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
                        Lit(DUMMY_SPAN, String("one")),
                        Call(DUMMY_SPAN, "map-get", [Var(DUMMY_SPAN, "my_map1"), Lit(DUMMY_SPAN, Int(1))]),
                    )
                ],
            ),
            ActionCommand(Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "my_map2"), Lit(DUMMY_SPAN, Int(0)))),
        )

        extract_report = egraph.extract_report()
        assert isinstance(extract_report, Best)
        assert extract_report.termdag.term_to_expr(extract_report.term, DUMMY_SPAN) == Call(
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
        assert repr(Variant(DUMMY_SPAN, "name", [])) == f"Variant({DUMMY_SPAN!r}, 'name', [], None)"

    def test_name(self):
        assert Variant(DUMMY_SPAN, "name", []).name == "name"

    def test_types(self):
        assert Variant(DUMMY_SPAN, "name", ["a", "b"]).types == ["a", "b"]

    def test_cost(self):
        assert Variant(DUMMY_SPAN, "name", [], cost=1).cost == 1

    def test_compare(self):
        assert Variant(DUMMY_SPAN, "name", []) == Variant(DUMMY_SPAN, "name", [])
        assert Variant(DUMMY_SPAN, "name", []) != Variant(DUMMY_SPAN, "name", ["a"])
        assert Variant(DUMMY_SPAN, "name", []) != 10  # type: ignore[comparison-overlap]


class TestThreads:
    """
    Verify that objects can be accessed from multiple threads at the same time.
    """

    def test_cmds(self):
        cmds = (
            Datatype(DUMMY_SPAN, "Math", [Variant(DUMMY_SPAN, "Add", ["Math", "Math"])]),
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
        _thread.start_new_thread(
            EGraph().run_program, (Datatype(DUMMY_SPAN, "Math", [Variant(DUMMY_SPAN, "Add", ["Math", "Math"])]),)
        )

    def test_serialized_egraph(self):
        egraph = EGraph()
        serialized = egraph.serialize([])
        _thread.start_new_thread(print, (serialized,))
