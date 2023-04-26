import datetime
import json
import os
import pathlib
import subprocess

import pytest
from egg_smol.bindings import *


def get_egg_smol_folder() -> pathlib.Path:
    """
    Return the egg-smol source folder
    """
    metadata_process = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "-q"],
        capture_output=True,
        check=True,
    )
    metadata = json.loads(metadata_process.stdout)
    (egg_smol_package,) = [package for package in metadata["packages"] if package["name"] == "egg-smol"]
    return pathlib.Path(egg_smol_package["manifest_path"]).parent


EGG_SMOL_FOLDER = get_egg_smol_folder()

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
    # TODO: Include currently relyies on the CWD instead of the fact directory. We should fix this upstream
    # and then remove this workaround.
    os.chdir(EGG_SMOL_FOLDER)
    egraph.run_program(*commands)


class TestEGraph:
    def test_parse_program(self):
        res = EGraph().parse_program(
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
                                Call("Var", [Lit(String("x"))]),
                                Call("Num", [Lit(Int(3))]),
                            ],
                        ),
                    ],
                ),
                None,
            ),
        ]

    def test_parse_and_run_program(self):
        program = "(check (= (+ 2 2) 4))"
        egraph = EGraph()

        assert egraph.run_program(*egraph.parse_program(program)) == ["Checked."]

    def test_parse_and_run_program_exception(self):
        program = "(check (= 1 1.0))"
        egraph = EGraph()

        with pytest.raises(
            EggSmolError,
            match="Type mismatch: expected f64, actual i64",
        ):
            egraph.run_program(*egraph.parse_program(program))

    def test_run_rules(self):
        egraph = EGraph()
        start_time = datetime.datetime.now()
        egraph.run_program(
            Datatype("Math", [Variant("Add", ["Math", "Math"])]),
            RewriteCommand("", Rewrite(Call("Add", [Var("a"), Var("b")]), Call("Add", [Var("b"), Var("a")]))),
            RunCommand(RunConfig("", 10, None)),
        )
        end_time = datetime.datetime.now()

        run_report = egraph.run_report()
        assert isinstance(run_report, RunReport)
        total_time = run_report.search_time + run_report.apply_time + run_report.rebuild_time
        # Verify  less than the total time (which includes time spent in Python).
        assert total_time < (end_time - start_time)

    def test_extract(self):
        # Example from extraction-cost
        egraph = EGraph()
        egraph.run_program(
            Datatype("Expr", [Variant("Num", ["i64"], cost=5)]),
            Define("x", Call("Num", [Lit(Int(1))]), 10),
            Define("y", Call("Num", [Lit(Int(2))]), 1),
            Extract(0, Var("x")),
        )
        assert egraph.extract_report() == ExtractReport(6, Call("Num", [Lit(Int(1))]), [])
        egraph.run_program(Extract(0, Var("y")))
        pytest.xfail(reason="https://github.com/mwillsey/egg-smol/issues/128")
        assert egraph.extract_report() == ExtractReport(1, Call("y", []), [])

    def test_extract_string(self):
        egraph = EGraph()
        egraph.run_program(Define("x", Lit(String("hello")), None), Extract(0, Var("x")))
        assert egraph.extract_report() == ExtractReport(0, Lit(String("hello")), [])

    def test_sort_alias(self):
        # From map example
        egraph = EGraph()
        egraph.run_program(
            Sort(
                "MyMap",
                ("Map", [Var("i64"), Var("String")]),
            ),
            Define("my_map1", Call("insert", [Call("empty", []), Lit(Int(1)), Lit(String("one"))]), None),
            Define("my_map2", Call("insert", [Var("my_map1"), Lit(Int(2)), Lit(String("two"))]), None),
            Check([Eq([Lit(String("one")), Call("get", [Var("my_map1"), Lit(Int(1))])])]),
            Extract(0, Var("my_map2")),
        )
        assert egraph.extract_report() == ExtractReport(
            0,
            Call(
                "insert",
                [
                    Call("insert", [Call("empty", []), Lit(Int(2)), Lit(String("two"))]),
                    Lit(Int(1)),
                    Lit(String("one")),
                ],
            ),
            [],
        )


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
