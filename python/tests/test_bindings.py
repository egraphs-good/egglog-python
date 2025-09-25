import concurrent.futures
import json
import os
import pathlib
import subprocess
from fractions import Fraction

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
SKIP_TESTS = {
    "eggcc-extraction",
    "math-microbenchmark",
    "python_array_optimize",
    "typeinfer",
    "repro-typechecking-schedule",
}


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
        res = egraph.run_program(
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

        assert len(res) == 1
        assert isinstance(res[0], RunScheduleOutput)

    def test_extract(self):
        # Example from extraction-cost
        egraph = EGraph()
        res = egraph.run_program(
            Datatype(DUMMY_SPAN, "Expr", [Variant(DUMMY_SPAN, "Num", ["i64"], cost=5)]),
            ActionCommand(Let(DUMMY_SPAN, "x", Call(DUMMY_SPAN, "Num", [Lit(DUMMY_SPAN, Int(1))]))),
            Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "x"), Lit(DUMMY_SPAN, Int(0))),
        )

        assert len(res) == 1
        extract_report = res[0]
        assert isinstance(extract_report, ExtractBest)
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
            Extract(DUMMY_SPAN, Var(DUMMY_SPAN, "my_map2"), Lit(DUMMY_SPAN, Int(0))),
        )


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
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(print, cmds).result()

    @pytest.mark.xfail(reason="egraphs are unsendable")
    def test_egraph(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                EGraph().run_program, Datatype(DUMMY_SPAN, "Math", [Variant(DUMMY_SPAN, "Add", ["Math", "Math"])])
            ).result()

    def test_serialized_egraph(self):
        egraph = EGraph()
        serialized = egraph.serialize([])
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(print, (serialized,)).result()


egraph = EGraph()


class TestValues:
    def test_i64(self):
        sort, value = egraph.eval_expr(Lit(DUMMY_SPAN, Int(42)))
        assert sort == "i64"
        assert egraph.value_to_i64(value) == 42

    def test_bigint(self):
        sort, value = egraph.eval_expr(Call(DUMMY_SPAN, "bigint", [Lit(DUMMY_SPAN, Int(100))]))
        assert sort == "BigInt"
        assert egraph.value_to_bigint(value) == 100

    def test_bigrat(self):
        sort, value = egraph.eval_expr(
            Call(
                DUMMY_SPAN,
                "bigrat",
                [
                    Call(DUMMY_SPAN, "bigint", [Lit(DUMMY_SPAN, Int(100))]),
                    Call(DUMMY_SPAN, "bigint", [Lit(DUMMY_SPAN, Int(21))]),
                ],
            )
        )
        assert sort == "BigRat"
        assert egraph.value_to_bigrat(value) == Fraction(100, 21)

    def test_f64(self):
        sort, value = egraph.eval_expr(Lit(DUMMY_SPAN, Float(3.14)))
        assert sort == "f64"
        assert egraph.value_to_f64(value) == 3.14

    def test_string(self):
        sort, value = egraph.eval_expr(Lit(DUMMY_SPAN, String("hello")))
        assert sort == "String"
        assert egraph.value_to_string(value) == "hello"

    def test_rational(self):
        sort, value = egraph.eval_expr(
            Call(
                DUMMY_SPAN,
                "rational",
                [Lit(DUMMY_SPAN, Int(22)), Lit(DUMMY_SPAN, Int(7))],
            )
        )
        assert sort == "Rational"
        assert egraph.value_to_rational(value) == Fraction(22, 7)

    def test_bool(self):
        sort, value = egraph.eval_expr(Lit(DUMMY_SPAN, Bool(True)))
        assert sort == "bool"
        assert egraph.value_to_bool(value) is True

    def test_py_object(self):
        py_object_sort = PyObjectSort()
        egraph = EGraph(py_object_sort)
        expr = py_object_sort.store("my object")
        sort, value = egraph.eval_expr(expr)
        assert sort == "PyObject"
        assert egraph.value_to_pyobject(py_object_sort, value) == "my object"

    def test_map(self):
        k = Lit(DUMMY_SPAN, Int(1))
        v = Lit(DUMMY_SPAN, String("one"))
        egraph.run_program(Sort(DUMMY_SPAN, "MyMap", ("Map", [Var(DUMMY_SPAN, "i64"), Var(DUMMY_SPAN, "String")])))
        sort, value = egraph.eval_expr(Call(DUMMY_SPAN, "map-insert", [Call(DUMMY_SPAN, "map-empty", []), k, v]))
        assert sort == "MyMap"
        m = egraph.value_to_map(value)
        assert m == {egraph.eval_expr(k)[1]: egraph.eval_expr(v)[1]}

    def test_multiset(self):
        egraph.run_program(Sort(DUMMY_SPAN, "MyMultiSet", ("MultiSet", [Var(DUMMY_SPAN, "i64")])))
        sort, value = egraph.eval_expr(
            Call(
                DUMMY_SPAN,
                "multiset-of",
                [
                    Lit(DUMMY_SPAN, Int(1)),
                    Lit(DUMMY_SPAN, Int(2)),
                    Lit(DUMMY_SPAN, Int(1)),
                ],
            )
        )
        assert sort == "MyMultiSet"
        ms = egraph.value_to_multiset(value)
        assert sorted(egraph.value_to_i64(v) for v in ms) == [1, 1, 2]

    def test_set(self):
        egraph.run_program(Sort(DUMMY_SPAN, "MySet", ("Set", [Var(DUMMY_SPAN, "i64")])))
        sort, value = egraph.eval_expr(
            Call(
                DUMMY_SPAN,
                "set-of",
                [
                    Lit(DUMMY_SPAN, Int(1)),
                    Lit(DUMMY_SPAN, Int(2)),
                ],
            )
        )
        assert sort == "MySet"
        s = egraph.value_to_set(value)
        assert isinstance(s, set)
        assert {egraph.value_to_i64(v) for v in s} == {1, 2}

    def test_vec(self):
        egraph.run_program(Sort(DUMMY_SPAN, "MyVec", ("Vec", [Var(DUMMY_SPAN, "i64")])))
        sort, value = egraph.eval_expr(
            Call(
                DUMMY_SPAN,
                "vec-of",
                [
                    Lit(DUMMY_SPAN, Int(1)),
                    Lit(DUMMY_SPAN, Int(2)),
                    Lit(DUMMY_SPAN, Int(3)),
                ],
            )
        )
        assert sort == "MyVec"
        v = egraph.value_to_vec(value)
        assert isinstance(v, list)
        assert [egraph.value_to_i64(vi) for vi in v] == [1, 2, 3]

    def test_fn(self):
        egraph.run_program(
            Sort(DUMMY_SPAN, "MyFn", ("UnstableFn", [Call(DUMMY_SPAN, "i64", []), Var(DUMMY_SPAN, "i64")]))
        )
        sort, value = egraph.eval_expr(
            Call(
                DUMMY_SPAN,
                "unstable-fn",
                [
                    Lit(DUMMY_SPAN, String("+")),
                    Lit(DUMMY_SPAN, Int(1)),
                ],
            )
        )
        assert sort == "MyFn"
        f, args = egraph.value_to_function(value)
        assert f == "+"
        assert len(args) == 1
        assert egraph.value_to_i64(args[0]) == 1


def test_lookup_function():
    egraph = EGraph()
    egraph.run_program(*egraph.parse_program("(function hi (i64) i64 :no-merge)\n(set (hi 1) 2)"))
    assert (
        egraph.lookup_function("hi", [egraph.eval_expr(Lit(DUMMY_SPAN, Int(1)))[1]])
        == egraph.eval_expr(Lit(DUMMY_SPAN, Int(2)))[1]
    )
