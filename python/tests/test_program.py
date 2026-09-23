from __future__ import annotations

import json

import jsonschema
import pytest

from egglog import EGraph, Expr, PyObject, back_off, eq, i64, i64Like, relation, rule, ruleset, run, union, vars_
from egglog import SharedProgram as Program
from egglog import bindings as b
from egglog.examples.shared_program import build_program


class ProgramMath(Expr):
    def __init__(self, value: i64Like) -> None: ...

    def __add__(self, other: ProgramMath) -> ProgramMath: ...  # type: ignore[empty-body]


def test_public_python_authored_fixture() -> None:
    program = build_program()
    payload = json.loads(program.to_json())
    jsonschema.Draft202012Validator(json.loads(Program.json_schema())).validate(payload)
    b.EGraph().run_shared_program(Program.from_json(program.to_json()))


def test_high_level_commands_export_and_execute() -> None:
    (x,) = vars_("x", i64)
    (result,) = vars_("result", ProgramMath)
    simplify = ruleset(
        rule(eq(ProgramMath(x) + ProgramMath(0)).to(result), name="program-add-zero").then(
            union(result).with_(ProgramMath(x))
        ),
        name="program-simplify",
    )
    graph = EGraph(record_program=True)
    expression = graph.let("root", ProgramMath(7) + ProgramMath(0))
    graph.run(run(simplify).saturate())
    graph.check(eq(expression).to(ProgramMath(7)))
    assert str(graph.extract(expression)) == str(ProgramMath(7))
    with graph:
        graph.register(ProgramMath(99))

    program = graph.recorded_program
    payload = json.loads(program.to_json())
    jsonschema.Draft202012Validator(json.loads(Program.json_schema())).validate(payload)
    assert payload["format"] == "egglog-program-v1"
    assert "program-simplify" in program.to_egglog()
    # This includes implicit sort/constructor/ruleset installation and extraction,
    # so a fresh graph can execute the complete Python-authored command stream.
    restored = Program.from_json(program.to_json())
    outputs = b.EGraph().run_shared_program(restored)
    assert any(isinstance(output, b.ExtractBest) for output in outputs)
    source = restored.to_replayable_egglog()
    assert any(isinstance(output, b.ExtractBest) for output in b.EGraph().run_shared_program(Program.parse(source)))


def test_existing_bindings_execution_records_native_programs() -> None:
    graph = b.EGraph(record_program=True)
    graph.run_program(*graph.parse_program("(function answer () i64 :no-merge)\n(set (answer) 42)"))
    program = graph.recorded_program()
    assert program is not None
    replay = b.EGraph()
    replay.run_shared_program(Program.from_json(program.to_json()))
    replay.parse_and_run_program("(check (= (answer) 42))")


def test_checked_source_export() -> None:
    program = Program.parse("(function answer () i64 :no-merge)\n(set (answer) 42)\n(check (= (answer) 42))")
    source = program.to_replayable_egglog()
    b.EGraph().run_shared_program(Program.parse(source))


def test_python_host_objects_and_experimental_schedulers_roundtrip() -> None:
    ready = relation("program_ready", i64)
    done = relation("program_done", i64)
    (x,) = vars_("x", i64)
    schedule = ruleset(rule(ready(x), name="program-host-rule").then(done(x)), name="program-host-rules")
    graph = EGraph(ready(i64(1)), record_program=True)
    value = graph.let("host_value", PyObject({"answer": 42}))
    graph.check(eq(value).to(PyObject({"answer": 42})))
    graph.run(run(schedule, scheduler=back_off(match_limit=10).persistent()))
    graph.check(done(i64(1)))
    program = graph.recorded_program
    assert "py-object" in program.to_egglog()
    assert "let-scheduler" in program.to_egglog()
    # Python's standard destination installs these host capabilities. JSON does
    # not make a Python object or experimental scheduler a core builtin.
    b.EGraph().run_shared_program(Program.from_json(program.to_json()))


def test_recording_disabled_and_reset() -> None:
    with pytest.raises(ValueError, match="record_program=True"):
        _ = EGraph().recorded_program
    graph = b.EGraph()
    assert graph.recorded_program() is None
    assert graph.stop_recording() is None
    graph.start_recording()
    graph.parse_and_run_program("(check (= 1 1))")
    graph.start_recording()
    program = graph.recorded_program()
    assert program is not None
    assert json.loads(program.to_json())["commands"] == []


def test_failed_attempts_remain_in_command_record() -> None:
    graph = b.EGraph(record_program=True)
    with pytest.raises(b.EggSmolError):
        graph.parse_and_run_program("(function prefix () i64 :no-merge)\n(set (prefix) 9)\n(check (= 1 2))")
    record = graph.stop_recording()
    assert record is not None
    data = json.loads(record.to_json())
    assert [entry["outcome"]["type"] for entry in data["entries"]] == ["Success", "Success", "Failure"]
    assert data["entries"][-1]["outcome"]["value"]["message"]
    graph.parse_and_run_program("(check (= (prefix) 9))")
    assert graph.recorded_program() is None
    with pytest.raises(b.EggSmolError):
        b.EGraph().run_shared_program(record.program())


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1])
def test_i64_json_roundtrip(value: int) -> None:
    span = b.RustSpan(__name__, 1, 1)
    program = Program(b.ActionCommand(b.Expr_(span, b.Lit(span, b.Int(value)))))
    restored = Program.from_json(program.to_json())
    assert restored.to_json() == program.to_json()
    assert str(value) in restored.to_egglog()


@pytest.mark.parametrize("value", [0.0, -0.0, 1.25, float("inf"), float("-inf"), float("nan")])
def test_float_json_roundtrip(value: float) -> None:
    span = b.RustSpan(__name__, 1, 1)
    program = Program(b.ActionCommand(b.Expr_(span, b.Lit(span, b.Float(value)))))
    jsonschema.Draft202012Validator(json.loads(Program.json_schema())).validate(json.loads(program.to_json()))
    assert Program.from_json(program.to_json()).to_json() == program.to_json()


@pytest.mark.parametrize(
    "payload", ["{", "{}", '{"format":"unknown","commands":[]}', '{"format":"egglog-program-v1","commands":[{}]}']
)
def test_malformed_program_rejected(payload: str) -> None:
    with pytest.raises(ValueError, match="invalid program JSON"):
        Program.from_json(payload)


def test_source_parse_error() -> None:
    with pytest.raises(ValueError, match=r"invalid-program\.egg"):
        Program.parse("(", filename="invalid-program.egg")


def test_unregistered_extension_is_an_execution_error() -> None:
    program = Program.parse("(shared-program-missing-extension)")
    restored = Program.from_json(program.to_json())
    with pytest.raises(b.EggSmolError):
        b.EGraph().run_shared_program(restored)


def test_import_preserves_fields_absent_from_legacy_bindings() -> None:
    span = b.RustSpan(__name__, 1, 1)
    program = Program(
        b.Sort(span, "NativeSort", None),
        b.Constructor(span, "NativeConstructor", b.Schema([], "NativeSort"), None, False),
    )
    payload = json.loads(program.to_json())
    payload["commands"][0]["value"]["unionable"] = False
    payload["commands"][1]["value"]["term_constructor"] = "InternalTermConstructor"
    restored = Program.from_json(json.dumps(payload))
    assert json.loads(restored.to_json()) == payload
    with pytest.raises(ValueError, match="replayable Egglog text"):
        restored.to_replayable_egglog()


def test_every_binding_command_variant_matches_the_generated_schema() -> None:
    span = b.RustSpan(__name__, 1, 1)
    literal = b.Lit(span, b.Int(1))
    call = b.Call(span, "f", [])
    fact = b.Eq(span, call, literal)
    rewrite = b.Rewrite(span, call, literal)
    variant = b.Variant(span, "C", [])
    commands: list[b._Command] = [
        b.Datatype(span, "D", [variant]),
        b.Datatypes(span, [(span, "D", b.SubVariants([variant]))]),
        b.Sort(span, "S", None),
        b.FunctionCommand(span, "f", b.Schema([], "i64"), None),
        b.AddRuleset(span, "rules"),
        b.RuleCommand(b.Rule(span, [b.Expr_(span, call)], [fact], "rule", "rules")),
        b.RewriteCommand("rules", rewrite, False),
        b.BiRewriteCommand("rules", rewrite),
        b.ActionCommand(b.Let(span, "$x", literal)),
        b.RunSchedule(b.Sequence(span, [b.Repeat(span, 2, b.Saturate(span, b.Run(span, b.RunConfig("rules"))))])),
        b.Extract(span, call, literal),
        b.Check(span, [fact]),
        b.Prove(span, [fact]),
        b.ProveExists(span, "f"),
        b.PrintFunction(span, "f", 10, None, b.DefaultPrintFunctionMode()),
        b.PrintSize(span, "f"),
        b.Output(span, "out.tsv", [literal]),
        b.Input(span, "f", "input.tsv"),
        b.Push(1),
        b.Pop(span, 1),
        b.Fail(span, b.Check(span, [fact])),
        b.Include(span, "library.egg"),
        b.Constructor(span, "C", b.Schema([], "S"), 1, False),
        b.Relation(span, "r", ["i64"]),
        b.PrintOverallStatistics(span, None),
        b.UserDefined(span, "host-extension", [literal]),
        b.UnstableCombinedRuleset(span, "combined", ["rules"]),
    ]
    program = Program(*commands)
    payload = json.loads(program.to_json())
    assert len({command["type"] for command in payload["commands"]}) == 27
    jsonschema.Draft202012Validator(json.loads(Program.json_schema())).validate(payload)
    assert json.loads(Program.from_json(program.to_json()).to_json()) == payload
