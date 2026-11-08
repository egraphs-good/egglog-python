"""Semantic run results, opaque diagnostics and synchronous live observations."""

import json
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import EggSmolError, EGraph, RunResult, i64, panic, relation, rule, ruleset, run, var
from egglog._program import Builder


@pytest.fixture
def growing_graph():
    includes = relation("stream_includes", i64)
    x = var("x", i64)
    selected = ruleset(rule(includes(x)).then(includes(x + 1)), name="stream-growth")
    return EGraph(includes(i64(0))), selected, includes


def test_ordinary_run_has_only_semantic_flags_and_no_diagnostics(growing_graph):
    graph, selected, includes = growing_graph
    kinds: list[str] = []

    def observe(chunk: pb.RunProgramResponse) -> None:
        if chunk.result is not None and chunk.result.field == "output":
            kinds.extend(output.kind.field for output in chunk.result.value.outputs if output.kind is not None)

    result = graph.run(selected, on_output=observe)
    assert isinstance(result, RunResult)
    assert result.updated
    assert not result.can_stop
    assert kinds == ["run"]
    assert not hasattr(result, "detail")
    assert not hasattr(result, "iterations")
    graph.check(includes(i64(1)))
    quiet = EGraph().run(run().saturate())
    assert not quiet.updated
    assert quiet.can_stop


def test_diagnostics_are_separate_opaque_documents(growing_graph):
    graph, selected, includes = growing_graph
    result, diagnostic = graph.run_with_diagnostics(selected)
    assert result.updated
    assert diagnostic.format.endswith("+json")
    document = json.loads(diagnostic.content)
    assert isinstance(document, dict)
    assert "updated" not in document
    assert "can_stop" not in document
    graph.check(includes(i64(1)))
    cumulative = graph.stats()
    assert isinstance(cumulative, pb.Diagnostic)
    assert cumulative.format == diagnostic.format
    assert isinstance(json.loads(cumulative.content), dict)


def test_explain_does_not_execute_installed_rules(growing_graph):
    graph, selected, includes = growing_graph
    graph.run(0, ruleset=selected)
    before = graph.freeze()
    diagnostic = graph.explain(str(selected.__egg_ident__))
    assert diagnostic.format == "application/vnd.egglog.query-plan+json"
    plans: object = json.loads(diagnostic.content)
    assert isinstance(plans, list)
    (entry,) = plans
    assert isinstance(entry, list)
    name, plan = entry
    assert isinstance(name, str)
    assert name
    assert isinstance(plan, dict)
    assert plan["stages"]
    assert graph.freeze() == before
    graph.check_fail(includes(i64(1)))


def test_loop_snapshots_arrive_before_completion_and_own_their_arenas(growing_graph):
    graph, selected, includes = growing_graph
    events: list[str] = []
    snapshots: list[pb.EGraphSnapshot] = []

    def observe(chunk: pb.RunProgramResponse) -> None:
        if chunk.result is None or chunk.result.field != "output":
            return
        for output in chunk.result.value.outputs:
            assert output.kind is not None
            if output.kind.field == "snapshot":
                events.append("snapshot")
                snapshot = output.kind.value
                assert snapshot.program is not None
                owner = Builder.from_program(snapshot.program).publish()
                assert owner.ref("nodes", len(snapshot.program.nodes) - 1).read() is not None
                snapshots.append(snapshot)
                with pytest.raises(RuntimeError, match="callback"):
                    graph.freeze()
            elif output.location == pb.CommandLocation(path=[0]) and output.kind.field == "run":
                events.append("result")

    result = graph.run(selected.with_snapshots() * 2, on_output=observe)
    events.append("returned")
    assert result.updated
    assert events == ["snapshot", "snapshot", "result", "returned"]
    assert len(snapshots) == 2
    first = snapshots[0].to_binary()
    assert snapshots[0] != snapshots[1]
    graph.register(includes(i64(100)))
    assert snapshots[0].to_binary() == first


def test_closed_consumer_cancels_without_losing_live_graph_ownership(growing_graph):
    graph, selected, includes = growing_graph
    snapshots: list[pb.EGraphSnapshot] = []

    def observe(chunk: pb.RunProgramResponse) -> bool:
        if chunk.result is None or chunk.result.field != "output":
            return True
        for output in chunk.result.value.outputs:
            if output.kind is not None and output.kind.field == "snapshot":
                snapshots.append(output.kind.value)
                return False
        return True

    with pytest.raises(ValueError, match="execution cancelled"):
        graph.run(selected.with_snapshots() * 10, on_output=observe)
    assert len(snapshots) == 1
    graph.check(includes(i64(1)))
    with graph:
        graph.register(includes(i64(100)))
        graph.check(includes(i64(100)))
    graph.check_fail(includes(i64(100)))


def test_output_callback_preserves_original_exception(growing_graph):
    graph, selected, includes = growing_graph
    original = ValueError("original stream consumer failure")

    def observe(chunk: pb.RunProgramResponse) -> None:
        if (
            chunk.result is not None
            and chunk.result.field == "output"
            and any(
                output.kind is not None and output.kind.field == "snapshot" for output in chunk.result.value.outputs
            )
        ):
            raise original

    with pytest.raises(ValueError, match="original stream consumer failure") as caught:
        graph.run(selected.with_snapshots() * 10, on_output=observe)
    assert caught.value is original
    assert original.__cause__ is None
    graph.check(includes(i64(1)))


def test_runtime_error_keeps_observation_prefix_when_terminal_callback_returns_false():
    graph = EGraph(record_history=True)
    builder = Builder()
    freeze = builder.add("commands", pb.Command(kind=Oneof[Literal["freeze"], pb.Freeze]("freeze", pb.Freeze())))
    snapshot = builder.publish().ref("commands", freeze)
    chunks: list[pb.RunProgramResponse] = []

    def observe(chunk: pb.RunProgramResponse) -> bool:
        chunks.append(chunk)
        return chunk.result is not None and chunk.result.field != "error"

    with pytest.raises(EggSmolError, match="streamed runtime marker") as caught:
        graph._state.run_program([snapshot, panic("streamed runtime marker").__egg_ref__, snapshot], on_output=observe)
    outputs = [
        output
        for chunk in chunks
        if chunk.result is not None and chunk.result.field == "output"
        for output in chunk.result.value.outputs
    ]
    assert len(outputs) == 1
    assert outputs[0].kind is not None
    assert outputs[0].kind.field == "snapshot"
    terminal = chunks[-1]
    assert terminal.result is not None
    assert terminal.result.field == "error"
    assert terminal.result.value.context is not None
    assert terminal.result.value.site is not None
    assert len(terminal.result.value.context.commands) == 1
    assert terminal.result.value.context.commands[0].kind == Oneof(
        "action", pb.Action(kind=Oneof[Literal["panic"], str]("panic", "streamed runtime marker"))
    )
    assert terminal.result.value.location == pb.CommandLocation(path=[1])
    assert caught.value.__dict__["_egglog_proto_error"] == terminal.result.value
    assert caught.value.__dict__["_egglog_proto_frame"] == terminal
    assert len(graph.history().commands) == 3
