# mypy: allow-empty-bodies
"""Accumulated protocol inputs and the optional replayable source view."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import EggSmolError, EGraph, bindings, function, i64, panic, set_
from egglog._program import Builder


@pytest.mark.parametrize(
    ("record_history", "save_egglog_string", "seminaive"),
    [(False, False, True), (True, False, True), (False, True, True), (True, True, True), (True, False, False)],
)
def test_history_and_source_export_share_native_input_recording(
    monkeypatch: pytest.MonkeyPatch, *, record_history: bool, save_egglog_string: bool, seminaive: bool
) -> None:
    created: list[pb.CreateEGraphRequest] = []

    class Transport:
        def create(self, encoded: bytes) -> bytes:
            created.append(pb.CreateEGraphRequest.from_binary(encoded))
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", Transport)
    graph = EGraph(record_history=record_history, save_egglog_string=save_egglog_string, seminaive=seminaive)
    assert len(created) == 1
    assert created[0].record_history is (record_history or save_egglog_string)
    assert created[0].options is not None
    assert created[0].options.seminaive is seminaive
    assert graph._state.record_history is (record_history or save_egglog_string)


@pytest.mark.parametrize("invalid", [0, 1, None, "yes"])
def test_history_choice_rejects_nonbooleans_before_creation(monkeypatch: pytest.MonkeyPatch, invalid: object) -> None:
    def forbidden() -> None:
        pytest.fail("Invalid recording choice allocated an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    with pytest.raises(TypeError, match="History recording must be a boolean"):
        EGraph(record_history=invalid)  # type: ignore[arg-type]


def test_history_getter_uses_current_handle_and_preserves_owned_result(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[pb.GetHistoryRequest] = []
    expected = pb.Program(files=[pb.SourceFile(name="input.egg")])

    class Transport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=19, threads=3).to_binary()

        def get_history(self, encoded: bytes) -> bytes:
            requests.append(pb.GetHistoryRequest.from_binary(encoded))
            return expected.to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", Transport)
    graph = EGraph(record_history=True)
    result = graph.history()
    assert requests == [pb.GetHistoryRequest(egraph_id=19)]
    assert result == expected
    result.files[0].name = "changed.egg"
    assert graph.history() == expected
    graph._state._native_running = True
    try:
        with pytest.raises(RuntimeError, match="cost-model callback"):
            graph.history()
    finally:
        graph._state._native_running = False
    assert len(requests) == 2
    graph._state.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.history()
    assert len(requests) == 2


def test_history_default_off_and_getter_does_not_record_itself() -> None:
    disabled = EGraph()
    disabled.register(i64(1))
    with pytest.raises(ValueError, match="record"):
        disabled.history()

    graph = EGraph(record_history=True, seminaive=False)
    graph.register(i64(2))
    history = graph.history()
    assert graph.num_threads() == 1
    assert len(history.commands) == 1
    assert graph.history().to_binary() == history.to_binary()
    graph.freeze()
    observed = graph.history()
    assert len(observed.commands) == len(history.commands) + 1
    assert observed.commands[-1].kind is not None
    assert observed.commands[-1].kind.field == "freeze"
    observed.commands.clear()
    assert graph.history().to_binary() != observed.to_binary()


def test_history_and_source_keep_accepted_failure_inputs_and_live_prefix() -> None:
    @function
    def history_row(key: i64) -> i64: ...

    graph = EGraph(save_egglog_string=True, record_history=True)
    graph.register(set_(history_row(i64(0))).to(i64(0)))
    builder = Builder()
    row = history_row(i64(0)).__egg_ref__.read().kind.value.func
    observe = builder.add(
        "commands",
        pb.Command(kind=Oneof[Literal["table_sizes"], pb.GetTableSizes]("table_sizes", pb.GetTableSizes(tables=[row]))),
    )
    commands = [
        set_(history_row(i64(1))).to(i64(11)).__egg_ref__,
        builder.publish().ref("commands", observe),
        panic("recorded prefix failure").__egg_ref__,
        set_(history_row(i64(2))).to(i64(22)).__egg_ref__,
    ]
    with pytest.raises(EggSmolError, match="recorded prefix failure"):
        graph._state.run_program(commands)
    source = graph.as_egglog_string
    assert "recorded prefix failure" in source
    history = graph.history()
    assert len(history.commands) >= 4
    failure = history.commands[-2].kind
    assert failure is not None
    assert failure.field == "action"
    assert failure.value.kind == Oneof[Literal["panic"], str]("panic", "recorded prefix failure")
    assert history.commands[-1].kind is not None
    assert history.commands[-1].kind.field == "action"
    assert graph.lookup_function_value(history_row(i64(1))) == i64(11)
    assert graph.lookup_function_value(history_row(i64(2))) is None


def test_history_parent_child_continuations_restore_saved_prefix_on_pop() -> None:
    graph = EGraph(record_history=True)
    graph.register(i64(3))
    graph.push()
    parent = graph._state_stack[-1]
    assert parent.handle is not None
    parent_before = pb.Program.from_binary(
        parent.transport.get_history(pb.GetHistoryRequest(egraph_id=parent.handle).to_binary())
    )
    graph.register(i64(4))
    graph.freeze()
    child = graph.history()
    parent_after = pb.Program.from_binary(
        parent.transport.get_history(pb.GetHistoryRequest(egraph_id=parent.handle).to_binary())
    )
    assert parent_before == parent_after
    assert len(child.commands) > len(parent_after.commands)
    graph.pop()
    assert graph.history() == parent_before
    assert graph.history() != child
