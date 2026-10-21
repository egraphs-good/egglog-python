# mypy: allow-empty-bodies
"""Shared protocol audit history and the optional replayable source view."""

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
def test_history_and_source_recording_are_independent_creation_choices(
    monkeypatch: pytest.MonkeyPatch, *, record_history: bool, save_egglog_string: bool, seminaive: bool
) -> None:
    created: list[pb.CreateEGraphRequest] = []
    source_enabled: list[bool] = []

    class Transport:
        def create(self, encoded: bytes) -> bytes:
            created.append(pb.CreateEGraphRequest.from_binary(encoded))
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def enable_recording(self) -> None:
            source_enabled.append(True)

        def destroy(self, encoded: bytes) -> bytes:
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", Transport)
    graph = EGraph(record_history=record_history, save_egglog_string=save_egglog_string, seminaive=seminaive)
    assert len(created) == 1
    assert created[0].record_history is record_history
    assert created[0].options is not None
    assert created[0].options.seminaive is seminaive
    assert source_enabled == ([True] if save_egglog_string else [])
    assert graph._state._recording is save_egglog_string


@pytest.mark.parametrize("invalid", [0, 1, None, "yes"])
def test_history_choice_rejects_nonbooleans_before_creation(monkeypatch: pytest.MonkeyPatch, invalid: object) -> None:
    def forbidden() -> None:
        pytest.fail("Invalid recording choice allocated an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    with pytest.raises(TypeError, match="History recording must be a boolean"):
        EGraph(record_history=invalid)  # type: ignore[arg-type]


def test_history_getter_uses_current_handle_and_preserves_owned_result(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[pb.GetHistoryRequest] = []
    expected = pb.GetHistoryResponse(initial_resources=pb.ConfigureEGraphResourcesResponse(threads=3))

    class Transport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=19).to_binary()

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
    assert result.initial_resources is not None
    result.initial_resources.threads = 9
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
    assert history.initial_resources is not None
    assert history.initial_resources.threads == 1
    assert not history.closed
    assert history.entries[0].request is not None
    assert history.entries[0].request.field == "create"
    assert history.entries[0].request.value.options is not None
    assert history.entries[0].request.value.options.seminaive is False
    assert graph.history().to_binary() == history.to_binary()

    graph.freeze()
    observed = graph.history()
    assert len(observed.entries) == len(history.entries) + 1
    entry = observed.entries[-1]
    assert entry.request is not None
    assert entry.request.field == "run_program"
    assert entry.request.value.program is not None
    assert entry.request.value.program.commands[0].kind is not None
    assert entry.request.value.program.commands[0].kind.field == "freeze"
    assert entry.outcome is not None
    assert entry.outcome.field == "run_program_response"
    assert entry.outcome.value.outputs[0].kind is not None
    assert entry.outcome.value.outputs[0].kind.field == "snapshot"
    observed.entries.clear()
    assert graph.history().to_binary() != observed.to_binary()


def test_history_keeps_failure_response_and_live_prefix_after_source_view_invalidates() -> None:
    @function
    def history_row(key: i64) -> i64: ...

    graph = EGraph(save_egglog_string=True, record_history=True)
    graph.register(set_(history_row(i64(0))).to(i64(0)))
    builder = Builder()
    row = history_row(i64(0)).__egg_ref__.read().kind.value.func
    observe = builder.add(
        "commands",
        pb.Command(kind=Oneof[Literal["print_size"], pb.PrintSize]("print_size", pb.PrintSize(tables=[row]))),
    )
    commands = [
        set_(history_row(i64(1))).to(i64(11)).__egg_ref__,
        builder.publish().ref("commands", observe),
        panic("recorded prefix failure").__egg_ref__,
        set_(history_row(i64(2))).to(i64(22)).__egg_ref__,
    ]
    with pytest.raises(EggSmolError, match="recorded prefix failure"):
        graph._state.run_program(commands)
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        _ = graph.as_egglog_string
    history = graph.history()
    entry = history.entries[-1]
    assert entry.request is not None
    assert entry.request.field == "run_program"
    assert entry.request.value.program is not None
    assert len(entry.request.value.program.commands) == 4
    assert entry.outcome is not None
    assert entry.outcome.field == "run_program_response"
    response = entry.outcome.value
    assert response.error is not None
    assert response.error.location is not None
    assert response.error.location.path == [2]
    assert len(response.outputs) == 1
    output = response.outputs[0].kind
    assert output is not None
    assert output.field == "sizes"
    assert output.value.sizes[0].rows == 2
    assert graph.lookup_function_value(history_row(i64(1))) == i64(11)
    assert graph.lookup_function_value(history_row(i64(2))) is None


def test_history_parent_child_continuations_join_on_pop_without_rewinding_audit() -> None:
    graph = EGraph(record_history=True)
    graph.register(i64(3))
    graph.push()
    parent = graph._state_stack[-1]
    assert parent.handle is not None
    parent_before = pb.GetHistoryResponse.from_binary(
        parent.transport.get_history(pb.GetHistoryRequest(egraph_id=parent.handle).to_binary())
    )
    graph.register(i64(4))
    graph.freeze()
    child = graph.history()
    parent_after = pb.GetHistoryResponse.from_binary(
        parent.transport.get_history(pb.GetHistoryRequest(egraph_id=parent.handle).to_binary())
    )
    assert parent_before == parent_after
    assert len(child.entries) > len(parent_after.entries)
    graph.pop()
    joined = graph.history()
    assert len(joined.entries) == len(child.entries) + 1
    assert joined.entries[:-1] == child.entries
    assert joined.entries[-1].request is not None
    assert joined.entries[-1].request.field == "return_to"
    assert graph.history() == joined
