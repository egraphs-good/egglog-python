# mypy: allow-empty-bodies
# Ellipsis bodies are DSL declarations or annotation-introspection fixtures.
"""Record/bytes plumbing tests; the inert responder is not engine conformance."""

import json
from collections.abc import Callable, Iterator
from operator import index
from pathlib import Path
from threading import Barrier, Event
from typing import Literal, Never, NotRequired, Self, SupportsIndex, TypedDict, cast

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import DescFieldValueList, Message, Oneof

from egglog import (
    Action,
    BaseExpr,
    EGraph,
    Expr,
    Fact,
    PrimitiveFailure,
    bindings,
    constant,
    eq,
    function,
    i64,
    method,
    set_,
)
from egglog._catalog import builtin_catalog
from egglog._presentation import CallKind, CallPresentation, Parameter, Presentation, presentation
from egglog._program import Builder, pack
from egglog.egraph_state import EggSmolError, _decode_run_frames
from egglog.runtime import RuntimeFunction, create_callable, resolve_callable


def output_frames(
    output: pb.RunProgramOutput | None = None, *, files: list[pb.SourceFile] | None = None
) -> list[bytes]:
    """Encode owned success data followed by the required terminal acknowledgement."""
    frames = []
    if output is not None:
        frames.append(
            pb.RunProgramResponse(
                result=Oneof[Literal["output"], pb.RunProgramOutput]("output", output), files=files or []
            ).to_binary()
        )
    frames.append(
        pb.RunProgramResponse(
            result=Oneof[Literal["output"], pb.RunProgramOutput]("output", pb.RunProgramOutput())
        ).to_binary()
    )
    return frames


def _record_frames(encoded: list[bytes], prefix: Path) -> list[pb.RunProgramResponse]:
    """Preserve each native frame's bytes and validate the observed completion."""
    for position, frame in enumerate(encoded):
        (prefix.parent / f"{prefix.name}-{position}.pb").write_bytes(frame)
    return _decode_run_frames(encoded)


def _nested_messages(root: Message) -> Iterator[Message]:
    """Inspect every nested record, including saved code and reconstruction Programs."""
    pending = [root]
    while pending:
        record = pending.pop()
        yield record
        for field in record.desc().fields:
            value = record[field]
            values = value if isinstance(field.value, DescFieldValueList) else (value,)
            pending.extend(child for child in values if isinstance(child, Message))


def _required_error(frames: list[pb.RunProgramResponse]) -> pb.Error:
    """Require a terminal error while retaining any earlier independent outputs."""
    assert frames
    terminal = frames[-1].result
    assert terminal is not None
    assert terminal.field == "error"
    assert all(frame.result is not None and frame.result.field == "output" for frame in frames[:-1])
    return terminal.value


def _required_data(frames: list[pb.RunProgramResponse], field: str) -> pb.RunProgramOutput:
    """Select one observation with its own arenas, never concatenate frame indices."""
    matches = [
        frame.result.value
        for frame in frames
        if frame.result is not None
        and frame.result.field == "output"
        and any(output.kind is not None and output.kind.field == field for output in frame.result.value.outputs)
    ]
    assert len(matches) == 1
    return matches[0]


def _program_rules(program: pb.Program) -> list[pb.Rule]:
    """Inspect setup definitions and literal selections, including nested controls."""
    rules = [rule for binding in program.rulesets if binding.value is not None for rule in binding.value.rules]
    pending = list(reversed(program.commands))
    while pending:
        kind = pending.pop().kind
        assert kind is not None
        if kind.field == "run":
            for selection in kind.value.rulesets:
                if selection.kind is not None and selection.kind.field == "ruleset":
                    rules.extend(selection.kind.value.rules)
        elif kind.field == "loop":
            pending.extend(reversed(kind.value.body))
        elif kind.field == "fail":
            pending.append(kind.value)
    return rules


def _required_oneof[T](kind: object, field: str, value_type: type[T]) -> T:
    """Require a populated observation arm and its exact generated payload type."""
    assert isinstance(kind, Oneof)
    assert kind.field == field
    assert isinstance(kind.value, value_type)
    return kind.value


class CallbackObservation(TypedDict):
    calls: int
    error: BaseException | None
    result: NotRequired[object]
    caught: NotRequired[BaseException]
    started: NotRequired[Event]
    release: NotRequired[Event]
    barrier: NotRequired[Barrier]


# Test-only observation of the exception actually raised inside Python. Functions
# pickle by module reference; this is not an engine error slot or value registry.
_callback_observations: dict[str, CallbackObservation | bool] = {}


@pytest.mark.parametrize("outcome", ["success", "transport", "decode", "engine", "host"])
def test_source_history_does_not_preflight_or_acknowledge_execution(monkeypatch, outcome):
    events = []
    original = ValueError("original history boundary")

    responses = {
        "decode": ([b"\x80"], None),
        "engine": (
            [
                pb.RunProgramResponse(
                    result=Oneof[Literal["error"], pb.Error](
                        "error", pb.Error(code=pb.ErrorCode.RUNTIME_FAILURE, message="native failure")
                    )
                ).to_binary()
            ],
            None,
        ),
        "host": (output_frames(), original),
    }

    class InertTransport:
        def create(self, encoded):
            assert pb.CreateEGraphRequest.from_binary(encoded).record_history
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.egraph_id == 1
            events.append("run")
            if outcome == "transport":
                raise original
            return responses.get(outcome, (output_frames(), None))

        def render_source(self, encoded):
            assert pb.GetHistoryRequest.from_binary(encoded).egraph_id == 1
            events.append("export")
            return "accepted input"

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph(save_egglog_string=True)
    if outcome == "success":
        graph.register(i64(1))
    else:
        with pytest.raises(EggSmolError if outcome == "engine" else ValueError) as caught:
            graph.register(i64(1))
        if outcome in {"transport", "host"}:
            assert caught.value is original
        if outcome == "engine":
            assert caught.value.__dict__["_egglog_proto_error"].code == pb.ErrorCode.RUNTIME_FAILURE
    assert events == ["run"]
    assert graph.as_egglog_string == "accepted input"
    assert events == ["run", "export"]


def test_source_history_submits_freeze_with_installation(monkeypatch):
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    submitted = []

    class InertTransport:
        def create(self, encoded):
            assert pb.CreateEGraphRequest.from_binary(encoded).record_history
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "freeze"
            assert request.program.declarations == []
            assert request.program.sorts == [pb.Sort(kind=Oneof("eq", "installed"))]
            submitted.append(request.program)
            return output_frames(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    state = EGraphState(save_egglog_string=True)
    builder = Builder()
    sort = builder.add("sorts", pb.Sort(kind=Oneof("eq", "installed")))
    command = builder.add("commands", pb.Command(kind=Oneof[Literal["freeze"], pb.Freeze]("freeze", pb.Freeze())))
    owner = builder.publish()
    state.run_program([owner.ref("commands", command)], roots=[owner.ref("sorts", sort)])
    assert len(submitted) == 1


def test_freeze_retains_unused_nominal_sorts_and_unset_binding_slots():
    from egglog._program import _definition_key  # noqa: PLC0415
    from egglog.runtime import RuntimeExpr  # noqa: PLC0415

    builder = Builder(environment=(builtin_catalog().owner,))
    nominal = builder.add("sorts", pb.Sort(kind=Oneof("eq", "unused"), doc="retained nominal documentation"))
    value_ref = i64(7).__egg_ref__
    value = builder.import_ref(value_ref)
    integer = builder.import_ref(value_ref.owner.ref("sorts", value_ref.read().sort_id))
    slot = builder.add("nodes", pb.Node(sort_id=integer, kind=Oneof("binding", "unset")))
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof(
                "loop",
                pb.Loop(max_iterations=0, body=[pb.Command(kind=Oneof("let", pb.Let(binding=slot, value=value)))]),
            )
        ),
    )
    owner = builder.publish()
    graph = EGraph()
    graph._state.run_program(
        [owner.ref("commands", command)], roots=[owner.ref("sorts", nominal), owner.ref("nodes", slot)]
    )
    snapshot = graph.freeze()
    assert snapshot.program is not None
    assert any(
        sort.kind == Oneof("eq", "unused") and sort.doc == "retained nominal documentation"
        for sort in snapshot.program.sorts
    )
    assert any(node.kind == Oneof("binding", "unset") for node in snapshot.program.nodes)
    assert not any(command.kind is not None and command.kind.field == "let" for command in snapshot.program.commands)
    assert graph.lookup_function_value(RuntimeExpr(owner.ref("nodes", slot))) is None
    restored_owner = Builder.from_program(snapshot.program, environment=(builtin_catalog().owner,)).publish()
    restored = EGraph()
    restored._state.run_program(
        roots=[
            restored_owner.ref(role, index)
            for role in ("sorts", "nodes", "declarations")
            for index, record in enumerate(getattr(snapshot.program, role))
            if _definition_key(record) is not None
        ],
        commands=[restored_owner.ref("commands", index) for index in range(len(snapshot.program.commands))],
    )
    assert restored.freeze() == snapshot
    restored.let("unset", i64(9))
    restored.check(eq(RuntimeExpr(owner.ref("nodes", slot))).to(i64(9)))
    assert graph.lookup_function_value(RuntimeExpr(owner.ref("nodes", slot))) is None


def test_source_producer_returns_only_program_bytes_and_scope_counts():
    source = bindings._ProtoEngine.source(
        "(function source_scoped () i64 :merge new)"
        " (relation source_scoped_seen (i64))"
        " (rule ((= e (source_scoped))) ((source_scoped_seen e)))"
        " (function e () i64 :merge new) (set (source_scoped) 7)"
        " (push 1) (set (source_scoped) 9) (check (= (source_scoped) 9))"
        " (pop 1) (check (= (source_scoped) 7))"
    )
    kinds = []
    commands = []
    variables = set()
    while (submission := source.next_submission()) is not None:
        kind, encoded, count = submission
        kinds.append(kind)
        if kind == "program":
            assert type(encoded) is bytes
            program = pb.Program.from_binary(encoded)
            assert program.to_binary() == encoded
            assert count == 0
            commands.append([command.kind.field for command in program.commands if command.kind is not None])
            variables.update(
                node.kind.value for node in program.nodes if node.kind is not None and node.kind.field == "var"
            )
        else:
            assert kind in {"push", "pop"}
            assert encoded is None
            assert count == 1
    assert kinds == ["program", "push", "program", "pop", "program"]
    assert commands == [["action"], ["action", "check"], ["check"]]
    assert variables == {"e", "new"}
    with pytest.raises(ValueError, match="requires a host callback"):
        bindings._ProtoEngine.source("(print-function missing :mode csv)").next_submission()


@pytest.mark.parametrize("barrier", [False, True])
def test_source_producer_later_error_rejects_only_unsubmitted_block(barrier):
    graph = EGraph()
    text = "(function source_prefix () i64 :merge new) (set (source_prefix) 7)"
    if barrier:
        text += " (push 1) (set (source_prefix) 9)"
    source = bindings._ProtoEngine.source(text + " (missing_source_command)")
    try:
        assert graph._state.handle is not None
        if barrier:
            submission = source.next_submission()
            assert submission is not None
            kind, encoded, count = submission
            assert (kind, count) == ("program", 0)
            assert type(encoded) is bytes
            response, host_error = graph._state.transport.run(
                pb.RunProgramRequest(egraph_id=graph._state.handle, program=pb.Program.from_binary(encoded)).to_binary()
            )
            assert host_error is None
            assert all(
                frame.result is not None and frame.result.field == "output" for frame in _decode_run_frames(response)
            )
            assert source.next_submission() == ("push", None, 1)
            graph.push()
        before = graph.freeze()
        with pytest.raises(ValueError, match="missing_source_command"):
            source.next_submission()
        assert graph.freeze() == before

        check = "(check (= (source_prefix) 7))"
        verification = bindings._ProtoEngine.source(
            "(function source_prefix () i64 :merge new) " + (check if barrier else f"(fail {check})")
        )
        while (submission := verification.next_submission()) is not None:
            kind, encoded, count = submission
            assert (kind, count) == ("program", 0)
            assert encoded is not None
            response, host_error = graph._state.transport.run(
                pb.RunProgramRequest(egraph_id=graph._state.handle, program=pb.Program.from_binary(encoded)).to_binary()
            )
            assert host_error is None
            assert all(
                frame.result is not None and frame.result.field == "output" for frame in _decode_run_frames(response)
            )
    finally:
        for state in [graph._state, *graph._state_stack]:
            state.destroy()


@pytest.mark.parametrize("located", [False, True])
def test_source_block_streams_outputs_before_runtime_failure(located):
    graph = EGraph(record_history=True)
    source = bindings._ProtoEngine.source(
        "(function source_streamed () i64 :merge new) (set (source_streamed) 7)"
        " (rule ((= e (source_streamed))) ((set (source_streamed) (+ e 1))))"
        ' (print-size source_streamed) (panic "source stream marker") (set (source_streamed) 9)'
    )
    try:
        assert graph._state.handle is not None
        submission = source.next_submission()
        assert submission is not None
        kind, encoded, count = submission
        assert (kind, count) == ("program", 0)
        assert type(encoded) is bytes
        program = pb.Program.from_binary(encoded)
        assert [command.kind.field for command in program.commands if command.kind is not None] == [
            "action",
            "table_sizes",
            "action",
            "action",
        ]
        assert len(program.files) == 1
        original_file = pb.SourceFile.from_binary(program.files[0].to_binary())
        program.files.insert(0, pb.SourceFile(name="unrelated.egg", contents="private unrelated source"))
        for record in _nested_messages(program):
            if isinstance(record, pb.Span):
                record.file += 1
            elif not located and any(field.name == "span" for field in record.desc().fields):
                record.span = None
        panic_span = program.commands[2].span
        assert (panic_span is not None) is located
        rule = program.rulesets[0].value.rules[0]
        rule.doc = "retained rule documentation"
        program.commands.insert(2, pb.Command(kind=Oneof("freeze", pb.Freeze())))
        events = []
        chunks = []
        encoded_chunks = []

        def observe(encoded):
            encoded_chunks.append(encoded)
            chunk = pb.RunProgramResponse.from_binary(encoded)
            chunks.append(chunk)
            assert chunk.result is not None
            if chunk.result.field == "error":
                events.append("error")
                return False
            assert all(not isinstance(record, pb.Span | pb.SourceFile) for record in _nested_messages(chunk))
            events.extend(output.kind.field for output in chunk.result.value.outputs if output.kind is not None)
            return True

        terminal, host_error = graph._state.transport.run_stream(
            pb.RunProgramRequest(egraph_id=graph._state.handle, program=program).to_binary(), observe
        )
        events.append("returned")
        assert host_error is None
        assert events == ["sizes", "snapshot", "error", "returned"]
        failure = _required_error(_decode_run_frames([terminal]))
        assert failure.code == pb.ErrorCode.RUNTIME_FAILURE
        assert failure.location == pb.CommandLocation(path=[3])
        assert "source stream marker" in failure.message
        assert encoded_chunks[-1] == terminal
        if located:
            assert panic_span is not None
            assert failure.span == pb.Span(file=0, start=panic_span.start, end=panic_span.end)
            assert chunks[-1].files == [original_file]
        else:
            assert failure.span is None
            assert chunks[-1].files == []
        frozen = _required_data(chunks, "snapshot").outputs[0].kind.value
        assert frozen.program is not None
        assert frozen.program.rulesets[0].value.rules[0].doc == rule.doc
        assert frozen.program.rulesets[0].value.rules[0].name == rule.name
        history = graph.history()
        if located:
            assert original_file in history.files
            assert any(isinstance(record, pb.Span) for record in _nested_messages(history))
        assert history.rulesets[0].value.rules[0].doc == rule.doc
        sizes = _required_data(chunks, "sizes")
        assert sizes.outputs[0].kind == Oneof(
            "sizes", pb.TableSizes(sizes=[pb.TableSize(table="source_streamed", rows=1)])
        )

        verification = bindings._ProtoEngine.source(
            "(function source_streamed () i64 :merge new) (check (= (source_streamed) 7))"
        )
        while (submission := verification.next_submission()) is not None:
            kind, encoded, count = submission
            assert (kind, count) == ("program", 0)
            assert encoded is not None
            response, host_error = graph._state.transport.run(
                pb.RunProgramRequest(egraph_id=graph._state.handle, program=pb.Program.from_binary(encoded)).to_binary()
            )
            assert host_error is None
            assert all(
                frame.result is not None and frame.result.field == "output" for frame in _decode_run_frames(response)
            )
    finally:
        graph._state.destroy()


@pytest.fixture
def scheduler_transport(monkeypatch):
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    class InertTransport:
        outcome = "success"
        original = ValueError("original scheduler deployment failure")

        def __init__(self) -> None:
            self.requests: list[pb.RunProgramRequest] = []
            self.next_handle = 1

        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=self.next_handle, threads=1).to_binary()

        def run(self, encoded):
            self.requests.append(pb.RunProgramRequest.from_binary(encoded))
            if self.outcome == "transport":
                raise self.original
            if self.outcome == "decode":
                return [b"\x80"], None
            if self.outcome == "engine":
                return [
                    pb.RunProgramResponse(
                        result=Oneof[Literal["error"], pb.Error](
                            "error", pb.Error(code=pb.ErrorCode.RUNTIME_FAILURE, message="scheduler native failure")
                        )
                    ).to_binary()
                ], None
            return output_frames(), self.original if self.outcome == "host" else None

        def clone_graph(self, encoded):
            self.next_handle += 1
            return pb.CloneEGraphResponse(egraph_id=self.next_handle).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    transport = InertTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport)
    return EGraphState(save_egglog_string=True), transport


@pytest.mark.parametrize("outcome", ["success", "transport", "decode", "engine", "host"])
def test_persistent_scheduler_deployment_requires_acknowledgement(scheduler_transport, outcome):
    from egglog import back_off  # noqa: PLC0415

    state, transport = scheduler_transport
    transport.outcome = outcome
    scheduler = back_off(match_limit=0, ban_length=0).persistent()
    reference = scheduler.__egg_ref__
    name = reference.read().kind.value.name
    if outcome == "success":
        state.bind_scheduler(reference)
        # Different ownership of the exact config is still the same binding.
        builder = Builder()
        index = builder.add("commands", reference.read())
        state.bind_scheduler(builder.publish().ref("commands", index))
        assert len(transport.requests) == 1
        request = transport.requests[0]
        assert len(request.program.commands) == 1
        assert request.program.commands[0].kind.field == "bind_scheduler"
        assert not request.diagnostics
        assert state._scheduler_bindings[name] == reference.read().kind.value
        for options in (pb.BackOff(match_limit=1, ban_length=0), pb.BackOff(ban_length=0)):
            conflicting = reference.read()
            conflicting.kind.value.back_off = options
            builder = Builder()
            index = builder.add("commands", conflicting)
            with pytest.raises(ValueError, match="different configuration"):
                state.bind_scheduler(builder.publish().ref("commands", index))
        assert len(transport.requests) == 1
        return
    expected = EggSmolError if outcome == "engine" else ValueError
    with pytest.raises(expected) as caught:
        state.bind_scheduler(reference)
    if outcome in {"transport", "host"}:
        assert caught.value is transport.original
    if outcome == "engine":
        assert caught.value.__dict__["_egglog_proto_error"].code == pb.ErrorCode.RUNTIME_FAILURE
    transport.outcome = "success"
    assert state._scheduler_bindings[name] is None
    with pytest.raises(RuntimeError, match="unacknowledged deployment"):
        state.bind_scheduler(reference)
    assert len(transport.requests) == 1


def test_persistent_scheduler_deployment_follows_clone_and_saved_context(scheduler_transport):
    from egglog import back_off  # noqa: PLC0415

    saved, transport = scheduler_transport
    parent_binding = back_off(match_limit=2).persistent().__egg_ref__
    saved.bind_scheduler(parent_binding)
    active = saved.copy()
    active.bind_scheduler(parent_binding)
    assert len(transport.requests) == 1
    child_binding = back_off(ban_length=0).persistent().__egg_ref__
    active.bind_scheduler(child_binding)
    active.destroy()
    assert active.handle is None
    saved.bind_scheduler(parent_binding)
    saved.bind_scheduler(child_binding)
    assert [request.egraph_id for request in transport.requests] == [1, 2, 1]
    assert saved._scheduler_bindings is not active._scheduler_bindings


@pytest.mark.parametrize("case", ["unknown", "annotation", "algorithm"])
def test_host_model_contract_rejects_before_callbacks(case):
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    state = EGraphState()
    sorts = pb.Program(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))
            ),
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="PyObject"))
            ),
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="bool"))
            ),
        ],
    )

    def unused(_observation: bytes) -> object:
        pytest.fail("Rejected model contract must not invoke callbacks")

    assert state.handle is not None
    name = state.transport.register_cost_model(
        state.handle, sorts.to_binary(), 2 if case == "annotation" else 0, 1, False, unused
    )
    try:
        builder = Builder()
        sort = builder.add("sorts", sorts.sorts[0])
        root = builder.add(
            "nodes",
            pb.Node(
                sort_id=sort,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                ),
            ),
        )
        command = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["extract"], pb.Extract](
                    "extract",
                    pb.Extract(
                        roots=[root],
                        variants=1,
                        extractor=pb.Extractor.GREEDY_DAG if case == "algorithm" else pb.Extractor.TREE,
                        cost_model="missing-model" if case == "unknown" else name,
                    ),
                )
            ),
        )
        message = {
            "unknown": "unknown cost model",
            "annotation": "stored annotation sort",
            "algorithm": "additive marginal",
        }[case]
        with pytest.raises(EggSmolError, match=message) as caught:
            state.run_program([builder.publish().ref("commands", command)])
        error = caught.value.__dict__["_egglog_proto_error"]
        assert error.code == (pb.ErrorCode.RUNTIME_FAILURE if case == "unknown" else pb.ErrorCode.INVALID_PROGRAM)
    finally:
        state.transport.release_cost_model(state.handle, name)
        state.destroy()


@pytest.mark.parametrize(
    "state",
    ["term", "constant", "set", "constant-set", "union", "relation", "subsumed", "cost", "let-vec", "nested-vec"],
)
def test_shared_snapshot_restores_logical_state_over_bytes(state):
    from collections import Counter  # noqa: PLC0415

    from egglog import Vec, expr_parts, let, relation, set_cost, subsume, union  # noqa: PLC0415
    from egglog._program import StructuralView, _definition_key  # noqa: PLC0415

    class Atom(Expr):
        def __init__(self, value: i64) -> None: ...

    class Wrapper(Expr):
        def __init__(self, values: Vec[Atom]) -> None: ...

    @function
    def cell() -> i64: ...

    a, b = Atom(i64(1)), Atom(i64(2))
    marker = constant("snapshot-marker", Atom)
    scalar = constant("snapshot-scalar", i64)
    member = relation("snapshot_member", Atom)
    cases: dict[str, Callable[[], tuple[tuple[Action | BaseExpr | Fact, ...], BaseExpr | None]]] = {
        "term": lambda: ((a,), a),
        "constant": lambda: ((marker,), marker),
        "set": lambda: ((set_(cell()).to(i64(7)),), cell()),
        "constant-set": lambda: ((set_(scalar).to(i64(7)),), scalar),
        "union": lambda: ((union(a).with_(b),), a),
        "relation": lambda: ((member(a),), a),
        "subsumed": lambda: ((a, subsume(a)), None),
        "cost": lambda: ((a, set_cost(a, 10)), a),
        "let-vec": lambda: ((let("vector", Vec(a)),), Vec(a)),
        "nested-vec": lambda: ((Wrapper(Vec(a)),), Wrapper(Vec(a))),
    }
    actions, probe = cases[state]()
    graph = EGraph(*actions)
    expected = graph.extract(probe, include_cost=True) if probe is not None else None
    snapshot = graph.freeze()
    assert isinstance(snapshot, pb.EGraphSnapshot)
    assert snapshot.options is not None
    assert snapshot.program is not None
    encoded = snapshot.to_binary()
    graph._state.destroy()
    snapshot = pb.EGraphSnapshot.from_binary(encoded)
    assert snapshot.program is not None
    owner = Builder.from_program(snapshot.program, environment=(builtin_catalog().owner,)).publish()
    restored = EGraph()
    restored._state.run_program(
        pack(
            roots=[
                owner.ref(role, index)
                for role in ("sorts", "nodes", "declarations")
                for index, record in enumerate(getattr(snapshot.program, role))
                if _definition_key(record) is not None
            ],
            commands=[owner.ref("commands", index) for index in range(len(snapshot.program.commands))],
        )
    )
    if probe is not None:
        result, cost = restored.extract(probe, include_cost=True)
        assert expected is not None
        assert expr_parts(result) == expr_parts(expected[0])
        assert cost == expected[1]
    if state == "union":
        restored.check(eq(a).to(b))
    elif state == "relation":
        restored.check(member(a))
    roundtrip = restored.freeze()
    assert roundtrip.options is not None
    assert roundtrip.program is not None
    roundtrip_owner = Builder.from_program(roundtrip.program, environment=(builtin_catalog().owner,)).publish()
    assert StructuralView(roundtrip_owner.ref("sorts", roundtrip.options.cost_sort)) == StructuralView(
        owner.ref("sorts", snapshot.options.cost_sort)
    )
    options = pb.EGraphOptions.from_binary(roundtrip.options.to_binary())
    options.cost_sort = snapshot.options.cost_sort
    assert options == snapshot.options
    action_counts = []
    for frozen in (roundtrip, snapshot):
        frozen_program = frozen.program
        assert frozen_program is not None
        action_fields = []
        for command in frozen_program.commands:
            command_kind = command.kind
            assert command_kind is not None
            if command_kind.field == "let":
                binding = frozen_program.nodes[command_kind.value.binding]
                assert binding.kind == Oneof("binding", "vector")
                action_fields.append("let")
                continue
            assert command_kind.field == "action"
            action_kind = command_kind.value.kind
            assert action_kind is not None
            action_fields.append(action_kind.field)
        action_counts.append(Counter(action_fields))
    assert action_counts[0] == action_counts[1]


def _observed_pyerr_callback(tag):
    observation = _callback_observations[tag]
    assert isinstance(observation, dict)
    observation["calls"] += 1
    if "started" in observation:
        observation["started"].set()
        assert observation["release"].wait(10), "callback release timed out"
    if "barrier" in observation:
        observation["barrier"].wait(timeout=10)
    error = observation["error"]
    if error is not None:
        observation["caught"] = error
        raise error
    return observation.get("result", 8)


class _UnprintableCallback:
    __call__ = staticmethod(_observed_pyerr_callback)

    def __str__(self) -> str:
        _callback_observations["formatted"] = True
        return "expected failures must not format the callable"


@pytest.fixture
def callback_observations(monkeypatch):
    import sys  # noqa: PLC0415

    observations: dict[str, CallbackObservation | bool] = {}
    monkeypatch.setattr(sys.modules[__name__], "_callback_observations", observations)
    return observations


def _callback_program(tag, callback=_observed_pyerr_callback):
    """Author a closed call using the actual configured registrar, not fake signatures."""
    import cloudpickle  # noqa: PLC0415

    # The generation-only exporter is intentionally absent from the public stub.
    catalog = pb.BuiltinCatalog.from_binary(getattr(bindings, "_export_pyobject_catalog")())  # noqa: B009
    program = pb.Program(
        sorts=[
            next(
                record for record in catalog.sorts if record.kind == Oneof("primitive", pb.PrimitiveSortRef(name=name))
            )
            for name in ("PyObject", "i64", "Unit")
        ],
        files=catalog.files,
    )
    sort = 0
    first = len(program.nodes)
    for value in (callback, tag):
        program.nodes.append(
            pb.Node(
                sort_id=sort,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(
                        value=Oneof[Literal["custom"], pb.CustomValue](
                            "custom", pb.CustomValue(payload=cloudpickle.dumps(value))
                        )
                    ),
                ),
            )
        )
    program.nodes.append(
        pb.Node(
            sort_id=sort,
            kind=Oneof[Literal["call"], pb.Call](
                "call", pb.Call(func="egglog.python.pyobject.call", args=[first, first + 1])
            ),
        )
    )
    program.commands.append(
        pb.Command(
            kind=Oneof[Literal["extract"], pb.Extract](
                "extract", pb.Extract(roots=[first + 2], variants=1, extractor=pb.Extractor.TREE)
            )
        )
    )
    return program, first


def _callback_exchange(state, program, path):
    """Keep the real encoded request/response before interpreting the atomic error pair."""
    encoded = pb.RunProgramRequest(egraph_id=state.handle, program=program).to_binary()
    assert pb.RunProgramRequest.from_binary(encoded).to_binary() == encoded
    path.with_suffix(".request.pb").write_bytes(encoded)
    response, error = state.transport.run(encoded)
    assert isinstance(response, list)
    decoded = _record_frames(response, path.with_suffix(".response"))
    assert [pb.RunProgramResponse.from_binary(frame.to_binary()) for frame in decoded] == decoded
    return decoded, error


def test_primitive_failure_public_exception_constructor():
    assert PrimitiveFailure is bindings.PrimitiveFailure
    assert PrimitiveFailure.__name__ == "PrimitiveFailure"
    assert PrimitiveFailure.__module__ == "egglog.bindings"
    assert PrimitiveFailure.__bases__ == (Exception,)

    class NoInit(PrimitiveFailure):
        def __init__(self, *args) -> None:
            pass

    class SuperNew(PrimitiveFailure):
        def __new__(cls, *args) -> Self:
            return super().__new__(cls, *args)

    marker = object()
    argument_sets: tuple[tuple[object, ...], ...] = ((), (marker,), (marker, "message", 17))
    for constructor in (PrimitiveFailure, NoInit, SuperNew):
        for args in argument_sets:
            error = constructor(*args)
            assert type(error) is constructor
            assert error.args == args
            if args:
                assert error.args[0] is marker
        with pytest.raises(TypeError):
            constructor(message="not a positional argument")  # type: ignore[call-arg]


def test_callback_python_none_is_a_value(callback_observations, tmp_path):
    import cloudpickle  # noqa: PLC0415

    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    callback_observations["none"] = {"error": None, "result": None, "calls": 0}
    state = EGraphState()
    try:
        response, error = _callback_exchange(state, _callback_program("none")[0], tmp_path / "none")
        assert error is None
        assert all(frame.result is not None and frame.result.field == "output" for frame in response)
        response = _required_data(response, "extraction")
        assert callback_observations["none"]["calls"] > 0
        extraction = _required_oneof(response.outputs[0].kind, "extraction", pb.ExtractResult)
        term = extraction.roots[0].variants[0].term
        primitive = _required_oneof(response.nodes[term].kind, "primitive_value", pb.PrimitiveValue)
        value = _required_oneof(primitive.value, "custom", pb.CustomValue)
        assert value.args == []
        assert cloudpickle.loads(value.payload) is None
    finally:
        state.destroy()


@pytest.mark.parametrize(
    "kind", ["value", "chain", "unpickleable", "expression", "hostile_notes", "malformed_notes", "same_name"]
)
def test_callback_error_original_identity_and_adoption(table_transport, callback_observations, monkeypatch, kind):
    class CustomError(ValueError):
        pass

    class HostileNotesError(CustomError):
        def add_note(self, note):
            msg = "annotation must not replace the original error"
            raise RuntimeError(msg)

    class Unpickleable:
        def __reduce_ex__(self, protocol) -> Never:
            pytest.fail("The original callback error argument was serialized")

    argument = Unpickleable() if kind == "unpickleable" else i64(7) if kind == "expression" else "actual error"
    error = (
        ValueError(argument)
        if kind == "value"
        else type("PrimitiveFailure", (ValueError,), {})(argument)
        if kind == "same_name"
        else HostileNotesError(argument)
        if kind == "hostile_notes"
        else CustomError(argument)
    )
    error.detail = {"owned": [1, 2]}  # type: ignore[union-attr]  # Fixture-owned dynamic exception metadata.
    BaseException.add_note(error, "original note")
    if kind == "malformed_notes":
        malformed_notes = {"preserve": "original note state"}
        error.__notes__ = malformed_notes
    if kind == "chain":
        cause, context = KeyError("cause"), LookupError("context")
        error.__cause__, error.__context__, error.__suppress_context__ = cause, context, True
        cause.__context__, context.__cause__ = context, error
    chain = error.__cause__, error.__context__, error.__suppress_context__
    callback_observations[kind] = {"error": error, "calls": 0}
    graph, requests, responses = table_transport
    program, first = _callback_program(kind)
    # A completed ordinary Custom output must survive the later callback error.
    program.commands.insert(
        0,
        pb.Command(
            kind=Oneof[Literal["extract"], pb.Extract](
                "extract", pb.Extract(roots=[first + 1], variants=1, extractor=pb.Extractor.TREE)
            )
        ),
    )
    source = Builder.from_program(program, environment=(builtin_catalog().owner,)).publish()
    packed = pack(commands=[source.ref("commands", index) for index in range(2)])
    # Python exposes the bound classmethod descriptor; MyPy represents only its callable signature.
    original_adopt = Builder.from_program.__func__  # type: ignore[attr-defined]
    adopted = []

    def adopt(cls, returned):
        adopted.append(returned)
        return original_adopt(cls, returned)

    with monkeypatch.context() as patch:
        patch.setattr(Builder, "from_program", classmethod(adopt))
        with pytest.raises(type(error)) as caught:
            graph._state.run_program(packed)
    assert caught.value is error is callback_observations[kind]["caught"]
    assert error.args == (argument,)
    assert error.args[0] is argument
    assert error.detail == {"owned": [1, 2]}
    assert (error.__cause__, error.__context__, error.__suppress_context__) == chain
    if kind == "chain":
        assert error.__cause__.__context__ is error.__context__
        assert error.__context__.__cause__ is error
    if kind == "hostile_notes":
        assert error.__notes__ == ["original note"]
    elif kind == "malformed_notes":
        assert error.__notes__ is malformed_notes
    else:
        assert error.__notes__[0] == "original note"
        assert error.__notes__[1].startswith("Calling function ")
        assert error.__notes__[2:] == ["While calling primitive 'py-call'"]
    traceback = error.__traceback__
    frames = []
    while traceback is not None:
        frames.append(traceback.tb_frame.f_code)
        traceback = traceback.tb_next
    assert _observed_pyerr_callback.__code__ in frames
    assert len(requests) == len(responses) == len(adopted) == 1
    failure = _required_error(responses[0])
    assert failure.code == pb.ErrorCode.RUNTIME_FAILURE
    assert failure.location is not None
    assert failure.location.path == [1]
    response = _required_data(responses[0], "extraction")
    assert len(response.outputs) == 1
    assert response.outputs[0].location is not None
    assert response.outputs[0].location.path == [0]
    assert adopted[0].nodes == response.nodes
    assert adopted[0].sorts == response.sorts
    extraction = _required_oneof(response.outputs[0].kind, "extraction", pb.ExtractResult)
    term = extraction.roots[0].variants[0].term
    primitive = _required_oneof(response.nodes[term].kind, "primitive_value", pb.PrimitiveValue)
    custom = _required_oneof(primitive.value, "custom", pb.CustomValue)
    original_primitive = _required_oneof(program.nodes[first + 1].kind, "primitive_value", pb.PrimitiveValue)
    original_custom = _required_oneof(original_primitive.value, "custom", pb.CustomValue)
    assert custom.payload == original_custom.payload
    assert custom.args == []


@pytest.mark.parametrize("failure", ["binary", "arena"])
def test_callback_error_protocol_failure_keeps_original_chain(monkeypatch, failure):
    error = ValueError("original host error")
    cause, context = KeyError("cause"), LookupError("context")
    error.__cause__, error.__context__, error.__suppress_context__ = cause, context, True
    cause.__context__, context.__cause__ = context, error
    response = output_frames()

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            pb.RunProgramRequest.from_binary(encoded)
            return ([b"\xff"] if failure == "binary" else response), error

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    packed = pack([])
    if failure == "arena":

        def reject(cls, program):
            msg = "response arena rejected"
            raise ValueError(msg)

        monkeypatch.setattr(Builder, "from_program", classmethod(reject))
    # Decode/adoption failures differ in type; this control asserts original-cause identity.
    with pytest.raises(Exception) as caught:  # noqa: PT011
        graph._state.run_program(packed)
    assert caught.value is not error
    assert caught.value.__cause__ is error
    assert (error.__cause__, error.__context__, error.__suppress_context__) == (cause, context, True)
    assert cause.__context__ is context
    assert context.__cause__ is error


def test_callback_error_prior_native_record_is_not_current_check_failure(monkeypatch):
    """An original host exception may itself retain an earlier native receipt."""
    original = EggSmolError("earlier native check")
    prior = pb.Error(
        code=pb.ErrorCode.CHECK_FAILED, message="earlier native check", location=pb.CommandLocation(path=[0])
    )
    original.__dict__["_egglog_proto_error"] = prior

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "check"
            return output_frames(), original

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(EggSmolError) as caught:
        graph.check_bool(eq(i64(1)).to(i64(1)))
    assert caught.value is original
    assert original.__dict__["_egglog_proto_error"] is prior


@pytest.mark.parametrize("domain", ["same", "clone", "independent"])
def test_callback_error_atomic_pairs_outlive_handles(callback_observations, tmp_path, domain):
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    first = EGraphState()
    second = first if domain == "same" else first.copy() if domain == "clone" else EGraphState()
    errors = [ValueError("request A"), LookupError("request B")]
    pairs = []
    try:
        for index, state in enumerate((first, second)):
            tag = str(index)
            callback_observations[tag] = {"error": errors[index], "calls": 0}
            program, _ = _callback_program(tag)
            pairs.append(_callback_exchange(state, program, tmp_path / f"request-{index}"))
        # The first returned object no longer occupies any mutable native slot.
        callback_observations["recovery"] = {"error": None, "calls": 0}
        program, _ = _callback_program("recovery")
        recovered, no_error = _callback_exchange(first, program, tmp_path / "recovery")
        assert all(frame.result is not None and frame.result.field == "output" for frame in recovered)
        assert no_error is None
    finally:
        if second is not first:
            second.destroy()
        first.destroy()
    for index in (1, 0):
        response, error = pairs[index]
        assert _required_error(response).code == pb.ErrorCode.RUNTIME_FAILURE
        assert error is errors[index] is callback_observations[str(index)]["caught"]
        with pytest.raises(type(error)) as caught:
            raise error
        assert caught.value is errors[index]


def test_callback_error_independent_engines_do_not_swap_concurrent_errors(callback_observations, tmp_path):
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from threading import Barrier  # noqa: PLC0415

    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    barrier = Barrier(2)
    states = [EGraphState(), EGraphState()]
    programs = []
    errors = [ValueError("parallel A"), KeyError("parallel B")]
    for ordinal in range(2):
        callback_observations[str(ordinal)] = {"error": errors[ordinal], "calls": 0, "barrier": barrier}
        programs.append(_callback_program(str(ordinal))[0])
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = [
                executor.submit(_callback_exchange, states[ordinal], programs[ordinal], tmp_path / str(ordinal))
                for ordinal in range(2)
            ]
            for ordinal, future in enumerate(pending):
                response, error = future.result(timeout=20)
                assert _required_error(response).code == pb.ErrorCode.RUNTIME_FAILURE
                assert error is errors[ordinal] is callback_observations[str(ordinal)]["caught"]
    finally:
        for state in states:
            state.destroy()


def test_callback_error_shared_transport_rejects_overlapping_borrow(callback_observations, tmp_path):
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from threading import Event  # noqa: PLC0415

    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    state = EGraphState()
    clone = state.copy()
    started, release = Event(), Event()
    original = ValueError("blocked callback")
    callback_observations["first"] = {"error": original, "calls": 0, "started": started, "release": release}
    callback_observations["second"] = {"error": LookupError("must not run"), "calls": 0}
    first, _ = _callback_program("first")
    second, _ = _callback_program("second")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(_callback_exchange, state, first, tmp_path / "first")
            try:
                assert started.wait(10)
                with pytest.raises(RuntimeError):
                    _callback_exchange(clone, second, tmp_path / "second")
                assert callback_observations["second"]["calls"] == 0
            finally:
                release.set()
            response, error = pending.result(timeout=20)
            assert _required_error(response).code == pb.ErrorCode.RUNTIME_FAILURE
            assert error is original
    finally:
        clone.destroy()
        state.destroy()


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "case",
    [
        "query_failure",
        "query_success",
        "extract_failure",
        "later_native_error",
        "query_expected",
        "query_expected_subclass",
        "extract_expected",
        "extract_expected_subclass",
    ],
)
def test_callback_error_native_prefix_suppression_and_recovery(  # noqa: C901, PLR0912
    callback_observations, tmp_path, threads, case
):
    # Shared setup preserves the same request-prefix, suppression and recovery boundary.
    import cloudpickle  # noqa: PLC0415

    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    class ExpectedSubclass(PrimitiveFailure):
        annotation_calls: int

        def add_note(self, note) -> None:
            self.annotation_calls += 1
            pytest.fail("Expected primitive failures must not be annotated")

    expected_failure = "expected" in case
    extracting = case.startswith("extract")
    original = (
        None
        if case == "query_success"
        else ExpectedSubclass("expected miss")
        if case.endswith("subclass")
        else PrimitiveFailure("expected miss")
        if expected_failure
        else ValueError("prefix error")
    )
    if expected_failure:
        # The case matrix installs this test counter only on expected-failure instances.
        original.annotation_calls = 0  # type: ignore[union-attr]
    callback_observations["first"] = {"error": original, "calls": 0}
    callback_observations["later"] = {"error": None, "calls": 0}
    program, first = _callback_program(
        "first", _UnprintableCallback() if case.endswith("subclass") else _observed_pyerr_callback
    )
    pyobject = program.nodes[first].sort_id
    integer, unit = (
        next(
            index
            for index, sort in enumerate(program.sorts)
            if sort.kind == Oneof("primitive", pb.PrimitiveSortRef(name=name))
        )
        for name in ("i64", "Unit")
    )
    marks = []
    for key, value in ((10, 1010), (20, 2020), (30, 3030)):
        marks.append((len(program.nodes), len(program.nodes) + 1))
        for number in (key, value):
            program.nodes.append(
                pb.Node(
                    sort_id=integer,
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", number))
                    ),
                )
            )
    unit_value = len(program.nodes)
    program.nodes.append(
        pb.Node(
            sort_id=unit,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["unit"], pb.Unit]("unit", pb.Unit()))
            ),
        )
    )
    x, y = len(program.nodes), len(program.nodes) + 1
    program.nodes.extend([
        pb.Node(sort_id=pyobject, kind=Oneof[Literal["var"], str]("var", name)) for name in ("x", "y")
    ])
    seed = len(program.nodes)
    program.nodes.append(
        pb.Node(sort_id=unit, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="handoff_seed", args=[x])))
    )
    query_call = len(program.nodes)
    program.nodes.append(
        pb.Node(
            sort_id=pyobject,
            kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.python.pyobject.call", args=[first, x])),
        )
    )
    later_argument = len(program.nodes)
    program.nodes.append(
        pb.Node(
            sort_id=pyobject,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["custom"], pb.CustomValue](
                        "custom", pb.CustomValue(payload=cloudpickle.dumps("later"))
                    )
                ),
            ),
        )
    )
    later_call = len(program.nodes)
    program.nodes.append(
        pb.Node(
            sort_id=pyobject,
            kind=Oneof[Literal["call"], pb.Call](
                "call", pb.Call(func="egglog.python.pyobject.call", args=[first, later_argument])
            ),
        )
    )
    program.declarations.extend([
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function](
                "function", pb.Function(name="handoff_mark", inputs=[pb.Arg(sort=integer)], output=integer)
            )
        ),
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function](
                "function", pb.Function(name="handoff_seed", inputs=[pb.Arg(sort=pyobject)], output=unit)
            )
        ),
    ])
    authored_rules = []
    for name, call in (("handoff_first", query_call), ("handoff_later", later_call)):
        query = len(program.nodes)
        program.nodes.append(
            pb.Node(sort_id=pyobject, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[y, call])))
        )
        authored_rules.append(
            pb.Rule(
                name=name,
                eval_mode=pb.RuleEvalMode.SEMINAIVE,
                kind=Oneof[Literal["rule"], pb.RuleBody](
                    "rule",
                    pb.RuleBody(
                        query=[seed, query],
                        head=[
                            pb.Action(
                                kind=Oneof[Literal["set"], pb.Set](
                                    "set",
                                    pb.Set(target=pb.Call(func="handoff_mark", args=[marks[2][0]]), value=marks[2][1]),
                                )
                            )
                        ],
                    ),
                ),
            )
        )
    program.rulesets = [
        pb.RulesetBinding(name=name, value=pb.Ruleset(rules=[authored_rules[index]]))
        for index, name in enumerate(("handoff_first", "handoff_later"))
    ]
    program.commands = [
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action",
                pb.Action(
                    kind=Oneof[Literal["set"], pb.Set](
                        "set", pb.Set(target=pb.Call(func="handoff_seed", args=[first + 1]), value=unit_value)
                    )
                ),
            )
        ),
    ]
    state = EGraphState(num_threads=threads)
    try:
        setup, error = _callback_exchange(state, program, tmp_path / "setup")
        assert all(frame.result is not None and frame.result.field == "output" for frame in setup)
        assert error is None
        assert callback_observations["first"]["calls"] == callback_observations["later"]["calls"] == 0
        program.commands = [
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["set"], pb.Set](
                            "set", pb.Set(target=pb.Call(func="handoff_mark", args=[key]), value=value)
                        )
                    ),
                )
            )
            for key, value in marks[:2]
        ]
        program.commands.insert(
            1,
            pb.Command(
                kind=(
                    Oneof[Literal["extract"], pb.Extract](
                        "extract", pb.Extract(roots=[first + 2], variants=1, extractor=pb.Extractor.TREE)
                    )
                    if extracting
                    else Oneof[Literal["run"], pb.Run](
                        "run",
                        pb.Run(
                            rulesets=[pb.RulesetSelection(kind=Oneof[Literal["name"], str]("name", "handoff_first"))]
                        ),
                    )
                )
            ),
        )
        program.commands.append(
            pb.Command(
                kind=(
                    Oneof[Literal["action"], pb.Action](
                        "action", pb.Action(kind=Oneof[Literal["panic"], str]("panic", "native after callback"))
                    )
                    if case == "later_native_error"
                    else Oneof[Literal["run"], pb.Run](
                        "run",
                        pb.Run(
                            rulesets=[pb.RulesetSelection(kind=Oneof[Literal["name"], str]("name", "handoff_later"))]
                        ),
                    )
                )
            )
        )
        if expected_failure and not extracting:
            # Observe the first rule's no-match before the later successful callback writes its head.
            program.commands.insert(
                3,
                pb.Command(
                    kind=Oneof[Literal["table_rows"], pb.GetTableRows](
                        "table_rows", pb.GetTableRows(table="handoff_mark")
                    )
                ),
            )
        response, captured = _callback_exchange(state, program, tmp_path / "execution")
        assert captured is (None if expected_failure else original)
        assert response[-1].result is not None
        assert (response[-1].result.field == "error") == (extracting or case == "later_native_error")
        if expected_failure:
            assert callback_observations["first"]["caught"] is original
            assert original.annotation_calls == 0
            assert not hasattr(original, "__notes__")
            assert not callback_observations.get("formatted", False)
            if extracting:
                assert callback_observations["later"]["calls"] == 0
            else:
                assert callback_observations["later"]["calls"] > 0
                (printed,) = _required_data(response, "table_rows").outputs
                assert printed.location is not None
                assert printed.location.path == [3]
                assert len(_required_oneof(printed.kind, "table_rows", pb.TableRows).rows) == 2
        elif original is not None:
            assert captured is callback_observations["first"]["caught"]
            assert callback_observations["later"]["calls"] == 0
        if case == "later_native_error":
            failure = _required_error(response)
            assert failure.location is not None
            assert failure.location.path == [3]
            assert "native after callback" in failure.message
        elif extracting:
            failure = _required_error(response)
            assert failure.code == pb.ErrorCode.RUNTIME_FAILURE
            assert failure.location is not None
            assert failure.location.path == [1]
        program.commands = [
            pb.Command(
                kind=Oneof[Literal["table_rows"], pb.GetTableRows]("table_rows", pb.GetTableRows(table="handoff_mark"))
            )
        ]
        observed, error = _callback_exchange(state, program, tmp_path / "markers")
        assert all(frame.result is not None and frame.result.field == "output" for frame in observed)
        observed = _required_data(observed, "table_rows")
        assert error is None
        rows = {}
        table = _required_oneof(observed.outputs[0].kind, "table_rows", pb.TableRows)
        for row in table.rows:
            values = []
            assert row.output is not None
            for cell in (*row.args, row.output):
                term = _required_oneof(cell.kind, "term", int)
                primitive = _required_oneof(observed.nodes[term].kind, "primitive_value", pb.PrimitiveValue)
                values.append(_required_oneof(primitive.value, "i64", int))
            assert len(values) == 2
            rows[values[0]] = values[1]
        expected = {10: 1010}
        if not extracting:
            expected[20] = 2020
        if case == "query_success" or (expected_failure and not extracting):
            expected[30] = 3030
        assert rows == expected
        callback_observations["recovery"] = {"error": None, "calls": 0}
        recovered, error = _callback_exchange(state, _callback_program("recovery")[0], tmp_path / "recovery")
        assert all(frame.result is not None and frame.result.field == "output" for frame in recovered)
        recovered = _required_data(recovered, "extraction")
        assert error is None
        extraction = _required_oneof(recovered.outputs[0].kind, "extraction", pb.ExtractResult)
        term = extraction.roots[0].variants[0].term
        primitive = _required_oneof(recovered.nodes[term].kind, "primitive_value", pb.PrimitiveValue)
        payload = _required_oneof(primitive.value, "custom", pb.CustomValue)
        assert payload.args == []
        assert cloudpickle.loads(payload.payload) == 8
    finally:
        state.destroy()


@pytest.fixture
def table_transport(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Table inspection used the legacy engine")

    graph = EGraph()
    transport = graph._state.transport
    requests: list[pb.RunProgramRequest] = []
    responses: list[list[pb.RunProgramResponse]] = []

    class AuditedTransport:
        destroy = transport.destroy
        register_cost_model = transport.register_cost_model
        release_cost_model = transport.release_cost_model

        def run(self, encoded):
            assert isinstance(encoded, bytes)
            position = len(requests)
            (tmp_path / f"request-{position}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned, host_error = transport.run(encoded)
            assert isinstance(returned, list)
            responses.append(_record_frames(returned, tmp_path / f"response-{position}"))
            return returned, host_error

    monkeypatch.setattr(graph._state, "transport", AuditedTransport())
    yield graph, requests, responses
    graph._state.transport = transport
    graph._state.destroy()


def test_tree_cost_callback_failure_does_not_publish_term_records(table_transport):
    graph, requests, responses = table_transport
    original = LookupError("base cost failed")

    def fail(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
        assert isinstance(expr, i64)
        raise original

    with pytest.raises(LookupError) as caught:
        graph.extract(i64(1), cost_model=fail)
    assert caught.value is original
    assert len(requests) == len(responses) == 1
    assert len(responses[0]) == 1
    assert _required_error(responses[0]).code == pb.ErrorCode.RUNTIME_FAILURE
    assert graph.extract(i64(2)) == i64(2)


def test_function_sizes_native_bytes_register_and_filter_unmapped(table_transport):
    graph, requests, responses = table_transport

    @function(egg_fn="z-known")
    def known(value: i64) -> i64: ...

    @function(egg_fn="a-unmapped")
    def unmapped(value: i64) -> i64: ...

    assert graph.function_size(known) == 0
    assert len(requests) == 1
    assert any(decl.kind.value.name == "z-known" for decl in requests[0].program.declarations)
    assert requests[0].program.commands[0].kind == Oneof("table_sizes", pb.GetTableSizes(tables=["z-known"]))
    observed = _required_data(responses[0], "sizes")
    assert observed.outputs[0].location == pb.CommandLocation(path=[0])
    assert observed.outputs[0].kind == Oneof("sizes", pb.TableSizes(sizes=[pb.TableSize(table="z-known", rows=0)]))
    graph.register(set_(known(i64(1))).to(i64(2)), set_(unmapped(i64(1))).to(i64(3)))
    # The legacy public API filtered native tables lacking a Python mapping.
    # Keep the native row, but remove only its canonical language lookup entry.
    del graph._state.definitions["callable", "a-unmapped"]
    assert graph.all_function_sizes() == [(known, 1)]
    assert requests[-1].program.declarations == []
    assert requests[-1].program.commands[0].kind == Oneof("table_sizes", pb.GetTableSizes())
    observed = _required_data(responses[-1], "sizes")
    table_sizes = _required_oneof(observed.outputs[0].kind, "sizes", pb.TableSizes)
    assert [entry.table for entry in table_sizes.sizes] == ["a-unmapped", "z-known"]


def test_function_sizes_include_unavailable_and_subsumed_native_bytes(table_transport):
    graph, _, _ = table_transport
    program = pb.Program(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))
            ),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "CountedHidden")),
        ],
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor",
                    pb.Constructor(name="hidden-count", inputs=[pb.Arg(sort=0)], output=1, unextractable=True),
                )
            ),
        ],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", n))
                ),
            )
            for n in (1, 2)
        ]
        + [
            pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="hidden-count", args=[n])))
            for n in (0, 1)
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", n))
                )
            )
            for n in (2, 3)
        ]
        + [
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["subsume"], pb.Call]("subsume", pb.Call(func="hidden-count", args=[0]))
                    ),
                )
            )
        ],
    )
    owner = Builder.from_program(program).publish()
    graph._state.run_program([owner.ref("commands", n) for n in range(3)])
    target = RuntimeFunction(owner.ref("declarations", 0))
    assert graph.function_size(target) == 2
    assert graph.all_function_sizes() == [(target, 2)]
    with pytest.raises(ValueError, match="has no extractable representative"):
        graph.function_values(target)


def test_table_callable_projection_keeps_first_view_without_invocation(monkeypatch):
    builder = Builder.from_program(
        pb.Program(
            sorts=[
                pb.Sort(
                    kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))
                )
            ],
            declarations=[
                pb.Declaration(
                    kind=Oneof[Literal["function"], pb.Function](
                        "function", pb.Function(name="two-views", inputs=[pb.Arg(sort=0)], output=0)
                    ),
                )
            ],
        )
    )
    builder.presentations["callable", "two-views"] = Presentation(
        views=tuple(
            CallPresentation(kind=CallKind.FUNCTION, path=(name,), params=(Parameter(name="value", core_input=0),))
            for name in ("first", "second")
        )
    )
    owner = builder.publish()

    def forbidden(*args, **kwargs):
        pytest.fail("Callable projection invoked a Python callable")

    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    reference = owner.ref("declarations", 0)
    projected = create_callable(reference)
    assert projected == RuntimeFunction(reference, 0)
    assert hash(projected) == hash(RuntimeFunction(reference, 0))
    assert projected != RuntimeFunction(reference, 1)


def test_table_constant_projection_does_not_invoke_callable(monkeypatch):
    current = constant("projected-current", i64)
    reference = resolve_callable(current)

    def forbidden(*args, **kwargs):
        pytest.fail("Constant projection invoked a Python callable")

    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    projected = create_callable(reference)
    assert projected == current
    assert hash(projected) == hash(current)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("corruption", ["outputs", "kind", "location", "table", "duplicate"])
def test_function_sizes_reject_malformed_response_without_engine(monkeypatch, selected, corruption):
    @function(egg_fn="size-table")
    def table(value: i64) -> i64: ...

    sizes = pb.TableSizes(sizes=[pb.TableSize(table="size-table", rows=1)])
    output = pb.CommandOutput(
        kind=Oneof[Literal["sizes"], pb.TableSizes]("sizes", sizes),
        location=pb.CommandLocation(path=[0]),
    )
    response = pb.RunProgramOutput(
        outputs=[output],
    )
    match corruption:
        case "outputs":
            response.outputs.clear()
        case "kind":
            output.kind = Oneof[Literal["table_rows"], pb.TableRows]("table_rows", pb.TableRows())
        case "location":
            output.location = pb.CommandLocation(path=[1])
        case "table":
            sizes.sizes[0].table = ""
        case "duplicate":
            sizes.sizes.append(pb.TableSize(table="size-table", rows=2))

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "table_sizes"
            return output_frames(response), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    message = "exactly one requested observation" if corruption == "outputs" else "table size"
    with pytest.raises(ValueError, match=message):
        graph.function_size(table) if selected else graph.all_function_sizes()


def test_function_values_native_bytes_and_shared_terms(table_transport):
    graph, requests, responses = table_transport

    @function
    def repeated(left: i64, right: i64) -> i64: ...

    graph.register(set_(repeated(i64(1), i64(1))).to(i64(1)), set_(repeated(i64(2), i64(3))).to(i64(7)))
    rows = graph.function_values(repeated)
    assert len(requests) == 2
    command = requests[-1].program.commands[0].kind
    assert command.field == "table_rows"
    assert command.value.table == resolve_callable(repeated).read().kind.value.name
    assert not command.value.has_field("max_rows")
    observed = _required_data(responses[-1], "table_rows")
    output = observed.outputs[0]
    assert output.location == pb.CommandLocation(path=[0])
    table = _required_oneof(output.kind, "table_rows", pb.TableRows)
    row = table.rows[0]
    assert row.args[0] == row.args[1] == row.output == pb.TableCell(kind=Oneof[Literal["term"], int]("term", 0))
    assert _required_oneof(observed.nodes[0].kind, "primitive_value", pb.PrimitiveValue).value == Oneof("i64", 1)
    key, result = next(iter(rows.items()))
    args = key.__egg_ref__.read().kind.value.args
    assert key.__egg_ref__.owner.ref("nodes", args[0]) == key.__egg_ref__.owner.ref("nodes", args[1])
    assert graph.function_values(repeated, 0) == {}
    assert requests[-2].program.commands == []
    assert requests[-1].program.commands[0].kind.value.has_field("max_rows")
    assert requests[-1].program.commands[0].kind.value.max_rows == 0
    graph._state.destroy()
    assert rows == {repeated(i64(1), i64(1)): i64(1), repeated(i64(2), i64(3)): i64(7)}
    assert result.value == 1
    assert str(key) == "repeated(1, 1)"
    assert len(requests) == 4


@pytest.mark.parametrize("value", [-1, 1])
def test_function_values_installs_before_index_conversion_native_bytes(table_transport, value):
    graph, requests, responses = table_transport

    @function
    def untouched(value: i64) -> i64: ...

    calls = []

    class IndexedLength:
        def __index__(self) -> int:
            calls.append(len(requests))
            assert len(requests) == 1
            assert requests[0].program.commands == []
            assert any(decl.kind.field == "function" for decl in requests[0].program.declarations)
            assert all(frame.result is not None and frame.result.field == "output" for frame in responses[0])
            return value

    if value < 0:
        with pytest.raises(OverflowError, match="can't convert negative int to unsigned") as caught:
            graph.function_values(untouched, IndexedLength())
        assert caught.value.__notes__ == ["while processing 'length'"]
        assert len(requests) == 1
    else:
        assert graph.function_values(untouched, IndexedLength()) == {}
        assert len(requests) == 2
    assert calls == [1]
    assert graph.function_values(untouched) == {}


@pytest.mark.parametrize(
    ("length", "error", "message"),
    [
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.0, TypeError, "'float' object cannot be interpreted as an integer"),
        (2**64, OverflowError, "int too big to convert"),
    ],
)
def test_function_values_bad_length_keeps_installed_table_native_bytes(table_transport, length, error, message):
    graph, requests, responses = table_transport

    @function
    def untouched(value: i64) -> i64: ...

    with pytest.raises(error, match=message) as caught:
        graph.function_values(untouched, length)
    assert caught.value.__notes__ == ["while processing 'length'"]
    assert len(requests) == 1
    assert requests[0].program.commands == []
    assert all(frame.result is not None and frame.result.field == "output" for frame in responses[0])
    assert graph.function_values(untouched) == {}


@pytest.mark.parametrize("unavailable", ["argument 1", "output"])
def test_function_values_unavailable_cells_raise_from_native_bytes(table_transport, unavailable):
    graph, requests, responses = table_transport
    # Canonical backend fixture: public unextractable authoring stays guarded.
    sorts = [
        pb.Sort(kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))),
        pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Hidden")),
    ]
    nodes = [
        pb.Node(
            sort_id=0,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", n))
            ),
        )
        for n in (1, 2)
    ]
    nodes.extend([
        pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="Hidden", args=[0]))),
        pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="Visible"))),
    ])
    inputs = [pb.Arg(sort=0), pb.Arg(sort=1)] if unavailable == "argument 1" else [pb.Arg(sort=0)]
    program = pb.Program(
        sorts=sorts,
        nodes=nodes,
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name="Hidden", inputs=[pb.Arg(sort=0)], output=1, unextractable=True)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name="Visible", output=1)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function",
                    pb.Function(name="inspected", inputs=inputs, output=0 if unavailable == "argument 1" else 1),
                )
            ),
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["set"], pb.Set](
                            "set",
                            pb.Set(
                                target=pb.Call(
                                    func="inspected", args=[number, value] if unavailable == "argument 1" else [number]
                                ),
                                value=number if unavailable == "argument 1" else value,
                            ),
                        )
                    ),
                )
            )
            for number, value in ((0, 3), (1, 2))
        ],
    )
    owner = Builder.from_program(program).publish()
    graph._state.run_program([owner.ref("commands", n) for n in range(2)])
    target = RuntimeFunction(owner.ref("declarations", 2))
    with pytest.raises(ValueError, match=f"Table 'inspected' row 1 {unavailable} has no extractable representative"):
        graph.function_values(target)
    observed = _required_data(responses[-1], "table_rows")
    table = _required_oneof(observed.outputs[0].kind, "table_rows", pb.TableRows)
    assert all(frame.result is not None and frame.result.field == "output" for frame in responses[-1])
    assert len(table.rows) == 2
    cell = table.rows[1].args[1] if unavailable == "argument 1" else table.rows[1].output
    assert cell is not None
    unavailable_sort = _required_oneof(cell.kind, "unavailable_sort", int)
    assert observed.sorts[unavailable_sort].kind == Oneof("eq", "Hidden")
    assert requests[-1].program.commands[0].kind.field == "table_rows"
    assert graph.function_values(target, 0) == {}


@pytest.mark.parametrize(
    "corruption",
    [
        "outputs",
        "kind",
        "location",
        "table",
        "arity",
        "output",
        "cell",
        "index",
        "node",
        "argument sort",
        "output sort",
    ],
)
def test_function_values_rejects_malformed_response_without_engine(monkeypatch, corruption):  # noqa: C901
    @function(egg_fn="table")
    def table(value: i64) -> i64: ...

    available = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 0))
    row = pb.FunctionRow(args=[available], output=available)
    output = pb.CommandOutput(
        kind=Oneof[Literal["table_rows"], pb.TableRows]("table_rows", pb.TableRows(table="table", rows=[row])),
        location=pb.CommandLocation(path=[0]),
    )
    response = pb.RunProgramOutput(
        sorts=[
            pb.Sort(kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name=name)))
            for name in ("i64", "Bool")
        ],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                ),
            ),
            pb.Node(
                sort_id=1,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["bool"], bool]("bool", True))
                ),
            ),
        ],
        outputs=[output],
    )
    # Generated messages retain the assigned child messages in this inert fixture.
    match corruption:
        case "outputs":
            response.outputs.clear()
        case "kind":
            output.kind = Oneof[Literal["extraction"], pb.ExtractResult]("extraction", pb.ExtractResult())
        case "location":
            output.location = pb.CommandLocation(path=[1])
        case "table":
            assert output.kind is not None
            assert output.kind.field == "table_rows"
            output.kind.value.table = "wrong"
        case "arity":
            row.args.clear()
        case "output":
            row.output = None
        case "cell":
            row.args[0] = pb.TableCell()
        case "index":
            row.args[0] = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 99))
        case "node":
            response.nodes[0].kind = None
        case "argument sort":
            row.args[0] = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 1))
        case "output sort":
            row.output = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 1))

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "table_rows"
            return output_frames(response), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    message = "exactly one requested observation" if corruption == "outputs" else r"table|Table"
    with pytest.raises(ValueError, match=message):
        EGraph().function_values(table)


def test_function_values_rejects_primitives_without_table_request(table_transport):
    graph, requests, _ = table_transport
    owner = Builder.from_program(
        pb.Program(
            sorts=[
                pb.Sort(
                    kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))
                )
            ],
            nodes=[
                pb.Node(
                    sort_id=0,
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                    ),
                )
            ],
            declarations=[
                pb.Declaration(
                    kind=Oneof[Literal["user_defined_primitive"], pb.UserDefinedPrimitive](
                        "user_defined_primitive", pb.UserDefinedPrimitive(name="eager", output=0, body=0)
                    )
                )
            ],
        )
    ).publish()
    for target in (i64.__add__, RuntimeFunction(owner.ref("declarations", 0))):
        with pytest.raises(ValueError, match="table-backed"):
            graph.function_values(target)
        with pytest.raises(ValueError, match="table-backed"):
            graph.function_size(target)
    assert requests == []


def test_scope_destroy_rejection_preserves_ownership_until_success(monkeypatch):
    calls = []

    class InertTransport:
        reject = True

        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def clone_graph(self, encoded):
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded):
            request = pb.DestroyEGraphRequest.from_binary(encoded)
            calls.append(request.egraph_id)
            if self.reject and request.egraph_id == 2:
                msg = "Destroy rejection"
                raise ValueError(msg)
            return pb.DestroyEGraphResponse().to_binary()

    transport = InertTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport)
    graph = EGraph()
    saved = graph._state
    graph.push()
    active = graph._state
    with pytest.raises(ValueError, match="Destroy rejection"):
        graph.pop()
    assert graph._state is active
    assert graph._state_stack == [saved]
    assert active.handle == 2
    assert saved.handle == 1
    transport.reject = False
    graph.pop()
    assert graph._state is saved
    assert not graph._state_stack
    assert cast("object", active.handle) is None
    active.destroy()
    assert calls == [2, 2]
    with pytest.raises(IndexError, match="without a pushed state"):
        graph.pop()
    saved.destroy()
    assert calls == [2, 2, 1]


def test_scope_destroy_corrupt_success_resumes_saved_without_double_disposal(monkeypatch):
    destroyed = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def clone_graph(self, encoded):
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded):
            handle = pb.DestroyEGraphRequest.from_binary(encoded).egraph_id
            destroyed.append(handle)
            return b"\x80" if handle == 2 else pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    saved = graph._state
    graph.push()
    active = graph._state
    with pytest.raises(ValueError, match="invalid varint"):
        graph.pop()
    assert graph._state is saved
    assert not graph._state_stack
    assert active.handle is None
    assert saved.handle == 1
    active.destroy()
    assert destroyed == [2]
    saved.destroy()
    assert destroyed == [2, 1]


@pytest.mark.parametrize("control", ["sequence", "bounded", "unbounded"])
def test_schedule_additions_are_ordered_once_before_control_loops(monkeypatch, control):
    from egglog import rule, ruleset, run, seq  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Schedule assembly accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    first = ruleset(rule(name="first").then(), name="same")
    last = ruleset(rule(name="last").then(), name="same")
    schedule = seq(run(first), run(last), run(first))
    if control == "bounded":
        schedule = schedule * 3
    elif control == "unbounded":
        schedule = schedule.saturate()
    resolved = pack(commands=[schedule._resolve(None)]).program
    assert len(resolved.rulesets) == 1
    binding = resolved.rulesets[0]
    assert binding.name == str(first.__egg_ident__)
    assert binding.value is not None
    assert [rule.name for rule in binding.value.rules] == ["first", "last"]
    pending = list(reversed(resolved.commands))
    selections = []
    while pending:
        command = pending.pop()
        assert command.kind is not None
        if command.kind.field == "loop":
            pending.extend(reversed(command.kind.value.body))
        elif command.kind.field == "run":
            selections.append([
                selection.kind.value for selection in command.kind.value.rulesets if selection.kind is not None
            ])
    assert selections == [[binding.name]] * 3


@pytest.mark.parametrize(
    ("operation", "malformed"),
    [(operation, malformed) for operation in ("run", "stats") for malformed in ("location", "kind", "missing")]
    + [("run", "flags")],
)
def test_public_run_and_stats_require_the_requested_top_level_output(monkeypatch, operation, malformed):
    from egglog import seq  # noqa: PLC0415

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            output = pb.CommandOutput(
                location=pb.CommandLocation(path=[1] if malformed == "location" else [0]),
                kind=Oneof[Literal["diagnostic"], pb.Diagnostic](
                    "diagnostic", pb.Diagnostic(format="application/json", content=b"{}")
                )
                if operation == "stats"
                else Oneof[Literal["run"], pb.RunResult]("run", pb.RunResult(can_stop=True)),
            )
            if malformed == "kind":
                output.kind = (
                    Oneof[Literal["run"], pb.RunResult]("run", pb.RunResult())
                    if operation == "stats"
                    else Oneof[Literal["diagnostic"], pb.Diagnostic]("diagnostic", pb.Diagnostic())
                )
            elif malformed == "missing":
                output.kind = None
            elif malformed == "flags":
                output.kind = Oneof[Literal["run"], pb.RunResult]("run", pb.RunResult(updated=True, can_stop=True))
            return output_frames(pb.RunProgramOutput(outputs=[output])), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(ValueError, match=r"Run|Stats|stopping"):
        graph.run(seq()) if operation == "run" else graph.stats()


def _run_ruleset_step(graph, selected=None):
    """Inspect backend match counts only in explicit cursor/identity diagnostics."""
    from egglog import run  # noqa: PLC0415

    _, diagnostic = graph.run_with_diagnostics(run(selected))
    document: object = json.loads(diagnostic.content)
    assert isinstance(document, dict)
    matches = document["num_matches_per_rule"]
    assert isinstance(matches, dict)
    counts: list[int] = []
    for count in matches.values():
        assert isinstance(count, int)
        if count:
            counts.append(count)
    return sorted(counts)


def test_mixed_registration_is_one_ordered_input_program(monkeypatch):
    from egglog import rule  # noqa: PLC0415

    requests = []

    class Transport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return output_frames(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", Transport)
    graph = EGraph()
    graph.register(i64(1), rule().then(), i64(2))
    assert len(requests) == 1
    program = requests[0].program
    assert program is not None
    assert [command.kind.field for command in program.commands if command.kind is not None] == [
        "action",
        "action",
    ]
    assert len(program.rulesets) == 1
    assert program.rulesets[0].name == ""
    assert program.rulesets[0].value is not None
    assert len(program.rulesets[0].value.rules) == 1


def test_default_registration_accumulates_complete_definitions_with_clone_scope(monkeypatch):
    from egglog import rule  # noqa: PLC0415

    requests: list[pb.RunProgramRequest] = []

    class Transport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return output_frames(), None

        def clone_graph(self, encoded):
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", Transport)
    graph = EGraph()
    graph.register(rule(name="first").then())
    graph.push()
    graph.register(rule(name="child").then())
    graph.pop()
    graph.register(rule(name="parent").then())
    graph.register(i64(1))
    names = []
    for request in requests[:3]:
        assert request.program is not None
        assert not request.program.commands
        assert len(request.program.rulesets) == 1
        binding = request.program.rulesets[0]
        assert binding.name == ""
        assert binding.value is not None
        names.append([member.name for member in binding.value.rules])
    assert names == [["first"], ["first", "child"], ["first", "parent"]]
    assert requests[-1].program is not None
    assert not requests[-1].program.rulesets


@pytest.mark.parametrize("reuse", ["separate", "same", "copy"])
def test_native_public_union_occurrence_boundary(tmp_path, reuse):
    from copy import copy  # noqa: PLC0415

    from egglog import union  # noqa: PLC0415

    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    lhs, rhs = UnionNode(1), UnionNode(2)  # type: ignore[arg-type]  # Retain runtime integer-to-i64 coercion.
    graph.register(lhs, rhs)
    first = union(lhs).with_(rhs)
    second = union(lhs).with_(rhs) if reuse == "separate" else first if reuse == "same" else copy(first)
    transport = graph._state.transport
    requests: list[pb.RunProgramRequest] = []
    responses: list[list[pb.RunProgramResponse]] = []

    class AuditedTransport:
        def run(self, encoded):
            index = len(requests)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned, host_error = transport.run(encoded)
            responses.append(_record_frames(returned, tmp_path / f"response-{index}"))
            return returned, host_error

        def destroy(self, encoded):
            return transport.destroy(encoded)

    # The audit proxy intentionally substitutes for the concrete native transport.
    graph._state.transport = AuditedTransport()  # type: ignore[assignment]
    graph.register(first, second)
    assert all(frame.result is not None and frame.result.field == "output" for frame in responses[0])
    graph.check(eq(lhs).to(rhs))
    submitted_program = requests[0].program
    assert submitted_program is not None
    assert len(submitted_program.commands) == 2
    roots = []
    for command in submitted_program.commands:
        command_kind = command.kind
        assert command_kind is not None
        assert command_kind.field == "action"
        action_kind = command_kind.value.kind
        assert action_kind is not None
        assert action_kind.field == "term"
        roots.append(action_kind.value)
    assert (roots[0] == roots[1]) is (reuse != "separate")
    root_kinds = []
    for root_index in roots:
        root_kind = submitted_program.nodes[root_index].kind
        assert root_kind is not None
        root_kinds.append(root_kind)
    assert all(kind.field == "union" for kind in root_kinds)


def test_ruleset_append_preserves_native_cursor(tmp_path):
    """Real encoded execution, with explicitly requested backend counts."""
    from egglog import rule, ruleset, var  # noqa: PLC0415

    class Input(Expr):
        def __init__(self, value: i64) -> None: ...

    class Old(Expr):
        def __init__(self, value: i64) -> None: ...

    class New(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    transport = graph._state.transport
    responses: list[list[pb.RunProgramResponse]] = []

    class AuditedTransport:
        def run(self, encoded):
            index = len(responses)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            decoded = pb.RunProgramRequest.from_binary(encoded)
            assert pb.RunProgramRequest.from_binary(decoded.to_binary()) == decoded
            returned, host_error = transport.run(encoded)
            response = _record_frames(returned, tmp_path / f"response-{index}")
            responses.append(response)
            return returned, host_error

        def destroy(self, encoded):
            return transport.destroy(encoded)

    graph._state.transport = AuditedTransport()  # type: ignore[assignment]  # Native-transport audit proxy.
    graph.register(Input(1))  # type: ignore[arg-type]  # Retain runtime integer-to-i64 coercion.
    x = var("x", i64)
    selected = ruleset(rule(Input(x), name="old").then(Old(x)), name="append_cursor")

    assert _run_ruleset_step(graph, selected) == [1]
    graph.check(Old(1))  # type: ignore[arg-type]  # Retain runtime integer-to-i64 coercion.
    selected.append(rule(Input(x), name="new").then(New(x)))
    assert _run_ruleset_step(graph, selected) == [1]
    graph.check(Old(1), New(1))  # type: ignore[arg-type]  # Retain runtime integer-to-i64 coercion.
    assert _run_ruleset_step(graph, selected) == []


@pytest.mark.parametrize("seeded", [False, True])
def test_native_class_default_rewrite_and_unregistered_closure(tmp_path, seeded):
    """Default authoring reaches real bytes; public report projection is separate."""
    from types import SimpleNamespace  # noqa: PLC0415

    from egglog import ruleset  # noqa: PLC0415

    class Result(Expr, egg_sort=f"NativeDefaultResult_{seeded}"):
        def __init__(self, value: i64) -> None: ...

    selected = ruleset()
    callbacks = []

    class Source(Expr, ruleset=selected):
        def __init__(self, value: i64) -> None: ...

        def wrapped(self, value: i64) -> Result:
            callbacks.append(1)
            return Result(value)

    expression = Source(1).wrapped(2)  # type: ignore[arg-type]
    assert callbacks == [1]
    graph = EGraph()
    transport = graph._state.transport
    requests: list[pb.RunProgramRequest] = []
    responses: list[list[pb.RunProgramResponse]] = []

    def audited_run(encoded):
        index = len(requests)
        request = pb.RunProgramRequest.from_binary(encoded)
        assert request.to_binary() == encoded
        requests.append(request)
        (tmp_path / f"request-{index}.pb").write_bytes(encoded)
        returned, host_error = transport.run(encoded)
        response = _record_frames(returned, tmp_path / f"response-{index}")
        responses.append(response)
        return returned, host_error

    graph._state.transport = SimpleNamespace(run=audited_run, destroy=transport.destroy)  # type: ignore[assignment]
    if seeded:
        graph.register(expression)
    matches = _run_ruleset_step(graph, selected)
    assert matches == ([1] if seeded else [])
    if seeded:
        graph.check(eq(expression).to(Result(2)))  # type: ignore[arg-type]
    else:
        assert _run_ruleset_step(graph, selected) == []
    assert callbacks == [1]
    declared = set()
    for request in requests:
        assert request.program is not None
        for declaration in request.program.declarations:
            declaration_kind = declaration.kind
            assert declaration_kind is not None
            declared.add(declaration_kind.value.name)
    assert resolve_callable(Source.wrapped).read().kind.value.name in declared
    assert resolve_callable(Result).read().kind.value.name in declared


@pytest.mark.parametrize(("seeded", "subsume_body"), [(False, False), (True, False), (True, True)])
def test_native_free_ruleset_body_is_deferred_with_complete_closure(table_transport, seeded, subsume_body):
    from egglog import expr_parts, ruleset  # noqa: PLC0415

    class FreeDefaultBox(Expr, egg_sort=f"FreeDefaultBox_{seeded}_{subsume_body}"):
        def __init__(self, value: i64) -> None: ...

    destination = ruleset()
    callbacks = []

    @function(ruleset=destination, subsume=subsume_body)
    def delayed(value: i64) -> FreeDefaultBox:
        callbacks.append(1)
        return FreeDefaultBox(value)

    expression = delayed(i64(4))
    graph, requests, responses = table_transport
    if seeded:
        graph.register(expression)
        graph.check_fail(eq(expression).to(FreeDefaultBox(i64(4))))
        assert not graph.run(ruleset()).updated
        graph.check_fail(eq(expression).to(FreeDefaultBox(i64(4))))
    assert graph.run(destination).updated is seeded
    if seeded:
        graph.check(eq(expression).to(FreeDefaultBox(i64(4))))
        if subsume_body:
            assert expr_parts(graph.extract(expression)) == expr_parts(FreeDefaultBox(i64(4)))
    else:
        assert not graph.run(destination).updated
    assert len(graph.function_values(delayed)) == int(seeded)
    output = _required_data(responses[-1], "table_rows").outputs[0]
    table = _required_oneof(output.kind, "table_rows", pb.TableRows)
    assert [row.subsumed for row in table.rows] == ([subsume_body] if seeded else [])
    declarations = {
        declaration.kind.value.name: declaration for request in requests for declaration in request.program.declarations
    }
    name = resolve_callable(delayed).read().kind.value.name
    assert declarations[name].kind.field == "constructor"
    assert resolve_callable(FreeDefaultBox).read().kind.value.name in declarations
    authored = [rule for request in requests for rule in _program_rules(request.program)]
    assert authored
    for rule in authored:
        kind = rule.kind
        assert kind is not None
        assert kind.field == "rewrite"
        assert kind.value.subsume is subsume_body
    assert callbacks == [1]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_birewrite_runs_each_guarded_direction(table_transport, reverse, enabled):
    from egglog import birewrite, ruleset  # noqa: PLC0415

    class Box(Expr):
        pass

    selected = ruleset()
    left, right = constant("left", Box), constant("right", Box)
    guard = constant("guard", i64)
    birewrite(left, selected).to(right, eq(guard).to(i64(5)))
    seed, target = (right, left) if reverse else (left, right)
    graph, requests, _responses = table_transport
    graph.register(seed, set_(guard).to(i64(5 if enabled else 6)))
    assert graph.function_size(seed) == 1
    assert graph.function_size(target) == 0
    graph.check_fail(eq(seed).to(target))
    report = graph.run(selected)
    assert report.updated is enabled
    assert graph.function_size(target) == int(enabled)
    if enabled:
        graph.check(eq(seed).to(target))
    else:
        graph.check_fail(eq(seed).to(target))
    encoded_rules = [rule for request in requests for rule in _program_rules(request.program)]
    assert encoded_rules
    assert all(rule.kind is not None and rule.kind.field == "birewrite" for rule in encoded_rules)
    assert all(len(_program_rules(request.program)) <= 1 for request in requests)


@pytest.fixture
def native_rule_program():
    from egglog import rule, var  # noqa: PLC0415

    class Seed(Expr):
        def __init__(self, value: i64) -> None: ...

    class First(Expr):
        def __init__(self, value: i64) -> None: ...

    class Second(Expr):
        def __init__(self, value: i64) -> None: ...

    x = var("x", i64)
    first = rule(Seed(x), name="first").then(First(x))
    second = rule(Seed(x), name="second").then(Second(x))
    return Seed, First, Second, first, second


def test_native_named_replacement_preserves_effects_and_matching_progress(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    r1, r2 = ruleset(first, name="same"), ruleset(second, name="same")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, r1) == [1]
    assert _run_ruleset_step(graph, r2) == [1]
    assert _run_ruleset_step(graph, r1) == []
    graph.register(Seed(2))
    assert _run_ruleset_step(graph, r1) == [1]
    graph.check(First(1), Second(1), First(2))
    graph.check_fail(Second(2))
    assert _run_ruleset_step(graph, r2) == [1]
    graph.check(Second(2))
    graph.register(Seed(3))
    assert _run_ruleset_step(graph, ruleset(name="same")) == []
    graph.check_fail(First(3), Second(3))
    assert _run_ruleset_step(graph, r1) == [1]
    graph.check(First(3))
    graph.check_fail(Second(3))


@pytest.mark.parametrize("reverse", [False, True])
def test_native_combined_same_group_adds_both_bodies_in_order(native_rule_program, reverse):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    r1, r2 = ruleset(first, name="same"), ruleset(second, name="same")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, r2 | r1 if reverse else r1 | r2) == [1, 1]
    graph.check(First(1), Second(1))


def test_native_direct_rule_and_named_object_compose_as_one_step(native_rule_program):
    from egglog import ruleset, run  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    named = ruleset(second, name="named-composition")
    graph = EGraph(Seed(1))
    graph.run(run(ruleset(first) | named))
    graph.check(First(1), Second(1))


def test_native_combined_materializes_current_child_additions(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    left = ruleset(first, name="left")
    combined = left | ruleset(name="empty")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, combined) == [1]
    left.append(second)
    assert _run_ruleset_step(graph, combined) == [1]
    graph.check(Second(1))
    assert _run_ruleset_step(graph, left) == []
    graph.register(Seed(2))
    assert _run_ruleset_step(graph, combined) == [1, 1]
    graph.check(First(2), Second(2))


def test_native_content_shared_across_groups_keeps_one_cursor(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, _, _, first, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))
    r1 = ruleset(first, first, name="independent_one")
    r2 = ruleset(first, name="independent_two")
    assert _run_ruleset_step(graph, r1) == [1]
    assert _run_ruleset_step(graph, r2) == []
    assert _run_ruleset_step(graph, r1) == []


@pytest.mark.parametrize("prefix_rule", [False, True])
def test_native_mixed_registration_rejects_invalid_program_before_any_command(native_rule_program, prefix_rule):
    from egglog import rule, var  # noqa: PLC0415

    Seed, First, Second, first, _ = native_rule_program
    bad = rule(Seed(var("x", i64)), name="unbound-head").then(Second(var("unbound_y", i64)))
    graph = EGraph(record_history=True)
    if prefix_rule:
        graph.register(Seed(1))
    before = graph.history()
    with pytest.raises(EggSmolError) as caught:
        graph.register(first if prefix_rule else Seed(1), bad, Second(2))
    assert caught.value.__dict__["_egglog_proto_error"].location is None
    assert graph.history() == before
    assert graph.check_bool(Seed(1)) is prefix_rule
    graph.check_fail(First(1))
    graph.check_fail(Second(2))
    assert _run_ruleset_step(graph) == []


@pytest.mark.parametrize("prefix_rule", [False, True])
def test_native_mixed_registration_keeps_runtime_failure_prefix(native_rule_program, prefix_rule):
    from egglog import panic  # noqa: PLC0415

    Seed, First, Second, first, _ = native_rule_program
    graph = EGraph()
    if prefix_rule:
        graph.register(Seed(1))
    with pytest.raises(EggSmolError, match="mixed runtime marker") as caught:
        graph.register(first if prefix_rule else Seed(1), panic("mixed runtime marker"), Second(2))
    assert caught.value.__dict__["_egglog_proto_error"].location == pb.CommandLocation(path=[0 if prefix_rule else 1])
    graph.check(Seed(1))
    graph.check_fail(First(1))
    graph.check_fail(Second(2))
    assert _run_ruleset_step(graph) == ([1] if prefix_rule else [])
    assert graph.check_bool(First(1)) is prefix_rule
    graph.check_fail(Second(2))


def test_native_valid_mixed_registration_and_scope_memberships(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    graph = EGraph()
    graph.register(Seed(1), first, Second(1))
    assert _run_ruleset_step(graph) == [1]
    graph.check(First(1), Second(1))
    graph.push()
    graph.register(second)
    assert _run_ruleset_step(graph) == [1]
    graph.pop()
    graph.register(Seed(2))
    assert _run_ruleset_step(graph) == [1]
    graph.check(First(2))
    graph.check_fail(Second(2))
    explicit = ruleset(second, name="explicit")
    assert _run_ruleset_step(graph, explicit) == [2]


def test_native_repeated_registration_is_idempotent_and_labels_do_not_collide(native_rule_program):
    from egglog import rule, ruleset, var  # noqa: PLC0415

    Seed, First, Second, first, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1), first, first)
    assert _run_ruleset_step(graph) == [1]
    graph.register(first)
    assert _run_ruleset_step(graph) == []
    x = var("x", i64)
    relabeled = rule(Seed(x), name="another label").then(First(x))
    graph.register(relabeled)
    assert _run_ruleset_step(graph) == []
    different = rule(Seed(x), name="first").then(Second(x))
    graph.register(different)
    assert _run_ruleset_step(graph) == [1]
    graph.check(First(1), Second(1))
    group = ruleset(first, relabeled, name="shared-content")
    assert _run_ruleset_step(graph, group) == []


def test_native_publication_clone_before_and_after_append(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    selected = ruleset(first, name="cloned")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, selected) == [1]
    graph.push()
    selected.append(second)
    assert _run_ruleset_step(graph, selected) == [1]
    graph.pop()
    graph.check(First(1))
    graph.check_fail(Second(1))
    assert _run_ruleset_step(graph, selected) == [1]
    graph.push()
    assert _run_ruleset_step(graph, selected) == []
    graph.pop()
    assert _run_ruleset_step(graph, selected) == []


def test_native_addition_keeps_independent_compatible_declaration_owners():
    from egglog import rule, ruleset  # noqa: PLC0415

    left = constant("publication_owned_target", i64)
    right = constant("publication_owned_target", i64)
    assert left.__egg_ref__.owner is not right.__egg_ref__.owner
    selected = ruleset(rule(name="first").then(set_(left).to(1)), name="owners")  # type: ignore[arg-type]
    graph = EGraph()
    assert _run_ruleset_step(graph, selected) == [1]
    selected.append(rule(name="second").then(set_(right).to(1)))  # type: ignore[arg-type]
    assert _run_ruleset_step(graph, selected) == []
    graph.check(eq(left).to(1), eq(right).to(1))  # type: ignore[arg-type]
    graph.check(eq(left).to(1))  # type: ignore[arg-type]
    graph.check(eq(right).to(1))  # type: ignore[arg-type]


def test_native_registration_materializes_generator_before_execution(native_rule_program):
    from egglog import rule  # noqa: PLC0415

    Seed, First, _, _, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))

    def broken(x: i64):
        yield rule(Seed(x)).then(First(x))
        msg = "authoring failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="authoring failed"):
        graph.register(broken)
    graph.check_fail(First(1))


def test_check_fail_authors_one_canonical_nested_command(monkeypatch):
    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return output_frames(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    graph.check_fail(eq(i64(1)).to(i64(2)))
    program = requests[0].program
    assert program is not None
    assert len(program.commands) == 1
    fail_kind = program.commands[0].kind
    assert fail_kind is not None
    assert fail_kind.field == "fail"
    check = fail_kind.value
    check_kind = check.kind
    assert check_kind is not None
    assert check_kind.field == "check"
    assert len(check_kind.value.facts) == 1
    fact = program.nodes[check_kind.value.facts[0]]
    fact_kind = fact.kind
    assert fact_kind is not None
    assert fact_kind.field == "union"
    member_values = []
    for node_index in fact_kind.value.members:
        member_kind = program.nodes[node_index].kind
        assert member_kind is not None
        assert member_kind.field == "primitive_value"
        member_value = member_kind.value.value
        assert member_value is not None
        assert member_value.field == "i64"
        member_values.append(member_value.value)
    assert member_values == [1, 2]
    graph.check_fail()
    empty_program = requests[1].program
    assert empty_program is not None
    empty_fail = empty_program.commands[0].kind
    assert empty_fail is not None
    assert empty_fail.field == "fail"
    empty_check = empty_fail.value.kind
    assert empty_check is not None
    assert empty_check.field == "check"
    assert empty_check.value.facts == []


@pytest.mark.parametrize("shape", ["empty", "other", "multiple"])
def test_failed_check_policy_requires_one_top_level_check(monkeypatch, shape):
    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            pytest.fail("Invalid failed-check policy must not execute a request")

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    commands = []
    if shape == "other":
        commands.append(pb.Command(kind=Oneof[Literal["stats"], pb.Unit]("stats", pb.Unit())))
    elif shape == "multiple":
        commands.extend(pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check())) for _ in range(2))
    owner = Builder.from_program(pb.Program(commands=commands)).publish()
    with pytest.raises(ValueError, match="exactly one top-level Check"):
        graph._state.run_program(
            [owner.ref("commands", index) for index in range(len(commands))], allow_failed_check=True
        )


@pytest.mark.parametrize("operation", ["check", "check_bool", "check_fail"])
@pytest.mark.parametrize(
    ("code", "location"),
    [
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.CHECK_FAILED, None),
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[1])),
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[0], iterations=[0])),
        (pb.ErrorCode.RUNTIME_FAILURE, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.INVALID_PROGRAM, None),
    ],
)
def test_check_errors_use_codes_and_locations_not_messages(monkeypatch, operation, code, location):
    record = pb.Error(code=code, message="Check failed: same diagnostic", location=location)

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            return [pb.RunProgramResponse(result=Oneof[Literal["error"], pb.Error]("error", record)).to_binary()], None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    method = getattr(graph, operation)
    if (
        operation == "check_bool"
        and code == pb.ErrorCode.CHECK_FAILED
        and location is not None
        and location.path == [0]
        and not location.iterations
    ):
        assert method(eq(i64(1)).to(i64(1))) is False
    else:
        with pytest.raises(EggSmolError) as caught:
            method(eq(i64(1)).to(i64(1)))
        assert type(caught.value) is EggSmolError
        assert caught.value.__dict__["_egglog_proto_error"] == record
        assert str(caught.value) == record.message
        assert caught.value.args == (record.message,)
        if operation == "check_bool":
            traceback = caught.value.__traceback__
            frames = []
            while traceback is not None:
                frames.append(traceback.tb_frame.f_code.co_name)
                traceback = traceback.tb_next
            assert frames.count("check_bool") == 1
            assert frames.count("run_program") == 1


@pytest.mark.parametrize("located", [False, True])
def test_execution_error_retains_only_confirmed_declaration_refs(monkeypatch, located):
    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            return [
                pb.RunProgramResponse(
                    result=Oneof[Literal["error"], pb.Error](
                        "error",
                        pb.Error(
                            code=pb.ErrorCode.RUNTIME_FAILURE,
                            message="installation or command failed",
                            location=pb.CommandLocation(path=[0]) if located else None,
                        ),
                    ),
                ).to_binary()
            ], None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    current = constant("failed_request_definition", i64)
    graph = EGraph()
    action = set_(current).to(i64(1))
    from egglog._program import _definition_key  # noqa: PLC0415

    expected = {
        _definition_key(reference.read()): reference
        for reference in pack(commands=[action.__egg_ref__]).definitions
        if reference.role == "declarations"
    }
    before = graph._state.definitions.copy()
    with pytest.raises(EggSmolError):
        graph.register(action)
    assert graph._state.definitions == (before | expected if located else before)


@pytest.mark.parametrize("error_type", [ValueError, EggSmolError])
@pytest.mark.parametrize("misleading_record", [False, True])
def test_check_bool_preserves_transport_exception_and_traceback(monkeypatch, error_type, misleading_record):
    sentinel = error_type("Check failed is only text")
    if misleading_record:
        sentinel.__dict__["_egglog_proto_error"] = {"code": pb.ErrorCode.CHECK_FAILED}

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            raise sentinel

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(error_type) as caught:
        graph.check_bool(eq(i64(1)).to(i64(1)))
    assert caught.value is sentinel
    traceback = caught.value.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    assert traceback.tb_frame.f_code is InertTransport.run.__code__


def test_constant_actions_submit_one_ordered_binary_program(monkeypatch):
    requests = []

    class InertTransport:
        def create(self, encoded):
            pb.CreateEGraphRequest.from_binary(encoded)
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return output_frames(), None

        def destroy(self, encoded):
            assert pb.DestroyEGraphRequest.from_binary(encoded).egraph_id == 1
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport, raising=False)
    current = constant("current", i64)
    later = constant("later", i64)
    graph = EGraph()
    graph.register(set_(current).to(i64(1)), current, set_(later).to(current + 2), later)
    assert len(requests) == 1
    program = requests[0].program
    assert program is not None
    action_kinds = []
    for command in program.commands:
        command_kind = command.kind
        assert command_kind is not None
        assert command_kind.field == "action"
        action_kind = command_kind.value.kind
        assert action_kind is not None
        action_kinds.append(action_kind)
    assert [kind.field for kind in action_kinds] == ["set", "term", "set", "term"]
    first_kind, first_term, second_kind, second_term = action_kinds
    assert first_kind.field == "set"
    assert first_term.field == "term"
    assert second_kind.field == "set"
    assert second_term.field == "term"
    first = first_kind.value
    second = second_kind.value
    assert first.target is not None
    assert second.target is not None
    first_call = program.nodes[first_term.value].kind
    assert first_call is not None
    assert first_call.field == "call"
    assert first_call.value.func == first.target.func
    second_call = program.nodes[second_term.value].kind
    assert second_call is not None
    assert second_call.field == "call"
    assert second_call.value.func == second.target.func
    definitions = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        definitions[declaration_kind.value.name] = declaration
    first_definition = definitions[first.target.func].kind
    assert first_definition is not None
    assert first_definition.field == "function"
    assert presentation(resolve_callable(later)).views[0].kind == CallKind.CONSTANT
    second_definition = definitions[second.target.func].kind
    assert second_definition is not None
    assert second_definition.value.name == resolve_callable(later).read().kind.value.name
    rhs_kind = program.nodes[second.value].kind
    assert rhs_kind is not None
    assert rhs_kind.field == "call"
    rhs = rhs_kind.value
    assert rhs.func == "egglog.core.i64.add"
    rhs_argument = program.nodes[rhs.args[0]].kind
    assert rhs_argument is not None
    assert rhs_argument.field == "call"
    assert rhs_argument.value.func == first.target.func
    condition = eq(i64(1)).to(i64(1))
    original = condition.__egg_ref__.read().to_binary()
    graph.register(set_(current).to(i64(1)), condition)
    assert len(requests) == 2
    program = requests[1].program
    assert program is not None
    action_kinds = []
    for command in program.commands:
        command_kind = command.kind
        assert command_kind is not None
        assert command_kind.field == "action"
        action_kind = command_kind.value.kind
        assert action_kind is not None
        action_kinds.append(action_kind)
    assert [kind.field for kind in action_kinds] == ["set", "term"]
    _, term_kind = action_kinds
    assert term_kind.field == "term"
    union = program.nodes[term_kind.value]
    union_kind = union.kind
    assert union_kind is not None
    assert union_kind.field == "union"
    member_values = []
    for node_index in union_kind.value.members:
        member_kind = program.nodes[node_index].kind
        assert member_kind is not None
        assert member_kind.field == "primitive_value"
        member_value = member_kind.value.value
        assert member_value is not None
        assert member_value.field == "i64"
        member_values.append(member_value.value)
    assert member_values == [1, 1]
    assert condition.__egg_ref__.read().to_binary() == original
    graph._state.destroy()


def test_response_owner_keeps_nominal_and_ambient_family_contexts(monkeypatch):
    family = builtin_catalog().definitions["primitive_sort", "Pair"]

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert any(sort.kind == Oneof("eq", "Pair") for sort in request.program.sorts)
            return output_frames(
                pb.RunProgramOutput(
                    sorts=[
                        pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Pair")),
                        pb.Sort(
                            kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef](
                                "primitive", pb.PrimitiveSortRef(name="i64")
                            )
                        ),
                    ],
                    nodes=[
                        pb.Node(sort_id=0, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="P"))),
                        pb.Node(
                            sort_id=1,
                            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                            ),
                        ),
                    ],
                    outputs=[
                        pb.CommandOutput(
                            kind=Oneof[Literal["extraction"], pb.ExtractResult](
                                "extraction",
                                pb.ExtractResult(roots=[pb.ExtractedRoot(variants=[pb.ExtractedTerm(term=0, cost=1)])]),
                            ),
                            location=pb.CommandLocation(path=[0]),
                        )
                    ],
                )
            ), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport, raising=False)

    class NominalPair(Expr, egg_sort="Pair"):
        @method(egg_fn="P")
        def __init__(self) -> None: ...

        @method(preserve=True)
        def tag(self):
            return "nominal hook"

    graph = EGraph()
    result = graph.extract(NominalPair())
    assert graph._state.definitions["primitive_sort", "Pair"] == family
    # The authored class is a RuntimeClass presentation with canonical declaration metadata.
    assert graph._state.definitions["eq_sort", "Pair"] == NominalPair.__egg_definition__  # type: ignore[attr-defined]
    assert result.tag() == "nominal hook"
    assert str(result) == "NominalPair()"
    graph._state.destroy()
    assert result.tag() == "nominal hook"


def test_public_bytes_requests_and_response_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[pb.RunProgramRequest] = []
    destroyed: list[int] = []

    class InertTransport:
        def create(self, encoded: bytes) -> bytes:
            assert isinstance(encoded, bytes)
            request = pb.CreateEGraphRequest.from_binary(encoded)
            assert request.has_field("threads")
            assert request.threads == 1
            assert request.options is not None
            cost_kind = request.sorts[request.options.cost_sort].kind
            assert cost_kind is not None
            assert cost_kind.field == "primitive"
            assert cost_kind.value.name == "i64"
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def run(self, encoded: bytes) -> tuple[list[bytes], BaseException | None]:
            assert isinstance(encoded, bytes)
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            # Fixed inert reply: no interpretation or computation of the request.
            return output_frames(
                pb.RunProgramOutput(
                    sorts=[
                        pb.Sort(
                            kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef](
                                "primitive", pb.PrimitiveSortRef(name="i64")
                            )
                        )
                    ],
                    nodes=[
                        pb.Node(
                            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                                "primitive_value",
                                pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 31)),
                            )
                        )
                    ],
                    outputs=[
                        pb.CommandOutput(
                            kind=Oneof[Literal["extraction"], pb.ExtractResult](
                                "extraction",
                                pb.ExtractResult(roots=[pb.ExtractedRoot(variants=[pb.ExtractedTerm(term=0, cost=0)])]),
                            ),
                            location=pb.CommandLocation(path=[0]),
                        )
                    ],
                )
            ), None

        def clone_graph(self, encoded: bytes) -> bytes:
            assert pb.CloneEGraphRequest.from_binary(encoded).egraph_id == 1
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            destroyed.append(pb.DestroyEGraphRequest.from_binary(encoded).egraph_id)
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport, raising=False)
    graph = EGraph()
    result, cost = graph.extract(i64(1) + 2, include_cost=True)
    assert result.value == cost == 31
    request = requests[0]
    assert request.program is not None
    program = request.program
    command = program.commands[0].kind
    assert command is not None
    assert command.field == "extract"
    call = program.nodes[command.value.roots[0]].kind
    assert call is not None
    assert call.field == "call"
    assert call.value.func == "egglog.core.i64.add"
    argument_values = []
    for node_index in call.value.args:
        argument_kind = program.nodes[node_index].kind
        assert argument_kind is not None
        assert argument_kind.field == "primitive_value"
        argument_value = argument_kind.value.value
        assert argument_value is not None
        assert argument_value.field == "i64"
        argument_values.append(argument_value.value)
    assert argument_values == [1, 2]
    declarations: dict[str, pb.Declaration] = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        declarations.setdefault(declaration_kind.value.name, declaration)
    assert not declarations
    assert presentation(builtin_catalog().definitions["callable", "egglog.core.i64.add"]).views[0].path == ("__add__",)
    local_type = presentation(builtin_catalog().definitions["primitive_sort", "i64"]).type_view
    assert local_type is not None
    assert local_type.path == ("egglog", "builtins", "i64")
    # Re-authoring from a returned value retains the original catalog refs,
    # not a second declaration copy reconstructed from the response.
    repacked = pack([(result + 4).__egg_ref__])
    assert not repacked.definitions
    assert builtin_catalog().owner in result.__egg_ref__.owner._environment
    with graph:
        assert graph._state.handle == 2
    assert graph._state.handle == 1
    assert destroyed == [2]
    graph.close()
    assert graph._state.handle == 1
    graph._state.destroy()
    graph._state.destroy()
    assert destroyed == [2, 1]


@pytest.mark.parametrize(
    "body_error_type", [None, ValueError, BaseException], ids=["no-body-error", "exception", "base-exception"]
)
def test_context_keeps_active_and_saved_state_when_destroy_rejected(
    monkeypatch: pytest.MonkeyPatch, body_error_type: type[BaseException] | None
) -> None:
    class BodyError(BaseException):
        pass

    if body_error_type is BaseException:
        body_error_type = BodyError

    class FaultTransport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=1, threads=1).to_binary()

        def clone_graph(self, encoded: bytes) -> bytes:
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            assert pb.DestroyEGraphRequest.from_binary(encoded).egraph_id == 2
            msg = "cleanup sentinel"
            raise RuntimeError(msg)

    monkeypatch.setattr(bindings, "_ProtoEngine", FaultTransport, raising=False)
    graph = EGraph()
    saved = graph._state
    body_error = None if body_error_type is None else body_error_type("body sentinel")
    with (  # noqa: PT012 -- exercise body and disposal together
        pytest.raises(
            RuntimeError if body_error_type is None else body_error_type,
            match="cleanup sentinel" if body_error is None else "body sentinel",
        ) as caught,
        graph,
    ):
        assert graph._state is not saved
        if body_error is not None:
            raise body_error
    if body_error is not None:
        assert caught.value is body_error
    assert graph._state is not saved
    assert graph._state.handle == 2
    assert saved.handle == 1
    assert graph._state_stack == [saved]


class ResourceTransport:
    def __init__(self) -> None:
        self.creations: list[pb.CreateEGraphRequest] = []
        self.actual_threads = 7

    def create(self, encoded: bytes) -> bytes:
        request = pb.CreateEGraphRequest.from_binary(encoded)
        self.creations.append(request)
        return pb.CreateEGraphResponse(egraph_id=1, threads=self.actual_threads).to_binary()

    def clone_graph(self, encoded: bytes) -> bytes:
        return pb.CloneEGraphResponse(egraph_id=2).to_binary()

    def destroy(self, encoded: bytes) -> bytes:
        return pb.DestroyEGraphResponse().to_binary()


@pytest.fixture
def resource_transport(monkeypatch: pytest.MonkeyPatch) -> ResourceTransport:
    transport = ResourceTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport)
    return transport


@pytest.mark.parametrize("seminaive", [False, True])
@pytest.mark.parametrize("no_decomp", [False, True])
def test_compilation_flags_and_effective_threads_are_creation_only(resource_transport, seminaive, no_decomp):
    graph = EGraph(seminaive=seminaive, no_decomp=no_decomp, num_threads=0)
    request = resource_transport.creations[0]
    assert request.options is not None
    assert request.options.seminaive is seminaive
    assert request.no_decomp is no_decomp
    assert request.threads == 0
    assert graph.num_threads() == 7
    assert graph.no_decomp() is no_decomp
    graph.push()
    assert graph.num_threads() == 7
    assert graph.no_decomp() is no_decomp
    graph.pop()
    assert len(resource_transport.creations) == 1
    for removed in ("set_num_threads", "set_no_decomp", "set_report_level"):
        assert not hasattr(graph, removed)


def test_naive_source_export_rejects_after_creation(resource_transport):
    graph = EGraph(save_egglog_string=True, seminaive=False)
    assert len(resource_transport.creations) == 1
    assert resource_transport.creations[0].record_history
    with pytest.raises(NotImplementedError, match="initial seminaive=False"):
        _ = graph.as_egglog_string


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_invalid_compilation_flags_have_no_transport_effect(resource_transport, value):
    for keyword in ("seminaive", "no_decomp"):
        with pytest.raises(TypeError, match="booleans"):
            EGraph(**{keyword: value})
    assert not resource_transport.creations


class IndexedThreadCount:
    def __index__(self) -> int:
        return 2


class IntOnlyThreadCount:
    def __int__(self) -> int:
        return 2


@pytest.mark.parametrize("value", [IndexedThreadCount(), True, False, (1 << 32) - 1])
def test_thread_count_accepts_index_protocol(resource_transport: ResourceTransport, value: SupportsIndex) -> None:
    graph = EGraph(num_threads=value)
    assert resource_transport.creations[0].threads == index(value)
    assert graph.num_threads() == resource_transport.actual_threads


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, OverflowError),
        (1 << 32, OverflowError),
        (1 << 64, OverflowError),
        (None, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (IntOnlyThreadCount(), TypeError),
    ],
)
def test_invalid_thread_count_has_no_transport_effect(resource_transport, value, error):
    with pytest.raises(error):
        EGraph(num_threads=value)
    assert not resource_transport.creations


def test_creation_requires_positive_effective_thread_count(resource_transport):
    resource_transport.actual_threads = 0
    with pytest.raises(ValueError, match="actual thread count"):
        EGraph()
