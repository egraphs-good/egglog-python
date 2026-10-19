# mypy: allow-empty-bodies
# Ellipsis bodies are DSL declarations or annotation-introspection fixtures.
"""Record/bytes plumbing tests; the inert responder is not engine conformance."""

from operator import index
from typing import Literal, Never, Self

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import BaseExpr, EGraph, Expr, PrimitiveFailure, bindings, constant, eq, function, i64, method, set_
from egglog._catalog import builtin_catalog
from egglog._program import Builder, pack
from egglog.egraph_state import EggSmolError
from egglog.runtime import RuntimeFunction, create_callable, resolve_callable

# Test-only observation of the exception actually raised inside Python. Functions
# pickle by module reference; this is not an engine error slot or value registry.
_callback_observations = {}


@pytest.mark.parametrize("outcome", ["success", "preflight", "transport", "decode", "receipt", "engine", "host"])
def test_source_recording_commits_only_valid_acknowledgements(monkeypatch, outcome):  # noqa: C901
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    events = []
    original = ValueError("original recording boundary")
    success = pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary()
    responses = {
        "decode": (b"\x80", None),
        "receipt": (pb.RunProgramResponse().to_binary(), None),
        "engine": (
            pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(),
                error=pb.Error(code=pb.ErrorCode.EVALUATION_FAILED, message="native recording failure"),
            ).to_binary(),
            None,
        ),
        "host": (success, original),
    }

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def enable_recording(self):
            events.append("enable")

        def prepare_recording(self, encoded):
            program = pb.Program.from_binary(encoded)
            assert len(program.commands) == 1
            events.append("prepare")
            if outcome == "preflight":
                raise original

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.egraph_id == 1
            events.append("run")
            if outcome == "transport":
                raise original
            return responses.get(outcome, (success, None))

        def finish_recording(self, success):
            events.append(("finish", success))

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    state = EGraphState(save_egglog_string=True)
    builder = Builder()
    root = builder.import_ref(i64(1).__egg_ref__)
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", root))
            )
        ),
    )
    expected = {
        "preflight": ValueError,
        "transport": ValueError,
        "decode": ValueError,
        "receipt": ValueError,
        "engine": EggSmolError,
        "host": ValueError,
    }
    if outcome == "success":
        state.run_program([builder.publish().ref("commands", command)])
    else:
        with pytest.raises(expected[outcome]) as caught:
            state.run_program([builder.publish().ref("commands", command)])
        if outcome in {"preflight", "transport", "host"}:
            assert caught.value is original
        if outcome == "engine":
            assert caught.value.__dict__["_egglog_proto_error"].code == pb.ErrorCode.EVALUATION_FAILED
    assert events == (
        ["enable", "prepare"]
        if outcome == "preflight"
        else ["enable", "prepare", "run", ("finish", outcome == "success")]
    )


def test_source_recording_does_not_exempt_freeze_with_installation(monkeypatch):
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    prepared = []

    class ClosedTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def enable_recording(self):
            pass

        def prepare_recording(self, encoded):
            program = pb.Program.from_binary(encoded)
            assert program.commands[0].kind is not None
            assert program.commands[0].kind.field == "freeze"
            assert len(program.declarations) == 1
            prepared.append(program)
            msg = "transcript has been closed"
            raise ValueError(msg)

        def run(self, encoded):
            pytest.fail("Freeze with installation bypassed closed recording preflight")

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", ClosedTransport)
    state = EGraphState(save_egglog_string=True)
    builder = Builder()
    declaration = builder.add(
        "declarations",
        pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="not-installed"))),
    )
    command = builder.add("commands", pb.Command(kind=Oneof[Literal["freeze"], pb.Freeze]("freeze", pb.Freeze())))
    owner = builder.publish()
    with pytest.raises(ValueError, match="has been closed"):
        state.run_program([owner.ref("commands", command)], roots=[owner.ref("declarations", declaration)])
    assert len(prepared) == 1


def test_source_producer_returns_only_program_bytes_and_scope_counts():
    source = bindings._ProtoEngine.source(
        "(function source_scoped () i64 :merge new) (push 1) (set (source_scoped) 7) (pop 1)"
    )
    kinds = []
    while (submission := source.next_submission()) is not None:
        kind, encoded, count = submission
        kinds.append(kind)
        if kind == "program":
            assert type(encoded) is bytes
            program = pb.Program.from_binary(encoded)
            assert program.ir_version == 1
            assert program.to_binary() == encoded
            assert count == 0
        else:
            assert kind in {"push", "pop"}
            assert encoded is None
            assert count == 1
    assert kinds == ["program", "push", "program", "pop"]


def test_source_producer_later_error_preserves_executed_byte_prefix():
    graph = EGraph()
    source = bindings._ProtoEngine.source(
        "(function source_prefix () i64 :merge new) (set (source_prefix) 7) (missing_source_command)"
    )
    try:
        for _ in range(2):
            kind, encoded, count = source.next_submission()
            assert (kind, count) == ("program", 0)
            assert type(encoded) is bytes
            response, host_error = graph._state.transport.run(
                pb.RunProgramRequest(
                    egraph_id=graph._state.handle, program=pb.Program.from_binary(encoded), profile=True
                ).to_binary()
            )
            decoded = pb.RunProgramResponse.from_binary(response)
            assert host_error is None
            assert decoded.error is None
            assert decoded.ruleset_publication is not None
        with pytest.raises(ValueError, match="missing_source_command"):
            source.next_submission()

        verification = bindings._ProtoEngine.source(
            "(function source_prefix () i64 :merge new) (check (= (source_prefix) 7))"
        )
        while (submission := verification.next_submission()) is not None:
            kind, encoded, count = submission
            assert (kind, count) == ("program", 0)
            response, host_error = graph._state.transport.run(
                pb.RunProgramRequest(
                    egraph_id=graph._state.handle, program=pb.Program.from_binary(encoded), profile=True
                ).to_binary()
            )
            assert host_error is None
            assert pb.RunProgramResponse.from_binary(response).error is None
    finally:
        graph._state.destroy()


@pytest.fixture
def scheduler_transport(monkeypatch):  # noqa: C901 -- finite acknowledgement/failure matrix, not engine conformance
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    class InertTransport:
        outcome = "success"
        original = ValueError("original scheduler deployment failure")

        def __init__(self) -> None:
            self.requests = []
            self.next_handle = 1

        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=self.next_handle).to_binary()

        def enable_recording(self):
            pass

        def prepare_recording(self, encoded):
            command_kind = pb.Program.from_binary(encoded).commands[0].kind
            assert command_kind is not None
            assert command_kind.field == "bind_scheduler"
            if self.outcome == "preflight":
                raise self.original

        def finish_recording(self, success):
            pass

        def prepare_recording_scope(self, push):
            pass

        def run(self, encoded):
            self.requests.append(pb.RunProgramRequest.from_binary(encoded))
            if self.outcome == "transport":
                raise self.original
            if self.outcome == "decode":
                return b"\x80", None
            if self.outcome == "receipt":
                return pb.RunProgramResponse().to_binary(), None
            response = pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication())
            if self.outcome == "engine":
                response.error = pb.Error(code=pb.ErrorCode.EVALUATION_FAILED, message="scheduler native failure")
            return response.to_binary(), self.original if self.outcome == "host" else None

        def clone_graph(self, encoded):
            self.next_handle += 1
            return pb.CloneEGraphResponse(egraph_id=self.next_handle).to_binary()

        def return_to(self, encoded):
            return pb.ReturnToEGraphResponse().to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    transport = InertTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport)
    return EGraphState(save_egglog_string=True), transport


@pytest.mark.parametrize("outcome", ["success", "preflight", "transport", "decode", "receipt", "engine", "host"])
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
        assert not request.include_run_reports
        assert state._scheduler_bindings[name] == reference.read().kind.value
        for options in (pb.BackOff(match_limit=1, ban_length=0), pb.BackOff(ban_length=0)):
            conflicting = reference.read()
            conflicting.kind.value.scheduler = pb.Scheduler(kind=Oneof("back_off", options))
            builder = Builder()
            index = builder.add("commands", conflicting)
            with pytest.raises(ValueError, match="different configuration"):
                state.bind_scheduler(builder.publish().ref("commands", index))
        assert len(transport.requests) == 1
        return
    expected = EggSmolError if outcome == "engine" else ValueError
    with pytest.raises(expected) as caught:
        state.bind_scheduler(reference)
    if outcome in {"preflight", "transport", "host"}:
        assert caught.value is transport.original
    if outcome == "engine":
        assert caught.value.__dict__["_egglog_proto_error"].code == pb.ErrorCode.EVALUATION_FAILED
    transport.outcome = "success"
    if outcome == "preflight":
        assert name not in state._scheduler_bindings
        assert not transport.requests
        state.bind_scheduler(reference)
        assert len(transport.requests) == 1
    else:
        assert state._scheduler_bindings[name] is None
        with pytest.raises(RuntimeError, match="unacknowledged deployment"):
            state.bind_scheduler(reference)
        assert len(transport.requests) == 1


def test_persistent_scheduler_deployment_follows_clone_and_return(scheduler_transport):
    from egglog import back_off  # noqa: PLC0415

    saved, transport = scheduler_transport
    parent_binding = back_off(match_limit=2).persistent().__egg_ref__
    saved.bind_scheduler(parent_binding)
    active = saved.copy()
    active.bind_scheduler(parent_binding)
    assert len(transport.requests) == 1
    child_binding = back_off(ban_length=0).persistent().__egg_ref__
    active.bind_scheduler(child_binding)
    active.return_to(saved)
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
        ir_version=1,
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="PyObject"))),
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="bool"))),
        ],
    )

    def unused(_observation: bytes) -> object:
        pytest.fail("Rejected model contract must not invoke callbacks")

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
        assert error.code == (pb.ErrorCode.UNKNOWN_NAME if case == "unknown" else pb.ErrorCode.INVALID_PROGRAM)
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
    cases = {
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
    owner = Builder.from_program(snapshot.program).publish()
    restored = EGraph()
    restored._state.run_program(
        pack(
            roots=[owner.ref("declarations", index) for index in range(len(snapshot.program.declarations))],
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
    assert roundtrip.options == snapshot.options
    action_counts = []
    for frozen in (roundtrip, snapshot):
        frozen_program = frozen.program
        assert frozen_program is not None
        action_fields = []
        for command in frozen_program.commands:
            command_kind = command.kind
            assert command_kind is not None
            assert command_kind.field == "action"
            action_kind = command_kind.value.kind
            assert action_kind is not None
            action_fields.append(action_kind.field)
        action_counts.append(Counter(action_fields))
    assert action_counts[0] == action_counts[1]


def _observed_pyerr_callback(tag):
    observation = _callback_observations[tag]
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

    observations = {}
    monkeypatch.setattr(sys.modules[__name__], "_callback_observations", observations)
    return observations


def _callback_program(tag, callback=_observed_pyerr_callback):
    """Author a closed call using the actual configured registrar, not fake signatures."""
    import cloudpickle  # noqa: PLC0415

    # The generation-only exporter is intentionally absent from the public stub.
    program = pb.Program.from_binary(getattr(bindings, "_export_pyobject_catalog")())  # noqa: B009
    sort = next(
        index
        for index, record in enumerate(program.sorts)
        if record.kind.field == "family" and record.kind.value == pb.HostSort(name="PyObject")
    )
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
    encoded = pb.RunProgramRequest(egraph_id=state.handle, program=program, profile=True).to_binary()
    assert pb.RunProgramRequest.from_binary(encoded).to_binary() == encoded
    path.with_suffix(".request.pb").write_bytes(encoded)
    response, error = state.transport.run(encoded)
    assert isinstance(response, bytes)
    path.with_suffix(".response.pb").write_bytes(response)
    decoded = pb.RunProgramResponse.from_binary(response)
    assert pb.RunProgramResponse.from_binary(decoded.to_binary()) == decoded
    assert decoded.ruleset_publication is not None
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
    for constructor in (PrimitiveFailure, NoInit, SuperNew):
        for args in ((), (marker,), (marker, "message", 17)):
            error = constructor(*args)
            assert type(error) is constructor
            assert error.args == args
            if args:
                assert error.args[0] is marker
        with pytest.raises(TypeError):
            constructor(message="not a positional argument")


def test_callback_python_none_is_a_value(callback_observations, tmp_path):
    import cloudpickle  # noqa: PLC0415

    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    callback_observations["none"] = {"error": None, "result": None, "calls": 0}
    state = EGraphState()
    try:
        response, error = _callback_exchange(state, _callback_program("none")[0], tmp_path / "none")
        assert error is None
        assert response.error is None
        assert callback_observations["none"]["calls"] > 0
        term = response.outputs[0].kind.value.roots[0].variants[0].term
        value = response.nodes[term].kind.value.value
        assert value.field == "custom"
        assert value.value.args == []
        assert cloudpickle.loads(value.value.payload) is None
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
    error.detail = {"owned": [1, 2]}
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
    program.rulesets.append(
        pb.Ruleset(name="host_error_published", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList()))
    )
    source = Builder.from_program(program).publish()
    published = source.ref("rulesets", len(program.rulesets) - 1)
    packed = pack(commands=[source.ref("commands", index) for index in range(2)], roots=[published])
    original_adopt = Builder.from_program.__func__
    adopted = []

    def adopt(cls, returned):
        assert graph._state.definitions["ruleset", "host_error_published"] == published
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
    response = responses[0]
    assert response.error.code == pb.ErrorCode.EVALUATION_FAILED
    assert response.error.location.path == [1]
    assert len(response.outputs) == 1
    assert response.outputs[0].location.path == [0]
    assert adopted[0].nodes == response.nodes
    assert adopted[0].sorts == response.sorts
    term = response.outputs[0].kind.value.roots[0].variants[0].term
    custom = response.nodes[term].kind.value.value
    assert custom.field == "custom"
    assert custom.value.payload == program.nodes[first + 1].kind.value.value.value.payload
    assert custom.value.args == []


@pytest.mark.parametrize("failure", ["binary", "receipt", "arena"])
def test_callback_error_protocol_failure_keeps_original_chain(monkeypatch, failure):
    error = ValueError("original host error")
    cause, context = KeyError("cause"), LookupError("context")
    error.__cause__, error.__context__, error.__suppress_context__ = cause, context, True
    cause.__context__, context.__cause__ = context, error
    response = pb.RunProgramResponse(
        ruleset_publication=None if failure == "receipt" else pb.RulesetPublication()
    ).to_binary()

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            pb.RunProgramRequest.from_binary(encoded)
            return (b"\xff" if failure == "binary" else response), error

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
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "check"
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary(), original

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
        assert recovered.error is None
        assert no_error is None
    finally:
        if second is not first:
            second.destroy()
        first.destroy()
    for index in (1, 0):
        response, error = pairs[index]
        assert response.error.code == pb.ErrorCode.EVALUATION_FAILED
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
                assert response.error.code == pb.ErrorCode.EVALUATION_FAILED
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
            assert response.error.code == pb.ErrorCode.EVALUATION_FAILED
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
        original.annotation_calls = 0
    callback_observations["first"] = {"error": original, "calls": 0}
    callback_observations["later"] = {"error": None, "calls": 0}
    program, first = _callback_program(
        "first", _UnprintableCallback() if case.endswith("subclass") else _observed_pyerr_callback
    )
    pyobject = program.nodes[first].sort_id
    integer, unit = (
        next(index for index, sort in enumerate(program.sorts) if sort.kind == Oneof("family", pb.HostSort(name=name)))
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
    for name, call in (("handoff_first", query_call), ("handoff_later", later_call)):
        query = len(program.nodes)
        program.nodes.append(
            pb.Node(sort_id=pyobject, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[y, call])))
        )
        rule = len(program.rules)
        program.rules.append(
            pb.RuleDecl(
                name=name,
                eval_mode=pb.RuleEvalMode.SEMINAIVE,
                kind=Oneof[Literal["rule"], pb.Rule](
                    "rule",
                    pb.Rule(
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
        program.rulesets.append(
            pb.Ruleset(name=name, kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList(rules=[rule])))
        )
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
        )
    ]
    state = EGraphState(num_threads=threads)
    try:
        setup, error = _callback_exchange(state, program, tmp_path / "setup")
        assert setup.error is None
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
                    else Oneof(
                        "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", "handoff_first")))
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
                    else Oneof(
                        "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", "handoff_later")))
                    )
                )
            )
        )
        if expected_failure and not extracting:
            # Observe the first rule's no-match before the later successful callback writes its head.
            program.commands.insert(
                3,
                pb.Command(
                    kind=Oneof[Literal["print_function"], pb.PrintFunction](
                        "print_function", pb.PrintFunction(table="handoff_mark")
                    )
                ),
            )
        response, captured = _callback_exchange(state, program, tmp_path / "execution")
        assert captured is (None if expected_failure else original)
        assert (response.error is not None) == (extracting or case == "later_native_error")
        if expected_failure:
            assert callback_observations["first"]["caught"] is original
            assert original.annotation_calls == 0
            assert not hasattr(original, "__notes__")
            assert not callback_observations.get("formatted", False)
            if extracting:
                assert callback_observations["later"]["calls"] == 0
            else:
                assert callback_observations["later"]["calls"] > 0
                (printed,) = [output for output in response.outputs if output.kind.field == "printed_function"]
                assert printed.location.path == [3]
                assert len(printed.kind.value.rows) == 2
        elif original is not None:
            assert captured is callback_observations["first"]["caught"]
            assert callback_observations["later"]["calls"] == 0
        if case == "later_native_error":
            assert response.error.location.path == [3]
            assert "native after callback" in response.error.message
        elif extracting:
            assert response.error.code == pb.ErrorCode.EVALUATION_FAILED
            assert response.error.location.path == [1]
        program.commands = [
            pb.Command(
                kind=Oneof[Literal["print_function"], pb.PrintFunction](
                    "print_function", pb.PrintFunction(table="handoff_mark")
                )
            )
        ]
        observed, error = _callback_exchange(state, program, tmp_path / "markers")
        assert observed.error is None
        assert error is None
        rows = {}
        for row in observed.outputs[0].kind.value.rows:
            values = []
            for cell in (*row.args, row.output):
                assert cell.kind.field == "term"
                node = observed.nodes[cell.kind.value]
                assert node.kind.value.value.field == "i64"
                values.append(node.kind.value.value.value)
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
        assert recovered.error is None
        assert error is None
        term = recovered.outputs[0].kind.value.roots[0].variants[0].term
        payload = recovered.nodes[term].kind.value.value
        assert payload.field == "custom"
        assert payload.value.args == []
        assert cloudpickle.loads(payload.value.payload) == 8
    finally:
        state.destroy()


@pytest.fixture
def table_transport(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Table inspection used the legacy engine")

    graph = EGraph()
    transport = graph._state.transport
    requests: list[pb.RunProgramRequest] = []
    responses: list[pb.RunProgramResponse] = []

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
            assert isinstance(returned, bytes)
            (tmp_path / f"response-{position}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
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
    assert responses[0].outputs == []
    assert responses[0].nodes == []
    assert responses[0].error is not None
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
    assert requests[0].program.commands[0].kind == Oneof("print_size", pb.PrintSize(tables=["z-known"]))
    assert responses[0].outputs[0].location == pb.CommandLocation(path=[0])
    assert responses[0].outputs[0].kind == Oneof("sizes", pb.TableSizes(sizes=[pb.TableSize(table="z-known", rows=0)]))
    graph.register(set_(known(i64(1))).to(i64(2)), set_(unmapped(i64(1))).to(i64(3)))
    # The legacy public API filtered native tables lacking a Python mapping.
    # Keep the native row, but remove only its canonical language lookup entry.
    del graph._state.definitions["callable", "a-unmapped"]
    assert graph.all_function_sizes() == [(known, 1)]
    assert requests[-1].program.declarations == []
    assert requests[-1].program.commands[0].kind == Oneof("print_size", pb.PrintSize())
    assert [entry.table for entry in responses[-1].outputs[0].kind.value.sizes] == ["a-unmapped", "z-known"]


def test_function_sizes_include_unavailable_and_subsumed_native_bytes(table_transport):
    graph, _, _ = table_transport
    program = pb.Program(
        ir_version=1,
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "CountedHidden")),
        ],
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="CountedHidden"))),
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
    target = RuntimeFunction(owner.ref("declarations", 1))
    assert graph.function_size(target) == 2
    assert graph.all_function_sizes() == [(target, 2)]
    with pytest.raises(ValueError, match="has no extractable representative"):
        graph.function_values(target)


def test_table_callable_projection_keeps_first_view_without_invocation(monkeypatch):
    owner = Builder.from_program(
        pb.Program(
            ir_version=1,
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            declarations=[
                pb.Declaration(
                    kind=Oneof[Literal["function"], pb.Function](
                        "function", pb.Function(name="two-views", inputs=[pb.Arg(sort=0)], output=0)
                    ),
                    bindings=pb.CallableBindings(
                        python=pb.PythonBindings(
                            views=[
                                pb.PythonCallable(
                                    kind=pb.PythonCallKind.FUNCTION,
                                    path=[name],
                                    params=[pb.PythonParameter(name="value", core_input=0)],
                                )
                                for name in ("first", "second")
                            ]
                        )
                    ),
                )
            ],
        )
    ).publish()

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
@pytest.mark.parametrize("corruption", ["outputs", "kind", "location", "run report", "table", "duplicate"])
def test_function_sizes_reject_malformed_response_without_engine(monkeypatch, selected, corruption):  # noqa: C901
    @function(egg_fn="size-table")
    def table(value: i64) -> i64: ...

    sizes = pb.TableSizes(sizes=[pb.TableSize(table="size-table", rows=1)])
    output = pb.CommandOutput(
        kind=Oneof[Literal["sizes"], pb.TableSizes]("sizes", sizes),
        location=pb.CommandLocation(path=[0]),
    )
    response = pb.RunProgramResponse(outputs=[output], ruleset_publication=pb.RulesetPublication())
    match corruption:
        case "outputs":
            response.outputs.clear()
        case "kind":
            output.kind = Oneof[Literal["printed_function"], pb.PrintedFunction](
                "printed_function", pb.PrintedFunction()
            )
        case "location":
            output.location = pb.CommandLocation(path=[1])
        case "run report":
            output.run_report = pb.NativeRunReport()
        case "table":
            sizes.sizes[0].table = ""
        case "duplicate":
            sizes.sizes.append(pb.TableSize(table="size-table", rows=2))

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "print_size"
            return response.to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(ValueError, match="table size"):
        graph.function_size(table) if selected else graph.all_function_sizes()


def test_function_values_native_bytes_and_shared_terms(table_transport):
    graph, requests, responses = table_transport

    @function
    def repeated(left: i64, right: i64) -> i64: ...

    graph.register(set_(repeated(i64(1), i64(1))).to(i64(1)), set_(repeated(i64(2), i64(3))).to(i64(7)))
    rows = graph.function_values(repeated)
    assert len(requests) == 2
    command = requests[-1].program.commands[0].kind
    assert command.field == "print_function"
    assert command.value.table == resolve_callable(repeated).read().kind.value.name
    assert not command.value.has_field("max_rows")
    output = responses[-1].outputs[0]
    assert output.location == pb.CommandLocation(path=[0])
    assert output.kind.field == "printed_function"
    row = output.kind.value.rows[0]
    assert row.args[0] == row.args[1] == row.output == pb.TableCell(kind=Oneof[Literal["term"], int]("term", 0))
    assert responses[-1].nodes[0].kind.value.value == Oneof("i64", 1)
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
            assert responses[0].error is None
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
    assert responses[0].error is None
    assert graph.function_values(untouched) == {}


@pytest.mark.parametrize("unavailable", ["argument 1", "output"])
def test_function_values_unavailable_cells_raise_from_native_bytes(table_transport, unavailable):
    graph, requests, responses = table_transport
    # Canonical backend fixture: public unextractable authoring stays guarded.
    sorts = [
        pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
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
        ir_version=1,
        sorts=sorts,
        nodes=nodes,
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Hidden"))),
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
    target = RuntimeFunction(owner.ref("declarations", 3))
    with pytest.raises(ValueError, match=f"Table 'inspected' row 1 {unavailable} has no extractable representative"):
        graph.function_values(target)
    table = responses[-1].outputs[0].kind.value
    assert responses[-1].error is None
    assert len(table.rows) == 2
    cell = table.rows[1].args[1] if unavailable == "argument 1" else table.rows[1].output
    assert cell.kind.field == "unavailable_sort"
    assert responses[-1].sorts[cell.kind.value].kind == Oneof("eq", "Hidden")
    assert requests[-1].program.commands[0].kind.field == "print_function"
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
        kind=Oneof[Literal["printed_function"], pb.PrintedFunction](
            "printed_function", pb.PrintedFunction(table="table", rows=[row])
        ),
        location=pb.CommandLocation(path=[0]),
    )
    response = pb.RunProgramResponse(
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=name)))
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
        ruleset_publication=pb.RulesetPublication(),
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
            assert output.kind.field == "printed_function"
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
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "print_function"
            return response.to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    with pytest.raises(ValueError, match=r"table|Table"):
        EGraph().function_values(table)


def test_function_values_rejects_primitives_without_table_request(table_transport):
    graph, requests, _ = table_transport
    owner = Builder.from_program(
        pb.Program(
            ir_version=1,
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
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
                    kind=Oneof[Literal["primitive"], pb.Primitive](
                        "primitive", pb.Primitive(name="eager", output=0, body=0)
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


def test_scope_return_consumption_and_rejection_without_engine():
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    calls = []

    class InertTransport:
        reject = True

        def return_to(self, encoded):
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            calls.append((request.active_id, request.saved_id))
            if self.reject:
                msg = "Return rejection"
                raise ValueError(msg)
            return pb.ReturnToEGraphResponse().to_binary()

    active, saved = object.__new__(EGraphState), object.__new__(EGraphState)
    active.handle, saved.handle = 2, 1
    active.transport = saved.transport = InertTransport()
    with pytest.raises(ValueError, match="Return rejection"):
        active.return_to(saved)
    assert (active.handle, saved.handle) == (2, 1)
    active.transport.reject = False
    active.return_to(saved)
    assert (active.handle, saved.handle) == (None, 1)
    with pytest.raises(RuntimeError, match="live"):
        active.return_to(saved)
    assert calls == [(2, 1), (2, 1)]
    active.handle = 2
    original = active.transport
    active.transport = InertTransport()
    with pytest.raises(ValueError, match="same transport"):
        active.return_to(saved)
    active.transport = original
    active.handle = 1
    with pytest.raises(ValueError, match="distinct"):
        active.return_to(saved)
    active.handle = None
    saved.handle = None


def test_scope_return_corrupt_success_consumes_active_without_selecting_saved(monkeypatch):
    destroyed = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def clone_graph(self, encoded):
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def return_to(self, encoded):
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            assert (request.active_id, request.saved_id) == (2, 1)
            return b"\x80"  # Successful transport, then a truncated acknowledgement.

        def destroy(self, encoded):
            destroyed.append(pb.DestroyEGraphRequest.from_binary(encoded).egraph_id)
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    saved = graph._state
    graph.push()
    active = graph._state
    with pytest.raises(ValueError, match="invalid varint"):
        graph.pop()
    assert graph._state is active
    assert graph._state_stack == [saved]
    assert active.handle is None
    assert saved.handle == 1
    active.destroy()
    assert destroyed == []  # Never Destroy a handle already consumed by Return.


def test_schedule_publication_and_report_requests_are_pure_and_ordered(monkeypatch):
    from egglog import rule, ruleset, run, seq  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415
    from egglog.egraph import set_current_ruleset  # noqa: PLC0415

    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            program = request.program
            assert program is not None
            outputs = []
            if program.commands:
                assert len(program.commands) == 1
                command = program.commands[0].kind
                report = pb.NativeRunReport(can_stop=True)
                assert command is not None
                if command.field == "stats":
                    assert not request.include_run_reports
                    outputs = [
                        pb.CommandOutput(
                            kind=Oneof[Literal["statistics"], pb.NativeRunReport]("statistics", report),
                            location=pb.CommandLocation(path=[0]),
                        )
                    ]
                else:
                    assert request.include_run_reports
                    outputs = [
                        pb.CommandOutput(
                            kind=Oneof("run", pb.RunOutcome(can_stop=True))
                            if command.field == "run"
                            else Oneof[Literal["loop"], pb.LoopOutcome](
                                "loop", pb.LoopOutcome(iterations=1, termination=pb.TerminationReason.COUNT_REACHED)
                            ),
                            run_report=report,
                            location=pb.CommandLocation(path=[0]),
                        )
                    ]
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(names=[r.name for r in program.rulesets]),
                outputs=outputs,
            ).to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

        def configure_resources(self, encoded):
            request = pb.ConfigureEGraphResourcesRequest.from_binary(encoded)
            requests.append(request)
            return pb.ConfigureEGraphResourcesResponse().to_binary()

    def forbidden(*args, **kwargs):
        pytest.fail("Schedule request assembly accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    first = ruleset(rule(name="first").then(), name="same")
    last = ruleset(rule(name="last").then(), name="same")
    a, b = run(first), run(last)
    schedule = seq(a, b, a)
    assert "first" in str(schedule)
    assert graph.run(schedule).can_stop
    installed = graph._state.rulesets[first.__egg_ident__]
    assert [ref.read().name for ref in graph._state._ruleset_members(installed)] == ["first"]
    execution = requests[-1]
    execution_program = execution.program
    assert execution_program is not None
    execution_kind = execution_program.commands[0].kind
    assert execution_kind is not None
    assert execution_kind.field == "repeat"
    assert len(execution_kind.value.body) == 3
    assert not any(request.program.commands for request in requests[:-1])
    before = len(requests)
    graph.run(schedule)
    assert len(requests) == before + 1  # Same members/cursors, no fresh publication.
    first.append(rule(name="append").then())
    graph.run(schedule)
    assert [ref.read().name for ref in graph._state._ruleset_members(graph._state.rulesets[first.__egg_ident__])] == [
        "first",
        "append",
    ]
    graph.run(seq(run()))
    assert Ident("") in graph._state.rulesets
    assert graph.stats().can_stop
    stats_program = requests[-1].program
    assert stats_program is not None
    stats_kind = stats_program.commands[0].kind
    assert stats_kind is not None
    assert stats_kind.field == "stats"
    for length in (-1, 2**64):
        before = len(requests)
        with pytest.raises(OverflowError):
            graph.run(run() * length)
        assert len(requests) == before  # No publication precedes unsigned-count rejection.
    other = ruleset(rule(name="other").then(), name="other")
    combined = first | other
    graph.run(combined)
    first.append(rule(name="deferred-append").then())
    graph.run(combined)
    installed = graph._state.rulesets[combined.__egg_ident__]
    assert [ref.read().name for ref in graph._state._ruleset_members(installed)] == ["first", "append", "other"]
    graph.run(first)
    graph.run(combined)
    installed = graph._state.rulesets[combined.__egg_ident__]
    assert [ref.read().name for ref in graph._state._ruleset_members(installed)] == [
        "first",
        "append",
        "deferred-append",
        "other",
    ]
    with set_current_ruleset(other):
        graph.run(run())
    selected_program = requests[-1].program
    assert selected_program is not None
    run_kind = selected_program.commands[0].kind
    assert run_kind is not None
    assert run_kind.field == "run"
    assert run_kind.value.ruleset is not None
    ruleset_kind = run_kind.value.ruleset.kind
    assert ruleset_kind is not None
    assert ruleset_kind.field == "index"
    selected = ruleset_kind.value
    assert selected_program.rulesets[selected].name == graph._state.rulesets[Ident("")].read().name
    for level in (pb.ReportLevel.TIME_ONLY, pb.ReportLevel.WITH_PLAN, pb.ReportLevel.STAGE_INFO):
        graph.set_report_level(level)
        assert requests[-1].operation.value == level


@pytest.mark.parametrize("operation", ["run", "stats"])
@pytest.mark.parametrize("malformed", ["location", "kind", "missing", "flags"])
def test_public_reports_require_the_requested_top_level_output(monkeypatch, operation, malformed):
    from egglog import seq  # noqa: PLC0415

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            report = pb.NativeRunReport(can_stop=True)
            output = pb.CommandOutput(
                location=pb.CommandLocation(path=[1] if malformed == "location" else [0]),
                kind=Oneof[Literal["statistics"], pb.NativeRunReport]("statistics", report)
                if operation == "stats"
                else Oneof[Literal["loop"], pb.LoopOutcome](
                    "loop", pb.LoopOutcome(termination=pb.TerminationReason.COUNT_REACHED)
                ),
                run_report=report if operation == "run" else None,
            )
            if malformed == "kind":
                output.kind = Oneof("run", pb.RunOutcome(can_stop=True))
            elif malformed == "missing":
                output.kind = None
            elif malformed == "flags":
                if operation == "run":
                    output.kind.value.updated = True
                else:
                    output.run_report = report
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(), outputs=[output]
            ).to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(ValueError, match=r"report|statistics"):
        graph.run(seq()) if operation == "run" else graph.stats()


def test_publication_derives_each_rule_key_once_per_call(monkeypatch):
    from collections import Counter  # noqa: PLC0415

    from egglog import _program as program  # noqa: PLC0415
    from egglog import rule, ruleset, var  # noqa: PLC0415

    x = var("x", i64)
    selected = ruleset(*(rule(eq(x).to(i64(i)), name=f"memo_{i}").then() for i in range(4)), name="memo")
    graph = EGraph()
    original = program._rule_key
    calls = Counter()

    def counted(reference):
        calls[reference] += 1
        return original(reference)

    monkeypatch.setattr(program, "_rule_key", counted)
    versions = []
    for _ in range(2):
        calls.clear()
        identifiers = {}
        snapshot = selected._snapshot(identifiers)
        versions.append(graph._state.publish_ruleset(snapshot, identifiers))
        assert calls
        assert max(calls.values()) == 1
        assert len(graph._state._ruleset_members(versions[-1])) == 4
    assert versions[0] == versions[1]
    assert len([key for key in graph._state.definitions if key[0] == "ruleset"]) == 7


def _run_ruleset_step(graph, selected=None):
    """Freeze a live target, execute canonical Run bytes, and read raw matches."""
    from egglog._program import Builder  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415

    if selected is None:
        published = graph._state.rulesets[Ident("")]
    else:
        identifiers = {}
        snapshot = selected._snapshot(identifiers)
        published = graph._state.publish_ruleset(snapshot, identifiers)
    builder = Builder()
    ruleset_index = builder.import_ref(published)
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["run"], pb.Run](
                "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", ruleset_index)))
            )
        ),
    )
    response, _ = graph._state.run_program([builder.publish().ref("commands", command)])
    assert response.error is None
    assert response.profile is not None
    assert len(response.profile.runs) == 1
    labels = {index: attribution.name for index, attribution in enumerate(response.rules)}
    matches = {}
    for summary in response.profile.runs[0].rules:
        assert summary.matches is not None
        label = labels[summary.rule]
        matches[label] = matches.get(label, 0) + summary.matches
    return matches


@pytest.mark.parametrize("completed", [False, True])
def test_publication_adopts_exact_receipt_before_install_error(monkeypatch, completed):
    from egglog import rule, ruleset  # noqa: PLC0415

    requests = []
    fail = True

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            names = [record.name for record in request.program.rulesets if record.has_field("name")]
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(names=names if completed or not fail else []),
                error=pb.Error(code=pb.ErrorCode.EVALUATION_FAILED, message="injected install boundary")
                if fail
                else None,
            ).to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    def forbidden(*args, **kwargs):
        pytest.fail("Publication bookkeeping used the legacy engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    selected = ruleset(rule(name="first").then(), name="receipt")
    identifiers = {}
    snapshot = selected._snapshot(identifiers)
    with pytest.raises(EggSmolError, match="injected install boundary"):
        graph._state.publish_ruleset(snapshot, identifiers)
    first_name = requests[0].program.rulesets[0].name
    assert (("ruleset", first_name) in graph._state.definitions) is completed
    assert (selected.__egg_ident__ in graph._state.rulesets) is completed
    if completed:
        confirmed = graph._state.rulesets[selected.__egg_ident__]
        assert graph._state._ruleset_identifiers[confirmed] == selected.__egg_ident__
    fail = False
    version = graph._state.publish_ruleset(snapshot, identifiers)
    assert version.read().name == first_name
    if completed:
        assert len(requests) == 1  # Retry must not reset the completed occurrence.
    else:
        assert len(requests) == 2
        assert requests[1].program == requests[0].program
    assert graph._state._ruleset_members(version)[0] != snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])


@pytest.mark.parametrize("receipt", [None, ["foreign"], ["requested", "requested"], []])
def test_invalid_publication_receipt_does_not_adopt_roots(monkeypatch, receipt):
    from egglog._program import Builder  # noqa: PLC0415

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            return pb.RunProgramResponse(
                ruleset_publication=None if receipt is None else pb.RulesetPublication(names=receipt)
            ).to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    builder = Builder()
    index = builder.add(
        "rulesets", pb.Ruleset(name="requested", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList()))
    )
    before = graph._state.definitions.copy()
    with pytest.raises(ValueError, match=r"receipt|confirmation"):
        graph._state.run_program([], roots=[builder.publish().ref("rulesets", index)])
    assert graph._state.definitions == before


@pytest.mark.parametrize("across_actions", [False, True])
def test_mixed_registration_cut_distinguishes_query_from_action_identity(monkeypatch, across_actions):
    from egglog import Action, Fact, expr_action, rule  # noqa: PLC0415
    from egglog._program import Builder  # noqa: PLC0415

    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(
                    names=[root.name for root in request.program.rulesets if root.has_field("name")]
                )
            ).to_binary(), None

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)

    class Box(Expr):
        def __init__(self) -> None: ...

    builder = Builder()
    union = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Box.__egg_sort__),
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[builder.import_ref(Box().__egg_ref__)])),
        ),
    )
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", union))
            )
        ),
    )
    owner = builder.publish()
    action = Action(owner.ref("commands", command))
    query = Fact(owner.ref("nodes", union))
    graph = EGraph()
    if across_actions:
        with pytest.raises(NotImplementedError, match="shared allocating Union"):
            graph.register(action, rule(query).then(), action)
        assert not requests
        assert not graph._state.rulesets
    else:
        graph.register(action, rule(query).then(), expr_action(i64(1)))
        assert [len(request.program.commands) for request in requests] == [1, 0, 1]
        assert len(requests[1].program.rules) == 1


@pytest.mark.parametrize("reuse", ["separate", "same", "copy"])
def test_native_public_union_occurrence_boundary(tmp_path, reuse):
    from copy import copy  # noqa: PLC0415

    from egglog import union  # noqa: PLC0415

    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    lhs, rhs = UnionNode(1), UnionNode(2)
    graph.register(lhs, rhs)
    first = union(lhs).with_(rhs)
    second = union(lhs).with_(rhs) if reuse == "separate" else first if reuse == "same" else copy(first)
    transport = graph._state.transport
    requests = []
    responses = []

    class AuditedTransport:
        def run(self, encoded):
            index = len(requests)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned, host_error = transport.run(encoded)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
            return returned, host_error

        def destroy(self, encoded):
            return transport.destroy(encoded)

    graph._state.transport = AuditedTransport()
    graph.register(first, second)
    assert responses[0].error is None
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
    assert all(submitted_program.nodes[index].kind.field == "union" for index in roots)


def test_ruleset_append_preserves_native_cursor(tmp_path):
    """Real encoded execution; raw summaries are not a public RunReport."""
    from egglog import rule, ruleset, var  # noqa: PLC0415

    class Input(Expr):
        def __init__(self, value: i64) -> None: ...

    class Old(Expr):
        def __init__(self, value: i64) -> None: ...

    class New(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    transport = graph._state.transport
    responses = []

    class AuditedTransport:
        def run(self, encoded):
            index = len(responses)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            decoded = pb.RunProgramRequest.from_binary(encoded)
            assert pb.RunProgramRequest.from_binary(decoded.to_binary()) == decoded
            returned, host_error = transport.run(encoded)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            response = pb.RunProgramResponse.from_binary(returned)
            assert response.ruleset_publication is not None
            responses.append(response)
            return returned, host_error

        def destroy(self, encoded):
            return transport.destroy(encoded)

    graph._state.transport = AuditedTransport()
    graph.register(Input(1))
    x = var("x", i64)
    selected = ruleset(rule(Input(x), name="old").then(Old(x)), name="append_cursor")

    assert _run_ruleset_step(graph, selected) == {"old": 1}
    graph.check(Old(1))
    selected.append(rule(Input(x), name="new").then(New(x)))
    assert _run_ruleset_step(graph, selected) == {"new": 1}
    graph.check(Old(1), New(1))
    assert _run_ruleset_step(graph, selected) == {}
    assert any(response.ruleset_publication.names for response in responses)


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

    expression = Source(1).wrapped(2)
    assert callbacks == [1]
    graph = EGraph()
    transport = graph._state.transport
    requests, responses = [], []

    def audited_run(encoded):
        index = len(requests)
        request = pb.RunProgramRequest.from_binary(encoded)
        assert request.to_binary() == encoded
        requests.append(request)
        (tmp_path / f"request-{index}.pb").write_bytes(encoded)
        returned, host_error = transport.run(encoded)
        (tmp_path / f"response-{index}.pb").write_bytes(returned)
        response = pb.RunProgramResponse.from_binary(returned)
        assert response.ruleset_publication is not None
        responses.append(response)
        return returned, host_error

    graph._state.transport = SimpleNamespace(run=audited_run, destroy=transport.destroy)
    if seeded:
        graph.register(expression)
    matches = _run_ruleset_step(graph, selected)
    assert matches == ({"": 1} if seeded else {})
    if seeded:
        graph.check(eq(expression).to(Result(2)))
    else:
        assert _run_ruleset_step(graph, selected) == {}
    assert callbacks == [1]
    assert any(response.ruleset_publication.names for response in responses)
    declared = {declaration.kind.value.name for request in requests for declaration in request.program.declarations}
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
    output = responses[-1].outputs[0]
    assert output.kind.field == "printed_function"
    assert [row.subsumed for row in output.kind.value.rows] == ([subsume_body] if seeded else [])
    declarations = {
        declaration.kind.value.name: declaration for request in requests for declaration in request.program.declarations
    }
    name = resolve_callable(delayed).read().kind.value.name
    assert declarations[name].kind.field == "constructor"
    assert resolve_callable(FreeDefaultBox).read().kind.value.name in declarations
    authored = [rule for request in requests for rule in request.program.rules]
    assert authored
    assert all(rule.kind.field == "rewrite" and rule.kind.value.subsume is subsume_body for rule in authored)
    assert callbacks == [1]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_birewrite_runs_each_guarded_direction(table_transport, reverse, enabled):
    from egglog import birewrite, ruleset  # noqa: PLC0415
    from egglog.run_report import RuleKey  # noqa: PLC0415

    class Box(Expr):
        pass

    selected = ruleset()
    left, right = constant("left", Box), constant("right", Box)
    guard = constant("guard", i64)
    authored = birewrite(left, selected).to(right, eq(guard).to(i64(5)))
    seed, target = (right, left) if reverse else (left, right)
    graph, requests, responses = table_transport
    graph.register(seed, set_(guard).to(i64(5 if enabled else 6)))
    assert graph.function_size(seed) == 1
    assert graph.function_size(target) == 0
    graph.check_fail(eq(seed).to(target))
    report = graph.run(selected)
    native_report = responses[-1].outputs[0].run_report
    assert native_report is not None
    assert report.updated is enabled
    assert graph.function_size(target) == int(enabled)
    if enabled:
        graph.check(eq(seed).to(target))
        assert list(report.search_and_apply_time_per_rule) == [RuleKey(authored.__egg_ref__)]
        assert "birewrite(" in str(report)
    else:
        graph.check_fail(eq(seed).to(target))
        assert not report.search_and_apply_time_per_rule
        assert not native_report.search_and_apply_time_per_rule
        assert not native_report.num_matches_per_rule
    encoded_rules = [rule for request in requests for rule in request.program.rules]
    assert encoded_rules
    assert all(rule.kind.field == "birewrite" for rule in encoded_rules)
    assert all(len(request.program.rules) <= 1 for request in requests)


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


def test_native_same_ident_retains_all_completed_memberships(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    r1, r2 = ruleset(first, name="same"), ruleset(second, name="same")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, r1) == {"first": 1}
    assert _run_ruleset_step(graph, r2) == {"second": 1}
    assert _run_ruleset_step(graph, r1) == {}
    graph.register(Seed(2))
    assert _run_ruleset_step(graph, r1) == {"first": 1, "second": 1}
    graph.check(First(1), Second(1), First(2), Second(2))


@pytest.mark.parametrize("reverse", [False, True])
def test_native_combined_same_ident_selects_rightmost_before_install(native_rule_program, reverse):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    r1, r2 = ruleset(first, name="same"), ruleset(second, name="same")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, r2 | r1 if reverse else r1 | r2) == {"first" if reverse else "second": 1}
    assert graph.check_bool(First(1)) is reverse
    assert graph.check_bool(Second(1)) is not reverse


def test_native_combined_uses_only_completed_child_updates(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    left = ruleset(first, name="left")
    combined = left | ruleset(name="empty")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, combined) == {"first": 1}
    left.append(second)
    assert _run_ruleset_step(graph, combined) == {}
    graph.check_fail(Second(1))
    assert _run_ruleset_step(graph, left) == {"second": 1}
    graph.register(Seed(2))
    assert _run_ruleset_step(graph, combined) == {"first": 1, "second": 1}
    graph.check(First(2), Second(2))


def test_native_distinct_ident_memberships_do_not_share_cursors(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, _, _, first, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))
    r1 = ruleset(first, first, name="independent_one")
    r2 = ruleset(first, name="independent_two")
    assert _run_ruleset_step(graph, r1) == {"first": 1}
    assert _run_ruleset_step(graph, r2) == {"first": 1}
    assert _run_ruleset_step(graph, r1) == {}
    roots = [graph._state.rulesets[r.__egg_ident__] for r in (r1, r2)]
    occurrences = [graph._state._ruleset_members(root) for root in roots]
    assert len(occurrences[0]) == len(occurrences[1]) == 1
    assert occurrences[0][0] != occurrences[1][0]


@pytest.mark.parametrize("prefix_rule", [False, True])
def test_native_mixed_registration_keeps_failure_prefix(native_rule_program, prefix_rule):
    from egglog import rule, var  # noqa: PLC0415

    Seed, First, Second, first, _ = native_rule_program
    bad = rule(Seed(var("x", i64)), name="unbound-head").then(Second(var("unbound_y", i64)))
    graph = EGraph()
    if prefix_rule:
        graph.register(Seed(1))
    with pytest.raises(EggSmolError):
        graph.register(first if prefix_rule else Seed(1), bad, Second(2))
    graph.check(Seed(1))
    graph.check_fail(First(1))
    graph.check_fail(Second(2))
    if prefix_rule:
        assert _run_ruleset_step(graph) == {"first": 1}
        graph.check(First(1))
    graph.check_fail(Second(2))


def test_native_valid_mixed_registration_and_scope_memberships(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    graph = EGraph()
    graph.register(Seed(1), first, Second(1))
    assert _run_ruleset_step(graph) == {"first": 1}
    graph.check(First(1), Second(1))
    original_direct = graph._state._direct_rules
    graph.push()
    assert graph._state._direct_rules == original_direct
    assert graph._state._direct_rules is not original_direct
    graph.register(second)
    assert len(graph._state._direct_rules) == 2
    assert len(original_direct) == 1
    assert _run_ruleset_step(graph) == {"second": 1}
    graph.pop()
    assert graph._state._direct_rules is original_direct
    graph.register(Seed(2))
    assert _run_ruleset_step(graph) == {"first": 1}
    graph.check(First(2))
    graph.check_fail(Second(2))
    explicit = ruleset(second, name="explicit")
    assert _run_ruleset_step(graph, explicit) == {"second": 2}


def test_native_direct_unnamed_registration_creates_fresh_occurrences(native_rule_program):
    from egglog import rule, var  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415

    Seed, First, _, _, _ = native_rule_program
    x = var("x", i64)
    authored = rule(Seed(x)).then(First(x))
    graph = EGraph()
    graph.register(Seed(1))
    graph.register(authored, authored)
    assert _run_ruleset_step(graph) == {"": 2}
    graph.register(authored)
    assert _run_ruleset_step(graph) == {"": 1}
    assert len(graph._state._ruleset_members(graph._state.rulesets[Ident("")])) == 3
    graph.check(First(1))


def test_native_direct_named_collision_keeps_prefix_and_error_shape(native_rule_program):
    Seed, First, Second, first, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))
    for prefix in [True, False]:
        with pytest.raises(EggSmolError) as raised:
            graph.register(first, first, Second(1)) if prefix else graph.register(first, Second(1))
        message = "At 0:0 of \nRule already exists: first"
        assert type(raised.value) is EggSmolError
        assert str(raised.value) == message
        assert raised.value.args == (message,)
        assert "_egglog_proto_error" not in raised.value.__dict__  # Frontend compatibility validation.
        assert _run_ruleset_step(graph) == ({"first": 1} if prefix else {})
        graph.check(First(1))
        graph.check_fail(Second(1))


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("direct_first", [False, True])
def test_native_direct_and_materialized_memberships(native_rule_program, default, named, direct_first):
    from egglog import rule, ruleset, var  # noqa: PLC0415

    Seed, First, _, _, _ = native_rule_program
    selected = ruleset(name="" if default else "explicit")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, selected) == {}
    authored = rule(Seed(var("x", i64)), name="first" if named else None, ruleset=selected).then(First(var("x", i64)))
    label = "first" if named else ""
    if direct_first:
        graph.register(authored)
        if named and default:
            with pytest.raises(EggSmolError, match="Rule already exists: first"):
                _run_ruleset_step(graph, selected)
            assert _run_ruleset_step(graph) == {label: 1}
        else:
            assert _run_ruleset_step(graph, selected) == {label: 2 if default else 1}
    else:
        assert _run_ruleset_step(graph, selected) == {label: 1}
        if named and default:
            with pytest.raises(EggSmolError, match="Rule already exists: first"):
                graph.register(authored)
        else:
            graph.register(authored)
            assert _run_ruleset_step(graph, selected) == ({label: 1} if default else {})
    graph.check(First(1))


def test_native_publication_clone_before_and_after_append(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    selected = ruleset(first, name="cloned")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, selected) == {"first": 1}
    graph.push()
    selected.append(second)
    assert _run_ruleset_step(graph, selected) == {"second": 1}
    graph.pop()
    graph.check(First(1))
    graph.check_fail(Second(1))
    assert _run_ruleset_step(graph, selected) == {"second": 1}
    graph.push()
    assert _run_ruleset_step(graph, selected) == {}
    graph.pop()
    assert _run_ruleset_step(graph, selected) == {}


def test_native_publication_keeps_independent_compatible_declaration_owners():
    from egglog import rule, ruleset  # noqa: PLC0415

    left = constant("publication_owned_target", i64)
    right = constant("publication_owned_target", i64)
    assert left.__egg_ref__.owner is not right.__egg_ref__.owner
    selected = ruleset(rule(name="first").then(set_(left).to(1)), name="owners")
    graph = EGraph()
    assert _run_ruleset_step(graph, selected) == {"first": 1}
    selected.append(rule(name="second").then(set_(right).to(1)))
    assert _run_ruleset_step(graph, selected) == {"second": 1}
    graph.check(eq(left).to(1), eq(right).to(1))
    graph.check(eq(left).to(1))
    graph.check(eq(right).to(1))


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
    assert not graph._state.rulesets
    graph.check_fail(First(1))


def test_check_fail_authors_one_canonical_nested_command(monkeypatch):
    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary(), None

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
    assert [program.nodes[index].kind.value.value.value for index in fact_kind.value.members] == [1, 2]
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
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

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
    owner = Builder.from_program(pb.Program(ir_version=1, commands=commands)).publish()
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
        (pb.ErrorCode.EVALUATION_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.PROOF_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.INVALID_PROGRAM, None),
    ],
)
def test_check_errors_use_codes_and_locations_not_messages(monkeypatch, operation, code, location):
    record = pb.Error(code=code, message="Check failed: same diagnostic", location=location)
    if code == pb.ErrorCode.PROOF_FAILED:
        record.proof_failure = pb.ProofFailure(reason=pb.ProofFailureReason.PROOFS_NOT_ENABLED)

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            return pb.RunProgramResponse(error=record, ruleset_publication=pb.RulesetPublication()).to_binary(), None

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
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            return pb.RunProgramResponse(
                error=pb.Error(
                    code=pb.ErrorCode.EVALUATION_FAILED,
                    message="installation or command failed",
                    location=pb.CommandLocation(path=[0]) if located else None,
                ),
                ruleset_publication=pb.RulesetPublication(),
            ).to_binary(), None

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
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

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
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary(), None

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
    second_bindings = definitions[second.target.func].bindings
    assert second_bindings is not None
    assert second_bindings.python is not None
    assert second_bindings.python.views[0].kind == pb.PythonCallKind.CONSTANT
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
    assert [program.nodes[index].kind.value.value.value for index in union_kind.value.members] == [1, 1]
    assert condition.__egg_ref__.read().to_binary() == original
    graph._state.destroy()


def test_response_owner_keeps_nominal_and_ambient_family_contexts(monkeypatch):
    family = builtin_catalog().definitions["host_sort_family", "Pair"]

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert any(d.kind.field == "eq_sort" and d.kind.value.name == "Pair" for d in request.program.declarations)
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(),
                sorts=[
                    pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Pair")),
                    pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
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
            ).to_binary(), None

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
    assert graph._state.definitions["host_sort_family", "Pair"] == family
    assert graph._state.definitions["eq_sort", "Pair"] == NominalPair.__egg_definition__
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
            assert not request.declarations
            assert request.options is not None
            assert request.sorts[request.options.cost_sort].kind.value.name == "i64"
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded: bytes) -> tuple[bytes, BaseException | None]:
            assert isinstance(encoded, bytes)
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            # Fixed inert reply: no interpretation or computation of the request.
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(),
                sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
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
            ).to_binary(), None

        def clone_graph(self, encoded: bytes) -> bytes:
            assert pb.CloneEGraphRequest.from_binary(encoded).egraph_id == 1
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def return_to(self, encoded: bytes) -> bytes:
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            assert (request.active_id, request.saved_id) == (2, 1)
            return pb.ReturnToEGraphResponse().to_binary()

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
    assert program.ir_version == 1
    command = program.commands[0].kind
    assert command is not None
    assert command.field == "extract"
    call = program.nodes[command.value.roots[0]].kind
    assert call is not None
    assert call.field == "call"
    assert call.value.func == "egglog.core.i64.add"
    assert [program.nodes[index].kind.value.value.value for index in call.value.args] == [1, 2]
    add = next(declaration for declaration in program.declarations if declaration.kind.value.name == call.value.func)
    assert add.bindings is not None
    assert add.bindings.python is not None
    assert add.bindings.python.views[0].path == ["__add__"]
    family = next(declaration for declaration in program.declarations if declaration.kind.value.name == "i64")
    assert family.kind.value.bindings.python.path == ["egglog", "builtins", "i64"]
    # Re-authoring from a returned value retains the original catalog refs,
    # not a second declaration copy reconstructed from the response.
    repacked = pack([(result + 4).__egg_ref__])
    assert builtin_catalog().definitions["callable", "egglog.core.i64.add"] in repacked.definitions
    with graph:
        assert graph._state.handle == 2
    assert graph._state.handle == 1
    assert destroyed == []
    graph.close()
    assert graph._state.handle == 1
    graph._state.destroy()
    graph._state.destroy()
    assert destroyed == [1]


@pytest.mark.parametrize(
    "body_error_type", [None, ValueError, BaseException], ids=["no-body-error", "exception", "base-exception"]
)
def test_context_keeps_active_and_saved_state_when_return_rejected(
    monkeypatch: pytest.MonkeyPatch, body_error_type: type[BaseException] | None
) -> None:
    class BodyError(BaseException):
        pass

    if body_error_type is BaseException:
        body_error_type = BodyError

    class FaultTransport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def clone_graph(self, encoded: bytes) -> bytes:
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def return_to(self, encoded: bytes) -> bytes:
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            assert (request.active_id, request.saved_id) == (2, 1)
            msg = "cleanup sentinel"
            raise RuntimeError(msg)

        def destroy(self, encoded: bytes) -> bytes:
            return pb.DestroyEGraphResponse().to_binary()

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
    """An inert bytes receiver, not a simulation of native resource allocation."""

    def __init__(self) -> None:
        self.creations: list[pb.CreateEGraphRequest] = []
        self.requests: list[pb.ConfigureEGraphResourcesRequest] = []
        self.threads: dict[int, int] = {}
        self.no_decomp: dict[int, bool] = {}
        self.error: ValueError | None = None
        self.acknowledgement: bytes | ValueError | None = None

    def create(self, encoded: bytes) -> bytes:
        request = pb.CreateEGraphRequest.from_binary(encoded)
        self.creations.append(request)
        handle = len(self.creations)
        self.threads[handle] = request.threads or 7
        self.no_decomp[handle] = request.no_decomp
        return pb.CreateEGraphResponse(egraph_id=handle).to_binary()

    def configure_resources(self, encoded: bytes) -> bytes:
        assert isinstance(encoded, bytes)
        request = pb.ConfigureEGraphResourcesRequest.from_binary(encoded)
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert request.operation is not None
        if request.operation.field == "threads":
            self.threads[request.egraph_id] = request.operation.value or 7
        elif request.operation.field == "no_decomp":
            self.no_decomp[request.egraph_id] = request.operation.value
        else:
            assert request.operation.field == "query"
            assert isinstance(request.operation.value, pb.Unit)
        if isinstance(self.acknowledgement, ValueError):
            raise self.acknowledgement
        if self.acknowledgement is not None:
            return self.acknowledgement
        return pb.ConfigureEGraphResourcesResponse(
            threads=self.threads[request.egraph_id], no_decomp=self.no_decomp[request.egraph_id]
        ).to_binary()

    def clone_graph(self, encoded: bytes) -> bytes:
        request = pb.CloneEGraphRequest.from_binary(encoded)
        handle = max(self.threads) + 1
        self.threads[handle] = self.threads[request.egraph_id]
        self.no_decomp[handle] = self.no_decomp[request.egraph_id]
        return pb.CloneEGraphResponse(egraph_id=handle).to_binary()

    def destroy(self, encoded: bytes) -> bytes:
        handle = pb.DestroyEGraphRequest.from_binary(encoded).egraph_id
        del self.threads[handle]
        del self.no_decomp[handle]
        return pb.DestroyEGraphResponse().to_binary()

    def return_to(self, encoded: bytes) -> bytes:
        request = pb.ReturnToEGraphRequest.from_binary(encoded)
        del self.threads[request.active_id]
        del self.no_decomp[request.active_id]
        assert request.saved_id in self.threads
        return pb.ReturnToEGraphResponse().to_binary()


@pytest.fixture
def resource_transport(monkeypatch: pytest.MonkeyPatch) -> ResourceTransport:
    transport = ResourceTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport, raising=False)
    return transport


@pytest.mark.parametrize("seminaive", [False, True])
@pytest.mark.parametrize("no_decomp", [False, True])
def test_compilation_flags_use_canonical_creation_bytes(resource_transport, seminaive, no_decomp):
    graph = EGraph(seminaive=seminaive, no_decomp=no_decomp)
    request = resource_transport.creations[0]
    assert request.options.has_field("seminaive")
    assert request.options.seminaive is seminaive
    assert request.no_decomp is no_decomp
    assert graph.no_decomp() is no_decomp
    assert resource_transport.requests[-1].operation == Oneof("query", pb.Unit())


@pytest.mark.parametrize("direct_state", [False, True])
def test_naive_recording_rejects_before_create(resource_transport, direct_state):
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    create = EGraphState if direct_state else EGraph
    with pytest.raises(NotImplementedError, match="initial seminaive=False"):
        create(seminaive=False, no_decomp=True, save_egglog_string=True)
    assert not resource_transport.creations
    assert not resource_transport.requests


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_invalid_compilation_flags_have_no_transport_effect(resource_transport, value):
    for keyword in ("seminaive", "no_decomp"):
        with pytest.raises(TypeError, match="booleans"):
            EGraph(**{keyword: value})
    assert not resource_transport.creations
    graph = EGraph()
    with pytest.raises(TypeError, match="boolean"):
        graph.set_no_decomp(value)
    assert not resource_transport.requests


def test_no_decomp_queries_are_uncached_and_updates_are_handle_local(resource_transport):
    graph = EGraph(no_decomp=True)
    assert graph.no_decomp()
    with graph:
        assert graph.no_decomp()
        assert graph.set_no_decomp(False) is None
        disabled = resource_transport.requests[-1]
        assert disabled.egraph_id == 2
        assert disabled.operation == Oneof("no_decomp", False)
        assert not graph.no_decomp()
        graph.set_no_decomp(True)
        assert resource_transport.requests[-1].to_binary() != disabled.to_binary()
    assert graph.no_decomp()
    assert resource_transport.requests[-1].egraph_id == 1
    resource_transport.no_decomp[1] = False
    assert not graph.no_decomp()
    graph._state.destroy()
    calls = len(resource_transport.requests)
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.no_decomp()
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.set_no_decomp(False)
    assert len(resource_transport.requests) == calls


@pytest.mark.parametrize("failure", ["transport", "decode", "value"])
def test_no_decomp_failed_acknowledgement_does_not_retry_or_cache(resource_transport, failure):
    graph = EGraph()
    original = ValueError("lost resource acknowledgement")
    resource_transport.acknowledgement = {
        "transport": original,
        "decode": b"\x80",
        "value": pb.ConfigureEGraphResourcesResponse(threads=1, no_decomp=False).to_binary(),
    }[failure]
    with pytest.raises(ValueError) as caught:  # noqa: PT011 -- decoder wording is provider-owned; identity/effects below.
        graph.set_no_decomp(True)
    if failure == "transport":
        assert caught.value is original
    assert len(resource_transport.requests) == 1
    resource_transport.acknowledgement = None
    assert graph.no_decomp()
    assert len(resource_transport.requests) == 2


@pytest.mark.parametrize("active", ["_cost_model_active", "_native_running"])
def test_no_decomp_resource_calls_reject_active_callbacks_before_transport(resource_transport, active):
    graph = EGraph()
    setattr(graph._state, active, True)
    with pytest.raises(RuntimeError, match="cost-model callback"):
        graph.no_decomp()
    with pytest.raises(RuntimeError, match="cost-model callback"):
        graph.set_no_decomp(True)
    assert not resource_transport.requests


@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_zero_threads_bytes_keep_presence_and_read_actual_count(
    resource_transport: ResourceTransport, *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=1 if use_setter else 0)
    if use_setter:
        assert graph.set_num_threads(0) is None
        assert resource_transport.requests[0].operation == Oneof[Literal["threads"], int]("threads", 0)
    else:
        assert resource_transport.creations[0].has_field("threads")
        assert resource_transport.creations[0].threads == 0
    assert graph.num_threads() == 7
    assert resource_transport.requests[-1].operation == Oneof[Literal["query"], pb.Unit]("query", pb.Unit())


def test_resource_queries_are_uncached_uint64_and_updates_are_handle_local(
    resource_transport: ResourceTransport,
) -> None:
    graph = EGraph(num_threads=2)
    assert graph.num_threads() == 2
    with graph:
        assert graph.num_threads() == 2
        assert graph.set_num_threads(1) is None
        assert graph.num_threads() == 1
        assert resource_transport.requests[-1].egraph_id == 2
    assert graph.num_threads() == 2
    assert resource_transport.requests[-1].egraph_id == 1
    resource_transport.threads[1] = (1 << 32) + 9
    assert graph.num_threads() == (1 << 32) + 9
    graph._state.destroy()
    calls = len(resource_transport.requests)
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.num_threads()
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.set_num_threads(1)
    assert len(resource_transport.requests) == calls


class IndexedThreadCount:
    def __index__(self) -> int:
        return 2


class IntOnlyThreadCount:
    def __int__(self) -> int:
        return 2


@pytest.mark.parametrize("value", [IndexedThreadCount(), True, False, (1 << 32) - 1])
@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_thread_count_accepts_index_protocol(
    resource_transport: ResourceTransport, value: object, *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=1 if use_setter else value)
    if use_setter:
        graph.set_num_threads(value)
        assert resource_transport.requests[0].operation.value == index(value)
    else:
        assert resource_transport.creations[0].threads == index(value)
    assert graph.num_threads() == (index(value) or 7)


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
@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_invalid_thread_count_has_no_transport_effect(
    resource_transport: ResourceTransport, value: object, error: type[Exception], *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=2) if use_setter else None
    creations = len(resource_transport.creations)
    if graph is None:
        with pytest.raises(error):
            EGraph(num_threads=value)
    else:
        with pytest.raises(error):
            graph.set_num_threads(value)
    assert len(resource_transport.creations) == creations
    assert not resource_transport.requests
    if graph is not None:
        assert graph.num_threads() == 2


@pytest.mark.parametrize("use_setter", [False, True], ids=["query", "setter"])
def test_resource_transport_failure_does_not_manufacture_success(
    resource_transport: ResourceTransport, *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=2)
    resource_transport.error = ValueError("resource sentinel")
    if use_setter:
        with pytest.raises(ValueError, match="resource sentinel"):
            graph.set_num_threads(1)
    else:
        with pytest.raises(ValueError, match="resource sentinel"):
            graph.num_threads()
    resource_transport.error = None
    assert graph.num_threads() == 2


def test_resource_response_requires_positive_actual_count(resource_transport: ResourceTransport) -> None:
    graph = EGraph()
    resource_transport.threads[1] = 0
    with pytest.raises(ValueError, match="thread count"):
        graph.num_threads()
