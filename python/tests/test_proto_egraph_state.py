"""Record/bytes plumbing tests; the inert responder is not engine conformance."""

from operator import index
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import EGraph, Expr, bindings, constant, eq, i64, method, set_
from egglog._catalog import builtin_catalog
from egglog._program import pack
from egglog.runtime import resolve_callable


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
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    def forbidden(*args, **kwargs):
        pytest.fail("Publication bookkeeping used the legacy engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    selected = ruleset(rule(name="first").then(), name="receipt")
    identifiers = {}
    snapshot = selected._snapshot(identifiers)
    with pytest.raises(bindings.EggSmolError, match="injected install boundary"):
        graph._state.publish_ruleset(snapshot, identifiers)
    first_name = requests[0].program.rulesets[0].name
    assert (("ruleset", first_name) in graph._state.definitions) is completed
    assert (selected.__egg_ident__ in graph._state.rulesets) is completed
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
            ).to_binary()

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
            ).to_binary()

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
            returned = transport.run(encoded)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            response = pb.RunProgramResponse.from_binary(returned)
            assert response.ruleset_publication is not None
            responses.append(response)
            return returned

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
        returned = transport.run(encoded)
        (tmp_path / f"response-{index}.pb").write_bytes(returned)
        response = pb.RunProgramResponse.from_binary(returned)
        assert response.ruleset_publication is not None
        responses.append(response)
        return returned

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
    with pytest.raises(bindings.EggSmolError):
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
        with pytest.raises(bindings.EggSmolError) as raised:
            graph.register(first, first, Second(1)) if prefix else graph.register(first, Second(1))
        message = "At 0:0 of \nRule already exists: first"
        assert type(raised.value) is bindings.EggSmolError
        assert str(raised.value) == message
        assert raised.value.args == (message, False)
        assert raised.value.replayable_by_fail is False
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
            with pytest.raises(bindings.EggSmolError, match="Rule already exists: first"):
                _run_ruleset_step(graph, selected)
            assert _run_ruleset_step(graph) == {label: 1}
        else:
            assert _run_ruleset_step(graph, selected) == {label: 2 if default else 1}
    else:
        assert _run_ruleset_step(graph, selected) == {label: 1}
        if named and default:
            with pytest.raises(bindings.EggSmolError, match="Rule already exists: first"):
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
    # Packing has an explicit, pre-existing same-Program reconciliation guard.
    with pytest.raises(ValueError, match="Ambiguous same-name definitions"):
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
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    graph.check_fail(eq(i64(1)).to(i64(2)))
    program = requests[0].program
    assert len(program.commands) == 1
    assert program.commands[0].kind.field == "fail"
    check = program.commands[0].kind.value
    assert check.kind.field == "check"
    assert len(check.kind.value.facts) == 1
    fact = program.nodes[check.kind.value.facts[0]]
    assert fact.kind.field == "union"
    assert [program.nodes[index].kind.value.value.value for index in fact.kind.value.members] == [1, 2]
    graph.check_fail()
    assert requests[1].program.commands[0].kind.value.kind.value.facts == []


@pytest.mark.parametrize("operation", ["check", "check_bool", "check_fail"])
@pytest.mark.parametrize(
    ("code", "location"),
    [
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.CHECK_FAILED, None),
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[1])),
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
            return pb.RunProgramResponse(error=record, ruleset_publication=pb.RulesetPublication()).to_binary()

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
    ):
        assert method(eq(i64(1)).to(i64(1))) is False
    else:
        with pytest.raises(bindings.EggSmolError) as caught:
            method(eq(i64(1)).to(i64(1)))
        assert type(caught.value) is bindings.EggSmolError
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
            ).to_binary()

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
    with pytest.raises(bindings.EggSmolError):
        graph.register(action)
    assert graph._state.definitions == (before | expected if located else before)


@pytest.mark.parametrize("error_type", [ValueError, bindings.EggSmolError])
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
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary()

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
    assert [command.kind.value.kind.field for command in program.commands] == ["set", "term", "set", "term"]
    first = program.commands[0].kind.value.kind.value
    second = program.commands[2].kind.value.kind.value
    assert program.nodes[program.commands[1].kind.value.kind.value].kind.value.func == first.target.func
    assert program.nodes[program.commands[3].kind.value.kind.value].kind.value.func == second.target.func
    definitions = {declaration.kind.value.name: declaration for declaration in program.declarations}
    assert definitions[first.target.func].kind.field == "function"
    assert definitions[second.target.func].bindings.python.views[0].kind == pb.PythonCallKind.CONSTANT
    rhs = program.nodes[second.value].kind.value
    assert rhs.func == "egglog.core.i64.add"
    assert program.nodes[rhs.args[0]].kind.value.func == first.target.func
    with pytest.raises(NotImplementedError, match="Fact-to-action"):
        graph.register(set_(current).to(i64(1)), eq(i64(1)).to(i64(1)))
    assert len(requests) == 1
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
            ).to_binary()

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

        def run(self, encoded: bytes) -> bytes:
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
            ).to_binary()

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
    assert program.ir_version == 1
    command = program.commands[0].kind
    assert command is not None
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
    assert destroyed == [2]
    graph.close()
    assert graph._state.handle == 1
    graph._state.destroy()
    graph._state.destroy()
    assert destroyed == [2, 1]


@pytest.mark.parametrize("body_raises", [False, True])
def test_context_restores_saved_state_when_clone_disposal_fails(
    monkeypatch: pytest.MonkeyPatch, body_raises: bool
) -> None:
    class FaultTransport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def clone_graph(self, encoded: bytes) -> bytes:
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            if pb.DestroyEGraphRequest.from_binary(encoded).egraph_id == 2:
                msg = "cleanup sentinel"
                raise RuntimeError(msg)
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", FaultTransport, raising=False)
    graph = EGraph()
    saved = graph._state
    with (  # noqa: PT012 -- exercise body and disposal together
        pytest.raises(
            ValueError if body_raises else RuntimeError, match="body sentinel" if body_raises else "cleanup sentinel"
        ),
        graph,
    ):
        assert graph._state is not saved
        if body_raises:
            msg = "body sentinel"
            raise ValueError(msg)
    assert graph._state is saved
    assert not graph._state_stack


class ResourceTransport:
    """An inert bytes receiver, not a simulation of native resource allocation."""

    def __init__(self) -> None:
        self.creations: list[pb.CreateEGraphRequest] = []
        self.requests: list[pb.ConfigureEGraphResourcesRequest] = []
        self.threads: dict[int, int] = {}
        self.error: ValueError | None = None

    def create(self, encoded: bytes) -> bytes:
        request = pb.CreateEGraphRequest.from_binary(encoded)
        self.creations.append(request)
        handle = len(self.creations)
        self.threads[handle] = request.threads or 7
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
        else:
            assert request.operation.field == "query"
            assert isinstance(request.operation.value, pb.Unit)
        return pb.ConfigureEGraphResourcesResponse(threads=self.threads[request.egraph_id]).to_binary()

    def clone_graph(self, encoded: bytes) -> bytes:
        request = pb.CloneEGraphRequest.from_binary(encoded)
        handle = max(self.threads) + 1
        self.threads[handle] = self.threads[request.egraph_id]
        return pb.CloneEGraphResponse(egraph_id=handle).to_binary()

    def destroy(self, encoded: bytes) -> bytes:
        del self.threads[pb.DestroyEGraphRequest.from_binary(encoded).egraph_id]
        return pb.DestroyEGraphResponse().to_binary()


@pytest.fixture
def resource_transport(monkeypatch: pytest.MonkeyPatch) -> ResourceTransport:
    transport = ResourceTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport, raising=False)
    return transport


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
