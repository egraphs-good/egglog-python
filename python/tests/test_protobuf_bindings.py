"""The native boundary accepts actual schema bytes, never Python AST objects."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import bindings


def test_binary_engine_lifecycle_and_scalar_extract():
    engine = bindings._ProtoEngine()
    sort = pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    created = pb.CreateEGraphResponse.from_binary(
        engine.create(pb.CreateEGraphRequest(sorts=[sort], options=pb.EGraphOptions(cost_sort=0)).to_binary()),
    )
    program = pb.Program(
        ir_version=1,
        sorts=[sort],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 7))
                ),
            )
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["extract"], pb.Extract](
                    "extract", pb.Extract(roots=[0], variants=1, extractor=pb.Extractor.TREE)
                )
            )
        ],
    )
    response = pb.RunProgramResponse.from_binary(
        engine.run(pb.RunProgramRequest(egraph_id=created.egraph_id, program=program, profile=True).to_binary()),
    )
    assert response.error is None
    output = response.outputs[0].kind
    assert output is not None
    assert output.field == "extraction"
    root = response.nodes[output.value.roots[0].variants[0].term]
    assert root.kind is not None
    assert root.kind.field == "primitive_value"
    assert root.kind.value.value == Oneof("i64", 7)

    cloned = pb.CloneEGraphResponse.from_binary(
        engine.clone_graph(pb.CloneEGraphRequest(egraph_id=created.egraph_id).to_binary()),
    )
    assert cloned.egraph_id != created.egraph_id
    for handle in (created.egraph_id, cloned.egraph_id):
        pb.DestroyEGraphResponse.from_binary(engine.destroy(pb.DestroyEGraphRequest(egraph_id=handle).to_binary()))


def test_binary_engine_rejects_non_bytes_and_malformed_payloads():
    engine = bindings._ProtoEngine()
    with pytest.raises(TypeError):
        engine.create(pb.CreateEGraphRequest())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r".+"):
        engine.create(b"\xff")


def test_maintenance_reader_preserves_site_and_metric_presence():
    """Generated response data, not Program indices or inferred native timings."""
    # Native prost uses unpacked paths; this reader writes packed paths. Both
    # protobuf encodings carry the same ordered indices, not different records.
    assert pb.CommandLocation.from_binary(bytes.fromhex("0801")) == pb.CommandLocation.from_binary(
        bytes.fromhex("0a0101")
    )
    sites = [
        pb.MaintenanceSite(
            target=Oneof[Literal["command"], pb.StaticCommandPath]("command", pb.StaticCommandPath(path=[0, 2]))
        ),
        pb.MaintenanceSite(target=Oneof[Literal["declaration"], int]("declaration", 0)),
        pb.MaintenanceSite(target=Oneof[Literal["sort"], int]("sort", 0)),
        pb.MaintenanceSite(target=Oneof[Literal["ruleset"], int]("ruleset", 0)),
    ]
    response = pb.RunProgramResponse(
        profile=pb.ProfileSummary(
            complete=False,
            runs=[pb.RunSummary(command_path=[0], invocations=1, search_and_apply_nanos=0)],
            maintenance=[pb.MaintenanceSummary(site=site, invocations=1, merge_nanos=0) for site in sites],
        )
    )
    encoded = response.to_binary()
    decoded = pb.RunProgramResponse.from_binary(encoded)
    assert decoded.to_binary() == encoded
    assert decoded.profile is not None
    assert not decoded.profile.complete
    assert decoded.profile.runs[0].has_field("search_and_apply_nanos")
    assert not decoded.profile.runs[0].has_field("merge_nanos")
    assert [summary.site for summary in decoded.profile.maintenance] == sites
    for summary in decoded.profile.maintenance:
        assert summary.invocations == 1
        assert summary.has_field("merge_nanos")
        assert summary.merge_nanos == 0
        assert not summary.has_field("search_and_apply_nanos")
        assert not summary.has_field("rebuild_nanos")


def _proof_program() -> pb.Program:
    """Actual wire fixture: only a rule derives Goal, avoiding fiat-only proof roots."""
    nodes = [
        pb.Node(
            sort_id=0,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
            ),
        ),
        pb.Node(sort_id=0, kind=Oneof[Literal["var"], str]("var", "x")),
    ]
    nodes.extend(
        pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func=name, args=[argument])))
        for name, argument in [("N", 0), ("N", 1), ("Goal", 1), ("Goal", 0)]
    )
    for value in (2, 3):
        literal = len(nodes)
        nodes.extend([
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
            pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="N", args=[literal]))),
        ])
    return pb.Program(
        ir_version=1,
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Num")),
        ],
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Num"))),
            *(
                pb.Declaration(
                    kind=Oneof[Literal["constructor"], pb.Constructor](
                        "constructor", pb.Constructor(name=name, inputs=[pb.Arg(sort=0, name="x")], output=1)
                    )
                )
                for name in ("N", "Goal", "Absent")
            ),
        ],
        nodes=nodes,
        rules=[
            pb.RuleDecl(
                name="derive",
                eval_mode=pb.RuleEvalMode.SEMINAIVE,
                kind=Oneof[Literal["rule"], pb.Rule](
                    "rule", pb.Rule(query=[3], head=[pb.Action(kind=Oneof[Literal["term"], int]("term", 4))])
                ),
            )
        ],
        rulesets=[
            pb.Ruleset(
                name="rebuilding_ruleset", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList(rules=[0]))
            )
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", 2))
                )
            ),
            pb.Command(
                kind=Oneof[Literal["run"], pb.Run](
                    "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", 0)))
                )
            ),
            pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[5]))),
            pb.Command(kind=Oneof[Literal["prove"], pb.Prove]("prove", pb.Prove(facts=[5]))),
            pb.Command(
                kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="Goal"))
            ),
        ],
    )


@pytest.fixture(
    params=[
        pb.ExecutionMode.NORMAL,
        pb.ExecutionMode.TERM_ENCODING,
        pb.ExecutionMode.PROOFS,
        pb.ExecutionMode.PROOF_TESTING,
    ]
)
def proof_engine(request):
    """Keep every test's original and cloned native handles scoped to that test."""
    engine = bindings._ProtoEngine()
    created = pb.CreateEGraphResponse.from_binary(
        engine.create(
            pb.CreateEGraphRequest(
                sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
                options=pb.EGraphOptions(cost_sort=0, execution_mode=request.param),
                threads=1,
            ).to_binary()
        )
    )
    handles = [created.egraph_id]
    try:
        yield engine, handles, request.param
    finally:
        for handle in handles:
            engine.destroy(pb.DestroyEGraphRequest(egraph_id=handle).to_binary())


def test_binary_proof_modes_requested_outputs_and_maintenance(proof_engine):
    engine, handles, mode = proof_engine
    encoded = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=_proof_program(), profile=True).to_binary())
    response = pb.RunProgramResponse.from_binary(encoded)
    assert pb.RunProgramResponse.from_binary(response.to_binary()) == response
    profile = response.profile
    assert profile is not None
    assert len(profile.runs) == 1
    assert profile.runs[0].command_path == [1]
    assert profile.runs[0].invocations == 1
    for field in ("search_and_apply_nanos", "merge_nanos", "rebuild_nanos"):
        assert profile.runs[0].has_field(field)
    requested_run = response.outputs[0].kind
    assert requested_run is not None
    assert requested_run.field == "run"
    assert requested_run.value.updated
    assert not requested_run.value.can_stop
    paths = [output.location.path for output in response.outputs if output.location is not None]
    if mode in (pb.ExecutionMode.NORMAL, pb.ExecutionMode.TERM_ENCODING):
        assert paths == [[1]]
        assert response.error is not None
        assert response.error.code == pb.ErrorCode.PROOF_FAILED
        assert response.error.proof_failure is not None
        assert response.error.proof_failure.reason == pb.ProofFailureReason.PROOFS_NOT_ENABLED
        assert response.error.location is not None
        assert response.error.location.path == [3]
        assert not profile.complete
    else:
        assert response.error is None
        assert profile.complete
        assert paths == ([[1], [2], [3], [4]] if mode == pb.ExecutionMode.PROOF_TESTING else [[1], [3], [4]])
        for output in response.outputs[1:]:
            assert output.kind is not None
            assert output.kind.field == "proof"
            proof = output.kind.value
            assert proof.has_field("root")
            assert proof.root < len(proof.proofs)
            assert proof.terms
            for step in proof.proofs:
                assert step.has_field("lhs")
                assert step.lhs < len(proof.terms)
                assert step.has_field("rhs")
                assert step.rhs < len(proof.terms)
                assert step.justification is not None
    if mode != pb.ExecutionMode.NORMAL:
        assert not response.rules
        assert not profile.runs[0].rules
        for command in (1, 3):
            site = pb.MaintenanceSite(
                target=Oneof[Literal["command"], pb.StaticCommandPath]("command", pb.StaticCommandPath(path=[command]))
            )
            (summary,) = [summary for summary in profile.maintenance if summary.site == site]
            assert summary.invocations > 1
            assert summary.has_field("search_and_apply_nanos")
        assert any(
            summary.site == pb.MaintenanceSite(target=Oneof[Literal["ruleset"], int]("ruleset", 0))
            for summary in profile.maintenance
        )


def test_binary_proof_prefix_clone_and_native_error_order(proof_engine):
    engine, handles, mode = proof_engine
    setup = _proof_program()
    del setup.commands[3:]
    response = pb.RunProgramResponse.from_binary(
        engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=setup, profile=True).to_binary())
    )
    assert response.error is None
    cloned = pb.CloneEGraphResponse.from_binary(
        engine.clone_graph(pb.CloneEGraphRequest(egraph_id=handles[0]).to_binary())
    )
    handles.append(cloned.egraph_id)
    proofs_enabled = mode in (pb.ExecutionMode.PROOFS, pb.ExecutionMode.PROOF_TESTING)
    program = _proof_program()
    program.declarations.clear()
    program.rules.clear()
    program.rulesets.clear()
    program.commands = [
        pb.Command(
            kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="Goal"))
        ),
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action]("action", pb.Action(kind=Oneof[Literal["term"], int]("term", 7)))
        ),
        pb.Command(
            kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="Absent"))
        ),
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action]("action", pb.Action(kind=Oneof[Literal["term"], int]("term", 9)))
        ),
    ]
    if not proofs_enabled:
        del program.commands[0]
    encoded = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program, profile=True).to_binary())
    response = pb.RunProgramResponse.from_binary(encoded)
    assert response.error is not None
    assert response.error.code == pb.ErrorCode.PROOF_FAILED
    failure = response.error.proof_failure
    assert failure is not None
    assert failure.reason == pb.ProofFailureReason.QUERY_DID_NOT_MATCH
    assert failure.has_field("constructor")
    assert failure.constructor == "Absent"
    assert response.error.location is not None
    assert response.error.location.path == ([2] if proofs_enabled else [1])
    assert len(response.outputs) == int(proofs_enabled)
    assert response.profile is not None
    assert not response.profile.complete
    for handle, node, succeeds in [(handles[0], 7, True), (handles[0], 9, False), (handles[1], 7, False)]:
        program.commands = [pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[node])))]
        checked = pb.RunProgramResponse.from_binary(
            engine.run(pb.RunProgramRequest(egraph_id=handle, program=program, profile=True).to_binary())
        )
        assert (checked.error is None) == succeeds
    program.commands = [
        pb.Command(
            kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="Goal"))
        )
    ]
    copied = pb.RunProgramResponse.from_binary(
        engine.run(pb.RunProgramRequest(egraph_id=handles[1], program=program, profile=True).to_binary())
    )
    assert (copied.error is None) == proofs_enabled
    if not proofs_enabled:
        assert copied.error is not None
        assert copied.error.proof_failure is not None
        assert copied.error.proof_failure.reason == pb.ProofFailureReason.PROOFS_NOT_ENABLED
    for handle in handles:
        engine.destroy(pb.DestroyEGraphRequest(egraph_id=handle).to_binary())
    handles.clear()
    retained = pb.RunProgramResponse.from_binary(response.to_binary())
    assert retained == pb.RunProgramResponse.from_binary(encoded)
    if proofs_enabled:
        output = retained.outputs[0].kind
        assert output is not None
        assert output.field == "proof"
        proof = output.value
        assert proof.has_field("root")
        root = proof.proofs[proof.root]
        assert root.has_field("lhs")
        assert root.has_field("rhs")
        assert root.lhs == root.rhs


@pytest.mark.parametrize("observation", ["extract", "print_function"])
@pytest.mark.parametrize(
    "proof_engine",
    [pb.ExecutionMode.TERM_ENCODING, pb.ExecutionMode.PROOFS, pb.ExecutionMode.PROOF_TESTING],
    indirect=True,
)
def test_binary_encoded_observations_stay_explicitly_unsupported(proof_engine, observation):
    engine, handles, _mode = proof_engine
    program = _proof_program()
    program.commands = [
        program.commands[0],
        pb.Command(
            kind=Oneof[Literal["extract"], pb.Extract](
                "extract", pb.Extract(roots=[2], variants=1, extractor=pb.Extractor.TREE)
            )
        )
        if observation == "extract"
        else pb.Command(
            kind=Oneof[Literal["print_function"], pb.PrintFunction](
                "print_function", pb.PrintFunction(table="N", max_rows=0)
            )
        ),
    ]
    failed = pb.RunProgramResponse.from_binary(
        engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program, profile=True).to_binary())
    )
    assert failed.error is not None
    assert failed.error.code == pb.ErrorCode.INVALID_PROGRAM
    assert not failed.outputs
    assert failed.profile is not None
    assert not failed.profile.complete
    assert not failed.profile.runs
    assert not failed.profile.maintenance
    program.commands = [pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[2])))]
    retry = pb.RunProgramResponse.from_binary(
        engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program, profile=True).to_binary())
    )
    assert retry.error is not None
    assert retry.error.location is not None


def test_binary_unknown_mode_and_profile_off_stay_explicit():
    engine = bindings._ProtoEngine()
    request = pb.CreateEGraphRequest(
        sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
        options=pb.EGraphOptions(cost_sort=0, execution_mode=pb.ExecutionMode(999)),
        threads=1,
    )
    with pytest.raises(ValueError, match="unknown execution mode"):
        engine.create(request.to_binary())
    assert request.options is not None
    request.options.execution_mode = pb.ExecutionMode.NORMAL
    created = pb.CreateEGraphResponse.from_binary(engine.create(request.to_binary()))
    try:
        response = pb.RunProgramResponse.from_binary(
            engine.run(
                pb.RunProgramRequest(
                    egraph_id=created.egraph_id, program=pb.Program(ir_version=1), profile=False
                ).to_binary()
            )
        )
        assert response.error is not None
        assert response.error.code == pb.ErrorCode.INVALID_PROGRAM
        assert "profile=false" in response.error.message
        assert response.profile is None
        assert not response.outputs
    finally:
        engine.destroy(pb.DestroyEGraphRequest(egraph_id=created.egraph_id).to_binary())
