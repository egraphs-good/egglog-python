"""The native boundary accepts actual schema bytes, never Python AST objects."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import bindings
from egglog.egraph_state import _decode_run_frames


def _one_observation(encoded: list[bytes]) -> pb.RunProgramOutput:
    """Require one success observation plus its exclusive completion frame."""
    outputs = []
    for frame in _decode_run_frames(encoded):
        assert frame.result is not None
        assert frame.result.field == "output", frame.result.value
        if frame.result.value.outputs:
            outputs.append(frame.result.value)
    assert len(outputs) == 1
    return outputs[0]


def test_native_surface_contains_only_encoded_transport_and_host_provider():
    assert {name for name in vars(bindings) if not name.startswith("_")} == {"PrimitiveFailure"}
    assert callable(bindings._export_pyobject_catalog)
    assert callable(bindings._ProtoEngine)


def test_table_cell_reader_preserves_availability_and_ownership_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Table response inspection accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    available = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 0))
    unavailable = pb.TableCell(kind=Oneof[Literal["unavailable_sort"], int]("unavailable_sort", 1))
    response = pb.RunProgramOutput(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="i64"))
            ),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Hidden")),
        ],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 7))
                ),
            )
        ],
        outputs=[
            pb.CommandOutput(
                kind=Oneof[Literal["table_rows"], pb.TableRows](
                    "table_rows",
                    pb.TableRows(
                        table="table",
                        rows=[
                            pb.FunctionRow(args=[available, unavailable, available], output=unavailable, subsumed=True),
                            pb.FunctionRow(args=[unavailable, available, unavailable], output=available),
                        ],
                    ),
                ),
                location=pb.CommandLocation(path=[0]),
            )
        ],
    )
    encoded = response.to_binary()
    decoded = pb.RunProgramOutput.from_binary(encoded)
    assert decoded == response
    assert decoded.to_binary() == encoded
    # Returned data own their values. Mutating the constructed input cannot
    # supply an unavailable cell with a fake representative in the decoded data.
    unavailable.kind = Oneof[Literal["term"], int]("term", 0)
    response.nodes.clear()
    output = decoded.outputs[0]
    assert output.location == pb.CommandLocation(path=[0])
    assert output.kind is not None
    assert output.kind.field == "table_rows"
    assert output.kind.value.table == "table"
    first, second = output.kind.value.rows
    assert first.subsumed
    assert not second.subsumed
    assert first.output is not None
    assert second.output is not None
    assert first.args[0] == first.args[2] == second.args[1] == second.output
    assert first.args[1] == first.output == second.args[0] == second.args[2]
    term = second.output.kind
    missing = first.output.kind
    assert term is not None
    assert term.field == "term"
    assert missing is not None
    assert missing.field == "unavailable_sort"
    assert term.value == 0  # oneof presence must retain index zero
    assert missing.value == 1
    assert decoded.nodes[term.value].kind == Oneof(
        "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 7))
    )
    assert decoded.sorts[decoded.nodes[term.value].sort_id].kind == Oneof("application", pb.SortApplication(name="i64"))
    assert decoded.sorts[missing.value].kind == Oneof("eq", "Hidden")
    assert len(decoded.nodes) == 1
    assert pb.TableCell().kind is None  # absence is neither term zero nor unavailable
    assert pb.FunctionRow().output is None
    assert pb.RunProgramOutput.from_binary(decoded.to_binary()) == decoded


def test_binary_engine_lifecycle_and_scalar_extract():
    engine = bindings._ProtoEngine()
    sort = pb.Sort(
        kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="i64"))
    )
    created = pb.CreateEGraphResponse.from_binary(
        engine.create(pb.CreateEGraphRequest(sorts=[sort], options=pb.EGraphOptions(cost_sort=0)).to_binary()),
    )
    program = pb.Program(
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
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=created.egraph_id, program=program).to_binary())
    assert host_error is None
    response = _one_observation(encoded)
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
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="i64"))
            ),
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
        rulesets=[
            pb.RulesetBinding(
                name="rebuilding_ruleset",
                value=pb.Ruleset(
                    rules=[
                        pb.Rule(
                            name="derive",
                            eval_mode=pb.RuleEvalMode.SEMINAIVE,
                            kind=Oneof[Literal["rule"], pb.RuleBody](
                                "rule",
                                pb.RuleBody(query=[3], head=[pb.Action(kind=Oneof[Literal["term"], int]("term", 4))]),
                            ),
                        )
                    ]
                ),
            ),
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", 2))
                )
            ),
            pb.Command(
                kind=Oneof[Literal["run"], pb.Run](
                    "run",
                    pb.Run(
                        rulesets=[pb.RulesetSelection(kind=Oneof[Literal["name"], str]("name", "rebuilding_ruleset"))]
                    ),
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
                sorts=[
                    pb.Sort(
                        kind=Oneof[Literal["application"], pb.SortApplication](
                            "application", pb.SortApplication(name="i64")
                        )
                    )
                ],
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


def test_binary_proof_modes_requested_outputs_and_control(proof_engine):
    engine, handles, mode = proof_engine
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=_proof_program()).to_binary())
    assert host_error is None
    frames = _decode_run_frames(encoded)
    assert [pb.RunProgramResponse.from_binary(frame.to_binary()) for frame in frames] == frames
    outputs = [
        output
        for frame in frames
        if frame.result is not None and frame.result.field == "output"
        for output in frame.result.value.outputs
    ]
    requested_run = outputs[0].kind
    assert requested_run is not None
    assert requested_run.field == "run"
    assert requested_run.value.updated
    assert not requested_run.value.can_stop
    paths = [output.location.path for output in outputs if output.location is not None]
    terminal = frames[-1].result
    assert terminal is not None
    if mode in (pb.ExecutionMode.NORMAL, pb.ExecutionMode.TERM_ENCODING):
        assert paths == [[1]]
        assert terminal.field == "error"
        assert terminal.value.code == pb.ErrorCode.RUNTIME_FAILURE
        assert terminal.value.location is not None
        assert terminal.value.location.path == [3]
    else:
        assert terminal.field == "output"
        assert paths == ([[1], [2], [3], [4]] if mode == pb.ExecutionMode.PROOF_TESTING else [[1], [3], [4]])
        for output in outputs[1:]:
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


def test_binary_proof_prefix_clone_and_native_error_order(proof_engine):
    engine, handles, mode = proof_engine
    setup = _proof_program()
    del setup.commands[2:]
    setup_bytes, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=setup).to_binary())
    assert host_error is None
    assert all(frame.result is not None and frame.result.field == "output" for frame in _decode_run_frames(setup_bytes))
    cloned = pb.CloneEGraphResponse.from_binary(
        engine.clone_graph(pb.CloneEGraphRequest(egraph_id=handles[0]).to_binary())
    )
    handles.append(cloned.egraph_id)
    proofs_enabled = mode in (pb.ExecutionMode.PROOFS, pb.ExecutionMode.PROOF_TESTING)
    program = _proof_program()
    program.declarations.clear()
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
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program).to_binary())
    assert host_error is None
    frames = _decode_run_frames(encoded)
    terminal = frames[-1].result
    assert terminal is not None
    assert terminal.field == "error"
    assert terminal.value.code == pb.ErrorCode.RUNTIME_FAILURE
    assert terminal.value.location is not None
    assert terminal.value.location.path == ([2] if proofs_enabled else [1])
    outputs = [
        output
        for frame in frames
        if frame.result is not None and frame.result.field == "output"
        for output in frame.result.value.outputs
    ]
    assert len(outputs) == int(proofs_enabled)
    for handle, node, succeeds in [(handles[0], 7, True), (handles[0], 9, False), (handles[1], 7, False)]:
        program.commands = [pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[node])))]
        checked_bytes, host_error = engine.run(pb.RunProgramRequest(egraph_id=handle, program=program).to_binary())
        assert host_error is None
        checked = _decode_run_frames(checked_bytes)[-1].result
        assert checked is not None
        assert (checked.field == "output") == succeeds
    program.commands = [
        pb.Command(
            kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="Goal"))
        )
    ]
    copied_bytes, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[1], program=program).to_binary())
    assert host_error is None
    copied = _decode_run_frames(copied_bytes)[-1].result
    assert copied is not None
    assert (copied.field == "output") == proofs_enabled
    if not proofs_enabled:
        assert copied.field == "error"
        assert copied.value.code == pb.ErrorCode.RUNTIME_FAILURE
    for handle in handles:
        engine.destroy(pb.DestroyEGraphRequest(egraph_id=handle).to_binary())
    handles.clear()
    retained = [pb.RunProgramResponse.from_binary(frame.to_binary()) for frame in frames]
    assert retained == _decode_run_frames(encoded)
    if proofs_enabled:
        first = retained[0].result
        assert first is not None
        assert first.field == "output"
        output = first.value.outputs[0].kind
        assert output is not None
        assert output.field == "proof"
        proof = output.value
        assert proof.has_field("root")
        root = proof.proofs[proof.root]
        assert root.has_field("lhs")
        assert root.has_field("rhs")
        assert root.lhs == root.rhs


@pytest.mark.parametrize(
    "proof_engine",
    [pb.ExecutionMode.TERM_ENCODING, pb.ExecutionMode.PROOFS, pb.ExecutionMode.PROOF_TESTING],
    indirect=True,
)
def test_binary_encoded_extract_stays_explicitly_unsupported(proof_engine):
    engine, handles, _mode = proof_engine
    program = _proof_program()
    program.commands = [
        program.commands[0],
        pb.Command(
            kind=Oneof[Literal["extract"], pb.Extract](
                "extract", pb.Extract(roots=[2], variants=1, extractor=pb.Extractor.TREE)
            )
        ),
    ]
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program).to_binary())
    assert host_error is None
    failed = _decode_run_frames(encoded)
    assert len(failed) == 1
    terminal = failed[0].result
    assert terminal is not None
    assert terminal.field == "error"
    assert terminal.value.code == pb.ErrorCode.INVALID_PROGRAM
    program.commands = [pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[2])))]
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program).to_binary())
    assert host_error is None
    retry = _decode_run_frames(encoded)[-1].result
    assert retry is not None
    assert retry.field == "error"
    assert retry.value.location is not None


@pytest.mark.parametrize("max_rows", [None, 0], ids=["print-all-rows", "print-zero-rows"])
@pytest.mark.parametrize(
    "proof_engine",
    [pb.ExecutionMode.TERM_ENCODING, pb.ExecutionMode.PROOFS, pb.ExecutionMode.PROOF_TESTING],
    indirect=True,
)
def test_binary_encoded_print_function_returns_typed_logical_rows(proof_engine, max_rows):
    engine, handles, _mode = proof_engine
    program = _proof_program()
    program.commands = [
        program.commands[0],
        pb.Command(
            kind=Oneof[Literal["table_rows"], pb.GetTableRows](
                "table_rows", pb.GetTableRows(table="N", max_rows=max_rows)
            )
        ),
    ]
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program).to_binary())
    assert host_error is None
    response = _one_observation(encoded)
    (output,) = response.outputs
    assert output.location == pb.CommandLocation(path=[1])
    assert output.kind is not None
    assert output.kind.field == "table_rows"
    table = output.kind.value
    assert table.table == "N"
    if max_rows == 0:
        assert not table.rows
        assert not response.nodes
        assert not response.sorts
    else:
        (row,) = table.rows
        assert not row.subsumed
        (argument,) = row.args
        assert argument.kind is not None
        assert argument.kind.field == "term"
        assert row.output is not None
        assert row.output.kind is not None
        assert row.output.kind.field == "term"
        argument_id, result_id = argument.kind.value, row.output.kind.value
        assert argument_id != result_id
        assert len(response.nodes) == len(response.sorts) == 2
        argument_node = response.nodes[argument_id]
        result_node = response.nodes[result_id]
        assert argument_node.kind == Oneof[Literal["primitive_value"], pb.PrimitiveValue](
            "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
        )
        assert response.sorts[argument_node.sort_id].kind == Oneof[Literal["application"], pb.SortApplication](
            "application", pb.SortApplication(name="i64")
        )
        assert result_node.kind == Oneof[Literal["call"], pb.Call]("call", pb.Call(func="N", args=[argument_id]))
        assert response.sorts[result_node.sort_id].kind == Oneof[Literal["eq"], str]("eq", "Num")
    assert pb.RunProgramOutput.from_binary(response.to_binary()) == response
    program.commands = [pb.Command(kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[2])))]
    encoded, host_error = engine.run(pb.RunProgramRequest(egraph_id=handles[0], program=program).to_binary())
    assert host_error is None
    assert all(frame.result is not None and frame.result.field == "output" for frame in _decode_run_frames(encoded))


def test_binary_unknown_mode_rejected_and_diagnostics_off_works():
    engine = bindings._ProtoEngine()
    request = pb.CreateEGraphRequest(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="i64"))
            )
        ],
        options=pb.EGraphOptions(cost_sort=0, execution_mode=pb.ExecutionMode(999)),
        threads=1,
    )
    with pytest.raises(ValueError, match="unknown execution mode"):
        engine.create(request.to_binary())
    assert request.options is not None
    request.options.execution_mode = pb.ExecutionMode.NORMAL
    created = pb.CreateEGraphResponse.from_binary(engine.create(request.to_binary()))
    try:
        encoded, host_error = engine.run(
            pb.RunProgramRequest(egraph_id=created.egraph_id, program=pb.Program()).to_binary()
        )
        assert host_error is None
        frames = _decode_run_frames(encoded)
        assert len(frames) == 1
        assert frames[0].result == Oneof[Literal["output"], pb.RunProgramOutput]("output", pb.RunProgramOutput())
    finally:
        engine.destroy(pb.DestroyEGraphRequest(egraph_id=created.egraph_id).to_binary())
