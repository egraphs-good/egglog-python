"""Record/bytes plumbing tests; the inert responder is not engine conformance."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import EGraph, bindings, i64
from egglog._catalog import builtin_catalog
from egglog._program import pack


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

    monkeypatch.setattr(bindings, "_ProtoEngine", FaultTransport)
    graph = EGraph()
    saved = graph._state
    with pytest.raises(  # noqa: PT012 -- exercise body and disposal together
        ValueError if body_raises else RuntimeError, match="body sentinel" if body_raises else "cleanup sentinel"
    ), graph:
        assert graph._state is not saved
        if body_raises:
            msg = "body sentinel"
            raise ValueError(msg)
    assert graph._state is saved
    assert not graph._state_stack
