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
