from __future__ import annotations

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog._program import Builder, Ref, StructuralView
from egglog.type_constraint_solver import TypeConstraintError, infer_sort, substitute_sort


@pytest.fixture
def sorts() -> dict[str, Ref]:
    builder = Builder.from_program(
        pb.Program(
            sorts=[
                pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)),
                pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)),
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Unit"))),
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Map", args=[0, 1]))),
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Map", args=[2, 3]))),
            ],
        )
    )
    owner = builder.publish()
    return {name: owner.ref("sorts", index) for index, name in enumerate(("K", "V", "i64", "Unit", "Map", "closed"))}


def test_simple(sorts: dict[str, Ref]) -> None:
    bindings: dict[int, Ref] = {}
    infer_sort(sorts["i64"], sorts["i64"], 0, bindings)
    assert StructuralView(substitute_sort(sorts["i64"], bindings)) == StructuralView(sorts["i64"])
    assert not bindings


def test_wrong_arg(sorts: dict[str, Ref]) -> None:
    bindings: dict[int, Ref] = {}
    with pytest.raises(TypeConstraintError, match="Expected i64, got Unit"):
        infer_sort(sorts["i64"], sorts["Unit"], 0, bindings)
    assert not bindings


def test_generic(sorts: dict[str, Ref]) -> None:
    bindings: dict[int, Ref] = {}
    infer_sort(sorts["Map"], sorts["closed"], 2, bindings)
    infer_sort(sorts["K"], sorts["i64"], 2, bindings)
    assert StructuralView(substitute_sort(sorts["V"], bindings)) == StructuralView(sorts["Unit"])
    assert StructuralView(substitute_sort(sorts["Map"], bindings)) == StructuralView(sorts["closed"])


def test_generic_wrong(sorts: dict[str, Ref]) -> None:
    bindings: dict[int, Ref] = {}
    infer_sort(sorts["Map"], sorts["closed"], 2, bindings)
    before = bindings.copy()
    with pytest.raises(TypeConstraintError, match="Inconsistent sort for type parameter 0"):
        infer_sort(sorts["K"], sorts["Unit"], 2, bindings)
    assert bindings == before


def test_bound(sorts: dict[str, Ref]) -> None:
    bindings = {0: sorts["i64"]}
    infer_sort(sorts["Map"], sorts["closed"], 2, bindings)
    assert bindings == {0: sorts["i64"], 1: sorts["Unit"]}
    assert StructuralView(substitute_sort(sorts["V"], bindings)) == StructuralView(sorts["Unit"])


def test_bound_wrong(sorts: dict[str, Ref]) -> None:
    bindings = {0: sorts["Unit"]}
    before = bindings.copy()
    with pytest.raises(TypeConstraintError, match="Inconsistent sort for type parameter 0"):
        infer_sort(sorts["Map"], sorts["closed"], 2, bindings)
    assert bindings == before
