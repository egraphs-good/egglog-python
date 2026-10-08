"""Canonical catalog indexing, not a replacement builtin signature inventory."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog._catalog import Catalog


def catalog_fixture() -> pb.Program:
    return pb.Program(
        ir_version=1,
        sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Number")))],
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family",
                    pb.HostSortFamily(
                        name="Number", bindings=pb.SortBindings(python=pb.TypeBinding(path=["example", "Number"]))
                    ),
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                    "host_primitive",
                    pb.HostPrimitive(
                        name="example.number.add",
                        typing=Oneof[Literal["signature"], pb.GenericSignature](
                            "signature", pb.GenericSignature(inputs=[pb.Arg(sort=0), pb.Arg(sort=0)], output=0)
                        ),
                    ),
                ),
                bindings=pb.CallableBindings(
                    python=pb.PythonBindings(
                        views=[
                            pb.PythonCallable(
                                kind=pb.PythonCallKind.METHOD,
                                path=["__add__"],
                                owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", 0)),
                                receiver=0,
                                params=[pb.PythonParameter(core_input=1, name="other")],
                            )
                        ]
                    )
                ),
            ),
        ],
    )


def test_catalog_indexes_records_without_evaluating_or_retyping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog loading cannot create an engine")

    monkeypatch.setattr("egglog.bindings.EGraph", forbidden)
    source = catalog_fixture()
    catalog = Catalog(source)
    family = catalog.types["example", "Number"]
    assert family.read().kind.value.name == "Number"
    declaration, view = catalog.members["Number", "__add__"]
    assert declaration.read().kind.value.name == "example.number.add"
    assert view == 0
    source.declarations.clear()
    assert declaration.read().kind.value.typing.value.inputs[0].sort == 0
    catalog.reindex()
    assert catalog.members["Number", "__add__"] == (declaration, view)


def test_explicit_index_does_not_persist_invented_python_bindings():
    source = catalog_fixture()
    source.declarations[1].bindings = None
    catalog = Catalog(source)
    assert not catalog.members
    declaration = catalog.definitions["callable", "example.number.add"].read()
    assert declaration.bindings is None


def test_local_scope_indexes_canonical_refs_without_adopting_them_again():
    source = Catalog(catalog_fixture())
    scoped = Catalog.from_refs([*source.roots, source.roots[0]])
    assert scoped.roots == source.roots
    assert scoped.members == source.members
    scoped.members.clear()
    scoped.reindex()
    assert scoped.members == source.members
    assert all(ref.owner is source.owner for ref in scoped.roots)


def test_catalog_rejects_ambiguous_presentation_and_invalid_paths():
    source = catalog_fixture()
    duplicate = pb.Declaration.from_binary(source.declarations[1].to_binary())
    assert duplicate.kind is not None
    assert duplicate.kind.field == "host_primitive"
    duplicate.kind.value.name = "example.other.add"
    source.declarations.append(duplicate)
    with pytest.raises(ValueError, match="binding collision"):
        Catalog(source)
    source = catalog_fixture()
    kind = source.declarations[0].kind
    assert kind is not None
    assert kind.field == "host_sort_family"
    assert kind.value.bindings is not None
    assert kind.value.bindings.python is not None
    kind.value.bindings.python.path = ["bad-name"]
    with pytest.raises(ValueError, match="Python path"):
        Catalog(source)
