"""Builtin descriptors remain owned typing context, never input declarations."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import bindings
from egglog._catalog import Catalog, builtin_catalog
from egglog._presentation import builtin_owner
from egglog._program import Builder, pack
from egglog.runtime import RuntimeExpr, RuntimeFunction, _definition_at


def _provider_catalog(*, inputs: int = 1) -> pb.BuiltinCatalog:
    """One actual test provider scheme with independent native and Python names."""
    return pb.BuiltinCatalog(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="Number"))
            )
        ],
        primitive_sorts=[pb.PrimitiveSort(name="Number")],
        primitives=[
            pb.Primitive(
                name="provider.identity",
                signature=pb.GenericSignature(inputs=[0 for _ in range(inputs)], output=0),
            )
        ],
    )


def test_custom_environment_resolves_without_package_fallback_and_packs_no_descriptors() -> None:
    catalog = _provider_catalog()
    mapping = {
        "types": {"Number": {"path": ["example", "Number"]}},
        "callables": {
            "provider.identity": [
                {"form": "FUNCTION", "path": ["example", "identity"], "params": [{"coreInput": 0, "name": "value"}]}
            ]
        },
        "fragment": {},
    }
    native = builtin_owner(catalog, mapping=mapping)
    indexed = Catalog(native)
    reference, view = indexed.functions["example", "identity"]
    builder = Builder(environment=(native,))
    sort = builder.import_ref(native.ref("sorts", 0))
    variable = builder.add("nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["var"], str]("var", "value")))
    root = builder.publish().ref("nodes", variable)
    result = RuntimeFunction(reference, view)(RuntimeExpr(root))
    assert result is not None
    assert _definition_at(result.__egg_ref__, "callable", "provider.identity") == reference
    with pytest.raises(LookupError, match="owned canonical definition"):
        _definition_at(result.__egg_ref__, "callable", "egglog.core.i64.add")
    packed = pack([result.__egg_ref__]).program
    assert not (*packed.constructors, *packed.functions, *packed.relations, *packed.user_defined_primitives)
    assert len(packed.sorts) == 1
    assert packed.nodes[0].kind is not None
    assert packed.nodes[0].kind.field == "call"
    assert packed.nodes[0].kind.value.func == "provider.identity"
    detached = Builder.from_program(packed).publish().ref("nodes", 0)
    with pytest.raises(LookupError, match="owned canonical definition"):
        _definition_at(detached, "callable", "provider.identity")


def test_conflicting_provider_signatures_cannot_be_combined() -> None:
    left = Builder.from_builtins(_provider_catalog()).publish()
    right = Builder.from_builtins(_provider_catalog(inputs=2)).publish()
    with pytest.raises(ValueError, match="Conflicting builtin environments"):
        Builder(environment=(left, right)).publish()


def test_backend_wide_get_builtins_matches_the_packaged_native_catalog() -> None:
    transport = bindings._ProtoEngine()
    actual = pb.BuiltinCatalog.from_binary(transport.get_builtins(pb.GetBuiltinsRequest().to_binary()))
    assert actual.to_binary() == getattr(bindings, "_export_pyobject_catalog")()  # noqa: B009
    assert {primitive.name for primitive in actual.primitives} == {
        reference.read().name for reference in builtin_catalog().roots if reference.role == "primitives"
    }
