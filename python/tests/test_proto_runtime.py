"""Record-native authoring gates for the single public runtime migration."""

from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog._program import Builder, Ref, StructuralView
from egglog.runtime import _author_call
from egglog.type_constraint_solver import TypeConstraintError, infer_sort, substitute_sort


def test_generic_inference_uses_binder_positions_and_is_transactional():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[0, 1])))
    )
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="String"))))
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[3, 4])))
    )
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[4, 3])))
    )
    owner = builder.publish()
    bindings: dict[int, Ref] = {}
    # GenericSignature labels may repeat: indices, not labels, identify binders.
    signature = pb.GenericSignature(type_params=["T", "T"], inputs=[pb.Arg(sort=2)], output=0)
    infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 5), len(signature.type_params), bindings)
    assert bindings == {0: owner.ref("sorts", 3), 1: owner.ref("sorts", 4)}
    assert StructuralView(substitute_sort(owner.ref("sorts", 2), bindings)) == StructuralView(owner.ref("sorts", 5))
    before = bindings.copy()
    with pytest.raises(TypeConstraintError):
        infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 6), len(signature.type_params), bindings)
    assert bindings == before
    independent: dict[int, Ref] = {}
    infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 6), len(signature.type_params), independent)
    assert independent == {0: owner.ref("sorts", 4), 1: owner.ref("sorts", 3)}


def test_sort_inference_rejects_missing_binders_arity_and_open_actuals():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Box")))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[1], result=1))))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(result=1))))
    owner = builder.publish()
    with pytest.raises(TypeConstraintError, match="binder"):
        infer_sort(owner.ref("sorts", 0), owner.ref("sorts", 1), 0, {})
    with pytest.raises(TypeConstraintError, match="closed"):
        infer_sort(owner.ref("sorts", 0), owner.ref("sorts", 0), 1, {})
    with pytest.raises(TypeConstraintError, match="arity"):
        infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 3), 0, {})
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        substitute_sort(owner.ref("sorts", 0), {})


def test_sort_inference_and_substitution_are_iterative():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Leaf")))
    pattern, concrete = 0, 1
    for _ in range(10_000):
        pattern = builder.add(
            "sorts",
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[pattern]))),
        )
        concrete = builder.add(
            "sorts",
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[concrete]))),
        )
    owner = builder.publish()
    bindings: dict[int, Ref] = {}
    infer_sort(owner.ref("sorts", pattern), owner.ref("sorts", concrete), 1, bindings)
    result = substitute_sort(owner.ref("sorts", pattern), bindings)
    assert StructuralView(result) == StructuralView(owner.ref("sorts", concrete))


def test_call_authoring_binds_keywords_defaults_and_core_input_order():
    builder = Builder()
    sort = builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    default = builder.add(
        "nodes",
        pb.Node(
            sort_id=sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 4))
            ),
        ),
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="sample.subtract",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature", pb.GenericSignature(inputs=[pb.Arg(sort=sort), pb.Arg(sort=sort)], output=sort)
                    ),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.FUNCTION,
                            path=["sample", "subtract_reversed"],
                            params=[
                                pb.PythonParameter(core_input=1, name="right"),
                                pb.PythonParameter(core_input=0, name="left", default_expr=default),
                            ],
                        )
                    ]
                )
            ),
        ),
    )
    owner = builder.publish()

    def convert(expected: Ref, value: object) -> Ref:
        assert isinstance(value, int)
        literal = Builder()
        literal.add(
            "nodes",
            pb.Node(
                sort_id=literal.import_ref(expected),
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
        )
        return literal.publish().ref("nodes", 0)

    root, mutated = _author_call(
        owner.ref("declarations", declaration), 0, (), {"right": 9}, node_of=lambda _: None, convert=convert
    )
    assert mutated is None
    node = root.read()
    assert node.kind.value.func == "sample.subtract"
    args = [root.owner.ref("nodes", index) for index in node.kind.value.args]
    assert [arg.read().kind.value.value.value for arg in args] == [4, 9]
    assert args[0] != owner.ref("nodes", default)
    with pytest.raises(TypeError, match="unexpected keyword"):
        _author_call(
            owner.ref("declarations", declaration),
            0,
            (),
            {"right": 9, "bogus": 9},
            node_of=lambda _: None,
            convert=convert,
        )


def test_generic_empty_call_requires_result_owner_and_keeps_signature_unchanged():
    builder = Builder()
    variable = builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    pattern = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[variable])))
    )
    scalar = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    )
    concrete = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[scalar])))
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="sample.vec.of",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature",
                        pb.GenericSignature(type_params=["T"], output=pattern, varargs=pb.Arg(sort=variable)),
                    ),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.INITIALIZER,
                            owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", pattern)),
                            params=[pb.PythonParameter(core_input=0, name="args")],
                        )
                    ]
                )
            ),
        ),
    )
    owner = builder.publish()
    ref = owner.ref("declarations", declaration)
    before = ref.read().to_binary()

    def forbidden(*args):
        pytest.fail("An empty constructor must not convert an argument")

    root, mutated = _author_call(
        ref, 0, (), {}, node_of=forbidden, convert=forbidden, owner=owner.ref("sorts", concrete)
    )
    assert mutated is None
    assert list(root.read().kind.value.args) == []
    assert StructuralView(root.owner.ref("sorts", root.read().sort_id)) == StructuralView(owner.ref("sorts", concrete))
    assert ref.read().to_binary() == before
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        _author_call(ref, 0, (), {}, node_of=forbidden, convert=forbidden)
