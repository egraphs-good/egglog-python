# mypy: allow-empty-bodies
# Ellipsis bodies are ordinary DSL declaration headers.
"""Canonical catalog indexing, not a replacement builtin signature inventory."""

from pathlib import Path
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import f64, i64, runtime
from egglog._catalog import Catalog
from egglog._presentation import (
    CallKind,
    CallPresentation,
    Parameter,
    Presentation,
    TypePresentation,
    presentation,
)
from egglog._program import Builder, StructuralView, pack
from egglog.pretty import pretty_ref
from egglog.runtime import RuntimeClass, RuntimeExpr, RuntimeFunction, _class_for_sort
from egglog.type_constraint_solver import TypeConstraintError


def _primitive_signature(primitive: pb.Primitive) -> pb.GenericSignature:
    """Narrow a real builtin descriptor before inspecting or corrupting its signature."""
    signature = primitive.signature
    assert signature is not None
    return signature


def _relocate_builtins(source: pb.BuiltinCatalog) -> pb.BuiltinCatalog:
    """Relocate every signature sort independently of descriptor and mapping order."""
    result = pb.BuiltinCatalog.from_binary(source.to_binary())
    remap = {index: len(source.sorts) - index - 1 for index in range(len(source.sorts))}
    result.sorts.reverse()
    for sort in result.sorts:
        kind = sort.kind
        assert kind is not None
        if kind.field == "primitive":
            kind.value.args = [remap[index] for index in kind.value.args]
        elif kind.field == "func":
            kind.value.params = [remap[index] for index in kind.value.params]
            kind.value.result = remap[kind.value.result]
    for primitive in result.primitives:
        signature = primitive.signature
        assert signature is not None
        signature.inputs = [remap[index] for index in signature.inputs]
        signature.varargs = [remap[index] for index in signature.varargs]
        assert signature.has_field("output")
        signature.output = remap[signature.output]
    result.primitives.reverse()
    result.primitive_sorts.reverse()
    return result


def catalog_fixture():
    return pb.BuiltinCatalog(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="Number"))
            )
        ],
        primitive_sorts=[pb.PrimitiveSort(name="Number")],
        primitives=[pb.Primitive(name="example.number.add", signature=pb.GenericSignature(inputs=[0, 0], output=0))],
    )


def number_presentations(owner):
    return {
        ("primitive_sort", "Number"): Presentation(type_view=TypePresentation(("example", "Number"))),
        ("callable", "example.number.add"): Presentation(
            views=(
                CallPresentation(
                    kind=CallKind.METHOD,
                    path=("__add__",),
                    owner=owner.ref("sorts", 0),
                    receiver=0,
                    params=(Parameter(core_input=1, name="other"),),
                ),
            )
        ),
    }


def test_catalog_indexes_records_without_evaluating_or_retyping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog loading cannot create an engine")

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)
    source = catalog_fixture()
    catalog = Catalog(Builder.from_builtins(source).publish(presentation_factory=number_presentations))
    family = catalog.types["example", "Number"]
    assert family.role == "primitive_sorts"
    assert family.read().name == "Number"
    primitive, view = catalog.members[("primitive_sort", "Number"), "__add__"]
    assert primitive.role == "primitives"
    assert primitive.read().name == "example.number.add"
    assert view == 0
    source.primitives.clear()
    assert _primitive_signature(primitive.read()).inputs[0] == 0
    catalog.reindex()
    assert catalog.members[("primitive_sort", "Number"), "__add__"] == (primitive, view)


def test_builtin_descriptors_and_local_views_never_enter_input_programs():
    source = catalog_fixture()
    encoded = source.to_binary()
    catalog = Catalog(Builder.from_builtins(source).publish(presentation_factory=number_presentations))
    with pytest.raises(TypeError, match="Builtin descriptors are ambient references, not Program roots"):
        pack(catalog.roots)
    packed = pack([(i64(1) + i64(2)).__egg_ref__]).program
    assert not (*packed.constructors, *packed.functions, *packed.relations, *packed.user_defined_primitives)
    assert packed.nodes
    assert not packed.commands
    assert source.to_binary() == encoded
    detached = Catalog(Builder.from_builtins(source).publish())
    assert not detached.members
    assert not presentation(detached.definitions["callable", "example.number.add"]).views


def test_local_scope_indexes_canonical_refs_without_adopting_them_again():
    program = catalog_fixture()
    source = Catalog(Builder.from_builtins(program).publish(presentation_factory=number_presentations))
    scoped = Catalog.from_refs([*source.roots, source.roots[0]])
    assert scoped.roots == source.roots
    assert scoped.members == source.members
    scoped.members.clear()
    scoped.reindex()
    assert scoped.members == source.members
    assert all(ref.owner is source.owner for ref in scoped.roots)


@pytest.mark.parametrize("generic", [False, True])
def test_nominal_and_primitive_sort_member_owners_recover_independently(generic):
    descriptors = catalog_fixture()
    signature = _primitive_signature(descriptors.primitives[0])
    signature.inputs = [0]
    if generic:
        descriptors.primitive_sorts[0].arity = 1
        signature.type_param_count = 1
        descriptors.sorts[0].kind.value.args = [1]
        descriptors.sorts.append(pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))

    def host_views(owner):
        result = number_presentations(owner)
        result["callable", "example.number.add"] = Presentation(
            views=(CallPresentation(kind=CallKind.METHOD, path=("tag",), owner=owner.ref("sorts", 0), receiver=0),)
        )
        return result

    ambient = Builder.from_builtins(descriptors).publish(presentation_factory=host_views)
    source = pb.Program(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef](
                    "primitive", pb.PrimitiveSortRef(name="Number", args=[1] if generic else [])
                )
            ),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Number")),
        ],
        nodes=[
            pb.Node(sort_id=0, kind=Oneof[Literal["var"], str]("var", "host")),
            pb.Node(sort_id=1, kind=Oneof[Literal["var"], str]("var", "nominal")),
        ],
        constructors=[pb.Constructor(name="nominal.tag", inputs=[1], output=1)],
    )

    def views(owner):
        return {
            ("eq_sort", "Number"): Presentation(type_view=TypePresentation(("example", "Nominal"))),
            ("callable", "nominal.tag"): Presentation(
                views=(CallPresentation(kind=CallKind.METHOD, path=("tag",), owner=owner.ref("sorts", 1), receiver=0),)
            ),
        }

    owner = Builder.from_program(source, environment=(ambient,)).publish(presentation_factory=views)
    catalog = Catalog.from_refs([*Catalog(ambient).roots, owner.ref("sorts", 1), owner.ref("constructors", 0)])
    host = _class_for_sort(owner.ref("sorts", 0))
    nominal = _class_for_sort(owner.ref("sorts", 1))
    assert host != nominal
    assert str(host) == ("Number[Nominal]" if generic else "Number")
    assert str(nominal) == "Nominal"
    for namespace, node, expected in [("primitive_sort", 0, "example.number.add"), ("eq_sort", 1, "nominal.tag")]:
        reference, _ = catalog.members[(namespace, "Number"), "tag"]
        tag = RuntimeExpr(owner.ref("nodes", node)).tag
        assert isinstance(tag, RuntimeFunction)
        tagged = tag()
        assert isinstance(tagged, RuntimeExpr)
        assert tagged.__egg_ref__.read().kind.value.func == expected
        assert (reference.read().name) == expected
    catalog.members.clear()
    catalog.reindex()
    host.__egg_attr_cache__.clear()
    nominal.__egg_attr_cache__.clear()
    nominal_tag = RuntimeExpr(owner.ref("nodes", 1)).tag
    assert isinstance(nominal_tag, RuntimeFunction)
    tagged = nominal_tag()
    assert isinstance(tagged, RuntimeExpr)
    assert tagged.__egg_ref__.read().kind.value.func == "nominal.tag"

    def colliding_views(local_owner):
        result = views(local_owner)
        result["eq_sort", "Number"] = Presentation(type_view=TypePresentation(("example", "Number")))
        return result

    colliding = Builder.from_program(source, environment=(ambient,)).publish(presentation_factory=colliding_views)
    with pytest.raises(ValueError, match="Python binding collision"):
        Catalog.from_refs([*Catalog(ambient).roots, colliding.ref("sorts", 1), colliding.ref("constructors", 0)])


@pytest.mark.parametrize("nominal", [False, True])
def test_class_and_pretty_recovery_never_use_other_sort_kind(nominal):
    descriptors = catalog_fixture()
    ambient = Builder.from_builtins(descriptors).publish(presentation_factory=number_presentations)
    source = pb.Program(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["eq"], str]("eq", "Number")
                if nominal
                else Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="Number"))
            )
        ],
    )
    owner = Builder.from_program(source, environment=(ambient,) if nominal else ()).publish()
    namespace = "eq_sort" if nominal else "primitive_sort"
    for recover in (_class_for_sort, pretty_ref):
        if nominal:
            assert str(recover(owner.ref("sorts", 0))) == "Number"
        else:
            with pytest.raises(LookupError, match=f"No owned canonical definition for {namespace} 'Number'"):
                recover(owner.ref("sorts", 0))


def test_catalog_rejects_ambiguous_presentation():
    source = catalog_fixture()
    duplicate = pb.Primitive.from_binary(source.primitives[0].to_binary())
    duplicate.name = "example.other.add"
    source.primitives.append(duplicate)

    def views(owner):
        result = number_presentations(owner)
        result["callable", duplicate.name] = result["callable", "example.number.add"]
        return result

    with pytest.raises(ValueError, match="binding collision"):
        Catalog(Builder.from_builtins(source).publish(presentation_factory=views))


@pytest.mark.parametrize("name", ["1.0", "not-a-python-identifier", "class"])
def test_runtime_type_names_preserve_exact_owned_metadata(name):
    source = catalog_fixture()
    path = ("example", name)

    def views(owner):
        result = number_presentations(owner)
        result["primitive_sort", "Number"] = Presentation(type_view=TypePresentation(path))
        if len(source.primitive_sorts) > 1:
            result["primitive_sort", "different.core"] = result["primitive_sort", "Number"]
        return result

    catalog = Catalog(Builder.from_builtins(source).publish(presentation_factory=views))
    reference = catalog.types[path]
    view = presentation(reference).type_view
    assert view is not None
    assert view.path[-1] == name
    with pytest.raises(TypeError, match="Builtin descriptors are ambient references, not Program roots"):
        pack(catalog.roots)
    restored = Catalog.from_refs(catalog.roots)
    assert presentation(restored.types[path]).type_view == view
    assert path not in Catalog(Builder.from_builtins(source).publish()).types
    source.primitive_sorts.append(pb.PrimitiveSort(name="different.core"))
    with pytest.raises(ValueError, match="Python binding collision"):
        Catalog(Builder.from_builtins(source).publish(presentation_factory=views))


@pytest.mark.parametrize(
    "path", [[], [""], ["example", ""], ["", "Number"], ["class", "Number"], ["bad-name", "Number"]]
)
def test_type_paths_keep_nonempty_terminal_and_identifier_qualifiers(path):
    source = catalog_fixture()

    def views(owner):
        result = number_presentations(owner)
        result["primitive_sort", "Number"] = Presentation(type_view=TypePresentation(tuple(path)))
        return result

    with pytest.raises(ValueError, match="Python path"):
        Catalog(Builder.from_builtins(source).publish(presentation_factory=views))


@pytest.mark.parametrize("name", ["", "123", "eval_mode_unsafe-seminaive", "class", "@internal"])
def test_runtime_free_function_names_preserve_exact_callee_and_core_order(monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog names and calls cannot create an engine")

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)
    ambient = Builder.from_builtins(
        pb.BuiltinCatalog.from_binary((Path(__file__).parents[1] / "egglog/_builtin_catalog.pb").read_bytes())
    ).publish()
    source = pb.Program(
        sorts=[
            pb.Sort(kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64")))
        ],
        functions=[pb.Function(name="exact.declared.table", inputs=[0, 0], output=0)],
    )
    surface = Presentation(
        views=(
            CallPresentation(
                kind=CallKind.FUNCTION,
                path=("example", name),
                params=(Parameter(core_input=1, name="right"), Parameter(core_input=0, name="left")),
            ),
        )
    )
    builder = Builder.from_program(source, environment=(ambient,))
    builder.presentations["callable", "exact.declared.table"] = surface
    owner = builder.publish()
    catalog = Catalog.from_refs([owner.ref("functions", 0)])
    reference, view = catalog.functions["example", name]
    result = RuntimeFunction(reference, view)(i64(9), left=i64(4))
    assert isinstance(result, RuntimeExpr)
    call = result.__egg_ref__.read().kind.value
    assert call.func == "exact.declared.table"
    assert [result.__egg_ref__.owner.ref("nodes", index).read().kind.value.value.value for index in call.args] == [4, 9]
    detached = pb.Program.from_binary(pack([result.__egg_ref__]).program.to_binary())
    detached_owner = Builder.from_program(detached, environment=(ambient,)).publish()
    assert ("example", name) not in Catalog(detached_owner).functions
    restored = Catalog.from_refs([reference])
    restored_reference, restored_view = restored.functions["example", name]
    assert restored_view == view
    assert StructuralView(restored_reference) == StructuralView(reference)
    restored.functions.clear()
    restored.reindex()
    assert restored.functions["example", name] == (restored_reference, restored_view)
    duplicate = pb.Function.from_binary(source.functions[0].to_binary())
    duplicate.name = "different.declared.table"
    source.functions.append(duplicate)
    builder = Builder.from_program(source, environment=(ambient,))
    for key in ("exact.declared.table", "different.declared.table"):
        builder.presentations["callable", key] = surface
    with pytest.raises(ValueError, match="binding collision"):
        Catalog(builder.publish())


@pytest.mark.parametrize("path", [[], ["", "f"], ["class", "f"], ["bad-name", "f"]])
def test_free_function_paths_keep_nonempty_identifier_qualifiers(path):
    source = catalog_fixture()

    def views(owner):
        result = number_presentations(owner)
        result["callable", "example.number.add"] = Presentation(
            views=(
                CallPresentation(
                    kind=CallKind.FUNCTION, path=tuple(path), params=(Parameter(0, "left"), Parameter(1, "right"))
                ),
            )
        )
        return result

    with pytest.raises(ValueError, match="Python path"):
        Catalog(Builder.from_builtins(source).publish(presentation_factory=views))


@pytest.mark.parametrize("kind", ["constructor", "function", "user_defined_primitive", "relation", "primitive"])
def test_constant_presentation_accepts_each_concrete_nullary_callable(kind):
    surface = Presentation(views=(CallPresentation(kind=CallKind.CONSTANT, path=("example", "c")),))
    if kind == "primitive":
        descriptors = pb.BuiltinCatalog(
            sorts=[
                pb.Sort(
                    kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))
                )
            ],
            primitive_sorts=[pb.PrimitiveSort(name="i64")],
            primitives=[
                pb.Primitive(
                    name="c",
                    signature=pb.GenericSignature(output=0),
                )
            ],
        )
        builder = Builder.from_builtins(descriptors)
        builder.presentations["callable", "c"] = surface
        owner = builder.publish()
        reference = owner.ref("primitives", 0)
    else:
        kinds = {
            "constructor": pb.Constructor(name="c", output=1),
            "function": pb.Function(name="c", output=0),
            "user_defined_primitive": pb.UserDefinedPrimitive(name="c", output=0, body=0),
            "relation": pb.Relation(name="c"),
        }
        role = {
            "constructor": "constructors",
            "function": "functions",
            "user_defined_primitive": "user_defined_primitives",
            "relation": "relations",
        }[kind]
        source = pb.Program(
            sorts=[
                pb.Sort(
                    kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="i64"))
                ),
                pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E")),
            ],
            nodes=[
                pb.Node(
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                    )
                )
            ],
        )
        getattr(source, role).append(kinds[kind])
        builder = Builder.from_program(source)
        builder.presentations["callable", "c"] = surface
        owner = builder.publish()
        reference = owner.ref(role, 0)
    catalog = Catalog.from_refs([reference])
    assert reference.read().name == "c"
    assert isinstance(reference.read(), pb.Primitive if kind == "primitive" else type(kinds[kind]))
    assert catalog.constants["example", "c"] == (reference, 0)
    catalog.constants.clear()
    catalog.reindex()
    assert catalog.constants["example", "c"] == (reference, 0)
    assert not catalog.functions


@pytest.mark.parametrize("invalid", ["input", "generic", "tail", "receiver", "owner", "params", "mutates"])
def test_constant_presentation_rejects_nonconstant_shapes(invalid):
    source = catalog_fixture()
    signature = _primitive_signature(source.primitives[0])
    signature.inputs.clear()
    if invalid == "input":
        signature.inputs.append(0)
    elif invalid == "generic":
        signature.type_param_count += 1
    elif invalid == "tail":
        signature.varargs.append(0)

    def views(owner):
        return {
            ("callable", "example.number.add"): Presentation(
                views=(
                    CallPresentation(
                        kind=CallKind.CONSTANT,
                        path=("constant",),
                        receiver=0 if invalid == "receiver" else None,
                        owner=owner.ref("sorts", 0) if invalid == "owner" else None,
                        params=(Parameter(0, "x"),) if invalid == "params" else (),
                        mutates=0 if invalid == "mutates" else None,
                    ),
                )
            )
        }

    expected = {
        "input": "Python view must map every core input exactly once",
        "tail": "Python view must map every core input exactly once",
        "receiver": "Tail receiver requires a homogeneous variadic method",
        "params": "Python view must map every core input exactly once",
        "mutates": "Invalid mutated input",
    }.get(invalid, r"Python (constant|member)")
    with pytest.raises(ValueError, match=expected):
        Catalog(Builder.from_builtins(source).publish(presentation_factory=views))


@pytest.mark.parametrize(
    ("path", "valid"),
    [
        ([], False),
        ([""], True),
        (["example", ""], True),
        (["", "c"], False),
        (["", ""], False),
        (["example", "", "c"], False),
    ],
)
def test_constant_path_allows_only_an_empty_final_name(path, valid):
    source = catalog_fixture()
    _primitive_signature(source.primitives[0]).inputs.clear()

    def views(owner):
        return {
            ("callable", "example.number.add"): Presentation(
                views=(CallPresentation(kind=CallKind.CONSTANT, path=tuple(path)),)
            )
        }

    if valid:
        assert tuple(path) in Catalog(Builder.from_builtins(source).publish(presentation_factory=views)).constants
    else:
        with pytest.raises(ValueError, match="Invalid Python constant path"):
            Catalog(Builder.from_builtins(source).publish(presentation_factory=views))


@pytest.mark.parametrize("relocated", [False, True])
def test_builtin_source_headers_resolve_owned_native_sort_addresses(native_catalog_program, monkeypatch, relocated):
    from egglog import BuiltinExpr, _catalog, egraph, method, runtime  # noqa: PLC0415  # noqa: PLC0415

    program = _relocate_builtins(native_catalog_program) if relocated else native_catalog_program
    catalog = Catalog(Builder.from_builtins(program).publish())
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(egraph, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", dict(runtime._BASE_CLASSES))

    class NativeInteger(BuiltinExpr, egg_sort="i64"):
        def __init__(self, value: int) -> None: ...

        @method(egg_fn="egglog.core.i64.add")
        def plus(self, rhs: "NativeInteger") -> "NativeInteger": ...

    result = NativeInteger(1).plus(rhs=NativeInteger(2))
    call = result.__egg_ref__.read().kind
    assert call is not None
    assert call.field == "call"
    assert call.value.func == "egglog.core.i64.add"
    assert [
        result.__egg_ref__.owner.ref("nodes", index).read().kind.value.value.value for index in call.value.args
    ] == [1, 2]


@pytest.mark.parametrize("corruption", ["input", "output", "arity", "generic"])
def test_builtin_source_headers_reject_corrupt_native_signatures_before_conversion(
    native_catalog_program, monkeypatch, corruption
):
    from egglog import BuiltinExpr, _catalog, egraph, method, runtime  # noqa: PLC0415  # noqa: PLC0415

    primitive = next(item for item in native_catalog_program.primitives if item.name == "egglog.core.i64.add")
    signature = _primitive_signature(primitive)
    foreign = next(
        index
        for index, sort in enumerate(native_catalog_program.sorts)
        if sort.kind is not None and sort.kind.field == "primitive" and sort.kind.value.name == "f64"
    )
    if corruption == "input":
        signature.inputs[1] = foreign
    elif corruption == "output":
        signature.output = foreign
    elif corruption == "arity":
        signature.inputs.pop()
    else:
        signature.type_param_count = 1
    catalog = Catalog(Builder.from_builtins(native_catalog_program).publish())
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(egraph, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", dict(runtime._BASE_CLASSES))
    events = []

    class Host:
        pass

    class NativeInteger(BuiltinExpr, egg_sort="i64"):
        def __init__(self, value: int) -> None: ...

        @method(egg_fn="egglog.core.i64.add")
        def plus(self, rhs: "NativeInteger") -> "NativeInteger": ...

    from egglog import converter  # noqa: PLC0415

    def convert(value):
        events.append(value)
        return NativeInteger(2)

    converter(Host, NativeInteger, convert)
    with pytest.raises((TypeError, ValueError, TypeConstraintError)):
        NativeInteger(1).plus(Host())  # type: ignore[arg-type]
    assert not events


def test_handwritten_builtin_defaults_share_only_their_authored_roots(class_authoring_without_engine):
    from egglog import py_eval, py_exec  # noqa: PLC0415
    from egglog.runtime import resolve_callable  # noqa: PLC0415

    roots = []
    for function in (py_eval, py_exec):
        view = presentation(resolve_callable(function)).views[0]
        first, second = [parameter.default for parameter in view.params[1:]]
        assert first is not None
        assert second == first
        node = first.read()
        assert node.kind is not None
        assert node.kind.field == "call"
        assert node.kind.value.func == "egglog.python.pyobject.dict"
        assert not node.kind.value.args
        roots.append(first)
    assert roots[0] != roots[1]


def test_catalog_export_runs_without_frontend_or_sdk_imports(tmp_path):
    import hashlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from egglog import bindings  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "catalog.pb"
    command = [
        sys.executable,
        "-c",
        """
import importlib.abc
import runpy
import sys
class RejectFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'egglog', 'egglog_proto'} or fullname.startswith(('egglog.', 'egglog_proto.')):
            raise AssertionError('Exporter imported frontend/SDK: ' + fullname)
sys.meta_path.insert(0, RejectFrontend())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
""",
        str(root / "scripts/export_builtin_catalog.py"),
        "--extension",
        bindings.__file__,
        "--output",
        str(output),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    expected = (root / "python/egglog/_builtin_catalog.pb").read_bytes()
    assert output.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() in result.stdout
    subprocess.run([*command, "--check"], check=True, capture_output=True, text=True)
    output.write_bytes(b"different catalog")
    failed = subprocess.run([*command, "--check"], check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert output.read_bytes() == b"different catalog"


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("arguments", ["same", "conflicting", "empty_tail"])
def test_classmethod_pretty_infers_owner_from_uniform_tail(native_catalog_program, monkeypatch, fixed, arguments):
    monkeypatch.setattr(runtime, "_BASE_CLASSES", dict(runtime._BASE_CLASSES))
    source = next(item for item in native_catalog_program.primitives if item.name == "egglog.core.vec.of")
    count = pb.Primitive.from_binary(source.to_binary())
    count.name = "test.vec.count"
    signature = _primitive_signature(count)
    owner_sort = signature.output
    signature.output = next(
        index
        for index, sort in enumerate(native_catalog_program.sorts)
        if sort.kind.field == "primitive" and sort.kind.value.name == "i64"
    )
    if fixed:
        signature.inputs.append(signature.varargs[0])
    native_catalog_program.primitives.append(count)

    def views(owner):
        return {
            ("primitive_sort", "Vec"): Presentation(type_view=TypePresentation(("egglog", "builtins", "Vec"), ("T",))),
            ("callable", count.name): Presentation(
                views=(
                    CallPresentation(
                        kind=CallKind.CLASS_METHOD,
                        path=("count",),
                        owner=owner.ref("sorts", owner_sort),
                        params=(Parameter(0, "first"), Parameter(1, "items")) if fixed else (Parameter(0, "items"),),
                    ),
                )
            ),
        }

    owner = Builder.from_builtins(native_catalog_program).publish(presentation_factory=views)
    family_index = next(
        index for index, item in enumerate(native_catalog_program.primitive_sorts) if item.name == "Vec"
    )
    family = RuntimeClass(owner.ref("primitive_sorts", family_index))
    target = family[i64]
    assert isinstance(target, RuntimeClass)
    count_function = target.count
    assert isinstance(count_function, RuntimeFunction)
    authored = count_function(i64(1), i64(2))
    assert isinstance(authored, RuntimeExpr)
    if arguments == "same":
        assert str(authored) == "Vec[i64].count(1, 2)"
        return
    if arguments == "empty_tail":
        expression = count_function(i64(1)) if fixed else count_function()
        assert isinstance(expression, RuntimeExpr)
        if fixed:
            assert str(expression) == "Vec[i64].count(1)"
        else:
            with pytest.raises(TypeConstraintError, match="Unresolved type variable: 0"):
                str(expression)
        return
    builder = Builder()
    builder.import_ref(authored.__egg_ref__)
    record = authored.__egg_ref__.read()
    record.sort_id = builder.import_ref(i64.__egg_sort__)  # type: ignore[attr-defined]
    record.kind.value.args = [builder.import_ref(item.__egg_ref__) for item in (i64(1), f64(2.0))]
    index = builder.add("nodes", record)
    malformed = RuntimeExpr(builder.publish().ref("nodes", index))
    with pytest.raises(TypeConstraintError, match="Inconsistent sort for type parameter 0"):
        str(malformed)
