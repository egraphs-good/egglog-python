"""Canonical catalog indexing, not a replacement builtin signature inventory."""

import builtins
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import f64, i64
from egglog._catalog import Catalog
from egglog._program import Builder
from egglog.pretty import pretty_ref
from egglog.runtime import RuntimeExpr, RuntimeFunction, _class_for_sort
from egglog.type_constraint_solver import TypeConstraintError


def test_set_generated_surface_from_actual_catalog(set_catalog_program):
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(set_catalog_program)
    body = stub.split("class Set(", 1)[1].split("\nclass ", 1)[0]
    for signature in (
        "def __init__(self, *args: _Set_T_0) -> None:",
        "def empty(cls) -> Set[_Set_T_0]:",
        "def value(self) -> set[_Set_T_0]:",
        "def __iter__(self) -> Iterator[_Set_T_0]:",
        "def __len__(self) -> int:",
        "def __contains__(self, item: _Set_T_0) -> bool:",
        "def contains(self, value: _Set_T_0) -> Unit:",
        "def not_contains(self, value: _Set_T_0) -> Unit:",
        "def insert(self, value: _Set_T_0) -> Set[_Set_T_0]:",
        "def remove(self, value: _Set_T_0) -> Set[_Set_T_0]:",
        "def length(self) -> i64:",
        "def __or__(self, other: Set[_Set_T_0]) -> Set[_Set_T_0]:",
        "def __sub__(self, other: Set[_Set_T_0]) -> Set[_Set_T_0]:",
        "def __and__(self, other: Set[_Set_T_0]) -> Set[_Set_T_0]:",
        "def eval(self) -> set[_Set_T_0]:",
    ):
        assert signature in body
    assert "SetLike: TypeAlias = Set[_Set_T_0] | set[_Set_HostT]" in stub
    assert "def get(" not in body
    assert "def __ror__(" not in body
    assert "def __getitem__(" not in body
    declarations = {d.kind.value.name: d for d in set_catalog_program.declarations}
    assert len([name for name in declarations if name.startswith("egglog.core.set.")]) == 11
    declarations["egglog.core.set.insert"].bindings.python.views[0].params[0].name = "item"
    declarations["Set"].kind.value.bindings.python.path[-1] = "FiniteSet"
    stub = render(set_catalog_program)
    assert "def insert(self, item: _FiniteSet_T_0) -> FiniteSet[_FiniteSet_T_0]:" in stub
    assert "FiniteSetLike: TypeAlias = FiniteSet[_FiniteSet_T_0] | set[_FiniteSet_HostT]" in stub


@pytest.mark.parametrize("name", ["set", "SetLike", "_Set_HostT"])
def test_set_generated_protocol_name_collisions(set_catalog_program, name):
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    family = next(d.kind.value for d in set_catalog_program.declarations if d.kind.value.name == "Vec")
    family.bindings.python.path[-1] = name
    with pytest.raises(ValueError, match="collision"):
        render(set_catalog_program)


def test_set_stub_uses_relocated_actual_sort_records(set_catalog_program):
    from egglog._program import pack  # noqa: PLC0415

    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    owner = Builder.from_program(set_catalog_program).publish()
    relocated = pack([
        owner.ref("declarations", index) for index in reversed(range(len(set_catalog_program.declarations)))
    ]).program
    assert relocated.sorts != set_catalog_program.sorts
    original = render(set_catalog_program)
    reordered = render(relocated)
    for line in original.splitlines():
        if "_Set_" in line or "class Set(" in line:
            assert line in reordered


def test_exact_generated_surface_keeps_native_constructors(rational_catalog_program):
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(rational_catalog_program)
    for signature in (
        "def __init__(self, num: i64Like, den: i64Like) -> None:",
        "def __add__(self, other: RationalLike) -> Rational:",
        "def __radd__(self, other: RationalLike) -> Rational:",
        "def __init__(self, value: i64Like) -> None:",
        "def from_string(cls, s: StringLike) -> BigInt:",
        "def __init__(self, num: BigIntLike, den: BigIntLike) -> None:",
        "RationalLike: TypeAlias = Rational | Fraction | i64Like",
        "BigIntLike: TypeAlias = BigInt | i64Like",
        "BigRatLike: TypeAlias = BigRat | Fraction | i64Like",
    ):
        assert signature in stub
    for family, value, hook in (
        ("Rational", "Fraction", "__float__"),
        ("BigRat", "Fraction", "__float__"),
        ("BigInt", "int", "__index__"),
    ):
        body = stub.split(f"class {family}(", 1)[1].split("\nclass ", 1)[0]
        assert body.count("def __init__(") == 1
        assert f"def value(self) -> {value}:" in body
        assert f"def eval(self) -> {value}:" in body
        assert f"def {hook}(self) ->" in body
        assert "from_value" not in body
    declaration = next(
        d for d in rational_catalog_program.declarations if d.kind.value.name == "egglog.experimental.rational.make"
    )
    declaration.bindings.python.views[0].params[0].name = "numerator"
    assert "def __init__(self, numerator: i64Like, den: i64Like)" in render(rational_catalog_program)


def test_scalar_generated_surface_uses_actual_variadic_providers():
    root = Path(__file__).resolve().parents[2]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declarations = {d.kind.value.name: d for d in program.declarations}
    for name, paths in (("string.concat", ["join", "__add__"]), ("bool.or", ["__or__"])):
        declaration = declarations[f"egglog.core.{name}"]
        signature = declaration.kind.value.typing.value
        assert not signature.inputs
        assert len(signature.varargs) == 1
        assert [v.path[-1] for v in declaration.bindings.python.views] == paths
        assert declaration.bindings.python.views[-1].tail_repetitions == 2
    stub = render(program)
    for signature in (
        "def join(*strings: StringLike) -> String:",
        "def __add__(self, other: StringLike) -> String:",
        "def __radd__(self, other: StringLike) -> String:",
        "def replace(self, old: StringLike, new: StringLike) -> String:",
        "def to_string(self) -> String:",
        "def __invert__(self) -> Bool:",
        "def __or__(self, other: BoolLike) -> Bool:",
        "def __ror__(self, other: BoolLike) -> Bool:",
        "def implies(self, other: BoolLike) -> Bool:",
    ):
        assert signature in stub
    unit = stub.split("class Unit(", 1)[1].split("class ", 1)[0]
    assert "def __bool__(self) -> bool:" in unit
    assert "value" not in unit
    assert "eval" not in unit
    assert "UnitLike" not in stub
    declarations["egglog.core.string.concat"].bindings.python.views[0].path[-1] = "concatenate"
    assert "def concatenate(*strings: StringLike) -> String:" in render(program)


@pytest.mark.parametrize(
    "mutation", ["fixed", "missing", "duplicate", "owner", "default", "mutation", "huge", "property"]
)
def test_scalar_specialized_stub_rejects_invalid_views(mutation):
    root = Path(__file__).resolve().parents[2]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declaration = next(d for d in program.declarations if d.kind.value.name == "egglog.core.bool.or")
    signature, view = declaration.kind.value.typing.value, declaration.bindings.python.views[0]
    if mutation == "fixed":
        signature.inputs = signature.varargs * 2
        signature.varargs.clear()
    elif mutation == "missing":
        view.params.clear()
    elif mutation == "duplicate":
        view.params[0].core_input = view.receiver
    elif mutation == "owner":
        view.owner.kind = Oneof[Literal["sort"], int](
            "sort",
            next(i for i, s in enumerate(program.sorts) if s.kind.field == "family" and s.kind.value.name == "String"),
        )
    elif mutation == "default":
        view.params[0].default_expr = 0
    elif mutation == "mutation":
        view.mutates = 0
    elif mutation == "property":
        view.kind = pb.PythonCallKind.PROPERTY
        view.params.clear()
        view.tail_repetitions = 1
    else:
        view.tail_repetitions = 2**32 - 1
    with pytest.raises(ValueError, match=r"signature|input|owner|Default|view"):
        render(program)


@pytest.mark.parametrize("mutation", ["generic", "foreign", "collision"])
def test_scalar_free_stub_rejects_unsupported_views(mutation):
    root = Path(__file__).resolve().parents[2]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declaration = next(d for d in program.declarations if d.kind.value.name == "egglog.core.string.concat")
    view = declaration.bindings.python.views[0]
    if mutation == "generic":
        declaration.kind.value.typing.value.type_params.append("T")
    elif mutation == "foreign":
        view.path[0] = "other"
    else:
        view.path[-1] = "String"
    with pytest.raises(ValueError, match=r"free functions|collision"):
        render(program)


def test_map_generated_surface_from_actual_catalog():
    root = Path(__file__).resolve().parents[2]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = render(program)
    assert "def from_pairs(cls, entries: Vec[Pair[_Map_K_0, _Map_V_1]]) -> Map[_Map_K_0, _Map_V_1]:" in stub
    assert "def empty(cls) -> Map[_Map_K_0, _Map_V_1]:" in stub
    assert "def insert(self, key: _Map_K_0, value: _Map_V_1) -> Map[_Map_K_0, _Map_V_1]:" in stub
    assert "def __getitem__(self, key: _Map_K_0) -> _Map_V_1:" in stub
    assert "def value(self) -> dict[_Map_K_0, _Map_V_1]:" in stub
    assert "def __iter__(self) -> Iterator[_Map_K_0]:" in stub
    assert "def __contains__(self, key: _Map_K_0) -> bool:" in stub
    assert "MapLike: TypeAlias = Map[_Map_K_0, _Map_V_1] | dict[_Map_HostK, _Map_HostV]" in stub


def test_absent_initializer_keeps_nonabstract_type_token(vec_catalog_program):
    of = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.vec.of")
    of.bindings.python = None
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(vec_catalog_program)
    vec = stub.split("class Vec(", 1)[1].split("class ", 1)[0]
    assert "def __init__(self, _unavailable: Never, /) -> None:" in vec
    assert "abstractmethod" not in stub
    assert "def __init__(self, value: int, /)" in stub
    assert "def __init__(self, left: _Pair_L_0, right: _Pair_R_1)" in stub


def test_vec_uniform_tail_and_classmethod_stub_from_native_signatures(vec_catalog_program):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = renderer(vec_catalog_program)
    assert "def __init__(self, *args: _Vec_T_0) -> None:" in stub
    assert "@classmethod\n    def empty(cls) -> Vec[_Vec_T_0]:" in stub
    assert "def __getitem__(self, index: i64Like) -> _Vec_T_0:" in stub
    assert "def value(self) -> tuple[_Vec_T_0, ...]:" in stub
    assert "def __iter__(self) -> Iterator[_Vec_T_0]:" in stub
    assert "def __len__(self) -> int:" in stub
    assert "def __contains__(self, item: _Vec_T_0) -> bool:" in stub
    assert "def eval(self) -> tuple[_Vec_T_0, ...]:" in stub
    assert "def range(self) -> Vec[i64]:" in stub
    assert "VecLike: TypeAlias = Vec[_Vec_T_0] | tuple[_Vec_HostT, ...] | list[_Vec_HostT]" in stub
    declarations = {d.kind.value.name: d for d in vec_catalog_program.declarations}
    declarations["Vec"].kind.value.bindings.python.type_params = ["Element"]
    declarations["egglog.core.vec.of"].bindings.python.views[0].params[0].name = "items"
    declarations["egglog.core.vec.empty"].bindings.python.views[0].path = ["vacant"]
    renamed = renderer(vec_catalog_program)
    assert "def __init__(self, *items: _Vec_Element_0) -> None:" in renamed
    assert "def vacant(cls) -> Vec[_Vec_Element_0]:" in renamed


@pytest.mark.parametrize(
    "mutation", ["grouped", "default", "missing", "duplicate", "nonfinal", "receiver", "tail_receiver"]
)
def test_vec_static_tail_rejects_unrepresentable_or_invalid_views(vec_catalog_program, mutation):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    of = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.vec.of")
    signature, view = of.kind.value.typing.value, of.bindings.python.views[0]
    if mutation == "grouped":
        signature.varargs.append(pb.Arg(sort=signature.varargs[0].sort))
    elif mutation == "default":
        view.params[0].default_expr = 0
    elif mutation == "missing":
        view.params.clear()
    elif mutation == "duplicate":
        view.params.append(pb.PythonParameter(core_input=0, name="again"))
    elif mutation == "nonfinal":
        signature.inputs.append(pb.Arg(sort=signature.varargs[0].sort))
        view.params[0].core_input = 1
        view.params.append(pb.PythonParameter(core_input=0, name="first"))
    elif mutation == "receiver":
        view.receiver = 0
    else:
        view.kind = pb.PythonCallKind.METHOD
        view.path = ["tail_receiver"]
        view.receiver = 0
        view.params.clear()
    with pytest.raises(ValueError, match=r"signature|parameter|input|view"):
        renderer(vec_catalog_program)


def test_uniform_tail_can_follow_fixed_inputs(vec_catalog_program):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    of = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.vec.of")
    signature, view = of.kind.value.typing.value, of.bindings.python.views[0]
    signature.inputs.append(pb.Arg(sort=signature.varargs[0].sort, name="first"))
    view.params[0].core_input = 1
    view.params.insert(0, pb.PythonParameter(core_input=0, name="first"))
    assert "def __init__(self, first: _Vec_T_0, *args: _Vec_T_0) -> None:" in renderer(vec_catalog_program)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("arguments", ["same", "conflicting", "empty_tail"])
def test_classmethod_pretty_infers_owner_from_uniform_tail(vec_catalog_program, request, fixed, arguments):
    """Test-only combined view shape, not a claim about native Vec primitives."""
    source = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.vec.of")
    count = pb.Declaration.from_binary(source.to_binary())
    count.kind.value.name = "test.vec.count"
    signature, view = count.kind.value.typing.value, count.bindings.python.views[0]
    signature.output = next(
        i
        for i, sort in enumerate(vec_catalog_program.sorts)
        if sort.kind is not None and sort.kind.field == "family" and sort.kind.value.name == "i64"
    )
    view.kind = pb.PythonCallKind.CLASS_METHOD
    view.path = ["count"]
    view.params = [pb.PythonParameter(core_input=int(fixed), name="items")]
    if fixed:
        signature.inputs.append(pb.Arg(sort=signature.varargs[0].sort, name="first"))
        view.params.insert(0, pb.PythonParameter(core_input=0, name="first"))
    vec_catalog_program.declarations.append(count)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    prefix = "first: _Vec_T_0, " if fixed else ""
    assert f"def count(cls, {prefix}*items: _Vec_T_0) -> i64:" in renderer(vec_catalog_program)
    Vec = request.getfixturevalue("vec_runtime")
    authored = Vec[i64].count(i64(1), i64(2))
    if arguments == "same":
        assert str(authored) == "Vec[i64].count(1, 2)"
        return
    if arguments == "empty_tail":
        expression = Vec[i64].count(i64(1)) if fixed else Vec[i64].count()
        if fixed:
            assert str(expression) == "Vec[i64].count(1)"
        else:
            # The owner helped author this call, but is not a semantic wire edge.
            with pytest.raises(TypeConstraintError, match="Unresolved type variable: 0"):
                str(expression)
        return
    # Bypass call authoring's sort checks to test the presentation reader itself.
    builder = Builder()
    builder.import_ref(authored.__egg_ref__)
    record = authored.__egg_ref__.read()
    record.sort_id = builder.import_ref(i64.__egg_sort__)
    record.kind.value.args = [builder.import_ref(item.__egg_ref__) for item in (i64(1), f64(2.0))]
    index = builder.add("nodes", record)
    malformed = RuntimeExpr(builder.publish().ref("nodes", index))
    with pytest.raises(TypeConstraintError, match="Inconsistent sort for type parameter 0"):
        str(malformed)


def test_pair_generic_stub_uses_native_signature_and_presentation(pair_catalog_program):
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = renderer(pair_catalog_program)
    assert "def __init__(self, left: _Pair_L_0, right: _Pair_R_1) -> None:" in stub
    assert "@property\n    def left(self) -> _Pair_L_0:" in stub
    assert "@property\n    def right(self) -> _Pair_R_1:" in stub
    assert "def value(self) -> tuple[_Pair_L_0, _Pair_R_1]:" in stub
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    declarations["Pair"].kind.value.bindings.python.type_params = ["X", "Y"]
    declarations["egglog.core.pair.make"].bindings.python.views[0].params[0].name = "head"
    declarations["egglog.core.pair.first"].bindings.python.views[0].path = ["head"]
    renamed = renderer(pair_catalog_program)
    assert "def __init__(self, head: _Pair_X_0, right: _Pair_Y_1) -> None:" in renamed
    assert "def head(self) -> _Pair_X_0:" in renamed


def test_generated_stub_matches_catalog_without_frontend_imports(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    original_import = builtins.__import__

    def reject_frontend(name, *args, **kwargs):
        if name == "egglog" or name.startswith("egglog."):
            pytest.fail(f"Generator imported frontend: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_frontend)
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    encoded = (root / "python/egglog/_builtin_catalog.pb").read_bytes()
    program = pb.Program.from_binary(encoded)
    assert renderer(program) == (root / "python/egglog/builtins.pyi").read_text()
    assert program.to_binary() == encoded


def test_stub_generation_uses_canonical_ordinary_method_view():
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(d for d in program.declarations if d.kind.value.name == "egglog.core.i64.add")
    view = addition.bindings.python.views[0]
    view.path = ["plus"]
    view.params[0].name = "rhs"
    assert "def plus(self, rhs: i64Like) -> i64:" in renderer(program)
    owner = Builder.from_program(program).publish()
    method = RuntimeFunction(owner.ref("declarations", program.declarations.index(addition)), 0, i64(1))
    result = method(rhs=i64(2))
    argument = result.__egg_ref__.read().kind.value.args[1]
    assert RuntimeExpr(result.__egg_ref__.owner.ref("nodes", argument)).value == 2


@pytest.mark.parametrize(
    "name", ["int", "_Vec_T_0", "AttributeError", "tuple", "dict", "Never", "_Map_HostK", "MapLike"]
)
def test_stub_generation_rejects_host_and_internal_name_collisions(name):
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    scalar = next(d.kind.value for d in program.declarations if d.kind.value.name == "i64")
    scalar.bindings.python.path[-1] = name
    with pytest.raises(ValueError, match="Python static name collision"):
        renderer(program)
    scalar.bindings.python.path[-1] = "Integer"
    assert "class Integer(BuiltinExpr):" in renderer(program)


@pytest.mark.parametrize("mutation", ["output", "receiver", "params", "owner_order", "unknown_binder", "default"])
def test_pair_stub_rejects_invalid_or_unadmitted_generic_views(pair_catalog_program, mutation):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    make = declarations["egglog.core.pair.make"]
    first = declarations["egglog.core.pair.first"]
    signature = make.kind.value.typing.value
    if mutation == "output":
        signature.output = signature.inputs[0].sort
    elif mutation == "receiver":
        first.bindings.python.views[0].clear_field("receiver")
    elif mutation == "params":
        first.bindings.python.views[0].params = [pb.PythonParameter(name="extra", core_input=0)]
    elif mutation == "owner_order":
        pair_catalog_program.sorts[signature.output].kind.value.args.reverse()
    elif mutation == "unknown_binder":
        signature.type_params.append("Unused")
    else:
        make.bindings.python.views[0].params[0].default_expr = 0
    with pytest.raises(ValueError, match=r"initializer output|static view|static owner|Default root"):
        renderer(pair_catalog_program)


def test_pair_stub_uses_sort_structure_after_relocation(pair_catalog_program):
    from egglog._program import pack  # noqa: PLC0415

    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    make = declarations["egglog.core.pair.make"]
    owner_sort = pair_catalog_program.sorts[make.kind.value.typing.value.output]
    duplicated = pb.Sort.from_binary(owner_sort.to_binary())
    duplicated.kind.value.args = [len(pair_catalog_program.sorts) + 1, len(pair_catalog_program.sorts) + 2]
    make.bindings.python.views[0].owner.kind = Oneof[Literal["sort"], int]("sort", len(pair_catalog_program.sorts))
    pair_catalog_program.sorts.extend([
        duplicated,
        *(pb.Sort.from_binary(pair_catalog_program.sorts[i].to_binary()) for i in owner_sort.kind.value.args),
    ])
    before = renderer(pair_catalog_program)
    owner = Builder.from_program(pair_catalog_program).publish()
    relocated = pack([
        owner.ref("declarations", i) for i in reversed(range(len(pair_catalog_program.declarations)))
    ]).program
    after = renderer(relocated)
    for line in before.splitlines():
        if "_Pair_" in line or "class Pair(" in line:
            assert line in after
    first = declarations["egglog.core.pair.first"]
    first.kind.value.typing.value.output = make.kind.value.typing.value.inputs[1].sort
    assert "def left(self) -> _Pair_R_1:" in renderer(pair_catalog_program)


def test_pair_stub_rejects_specialized_owner_as_universal_class_view(pair_catalog_program):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    for name in ("first", "second"):
        declarations[f"egglog.core.pair.{name}"].bindings.python = None
    make = declarations["egglog.core.pair.make"]
    signature = make.kind.value.typing.value
    signature.type_params.clear()
    i64_sort = next(
        i for i, s in enumerate(pair_catalog_program.sorts) if s.kind.field == "family" and s.kind.value.name == "i64"
    )
    signature.inputs[0].sort = i64_sort
    signature.inputs[1].sort = i64_sort
    pair_catalog_program.sorts[signature.output].kind.value.args = [i64_sort, i64_sort]
    with pytest.raises(ValueError, match="ordered supported family parameters"):
        renderer(pair_catalog_program)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("receiver", "cover each input exactly once"),
        ("tail", "fixed concrete signature"),
        ("collision", "Python member collision"),
        ("missing_output", "fixed concrete signature"),
        ("application", "Function application static generation"),
        ("numeric_keyword", "Numeric forwarding requires one required 'other' parameter"),
        ("numeric_default", "Numeric forwarding requires one required 'other' parameter"),
    ],
)
def test_stub_generation_rejects_invalid_or_unmigrated_shapes(mutation, message):
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(d for d in program.declarations if d.kind.value.name == "egglog.core.i64.add")
    signature = addition.kind.value.typing.value
    view = addition.bindings.python.views[0]
    if mutation == "receiver":
        view.receiver = 1
    elif mutation == "tail":
        signature.varargs = [pb.Arg(sort=1), pb.Arg(sort=0)]
    elif mutation == "collision":
        addition.bindings.python.views.append(pb.PythonCallable.from_binary(view.to_binary()))
    elif mutation == "missing_output":
        signature.clear_field("output")
    elif mutation == "numeric_keyword":
        view.params[0].name = "rhs"
    elif mutation == "numeric_default":
        program.nodes.append(
            pb.Node(
                sort_id=signature.inputs[1].sort,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 2))
                ),
            )
        )
        view.params[0].default_expr = 0
    else:
        addition.kind.value.typing = Oneof[Literal["application"], pb.FunctionApplication](
            "application", pb.FunctionApplication()
        )
    with pytest.raises(ValueError, match=message):
        renderer(program)


@pytest.mark.parametrize(
    ("method", "argument", "output"),
    [("__lshift__", "f64", "i64"), ("__lt__", "i64", "Unit"), ("__lt__", "i64", "i64")],
)
def test_numeric_forward_signature_does_not_imply_reflection(method, argument, output):
    """Mutated descriptor shapes test emission policy, not native availability."""
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(
        d for d in program.declarations if d.kind is not None and d.kind.value.name == "egglog.core.i64.add"
    )
    assert addition.kind is not None
    assert isinstance(addition.kind.value, pb.HostPrimitive)
    typing = addition.kind.value.typing
    assert typing is not None
    assert isinstance(typing.value, pb.GenericSignature)
    signature = typing.value
    sorts = {
        s.kind.value.name: i
        for i, s in enumerate(program.sorts)
        if s.kind is not None and isinstance(s.kind.value, pb.HostSort)
    }
    signature.inputs[1].sort = sorts[argument]
    signature.output = sorts[output]
    for declaration in program.declarations:
        if declaration is not addition and declaration.bindings is not None and declaration.bindings.python is not None:
            declaration.bindings.python.views[:] = [
                view for view in declaration.bindings.python.views if view.path != [method]
            ]
    assert addition.bindings is not None
    assert addition.bindings.python is not None
    addition.bindings.python.views[0].path = [method]
    stub = renderer(program)
    body = stub.split("class i64(", 1)[1].split("\nclass ", 1)[0]
    assert f"def {method}(self, other: {argument}Like) -> {output}:" in body
    assert f"def __r{method[2:]}(" not in body


def test_bignum_generated_surface_uses_actual_forward_signatures(rational_catalog_program):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.kind.value.name: d for d in rational_catalog_program.declarations}
    assert len([name for name in declarations if name.startswith("egglog.core.bigrat.")]) == 24
    assert len([name for name in declarations if name.startswith("egglog.core.bigint.")]) == 26
    stub = renderer(rational_catalog_program)
    integer = stub.split("class BigInt(", 1)[1].split("\nclass ", 1)[0]
    ratio = stub.split("class BigRat(", 1)[1].split("\nclass ", 1)[0]
    for member in ("numer", "denom"):
        assert f"@property\n    def {member}(self) -> BigInt:" in ratio
        view = declarations[f"egglog.core.bigrat.{member}"].bindings.python.views[0]
        assert view.kind == pb.PythonCallKind.PROPERTY
        assert view.receiver == 0
    for member in ("__lshift__", "__rshift__"):
        assert f"def {member}(self, other: i64Like) -> BigInt:" in integer
        assert f"def __r{member[2:]}(" not in integer
    for body, family in ((integer, "BigInt"), (ratio, "BigRat")):
        for member in ("__lt__", "__gt__", "__le__", "__ge__"):
            assert f"def {member}(self, other: {family}Like) -> Unit:" in body
            assert f"def __r{member[2:]}(" not in body
        for member in ("__add__", "__sub__", "__mul__", "__truediv__"):
            assert f"def {member}(self, other: {family}Like) -> {family}:" in body
            assert f"def __r{member[2:]}(self, other: {family}Like) -> {family}:" in body


def test_stub_defaults_are_inert_and_presence_controls_arity():
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(d for d in program.declarations if d.kind.value.name == "egglog.core.i64.add")
    addition.bindings.python.views[0].path = ["plus"]
    parameter = addition.bindings.python.views[0].params[0]
    scalar_sort = addition.kind.value.typing.value.inputs[parameter.core_input].sort
    program.nodes.extend([
        pb.Node(
            sort_id=scalar_sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 2))
            ),
        ),
        pb.Node(
            sort_id=scalar_sort,
            kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.i64.add", args=[0, 0])),
        ),
    ])
    parameter.default_expr = 1
    assert "def plus(self, other: i64Like = ...) -> i64:" in renderer(program)
    owner = Builder.from_program(program).publish()
    method = RuntimeFunction(owner.ref("declarations", program.declarations.index(addition)), 0, i64(1))
    result = method()
    argument = result.__egg_ref__.read().kind.value.args[1]
    assert result.__egg_ref__.owner.ref("nodes", argument).read().kind.value.func == "egglog.core.i64.add"
    parameter.clear_field("default_expr")
    assert "other: i64Like = ..." not in renderer(program)
    parameter.default_expr = 2
    with pytest.raises(ValueError, match="Default root"):
        renderer(program)


def test_generator_cold_import_and_failed_render_preserve_both_outputs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root / "python/egglog/_builtin_catalog.pb"
    binary, stub = tmp_path / "catalog.pb", tmp_path / "builtins.pyi"
    command = [
        sys.executable,
        "-c",
        """
import importlib.abc
import runpy
import sys
class RejectFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'egglog' or fullname.startswith('egglog.'):
            raise AssertionError('Generator imported frontend: ' + fullname)
sys.meta_path.insert(0, RejectFrontend())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
""",
        str(root / "scripts/generate_builtin_catalog.py"),
        str(source),
        "--output",
        str(binary),
        "--stubs-output",
        str(stub),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert binary.read_bytes() == source.read_bytes()
    assert stub.read_text() == (root / "python/egglog/builtins.pyi").read_text()
    old_binary, old_stub = binary.read_bytes(), stub.read_bytes()
    malformed = pb.Program.from_binary(old_binary)
    addition = next(d for d in malformed.declarations if d.kind.value.name == "egglog.core.i64.add")
    assert addition.bindings is not None
    assert addition.bindings.python is not None
    addition.bindings.python.views[0].params[0].name = "invalid_operand"
    invalid = tmp_path / "invalid.pb"
    invalid.write_bytes(malformed.to_binary())
    command[4] = str(invalid)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Numeric forwarding requires one required 'other' parameter" in result.stderr
    assert binary.read_bytes() == old_binary
    assert stub.read_bytes() == old_stub


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
    declaration, view = catalog.members[("host_sort_family", "Number"), "__add__"]
    assert declaration.read().kind.value.name == "example.number.add"
    assert view == 0
    source.declarations.clear()
    assert declaration.read().kind.value.typing.value.inputs[0].sort == 0
    catalog.reindex()
    assert catalog.members[("host_sort_family", "Number"), "__add__"] == (declaration, view)


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


@pytest.mark.parametrize("generic", [False, True])
def test_nominal_and_family_member_owners_recover_independently(monkeypatch, generic):
    source = catalog_fixture()
    source.declarations[1].kind.value.typing.value.inputs = [pb.Arg(sort=0)]
    host_view = source.declarations[1].bindings.python.views[0]
    host_view.path = ["tag"]
    host_view.params = []
    source.sorts.append(pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Number")))
    source.declarations.extend([
        pb.Declaration(
            kind=Oneof[Literal["eq_sort"], pb.EqSort](
                "eq_sort",
                pb.EqSort(name="Number", bindings=pb.SortBindings(python=pb.TypeBinding(path=["example", "Nominal"]))),
            )
        ),
        pb.Declaration(
            kind=Oneof[Literal["constructor"], pb.Constructor](
                "constructor", pb.Constructor(name="nominal.tag", inputs=[pb.Arg(sort=1)], output=1)
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.METHOD,
                            path=["tag"],
                            owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", 1)),
                            receiver=0,
                        )
                    ]
                )
            ),
        ),
    ])
    host_sort = 0
    if generic:
        source.declarations[0].kind.value.arity = 1
        source.declarations[1].kind.value.typing.value.type_params = ["T"]
        source.sorts[0].kind.value.args = [2]
        source.sorts.extend([
            pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)),
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Number", args=[1]))),
        ])
        host_sort = 3
    source.nodes.extend([
        pb.Node(sort_id=host_sort, kind=Oneof[Literal["var"], str]("var", "host")),
        pb.Node(sort_id=1, kind=Oneof[Literal["var"], str]("var", "nominal")),
    ])
    catalog = Catalog(source)
    monkeypatch.setattr("egglog.runtime.builtin_catalog", lambda: catalog)
    host = _class_for_sort(catalog.owner.ref("sorts", host_sort))
    nominal = _class_for_sort(catalog.owner.ref("sorts", 1))
    assert host != nominal
    assert str(host) == ("Number[Nominal]" if generic else "Number")
    assert str(nominal) == "Nominal"
    for namespace, node, expected in [
        ("host_sort_family", 0, "example.number.add"),
        ("eq_sort", 1, "nominal.tag"),
    ]:
        declaration, _ = catalog.members[(namespace, "Number"), "tag"]
        expression = RuntimeExpr(catalog.owner.ref("nodes", node))
        assert expression.tag().__egg_ref__.read().kind.value.func == expected
        assert declaration.read().kind.value.name == expected
    catalog.members.clear()
    catalog.reindex()
    host.__egg_attr_cache__.clear()
    nominal.__egg_attr_cache__.clear()
    assert RuntimeExpr(catalog.owner.ref("nodes", 1)).tag().__egg_ref__.read().kind.value.func == "nominal.tag"
    source.declarations[2].kind.value.bindings.python.path = ["example", "Number"]
    with pytest.raises(ValueError, match="Python binding collision"):
        Catalog(source)


@pytest.mark.parametrize("nominal", [False, True])
def test_class_and_pretty_recovery_never_use_other_sort_kind(monkeypatch, nominal):
    source = pb.Program(
        ir_version=1,
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["eq"], str]("eq", "Number")
                if nominal
                else Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Number"))
            )
        ],
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="Number")
                )
                if nominal
                else Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Number"))
            )
        ],
    )
    catalog = Catalog(source)
    monkeypatch.setattr("egglog.runtime.builtin_catalog", lambda: catalog)
    reference = catalog.owner.ref("sorts", 0)
    for recover in (_class_for_sort, pretty_ref):
        with pytest.raises(LookupError, match="No canonical definition"):
            recover(reference)


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


@pytest.mark.parametrize("kind", ["constructor", "function", "primitive", "relation", "host_primitive"])
def test_constant_presentation_accepts_each_concrete_nullary_callable(kind):
    kinds = {
        "constructor": Oneof[Literal["constructor"], pb.Constructor]("constructor", pb.Constructor(name="c", output=1)),
        "function": Oneof[Literal["function"], pb.Function]("function", pb.Function(name="c", output=0)),
        "primitive": Oneof[Literal["primitive"], pb.Primitive]("primitive", pb.Primitive(name="c", output=0, body=0)),
        "relation": Oneof[Literal["relation"], pb.Relation]("relation", pb.Relation(name="c")),
        "host_primitive": Oneof[Literal["host_primitive"], pb.HostPrimitive](
            "host_primitive",
            pb.HostPrimitive(
                name="c",
                typing=Oneof[Literal["signature"], pb.GenericSignature]("signature", pb.GenericSignature(output=0)),
            ),
        ),
    }
    source = pb.Program(
        ir_version=1,
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E")),
        ],
        nodes=[
            pb.Node(
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                )
            )
        ],
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="i64")
                )
            ),
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="E"))),
            pb.Declaration(
                kind=kinds[kind],
                bindings=pb.CallableBindings(
                    python=pb.PythonBindings(
                        views=[pb.PythonCallable(kind=pb.PythonCallKind.CONSTANT, path=["example", "c"])]
                    )
                ),
            ),
        ],
    )
    catalog = Catalog(source)
    reference, view = catalog.constants["example", "c"]
    assert reference.read().kind.field == kind
    assert view == 0
    catalog.constants.clear()
    catalog.reindex()
    assert catalog.constants["example", "c"] == (reference, view)
    assert not catalog.functions
    duplicate = pb.Declaration.from_binary(source.declarations[-1].to_binary())
    duplicate.kind.value.name = "different_core_name"
    duplicate.bindings.python.views[0].kind = pb.PythonCallKind.FUNCTION
    source.declarations.append(duplicate)
    with pytest.raises(ValueError, match="binding collision"):
        Catalog(source)


@pytest.mark.parametrize(
    "invalid", ["input", "generic", "tail", "application", "receiver", "owner", "params", "mutates"]
)
def test_constant_presentation_rejects_nonconstant_shapes(invalid):
    source = catalog_fixture()
    declaration = source.declarations[1]
    declaration.kind.value.typing.value.inputs.clear()
    declaration.bindings.python.views = [pb.PythonCallable(kind=pb.PythonCallKind.CONSTANT, path=["constant"])]
    signature = declaration.kind.value.typing.value
    view = declaration.bindings.python.views[0]
    match invalid:
        case "input":
            signature.inputs.append(pb.Arg(sort=0))
        case "generic":
            signature.type_params.append("T")
        case "tail":
            signature.varargs.append(pb.Arg(sort=0))
        case "application":
            declaration.kind.value.typing = Oneof[Literal["application"], pb.FunctionApplication](
                "application", pb.FunctionApplication()
            )
        case "receiver":
            view.receiver = 0
        case "owner":
            view.owner = pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", 0))
        case "params":
            view.params.append(pb.PythonParameter(core_input=0, name="x"))
        case "mutates":
            view.mutates = 0
    with pytest.raises(ValueError, match=r"Python (constant|member)"):
        Catalog(source)


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
    declaration = source.declarations[1]
    declaration.kind.value.typing.value.inputs.clear()
    declaration.bindings.python.views = [pb.PythonCallable(kind=pb.PythonCallKind.CONSTANT, path=path)]
    if valid:
        assert tuple(path) in Catalog(source).constants
    else:
        with pytest.raises(ValueError, match="Invalid Python constant path"):
            Catalog(source)
