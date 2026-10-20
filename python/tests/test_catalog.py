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
from egglog._program import Builder, StructuralView, pack
from egglog.pretty import pretty_ref
from egglog.runtime import RuntimeExpr, RuntimeFunction, _class_for_sort
from egglog.type_constraint_solver import TypeConstraintError


def _required_named_declaration(program: pb.Program, name: str) -> pb.Declaration:
    """Require populated declaration arms while retaining first-match lookup semantics."""
    for declaration in program.declarations:
        kind = declaration.kind
        assert isinstance(kind, Oneof)
        if kind.value.name == name:
            return declaration
    raise StopIteration


def _host_signature(declaration: pb.Declaration) -> tuple[pb.HostPrimitive, pb.GenericSignature]:
    """Require the concrete host signature before inspecting or corrupting a catalog fixture."""
    kind = declaration.kind
    assert isinstance(kind, Oneof)
    assert kind.field == "host_primitive"
    primitive = kind.value
    assert isinstance(primitive, pb.HostPrimitive)
    typing = primitive.typing
    assert isinstance(typing, Oneof)
    assert typing.field == "signature"
    signature = typing.value
    assert isinstance(signature, pb.GenericSignature)
    return primitive, signature


def _python_bindings(declaration: pb.Declaration) -> pb.PythonBindings:
    """Require callable presentation metadata before inspecting or corrupting a fixture."""
    bindings = declaration.bindings
    assert bindings is not None
    python = bindings.python
    assert python is not None
    return python


def _type_binding(payload: object) -> pb.TypeBinding:
    """Require an actual sort declaration with Python type presentation metadata."""
    assert isinstance(payload, (pb.HostSortFamily, pb.EqSort))
    bindings = payload.bindings
    assert bindings is not None
    python = bindings.python
    assert python is not None
    return python


def test_pyobject_generated_surface_uses_real_defaults_and_host_policy():
    from egglog._catalog import builtin_catalog  # noqa: PLC0415
    from egglog._program import pack  # noqa: PLC0415

    program = pack(builtin_catalog().roots).program
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = renderer(program)
    for signature in (
        "def __init__(self, value: object, /) -> None:",
        "def value(self) -> object:",
        "def __call__(self, *args: PyObjectLike) -> PyObject:",
        "def dict(cls, *keys_and_values: PyObjectLike) -> PyObject:",
        "def dict_update(self, *keys_and_values: PyObjectLike) -> PyObject:",
        "def py_eval(code: StringLike, globals_: PyObjectLike = ..., locals_: PyObjectLike = ...) -> PyObject:",
        "def py_exec(code: StringLike, globals_: PyObjectLike = ..., locals_: PyObjectLike = ...) -> PyObject:",
        "class PyObjectFunction(Protocol):",
        "def py_eval_fn(fn: Callable) -> PyObjectFunction:",
    ):
        assert signature in stub
    assert '"py_eval_fn"' in stub.split("__all__ = ", 1)[1].splitlines()[0]
    assert '"PyObjectFunction"' not in stub.split("__all__ = ", 1)[1].splitlines()[0]
    for name in ("eval", "exec"):
        declaration = builtin_catalog().definitions["callable", f"egglog.python.pyobject.{name}"]
        view = declaration.read().bindings.python.views[0]
        assert view.path == ["egglog", "builtins", f"py_{name}"]
        assert view.params[0].name == "code"
        assert not view.params[0].has_field("default_expr")
        for parameter, expected in zip(view.params[1:], ("globals_", "locals_"), strict=True):
            assert parameter.name == expected
            default = declaration.owner.ref("nodes", parameter.default_expr).read()
            assert default.kind.field == "call"
            assert default.kind.value.func == "egglog.python.pyobject.dict"
            assert not default.kind.value.args
    # The same real export must remain usable after all arena IDs relocate.
    relocated = pack(reversed(builtin_catalog().roots)).program
    assert sorted(renderer(relocated).splitlines()) == sorted(stub.splitlines())


@pytest.mark.parametrize("corruption", ["index", "sort"])
def test_pyobject_generator_rejects_invalid_default(corruption):
    from egglog._catalog import builtin_catalog  # noqa: PLC0415
    from egglog._program import pack  # noqa: PLC0415

    program = pack(builtin_catalog().roots).program
    declaration = _required_named_declaration(program, "egglog.python.pyobject.eval")
    parameter = _python_bindings(declaration).views[0].params[1]
    if corruption == "index":
        parameter.default_expr = len(program.nodes)
    else:
        for index, sort in enumerate(program.sorts):
            sort_kind = sort.kind
            assert sort_kind is not None
            if sort_kind.field == "family" and sort_kind.value.name == "i64":
                integer_sort = index
                break
        else:
            raise StopIteration
        program.nodes[parameter.default_expr].sort_id = integer_sort
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    with pytest.raises(ValueError, match="Default"):
        renderer(program)


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
    declarations: dict[str, pb.Declaration] = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        declarations[declaration_kind.value.name] = declaration
    for name, paths in (("string.concat", ["join", "__add__"]), ("bool.or", ["__or__"])):
        declaration = declarations[f"egglog.core.{name}"]
        _, signature = _host_signature(declaration)
        assert not signature.inputs
        assert len(signature.varargs) == 1
        assert [v.path[-1] for v in _python_bindings(declaration).views] == paths
        assert _python_bindings(declaration).views[-1].tail_repetitions == 2
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
    _python_bindings(declarations["egglog.core.string.concat"]).views[0].path[-1] = "concatenate"
    assert "def concatenate(*strings: StringLike) -> String:" in render(program)


@pytest.mark.parametrize(
    "mutation", ["fixed", "missing", "duplicate", "owner", "default", "mutation", "huge", "property"]
)
def test_scalar_specialized_stub_rejects_invalid_views(mutation):
    root = Path(__file__).resolve().parents[2]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declaration = _required_named_declaration(program, "egglog.core.bool.or")
    _, signature = _host_signature(declaration)
    view = _python_bindings(declaration).views[0]
    if mutation == "fixed":
        signature.inputs = signature.varargs * 2
        signature.varargs.clear()
    elif mutation == "missing":
        view.params.clear()
    elif mutation == "duplicate":
        view.params[0].core_input = view.receiver
    elif mutation == "owner":
        assert view.owner is not None
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
    declaration = _required_named_declaration(program, "egglog.core.string.concat")
    view = _python_bindings(declaration).views[0]
    if mutation == "generic":
        _, signature = _host_signature(declaration)
        signature.type_params.append("T")
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


def test_vec_map_view_and_static_signature_come_from_actual_catalog():
    root = Path(__file__).resolve().parents[2]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    declaration = _required_named_declaration(program, "egglog.core.vec.map")
    _, signature = _host_signature(declaration)
    assert declaration.bindings is not None
    assert declaration.bindings.rust is not None
    rust = declaration.bindings.rust.views[0]
    view = _python_bindings(declaration).views[0]
    assert view.kind == pb.PythonCallKind.METHOD
    assert view.path == ["map"]
    assert view.owner == rust.owner
    assert rust.receiver is not None
    assert view.receiver == rust.receiver.core_input == 1
    assert [(p.core_input, p.name) for p in view.params] == [(0, signature.inputs[0].name)]
    assert not view.has_field("mutates")
    assert not view.params[0].has_field("default_expr")
    callback_kind = program.sorts[signature.inputs[0].sort].kind
    assert callback_kind is not None
    assert callback_kind.field == "func"
    assert view.owner is not None
    assert view.owner.kind is not None
    assert view.owner.kind.field == "sort"
    assert signature.inputs[1].sort == view.owner.kind.value
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = render(program)
    assert "def map(self, f: Callable[[_Vec_T_0], _Vec_map_T1]) -> Vec[_Vec_map_T1]:" in stub
    assert '_Vec_map_T1 = TypeVar("_Vec_map_T1", bound=BaseExpr)' in stub
    owner = Builder.from_program(program).publish()
    relocated = pack(owner.ref("declarations", i) for i in reversed(range(len(program.declarations)))).program
    assert sorted(render(relocated).splitlines()) == sorted(stub.splitlines())


@pytest.mark.parametrize(
    "mutation", ["grouped", "default", "missing", "duplicate", "nonfinal", "receiver", "tail_receiver"]
)
def test_vec_static_tail_rejects_unrepresentable_or_invalid_views(vec_catalog_program, mutation):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    of = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.vec.of")
    _, signature = _host_signature(of)
    view = of.bindings.python.views[0]
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
    _, signature = _host_signature(of)
    view = of.bindings.python.views[0]
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
    assert count.kind is not None
    count.kind.value.name = "test.vec.count"
    _, signature = _host_signature(count)
    view = _python_bindings(count).views[0]
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
    record.sort_id = builder.import_ref(i64.__egg_sort__)  # type: ignore[attr-defined]  # RuntimeClass metadata.
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


def test_function_generated_surface_uses_actual_application_metadata():
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    application = next(
        d for d in program.declarations if d.kind is not None and d.kind.value.name == "egglog.core.fn.apply"
    )
    assert application.kind is not None
    assert application.kind.field == "host_primitive"
    assert application.kind.value.typing is not None
    assert application.kind.value.typing.field == "application"
    assert application.bindings is not None
    assert application.bindings.python is not None
    (view,) = application.bindings.python.views
    assert view.owner is not None
    assert view.owner.kind is not None
    assert view.owner.kind.field == "function"
    stub = render(program)
    for signature in (
        "class UnstableFn(BuiltinExpr, Generic[_UnstableFn_T, *_UnstableFn_TS]):",
        '    __match_args__ = ("value",)',
        "def __init__(self, f: Callable[[Unpack[_UnstableFn_TS]], _UnstableFn_T]) -> None:",
        "def __init__(self, f: Callable[[_UnstableFn_T1, Unpack[_UnstableFn_TS]], _UnstableFn_T], _a: _UnstableFn_T1, /) -> None:",
        "def __init__(self, f: Callable[[_UnstableFn_T1, _UnstableFn_T2, Unpack[_UnstableFn_TS]], _UnstableFn_T], _a: _UnstableFn_T1, _b: _UnstableFn_T2, /) -> None:",
        "def __call__(self, *args: *_UnstableFn_TS) -> _UnstableFn_T:",
        "def value(self) -> Callable[[Unpack[_UnstableFn_TS]], _UnstableFn_T]:",
        '    @deprecated("use .value")',
        "def eval(self) -> Callable[[Unpack[_UnstableFn_TS]], _UnstableFn_T]:",
    ):
        assert signature in stub
    assert stub.count("    @overload\n") == 3
    assert "Any" not in stub
    application.bindings.python = None
    with pytest.raises(ValueError, match="Function sort requires an admitted structural function view"):
        render(program)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("view", "kind", pb.PythonCallKind.PROPERTY),
        ("view", "path", ["apply"]),
        ("view", "owner", None),
        ("view", "owner", pb.BindingOwner()),
        ("view", "owner", pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", 0))),
        ("view", "receiver", 1),
        ("view", "receiver", None),
        ("parameter", "core_input", 0),
        ("parameter", "core_input", None),
        ("view", "params", []),
        (
            "view",
            "params",
            [pb.PythonParameter(core_input=1, name="args"), pb.PythonParameter(core_input=2, name="extra")],
        ),
        ("parameter", "name", "values"),
        ("parameter", "default_expr", 0),
        ("view", "mutates", 0),
        ("view", "tail_repetitions", 0),
        ("conflict", "", None),
        ("collision", "", None),
        ("signature", "", None),
    ],
)
def test_function_generated_views_reject_invalid_metadata(target, field, value):
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    application = next(
        d for d in program.declarations if d.kind is not None and d.kind.value.name == "egglog.core.fn.apply"
    )
    assert application.kind is not None
    assert application.kind.field == "host_primitive"
    assert application.bindings is not None
    assert application.bindings.python is not None
    (view,) = application.bindings.python.views
    if target == "view":
        setattr(view, field, value)
    elif target == "parameter":
        setattr(view.params[0], field, value)
    elif target == "conflict":
        other = pb.Declaration.from_binary(application.to_binary())
        assert other.kind is not None
        other.kind.value.name = "test.other_application"
        program.declarations.append(other)
    elif target == "collision":
        family = next(d.kind.value for d in program.declarations if d.kind is not None and d.kind.value.name == "Vec")
        assert isinstance(family, pb.HostSortFamily)
        assert family.bindings is not None
        assert family.bindings.python is not None
        family.bindings.python.path[-1] = "UnstableFn"
    else:
        application.kind.value.typing = Oneof[Literal["signature"], pb.GenericSignature](
            "signature", pb.GenericSignature(output=0)
        )
    with pytest.raises(ValueError, match=r"function|Function|collision|static view"):
        render(program)
    if target != "collision":
        with pytest.raises(ValueError, match=r"function|collision"):
            Catalog(program)


def test_function_sort_annotations_recurse_through_canonical_records():
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    declarations = {d.kind.value.name: d for d in program.declarations if d.kind is not None}
    signatures = {
        d.kind.value.name: d.kind.value.typing.value
        for d in program.declarations
        if d.kind is not None
        and d.kind.field == "host_primitive"
        and d.kind.value.typing is not None
        and d.kind.value.typing.field == "signature"
    }
    signature = signatures["egglog.core.pair.first"]
    left, right = (arg.sort for arg in signatures["egglog.core.pair.make"].inputs)
    index = len(program.sorts)
    program.sorts.extend([
        pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[left], result=right))),
        pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[index], result=index))),
        pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[index + 1]))),
    ])
    signature.output = index + 2
    assert (
        "def left(self) -> Vec[UnstableFn[UnstableFn[_Pair_R_1, _Pair_L_0], UnstableFn[_Pair_R_1, _Pair_L_0]]]:"
        in render(program)
    )
    i64_sort = signatures["egglog.core.i64.add"].output
    kind = program.sorts[index].kind
    assert kind is not None
    assert kind.field == "func"
    kind.value.params = [i64_sort]
    kind.value.result = i64_sort
    join = declarations["egglog.core.string.concat"]
    assert join.bindings is not None
    assert join.bindings.python is not None
    join.bindings.python.views = [join.bindings.python.views[0]]
    signatures["egglog.core.string.concat"].varargs[0].sort = index
    assert "def join(*strings: Callable[[i64], i64]) -> String:" in render(program)
    kind.value.params.clear()
    assert "def join(*strings: Callable[[], i64]) -> String:" in render(program)
    kind.value.result = 0
    zero_sort = program.sorts[0].kind
    assert zero_sort is not None
    assert zero_sort.field == "family"
    zero_declaration = declarations[zero_sort.value.name].kind
    assert zero_declaration is not None
    assert zero_declaration.field == "host_sort_family"
    zero_family = zero_declaration.value
    assert zero_family.bindings is not None
    assert zero_family.bindings.python is not None
    zero_name = zero_family.bindings.python.path[-1]
    assert f"def join(*strings: Callable[[], {zero_name}]) -> String:" in render(program)
    kind.value.result = index
    with pytest.raises(ValueError, match="cyclic signature sort"):
        render(program)
    kind.value.result = len(program.sorts)
    with pytest.raises(ValueError, match="Invalid or cyclic signature sort"):
        render(program)


def test_stub_generation_uses_canonical_ordinary_method_view():
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = _required_named_declaration(program, "egglog.core.i64.add")
    view = _python_bindings(addition).views[0]
    view.path = ["plus"]
    view.params[0].name = "rhs"
    assert "def plus(self, rhs: i64Like) -> i64:" in renderer(program)
    owner = Builder.from_program(program).publish()
    method = RuntimeFunction(owner.ref("declarations", program.declarations.index(addition)), 0, i64(1))
    result = method(rhs=i64(2))
    assert result is not None
    argument = result.__egg_ref__.read().kind.value.args[1]
    assert RuntimeExpr(result.__egg_ref__.owner.ref("nodes", argument)).value == 2


@pytest.mark.parametrize(
    "name", ["int", "_Vec_T_0", "AttributeError", "tuple", "dict", "Never", "_Map_HostK", "MapLike"]
)
def test_stub_generation_rejects_host_and_internal_name_collisions(name):
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    scalar_kind = _required_named_declaration(program, "i64").kind
    assert scalar_kind is not None
    scalar = scalar_kind.value
    _type_binding(scalar).path[-1] = name
    with pytest.raises(ValueError, match="Python static name collision"):
        renderer(program)
    _type_binding(scalar).path[-1] = "Integer"
    assert "class Integer(BuiltinExpr):" in renderer(program)


@pytest.mark.parametrize("mutation", ["output", "receiver", "params", "owner_duplicate", "unknown_binder", "default"])
def test_pair_stub_rejects_invalid_or_unadmitted_generic_views(pair_catalog_program, mutation):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    make = declarations["egglog.core.pair.make"]
    first = declarations["egglog.core.pair.first"]
    _, signature = _host_signature(make)
    if mutation == "output":
        signature.output = signature.inputs[0].sort
    elif mutation == "receiver":
        first.bindings.python.views[0].clear_field("receiver")
    elif mutation == "params":
        first.bindings.python.views[0].params = [pb.PythonParameter(name="extra", core_input=0)]
    elif mutation == "owner_duplicate":
        owner_args = pair_catalog_program.sorts[signature.output].kind.value.args
        owner_args[1] = owner_args[0]
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
    _, signature = _host_signature(make)
    owner_sort = pair_catalog_program.sorts[signature.output]
    duplicated = pb.Sort.from_binary(owner_sort.to_binary())
    duplicated_kind = duplicated.kind
    assert duplicated_kind is not None
    assert duplicated_kind.field == "family"
    duplicated_kind.value.args = [len(pair_catalog_program.sorts) + 1, len(pair_catalog_program.sorts) + 2]
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
    _, first_signature = _host_signature(first)
    first_signature.output = signature.inputs[1].sort
    assert "def left(self) -> _Pair_R_1:" in renderer(pair_catalog_program)


def test_pair_stub_rejects_specialized_owner_as_universal_class_view(pair_catalog_program):
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    for name in ("first", "second"):
        declarations[f"egglog.core.pair.{name}"].bindings.python = None
    make = declarations["egglog.core.pair.make"]
    _, signature = _host_signature(make)
    signature.type_params.clear()
    i64_sort = next(
        i for i, s in enumerate(pair_catalog_program.sorts) if s.kind.field == "family" and s.kind.value.name == "i64"
    )
    signature.inputs[0].sort = i64_sort
    signature.inputs[1].sort = i64_sort
    pair_catalog_program.sorts[signature.output].kind.value.args = [i64_sort, i64_sort]
    with pytest.raises(ValueError, match="distinct in-range family parameters"):
        renderer(pair_catalog_program)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("receiver", "cover each input exactly once"),
        ("tail", "fixed concrete signature"),
        ("collision", "Python member collision"),
        ("missing_output", "fixed concrete signature"),
        ("application", "Unsupported structural function Python static view"),
        ("numeric_keyword", "Numeric forwarding requires one required 'other' parameter"),
        ("numeric_default", "Numeric forwarding requires one required 'other' parameter"),
    ],
)
def test_stub_generation_rejects_invalid_or_unmigrated_shapes(mutation, message):
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = _required_named_declaration(program, "egglog.core.i64.add")
    primitive, signature = _host_signature(addition)
    view = _python_bindings(addition).views[0]
    if mutation == "receiver":
        view.receiver = 1
    elif mutation == "tail":
        signature.varargs = [pb.Arg(sort=1), pb.Arg(sort=0)]
    elif mutation == "collision":
        _python_bindings(addition).views.append(pb.PythonCallable.from_binary(view.to_binary()))
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
        view.params[0].default_expr = len(program.nodes) - 1
    else:
        primitive.typing = Oneof[Literal["application"], pb.FunctionApplication](
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
    addition = _required_named_declaration(program, "egglog.core.i64.add")
    _python_bindings(addition).views[0].path = ["plus"]
    parameter = _python_bindings(addition).views[0].params[0]
    _, signature = _host_signature(addition)
    scalar_sort = signature.inputs[parameter.core_input].sort
    literal = len(program.nodes)
    program.nodes.extend([
        pb.Node(
            sort_id=scalar_sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 2))
            ),
        ),
        pb.Node(
            sort_id=scalar_sort,
            kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.i64.add", args=[literal, literal])),
        ),
    ])
    parameter.default_expr = literal + 1
    assert "def plus(self, other: i64Like = ...) -> i64:" in renderer(program)
    owner = Builder.from_program(program).publish()
    method = RuntimeFunction(owner.ref("declarations", program.declarations.index(addition)), 0, i64(1))
    result = method()
    assert result is not None
    argument = result.__egg_ref__.read().kind.value.args[1]
    assert result.__egg_ref__.owner.ref("nodes", argument).read().kind.value.func == "egglog.core.i64.add"
    parameter.clear_field("default_expr")
    assert "other: i64Like = ..." not in renderer(program)
    parameter.default_expr = len(program.nodes)
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
    addition = _required_named_declaration(malformed, "egglog.core.i64.add")
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

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)
    source = catalog_fixture()
    catalog = Catalog(source)
    family = catalog.types["example", "Number"]
    assert family.read().kind.value.name == "Number"
    declaration, view = catalog.members[("host_sort_family", "Number"), "__add__"]
    assert declaration.read().kind.value.name == "example.number.add"
    assert view == 0
    source.declarations.clear()
    _, signature = _host_signature(declaration.read())
    assert signature.inputs[0].sort == 0
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
    _, signature = _host_signature(source.declarations[1])
    signature.inputs = [pb.Arg(sort=0)]
    host_view = _python_bindings(source.declarations[1]).views[0]
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
        family_kind = source.declarations[0].kind
        assert family_kind is not None
        assert family_kind.field == "host_sort_family"
        family_kind.value.arity = 1
        signature.type_params = ["T"]
        host_kind = source.sorts[0].kind
        assert host_kind is not None
        assert host_kind.field == "family"
        host_kind.value.args = [2]
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
        tag = expression.tag
        assert isinstance(tag, RuntimeFunction)
        tagged = tag()
        assert tagged is not None
        assert tagged.__egg_ref__.read().kind.value.func == expected
        assert declaration.read().kind.value.name == expected
    catalog.members.clear()
    catalog.reindex()
    host.__egg_attr_cache__.clear()
    nominal.__egg_attr_cache__.clear()
    nominal_expression = RuntimeExpr(catalog.owner.ref("nodes", 1))
    nominal_tag = nominal_expression.tag
    assert isinstance(nominal_tag, RuntimeFunction)
    nominal_tagged = nominal_tag()
    assert nominal_tagged is not None
    assert nominal_tagged.__egg_ref__.read().kind.value.func == "nominal.tag"
    nominal_kind = source.declarations[2].kind
    assert nominal_kind is not None
    _type_binding(nominal_kind.value).path = ["example", "Number"]
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
    kind.value.bindings.python.path = ["bad-name", "Number"]
    with pytest.raises(ValueError, match="Python path"):
        Catalog(source)


@pytest.mark.parametrize("name", ["1.0", "not-a-python-identifier", "class"])
def test_runtime_type_names_do_not_weaken_static_generator_identifiers(name):
    root = Path(__file__).resolve().parents[2]
    source = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    declaration = _required_named_declaration(source, "i64")
    declaration_kind = declaration.kind
    assert declaration_kind is not None
    type_binding = _type_binding(declaration_kind.value)
    type_binding.path[-1] = name
    path = tuple(type_binding.path)
    catalog = Catalog(source)
    reference = catalog.types[path]
    reference_kind = reference.read().kind
    assert reference_kind is not None
    assert _type_binding(reference_kind.value).path[-1] == name
    restored = Catalog(pb.Program.from_binary(pack(catalog.roots).program.to_binary()))
    restored_kind = restored.types[path].read().kind
    assert restored_kind is not None
    assert _type_binding(restored_kind.value).path[-1] == name
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    with pytest.raises(ValueError, match="Invalid Python identifier"):
        render(source)
    source.declarations.append(
        pb.Declaration(
            kind=Oneof[Literal["eq_sort"], pb.EqSort](
                "eq_sort",
                pb.EqSort(name="different.core", bindings=pb.SortBindings(python=pb.TypeBinding(path=list(path)))),
            )
        )
    )
    with pytest.raises(ValueError, match="Python binding collision"):
        Catalog(source)


@pytest.mark.parametrize(
    "path", [[], [""], ["example", ""], ["", "Number"], ["class", "Number"], ["bad-name", "Number"]]
)
def test_type_paths_keep_nonempty_terminal_and_identifier_qualifiers(path):
    source = catalog_fixture()
    declaration_kind = source.declarations[0].kind
    assert declaration_kind is not None
    _type_binding(declaration_kind.value).path = path
    with pytest.raises(ValueError, match="Python path"):
        Catalog(source)


@pytest.mark.parametrize("name", ["", "123", "eval_mode_unsafe-seminaive", "class", "@internal"])
def test_runtime_free_function_names_preserve_exact_callee_and_core_order(monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog names and calls cannot create an engine")

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)
    source = pb.Program.from_binary((Path(__file__).parents[1] / "egglog/_builtin_catalog.pb").read_bytes())
    for index, sort in enumerate(source.sorts):
        sort_kind = sort.kind
        assert sort_kind is not None
        if sort_kind.field == "family" and sort_kind.value.name == "i64" and not sort_kind.value.args:
            integer = index
            break
    else:
        raise StopIteration
    source.declarations.append(
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function](
                "function",
                pb.Function(
                    name="exact.declared.table",
                    inputs=[pb.Arg(sort=integer), pb.Arg(sort=integer)],
                    output=integer,
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.FUNCTION,
                            path=["example", name],
                            params=[
                                pb.PythonParameter(core_input=1, name="right"),
                                pb.PythonParameter(core_input=0, name="left"),
                            ],
                        )
                    ]
                )
            ),
        )
    )
    catalog = Catalog(source)
    reference, view = catalog.functions["example", name]
    result = RuntimeFunction(reference, view)(i64(9), left=i64(4))
    assert isinstance(result, RuntimeExpr)
    call = result.__egg_ref__.read().kind.value
    assert call.func == "exact.declared.table"
    assert [result.__egg_ref__.owner.ref("nodes", index).read().kind.value.value.value for index in call.args] == [4, 9]
    restored = Catalog(pb.Program.from_binary(pack([result.__egg_ref__]).program.to_binary()))
    restored_reference, restored_view = restored.functions["example", name]
    assert restored_view == view
    assert StructuralView(restored_reference) == StructuralView(reference)
    restored.functions.clear()
    restored.reindex()
    assert restored.functions["example", name] == (restored_reference, restored_view)
    duplicate = pb.Declaration.from_binary(source.declarations[-1].to_binary())
    assert duplicate.kind is not None
    duplicate.kind.value.name = "different.declared.table"
    source.declarations.append(duplicate)
    with pytest.raises(ValueError, match="binding collision"):
        Catalog(source)


@pytest.mark.parametrize("path", [[], ["", "f"], ["class", "f"], ["bad-name", "f"]])
def test_free_function_paths_keep_nonempty_identifier_qualifiers(path):
    source = catalog_fixture()
    _python_bindings(source.declarations[1]).views = [pb.PythonCallable(kind=pb.PythonCallKind.FUNCTION, path=path)]
    with pytest.raises(ValueError, match="Python path"):
        Catalog(source)


@pytest.mark.parametrize("name", ["", "not-a-python-identifier", "class"])
def test_runtime_free_names_do_not_weaken_static_generator_identifiers(name):
    root = Path(__file__).resolve().parents[2]
    source = pb.Program.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    declaration = _required_named_declaration(source, "egglog.core.string.concat")
    view = _python_bindings(declaration).views[0]
    view.path[-1] = name
    assert tuple(view.path) in Catalog(source).functions
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    with pytest.raises(ValueError, match="Invalid Python identifier"):
        render(source)


@pytest.mark.parametrize("kind", ["constructor", "function", "primitive", "relation", "host_primitive"])
def test_constant_presentation_accepts_each_concrete_nullary_callable(kind):
    kinds: dict[
        str,
        Oneof[Literal["constructor"], pb.Constructor]
        | Oneof[Literal["function"], pb.Function]
        | Oneof[Literal["primitive"], pb.Primitive]
        | Oneof[Literal["relation"], pb.Relation]
        | Oneof[Literal["host_primitive"], pb.HostPrimitive],
    ] = {
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
    assert duplicate.kind is not None
    duplicate.kind.value.name = "different_core_name"
    _python_bindings(duplicate).views[0].kind = pb.PythonCallKind.FUNCTION
    source.declarations.append(duplicate)
    with pytest.raises(ValueError, match="binding collision"):
        Catalog(source)


@pytest.mark.parametrize(
    "invalid", ["input", "generic", "tail", "application", "receiver", "owner", "params", "mutates"]
)
def test_constant_presentation_rejects_nonconstant_shapes(invalid):
    source = catalog_fixture()
    declaration = source.declarations[1]
    primitive, signature = _host_signature(declaration)
    signature.inputs.clear()
    _python_bindings(declaration).views = [pb.PythonCallable(kind=pb.PythonCallKind.CONSTANT, path=["constant"])]
    view = _python_bindings(declaration).views[0]
    match invalid:
        case "input":
            signature.inputs.append(pb.Arg(sort=0))
        case "generic":
            signature.type_params.append("T")
        case "tail":
            signature.varargs.append(pb.Arg(sort=0))
        case "application":
            primitive.typing = Oneof[Literal["application"], pb.FunctionApplication](
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
    _, signature = _host_signature(declaration)
    signature.inputs.clear()
    _python_bindings(declaration).views = [pb.PythonCallable(kind=pb.PythonCallKind.CONSTANT, path=path)]
    if valid:
        assert tuple(path) in Catalog(source).constants
    else:
        with pytest.raises(ValueError, match="Invalid Python constant path"):
            Catalog(source)


def test_multiset_generated_surface_from_actual_catalog():
    from egglog._catalog import builtin_catalog  # noqa: PLC0415
    from egglog._program import pack  # noqa: PLC0415

    program = pack(builtin_catalog().roots).program
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(program)
    body = stub.split("class MultiSet(", 1)[1].split("\nclass ", 1)[0]
    for signature in (
        "def __init__(self, *args: _MultiSet_T_0) -> None:",
        "def single(cls, x: _MultiSet_T_0, i: i64Like) -> MultiSet[_MultiSet_T_0]:",
        "def pick(self) -> _MultiSet_T_0:",
        "def insert(self, value: _MultiSet_T_0) -> MultiSet[_MultiSet_T_0]:",
        "def remove(self, value: _MultiSet_T_0) -> MultiSet[_MultiSet_T_0]:",
        "def __sub__(self, other: MultiSet[_MultiSet_T_0]) -> MultiSet[_MultiSet_T_0]:",
        "def length(self) -> i64:",
        "def contains(self, value: _MultiSet_T_0) -> Unit:",
        "def not_contains(self, value: _MultiSet_T_0) -> Unit:",
        "def __and__(self, other: MultiSet[_MultiSet_T_0]) -> MultiSet[_MultiSet_T_0]:",
        "def __add__(self, other: MultiSet[_MultiSet_T_0]) -> MultiSet[_MultiSet_T_0]:",
        "def reset_counts(self) -> MultiSet[_MultiSet_T_0]:",
        "def pick_max(self) -> _MultiSet_T_0:",
        "def count(self, value: _MultiSet_T_0) -> i64:",
        "def sum_multisets(cls, xs: MultiSet[MultiSet[_MultiSet_T_0]]) -> MultiSet[_MultiSet_T_0]:",
        "def union_values(self) -> _MultiSet_T_0:",
        "def value(self) -> list[_MultiSet_T_0]:",
        "def __iter__(self) -> Iterator[_MultiSet_T_0]:",
        "def __len__(self) -> int:",
        "def __contains__(self, item: _MultiSet_T_0) -> bool:",
        "def eval(self) -> list[_MultiSet_T_0]:",
    ):
        assert signature in body
    assert "MultiSetLike: TypeAlias = MultiSet[_MultiSet_T_0] | tuple[_MultiSet_HostT, ...]" in stub
    for suffix in ("remove_swapped", "subtract_swapped", "contains_swapped", "not_contains_swapped"):
        name = f"multiset_{suffix}"
        parameter = f"_{name}_T0"
        first = f"MultiSet[{parameter}]" if suffix == "subtract_swapped" else parameter
        output = f"MultiSet[{parameter}]" if suffix in {"remove_swapped", "subtract_swapped"} else "Unit"
        assert f'{parameter} = TypeVar("{parameter}", bound=BaseExpr)' in stub
        assert f"def {name}(x: {first}, xs: MultiSet[{parameter}]) -> {output}:" in stub
        assert f'"{name}"' in stub.split("__all__ = ", 1)[1].splitlines()[0]
    declarations: dict[str, pb.Declaration] = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        declarations[declaration_kind.value.name] = declaration
    assert len([name for name in declarations if name.startswith("egglog.core.multiset.")]) == 27
    assert "def empty(" not in body
    assert "def map(self, f: Callable[[_MultiSet_T_0], _MultiSet_map_T1]) -> MultiSet[_MultiSet_map_T1]:" in body
    multiset_kind = declarations["MultiSet"].kind
    assert multiset_kind is not None
    _type_binding(multiset_kind.value).path[-1] = "Bag"
    view = _python_bindings(declarations["egglog.core.multiset.remove_swapped"]).views[0]
    view.path[-1] = "take_one"
    view.params[0].name, view.params[1].name = "item", "values"
    renamed = render(program)
    assert "BagLike: TypeAlias = Bag[_Bag_T_0] | tuple[_Bag_HostT, ...]" in renamed
    assert "def take_one(item: _take_one_T0, values: Bag[_take_one_T0]) -> Bag[_take_one_T0]:" in renamed


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("unused", "Unused static signature parameter"),
        ("unbound", "Unbound static signature parameter"),
        ("default", "Default root"),
        ("tail", "Generic static free functions require fixed signatures"),
        ("collision", "collision"),
    ],
)
def test_multiset_generic_free_static_guards(corruption, message):
    from egglog._catalog import builtin_catalog  # noqa: PLC0415
    from egglog._program import pack  # noqa: PLC0415

    program = pack(builtin_catalog().roots).program
    declarations: dict[str, pb.Declaration] = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        declarations[declaration_kind.value.name] = declaration
    declaration = declarations["egglog.core.multiset.remove_swapped"]
    _, signature = _host_signature(declaration)
    if corruption == "unused":
        signature.type_params.append("Unused")
    elif corruption == "unbound":
        signature.inputs[0].sort = len(program.sorts)
        program.sorts.append(pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    elif corruption == "default":
        _python_bindings(declaration).views[0].params[0].default_expr = len(program.nodes)
    elif corruption == "tail":
        signature.varargs.append(signature.inputs[0])
    else:
        multiset_kind = declarations["MultiSet"].kind
        assert multiset_kind is not None
        _type_binding(multiset_kind.value).path[-1] = "_multiset_remove_swapped_T0"
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    with pytest.raises(ValueError, match=message):
        render(program)


@pytest.mark.parametrize("label", ["", "Element type", "class", "duplicate"])
def test_multiset_free_signature_labels_are_diagnostic_only(label, monkeypatch):
    from egglog import bindings  # noqa: PLC0415
    from egglog._catalog import builtin_catalog  # noqa: PLC0415
    from egglog._program import pack  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Signature label rendering authored or executed a Call")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    monkeypatch.setattr("egglog.runtime._author_call", forbidden)
    program = pack(builtin_catalog().roots).program
    signatures = []
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        if declaration_kind.value.name in {
            f"egglog.core.multiset.{suffix}"
            for suffix in ("remove_swapped", "subtract_swapped", "contains_swapped", "not_contains_swapped")
        }:
            signatures.append(_host_signature(declaration)[1])
    assert len(signatures) == 4
    if label == "duplicate":
        # Use a second positional binder in a real fixed free signature so the
        # duplicate-label control is within one signature, not across functions.
        signatures[0].type_params.append("Other")
        signatures[0].inputs[0].sort = len(program.sorts)
        program.sorts.append(pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    before = render(program)
    for signature in signatures:
        signature.type_params = [label] * len(signature.type_params)
    assert render(program) == before


def test_multiset_method_local_binders_from_actual_catalog():
    from os import environ  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    source = Path(environ.get("EGGLOG_TEST_HIGHER_ORDER_CATALOG", root / "python/egglog/_builtin_catalog.pb"))
    program = pb.Program.from_binary(source.read_bytes())
    declaration = _required_named_declaration(program, "egglog.core.multiset.map")
    _, signature = _host_signature(declaration)
    view = _python_bindings(declaration).views[0]
    assert len(signature.type_params) == 2
    assert view.owner is not None
    assert view.owner.kind is not None
    assert view.owner.kind.field == "sort"
    owner_kind = program.sorts[view.owner.kind.value].kind
    assert owner_kind is not None
    assert owner_kind.field == "family"
    owner = owner_kind.value
    assert owner.name == "MultiSet"
    assert len(owner.args) == 1
    argument_kind = program.sorts[owner.args[0]].kind
    assert argument_kind is not None
    assert argument_kind.value == 0
    callback = program.sorts[signature.inputs[view.params[0].core_input].sort]
    assert callback.kind is not None
    assert callback.kind.field == "func"
    assert len(callback.kind.value.params) == 1
    parameter_kind = program.sorts[callback.kind.value.params[0]].kind
    assert parameter_kind is not None
    assert parameter_kind.value == 0
    result_kind = program.sorts[callback.kind.value.result].kind
    assert result_kind is not None
    assert result_kind.value == 1
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = render(program)
    local = "_MultiSet_map_T1"
    assert f'{local} = TypeVar("{local}", bound=BaseExpr)' in stub
    assert f"def map(self, f: Callable[[_MultiSet_T_0], {local}]) -> MultiSet[{local}]:" in stub
    assert "class MultiSet(BuiltinExpr, Generic[_MultiSet_T_0]):" in stub
    assert "Any" not in stub
    assert "Callable[...," not in stub
    for labels in (["", ""], ["class", "Element type"], ["duplicate", "duplicate"]):
        signature.type_params = labels
        assert render(program) == stub

    def relocated(order):
        # Alpha-permute only this real signature's owned sort closure. Other
        # declarations and every operation's authoritative signature stay intact.
        result = pb.Program.from_binary(program.to_binary())
        selected = _required_named_declaration(result, "egglog.core.multiset.map")
        indices: dict[int, int] = {}

        def sort(index):
            if index in indices:
                return indices[index]
            record = pb.Sort.from_binary(program.sorts[index].to_binary())
            kind = record.kind
            assert kind is not None
            if kind.field == "var":
                record.kind = Oneof[Literal["var"], int]("var", order[kind.value])
            elif kind.field == "family":
                kind.value.args = [sort(child) for child in kind.value.args]
            elif kind.field == "func":
                kind.value.params = [sort(child) for child in kind.value.params]
                kind.value.result = sort(kind.value.result)
            else:
                pytest.fail("The actual map signature has an unexpected sort form")
            indices[index] = len(result.sorts)
            result.sorts.append(record)
            return indices[index]

        _, bound = _host_signature(selected)
        for argument in bound.inputs:
            argument.sort = sort(argument.sort)
        bound.output = sort(bound.output)
        selected_owner = _python_bindings(selected).views[0].owner
        assert selected_owner is not None
        assert view.owner is not None
        assert view.owner.kind is not None
        assert view.owner.kind.field == "sort"
        selected_owner.kind = Oneof[Literal["sort"], int]("sort", sort(view.owner.kind.value))
        return result

    reordered = relocated((1, 0))
    reordered_stub = render(reordered)
    assert "class MultiSet(BuiltinExpr, Generic[_MultiSet_T_0]):" in reordered_stub
    assert (
        "def map(self, f: Callable[[_MultiSet_T_0], _MultiSet_map_T0]) -> MultiSet[_MultiSet_map_T0]:" in reordered_stub
    )
    nested = relocated((0, 1))
    selected = next(d for d in nested.declarations if d.kind.value.name == "egglog.core.multiset.map")
    _, bound = _host_signature(selected)
    function = nested.sorts[bound.inputs[0].sort].kind.value
    nested.sorts.append(pb.Sort.from_binary(nested.sorts[bound.inputs[0].sort].to_binary()))
    nested_function = len(nested.sorts) - 1
    function.result = nested_function
    nested.sorts[bound.output].kind.value.args = [nested_function]
    assert (
        "def map(self, f: Callable[[_MultiSet_T_0], UnstableFn[_MultiSet_map_T1, _MultiSet_T_0]])"
        " -> MultiSet[UnstableFn[_MultiSet_map_T1, _MultiSet_T_0]]:" in render(nested)
    )
    malformed = relocated((0, 1))
    selected = next(d for d in malformed.declarations if d.kind.value.name == "egglog.core.multiset.map")
    _, malformed_signature = _host_signature(selected)
    malformed_signature.type_params.append("Unused")
    with pytest.raises(ValueError, match="Unused static signature parameter"):
        render(malformed)
    malformed_signature.type_params.pop()
    selected.bindings.python.views[0].params[0].default_expr = len(malformed.nodes)
    with pytest.raises(ValueError, match="Default root"):
        render(malformed)
    collision = relocated((0, 1))
    scalar = next(d for d in collision.declarations if d.kind.value.name == "String")
    scalar.kind.value.bindings.python.path[-1] = local
    with pytest.raises(ValueError, match="collision"):
        render(collision)
    for direction in (1, -1):
        collision = relocated((0, 1))
        collision.declarations = collision.declarations[::direction]
        assert f"def map(self, f: Callable[[_MultiSet_T_0], {local}]) -> MultiSet[{local}]:" in render(collision)
        member = next(d for d in collision.declarations if d.kind.value.name == "egglog.core.multiset.clear_index")
        member.bindings.python.views[0].path = [local]
        with pytest.raises(ValueError, match="collision"):
            render(collision)
