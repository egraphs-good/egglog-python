"""Canonical catalog indexing, not a replacement builtin signature inventory."""

import builtins
import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import UnstableFn, f64, i64
from egglog._builtin_presentation import DATA
from egglog._catalog import Catalog
from egglog._presentation import (
    CallKind,
    CallPresentation,
    Parameter,
    Presentation,
    TypePresentation,
    builtin_owner,
    mapping_views,
    presentation,
)
from egglog._program import Builder, StructuralView, pack
from egglog.egraph import to_runtime_expr
from egglog.pretty import pretty_ref
from egglog.runtime import RuntimeExpr, RuntimeFunction, _class_for_sort
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


def _permute_local_owner(mapping, primitive, order):
    """Alpha-permute one owned presentation root without changing other members."""
    local_mapping = copy.deepcopy(mapping)
    local_sorts = local_mapping["fragment"]["sorts"]
    local_indices: dict[int, int] = {}

    def local_sort(index):
        if index in local_indices:
            return local_indices[index]
        record = copy.deepcopy(mapping["fragment"]["sorts"][index])
        if "var" in record:
            record["var"] = order[record["var"]]
        elif "primitive" in record:
            record["primitive"]["args"] = [local_sort(child) for child in record["primitive"].get("args", [])]
        else:
            pytest.fail("The actual local owner has an unexpected sort form")
        local_indices[index] = len(local_sorts)
        local_sorts.append(record)
        return local_indices[index]

    view = local_mapping["callables"][primitive][0]
    view["owner"] = local_sort(view["owner"])
    return local_mapping


def test_sparse_builtin_views_preserve_explicit_layout_and_enrollment():
    source = pb.BuiltinCatalog.from_binary(
        (Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes()
    )
    semantic = Builder.from_builtins(source).publish()
    templates = Builder.from_program(
        pb.Program.from_json(json.dumps(DATA["fragment"])), environment=(semantic,)
    ).publish()
    expected = mapping_views(source, templates, DATA)
    explicit = copy.deepcopy(DATA)
    sparse = copy.deepcopy(DATA)
    for name, rows in explicit["callables"].items():
        for row, compact, view in zip(rows, sparse["callables"][name], expected["callable", name].views, strict=True):
            row["path"] = list(view.path)
            row["receiver"] = view.receiver
            for parameter, resolved in zip(row.get("params", ()), view.params, strict=True):
                parameter["coreInput"] = resolved.core_input
            if view.kind in {CallKind.METHOD, CallKind.PROPERTY} and view.receiver == 0:
                compact.pop("receiver", None)
            if view.kind in {CallKind.METHOD, CallKind.CLASS_METHOD, CallKind.PROPERTY} and view.path == (
                name.rsplit(".", 1)[-1],
            ):
                compact.pop("path", None)
            for position, (parameter, resolved) in enumerate(zip(compact.get("params", ()), view.params, strict=True)):
                slot = position + (view.receiver is not None and position >= view.receiver)
                if resolved.core_input == slot:
                    parameter.pop("coreInput", None)
    assert mapping_views(source, templates, explicit) == expected
    assert mapping_views(source, templates, sparse) == expected
    del sparse["callables"]["egglog.core.i64.abs"]
    assert ("callable", "egglog.core.i64.abs") not in mapping_views(source, templates, sparse)


def test_pyobject_generated_surface_uses_real_defaults_and_host_policy():
    mapping = copy.deepcopy(DATA)
    from egglog._catalog import builtin_catalog  # noqa: PLC0415

    program = pb.BuiltinCatalog.from_binary(
        (Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes()
    )
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = renderer(program, mapping=mapping)
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
        view = presentation(declaration).views[0]
        assert view.path == ("egglog", "builtins", f"py_{name}")
        assert view.params[0].name == "code"
        assert view.params[0].default is None
        for parameter, expected in zip(view.params[1:], ("globals_", "locals_"), strict=True):
            assert parameter.name == expected
            assert parameter.default is not None
            default = parameter.default.read()
            assert default.kind.field == "call"
            assert default.kind.value.func == "egglog.python.pyobject.dict"
            assert not default.kind.value.args
    # The same real export must remain usable after all arena IDs relocate.
    relocated = _relocate_builtins(program)
    assert sorted(renderer(relocated, mapping=mapping).splitlines()) == sorted(stub.splitlines())


@pytest.mark.parametrize("corruption", ["index", "sort"])
def test_pyobject_generator_rejects_invalid_default(corruption):
    mapping = copy.deepcopy(DATA)

    program = pb.BuiltinCatalog.from_binary(
        (Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes()
    )
    declaration = next(primitive for primitive in program.primitives if primitive.name == "egglog.python.pyobject.eval")
    parameter = mapping["callables"][declaration.name][0]["params"][1]
    if corruption == "index":
        parameter["default"] = len(mapping["fragment"]["nodes"])
    else:
        integer_sort = next(
            i for i, sort in enumerate(mapping["fragment"]["sorts"]) if sort.get("primitive", {}).get("name") == "i64"
        )
        mapping["fragment"]["nodes"][parameter["default"]]["sortId"] = integer_sort
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    expected = "Default root has an invalid index" if corruption == "index" else "Default root has an invalid sort"
    with pytest.raises(ValueError, match=expected):
        renderer(program, mapping=mapping)


def test_set_generated_surface_from_actual_catalog(set_catalog_program):
    mapping = copy.deepcopy(DATA)
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(set_catalog_program, mapping=mapping)
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
    declarations = {d.name: d for d in set_catalog_program.primitives}
    assert len([name for name in declarations if name.startswith("egglog.core.set.")]) == 11
    mapping["callables"][declarations["egglog.core.set.insert"].name][0]["params"][0]["name"] = "item"
    mapping["types"]["Set"]["path"][-1] = "FiniteSet"
    stub = render(set_catalog_program, mapping=mapping)
    assert "def insert(self, item: _FiniteSet_T_0) -> FiniteSet[_FiniteSet_T_0]:" in stub
    assert "FiniteSetLike: TypeAlias = FiniteSet[_FiniteSet_T_0] | set[_FiniteSet_HostT]" in stub


@pytest.mark.parametrize("name", ["set", "SetLike", "_Set_HostT"])
def test_set_generated_protocol_name_collisions(set_catalog_program, name):
    mapping = copy.deepcopy(DATA)
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    family = next(d for d in set_catalog_program.primitive_sorts if d.name == "Vec")
    mapping["types"][family.name]["path"][-1] = name
    with pytest.raises(ValueError, match="collision"):
        render(set_catalog_program, mapping=mapping)


def test_set_stub_uses_relocated_actual_sort_records(set_catalog_program):
    mapping = copy.deepcopy(DATA)

    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    relocated = _relocate_builtins(set_catalog_program)
    assert relocated.sorts != set_catalog_program.sorts
    original = render(set_catalog_program, mapping=mapping)
    reordered = render(relocated, mapping=mapping)
    for line in original.splitlines():
        if "_Set_" in line or "class Set(" in line:
            assert line in reordered


def test_exact_generated_surface_includes_host_and_symbolic_constructors(rational_catalog_program):
    mapping = copy.deepcopy(DATA)
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(rational_catalog_program, mapping=mapping)
    for signature in (
        "def __init__(self, num: i64Like, den: i64Like, /) -> None:",
        "def __add__(self, other: RationalLike) -> Rational:",
        "def __radd__(self, other: RationalLike) -> Rational:",
        "def __init__(self, value: i64Like, /) -> None:",
        "def from_string(cls, s: StringLike) -> BigInt:",
        "def __init__(self, num: BigIntLike, den: BigIntLike, /) -> None:",
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
        assert body.count("def __init__(") == (1 if family == "BigInt" else 2)
        assert f"def value(self) -> {value}:" in body
        assert f"def eval(self) -> {value}:" in body
        assert f"def {hook}(self) ->" in body
        if family != "BigInt":
            assert "def __init__(self, value: int | Fraction, /)" in body
        assert "from_value" not in body
    declaration = next(d for d in rational_catalog_program.primitives if d.name == "egglog.experimental.rational.make")
    mapping["callables"][declaration.name][0]["params"][0]["name"] = "numerator"
    assert "def __init__(self, numerator: i64Like, den: i64Like, /)" in render(
        rational_catalog_program, mapping=mapping
    )


def test_scalar_generated_surface_uses_actual_variadic_providers():
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declarations: dict[str, pb.Primitive] = {}
    for declaration in program.primitives:
        declarations[declaration.name] = declaration
    for name, paths in (("string.concat", ["join", "__add__"]), ("bool.or", ["__or__"])):
        declaration = declarations[f"egglog.core.{name}"]
        signature = _primitive_signature(declaration)
        assert not signature.inputs
        assert len(signature.varargs) == 1
        assert [v["path"][-1] for v in mapping["callables"][declaration.name]] == paths
        assert mapping["callables"][declaration.name][-1]["tailRepetitions"] == 2
    stub = render(program, mapping=mapping)
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
    mapping["callables"][declarations["egglog.core.string.concat"].name][0]["path"][-1] = "concatenate"
    assert "def concatenate(*strings: StringLike) -> String:" in render(program, mapping=mapping)


@pytest.mark.parametrize(
    "mutation", ["fixed", "missing", "duplicate", "owner", "default", "mutation", "huge", "property"]
)
def test_scalar_specialized_stub_rejects_invalid_views(mutation):
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declaration = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.bool.or")
    signature = _primitive_signature(declaration)
    view = mapping["callables"][declaration.name][0]
    if mutation == "fixed":
        signature.inputs = signature.varargs * 2
        signature.varargs.clear()
    elif mutation == "missing":
        view["params"].clear()
    elif mutation == "duplicate":
        view["params"][0]["coreInput"] = 0
    elif mutation == "owner":
        assert "owner" in view
        view["owner"] = next(
            i
            for i, sort in enumerate(mapping["fragment"]["sorts"])
            if sort.get("primitive", {}).get("name") == "String"
        )
    elif mutation == "default":
        view["params"][0]["default"] = 0
    elif mutation == "mutation":
        view["mutates"] = 0
    elif mutation == "property":
        view["form"] = "PROPERTY"
        view["params"].clear()
        view["tailRepetitions"] = 1
    else:
        view["tailRepetitions"] = 2**32 - 1
    expected = {
        "fixed": "Tail specialization requires a variadic function or method",
        "property": "Tail specialization requires a variadic function or method",
        "owner": "Python receiver input must match its owner",
        "default": "Specialized tail inputs cannot have defaults",
        "mutation": "Invalid mutated input",
    }.get(mutation, "Python view must map every core input exactly once")
    with pytest.raises(ValueError, match=expected):
        render(program, mapping=mapping)


@pytest.mark.parametrize("mutation", ["generic", "foreign", "collision"])
def test_scalar_free_stub_rejects_unsupported_views(mutation):
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    declaration = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.string.concat")
    view = mapping["callables"][declaration.name][0]
    if mutation == "generic":
        signature = _primitive_signature(declaration)
        signature.type_param_count += 1
    elif mutation == "foreign":
        view["path"][0] = "other"
    else:
        view["path"][-1] = "String"
    expected = (
        "Static signature requires all generic parameters in its sort closure"
        if mutation == "generic"
        else r"free functions|collision"
    )
    with pytest.raises(ValueError, match=expected):
        render(program, mapping=mapping)


def test_map_generated_surface_from_actual_catalog():
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = render(program, mapping=mapping)
    assert "def from_pairs(cls, entries: Vec[Pair[_Map_K_0, _Map_V_1]]) -> Map[_Map_K_0, _Map_V_1]:" in stub
    assert "def empty(cls) -> Map[_Map_K_0, _Map_V_1]:" in stub
    assert "def insert(self, key: _Map_K_0, value: _Map_V_1) -> Map[_Map_K_0, _Map_V_1]:" in stub
    assert "def __getitem__(self, key: _Map_K_0) -> _Map_V_1:" in stub
    assert "def value(self) -> dict[_Map_K_0, _Map_V_1]:" in stub
    assert "def __iter__(self) -> Iterator[_Map_K_0]:" in stub
    assert "def __contains__(self, key: _Map_K_0) -> bool:" in stub
    assert "MapLike: TypeAlias = Map[_Map_K_0, _Map_V_1] | dict[_Map_HostK, _Map_HostV]" in stub


def test_absent_initializer_keeps_nonabstract_type_token(vec_catalog_program):
    mapping = copy.deepcopy(DATA)
    mapping["callables"]["egglog.core.vec.of"] = []
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(vec_catalog_program, mapping=mapping)
    vec = stub.split("class Vec(", 1)[1].split("class ", 1)[0]
    assert "def __init__(self, _unavailable: Never, /) -> None:" in vec
    assert "abstractmethod" not in stub
    assert "def __init__(self, value: int, /)" in stub
    assert "def __init__(self, left: _Pair_L_0, right: _Pair_R_1)" in stub


def test_vec_uniform_tail_and_classmethod_stub_from_native_signatures(vec_catalog_program):
    mapping = copy.deepcopy(DATA)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = renderer(vec_catalog_program, mapping=mapping)
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
    declarations = {d.name: d for d in vec_catalog_program.primitives}
    mapping["types"]["Vec"]["typeParams"] = ["Element"]
    mapping["callables"][declarations["egglog.core.vec.of"].name][0]["params"][0]["name"] = "items"
    mapping["callables"][declarations["egglog.core.vec.empty"].name][0]["path"] = ["vacant"]
    renamed = renderer(vec_catalog_program, mapping=mapping)
    assert "def __init__(self, *items: _Vec_Element_0) -> None:" in renamed
    assert "def vacant(cls) -> Vec[_Vec_Element_0]:" in renamed


def test_vec_map_view_and_static_signature_come_from_actual_catalog():
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    declaration = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.vec.map")
    signature = _primitive_signature(declaration)
    view = mapping["callables"][declaration.name][0]
    owner = builtin_owner(program, mapping=mapping)
    reference = owner.ref("primitives", program.primitives.index(declaration))
    resolved = presentation(reference).views[0]
    assert view["form"] == "METHOD"
    assert resolved.path == ("map",)
    assert resolved.receiver == 1
    assert [(p.core_input, p.name) for p in resolved.params] == [(0, view["params"][0]["name"])]
    assert "mutates" not in view
    assert "default" not in view["params"][0]
    callback_kind = program.sorts[signature.inputs[0]].kind
    assert callback_kind is not None
    assert callback_kind.field == "func"
    assert "owner" in view
    local_owner = resolved.owner
    assert local_owner is not None
    assert StructuralView(owner.ref("sorts", signature.inputs[1])) == StructuralView(local_owner)
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = render(program, mapping=mapping)
    assert "def map(self, f: Callable[[_Vec_T_0], _Vec_map_T1]) -> Vec[_Vec_map_T1]:" in stub
    assert '_Vec_map_T1 = TypeVar("_Vec_map_T1", bound=BaseExpr)' in stub
    relocated = _relocate_builtins(program)
    assert sorted(render(relocated, mapping=mapping).splitlines()) == sorted(stub.splitlines())


@pytest.mark.parametrize(
    "mutation", ["grouped", "default", "missing", "duplicate", "nonfinal", "receiver", "tail_receiver"]
)
def test_vec_static_tail_rejects_unrepresentable_or_invalid_views(vec_catalog_program, mutation):
    mapping = copy.deepcopy(DATA)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    of = next(d for d in vec_catalog_program.primitives if d.name == "egglog.core.vec.of")
    signature = _primitive_signature(of)
    view = mapping["callables"][of.name][0]
    if mutation == "grouped":
        signature.varargs.append(signature.varargs[0])
    elif mutation == "default":
        view["params"][0]["default"] = 0
    elif mutation == "missing":
        view["params"].clear()
    elif mutation == "duplicate":
        view["params"].append({"coreInput": 0, "name": "again"})
    elif mutation == "nonfinal":
        signature.inputs.append(signature.varargs[0])
        view["params"][0]["coreInput"] = 1
        view["params"].append({"coreInput": 0, "name": "first"})
    elif mutation == "receiver":
        view["receiver"] = 0
    else:
        view["form"] = "METHOD"
        view["path"] = ["tail_receiver"]
        view["receiver"] = 0
        view["params"].clear()
    expected = {
        "default": "A uniform tail must be final and have no default",
        "nonfinal": "A uniform tail must be final and have no default",
        "receiver": "Tail receiver requires a homogeneous variadic method",
        "grouped": "Static callable generation requires a fixed concrete signature",
    }.get(mutation, "Python view must map every core input exactly once")
    with pytest.raises(ValueError, match=expected):
        renderer(vec_catalog_program, mapping=mapping)


def test_uniform_tail_can_follow_fixed_inputs(vec_catalog_program):
    mapping = copy.deepcopy(DATA)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    of = next(d for d in vec_catalog_program.primitives if d.name == "egglog.core.vec.of")
    signature = _primitive_signature(of)
    view = mapping["callables"][of.name][0]
    signature.inputs.append(signature.varargs[0])
    view["params"][0]["coreInput"] = 1
    view["params"].insert(0, {"coreInput": 0, "name": "first"})
    assert "def __init__(self, first: _Vec_T_0, *args: _Vec_T_0) -> None:" in renderer(
        vec_catalog_program, mapping=mapping
    )


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("arguments", ["same", "conflicting", "empty_tail"])
def test_classmethod_pretty_infers_owner_from_uniform_tail(
    vec_catalog_program, python_presentation_mapping, request, fixed, arguments
):
    """Test-only combined view shape, not a claim about native Vec primitives."""
    mapping = python_presentation_mapping
    source = next(d for d in vec_catalog_program.primitives if d.name == "egglog.core.vec.of")
    count = pb.Primitive.from_binary(source.to_binary())
    count.name = "test.vec.count"
    signature = _primitive_signature(count)
    mapping["callables"][count.name] = copy.deepcopy(mapping["callables"][source.name])
    view = mapping["callables"][count.name][0]
    signature.output = next(
        i
        for i, sort in enumerate(vec_catalog_program.sorts)
        if sort.kind is not None and sort.kind.field == "primitive" and sort.kind.value.name == "i64"
    )
    view["form"] = "CLASS_METHOD"
    view["path"] = ["count"]
    view["params"] = [{"coreInput": int(fixed), "name": "items"}]
    if fixed:
        signature.inputs.append(signature.varargs[0])
        view["params"].insert(0, {"coreInput": 0, "name": "first"})
    vec_catalog_program.primitives.append(count)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    prefix = "first: _Vec_T_0, " if fixed else ""
    assert f"def count(cls, {prefix}*items: _Vec_T_0) -> i64:" in renderer(vec_catalog_program, mapping=mapping)
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
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    stub = renderer(pair_catalog_program, mapping=mapping)
    assert "def __init__(self, left: _Pair_L_0, right: _Pair_R_1) -> None:" in stub
    assert "@property\n    def left(self) -> _Pair_L_0:" in stub
    assert "@property\n    def right(self) -> _Pair_R_1:" in stub
    assert "def value(self) -> tuple[_Pair_L_0, _Pair_R_1]:" in stub
    declarations = {d.name: d for d in pair_catalog_program.primitives}
    mapping["types"]["Pair"]["typeParams"] = ["X", "Y"]
    mapping["callables"][declarations["egglog.core.pair.make"].name][0]["params"][0]["name"] = "head"
    mapping["callables"][declarations["egglog.core.pair.first"].name][0]["path"] = ["head"]
    renamed = renderer(pair_catalog_program, mapping=mapping)
    assert "def __init__(self, head: _Pair_X_0, right: _Pair_Y_1) -> None:" in renamed
    assert "def head(self) -> _Pair_X_0:" in renamed


def test_generated_stub_matches_catalog_without_frontend_imports(monkeypatch):
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    original_import = builtins.__import__

    def reject_frontend(name, *args, **kwargs):
        if name == "egglog" or name.startswith("egglog."):
            pytest.fail(f"Generator imported frontend: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_frontend)
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    encoded = (root / "python/egglog/_builtin_catalog.pb").read_bytes()
    program = pb.BuiltinCatalog.from_binary(encoded)
    assert renderer(program, mapping=mapping) == (root / "python/egglog/builtins.pyi").read_text()
    assert program.to_binary() == encoded


def test_function_generated_surface_uses_intrinsic_application_without_a_descriptor():
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    assert all(primitive.name != "egglog.core.fn.apply" for primitive in program.primitives)
    assert "egglog.core.fn.apply" not in mapping["callables"]
    stub = render(program, mapping=mapping)
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
    function_body = stub.split("class UnstableFn(", 1)[1].split("\nclass ", 1)[0]
    assert function_body.count("    @overload\n") == 3
    assert "Any" not in stub
    indexed = Catalog(builtin_owner(program, mapping=mapping))
    assert (("function", ""), "__call__") not in indexed.members
    function_value = UnstableFn[i64, i64](i64.__add__, i64(1))
    applied = function_value(i64(2))
    kind = applied.__egg_ref__.read().kind
    assert kind is not None
    assert kind.field == "apply"
    assert len(kind.value.args) == 1


def test_function_sort_annotations_recurse_through_canonical_records():
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    declarations = {d.name: d for d in program.primitives}
    signatures = {d.name: d.signature for d in program.primitives if d.signature is not None}
    signature = signatures["egglog.core.pair.first"]
    left, right = signatures["egglog.core.pair.make"].inputs
    index = len(program.sorts)
    program.sorts.extend([
        pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[left], result=right))),
        pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[index], result=index))),
        pb.Sort(
            kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef](
                "primitive", pb.PrimitiveSortRef(name="Vec", args=[index + 1])
            )
        ),
    ])
    signature.output = index + 2
    assert (
        "def left(self) -> Vec[UnstableFn[UnstableFn[_Pair_R_1, _Pair_L_0], UnstableFn[_Pair_R_1, _Pair_L_0]]]:"
        in render(program, mapping=mapping)
    )
    i64_sort = signatures["egglog.core.i64.add"].output
    kind = program.sorts[index].kind
    assert kind is not None
    assert kind.field == "func"
    kind.value.params = [i64_sort]
    kind.value.result = i64_sort
    join = declarations["egglog.core.string.concat"]
    mapping["callables"][join.name] = [mapping["callables"][join.name][0]]
    signatures["egglog.core.string.concat"].varargs[0] = index
    assert "def join(*strings: Callable[[i64], i64]) -> String:" in render(program, mapping=mapping)
    kind.value.params.clear()
    assert "def join(*strings: Callable[[], i64]) -> String:" in render(program, mapping=mapping)
    kind.value.result = index
    with pytest.raises(ValueError, match="cyclic signature sort"):
        render(program, mapping=mapping)
    kind.value.result = len(program.sorts)
    with pytest.raises(ValueError, match="Invalid or cyclic signature sort"):
        render(program, mapping=mapping)


def test_stub_generation_uses_canonical_ordinary_method_view():
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.i64.add")
    view = mapping["callables"][addition.name][0]
    view["path"] = ["plus"]
    view["params"][0]["name"] = "rhs"
    assert "def plus(self, rhs: i64Like) -> i64:" in renderer(program, mapping=mapping)
    owner = builtin_owner(program, mapping=mapping)
    method = RuntimeFunction(owner.ref("primitives", program.primitives.index(addition)), 0, to_runtime_expr(i64(1)))
    result = method(rhs=i64(2))
    assert result is not None
    argument = result.__egg_ref__.read().kind.value.args[1]
    assert RuntimeExpr(result.__egg_ref__.owner.ref("nodes", argument)).value == 2


@pytest.mark.parametrize(
    "name", ["int", "_Vec_T_0", "AttributeError", "tuple", "dict", "Never", "_Map_HostK", "MapLike"]
)
def test_stub_generation_rejects_host_and_internal_name_collisions(name):
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    scalar = next(sort for sort in program.primitive_sorts if sort.name == "i64")
    mapping["types"][scalar.name]["path"][-1] = name
    with pytest.raises(ValueError, match="Python static name collision"):
        renderer(program, mapping=mapping)
    mapping["types"][scalar.name]["path"][-1] = "Integer"
    assert "class Integer(BuiltinExpr):" in renderer(program, mapping=mapping)


@pytest.mark.parametrize("mutation", ["output", "receiver", "params", "owner_duplicate", "unknown_binder", "default"])
def test_pair_stub_rejects_invalid_or_unadmitted_generic_views(pair_catalog_program, mutation):
    mapping = copy.deepcopy(DATA)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.name: d for d in pair_catalog_program.primitives}
    make = declarations["egglog.core.pair.make"]
    first = declarations["egglog.core.pair.first"]
    signature = _primitive_signature(make)
    if mutation == "output":
        signature.output = signature.inputs[0]
    elif mutation == "receiver":
        mapping["callables"][first.name][0]["receiver"] = None
    elif mutation == "params":
        mapping["callables"][first.name][0]["params"] = [{"name": "extra", "coreInput": 0}]
    elif mutation == "owner_duplicate":
        view = mapping["callables"][make.name][0]
        owner_args = mapping["fragment"]["sorts"][view["owner"]]["primitive"]["args"]
        owner_args[1] = owner_args[0]
    elif mutation == "unknown_binder":
        signature.type_param_count += 1
    else:
        mapping["callables"][make.name][0]["params"][0]["default"] = 0
    expected = {
        "receiver": "Python view must map every core input exactly once",
        "params": "Python view must map every core input exactly once",
        "owner_duplicate": "Python receiver input must match its owner",
        "unknown_binder": "Static signature requires all generic parameters in its sort closure",
        "default": "Required parameter follows a default",
    }.get(mutation, r"initializer output|static owner|Default root")
    with pytest.raises(ValueError, match=expected):
        renderer(pair_catalog_program, mapping=mapping)


def test_pair_stub_uses_sort_structure_after_relocation(pair_catalog_program):
    mapping = copy.deepcopy(DATA)

    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.name: d for d in pair_catalog_program.primitives}
    make = declarations["egglog.core.pair.make"]
    signature = _primitive_signature(make)
    view = mapping["callables"][make.name][0]
    sorts = mapping["fragment"]["sorts"]
    duplicated = copy.deepcopy(sorts[view["owner"]])
    arguments = duplicated["primitive"]["args"]
    duplicated["primitive"]["args"] = [len(sorts) + 1, len(sorts) + 2]
    view["owner"] = len(sorts)
    sorts.extend([duplicated, *(copy.deepcopy(sorts[index]) for index in arguments)])
    before = renderer(pair_catalog_program, mapping=mapping)
    relocated = _relocate_builtins(pair_catalog_program)
    after = renderer(relocated, mapping=mapping)
    for line in before.splitlines():
        if "_Pair_" in line or "class Pair(" in line:
            assert line in after
    first = declarations["egglog.core.pair.first"]
    first_signature = _primitive_signature(first)
    first_signature.output = signature.inputs[1]
    assert "def left(self) -> _Pair_R_1:" in renderer(pair_catalog_program, mapping=mapping)


def test_pair_stub_rejects_specialized_owner_as_universal_class_view(pair_catalog_program):
    mapping = copy.deepcopy(DATA)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.name: d for d in pair_catalog_program.primitives}
    for name in ("first", "second"):
        mapping["callables"][f"egglog.core.pair.{name}"] = []
    make = declarations["egglog.core.pair.make"]
    signature = _primitive_signature(make)
    signature.type_param_count = 0
    i64_sort = next(
        i
        for i, s in enumerate(pair_catalog_program.sorts)
        if s.kind.field == "primitive" and s.kind.value.name == "i64"
    )
    signature.inputs[0] = i64_sort
    signature.inputs[1] = i64_sort
    pair_catalog_program.sorts[signature.output].kind.value.args = [i64_sort, i64_sort]
    view = mapping["callables"][make.name][0]
    local_i64 = next(
        i for i, sort in enumerate(mapping["fragment"]["sorts"]) if sort.get("primitive", {}).get("name") == "i64"
    )
    mapping["fragment"]["sorts"][view["owner"]]["primitive"]["args"] = [local_i64, local_i64]
    with pytest.raises(ValueError, match="distinct in-range family parameters"):
        renderer(pair_catalog_program, mapping=mapping)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("receiver", "Python view must map every core input exactly once"),
        ("tail", "fixed concrete signature"),
        ("collision", "Python member collision"),
        ("missing_output", "Python callable signature has no output"),
        ("numeric_keyword", "Numeric forwarding requires one required 'other' parameter"),
        ("numeric_default", "Numeric forwarding requires one required 'other' parameter"),
    ],
)
def test_stub_generation_rejects_invalid_or_unmigrated_shapes(mutation, message):
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.i64.add")
    primitive = addition
    signature = _primitive_signature(primitive)
    view = mapping["callables"][addition.name][0]
    if mutation == "receiver":
        view["receiver"] = 1
        view["params"][0]["coreInput"] = 1
    elif mutation == "tail":
        signature.varargs = [1, 0]
        view["params"].append({"coreInput": len(signature.inputs), "name": "rest"})
    elif mutation == "collision":
        mapping["callables"][addition.name].append(copy.deepcopy(view))
    elif mutation == "missing_output":
        signature.clear_field("output")
    elif mutation == "numeric_keyword":
        view["params"][0]["name"] = "rhs"
    elif mutation == "numeric_default":
        fragment = mapping["fragment"]
        integer = next(i for i, sort in enumerate(fragment["sorts"]) if sort.get("primitive", {}).get("name") == "i64")
        view["params"][0]["default"] = len(fragment["nodes"])
        fragment["nodes"].append({"sortId": integer, "primitiveValue": {"i64": "2"}})
    with pytest.raises(ValueError, match=message):
        renderer(program, mapping=mapping)


@pytest.mark.parametrize(
    ("method", "argument", "output"),
    [("__lshift__", "f64", "i64"), ("__lt__", "i64", "Unit"), ("__lt__", "i64", "i64")],
)
def test_numeric_forward_signature_does_not_imply_reflection(method, argument, output):
    """Mutated descriptor shapes test emission policy, not native availability."""
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(d for d in program.primitives if d.name == "egglog.core.i64.add")
    signature = addition.signature
    assert signature is not None
    sorts = {
        s.kind.value.name: i
        for i, s in enumerate(program.sorts)
        if s.kind is not None and isinstance(s.kind.value, pb.PrimitiveSortRef)
    }
    signature.inputs[1] = sorts[argument]
    signature.output = sorts[output]
    for key, views in mapping["callables"].items():
        if key != addition.name:
            views[:] = [view for view in views if view.get("path") != [method]]
    mapping["callables"][addition.name][0]["path"] = [method]
    stub = renderer(program, mapping=mapping)
    body = stub.split("class i64(", 1)[1].split("\nclass ", 1)[0]
    assert f"def {method}(self, other: {argument}Like) -> {output}:" in body
    assert f"def __r{method[2:]}(" not in body


def test_bignum_generated_surface_uses_actual_forward_signatures(rational_catalog_program):
    mapping = copy.deepcopy(DATA)
    renderer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    declarations = {d.name: d for d in rational_catalog_program.primitives}
    assert len([name for name in declarations if name.startswith("egglog.core.bigrat.")]) == 24
    assert len([name for name in declarations if name.startswith("egglog.core.bigint.")]) == 26
    stub = renderer(rational_catalog_program, mapping=mapping)
    integer = stub.split("class BigInt(", 1)[1].split("\nclass ", 1)[0]
    ratio = stub.split("class BigRat(", 1)[1].split("\nclass ", 1)[0]
    catalog = Catalog(builtin_owner(rational_catalog_program, mapping=mapping))
    for member in ("numer", "denom"):
        assert f"@property\n    def {member}(self) -> BigInt:" in ratio
        view = presentation(catalog.definitions["callable", declarations[f"egglog.core.bigrat.{member}"].name]).views[0]
        assert view.kind == CallKind.PROPERTY
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
    mapping = copy.deepcopy(DATA)
    root = Path(__file__).resolve().parents[2]
    renderer = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    addition = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.i64.add")
    mapping["callables"][addition.name][0]["path"] = ["plus"]
    parameter = mapping["callables"][addition.name][0]["params"][0]
    fragment = mapping["fragment"]
    scalar_sort = next(i for i, sort in enumerate(fragment["sorts"]) if sort.get("primitive", {}).get("name") == "i64")
    literal = len(fragment["nodes"])
    fragment["nodes"].extend([
        {"sortId": scalar_sort, "primitiveValue": {"i64": "2"}},
        {"sortId": scalar_sort, "call": {"func": "egglog.core.i64.add", "args": [literal, literal]}},
    ])
    parameter["default"] = literal + 1
    assert "def plus(self, other: i64Like = ...) -> i64:" in renderer(program, mapping=mapping)
    owner = builtin_owner(program, mapping=mapping)
    method = RuntimeFunction(owner.ref("primitives", program.primitives.index(addition)), 0, to_runtime_expr(i64(1)))
    result = method()
    assert result is not None
    argument = result.__egg_ref__.read().kind.value.args[1]
    assert result.__egg_ref__.owner.ref("nodes", argument).read().kind.value.func == "egglog.core.i64.add"
    parameter.pop("default", None)
    assert "other: i64Like = ..." not in renderer(program, mapping=mapping)
    parameter["default"] = len(fragment["nodes"])
    with pytest.raises(ValueError, match="Default root"):
        renderer(program, mapping=mapping)


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
    malformed = pb.BuiltinCatalog.from_binary(old_binary)
    addition = next(primitive for primitive in malformed.primitives if primitive.name == "egglog.core.i64.add")
    signature = _primitive_signature(addition)
    signature.inputs.clear()
    invalid = tmp_path / "invalid.pb"
    invalid.write_bytes(malformed.to_binary())
    command[4] = str(invalid)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Tail receiver requires a homogeneous variadic method" in result.stderr
    assert binary.read_bytes() == old_binary
    assert stub.read_bytes() == old_stub


def catalog_fixture():
    """Pair provider descriptors with independent Python choices and owner roots."""
    catalog = pb.BuiltinCatalog(
        sorts=[
            pb.Sort(
                kind=Oneof[Literal["primitive"], pb.PrimitiveSortRef]("primitive", pb.PrimitiveSortRef(name="Number"))
            )
        ],
        primitive_sorts=[pb.PrimitiveSort(name="Number")],
        primitives=[
            pb.Primitive(
                name="example.number.add",
                signature=pb.GenericSignature(inputs=[0, 0], output=0),
            )
        ],
    )
    mapping = {
        "types": {"Number": {"path": ["example", "Number"]}},
        "callables": {
            "example.number.add": [
                {
                    "form": "METHOD",
                    "path": ["__add__"],
                    "owner": 0,
                    "receiver": 0,
                    "params": [{"coreInput": 1, "name": "other"}],
                }
            ]
        },
        "fragment": {"sorts": [{"primitive": {"name": "Number"}}], "nodes": []},
    }
    return catalog, mapping


def test_catalog_indexes_records_without_evaluating_or_retyping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog loading cannot create an engine")

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)
    source, mapping = catalog_fixture()
    catalog = Catalog(builtin_owner(source, mapping=mapping))
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
    source, mapping = catalog_fixture()
    encoded = source.to_binary()
    catalog = Catalog(builtin_owner(source, mapping=mapping))
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
    program, mapping = catalog_fixture()
    source = Catalog(builtin_owner(program, mapping=mapping))
    scoped = Catalog.from_refs([*source.roots, source.roots[0]])
    assert scoped.roots == source.roots
    assert scoped.members == source.members
    scoped.members.clear()
    scoped.reindex()
    assert scoped.members == source.members
    assert all(ref.owner is source.owner for ref in scoped.roots)


@pytest.mark.parametrize("generic", [False, True])
def test_nominal_and_primitive_sort_member_owners_recover_independently(generic):
    descriptors, mapping = catalog_fixture()
    signature = _primitive_signature(descriptors.primitives[0])
    signature.inputs = [0]
    mapping["callables"]["example.number.add"][0].update(path=["tag"], params=[])
    if generic:
        descriptors.primitive_sorts[0].arity = 1
        signature.type_param_count = 1
        descriptors.sorts[0].kind.value.args = [1]
        descriptors.sorts.append(pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
        mapping["fragment"]["sorts"][0]["primitive"]["args"] = [1]
        mapping["fragment"]["sorts"].append({"var": 0})
    ambient = builtin_owner(descriptors, mapping=mapping)
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
    descriptors, mapping = catalog_fixture()
    ambient = builtin_owner(descriptors, mapping=mapping)
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
    source, mapping = catalog_fixture()
    duplicate = pb.Primitive.from_binary(source.primitives[0].to_binary())
    duplicate.name = "example.other.add"
    source.primitives.append(duplicate)
    mapping["callables"][duplicate.name] = copy.deepcopy(mapping["callables"]["example.number.add"])
    with pytest.raises(ValueError, match="binding collision"):
        Catalog(builtin_owner(source, mapping=mapping))


@pytest.mark.parametrize("name", ["1.0", "not-a-python-identifier", "class"])
def test_runtime_type_names_do_not_weaken_static_generator_identifiers(name):
    root = Path(__file__).resolve().parents[2]
    source = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    mapping = copy.deepcopy(DATA)
    mapping["types"]["i64"]["path"][-1] = name
    path = tuple(mapping["types"]["i64"]["path"])
    catalog = Catalog(builtin_owner(source, mapping=mapping))
    reference = catalog.types[path]
    view = presentation(reference).type_view
    assert view is not None
    assert view.path[-1] == name
    with pytest.raises(TypeError, match="Builtin descriptors are ambient references, not Program roots"):
        pack(catalog.roots)
    restored = Catalog.from_refs(catalog.roots)
    restored_view = presentation(restored.types[path]).type_view
    assert restored_view is not None
    assert restored_view.path[-1] == name
    assert path not in Catalog(Builder.from_builtins(source).publish()).types
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    with pytest.raises(ValueError, match="Invalid Python identifier"):
        render(source, mapping=mapping)
    source.primitive_sorts.append(pb.PrimitiveSort(name="different.core"))
    mapping["types"]["different.core"] = {"path": list(path)}
    with pytest.raises(ValueError, match="Python binding collision"):
        Catalog(builtin_owner(source, mapping=mapping))


@pytest.mark.parametrize(
    "path", [[], [""], ["example", ""], ["", "Number"], ["class", "Number"], ["bad-name", "Number"]]
)
def test_type_paths_keep_nonempty_terminal_and_identifier_qualifiers(path):
    source, mapping = catalog_fixture()
    mapping["types"]["Number"]["path"] = path
    with pytest.raises(ValueError, match="Python path"):
        Catalog(builtin_owner(source, mapping=mapping))


@pytest.mark.parametrize("name", ["", "123", "eval_mode_unsafe-seminaive", "class", "@internal"])
def test_runtime_free_function_names_preserve_exact_callee_and_core_order(monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog names and calls cannot create an engine")

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)
    ambient = builtin_owner(
        pb.BuiltinCatalog.from_binary((Path(__file__).parents[1] / "egglog/_builtin_catalog.pb").read_bytes())
    )
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
    source, mapping = catalog_fixture()
    mapping["callables"]["example.number.add"] = [
        {
            "form": "FUNCTION",
            "path": path,
            "params": [{"coreInput": 0, "name": "left"}, {"coreInput": 1, "name": "right"}],
        }
    ]
    with pytest.raises(ValueError, match="Python path"):
        Catalog(builtin_owner(source, mapping=mapping))


@pytest.mark.parametrize("name", ["", "not-a-python-identifier", "class"])
def test_runtime_free_names_do_not_weaken_static_generator_identifiers(name):
    root = Path(__file__).resolve().parents[2]
    source = pb.BuiltinCatalog.from_binary((root / "python/egglog/_builtin_catalog.pb").read_bytes())
    mapping = copy.deepcopy(DATA)
    view = mapping["callables"]["egglog.core.string.concat"][0]
    view["path"][-1] = name
    assert tuple(view["path"]) in Catalog(builtin_owner(source, mapping=mapping)).functions
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    with pytest.raises(ValueError, match="Invalid Python identifier"):
        render(source, mapping=mapping)


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
    source, mapping = catalog_fixture()
    primitive = source.primitives[0]
    signature = _primitive_signature(primitive)
    signature.inputs.clear()
    mapping["callables"]["example.number.add"] = [{"form": "CONSTANT", "path": ["constant"]}]
    view = mapping["callables"]["example.number.add"][0]
    match invalid:
        case "input":
            signature.inputs.append(0)
        case "generic":
            signature.type_param_count += 1
        case "tail":
            signature.varargs.append(0)
        case "receiver":
            view["receiver"] = 0
        case "owner":
            view["owner"] = 0
        case "params":
            view["params"] = [{"coreInput": 0, "name": "x"}]
        case "mutates":
            view["mutates"] = 0
    expected = {
        "input": "Python view must map every core input exactly once",
        "tail": "Python view must map every core input exactly once",
        "receiver": "Tail receiver requires a homogeneous variadic method",
        "params": "Python view must map every core input exactly once",
        "mutates": "Invalid mutated input",
    }.get(invalid, r"Python (constant|member)")
    with pytest.raises(ValueError, match=expected):
        Catalog(builtin_owner(source, mapping=mapping))


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
    source, mapping = catalog_fixture()
    signature = _primitive_signature(source.primitives[0])
    signature.inputs.clear()
    mapping["callables"]["example.number.add"] = [{"form": "CONSTANT", "path": path}]
    if valid:
        assert tuple(path) in Catalog(builtin_owner(source, mapping=mapping)).constants
    else:
        with pytest.raises(ValueError, match="Invalid Python constant path"):
            Catalog(builtin_owner(source, mapping=mapping))


def test_multiset_generated_surface_from_actual_catalog():
    mapping = copy.deepcopy(DATA)

    program = pb.BuiltinCatalog.from_binary(
        (Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes()
    )
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    stub = render(program, mapping=mapping)
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
    declarations: dict[str, pb.Primitive] = {}
    for declaration in program.primitives:
        declarations[declaration.name] = declaration
    assert len([name for name in declarations if name.startswith("egglog.core.multiset.")]) == 27
    assert "def empty(" not in body
    assert "def map(self, f: Callable[[_MultiSet_T_0], _MultiSet_map_T1]) -> MultiSet[_MultiSet_map_T1]:" in body
    mapping["types"]["MultiSet"]["path"][-1] = "Bag"
    view = mapping["callables"][declarations["egglog.core.multiset.remove_swapped"].name][0]
    view["path"][-1] = "take_one"
    view["params"][0]["name"], view["params"][1]["name"] = "item", "values"
    renamed = render(program, mapping=mapping)
    assert "BagLike: TypeAlias = Bag[_Bag_T_0] | tuple[_Bag_HostT, ...]" in renamed
    assert "def take_one(item: _take_one_T0, values: Bag[_take_one_T0]) -> Bag[_take_one_T0]:" in renamed


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("unused", "Static signature requires all generic parameters in its sort closure"),
        ("unbound", "Static signature requires all generic parameters in its sort closure"),
        ("default", "Default root"),
        ("tail", "Generic static free functions require fixed signatures"),
        ("collision", "collision"),
    ],
)
def test_multiset_generic_free_static_guards(corruption, message):
    mapping = copy.deepcopy(DATA)

    program = pb.BuiltinCatalog.from_binary(
        (Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes()
    )
    declarations: dict[str, pb.Primitive] = {}
    for declaration in program.primitives:
        declarations[declaration.name] = declaration
    declaration = declarations["egglog.core.multiset.remove_swapped"]
    signature = _primitive_signature(declaration)
    if corruption == "unused":
        signature.type_param_count += 1
    elif corruption == "unbound":
        signature.inputs[0] = len(program.sorts)
        program.sorts.append(pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    elif corruption == "default":
        mapping["callables"][declaration.name][0]["params"][0]["default"] = len(mapping["fragment"]["nodes"])
    elif corruption == "tail":
        signature.varargs.append(signature.inputs[0])
        mapping["callables"][declaration.name][0]["params"].append({
            "coreInput": len(signature.inputs),
            "name": "rest",
        })
    else:
        mapping["types"]["MultiSet"]["path"][-1] = "_multiset_remove_swapped_T0"
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    with pytest.raises(ValueError, match=message):
        render(program, mapping=mapping)


def test_multiset_method_local_binders_from_actual_catalog():
    mapping = copy.deepcopy(DATA)
    from os import environ  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    source = Path(environ.get("EGGLOG_TEST_HIGHER_ORDER_CATALOG", root / "python/egglog/_builtin_catalog.pb"))
    program = pb.BuiltinCatalog.from_binary(source.read_bytes())
    declaration = next(primitive for primitive in program.primitives if primitive.name == "egglog.core.multiset.map")
    signature = _primitive_signature(declaration)
    view = mapping["callables"][declaration.name][0]
    assert signature.type_param_count == 2
    assert "owner" in view
    owner = mapping["fragment"]["sorts"][view["owner"]]["primitive"]
    assert owner["name"] == "MultiSet"
    assert len(owner["args"]) == 1
    assert mapping["fragment"]["sorts"][owner["args"][0]] == {"var": 0}
    callback = program.sorts[signature.inputs[0]]
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
    stub = render(program, mapping=mapping)
    local = "_MultiSet_map_T1"
    assert f'{local} = TypeVar("{local}", bound=BaseExpr)' in stub
    assert f"def map(self, f: Callable[[_MultiSet_T_0], {local}]) -> MultiSet[{local}]:" in stub
    assert "class MultiSet(BuiltinExpr, Generic[_MultiSet_T_0]):" in stub
    assert "Any" not in stub
    assert "Callable[...," not in stub

    def relocated(order):
        # Alpha-permute only this real signature's owned sort closure. Other
        # declarations and every operation's authoritative signature stay intact.
        result = pb.BuiltinCatalog.from_binary(program.to_binary())
        selected = next(primitive for primitive in result.primitives if primitive.name == "egglog.core.multiset.map")
        indices: dict[int, int] = {}

        def sort(index):
            if index in indices:
                return indices[index]
            record = pb.Sort.from_binary(program.sorts[index].to_binary())
            kind = record.kind
            assert kind is not None
            if kind.field == "var":
                record.kind = Oneof[Literal["var"], int]("var", order[kind.value])
            elif kind.field == "primitive":
                kind.value.args = [sort(child) for child in kind.value.args]
            elif kind.field == "func":
                kind.value.params = [sort(child) for child in kind.value.params]
                kind.value.result = sort(kind.value.result)
            else:
                pytest.fail("The actual map signature has an unexpected sort form")
            indices[index] = len(result.sorts)
            result.sorts.append(record)
            return indices[index]

        bound = _primitive_signature(selected)
        bound.inputs = [sort(argument) for argument in bound.inputs]
        bound.output = sort(bound.output)
        return result, _permute_local_owner(mapping, selected.name, order)

    reordered, reordered_mapping = relocated((1, 0))
    reordered_stub = render(reordered, mapping=reordered_mapping)
    assert "class MultiSet(BuiltinExpr, Generic[_MultiSet_T_0]):" in reordered_stub
    assert (
        "def map(self, f: Callable[[_MultiSet_T_0], _MultiSet_map_T0]) -> MultiSet[_MultiSet_map_T0]:" in reordered_stub
    )
    nested, nested_mapping = relocated((0, 1))
    selected = next(d for d in nested.primitives if d.name == "egglog.core.multiset.map")
    bound = _primitive_signature(selected)
    function = nested.sorts[bound.inputs[0]].kind.value
    nested.sorts.append(pb.Sort.from_binary(nested.sorts[bound.inputs[0]].to_binary()))
    nested_function = len(nested.sorts) - 1
    function.result = nested_function
    nested.sorts[bound.output].kind.value.args = [nested_function]
    assert (
        "def map(self, f: Callable[[_MultiSet_T_0], UnstableFn[_MultiSet_map_T1, _MultiSet_T_0]])"
        " -> MultiSet[UnstableFn[_MultiSet_map_T1, _MultiSet_T_0]]:" in render(nested, mapping=nested_mapping)
    )
    malformed, malformed_mapping = relocated((0, 1))
    selected = next(d for d in malformed.primitives if d.name == "egglog.core.multiset.map")
    malformed_signature = _primitive_signature(selected)
    malformed_signature.type_param_count += 1
    with pytest.raises(ValueError, match="Static signature requires all generic parameters in its sort closure"):
        render(malformed, mapping=malformed_mapping)
    malformed_signature.type_param_count -= 1
    malformed_mapping["callables"][selected.name][0]["params"][0]["default"] = len(
        malformed_mapping["fragment"]["nodes"]
    )
    with pytest.raises(ValueError, match="Default root"):
        render(malformed, mapping=malformed_mapping)
    collision, collision_mapping = relocated((0, 1))
    scalar = next(d for d in collision.primitive_sorts if d.name == "String")
    collision_mapping["types"][scalar.name]["path"][-1] = local
    with pytest.raises(ValueError, match="collision"):
        render(collision, mapping=collision_mapping)
    for direction in (1, -1):
        collision, collision_mapping = relocated((0, 1))
        collision.primitives = collision.primitives[::direction]
        assert f"def map(self, f: Callable[[_MultiSet_T_0], {local}]) -> MultiSet[{local}]:" in render(
            collision, mapping=collision_mapping
        )
        member = next(d for d in collision.primitives if d.name == "egglog.core.multiset.clear_index")
        collision_mapping["callables"][member.name][0]["path"] = [local]
        with pytest.raises(ValueError, match="collision"):
            render(collision, mapping=collision_mapping)
