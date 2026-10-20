import copy
import os
import runpy
import sys
from pathlib import Path
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof
from syrupy.extensions.single_file import SingleFileSnapshotExtension


@pytest.fixture(autouse=True)
def _reset_conversions():
    from egglog import conversion  # noqa: PLC0415

    old_conversions = copy.copy(conversion.CONVERSIONS)
    yield
    conversion.CONVERSIONS = old_conversions


@pytest.fixture(autouse=True)
def _reset_current_egraph():
    yield
    if (array_api := sys.modules.get("egglog.exp.array_api")) is not None:
        array_api._CURRENT_EGRAPH = None  # type: ignore[attr-defined]  # Reset the loaded module's dynamic context slot.


class PythonSnapshotExtension(SingleFileSnapshotExtension):
    file_extension = "py"

    def serialize(self, data, **kwargs) -> bytes:
        return str(data).encode()


@pytest.fixture
def snapshot_py(snapshot):
    return snapshot.with_defaults(extension_class=PythonSnapshotExtension)


@pytest.fixture
def pair_catalog_program():
    """Proposed presentation only; signatures stay in the real native snapshot."""
    program = pb.Program.from_binary((Path(__file__).parents[1] / "egglog/_builtin_catalog.pb").read_bytes())
    declarations: dict[str, pb.Declaration] = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        declarations[declaration_kind.value.name] = declaration
    family_kind = declarations["Pair"].kind
    assert family_kind is not None
    assert family_kind.field == "host_sort_family"
    family = family_kind.value
    family.bindings = pb.SortBindings(
        python=pb.TypeBinding(path=["egglog", "builtins", "Pair"], type_params=["L", "R"])
    )
    make = declarations["egglog.core.pair.make"]
    make_kind = make.kind
    assert make_kind is not None
    assert make_kind.field == "host_primitive"
    signature_kind = make_kind.value.typing
    assert signature_kind is not None
    assert signature_kind.field == "signature"
    owner = signature_kind.value.output
    assert make.bindings is not None
    make.bindings.python = pb.PythonBindings(
        views=[
            pb.PythonCallable(
                kind=pb.PythonCallKind.INITIALIZER,
                owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", owner)),
                params=[pb.PythonParameter(name="left", core_input=0), pb.PythonParameter(name="right", core_input=1)],
            )
        ]
    )
    for core, public in [("first", "left"), ("second", "right")]:
        member_bindings = declarations[f"egglog.core.pair.{core}"].bindings
        assert member_bindings is not None
        member_bindings.python = pb.PythonBindings(
            views=[
                pb.PythonCallable(
                    kind=pb.PythonCallKind.PROPERTY,
                    path=[public],
                    owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", owner)),
                    receiver=0,
                )
            ]
        )
    return program


@pytest.fixture
def pair_runtime(pair_catalog_program, monkeypatch, request):
    """Run real wrapper generation over one coherent fixture-owned catalog."""
    from egglog import _catalog, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415

    catalog = Catalog(pair_catalog_program)
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", dict(runtime._BASE_CLASSES))
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    generated = runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")
    for name in ("i64", "f64", "Pair", "Vec", "ExprValueError", "_pair_value"):
        if hasattr(request.module, name):
            monkeypatch.setattr(request.module, name, generated[name])
    return generated["Pair"]


@pytest.fixture
def vec_catalog_program(pair_catalog_program):
    """Proposed Vec views over copied actual signatures; never a package export."""
    declarations = {d.kind.value.name: d for d in pair_catalog_program.declarations}
    declarations["Vec"].kind.value.bindings.python = pb.TypeBinding(
        path=["egglog", "builtins", "Vec"], type_params=["T"]
    )
    owner = declarations["egglog.core.vec.of"].kind.value.typing.value.output
    for core, kind, path, receiver, params in [
        ("of", pb.PythonCallKind.INITIALIZER, [], None, [pb.PythonParameter(core_input=0, name="args")]),
        ("empty", pb.PythonCallKind.CLASS_METHOD, ["empty"], None, []),
        ("get", pb.PythonCallKind.METHOD, ["__getitem__"], 0, [pb.PythonParameter(core_input=1, name="index")]),
    ]:
        declarations[f"egglog.core.vec.{core}"].bindings.python = pb.PythonBindings(
            views=[
                pb.PythonCallable(
                    kind=kind,
                    path=path,
                    owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", owner)),
                    receiver=receiver,
                    params=params,
                )
            ]
        )
    return pair_catalog_program


@pytest.fixture
def vec_runtime(vec_catalog_program, monkeypatch, request):
    """Run ordinary wrapper generation with the proposed fixture-only Vec views."""
    from egglog import _catalog, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415

    catalog = Catalog(vec_catalog_program)
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", dict(runtime._BASE_CLASSES))
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    generated = runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")
    for name in ("i64", "f64", "Pair", "Vec", "VecLike", "ExprValueError"):
        if hasattr(request.module, name):
            monkeypatch.setattr(request.module, name, generated[name])
    return generated["Vec"]


@pytest.fixture
def rational_catalog_program():
    """Actual native export, optionally supplied before package regeneration."""
    path = Path(os.environ.get("EGGLOG_TEST_BUILTIN_CATALOG", Path(__file__).parents[1] / "egglog/_builtin_catalog.pb"))
    program = pb.Program.from_binary(path.read_bytes())
    kinds = []
    for declaration in program.declarations:
        kind = declaration.kind
        assert kind is not None
        kinds.append(kind)
    if not any(kind.value.name == "egglog.experimental.rational.make" for kind in kinds):
        pytest.skip("Rational native export not packaged or supplied via EGGLOG_TEST_BUILTIN_CATALOG")
    return program


@pytest.fixture
def set_catalog_program():
    """Actual native export; no synthetic Set signatures or views."""
    path = Path(os.environ.get("EGGLOG_TEST_BUILTIN_CATALOG", Path(__file__).parents[1] / "egglog/_builtin_catalog.pb"))
    program = pb.Program.from_binary(path.read_bytes())
    kinds = []
    for declaration in program.declarations:
        kind = declaration.kind
        assert kind is not None
        kinds.append(kind)
    assert any(kind.value.name == "egglog.core.set.of" for kind in kinds)
    return program


@pytest.fixture
def set_runtime(set_catalog_program, monkeypatch):
    """Exercise catalog-derived Set authoring while forbidding native execution."""
    from egglog import _catalog, bindings, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Set source authoring accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    catalog = Catalog(set_catalog_program)
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", {})
    monkeypatch.setattr(runtime, "_CLASS_CACHE", runtime.WeakKeyDictionary())
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    return runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")


@pytest.fixture
def rational_runtime(rational_catalog_program, monkeypatch):
    """Use real catalog records without replacing package data or a native engine."""
    from egglog import _catalog, bindings, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Rational source authoring accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    catalog = Catalog(rational_catalog_program)
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", {})
    monkeypatch.setattr(runtime, "_CLASS_CACHE", runtime.WeakKeyDictionary())
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    return runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")
