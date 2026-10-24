import copy
import os
import runpy
import sys
from pathlib import Path

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
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
def native_catalog_program():
    """Fresh actual backend descriptors, independent of executable Programs."""
    return pb.BuiltinCatalog.from_binary((Path(__file__).parents[1] / "egglog/_builtin_catalog.pb").read_bytes())


@pytest.fixture
def snapshot_py(snapshot):
    return snapshot.with_defaults(extension_class=PythonSnapshotExtension)


@pytest.fixture
def python_presentation_mapping():
    """Give each authoring test its own frontend map, never native signatures."""
    from egglog._builtin_presentation import DATA  # noqa: PLC0415

    return copy.deepcopy(DATA)


@pytest.fixture
def pair_catalog_program():
    """Signatures stay in the actual native snapshot; Python views are separate."""
    return pb.BuiltinCatalog.from_binary((Path(__file__).parents[1] / "egglog/_builtin_catalog.pb").read_bytes())


@pytest.fixture
def pair_runtime(pair_catalog_program, python_presentation_mapping, monkeypatch, request):
    """Run real wrapper generation over one coherent fixture-owned catalog."""
    from egglog import _catalog, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415
    from egglog._presentation import builtin_owner  # noqa: PLC0415

    catalog = Catalog(builtin_owner(pair_catalog_program, mapping=python_presentation_mapping))
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", dict(runtime._BASE_CLASSES))
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    generated = runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")
    for name in ("i64", "f64", "Pair", "Vec", "ExprValueError", "_pair_value"):
        if hasattr(request.module, name):
            monkeypatch.setattr(request.module, name, generated[name])
    return generated["Pair"]


@pytest.fixture
def vec_catalog_program(pair_catalog_program):
    """Real Vec signatures; fixture-local names live in python_presentation_mapping."""
    return pair_catalog_program


@pytest.fixture
def vec_runtime(vec_catalog_program, python_presentation_mapping, monkeypatch, request):
    """Run ordinary wrapper generation with the proposed fixture-only Vec views."""
    from egglog import _catalog, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415
    from egglog._presentation import builtin_owner  # noqa: PLC0415

    catalog = Catalog(builtin_owner(vec_catalog_program, mapping=python_presentation_mapping))
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
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
    program = pb.BuiltinCatalog.from_binary(path.read_bytes())
    if not any(primitive.name == "egglog.experimental.rational.make" for primitive in program.primitives):
        pytest.skip("Rational native export not packaged or supplied via EGGLOG_TEST_BUILTIN_CATALOG")
    return program


@pytest.fixture
def set_catalog_program():
    """Actual native export; no synthetic Set signatures or views."""
    path = Path(os.environ.get("EGGLOG_TEST_BUILTIN_CATALOG", Path(__file__).parents[1] / "egglog/_builtin_catalog.pb"))
    program = pb.BuiltinCatalog.from_binary(path.read_bytes())
    assert any(primitive.name == "egglog.core.set.of" for primitive in program.primitives)
    return program


@pytest.fixture
def set_runtime(set_catalog_program, python_presentation_mapping, monkeypatch):
    """Exercise catalog-derived Set authoring while forbidding native execution."""
    from egglog import _catalog, bindings, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415
    from egglog._presentation import builtin_owner  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Set source authoring accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    catalog = Catalog(builtin_owner(set_catalog_program, mapping=python_presentation_mapping))
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", {})
    monkeypatch.setattr(runtime, "_CLASS_CACHE", runtime.WeakKeyDictionary())
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    return runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")


@pytest.fixture
def rational_runtime(rational_catalog_program, python_presentation_mapping, monkeypatch):
    """Use real catalog records without replacing package data or a native engine."""
    from egglog import _catalog, bindings, conversion, runtime  # noqa: PLC0415
    from egglog._catalog import Catalog  # noqa: PLC0415
    from egglog._presentation import builtin_owner  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Rational source authoring accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    catalog = Catalog(builtin_owner(rational_catalog_program, mapping=python_presentation_mapping))
    monkeypatch.setattr(_catalog, "builtin_catalog", lambda: catalog)
    monkeypatch.setattr(runtime, "_BASE_CLASSES", {})
    monkeypatch.setattr(runtime, "_CLASS_CACHE", runtime.WeakKeyDictionary())
    monkeypatch.setattr(conversion, "CONVERSIONS", {})
    return runpy.run_path(str(Path(__file__).parents[1] / "egglog/builtins.py"), run_name="egglog.builtins")
