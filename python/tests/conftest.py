import copy
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
def class_authoring_without_engine(monkeypatch):
    """Ordinary source authoring and local inspection cannot create an engine."""
    from egglog import bindings  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Class authoring or local inspection accessed a native engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
