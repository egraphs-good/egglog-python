import copy
import sys

import pytest
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
        array_api._CURRENT_EGRAPH = None


class PythonSnapshotExtension(SingleFileSnapshotExtension):
    file_extension = "py"

    def serialize(self, data, **kwargs) -> bytes:
        return str(data).encode()


@pytest.fixture
def snapshot_py(snapshot):
    return snapshot.with_defaults(extension_class=PythonSnapshotExtension)
