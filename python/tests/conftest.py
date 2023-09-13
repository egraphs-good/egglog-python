import copy

import egglog.runtime
import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension


@pytest.fixture(autouse=True)
def reset_conversions():
    old_conversions = copy.copy(egglog.runtime.CONVERSIONS)
    yield
    egglog.runtime.CONVERSIONS = old_conversions


class PythonSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "py"

    def serialize(self, data, **kwargs) -> bytes:
        return str(data).encode()


@pytest.fixture
def snapshot_py(snapshot):
    return snapshot.with_defaults(extension_class=PythonSnapshotExtension)
