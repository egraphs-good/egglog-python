import copy

import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension

import egglog.conversion
import egglog.exp.array_api


@pytest.fixture(autouse=True)
def _reset_conversions():
    old_conversions = copy.copy(egglog.conversion.CONVERSIONS)
    old_conversion_decls = copy.copy(egglog.conversion._TO_PROCESS_DECLS)
    yield
    egglog.conversion.CONVERSIONS = old_conversions
    egglog.conversion._TO_PROCESS_DECLS = old_conversion_decls


@pytest.fixture(autouse=True)
def _reset_current_egraph():
    yield
    egglog.exp.array_api._CURRENT_EGRAPH = None


class PythonSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "py"

    def serialize(self, data, **kwargs) -> bytes:
        return str(data).encode()


@pytest.fixture
def snapshot_py(snapshot):
    return snapshot.with_defaults(extension_class=PythonSnapshotExtension)
