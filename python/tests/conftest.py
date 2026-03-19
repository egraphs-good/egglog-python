import copy
from importlib import import_module

import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension


@pytest.fixture(autouse=True)
def _reset_conversions():
    conversion = import_module("egglog.conversion")

    old_conversions = copy.copy(conversion.CONVERSIONS)
    old_conversion_decls = copy.copy(conversion._TO_PROCESS_DECLS)
    yield
    conversion.CONVERSIONS = old_conversions
    conversion._TO_PROCESS_DECLS = old_conversion_decls


@pytest.fixture(autouse=True)
def _reset_current_egraph():
    array_api = import_module("egglog.exp.array_api")

    yield
    array_api._CURRENT_EGRAPH = None


class PythonSnapshotExtension(SingleFileSnapshotExtension):
    file_extension = "py"

    def serialize(self, data, **kwargs) -> bytes:
        return str(data).encode()


@pytest.fixture
def snapshot_py(snapshot):
    return snapshot.with_defaults(extension_class=PythonSnapshotExtension)
