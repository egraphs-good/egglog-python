from __future__ import annotations

import sys

from mypy.test import helpers

# Monkeypatch testfile_pyversion so that we can test against the current python version, instead of the default (3.7)


def monkeypatched_testfile_pyversion(path: str) -> tuple[int, int]:
    return sys.version_info[:2]


helpers.testfile_pyversion = monkeypatched_testfile_pyversion


# This is required using the `TypeCheckSuite`, because it calls remove on a number of builtin files.
# we can either make empty versions of them or just monkypatch the function so that it works.

# Monkeypatch find_test_files so that we can "remove" any files that don't really exist.

original_find_test_files = helpers.find_test_files


class ListRemoveAnything(list):
    def remove(self, item):
        pass


def new_find_test_files(*args, **kwargs):
    original_output = original_find_test_files(*args, **kwargs)
    return ListRemoveAnything(original_output)


helpers.find_test_files = new_find_test_files

# Import TypeCheckSuite so it is picked up by pytest.
from mypy.test.testcheck import TypeCheckSuite

__all__ = ["TypeCheckSuite"]
