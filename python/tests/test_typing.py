# Monkeypatch find_test_files so that we can "remove" any files that don't really exist.

# This is required using the `TypeCheckSuite`, because it calls remove on a number of builtin files.
# we can either make empty versions of them or just monkypatch the function so that it works.
from mypy.test import helpers

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
