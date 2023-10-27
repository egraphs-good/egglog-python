import os
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
# So that it finds the local typings
os.environ["MYPYPATH"] = str(ROOT_DIR / "python")


# Add mypy's test searcher so we can write test files to verify type checking
pytest_plugins = ["mypy.test.data"]
# Set this to the root directory so it finds the `test-data` directory
os.environ["MYPY_TEST_PREFIX"] = str(ROOT_DIR)
