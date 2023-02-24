import os

ROOT_DIR = os.path.dirname(__file__)
# So that it finds the local typings
os.environ["MYPYPATH"] = os.path.join(ROOT_DIR, "python")


# Add mypy's test searcher so we can write test files to verify type checking
pytest_plugins = ["mypy.test.data"]
# Set this to the root directory so it finds the `test-data` directory
os.environ["MYPY_TEST_PREFIX"] = ROOT_DIR
