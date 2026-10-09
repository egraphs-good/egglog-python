from __future__ import annotations

import json
import os
import pathlib
import re
import runpy
import subprocess
import sys

import pytest
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


@pytest.mark.parametrize("family", ["scalar", "pair", "vec", "map", "concrete_members", "rational"])
@pytest.mark.parametrize("kind", ["positive", "negative"])
def test_generated_builtin_installed_types(family, kind, tmp_path):
    """Check actual installed public imports, independent of source MYPYPATH."""
    root = pathlib.Path(__file__).resolve().parents[2]
    prefix = family if family == "concrete_members" else f"generated_{family}"
    fixture = root / "test-data" / "typing" / f"{prefix}_{kind}.py"
    environment = os.environ.copy()
    environment.pop("MYPYPATH", None)
    environment.pop("PYTHONPATH", None)
    provenance = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import egglog, json; from pathlib import Path; "
                "p = Path(egglog.__file__).resolve().parent; "
                "s = p / 'builtins.pyi'; "
                "print(json.dumps([str(p), s.read_text() if s.is_file() else None, (p / 'py.typed').is_file()]))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    package, stub, has_marker = json.loads(provenance.stdout)
    assert "site-packages" in pathlib.Path(package).parts
    assert stub is not None, "Install the wheel containing generated builtins.pyi before this gate"
    assert has_marker
    assert stub == (root / "python/egglog/builtins.pyi").read_text()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            "--no-incremental",
            "--allow-empty-bodies",
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            str(fixture),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = [
        (number, line.split("# error: ", 1)[1])
        for number, line in enumerate(fixture.read_text().splitlines(), 1)
        if "# error: " in line
    ]
    diagnostics = [line for line in result.stdout.splitlines() if ": error:" in line]
    actual = []
    for line in diagnostics:
        match = re.fullmatch(re.escape(str(fixture)) + r":(\d+): error: .*  \[([^]]+)\]", line)
        assert match is not None, line
        actual.append((int(match[1]), match[2]))
    assert result.returncode == bool(expected), result.stdout + result.stderr
    assert actual == expected, result.stdout + result.stderr
    assert not result.stderr, result.stderr


@pytest.mark.parametrize("family", ["scalar", "pair", "vec", "map", "concrete_members", "rational"])
@pytest.mark.parametrize("kind", ["positive", "negative"])
def test_generated_builtin_source_types(family, kind, tmp_path, request):
    """Actual catalog-derived source stub; shadowing cannot replace the installed gate."""
    root = pathlib.Path(__file__).resolve().parents[2]
    stub = root / "python/egglog/builtins.pyi"
    if family == "rational":
        program = request.getfixturevalue("rational_catalog_program")
        render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
        stub = tmp_path / "generated_builtins.pyi"
        stub.write_text(render(program))
    environment = os.environ.copy()
    environment.pop("MYPYPATH", None)
    environment.pop("PYTHONPATH", None)
    installed = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import egglog; from pathlib import Path; print(Path(egglog.__file__).parent / 'builtins.pyi')",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
    ).strip()
    assert "site-packages" in pathlib.Path(installed).parts
    prefix = family if family == "concrete_members" else f"generated_{family}"
    fixture = root / "test-data/typing" / f"{prefix}_{kind}.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            "--no-incremental",
            "--allow-empty-bodies",
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            "--shadow-file",
            installed,
            str(stub),
            str(fixture),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = [
        (number, line.split("# error: ", 1)[1])
        for number, line in enumerate(fixture.read_text().splitlines(), 1)
        if "# error: " in line
    ]
    actual = []
    for line in result.stdout.splitlines():
        if ": error:" not in line:
            continue
        match = re.fullmatch(re.escape(str(fixture)) + r":(\d+): error: .*  \[([^]]+)\]", line)
        assert match is not None, line
        actual.append((int(match[1]), match[2]))
    assert result.returncode == bool(expected), result.stdout + result.stderr
    assert actual == expected, result.stdout + result.stderr
    assert not result.stderr, result.stderr
