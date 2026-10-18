from __future__ import annotations

import ast
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


@pytest.mark.parametrize("defect", [None, "missing", "annotation"])
def test_generated_private_builtin_helpers_match_runtime_annotations(defect, monkeypatch, tmp_path):
    from egglog_proto.egglog.v1 import egglog_pb as pb  # noqa: PLC0415

    root = pathlib.Path(__file__).resolve().parents[2]
    package = root / "python/egglog"
    implementation = ast.parse((package / "builtins.py").read_text())
    pretty = ast.parse((package / "pretty.py").read_text())
    names = {
        alias.name
        for node in ast.walk(pretty)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "builtins"
        for alias in node.names
        if alias.name.startswith("_")
    }
    helpers = {
        node.name: node for node in implementation.body if isinstance(node, ast.FunctionDef) and node.name in names
    }
    render = runpy.run_path(str(root / "scripts/generate_builtin_catalog.py"))["render_stubs"]
    program = pb.Program.from_binary((package / "_builtin_catalog.pb").read_bytes())
    if defect is not None:
        helper = helpers[min(names)]
        if defect == "missing":
            implementation.body.remove(helper)
        else:
            helper.args.args[0].annotation = None
        (tmp_path / "builtins.py").write_text(ast.unparse(implementation))
        monkeypatch.setitem(render.__globals__, "_PACKAGE", tmp_path)
        with pytest.raises(ValueError, match="definitions are missing" if defect == "missing" else "fully annotated"):
            render(program)
        return
    rendered = render(program)
    assert rendered == (package / "builtins.pyi").read_text()
    stub = ast.parse(rendered)
    signatures = {
        node.name: node for node in stub.body if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
    }
    assert signatures.keys() == names
    for name, source in helpers.items():
        signature = signatures[name]
        assert ast.dump(signature.args) == ast.dump(source.args)
        assert signature.returns is not None
        assert source.returns is not None
        assert ast.dump(signature.returns) == ast.dump(source.returns)
        body = signature.body[0]
        assert isinstance(body, ast.Expr)
        assert ast.literal_eval(body.value) is Ellipsis
    exports = next(
        ast.literal_eval(node.value)
        for node in stub.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert names.isdisjoint(exports)


@pytest.mark.parametrize(
    "family", ["scalar", "pair", "vec", "map", "concrete_members", "rational", "set", "maybe", "multiset"]
)
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


@pytest.mark.parametrize(
    "family", ["scalar", "pair", "vec", "map", "concrete_members", "rational", "set", "maybe", "multiset"]
)
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
    if family == "vec":
        extended = tmp_path / fixture.name
        extra = (
            """
from egglog import String

def stringify(value: i64) -> String: ...
def atomize(value: i64) -> Atom: ...
assert_type(owned.map(stringify), Vec[String])
assert_type(owned.map(atomize), Vec[Atom])
assert_type(owned.map(lambda value: value + 1), Vec[i64])
"""
            if kind == "positive"
            else """
def wrong_input(value: f64) -> i64: ...
def wrong_output(value: i64) -> int: ...
value.map(wrong_input)  # error: arg-type
value.map(wrong_output)  # error: type-var
value.map()  # error: call-arg
value.map(lambda item: item, i64(1))  # error: call-arg
mapped = value.map(lambda item: item)
wrong_mapped: Vec[f64] = mapped  # error: assignment
"""
        )
        extended.write_text(fixture.read_text() + extra)
        fixture = extended
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
            "--show-absolute-path",
            "--shadow-file",
            installed,
            str(stub),
            *(
                [
                    "--shadow-file",
                    str(pathlib.Path(installed).with_name("egraph.py")),
                    str(root / "python/egglog/egraph.py"),
                ]
                if family == "concrete_members"
                else []
            ),
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
