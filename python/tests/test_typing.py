from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("family", "kind", "source_egraph"),
    [
        *(
            pytest.param(family, kind, False, id=f"{kind}-{family}")
            for kind in ("positive", "negative")
            for family in (
                "scalar",
                "pair",
                "vec",
                "map",
                "concrete_members",
                "rational",
                "set",
                "maybe",
                "multiset",
                "extract_multiple",
            )
        ),
        *(
            pytest.param("high_level", case, False, id=case)
            for case in re.findall(
                r"^\[case ([^]]+)\]$",
                (pathlib.Path(__file__).resolve().parents[2] / "test-data/unit/check-high-level.test").read_text(),
                re.MULTILINE,
            )
        ),
        *(
            pytest.param("concrete_members", kind, True, id=f"source-{kind}-concrete_members")
            for kind in ("positive", "negative")
        ),
    ],
)
def test_builtin_installed_types(family, kind, source_egraph, tmp_path):
    """Check actual installed public imports, independent of source MYPYPATH."""
    root = pathlib.Path(__file__).resolve().parents[2]
    if family == "high_level":
        cases = dict(
            re.findall(
                r"^\[case ([^]]+)\]\n(.*?)(?=^\[case |\Z)",
                (root / "test-data/unit/check-high-level.test").read_text(),
                re.MULTILINE | re.DOTALL,
            )
        )
        fixture = tmp_path / f"{kind}.py"
        fixture.write_text(cases[kind])
    else:
        prefix = family if family in {"concrete_members", "extract_multiple"} else f"builtin_{family}"
        fixture = root / "test-data" / "typing" / f"{prefix}_{kind}.py"
    environment = os.environ.copy()
    environment.pop("MYPYPATH", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("MYPY_TEST_PREFIX", None)
    provenance = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import egglog, json; from pathlib import Path; "
                "p = Path(egglog.__file__).resolve().parent; "
                "s = p / 'builtins.py'; "
                "print(json.dumps([str(p), s.read_text() if s.is_file() else None, "
                "(p / 'py.typed').is_file(), (p / 'builtins.pyi').exists()]))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    package, source, has_marker, has_stub = json.loads(provenance.stdout)
    assert "site-packages" in pathlib.Path(package).parts
    assert source is not None, "Install the wheel containing handwritten builtins.py before this gate"
    assert has_marker
    assert not has_stub, "The handwritten Python declarations must be the installed typing source"
    assert source == (root / "python/egglog/builtins.py").read_text()
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
            *(
                [
                    "--shadow-file",
                    str(pathlib.Path(package) / "egraph.py"),
                    str(root / "python/egglog/egraph.py"),
                ]
                if source_egraph
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
    expected: list[tuple[int, str, str | None, str | None]] = []
    for number, line in enumerate(fixture.read_text().splitlines(), 1):
        if "# error: " in line:
            expected.append((number, "error", None, line.split("# error: ", 1)[1]))
        elif family == "high_level" and "# N: " in line:
            expected.append((number, "note", line.split("# N: ", 1)[1], None))
        elif family == "high_level" and "# E: " in line:
            codes = {
                "eqToNotAllowed": "arg-type",
                "functionPrimitiveCostRejected": "type-var",
                "functionPrimitiveRulesetSubsumeRejected": "type-var",
            }
            expected.append((number, "error", line.split("# E: ", 1)[1], codes[kind]))
    diagnostics = [line for line in result.stdout.splitlines() if family == "high_level" or ": error:" in line]
    actual = []
    diagnostic_path = fixture.name if family == "high_level" else str(fixture)
    for line in diagnostics:
        match = re.fullmatch(re.escape(diagnostic_path) + r":(\d+): (error|note): (.*?)(?:  \[([^]]+)\])?", line)
        assert match is not None, line
        actual.append((int(match[1]), match[2], match[3] if family == "high_level" else None, match[4]))
    assert result.returncode == any(kind == "error" for _, kind, _, _ in expected), result.stdout + result.stderr
    assert actual == expected, result.stdout + result.stderr
    assert not result.stderr, result.stderr
