"""Smoke test for running the retained param-eq notebook source in-process."""

from __future__ import annotations

import json
import runpy
from pathlib import Path


def test_replication_notebook_runs_in_process() -> None:
    notebook_source = Path(__file__).with_name("replication.py")
    notebook_output = notebook_source.with_suffix(".ipynb")

    runpy.run_path(str(notebook_source), run_name="__main__")

    payload = json.loads(notebook_output.read_text())
    assert payload["cells"]
    assert any(cell.get("outputs") for cell in payload["cells"] if cell.get("cell_type") == "code")
