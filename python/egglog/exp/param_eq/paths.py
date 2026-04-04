"""Shared path helpers for the retained param-eq replication package."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PARAM_EQ_DATA_DIR_ENV = "EGGLOG_PARAM_EQ_DATA_DIR"
PARAM_EQ_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARAM_EQ_DIR.parents[3]
ARTIFACT_DIR = PARAM_EQ_DIR / "artifacts"
GOLDEN_PATH = PARAM_EQ_DIR / "haskell_golden.json"


def default_data_dir() -> Path:
    return (REPO_ROOT.parent / "param-eq-haskell").resolve()


def param_eq_data_dir() -> Path:
    configured = os.environ.get(PARAM_EQ_DATA_DIR_ENV)
    return Path(configured).expanduser().resolve() if configured else default_data_dir()


def llvm_bin_dir() -> Path | None:
    for tool in ("opt", "llvm-config"):
        resolved = shutil.which(tool)
        if resolved is not None:
            return Path(resolved).resolve().parent
    return None
