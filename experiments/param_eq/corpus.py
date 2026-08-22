"""Load the author-supplied Param-Eq corpus without copying it into this repository."""

from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATASETS = ("pagie", "kotanchek")
RAW_ALGORITHMS = ("Bingo", "EPLEX", "FEAT", "GOMEA", "Operon", "SBP", "SRjl")
DROP_INDEXES: dict[str, set[int]] = {"pagie": {16, 162}, "kotanchek": {1}}
ALGORITHM_RENAMES = {"GOMEA": "GP-GOMEA", "SRjl": "PySR"}


class ArchiveLayoutError(ValueError):
    """Raised when the external research archive is absent or incomplete."""


@dataclass(frozen=True)
class CorpusRow:
    """One in-memory source expression; source text is never written to tracked output."""

    row_id: str
    dataset: str
    raw_index: int
    algorithm_raw: str
    algorithm: str
    algorithm_row: int
    input_kind: str
    source: str
    source_n_rank: float


def external_archive_root(root: Path | None = None) -> Path:
    """Resolve and validate the external archive root."""
    if root is None:
        configured = os.environ.get("EGGLOG_PARAM_EQ_DATA_DIR")
        if not configured:
            msg = "Set EGGLOG_PARAM_EQ_DATA_DIR to the private param-eq-haskell archive checkout."
            raise ArchiveLayoutError(msg)
        root = Path(configured)
    resolved = root.expanduser().resolve()
    required = [resolved / "results" / f"{dataset}_table_counts.csv" for dataset in DATASETS]
    required.extend(resolved / "results" / f"{dataset}_results" for dataset in DATASETS)
    required.extend(resolved / "results" / directory for directory in ("exprs", "exprs_simpl"))
    missing = [path.relative_to(resolved).as_posix() for path in required if not path.exists()]
    if missing:
        msg = f"External Param-Eq archive is incomplete; missing: {', '.join(missing)}"
        raise ArchiveLayoutError(msg)
    return resolved


def should_keep_row(dataset: str, raw_index: int, algorithm: str, n_rank: str | None) -> bool:
    """Apply the documented retained-paper cleaning policy."""
    return algorithm != "FEAT" and raw_index not in DROP_INDEXES[dataset] and bool((n_rank or "").strip())


def _read_expression_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_sympy_by_algorithm(path: Path) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["algorithm"].strip()].append(row["expr_sympy"].strip())
    return dict(grouped)


def _raw_index(row: dict[str, str | None]) -> int:
    for key in ("raw_index", "", "Unnamed: 0"):
        value = row.get(key)
        if value is not None and value != "":
            return int(value)
    msg = "A table-count row has no raw index column."
    raise ArchiveLayoutError(msg)


def load_corpus_rows(  # noqa: C901
    root: Path | None = None,
    *,
    dataset: str | None = None,
    algorithm: str | None = None,
    input_kind: str | None = None,
    limit: int | None = None,
) -> list[CorpusRow]:
    """Load retained rows in stable order while keeping expressions in memory only."""
    archive = external_archive_root(root)
    if dataset is not None and dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if input_kind is not None and input_kind not in {"original", "sympy"}:
        raise ValueError(f"Unknown input kind: {input_kind}")

    rows: list[CorpusRow] = []
    for current_dataset in DATASETS:
        if dataset is not None and current_dataset != dataset:
            continue
        results_root = archive / "results"
        sympy_by_algorithm = _read_sympy_by_algorithm(results_root / f"{current_dataset}_results")
        originals = {
            raw_algorithm: _read_expression_lines(results_root / "exprs" / f"{raw_algorithm}_exprs_{current_dataset}")
            for raw_algorithm in RAW_ALGORITHMS
        }
        algorithm_positions: defaultdict[str, int] = defaultdict(int)
        with (results_root / f"{current_dataset}_table_counts.csv").open(newline="", encoding="utf-8") as handle:
            for count_row in csv.DictReader(handle):
                raw_index = _raw_index(count_row)
                raw_algorithm = (count_row.get("algorithm") or "").strip()
                n_rank = count_row.get("n_rank")
                # The known dropped indexes have no corresponding expression
                # line. Rows missing rank data do have a line, so advance the
                # per-algorithm position before omitting them from the study.
                if raw_algorithm == "FEAT" or raw_index in DROP_INDEXES[current_dataset]:
                    continue
                algorithm_positions[raw_algorithm] += 1
                algorithm_row = algorithm_positions[raw_algorithm]
                if not should_keep_row(current_dataset, raw_index, raw_algorithm, n_rank):
                    continue
                public_algorithm = ALGORITHM_RENAMES.get(raw_algorithm, raw_algorithm)
                if algorithm not in (None, raw_algorithm, public_algorithm):
                    continue
                try:
                    source_by_kind = {
                        "original": originals[raw_algorithm][algorithm_row - 1],
                        "sympy": sympy_by_algorithm[raw_algorithm][algorithm_row - 1],
                    }
                except (KeyError, IndexError) as exc:
                    msg = (
                        f"External archive rows are misaligned for {current_dataset}/{raw_algorithm} "
                        f"at retained algorithm row {algorithm_row}."
                    )
                    raise ArchiveLayoutError(msg) from exc
                for current_kind in ("original", "sympy"):
                    if input_kind is not None and current_kind != input_kind:
                        continue
                    rows.append(
                        CorpusRow(
                            row_id=(f"{current_dataset}/{raw_index}/{public_algorithm}/{algorithm_row}/{current_kind}"),
                            dataset=current_dataset,
                            raw_index=raw_index,
                            algorithm_raw=raw_algorithm,
                            algorithm=public_algorithm,
                            algorithm_row=algorithm_row,
                            input_kind=current_kind,
                            source=source_by_kind[current_kind],
                            source_n_rank=float(n_rank or "nan"),
                        )
                    )
                    if limit is not None and len(rows) >= limit:
                        return rows
    return rows


def external_archive_hash(root: Path | None = None) -> str:
    """Hash corpus rows plus the optional live-Haskell source/toolchain inputs."""
    archive = external_archive_root(root)
    files = {path for path in (archive / "results").rglob("*") if path.is_file()}
    files.update(path for path in (archive / "src").rglob("*.hs") if path.is_file())
    files.update(
        path
        for name in ("rewrite.cabal", "cabal.project", "stack.yaml", "stack.yaml.lock")
        if (path := archive / name).is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(archive).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
