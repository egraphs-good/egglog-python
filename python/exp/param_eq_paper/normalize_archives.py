from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASETS = ("pagie", "kotanchek")
DROP_INDEXES: dict[str, set[int]] = {"pagie": {16, 162}, "kotanchek": {1}}
ALGORITHM_RENAMES = {"SRjl": "PySR", "GOMEA": "GP-GOMEA"}
RUNTIME_MEMBER = "param-eq-haskell/runtimes"


@dataclass(frozen=True)
class ArchiveBundle:
    param_eq_haskell: Path
    pandoc_symreg: Path


def _discover_tarball(filename: str) -> Path:
    candidates = [
        Path.cwd() / filename,
        Path.home() / "Downloads" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = f"Could not find {filename!r}; pass it explicitly with a CLI flag."
    raise FileNotFoundError(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param-eq-haskell-tar",
        type=Path,
        default=None,
        help="Path to param-eq-haskell.tar.gz",
    )
    parser.add_argument(
        "--pandoc-symreg-tar",
        type=Path,
        default=None,
        help="Path to pandoc-symreg tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory to write normalized artifacts into",
    )
    return parser.parse_args()


def _open_bundle(args: argparse.Namespace) -> ArchiveBundle:
    param_eq_haskell = args.param_eq_haskell_tar or _discover_tarball("param-eq-haskell.tar.gz")
    pandoc_symreg = args.pandoc_symreg_tar or _discover_tarball("pandoc-symreg (1).tar.gz")
    return ArchiveBundle(param_eq_haskell=param_eq_haskell, pandoc_symreg=pandoc_symreg)


def _read_text(tf: tarfile.TarFile, member: str) -> str:
    handle = tf.extractfile(member)
    if handle is None:
        msg = f"Archive member not found: {member}"
        raise FileNotFoundError(msg)
    return handle.read().decode("utf-8", errors="replace")


def _read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def _clean_algorithm(name: str) -> str:
    return ALGORITHM_RENAMES.get(name, name)


def _paper_row_status(dataset: str, raw_index: int, raw_algorithm: str, n_rank: str) -> tuple[bool, str]:
    if raw_algorithm == "FEAT":
        return False, "drop_feat"
    if n_rank == "" or n_rank.lower() == "nan":
        return False, "missing_rank"
    if raw_index in DROP_INDEXES[dataset]:
        return False, "paper_manual_drop"
    return True, "kept"


def _load_expression_lines(tf: tarfile.TarFile, member: str) -> list[str]:
    return [line.strip() for line in _read_text(tf, member).splitlines() if line.strip()]


def _normalize_haskell_rows(tf: tarfile.TarFile) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        table_member = f"param-eq-haskell/results/{dataset}_table_counts.csv"
        raw_rows = _read_csv_text(_read_text(tf, table_member))
        exprs_by_algorithm = {
            path_algorithm: _load_expression_lines(tf, f"param-eq-haskell/results/exprs/{path_algorithm}_exprs_{dataset}")
            for path_algorithm in ("Bingo", "EPLEX", "FEAT", "GOMEA", "Operon", "SBP", "SRjl")
        }
        sympy_by_algorithm = {
            path_algorithm: _load_expression_lines(
                tf, f"param-eq-haskell/results/exprs_simpl/{path_algorithm}_exprs_{dataset}"
            )
            for path_algorithm in ("Bingo", "EPLEX", "FEAT", "GOMEA", "Operon", "SBP", "SRjl")
        }
        algo_counts: defaultdict[str, int] = defaultdict(int)
        for raw_index, raw_row in enumerate(raw_rows):
            raw_algorithm = raw_row["algorithm"]
            algo_counts[raw_algorithm] += 1
            algo_row = algo_counts[raw_algorithm]
            exprs = exprs_by_algorithm[raw_algorithm]
            sympy_exprs = sympy_by_algorithm[raw_algorithm]
            is_paper_row, drop_reason = _paper_row_status(dataset, raw_index, raw_algorithm, raw_row["n_rank"])
            row = {
                "dataset": dataset,
                "raw_index": str(raw_index),
                "algorithm_raw": raw_algorithm,
                "algorithm": _clean_algorithm(raw_algorithm),
                "algo_row": str(algo_row),
                "is_paper_row": "1" if is_paper_row else "0",
                "drop_reason": drop_reason,
                "original_expr": exprs[algo_row - 1],
                "sympy_expr": sympy_exprs[algo_row - 1],
            }
            for key, value in raw_row.items():
                if key == "":
                    row["raw_csv_index"] = value
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _to_runtime_multiplier(unit: str) -> float:
    return {"ms": 1.0, "μs": 0.001, "ns": 1e-6, "s": 1000.0}[unit]


def _normalize_runtime_rows(tf: tarfile.TarFile) -> list[dict[str, str]]:
    lines = _read_text(tf, RUNTIME_MEMBER).splitlines()
    benchmark_lines = [line for line in lines if line.startswith("benchmarking ")]
    time_lines = [line for line in lines if line.startswith("time")]
    rows: list[dict[str, str]] = []
    for benchmark_line, time_line in zip(benchmark_lines, time_lines, strict=True):
        benchmark_name = benchmark_line.removeprefix("benchmarking ").strip()
        node_count = int(benchmark_name.split("/")[-1])
        _, numeric, unit, *_ = time_line.split()
        runtime_ms = float(numeric) * _to_runtime_multiplier(unit)
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "node_count": str(node_count),
                "runtime_ms": f"{runtime_ms:.9f}",
            }
        )
    return rows


def _normalize_manifest(bundle: ArchiveBundle, haskell_rows: Iterable[dict[str, str]], runtime_rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    haskell_rows = list(haskell_rows)
    runtime_rows = list(runtime_rows)
    return {
        "param_eq_haskell_tar": bundle.param_eq_haskell.name,
        "pandoc_symreg_tar": bundle.pandoc_symreg.name,
        "haskell_row_count": len(haskell_rows),
        "paper_row_count": sum(row["is_paper_row"] == "1" for row in haskell_rows),
        "pagie_paper_rows": sum(row["dataset"] == "pagie" and row["is_paper_row"] == "1" for row in haskell_rows),
        "kotanchek_paper_rows": sum(row["dataset"] == "kotanchek" and row["is_paper_row"] == "1" for row in haskell_rows),
        "runtime_row_count": len(runtime_rows),
        "matcher_height_limit": 8,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        msg = f"No rows available for {path.name}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    bundle = _open_bundle(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle.param_eq_haskell, "r:gz") as param_eq_haskell_tar:
        haskell_rows = _normalize_haskell_rows(param_eq_haskell_tar)
        runtime_rows = _normalize_runtime_rows(param_eq_haskell_tar)

    with tarfile.open(bundle.pandoc_symreg, "r:gz") as pandoc_symreg_tar:
        matcher_member = "pandoc-symreg/src/Data/Equality/Matching.hs"
        matcher_source = _read_text(pandoc_symreg_tar, matcher_member)
        matcher_height_limit = 8 if "getHeight 8" in matcher_source else None

    manifest = _normalize_manifest(bundle, haskell_rows, runtime_rows)
    if matcher_height_limit is not None:
        manifest["matcher_height_limit"] = matcher_height_limit

    _write_csv(args.output_dir / "haskell_paper_rows.csv", haskell_rows)
    _write_csv(args.output_dir / "pagie_runtime_scatter.csv", runtime_rows)
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
