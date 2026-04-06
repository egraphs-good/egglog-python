"""Normalize extracted param-eq-haskell result files into checked-in CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from egglog.exp.param_eq.paths import ARTIFACT_DIR, param_eq_data_dir

DATASETS = ("pagie", "kotanchek")
DROP_INDEXES: dict[str, set[int]] = {"pagie": {16, 162}, "kotanchek": {1}}
ALGORITHM_RENAMES = {"SRjl": "PySR", "GOMEA": "GP-GOMEA"}
RUNTIME_PATH = Path("runtimes")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=param_eq_data_dir(),
        help="Path to the extracted param-eq-haskell checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Directory to write normalized artifacts into",
    )
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    return args


def _read_text(source_dir: Path, relative_path: Path) -> str:
    path = source_dir / relative_path
    return path.read_text(encoding="utf-8", errors="replace")


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


def _load_expression_lines(source_dir: Path, relative_path: Path) -> list[str]:
    return [line.strip() for line in _read_text(source_dir, relative_path).splitlines() if line.strip()]


def _load_results_rows_by_algorithm(source_dir: Path, dataset: str) -> dict[str, list[dict[str, str]]]:
    rows = _read_csv_text(_read_text(source_dir, Path("results") / f"{dataset}_results"))
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)
    return dict(grouped)


def _normalize_haskell_rows(source_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        table_path = Path("results") / f"{dataset}_table_counts.csv"
        raw_rows = _read_csv_text(_read_text(source_dir, table_path))
        results_rows_by_algorithm = _load_results_rows_by_algorithm(source_dir, dataset)
        exprs_by_algorithm = {
            path_algorithm: _load_expression_lines(
                source_dir,
                Path("results") / "exprs" / f"{path_algorithm}_exprs_{dataset}",
            )
            for path_algorithm in ("Bingo", "EPLEX", "FEAT", "GOMEA", "Operon", "SBP", "SRjl")
        }
        algo_counts: defaultdict[str, int] = defaultdict(int)
        for raw_index, raw_row in enumerate(raw_rows):
            raw_algorithm = raw_row["algorithm"]
            algo_counts[raw_algorithm] += 1
            algo_row = algo_counts[raw_algorithm]
            exprs = exprs_by_algorithm[raw_algorithm]
            sympy_exprs = results_rows_by_algorithm[raw_algorithm]
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
                "sympy_expr": sympy_exprs[algo_row - 1]["expr_sympy"].strip(),
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


def _normalize_runtime_rows(source_dir: Path) -> list[dict[str, str]]:
    lines = _read_text(source_dir, RUNTIME_PATH).splitlines()
    benchmark_lines = [line for line in lines if line.startswith("benchmarking ")]
    time_lines = [line for line in lines if line.startswith("time")]
    rows: list[dict[str, str]] = []
    for benchmark_line, time_line in zip(benchmark_lines, time_lines, strict=True):
        benchmark_name = benchmark_line.removeprefix("benchmarking ").strip()
        node_count = int(benchmark_name.split("/")[-1])
        _, numeric, unit, *_ = time_line.split()
        runtime_ms = float(numeric) * _to_runtime_multiplier(unit)
        rows.append({
            "benchmark_name": benchmark_name,
            "node_count": str(node_count),
            "runtime_ms": f"{runtime_ms:.9f}",
        })
    return rows


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    haskell_rows = _normalize_haskell_rows(args.source_dir)
    runtime_rows = _normalize_runtime_rows(args.source_dir)

    _write_csv(args.output_dir / "haskell_paper_rows.csv", haskell_rows)
    _write_csv(args.output_dir / "pagie_runtime_scatter.csv", runtime_rows)


if __name__ == "__main__":
    main()
