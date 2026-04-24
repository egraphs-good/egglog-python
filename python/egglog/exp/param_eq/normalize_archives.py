"""Vendor the minimal raw `param_eq` archive subset into `artifacts/original`."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from egglog.exp.param_eq.original_results import DATASETS, RAW_ALGORITHMS
from egglog.exp.param_eq.paths import original_artifact_dir, param_eq_data_dir

RAW_RELATIVE_PATHS = [
    *(Path("results") / f"{dataset}_table_counts.csv" for dataset in DATASETS),
    *(Path("results") / f"{dataset}_results" for dataset in DATASETS),
    *(Path("results") / "exprs" / f"{algorithm}_exprs_{dataset}" for dataset in DATASETS for algorithm in RAW_ALGORITHMS),
    *(
        Path("results") / "exprs_simpl" / f"{algorithm}_exprs_{dataset}"
        for dataset in DATASETS
        for algorithm in RAW_ALGORITHMS
    ),
    Path("runtimes"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=param_eq_data_dir(),
        help="Path to the extracted param-eq-haskell checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=original_artifact_dir(),
        help="Path to write the vendored raw archive subset.",
    )
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def vendor_raw_sources(source_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in RAW_RELATIVE_PATHS:
        source_path = source_dir / relative_path
        target_path = output_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def main() -> None:
    args = _parse_args()
    vendor_raw_sources(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
