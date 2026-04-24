"""Write the retained Egglog baseline rows that miss the archived rank target."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from egglog.exp.param_eq.egglog_results import load_egglog_results
from egglog.exp.param_eq.original_results import RESULT_KEY_COLUMNS
from egglog.exp.param_eq.paths import artifact_dir


def default_rank_misses_path() -> Path:
    return artifact_dir() / "egglog_rank_misses.csv"


def _sort_by_smallest_expression(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["_expr_len"] = result["source_orig_parsed_expr"].map(len)
    result = result.sort_values(
        [
            "source_orig_parsed_n_params",
            "source_before_nodes",
            "_expr_len",
            "dataset",
            "raw_index",
            "algo_row",
            "input_kind",
        ]
    )
    return result.drop(columns="_expr_len")


def egglog_rank_misses_frame(*, variant: str = "baseline") -> pd.DataFrame:
    """Return rows where Egglog's raw final parameter count remains above `n_rank`."""

    egglog = load_egglog_results(variant=variant).copy()
    egglog["rank_gap"] = egglog["after_rank_difference"]
    egglog["parsed_rank_gap"] = egglog["after_parsed_rank_difference"]
    misses = egglog[
        (egglog["status"] == "saturated")
        & egglog["n_rank"].notna()
        & egglog["rank_gap"].notna()
        & (egglog["rank_gap"] > 0)
    ].copy()
    columns = [
        *RESULT_KEY_COLUMNS,
        "algorithm",
        "source_orig_parsed_expr",
        "source_orig_parsed_n_params",
        "source_before_nodes",
        "n_rank",
        "before_params",
        "after_params",
        "rank_gap",
        "simpl_parsed_n_params",
        "parsed_rank_gap",
        "runtime_ms",
        "peak_rss_mb",
        "passes",
        "egraph_total_size",
        "extracted_cost",
        "rendered",
    ]
    return _sort_by_smallest_expression(misses.loc[:, columns])


def write_egglog_rank_misses(path: Path, *, variant: str = "baseline") -> pd.DataFrame:
    misses = egglog_rank_misses_frame(variant=variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    misses.to_csv(path, index=False, na_rep="na")
    return misses


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_rank_misses_path())
    parser.add_argument("--variant", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    misses = write_egglog_rank_misses(args.output, variant=args.variant)
    print(f"{args.output} ({len(misses)} rows)")


if __name__ == "__main__":
    main()
