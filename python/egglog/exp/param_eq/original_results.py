"""Vendored raw `param_eq` archive loaders plus the cleaned retained-paper source DataFrame."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from egglog.exp.param_eq.domain import count_params, parse_expression, render_num
from egglog.exp.param_eq.paths import original_artifact_dir

NA_REPR = "na"
DATASETS = ("pagie", "kotanchek")
RAW_ALGORITHMS = ("Bingo", "EPLEX", "FEAT", "GOMEA", "Operon", "SBP", "SRjl")
INPUT_KIND_ORDER = ("original", "sympy")
RESULT_KEY_COLUMNS = ["dataset", "raw_index", "algorithm_raw", "algo_row", "input_kind"]
SOURCE_METADATA_COLUMNS = [*RESULT_KEY_COLUMNS, "algorithm", "n_params", "n_rank"]
DROP_INDEXES: dict[str, set[int]] = {"pagie": {16, 162}, "kotanchek": {1}}
ALGORITHM_RENAMES = {"SRjl": "PySR", "GOMEA": "GP-GOMEA"}

TABLE_COUNTS_SCHEMA = pa.DataFrameSchema(
    {
        "raw_index": pa.Column(int),
        "orig_nodes": pa.Column(float),
        "orig_params": pa.Column(float),
        "simpl_nodes": pa.Column(float),
        "simpl_params": pa.Column(float),
        "orig_nodes_sympy": pa.Column(float),
        "orig_params_sympy": pa.Column(float),
        "simpl_nodes_sympy": pa.Column(float),
        "simpl_params_sympy": pa.Column(float),
        "algorithm": pa.Column(str),
        "n_params": pa.Column(float, nullable=True),
        "n_rank": pa.Column(float, nullable=True),
    },
    strict=True,
    ordered=True,
    coerce=True,
)

DATASET_RESULTS_SCHEMA = pa.DataFrameSchema(
    {
        "algorithm": pa.Column(str),
        "expr": pa.Column(str),
        "expr_sympy": pa.Column(str),
    },
    strict=True,
    ordered=True,
    coerce=True,
)

EXPR_LINES_SCHEMA = pa.DataFrameSchema(
    {
        "expr": pa.Column(str),
    },
    strict=True,
    ordered=True,
    coerce=True,
)

ORIGINAL_RESULTS_SCHEMA = pa.DataFrameSchema(
    {
        "dataset": pa.Column(str, checks=pa.Check.isin(DATASETS)),
        "raw_index": pa.Column(int),
        "algorithm_raw": pa.Column(str, checks=pa.Check.isin(RAW_ALGORITHMS)),
        "algorithm": pa.Column(str),
        "algo_row": pa.Column(int),
        "input_kind": pa.Column(str, checks=pa.Check.isin(INPUT_KIND_ORDER)),
        "n_params": pa.Column(float, nullable=True),
        "n_rank": pa.Column(float, nullable=True),
        "before_nodes": pa.Column(float),
        "before_params": pa.Column(float),
        "after_nodes": pa.Column(float),
        "after_params": pa.Column(float),
        "orig_expr": pa.Column(str),
        "simpl_expr": pa.Column(str, nullable=True),
        "orig_parsed_expr": pa.Column(str),
        "simpl_parsed_expr": pa.Column(str, nullable=True),
        "orig_parsed_n_params": pa.Column(float),
        "simpl_parsed_n_params": pa.Column(float, nullable=True),
        "before_rank_difference": pa.Column(float, nullable=True),
        "after_rank_difference": pa.Column(float, nullable=True),
        "before_parsed_rank_difference": pa.Column(float, nullable=True),
        "after_parsed_rank_difference": pa.Column(float, nullable=True),
    },
    strict=True,
    ordered=True,
    coerce=True,
)

ARCHIVED_RUNTIMES_SCHEMA = pa.DataFrameSchema(
    {
        "benchmark_name": pa.Column(str),
        "node_count": pa.Column(int),
        "runtime_ms": pa.Column(float),
    },
    strict=True,
    ordered=True,
    coerce=True,
)


def _source_root(root: Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else original_artifact_dir()


def _results_root(root: Path | None = None) -> Path:
    return _source_root(root) / "results"


def _table_counts_path(dataset: str, root: Path | None = None) -> Path:
    return _results_root(root) / f"{dataset}_table_counts.csv"


def _dataset_results_path(dataset: str, root: Path | None = None) -> Path:
    return _results_root(root) / f"{dataset}_results"


def _expr_lines_path(dataset: str, algorithm: str, *, simplified: bool, root: Path | None = None) -> Path:
    directory = "exprs_simpl" if simplified else "exprs"
    return _results_root(root) / directory / f"{algorithm}_exprs_{dataset}"


def _runtimes_path(root: Path | None = None) -> Path:
    return _source_root(root) / "runtimes"


def _clean_algorithm(name: str) -> str:
    return ALGORITHM_RENAMES.get(name, name)


def _should_keep_row(dataset: str, raw_index: int, raw_algorithm: str) -> bool:
    return raw_algorithm != "FEAT" and raw_index not in DROP_INDEXES[dataset]


def canonical_expr(expr: object) -> str | None:
    if pd.isna(expr):
        return None
    return render_num(parse_expression(str(expr)))


def parsed_n_params(expr: object) -> float | None:
    if pd.isna(expr):
        return None
    return float(count_params(parse_expression(str(expr))))


def rank_gap(param_count: object, n_rank: object) -> float | None:
    if pd.isna(param_count) or pd.isna(n_rank):
        return None
    return float(param_count) - float(n_rank)


def load_table_counts(dataset: str, root: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(_table_counts_path(dataset, root))
    first_column = frame.columns[0]
    if str(first_column).startswith("Unnamed:"):
        frame = frame.rename(columns={first_column: "raw_index"})
    return TABLE_COUNTS_SCHEMA.validate(frame)


def load_dataset_results(dataset: str, root: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(_dataset_results_path(dataset, root), dtype=str).fillna("")
    return DATASET_RESULTS_SCHEMA.validate(frame)


def load_expr_lines(
    dataset: str, algorithm: str, *, simplified: bool = False, root: Path | None = None
) -> pd.DataFrame:
    with open(_expr_lines_path(dataset, algorithm, simplified=simplified, root=root), encoding="utf-8") as handle:
        frame = pd.DataFrame({"expr": [line.strip() for line in handle if line.strip()]})
    return EXPR_LINES_SCHEMA.validate(frame)


def _parse_archived_runtimes(root: Path | None = None) -> pd.DataFrame:
    lines = _runtimes_path(root).read_text(encoding="utf-8", errors="replace").splitlines()
    benchmark_lines = [line for line in lines if line.startswith("benchmarking ")]
    time_lines = [line for line in lines if line.startswith("time")]
    rows: list[dict[str, object]] = []
    for benchmark_line, time_line in zip(benchmark_lines, time_lines, strict=True):
        benchmark_name = benchmark_line.removeprefix("benchmarking ").strip()
        node_count = int(benchmark_name.split("/")[-1])
        _, numeric, unit, *_ = time_line.split()
        runtime_ms = float(numeric) * {"ms": 1.0, "μs": 0.001, "ns": 1e-6, "s": 1000.0}[unit]
        rows.append({"benchmark_name": benchmark_name, "node_count": node_count, "runtime_ms": runtime_ms})
    return ARCHIVED_RUNTIMES_SCHEMA.validate(pd.DataFrame.from_records(rows))


@lru_cache(maxsize=1)
def _load_original_results_cached(root_str: str) -> pd.DataFrame:
    root = Path(root_str)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        counts = load_table_counts(dataset, root)
        results_rows = load_dataset_results(dataset, root)
        results_by_algorithm = {
            algorithm: group.reset_index(drop=True)
            for algorithm, group in results_rows.groupby("algorithm", sort=False)
        }
        exprs_by_algorithm = {algorithm: load_expr_lines(dataset, algorithm, root=root) for algorithm in RAW_ALGORITHMS}
        simplified_exprs_by_algorithm = {
            algorithm: load_expr_lines(dataset, algorithm, simplified=True, root=root) for algorithm in RAW_ALGORITHMS
        }
        algo_counts: defaultdict[str, int] = defaultdict(int)
        for raw_row in counts.itertuples(index=False):
            raw_index = int(raw_row.raw_index)
            raw_algorithm = str(raw_row.algorithm)
            if not _should_keep_row(dataset, raw_index, raw_algorithm):
                continue
            algo_counts[raw_algorithm] += 1
            algo_row = algo_counts[raw_algorithm]
            original_expr = str(exprs_by_algorithm[raw_algorithm].iloc[algo_row - 1]["expr"]).strip()
            original_simpl_expr = str(simplified_exprs_by_algorithm[raw_algorithm].iloc[algo_row - 1]["expr"]).strip()
            sympy_expr = str(results_by_algorithm[raw_algorithm].iloc[algo_row - 1]["expr_sympy"]).strip()
            rows.extend([
                {
                    "dataset": dataset,
                    "raw_index": raw_index,
                    "algorithm_raw": raw_algorithm,
                    "algorithm": _clean_algorithm(raw_algorithm),
                    "algo_row": algo_row,
                    "input_kind": "original",
                    "n_params": raw_row.n_params,
                    "n_rank": raw_row.n_rank,
                    "before_nodes": raw_row.orig_nodes,
                    "before_params": raw_row.orig_params,
                    "after_nodes": raw_row.simpl_nodes,
                    "after_params": raw_row.simpl_params,
                    "orig_expr": original_expr,
                    "simpl_expr": original_simpl_expr,
                },
                {
                    "dataset": dataset,
                    "raw_index": raw_index,
                    "algorithm_raw": raw_algorithm,
                    "algorithm": _clean_algorithm(raw_algorithm),
                    "algo_row": algo_row,
                    "input_kind": "sympy",
                    "n_params": raw_row.n_params,
                    "n_rank": raw_row.n_rank,
                    "before_nodes": raw_row.orig_nodes_sympy,
                    "before_params": raw_row.orig_params_sympy,
                    "after_nodes": raw_row.simpl_nodes_sympy,
                    "after_params": raw_row.simpl_params_sympy,
                    "orig_expr": sympy_expr,
                    "simpl_expr": None,
                },
            ])
    frame = pd.DataFrame.from_records(rows)
    frame["orig_parsed_expr"] = frame["orig_expr"].map(canonical_expr)
    frame["simpl_parsed_expr"] = frame["simpl_expr"].map(canonical_expr)
    frame["orig_parsed_n_params"] = frame["orig_parsed_expr"].map(parsed_n_params)
    frame["simpl_parsed_n_params"] = frame["simpl_parsed_expr"].map(parsed_n_params)
    frame["before_rank_difference"] = [
        rank_gap(before_params, n_rank)
        for before_params, n_rank in zip(frame["before_params"], frame["n_rank"], strict=True)
    ]
    frame["after_rank_difference"] = [
        rank_gap(after_params, n_rank)
        for after_params, n_rank in zip(frame["after_params"], frame["n_rank"], strict=True)
    ]
    frame["before_parsed_rank_difference"] = [
        rank_gap(parsed_params, n_rank)
        for parsed_params, n_rank in zip(frame["orig_parsed_n_params"], frame["n_rank"], strict=True)
    ]
    frame["after_parsed_rank_difference"] = [
        rank_gap(parsed_params, n_rank)
        for parsed_params, n_rank in zip(frame["simpl_parsed_n_params"], frame["n_rank"], strict=True)
    ]
    validated = ORIGINAL_RESULTS_SCHEMA.validate(frame)
    return validated.sort_values(RESULT_KEY_COLUMNS).reset_index(drop=True)


def load_original_results(root: Path | None = None) -> pd.DataFrame:
    return _load_original_results_cached(str(_source_root(root))).copy(deep=True)


@lru_cache(maxsize=1)
def _load_archived_runtimes_cached(root_str: str) -> pd.DataFrame:
    return _parse_archived_runtimes(Path(root_str))


def load_archived_runtimes(root: Path | None = None) -> pd.DataFrame:
    return _load_archived_runtimes_cached(str(_source_root(root))).copy(deep=True)


def load_original_results_for_join(root: Path | None = None) -> pd.DataFrame:
    source = load_original_results(root)
    rename_map = {column: f"source_{column}" for column in source.columns if column not in SOURCE_METADATA_COLUMNS}
    return source.rename(columns=rename_map)
