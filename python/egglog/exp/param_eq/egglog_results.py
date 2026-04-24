"""Minimal Egglog result storage plus joined loaders for `param_eq`."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from egglog.exp.param_eq.domain import parse_expression
from egglog.exp.param_eq.original_results import (
    INPUT_KIND_ORDER,
    NA_REPR,
    RESULT_KEY_COLUMNS,
    canonical_expr,
    load_original_results,
    load_original_results_for_join,
    parsed_n_params,
    rank_gap,
)
from egglog.exp.param_eq.paths import artifact_dir
from egglog.exp.param_eq.pipeline import parse_expression_container, run_paper_pipeline, run_paper_pipeline_container
from egglog.exp.param_eq.resource_guard import (
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_SAMPLE_INTERVAL_SEC,
    cap_workers_for_memory,
    total_system_memory_mb,
    watch_process,
)

TIMEOUT_SEC = 180.0
SUCCESS_STATUS = "saturated"
DEFAULT_WORKERS = 10

EGGLOG_RESULTS_SCHEMA = pa.DataFrameSchema(
    {
        "dataset": pa.Column(str),
        "raw_index": pa.Column(int),
        "algorithm_raw": pa.Column(str),
        "algo_row": pa.Column(int),
        "input_kind": pa.Column(str, checks=pa.Check.isin(INPUT_KIND_ORDER)),
        "variant": pa.Column(str, checks=pa.Check.isin(["baseline", "container"])),
        "status": pa.Column(str),
        "runtime_ms": pa.Column(float, nullable=True),
        "before_nodes": pa.Column(float, nullable=True),
        "before_params": pa.Column(float, nullable=True),
        "after_nodes": pa.Column(float, nullable=True),
        "after_params": pa.Column(float, nullable=True),
        "egraph_total_size": pa.Column(float, nullable=True),
        "passes": pa.Column(float, nullable=True),
        "extracted_cost": pa.Column(float, nullable=True),
        "peak_rss_mb": pa.Column(float, nullable=True),
        "rendered": pa.Column(str, nullable=True),
    },
    strict=True,
    ordered=True,
    coerce=True,
)


def default_egglog_results_path() -> Path:
    return artifact_dir() / "egglog_results.csv"


def load_egglog_results_raw(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or default_egglog_results_path(), na_values=[NA_REPR], keep_default_na=True)
    return EGGLOG_RESULTS_SCHEMA.validate(frame)


def write_egglog_results(frame: pd.DataFrame, path: Path | None = None) -> None:
    validated = EGGLOG_RESULTS_SCHEMA.validate(frame)
    output = path or default_egglog_results_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False, na_rep=NA_REPR)


def load_egglog_results(path: Path | None = None, *, variant: str | None = None, source_root: Path | None = None) -> pd.DataFrame:
    source = load_original_results_for_join(source_root)
    egglog = load_egglog_results_raw(path)
    if variant is not None:
        egglog = egglog[egglog["variant"] == variant].copy()
    joined = source.merge(egglog, on=RESULT_KEY_COLUMNS, how="inner", validate="one_to_one")
    joined["implementation"] = "egglog"
    joined["baseline_source"] = ["egglog_" + str(v) for v in joined["variant"]]
    joined["simpl_parsed_expr"] = joined["rendered"].map(canonical_expr)
    joined["simpl_parsed_n_params"] = joined["simpl_parsed_expr"].map(parsed_n_params)
    joined["before_rank_difference"] = [
        rank_gap(before_params, n_rank)
        for before_params, n_rank in zip(joined["before_params"], joined["n_rank"], strict=True)
    ]
    joined["after_rank_difference"] = [
        rank_gap(after_params, n_rank)
        for after_params, n_rank in zip(joined["after_params"], joined["n_rank"], strict=True)
    ]
    joined["after_parsed_rank_difference"] = [
        rank_gap(parsed_params, n_rank)
        for parsed_params, n_rank in zip(joined["simpl_parsed_n_params"], joined["n_rank"], strict=True)
    ]
    return joined.sort_values([*RESULT_KEY_COLUMNS, "variant"]).reset_index(drop=True)


def _filter_source_rows(
    frame: pd.DataFrame,
    *,
    dataset: str | None,
    algorithm: str | None,
    limit_per_dataset: int | None,
    offset: int,
    limit: int | None,
) -> pd.DataFrame:
    filtered = frame
    if dataset is not None:
        filtered = filtered[filtered["dataset"] == dataset]
    if algorithm is not None:
        filtered = filtered[(filtered["algorithm"] == algorithm) | (filtered["algorithm_raw"] == algorithm)]
    if limit_per_dataset is not None:
        filtered = filtered.groupby("dataset", sort=False, group_keys=False).head(limit_per_dataset)
    if offset:
        filtered = filtered.iloc[offset:]
    if limit is not None:
        filtered = filtered.iloc[:limit]
    return filtered


def _load_rows(
    *,
    dataset: str | None = None,
    algorithm: str | None = None,
    limit_per_dataset: int | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, object]]:
    frame = load_original_results().loc[:, [*RESULT_KEY_COLUMNS, "algorithm", "orig_parsed_expr"]]
    filtered = _filter_source_rows(
        frame,
        dataset=dataset,
        algorithm=algorithm,
        limit_per_dataset=limit_per_dataset,
        offset=offset,
        limit=limit,
    )
    return filtered.sort_values(RESULT_KEY_COLUMNS).to_dict(orient="records")


def _timeout_status() -> dict[str, object]:
    return {
        "status": "timeout",
        "runtime_ms": None,
        "before_nodes": None,
        "before_params": None,
        "after_nodes": None,
        "after_params": None,
        "egraph_total_size": None,
        "passes": None,
        "extracted_cost": None,
        "peak_rss_mb": None,
        "rendered": None,
    }


def _failed_status() -> dict[str, object]:
    return {
        "status": "failed",
        "runtime_ms": None,
        "before_nodes": None,
        "before_params": None,
        "after_nodes": None,
        "after_params": None,
        "egraph_total_size": None,
        "passes": None,
        "extracted_cost": None,
        "peak_rss_mb": None,
        "rendered": None,
    }


def _memory_limit_status() -> dict[str, object]:
    return {
        **_failed_status(),
        "status": "memory_limit",
    }


def _evaluate_expression(source: str, variant: str) -> dict[str, object]:
    start = time.perf_counter()
    try:
        if variant == "baseline":
            report = run_paper_pipeline(parse_expression(source))
        else:
            report = run_paper_pipeline_container(parse_expression_container(source))
    except Exception:
        return _failed_status()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "status": SUCCESS_STATUS,
        "runtime_ms": elapsed_ms,
        "before_nodes": float(report.before_nodes),
        "before_params": float(report.before_params),
        "after_nodes": float(report.extracted_nodes),
        "after_params": float(report.extracted_params),
        "egraph_total_size": float(report.total_size),
        "passes": float(report.passes),
        "extracted_cost": float(report.extracted_nodes),
        "peak_rss_mb": None,
        "rendered": report.extracted,
    }


def _worker_entry(connection, source: str, variant: str) -> None:
    try:
        connection.send(_evaluate_expression(source, variant))
    finally:
        connection.close()


def _run_worker_process(
    source: str,
    variant: str,
    *,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> dict[str, object]:
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(child_connection, source, variant))
    process.start()
    watch = watch_process(
        process,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
    )
    child_connection.close()
    if watch.status == "timeout":
        result = _timeout_status()
    elif watch.status == "memory_limit":
        result = _memory_limit_status()
    elif parent_connection.poll():
        try:
            result = parent_connection.recv()
        except EOFError:
            result = _failed_status()
    else:
        result = _failed_status()
    parent_connection.close()
    result["peak_rss_mb"] = watch.peak_rss_mb
    return result


def _run_source_row(
    row: dict[str, object],
    variant: str,
    *,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> dict[str, object]:
    result = _run_worker_process(
        str(row["orig_parsed_expr"]),
        variant,
        timeout_sec=TIMEOUT_SEC,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
    )
    return {
        "dataset": row["dataset"],
        "raw_index": row["raw_index"],
        "algorithm_raw": row["algorithm_raw"],
        "algo_row": row["algo_row"],
        "input_kind": row["input_kind"],
        "variant": variant,
        **result,
    }


def _progress_description(
    *,
    completed: int,
    total: int,
    kill_count: int,
    max_peak_rss_mb: float | None,
    counts: Counter[str],
) -> str:
    peak = "na" if max_peak_rss_mb is None else f"{max_peak_rss_mb:.1f}MB"
    status_summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    return f"egglog rows {completed}/{total} killed={kill_count} max_rss={peak} {status_summary}".strip()


def run_egglog_results_frame(
    *,
    variant: str = "baseline",
    workers: int = DEFAULT_WORKERS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
    dataset: str | None = None,
    algorithm: str | None = None,
    limit_per_dataset: int | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    rows = _load_rows(
        dataset=dataset,
        algorithm=algorithm,
        limit_per_dataset=limit_per_dataset,
        offset=offset,
        limit=limit,
    )
    workers = cap_workers_for_memory(workers, memory_limit_mb=memory_limit_mb)
    results: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    killed_count = 0
    max_peak_rss_mb = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            _progress_description(
                completed=0,
                total=len(rows),
                kill_count=0,
                max_peak_rss_mb=None,
                counts=Counter(),
            ),
            total=len(rows),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_source_row,
                    row,
                    variant,
                    memory_limit_mb=memory_limit_mb,
                    sample_interval_sec=sample_interval_sec,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status_counts[str(result["status"])] += 1
                killed_count += int(result["status"] == "memory_limit")
                peak_rss_mb = result["peak_rss_mb"]
                if peak_rss_mb is not None:
                    max_peak_rss_mb = peak_rss_mb if max_peak_rss_mb is None else max(max_peak_rss_mb, peak_rss_mb)
                progress.advance(task)
                progress.update(
                    task,
                    description=_progress_description(
                        completed=len(results),
                        total=len(rows),
                        kill_count=killed_count,
                        max_peak_rss_mb=max_peak_rss_mb,
                        counts=status_counts,
                    ),
                )
    Console().print(f"memory_limit_mb={memory_limit_mb} workers={workers} total_system_memory_mb={total_system_memory_mb():.1f}")
    return EGGLOG_RESULTS_SCHEMA.validate(pd.DataFrame.from_records(results))


def upsert_egglog_results(frame: pd.DataFrame, path: Path | None = None) -> None:
    output = path or default_egglog_results_path()
    if output.exists():
        existing = load_egglog_results_raw(output)
        existing = existing.merge(
            frame.loc[:, [*RESULT_KEY_COLUMNS, "variant"]],
            on=[*RESULT_KEY_COLUMNS, "variant"],
            how="left",
            indicator=True,
        )
        existing = existing[existing["_merge"] == "left_only"].drop(columns="_merge")
        frame = pd.concat([existing, frame], ignore_index=True)
    write_egglog_results(frame.sort_values([*RESULT_KEY_COLUMNS, "variant"]).reset_index(drop=True), output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_egglog_results_path())
    parser.add_argument("--variant", choices=("baseline", "container"), default="baseline")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--memory-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--sample-interval-sec", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--algorithm", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    frame = run_egglog_results_frame(
        variant=args.variant,
        workers=args.workers,
        memory_limit_mb=args.memory_limit_mb,
        sample_interval_sec=args.sample_interval_sec,
        dataset=args.dataset,
        algorithm=args.algorithm,
        limit_per_dataset=args.limit_per_dataset,
        offset=args.offset,
        limit=args.limit,
    )
    write_egglog_results(frame.sort_values([*RESULT_KEY_COLUMNS, "variant"]).reset_index(drop=True), args.output)
    Console().print(f"[green]wrote[/green] {args.output}")


if __name__ == "__main__":
    main()
