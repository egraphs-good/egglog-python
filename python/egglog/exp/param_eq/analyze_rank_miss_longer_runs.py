"""Probe whether longer Egglog runs reduce retained rank-miss expressions."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from egglog.exp.param_eq.analyze_baseline_results import default_rank_misses_path
from egglog.exp.param_eq.original_results import NA_REPR, RESULT_KEY_COLUMNS
from egglog.exp.param_eq.paths import artifact_dir
from egglog.exp.param_eq.resource_guard import (
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_SAMPLE_INTERVAL_SEC,
    cap_workers_for_memory,
    watch_process,
)

ProbeMode = Literal["long_backoff", "no_backoff"]

DEFAULT_MAX_PASSES = 8
DEFAULT_INNER_LIMIT = 180
DEFAULT_MATCH_LIMIT = 10000
DEFAULT_BAN_LENGTH = 30
DEFAULT_WORKERS = 5
DEFAULT_LONG_BACKOFF_TIMEOUT_SEC = 60.0
DEFAULT_NO_BACKOFF_TIMEOUT_SEC = 30.0

OUTPUT_COLUMNS = [
    *RESULT_KEY_COLUMNS,
    "algorithm",
    "source_orig_parsed_expr",
    "source_orig_parsed_n_params",
    "source_before_nodes",
    "n_rank",
    "baseline_after_params",
    "baseline_rank_gap",
    "baseline_rendered",
    "probe_mode",
    "probe_max_passes",
    "probe_inner_limit",
    "probe_match_limit",
    "probe_ban_length",
    "probe_scheduler",
    "probe_status",
    "probe_runtime_ms",
    "probe_peak_rss_mb",
    "probe_passes",
    "probe_total_size",
    "probe_after_params",
    "probe_rank_gap",
    "probe_after_nodes",
    "probe_rendered",
    "probe_error",
    "param_delta",
    "rank_gap_delta",
    "improved",
    "reached_rank",
]


@dataclass(frozen=True)
class ProbeConfig:
    mode: ProbeMode
    max_passes: int
    inner_limit: int
    match_limit: int
    ban_length: int
    timeout_sec: float
    memory_limit_mb: int
    sample_interval_sec: float

    @property
    def scheduler_kind(self) -> str:
        return "fresh_rematch_backoff" if self.mode == "long_backoff" else "none"


def default_longer_runs_path() -> Path:
    return artifact_dir() / "egglog_rank_miss_longer_runs.csv"


def _notna(value: object) -> bool:
    return not pd.isna(value)


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _base_output_row(row: dict[str, object], config: ProbeConfig) -> dict[str, object]:
    return {
        **{key: row[key] for key in RESULT_KEY_COLUMNS},
        "algorithm": row["algorithm"],
        "source_orig_parsed_expr": row["source_orig_parsed_expr"],
        "source_orig_parsed_n_params": _optional_float(row["source_orig_parsed_n_params"]),
        "source_before_nodes": _optional_float(row["source_before_nodes"]),
        "n_rank": _optional_float(row["n_rank"]),
        "baseline_after_params": _optional_float(row["after_params"]),
        "baseline_rank_gap": _optional_float(row["rank_gap"]),
        "baseline_rendered": row.get("rendered"),
        "probe_mode": config.mode,
        "probe_max_passes": config.max_passes,
        "probe_inner_limit": config.inner_limit,
        "probe_match_limit": config.match_limit if config.mode == "long_backoff" else None,
        "probe_ban_length": config.ban_length if config.mode == "long_backoff" else None,
        "probe_scheduler": config.scheduler_kind,
        "probe_status": None,
        "probe_runtime_ms": None,
        "probe_peak_rss_mb": None,
        "probe_passes": None,
        "probe_total_size": None,
        "probe_after_params": None,
        "probe_rank_gap": None,
        "probe_after_nodes": None,
        "probe_rendered": None,
        "probe_error": None,
        "param_delta": None,
        "rank_gap_delta": None,
        "improved": False,
        "reached_rank": False,
    }


def configure_pipeline_for_probe(config: ProbeConfig) -> str:
    """Mutate `pipeline` globals for one child-process probe run."""

    from egglog import back_off
    from egglog.exp.param_eq import pipeline

    pipeline.MAX_PASSES = config.max_passes
    pipeline.HASKELL_INNER_ITERATION_LIMIT = config.inner_limit
    pipeline.BACKOFF_MATCH_LIMIT = config.match_limit
    pipeline.BACKOFF_BAN_LENGTH = config.ban_length
    if config.mode == "long_backoff":
        pipeline.rewrite_scheduler = back_off(
            match_limit=config.match_limit,
            ban_length=config.ban_length,
            fresh_rematch=True,
        ).persistent()
    else:
        pipeline.rewrite_scheduler = None
    return config.scheduler_kind


def _evaluate_expression(source: str, config: ProbeConfig) -> dict[str, object]:
    start = time.perf_counter()
    try:
        from egglog.exp.param_eq.domain import parse_expression
        from egglog.exp.param_eq.pipeline import run_paper_pipeline

        scheduler_kind = configure_pipeline_for_probe(config)
        report = run_paper_pipeline(parse_expression(source))
    except Exception as exc:  # noqa: BLE001 - probe output should preserve failures as data.
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_ms": (time.perf_counter() - start) * 1000.0,
        }
    return {
        "status": "ok",
        "error": None,
        "runtime_ms": report.total_sec * 1000.0,
        "passes": report.passes,
        "total_size": report.total_size,
        "after_params": report.extracted_params,
        "after_nodes": report.extracted_nodes,
        "rendered": report.extracted,
        "scheduler": scheduler_kind,
    }


def _worker_entry(connection: Any, row: dict[str, object], config: ProbeConfig) -> None:
    try:
        connection.send(_evaluate_expression(str(row["source_orig_parsed_expr"]), config))
    finally:
        connection.close()


def run_rank_miss_probe_row(row: dict[str, object], config: ProbeConfig) -> dict[str, object]:
    output = _base_output_row(row, config)
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(child_connection, row, config))
    process.start()
    watch = watch_process(
        process,
        timeout_sec=config.timeout_sec,
        memory_limit_mb=config.memory_limit_mb,
        sample_interval_sec=config.sample_interval_sec,
    )
    child_connection.close()
    output["probe_peak_rss_mb"] = watch.peak_rss_mb
    if watch.status in {"timeout", "memory_limit"}:
        output["probe_status"] = watch.status
        output["probe_error"] = watch.status
        parent_connection.close()
        return output
    if parent_connection.poll():
        try:
            payload = parent_connection.recv()
        except EOFError:
            payload = {"status": "failed", "error": "worker exited without a result"}
    else:
        payload = {"status": "failed", "error": "worker exited without a result"}
    parent_connection.close()

    output["probe_status"] = payload.get("status")
    output["probe_error"] = payload.get("error")
    output["probe_runtime_ms"] = payload.get("runtime_ms")
    output["probe_scheduler"] = payload.get("scheduler", config.scheduler_kind)
    if payload.get("status") != "ok":
        return output

    probe_after_params = float(payload["after_params"])
    baseline_after_params = float(output["baseline_after_params"])
    n_rank = float(output["n_rank"])
    probe_rank_gap = probe_after_params - n_rank
    baseline_rank_gap = float(output["baseline_rank_gap"])
    output.update(
        {
            "probe_passes": payload.get("passes"),
            "probe_total_size": payload.get("total_size"),
            "probe_after_params": probe_after_params,
            "probe_rank_gap": probe_rank_gap,
            "probe_after_nodes": payload.get("after_nodes"),
            "probe_rendered": payload.get("rendered"),
            "param_delta": baseline_after_params - probe_after_params,
            "rank_gap_delta": baseline_rank_gap - probe_rank_gap,
            "improved": probe_after_params < baseline_after_params,
            "reached_rank": probe_after_params <= n_rank,
        }
    )
    return output


def load_rank_miss_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=[NA_REPR], keep_default_na=True)


def filter_rank_miss_rows(
    frame: pd.DataFrame,
    *,
    dataset: str | None = None,
    algorithm: str | None = None,
    input_kind: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    filtered = frame
    if dataset is not None:
        filtered = filtered[filtered["dataset"] == dataset]
    if algorithm is not None:
        filtered = filtered[(filtered["algorithm"] == algorithm) | (filtered["algorithm_raw"] == algorithm)]
    if input_kind is not None:
        filtered = filtered[filtered["input_kind"] == input_kind]
    if offset:
        filtered = filtered.iloc[offset:]
    if limit is not None:
        filtered = filtered.iloc[:limit]
    return filtered.reset_index(drop=True)


def _mode_configs(args: argparse.Namespace) -> list[ProbeConfig]:
    modes: list[ProbeMode]
    if args.mode == "both":
        modes = ["long_backoff", "no_backoff"]
    else:
        modes = [args.mode]
    configs = []
    for mode in modes:
        timeout_sec = args.long_backoff_timeout_sec if mode == "long_backoff" else args.no_backoff_timeout_sec
        configs.append(
            ProbeConfig(
                mode=mode,
                max_passes=args.max_passes,
                inner_limit=args.inner_limit,
                match_limit=args.match_limit,
                ban_length=args.ban_length,
                timeout_sec=timeout_sec,
                memory_limit_mb=args.memory_limit_mb,
                sample_interval_sec=args.sample_interval_sec,
            )
        )
    return configs


def _sort_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = frame.copy()
    result["_expr_len"] = result["source_orig_parsed_expr"].map(len)
    result = result.sort_values(
        [
            "probe_mode",
            "source_orig_parsed_n_params",
            "source_before_nodes",
            "_expr_len",
            "dataset",
            "raw_index",
            "algo_row",
            "input_kind",
        ]
    )
    return result.loc[:, [*OUTPUT_COLUMNS, "_expr_len"]].drop(columns="_expr_len")


def run_probe_frame(
    rows: pd.DataFrame,
    configs: list[ProbeConfig],
    *,
    workers: int = DEFAULT_WORKERS,
    console: Console | None = None,
    runner=run_rank_miss_probe_row,
) -> pd.DataFrame:
    console = console or Console()
    all_results: list[dict[str, object]] = []
    row_dicts = rows.to_dict(orient="records")
    if not row_dicts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    for config in configs:
        capped_workers = cap_workers_for_memory(workers, memory_limit_mb=config.memory_limit_mb)
        status_counts: Counter[str] = Counter()
        max_peak_rss = 0.0
        console.log(
            f"running {config.mode}: rows={len(row_dicts)} workers={capped_workers} "
            f"timeout={config.timeout_sec}s rss_cap={config.memory_limit_mb}MB"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(config.mode, total=len(row_dicts))
            with ThreadPoolExecutor(max_workers=capped_workers) as executor:
                futures = [executor.submit(runner, row, config) for row in row_dicts]
                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)
                    status_counts[str(result["probe_status"])] += 1
                    peak_rss = result.get("probe_peak_rss_mb")
                    if _notna(peak_rss):
                        max_peak_rss = max(max_peak_rss, float(peak_rss))
                    progress.update(
                        task,
                        description=(
                            f"{config.mode} ok={status_counts['ok']} timeout={status_counts['timeout']} "
                            f"mem={status_counts['memory_limit']} max_rss={max_peak_rss:.1f}MB"
                        ),
                    )
                    progress.advance(task)
        _print_mode_summary(console, pd.DataFrame.from_records(all_results), config.mode)
    return _sort_output(pd.DataFrame.from_records(all_results))


def _print_mode_summary(console: Console, results: pd.DataFrame, mode: ProbeMode) -> None:
    mode_results = results[results["probe_mode"] == mode].copy()
    if mode_results.empty:
        console.log(f"{mode}: no rows")
        return
    status_counts = mode_results["probe_status"].value_counts(dropna=False)
    ok = mode_results[mode_results["probe_status"] == "ok"]
    improved = ok[ok["improved"] == True]  # noqa: E712 - pandas boolean mask.
    reached = ok[ok["reached_rank"] == True]  # noqa: E712 - pandas boolean mask.
    table = Table(title=f"{mode} summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("rows", str(len(mode_results)))
    table.add_row("ok", str(int(status_counts.get("ok", 0))))
    table.add_row("improved", str(len(improved)))
    table.add_row("reached_rank", str(len(reached)))
    table.add_row("timeout", str(int(status_counts.get("timeout", 0))))
    table.add_row("memory_limit", str(int(status_counts.get("memory_limit", 0))))
    max_rss = pd.to_numeric(mode_results["probe_peak_rss_mb"], errors="coerce").max()
    table.add_row("max_peak_rss_mb", "na" if math.isnan(max_rss) else f"{max_rss:.1f}")
    console.print(table)
    if improved.empty:
        console.log(f"{mode}: no improved rows")
        return
    examples = improved.sort_values(["source_orig_parsed_n_params", "source_before_nodes", "param_delta"], ascending=[True, True, False]).head(10)
    example_table = Table(title=f"{mode} smallest improved examples")
    for column in ["dataset", "algorithm", "raw_index", "input_kind", "baseline_after_params", "probe_after_params", "n_rank"]:
        example_table.add_column(column)
    for row in examples.itertuples(index=False):
        example_table.add_row(
            str(row.dataset),
            str(row.algorithm),
            str(row.raw_index),
            str(row.input_kind),
            str(row.baseline_after_params),
            str(row.probe_after_params),
            str(row.n_rank),
        )
    console.print(example_table)


def write_probe_results(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep=NA_REPR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_rank_misses_path())
    parser.add_argument("--output", type=Path, default=default_longer_runs_path())
    parser.add_argument("--mode", choices=["long_backoff", "no_backoff", "both"], default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dataset")
    parser.add_argument("--algorithm")
    parser.add_argument("--input-kind", choices=["original", "sympy"])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--memory-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--sample-interval-sec", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC)
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    parser.add_argument("--inner-limit", type=int, default=DEFAULT_INNER_LIMIT)
    parser.add_argument("--match-limit", type=int, default=DEFAULT_MATCH_LIMIT)
    parser.add_argument("--ban-length", type=int, default=DEFAULT_BAN_LENGTH)
    parser.add_argument("--long-backoff-timeout-sec", type=float, default=DEFAULT_LONG_BACKOFF_TIMEOUT_SEC)
    parser.add_argument("--no-backoff-timeout-sec", type=float, default=DEFAULT_NO_BACKOFF_TIMEOUT_SEC)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = filter_rank_miss_rows(
        load_rank_miss_rows(args.input),
        dataset=args.dataset,
        algorithm=args.algorithm,
        input_kind=args.input_kind,
        offset=args.offset,
        limit=args.limit,
    )
    configs = _mode_configs(args)
    frame = run_probe_frame(rows, configs, workers=args.workers)
    write_probe_results(frame, args.output)
    print(f"{args.output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
