"""Minimal live-Haskell result storage plus joined loaders for `param_eq`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

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
from egglog.exp.param_eq.paths import artifact_dir, llvm_bin_dir, param_eq_data_dir
from egglog.exp.param_eq.resource_guard import (
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_SAMPLE_INTERVAL_SEC,
    WatchResult,
    cap_workers_for_memory,
    total_system_memory_mb,
    watch_subprocess,
)

HASKELL_TIMEOUT_SECONDS = 10
DEFAULT_WORKERS = 2
SUCCESS_STATUS = "saturated"
MISSING_HASKELL_SOURCE = "missing_haskell"

LIVE_RESULTS_SCHEMA = pa.DataFrameSchema(
    {
        "dataset": pa.Column(str),
        "raw_index": pa.Column(int),
        "algorithm_raw": pa.Column(str),
        "algo_row": pa.Column(int),
        "input_kind": pa.Column(str, checks=pa.Check.isin(INPUT_KIND_ORDER)),
        "status": pa.Column(str),
        "runtime_ms": pa.Column(float, nullable=True),
        "before_nodes": pa.Column(float, nullable=True),
        "before_params": pa.Column(float, nullable=True),
        "after_nodes": pa.Column(float, nullable=True),
        "after_params": pa.Column(float, nullable=True),
        "peak_rss_mb": pa.Column(float, nullable=True),
        "rendered": pa.Column(str, nullable=True),
    },
    strict=True,
    ordered=True,
    coerce=True,
)


def default_live_results_path() -> Path:
    return artifact_dir() / "live_results.csv"


def load_live_results_raw(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or default_live_results_path(), na_values=[NA_REPR], keep_default_na=True)
    return LIVE_RESULTS_SCHEMA.validate(frame)


def write_live_results(frame: pd.DataFrame, path: Path | None = None) -> None:
    validated = LIVE_RESULTS_SCHEMA.validate(frame)
    output = path or default_live_results_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False, na_rep=NA_REPR)


def load_live_results(path: Path | None = None, *, source_root: Path | None = None) -> pd.DataFrame:
    source = load_original_results_for_join(source_root)
    live = load_live_results_raw(path)
    joined = source.merge(live, on=RESULT_KEY_COLUMNS, how="inner", validate="one_to_one")
    joined["implementation"] = "haskell"
    joined["variant"] = "live"
    joined["baseline_source"] = [
        "live_haskell" if status == SUCCESS_STATUS else MISSING_HASKELL_SOURCE for status in joined["status"]
    ]
    joined["egraph_total_size"] = None
    joined["passes"] = None
    joined["extracted_cost"] = None
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
    return joined.sort_values(RESULT_KEY_COLUMNS).reset_index(drop=True)


def _load_rows() -> list[dict[str, object]]:
    frame = load_original_results().loc[:, RESULT_KEY_COLUMNS]
    return frame.sort_values(RESULT_KEY_COLUMNS).to_dict(orient="records")


def _row_label(row: dict[str, object]) -> str:
    return f"{row['dataset']}/{row['algorithm_raw']} row={row['algo_row']} raw={row['raw_index']} {row['input_kind']}"


def _build_haskell_program(row: dict[str, object]) -> str:
    dataset = json.dumps(str(row["dataset"]))
    raw_algorithm = json.dumps(str(row["algorithm_raw"]))
    zero_index = int(row["algo_row"]) - 1
    lookup = "lookupOriginal" if row["input_kind"] == "original" else "lookupSympy"
    return "\n".join([
        "import Data.List (intercalate)",
        "import qualified Data.Map as M",
        "import Data.SRTree",
        "import Data.SRTree.Print",
        "import Data.Time.Clock.POSIX (getPOSIXTime)",
        "import FixTree",
        "import KotanchekSR (kotanchekSR)",
        "import KotanchekSympy (kotanchekSympy)",
        "import PagieSR (pagieSR)",
        "import PagieSympy (pagieSympy)",
        "import Reparam (replaceConstsWithParams)",
        "",
        "sanitize :: String -> String",
        "sanitize = map (\\c -> if c == '\\t' || c == '\\n' then ' ' else c)",
        "",
        "lookupOriginal :: String -> String -> Int -> SRTree Int Double",
        "lookupOriginal dataset algorithm rowIndex = case dataset of",
        '  "pagie" -> (pagieSR M.! algorithm) !! rowIndex',
        '  "kotanchek" -> (kotanchekSR M.! algorithm) !! rowIndex',
        '  _ -> error "unknown dataset"',
        "",
        "lookupSympy :: String -> String -> Int -> SRTree Int Double",
        "lookupSympy dataset algorithm rowIndex = case dataset of",
        '  "pagie" -> (pagieSympy M.! algorithm) !! rowIndex',
        '  "kotanchek" -> (kotanchekSympy M.! algorithm) !! rowIndex',
        '  _ -> error "unknown dataset"',
        "",
        "emitExpr :: SRTree Int Double -> IO ()",
        "emitExpr expr = do",
        "  start <- getPOSIXTime",
        "  let simplified = simplifyE expr",
        "  end <- getPOSIXTime",
        "  let beforeNodes = countNodes expr",
        "      beforeParams = recountParams (replaceConstsWithParams expr)",
        "      afterNodes = countNodes simplified",
        "      afterParams = recountParams (replaceConstsWithParams simplified)",
        "      runtimeMs = (realToFrac (end - start) :: Double) * 1000.0",
        "      rendered = showDefault simplified",
        "      fields =",
        "        [ show beforeNodes",
        "        , show beforeParams",
        "        , show afterNodes",
        "        , show afterParams",
        "        , show runtimeMs",
        "        , rendered",
        "        ]",
        '  putStrLn (intercalate "\\t" (map sanitize fields))',
        "",
        "main :: IO ()",
        f"main = emitExpr ({lookup} {dataset} {raw_algorithm} {zero_index})",
        "",
    ])


def _decode_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _short_error(text: str, *, limit: int = 260) -> str:
    condensed = " ".join(text.split())
    if not condensed:
        return "no process output"
    if len(condensed) <= limit:
        return condensed
    return f"{condensed[: limit - 1]}..."


def _failure_status(stderr: str) -> str:
    lowered = stderr.lower()
    if "stack overflow" in lowered or "stack space overflow" in lowered:
        return "missing_stack_overflow"
    return "missing_haskell_error"


def _result_row(
    row: dict[str, object],
    *,
    status: str,
    runtime_ms: object,
    before_nodes: object,
    before_params: object,
    after_nodes: object,
    after_params: object,
    peak_rss_mb: object,
    rendered: object,
) -> dict[str, object]:
    return {
        "dataset": row["dataset"],
        "raw_index": row["raw_index"],
        "algorithm_raw": row["algorithm_raw"],
        "algo_row": row["algo_row"],
        "input_kind": row["input_kind"],
        "status": status,
        "runtime_ms": runtime_ms,
        "before_nodes": before_nodes,
        "before_params": before_params,
        "after_nodes": after_nodes,
        "after_params": after_params,
        "peak_rss_mb": peak_rss_mb,
        "rendered": canonical_expr(rendered),
    }


def _missing_result_row(row: dict[str, object], *, status: str, peak_rss_mb: object = None) -> dict[str, object]:
    return _result_row(
        row,
        status=status,
        runtime_ms=None,
        before_nodes=None,
        before_params=None,
        after_nodes=None,
        after_params=None,
        peak_rss_mb=peak_rss_mb,
        rendered=None,
    )


def _parse_haskell_stdout(row: dict[str, object], stdout: str) -> tuple[dict[str, object], str | None]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return (
            _missing_result_row(row, status="missing_bad_output"),
            f"expected one output row, got {len(lines)}: {_short_error(stdout)}",
        )
    try:
        before_nodes, before_params, after_nodes, after_params, runtime_ms, rendered = lines[0].split("\t", maxsplit=5)
    except ValueError:
        return (
            _missing_result_row(row, status="missing_bad_output"),
            f"could not parse output row: {_short_error(lines[0])}",
        )
    return (
        _result_row(
            row,
            status=SUCCESS_STATUS,
            runtime_ms=float(runtime_ms),
            before_nodes=float(before_nodes),
            before_params=float(before_params),
            after_nodes=float(after_nodes),
            after_params=float(after_params),
            peak_rss_mb=None,
            rendered=rendered,
        ),
        None,
    )


def _failure_row_from_watch(row: dict[str, object], watch: WatchResult) -> tuple[dict[str, object], str | None]:
    status = "missing_timeout" if watch.status == "timeout" else "missing_memory_limit"
    detail = f"peak_rss_mb={watch.peak_rss_mb:.1f}" if watch.peak_rss_mb is not None else "no rss sample"
    return _missing_result_row(row, status=status, peak_rss_mb=watch.peak_rss_mb), detail


def _run_haskell_row(
    row: dict[str, object],
    *,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> tuple[dict[str, object], str | None]:
    program = _build_haskell_program(row)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".hs", delete=False) as handle:
            handle.write(program)
            temp_path = Path(handle.name)
        env = dict(os.environ)
        llvm_bin = llvm_bin_dir()
        if llvm_bin is not None:
            env["PATH"] = f"{llvm_bin}:{env['PATH']}"
        process = subprocess.Popen(
            ["stack", "exec", "--", "runghc", "-isrc", str(temp_path), "+RTS", "-K3G", "-RTS"],  # noqa: S607
            cwd=param_eq_data_dir(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        watch = watch_subprocess(
            process,
            timeout_sec=HASKELL_TIMEOUT_SECONDS,
            memory_limit_mb=memory_limit_mb,
            sample_interval_sec=sample_interval_sec,
        )
        stdout, stderr = process.communicate()
    except OSError as error:
        return _missing_result_row(row, status="missing_haskell_error"), _short_error(str(error))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    if watch.status != "completed":
        return _failure_row_from_watch(row, watch)

    if process.returncode != 0:
        stderr = stderr or stdout
        status = _failure_status(stderr)
        return _missing_result_row(row, status=status, peak_rss_mb=watch.peak_rss_mb), _short_error(stderr)
    parsed, detail = _parse_haskell_stdout(row, stdout)
    parsed["peak_rss_mb"] = watch.peak_rss_mb
    return parsed, detail


def _log_missing(console: Console, row: dict[str, object], result: dict[str, object], detail: str | None) -> None:
    if result["status"] == SUCCESS_STATUS:
        return
    console.log(f"[yellow]{_row_label(row)}[/yellow] {result['status']}: {detail or 'no detail'}")


def _status_table(rows: list[dict[str, object]]) -> Table:
    counts = Counter(str(row["status"]) for row in rows)
    table = Table(title="Live Haskell Status")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    for status, count in sorted(counts.items()):
        style = "green" if status == SUCCESS_STATUS else "yellow"
        table.add_row(f"[{style}]{status}[/{style}]", str(count))
    return table


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
    return f"live Haskell rows {completed}/{total} killed={kill_count} max_rss={peak} {status_summary}".strip()


def run_live_results_frame(
    *,
    workers: int = DEFAULT_WORKERS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
) -> pd.DataFrame:
    rows = _load_rows()
    console = Console()
    workers = cap_workers_for_memory(workers, memory_limit_mb=memory_limit_mb)
    results: list[dict[str, object] | None] = [None] * len(rows)
    status_counts: Counter[str] = Counter()
    killed_count = 0
    max_peak_rss_mb = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
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
        if workers <= 1:
            for index, row in enumerate(rows):
                try:
                    result, detail = _run_haskell_row(
                        row,
                        memory_limit_mb=memory_limit_mb,
                        sample_interval_sec=sample_interval_sec,
                    )
                except Exception as error:  # pragma: no cover
                    result = _missing_result_row(row, status="missing_haskell_error")
                    detail = _short_error(str(error))
                results[index] = result
                _log_missing(console, row, result, detail)
                status_counts[str(result["status"])] += 1
                killed_count += int(result["status"] == "missing_memory_limit")
                peak_rss_mb = result["peak_rss_mb"]
                if peak_rss_mb is not None:
                    max_peak_rss_mb = peak_rss_mb if max_peak_rss_mb is None else max(max_peak_rss_mb, peak_rss_mb)
                progress.advance(task)
                progress.update(
                    task,
                    description=_progress_description(
                        completed=index + 1,
                        total=len(rows),
                        kill_count=killed_count,
                        max_peak_rss_mb=max_peak_rss_mb,
                        counts=status_counts,
                    ),
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_haskell_row,
                        row,
                        memory_limit_mb=memory_limit_mb,
                        sample_interval_sec=sample_interval_sec,
                    ): index
                    for index, row in enumerate(rows)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    row = rows[index]
                    try:
                        result, detail = future.result()
                    except Exception as error:  # pragma: no cover
                        result = _missing_result_row(row, status="missing_haskell_error")
                        detail = _short_error(str(error))
                    results[index] = result
                    _log_missing(console, row, result, detail)
                    status_counts[str(result["status"])] += 1
                    killed_count += int(result["status"] == "missing_memory_limit")
                    peak_rss_mb = result["peak_rss_mb"]
                    if peak_rss_mb is not None:
                        max_peak_rss_mb = peak_rss_mb if max_peak_rss_mb is None else max(max_peak_rss_mb, peak_rss_mb)
                    progress.advance(task)
                    progress.update(
                        task,
                        description=_progress_description(
                            completed=sum(status_counts.values()),
                            total=len(rows),
                            kill_count=killed_count,
                            max_peak_rss_mb=max_peak_rss_mb,
                            counts=status_counts,
                        ),
                    )
    materialized = [row for row in results if row is not None]
    console.print(
        f"memory_limit_mb={memory_limit_mb} workers={workers} total_system_memory_mb={total_system_memory_mb():.1f}"
    )
    console.print(_status_table(materialized))
    return LIVE_RESULTS_SCHEMA.validate(pd.DataFrame.from_records(materialized))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_live_results_path(),
        help="Path to write the minimal live Haskell results CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of Haskell rows to evaluate in parallel.",
    )
    parser.add_argument("--memory-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--sample-interval-sec", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    frame = run_live_results_frame(
        workers=args.workers,
        memory_limit_mb=args.memory_limit_mb,
        sample_interval_sec=args.sample_interval_sec,
    )
    write_live_results(frame, args.output)
    Console().print(f"[green]wrote[/green] {args.output}")


if __name__ == "__main__":
    main()
