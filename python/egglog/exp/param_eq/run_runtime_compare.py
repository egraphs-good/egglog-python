"""Build an apples-to-apples Pagie runtime sweep for archived Haskell, live Haskell, and Egglog."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from egglog.exp.param_eq.original_results import NA_REPR, load_archived_runtimes, load_dataset_results
from egglog.exp.param_eq.paths import artifact_dir, llvm_bin_dir, param_eq_data_dir
from egglog.exp.param_eq.resource_guard import (
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_SAMPLE_INTERVAL_SEC,
    cap_workers_for_memory,
    total_system_memory_mb,
    watch_process,
    watch_subprocess,
)

HASKELL_TIMEOUT_SECONDS = 10
EGGLOG_TIMEOUT_SECONDS = 180
DEFAULT_HASKELL_WORKERS = 2
DEFAULT_EGGLOG_WORKERS = 5
SUCCESS_STATUS = "saturated"
ALGORITHM_RENAMES = {"SRjl": "PySR", "GOMEA": "GP-GOMEA"}

RUNTIME_COMPARE_SCHEMA = pa.DataFrameSchema(
    {
        "implementation": pa.Column(str),
        "algorithm_raw": pa.Column(str, nullable=True),
        "algorithm": pa.Column(str, nullable=True),
        "algo_row": pa.Column(float, nullable=True),
        "node_count": pa.Column(float, nullable=True),
        "after_nodes": pa.Column(float, nullable=True),
        "runtime_ms": pa.Column(float, nullable=True),
        "peak_rss_mb": pa.Column(float, nullable=True),
        "status": pa.Column(str),
    },
    strict=True,
    ordered=True,
    coerce=True,
)


def default_runtime_compare_path() -> Path:
    return artifact_dir() / "runtime_compare.csv"


def load_runtime_compare(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or default_runtime_compare_path(), na_values=[NA_REPR], keep_default_na=True)
    return RUNTIME_COMPARE_SCHEMA.validate(frame)


def write_runtime_compare(frame: pd.DataFrame, path: Path | None = None) -> None:
    validated = RUNTIME_COMPARE_SCHEMA.validate(frame)
    output = path or default_runtime_compare_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False, na_rep=NA_REPR)


def _clean_algorithm(name: str) -> str:
    return ALGORITHM_RENAMES.get(name, name)


def _load_pagie_rows() -> list[dict[str, object]]:
    raw_rows = load_dataset_results("pagie")
    counts: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, object]] = []
    for raw_index, row in enumerate(raw_rows.itertuples(index=False), start=0):
        raw_algorithm = str(row.algorithm)
        counts[raw_algorithm] += 1
        rows.append(
            {
                "raw_index": raw_index,
                "algorithm_raw": raw_algorithm,
                "algorithm": _clean_algorithm(raw_algorithm),
                "algo_row": counts[raw_algorithm],
                "expr": str(row.expr).strip(),
            }
        )
    return rows


def _build_haskell_program(row: dict[str, object]) -> str:
    algorithm = json.dumps(str(row["algorithm_raw"]))
    zero_index = int(row["algo_row"]) - 1
    return "\n".join(
        [
            "import Control.Exception (evaluate)",
            "import Data.List (intercalate)",
            "import qualified Data.Map as M",
            "import Data.SRTree",
            "import Data.Time.Clock.POSIX (getPOSIXTime)",
            "import FixTree (simplifyE)",
            "import PagieSR (pagieSR)",
            "",
            "sanitize :: String -> String",
            "sanitize = map (\\c -> if c == '\\t' || c == '\\n' then ' ' else c)",
            "",
            "emitCase :: String -> Int -> IO ()",
            "emitCase algorithm rowIndex = do",
            "  let expr = (pagieSR M.! algorithm) !! rowIndex",
            "      beforeNodes = countNodes expr",
            "  start <- getPOSIXTime",
            "  afterNodes <- evaluate (countNodes (simplifyE expr))",
            "  end <- getPOSIXTime",
            "  let runtimeMs = (realToFrac (end - start) :: Double) * 1000.0",
            "      fields = [show beforeNodes, show afterNodes, show runtimeMs]",
            "  putStrLn (intercalate \"\\t\" (map sanitize fields))",
            "",
            "main :: IO ()",
            f"main = emitCase {algorithm} {zero_index}",
            "",
        ]
    )


def _missing_haskell_runtime_row(row: dict[str, object], *, status: str) -> dict[str, object]:
    return {
        "implementation": "Live Haskell",
        "algorithm_raw": row["algorithm_raw"],
        "algorithm": row["algorithm"],
        "algo_row": row["algo_row"],
        "node_count": None,
        "after_nodes": None,
        "runtime_ms": None,
        "peak_rss_mb": None,
        "status": status,
    }


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


def _run_haskell_row(
    row: dict[str, object],
    *,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> tuple[dict[str, object], str | None]:
    program = _build_haskell_program(row)
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile("w", suffix=".hs", delete=False) as handle:
        handle.write(program)
        temp_path = Path(handle.name)
    try:
        env = dict(os.environ)
        llvm_bin = llvm_bin_dir()
        if llvm_bin is not None:
            env["PATH"] = f"{llvm_bin}:{env['PATH']}"
        process = subprocess.Popen(  # noqa: S603,S607
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
        return _missing_haskell_runtime_row(row, status="missing_haskell_error"), _short_error(str(error))
    finally:
        temp_path.unlink(missing_ok=True)

    if watch.status == "timeout":
        detail = f"peak_rss_mb={watch.peak_rss_mb:.1f}" if watch.peak_rss_mb is not None else "no rss sample"
        result = _missing_haskell_runtime_row(row, status="missing_timeout")
        result["peak_rss_mb"] = watch.peak_rss_mb
        return result, detail
    if watch.status == "memory_limit":
        detail = f"peak_rss_mb={watch.peak_rss_mb:.1f}" if watch.peak_rss_mb is not None else "no rss sample"
        result = _missing_haskell_runtime_row(row, status="missing_memory_limit")
        result["peak_rss_mb"] = watch.peak_rss_mb
        return result, detail

    if process.returncode != 0:
        stderr = stderr or stdout
        status = _failure_status(stderr)
        result = _missing_haskell_runtime_row(row, status=status)
        result["peak_rss_mb"] = watch.peak_rss_mb
        return result, _short_error(stderr)

    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        result = _missing_haskell_runtime_row(row, status="missing_bad_output")
        result["peak_rss_mb"] = watch.peak_rss_mb
        return (result, f"expected one output row, got {len(lines)}: {_short_error(stdout)}")
    try:
        before_nodes, after_nodes, runtime_ms = lines[0].split("\t", maxsplit=2)
    except ValueError:
        result = _missing_haskell_runtime_row(row, status="missing_bad_output")
        result["peak_rss_mb"] = watch.peak_rss_mb
        return (result, f"could not parse output row: {_short_error(lines[0])}")
    return (
        {
            "implementation": "Live Haskell",
            "algorithm_raw": row["algorithm_raw"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "node_count": float(before_nodes),
            "after_nodes": float(after_nodes),
            "runtime_ms": float(runtime_ms),
            "peak_rss_mb": watch.peak_rss_mb,
            "status": SUCCESS_STATUS,
        },
        None,
    )


def _runtime_row_label(row: dict[str, object]) -> str:
    return f"{row['algorithm_raw']}#{row['algo_row']} raw={row['raw_index']}"


def _log_missing(console: Console, row: dict[str, object], result: dict[str, object], detail: str | None) -> None:
    if result["status"] == SUCCESS_STATUS:
        return
    console.log(f"[yellow]{_runtime_row_label(row)}[/yellow] {result['status']}: {detail or 'no detail'}")


def _status_table(rows: list[dict[str, object]]) -> Table:
    counts = Counter(str(row["status"]) for row in rows)
    table = Table(title="Live Haskell Runtime Status")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    for status, count in sorted(counts.items()):
        style = "green" if status == SUCCESS_STATUS else "yellow"
        table.add_row(f"[{style}]{status}[/{style}]", str(count))
    return table


def _progress_description(
    *,
    label: str,
    completed: int,
    total: int,
    kill_count: int,
    max_peak_rss_mb: float | None,
    counts: Counter[str],
) -> str:
    peak = "na" if max_peak_rss_mb is None else f"{max_peak_rss_mb:.1f}MB"
    status_summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    return f"{label} {completed}/{total} killed={kill_count} max_rss={peak} {status_summary}".strip()


def _run_haskell_rows(
    rows: list[dict[str, object]],
    *,
    workers: int,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> list[dict[str, object]]:
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
                label="live Haskell runtime rows",
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
                result, detail = _run_haskell_row(
                    row,
                    memory_limit_mb=memory_limit_mb,
                    sample_interval_sec=sample_interval_sec,
                )
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
                        label="live Haskell runtime rows",
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
                    result, detail = future.result()
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
                            label="live Haskell runtime rows",
                            completed=sum(status_counts.values()),
                            total=len(rows),
                            kill_count=killed_count,
                            max_peak_rss_mb=max_peak_rss_mb,
                            counts=status_counts,
                        ),
                    )
    materialized = [row for row in results if row is not None]
    console.print(f"memory_limit_mb={memory_limit_mb} workers={workers} total_system_memory_mb={total_system_memory_mb():.1f}")
    console.print(_status_table(materialized))
    return materialized


def _run_egglog_row(row: dict[str, object]) -> dict[str, object]:
    from egglog.exp.param_eq.domain import parse_expression
    from egglog.exp.param_eq.pipeline import run_paper_pipeline

    start = time.perf_counter()
    try:
        report = run_paper_pipeline(parse_expression(str(row["expr"])))
    except Exception:
        return {
            "implementation": "Egglog",
            "algorithm_raw": row["algorithm_raw"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "node_count": None,
            "after_nodes": None,
            "runtime_ms": None,
            "peak_rss_mb": None,
            "status": "failed",
        }

    return {
        "implementation": "Egglog",
        "algorithm_raw": row["algorithm_raw"],
        "algorithm": row["algorithm"],
        "algo_row": row["algo_row"],
        "node_count": float(report.before_nodes),
        "after_nodes": float(report.extracted_nodes),
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
        "peak_rss_mb": None,
        "status": SUCCESS_STATUS,
    }


def _egglog_worker_entry(connection, row: dict[str, object]) -> None:
    try:
        connection.send(_run_egglog_row(row))
    finally:
        connection.close()


def _run_egglog_row_isolated(
    row: dict[str, object],
    *,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> dict[str, object]:
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_egglog_worker_entry, args=(child_connection, row))
    process.start()
    watch = watch_process(
        process,
        timeout_sec=EGGLOG_TIMEOUT_SECONDS,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
    )
    child_connection.close()
    if watch.status == "timeout":
        result = {
            "implementation": "Egglog",
            "algorithm_raw": row["algorithm_raw"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "node_count": None,
            "after_nodes": None,
            "runtime_ms": None,
            "peak_rss_mb": watch.peak_rss_mb,
            "status": "timeout",
        }
    elif watch.status == "memory_limit":
        result = {
            "implementation": "Egglog",
            "algorithm_raw": row["algorithm_raw"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "node_count": None,
            "after_nodes": None,
            "runtime_ms": None,
            "peak_rss_mb": watch.peak_rss_mb,
            "status": "memory_limit",
        }
    elif parent_connection.poll():
        result = parent_connection.recv()
        result["peak_rss_mb"] = watch.peak_rss_mb
    else:
        result = {
            "implementation": "Egglog",
            "algorithm_raw": row["algorithm_raw"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "node_count": None,
            "after_nodes": None,
            "runtime_ms": None,
            "peak_rss_mb": watch.peak_rss_mb,
            "status": "failed",
        }
    parent_connection.close()
    return result


def _run_egglog_rows(
    rows: list[dict[str, object]],
    *,
    workers: int,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    workers = cap_workers_for_memory(workers, memory_limit_mb=memory_limit_mb)
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
                label="egglog runtime rows",
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
                    _run_egglog_row_isolated,
                    row,
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
                        label="egglog runtime rows",
                        completed=len(results),
                        total=len(rows),
                        kill_count=killed_count,
                        max_peak_rss_mb=max_peak_rss_mb,
                        counts=status_counts,
                    ),
                )
    Console().print(f"memory_limit_mb={memory_limit_mb} workers={workers} total_system_memory_mb={total_system_memory_mb():.1f}")
    return results


def _load_archived_runtime_rows() -> list[dict[str, object]]:
    rows = load_archived_runtimes().to_dict(orient="records")
    return [
        {
            "implementation": "Archived Haskell",
            "algorithm_raw": None,
            "algorithm": None,
            "algo_row": None,
            "node_count": row["node_count"],
            "after_nodes": None,
            "runtime_ms": row["runtime_ms"],
            "peak_rss_mb": None,
            "status": "archived_benchmark",
        }
        for row in rows
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_runtime_compare_path())
    parser.add_argument("--haskell-workers", type=int, default=DEFAULT_HASKELL_WORKERS)
    parser.add_argument("--egglog-workers", type=int, default=DEFAULT_EGGLOG_WORKERS)
    parser.add_argument("--memory-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--sample-interval-sec", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    rows = _load_pagie_rows()
    archived_rows = _load_archived_runtime_rows()
    live_rows = _run_haskell_rows(
        rows,
        workers=args.haskell_workers,
        memory_limit_mb=args.memory_limit_mb,
        sample_interval_sec=args.sample_interval_sec,
    )
    egglog_rows = _run_egglog_rows(
        rows,
        workers=args.egglog_workers,
        memory_limit_mb=args.memory_limit_mb,
        sample_interval_sec=args.sample_interval_sec,
    )
    write_runtime_compare(pd.DataFrame.from_records(archived_rows + live_rows + egglog_rows), args.output)
    Console().print(f"[green]wrote[/green] {args.output}")


if __name__ == "__main__":
    main()
