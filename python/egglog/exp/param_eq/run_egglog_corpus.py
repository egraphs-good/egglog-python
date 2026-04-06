"""Run the retained Egglog param-eq baseline across the paper rows."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from egglog.exp.param_eq.paths import ARTIFACT_DIR

TIMEOUT_SEC = 180.0

OUTPUT_DIR = ARTIFACT_DIR
HASKELL_ROWS_PATH = OUTPUT_DIR / "haskell_paper_rows.csv"
EGGLOG_ROWS_PATH = OUTPUT_DIR / "egglog_paper_rows.csv"
DEFAULT_WORKERS = max(1, min(os.cpu_count() or 1, 4))


def _load_rows() -> list[dict[str, str]]:
    with HASKELL_ROWS_PATH.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["is_paper_row"] == "1"]


def _run_one(source: str) -> dict[str, str]:
    start = time.perf_counter()
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "egglog.exp.param_eq",
                f"--expr={source}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.communicate(timeout=1.0)
        return {
            "status": "timeout",
            "runtime_ms": f"{TIMEOUT_SEC * 1000.0:.6f}",
            "before_nodes": "na",
            "before_params": "na",
            "after_nodes": "na",
            "after_params": "na",
            "total_size": "na",
            "egraph_nodes": "na",
            "eclass_count": "na",
            "passes": "na",
            "extracted_cost": "na",
            "rendered": "timeout",
        }
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        message = stderr.strip() or stdout.strip() or "subprocess failure"
        return {
            "status": "failed",
            "runtime_ms": f"{elapsed_ms:.6f}",
            "before_nodes": "na",
            "before_params": "na",
            "after_nodes": "na",
            "after_params": "na",
            "total_size": "na",
            "egraph_nodes": "na",
            "eclass_count": "na",
            "passes": "na",
            "extracted_cost": "na",
            "rendered": message.splitlines()[0],
        }
    payload = json.loads(stdout)
    return {
        "status": str(payload["status"]),
        "runtime_ms": f"{elapsed_ms:.6f}",
        "before_nodes": str(payload["before_nodes"]),
        "before_params": str(payload["before_params"]),
        "after_nodes": str(payload["after_nodes"]),
        "after_params": str(payload["after_params"]),
        "total_size": str(payload["total_size"]),
        "egraph_nodes": str(payload["node_count"]),
        "eclass_count": str(payload["eclass_count"]),
        "passes": str(payload["passes"]),
        "extracted_cost": str(payload["extracted_cost"]),
        "rendered": str(payload["rendered"]),
    }


def _run_row(source_row: dict[str, str]) -> dict[str, str]:
    original = _run_one(source_row["original_expr"])
    sympy = _run_one(source_row["sympy_expr"])
    return {
        "dataset": source_row["dataset"],
        "raw_index": source_row["raw_index"],
        "algorithm_raw": source_row["algorithm_raw"],
        "algorithm": source_row["algorithm"],
        "algo_row": source_row["algo_row"],
        "is_paper_row": source_row["is_paper_row"],
        "drop_reason": source_row["drop_reason"],
        "n_params": source_row["n_params"],
        "n_rank": source_row["n_rank"],
        "original_expr": source_row["original_expr"],
        "sympy_expr": source_row["sympy_expr"],
        "orig_status": original["status"],
        "orig_runtime_ms": original["runtime_ms"],
        "orig_nodes": original["before_nodes"],
        "orig_params": original["before_params"],
        "simpl_nodes": original["after_nodes"],
        "simpl_params": original["after_params"],
        "orig_total_size": original["total_size"],
        "orig_egraph_nodes": original["egraph_nodes"],
        "orig_eclass_count": original["eclass_count"],
        "orig_passes": original["passes"],
        "orig_extracted_cost": original["extracted_cost"],
        "orig_rendered": original["rendered"],
        "sympy_status": sympy["status"],
        "sympy_runtime_ms": sympy["runtime_ms"],
        "orig_nodes_sympy": sympy["before_nodes"],
        "orig_params_sympy": sympy["before_params"],
        "simpl_nodes_sympy": sympy["after_nodes"],
        "simpl_params_sympy": sympy["after_params"],
        "sympy_total_size": sympy["total_size"],
        "sympy_egraph_nodes": sympy["egraph_nodes"],
        "sympy_eclass_count": sympy["eclass_count"],
        "sympy_passes": sympy["passes"],
        "sympy_extracted_cost": sympy["extracted_cost"],
        "sympy_rendered": sympy["rendered"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(EGGLOG_ROWS_PATH),
        help="Where to write the resulting CSV artifact.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of Egglog rows to evaluate in parallel.",
    )
    args = parser.parse_args()
    output_path = pathlib.Path(args.output)

    rows = _load_rows()
    total = len(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".csv.partial")
    completed_keys: set[tuple[str, str, str, str]] = set()
    fieldnames: list[str] | None = None
    if temp_path.exists():
        with temp_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames) if reader.fieldnames is not None else None
            for row in reader:
                completed_keys.add((row["dataset"], row["raw_index"], row["algorithm"], row["algo_row"]))

    pending_rows = [
        row
        for row in rows
        if (
            row["dataset"],
            row["raw_index"],
            row["algorithm"],
            row["algo_row"],
        )
        not in completed_keys
    ]
    with temp_path.open("a" if completed_keys else "w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        if fieldnames is not None:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("egglog rows", total=total, completed=len(completed_keys))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures: dict[Future[dict[str, str]], tuple[str, str, str, str]] = {
                    executor.submit(_run_row, row): (
                        row["dataset"],
                        row["raw_index"],
                        row["algorithm"],
                        row["algo_row"],
                    )
                    for row in pending_rows
                }
                for future in as_completed(futures):
                    output_row = future.result()
                    if fieldnames is None:
                        fieldnames = list(output_row)
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                    assert writer is not None
                    writer.writerow(output_row)
                    handle.flush()
                    completed_keys.add(futures[future])
                    progress.advance(task)
    temp_path.replace(output_path)


if __name__ == "__main__":
    main()
