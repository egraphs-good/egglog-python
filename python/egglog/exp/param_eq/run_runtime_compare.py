"""Build an apples-to-apples Pagie runtime sweep for archived Haskell, live Haskell, and Egglog."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from egglog.exp.param_eq.pipeline import parse_expression, run_paper_pipeline
from egglog.exp.param_eq.paths import ARTIFACT_DIR, llvm_bin_dir, param_eq_data_dir
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ARCHIVED_RUNTIME_PATH = ARTIFACT_DIR / "pagie_runtime_scatter.csv"
RUNTIME_COMPARE_PATH = ARTIFACT_DIR / "pagie_runtime_compare.csv"
HASKELL_ROOT = param_eq_data_dir()
DEFAULT_BATCH_SIZE = 18
DEFAULT_HASKELL_WORKERS = 2
DEFAULT_EGGLOG_WORKERS = max(1, min(os.cpu_count() or 1, 4))
ALGORITHM_RENAMES = {"SRjl": "PySR", "GOMEA": "GP-GOMEA"}


def _clean_algorithm(name: str) -> str:
    return ALGORITHM_RENAMES.get(name, name)


def _load_pagie_rows() -> list[dict[str, str]]:
    results_path = HASKELL_ROOT / "results" / "pagie_results"
    with results_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    counts: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, str]] = []
    for raw_index, row in enumerate(raw_rows):
        raw_algorithm = row["algorithm"]
        counts[raw_algorithm] += 1
        rows.append(
            {
                "raw_index": str(raw_index),
                "algorithm_raw": raw_algorithm,
                "algorithm": _clean_algorithm(raw_algorithm),
                "algo_row": str(counts[raw_algorithm]),
                "expr": row["expr"].strip(),
            }
        )
    return rows


def _build_haskell_program(rows: list[dict[str, str]]) -> str:
    case_lines: list[str] = []
    for index, row in enumerate(rows):
        algorithm = json.dumps(row["algorithm_raw"])
        algo_row = json.dumps(row["algo_row"])
        zero_index = int(row["algo_row"]) - 1
        prefix = "  " if index == 0 else "  , "
        case_lines.append(f"{prefix}({algorithm}, {algo_row}, {zero_index})")
    joined_case_lines = "\n".join(case_lines)
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
            "type RowCase = (String, String, Int)",
            "",
            "cases :: [RowCase]",
            "cases =",
            "  [",
            joined_case_lines,
            "  ]",
            "",
            "sanitize :: String -> String",
            "sanitize = map (\\c -> if c == '\\t' || c == '\\n' then ' ' else c)",
            "",
            "emitCase :: RowCase -> IO ()",
            "emitCase (algorithm, algoRow, rowIndex) = do",
            "  let expr = (pagieSR M.! algorithm) !! rowIndex",
            "      beforeNodes = countNodes expr",
            "  start <- getPOSIXTime",
            "  afterNodes <- evaluate (countNodes (simplifyE expr))",
            "  end <- getPOSIXTime",
            "  let runtimeMs = (realToFrac (end - start) :: Double) * 1000.0",
            "      fields = [algorithm, algoRow, show beforeNodes, show afterNodes, show runtimeMs]",
            "  putStrLn (intercalate \"\\t\" (map sanitize fields))",
            "",
            "main :: IO ()",
            "main = mapM_ emitCase cases",
            "",
        ]
    )


def _run_haskell_chunk(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    program = _build_haskell_program(rows)
    with tempfile.NamedTemporaryFile("w", suffix=".hs", delete=False) as handle:
        handle.write(program)
        temp_path = Path(handle.name)
    try:
        env = dict(os.environ)
        llvm_bin = llvm_bin_dir()
        if llvm_bin is not None:
            env["PATH"] = f"{llvm_bin}:{env['PATH']}"
        output = subprocess.check_output(
            ["stack", "exec", "--", "runghc", "-isrc", str(temp_path), "+RTS", "-K3G", "-RTS"],
            cwd=HASKELL_ROOT,
            env=env,
            text=True,
            timeout=3600,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    results: list[dict[str, str]] = []
    for line in output.splitlines():
        algorithm_raw, algo_row, before_nodes, after_nodes, runtime_ms = line.split("\t", maxsplit=4)
        results.append(
            {
                "implementation": "Live Haskell",
                "algorithm_raw": algorithm_raw,
                "algorithm": _clean_algorithm(algorithm_raw),
                "algo_row": algo_row,
                "node_count": before_nodes,
                "after_nodes": after_nodes,
                "runtime_ms": runtime_ms,
                "status": "saturated",
            }
        )
    return results


def _run_haskell_rows_serial(rows: list[dict[str, str]], *, batch_size: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            results.extend(_run_haskell_chunk(batch))
        except subprocess.CalledProcessError:
            if len(batch) == 1:
                row = batch[0]
                print(
                    f"live Haskell runtime sweep failed on {row['algorithm_raw']}#{row['algo_row']}",
                    flush=True,
                )
                results.append(
                    {
                        "implementation": "Live Haskell",
                        "algorithm_raw": row["algorithm_raw"],
                        "algorithm": row["algorithm"],
                        "algo_row": row["algo_row"],
                        "node_count": "na",
                        "after_nodes": "na",
                        "runtime_ms": "na",
                        "status": "stack_overflow",
                    }
                )
                continue
            mid = len(batch) // 2
            results.extend(_run_haskell_rows_serial(batch[:mid], batch_size=max(1, mid)))
            results.extend(_run_haskell_rows_serial(batch[mid:], batch_size=max(1, len(batch) - mid)))
    return results


def _run_haskell_rows(rows: list[dict[str, str]], *, batch_size: int, workers: int) -> list[dict[str, str]]:
    if workers <= 1:
        return _run_haskell_rows_serial(rows, batch_size=batch_size)

    batches = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
    results: list[dict[str, str]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("live Haskell runtime rows", total=len(rows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_haskell_rows_serial, batch, batch_size=batch_size): len(batch) for batch in batches
            }
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
                progress.advance(task, futures[future])
    return results


def _run_egglog_row(row: dict[str, str]) -> dict[str, str]:
    num = parse_expression(row["expr"])
    start = time.perf_counter()
    report = run_paper_pipeline(num)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "implementation": "Egglog",
        "algorithm_raw": row["algorithm_raw"],
        "algorithm": row["algorithm"],
        "algo_row": row["algo_row"],
        "node_count": str(report.before_nodes),
        "after_nodes": str(report.after_nodes),
        "runtime_ms": f"{elapsed_ms:.6f}",
        "status": report.status,
    }


def _run_egglog_rows(rows: list[dict[str, str]], *, workers: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("egglog runtime rows", total=len(rows))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_egglog_row, row) for row in rows]
            for future in as_completed(futures):
                results.append(future.result())
                progress.advance(task)
    return results


def _load_archived_runtime_rows() -> list[dict[str, str]]:
    with ARCHIVED_RUNTIME_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "implementation": "Archived Haskell",
            "algorithm_raw": "",
            "algorithm": "",
            "algo_row": "",
            "node_count": row["node_count"],
            "after_nodes": "",
            "runtime_ms": row["runtime_ms"],
            "status": "archived_benchmark",
        }
        for row in rows
    ]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNTIME_COMPARE_PATH)
    parser.add_argument("--haskell-workers", type=int, default=DEFAULT_HASKELL_WORKERS)
    parser.add_argument("--egglog-workers", type=int, default=DEFAULT_EGGLOG_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    rows = _load_pagie_rows()
    archived_rows = _load_archived_runtime_rows()
    live_rows = _run_haskell_rows(rows, batch_size=args.batch_size, workers=args.haskell_workers)
    egglog_rows = _run_egglog_rows(rows, workers=args.egglog_workers)
    _write_rows(args.output, archived_rows + live_rows + egglog_rows)
    print(args.output)


if __name__ == "__main__":
    main()
