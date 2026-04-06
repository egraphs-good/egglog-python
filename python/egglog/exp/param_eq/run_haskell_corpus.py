"""Run the current local param-eq Haskell pipeline across the retained corpus."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from egglog.exp.param_eq.generate_haskell_golden import _canonicalize
from egglog.exp.param_eq.paths import ARTIFACT_DIR, llvm_bin_dir, param_eq_data_dir
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ARCHIVED_ROWS_PATH = ARTIFACT_DIR / "haskell_paper_rows.csv"
LIVE_ROWS_PATH = ARTIFACT_DIR / "haskell_live_rows.csv"
HASKELL_ROOT = param_eq_data_dir()
DEFAULT_BATCH_SIZE = 24
DEFAULT_WORKERS = 2


def _load_rows() -> list[dict[str, str]]:
    with ARCHIVED_ROWS_PATH.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["is_paper_row"] == "1"]


def _build_haskell_program(rows: list[dict[str, str]]) -> str:
    case_lines: list[str] = []
    for index, row in enumerate(rows):
        dataset = json.dumps(row["dataset"])
        raw_index = json.dumps(row["raw_index"])
        algorithm = json.dumps(row["algorithm"])
        raw_algorithm = json.dumps(row["algorithm_raw"])
        algo_row = json.dumps(row["algo_row"])
        zero_index = int(row["algo_row"]) - 1
        prefix = "  " if index == 0 else "  , "
        case_lines.append(
            f'{prefix}(({dataset}, {raw_index}, {algorithm}, {algo_row}), {raw_algorithm}, {zero_index})'
        )
    joined_case_lines = "\n".join(case_lines)
    return "\n".join(
        [
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
            "type RowId = (String, String, String, String)",
            "type RowCase = (RowId, String, Int)",
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
            "lookupOriginal :: String -> String -> Int -> SRTree Int Double",
            "lookupOriginal dataset algorithm rowIndex = case dataset of",
            "  \"pagie\" -> (pagieSR M.! algorithm) !! rowIndex",
            "  \"kotanchek\" -> (kotanchekSR M.! algorithm) !! rowIndex",
            "  _ -> error \"unknown dataset\"",
            "",
            "lookupSympy :: String -> String -> Int -> SRTree Int Double",
            "lookupSympy dataset algorithm rowIndex = case dataset of",
            "  \"pagie\" -> (pagieSympy M.! algorithm) !! rowIndex",
            "  \"kotanchek\" -> (kotanchekSympy M.! algorithm) !! rowIndex",
            "  _ -> error \"unknown dataset\"",
            "",
            "emitExpr :: RowId -> String -> SRTree Int Double -> IO ()",
            "emitExpr (dataset, rawIndex, algorithm, algoRow) label expr = do",
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
            "        [ dataset",
            "        , rawIndex",
            "        , algorithm",
            "        , algoRow",
            "        , label",
            "        , show beforeNodes",
            "        , show beforeParams",
            "        , show afterNodes",
            "        , show afterParams",
            "        , show runtimeMs",
            "        , rendered",
            "        ]",
            "  putStrLn (intercalate \"\\t\" (map sanitize fields))",
            "",
            "emitCase :: RowCase -> IO ()",
            "emitCase (rowId@(dataset, _, _, _), rawAlgorithm, rowIndex) = do",
            "  let originalExpr = lookupOriginal dataset rawAlgorithm rowIndex",
            "      sympyExpr = lookupSympy dataset rawAlgorithm rowIndex",
            "  emitExpr rowId \"original\" originalExpr",
            "  emitExpr rowId \"sympy\" sympyExpr",
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
        (
            dataset,
            raw_index,
            algorithm,
            algo_row,
            label,
            before_nodes,
            before_params,
            after_nodes,
            after_params,
            runtime_ms,
            rendered,
        ) = line.split("\t", maxsplit=10)
        results.append(
            {
                "dataset": dataset,
                "raw_index": raw_index,
                "algorithm": algorithm,
                "algo_row": algo_row,
                "label": label,
                "status": "saturated",
                "before_nodes": before_nodes,
                "before_params": before_params,
                "after_nodes": after_nodes,
                "after_params": after_params,
                "runtime_ms": runtime_ms,
                "rendered_haskell": rendered,
                "rendered_python": _canonicalize(rendered),
            }
        )
    return results


def _archived_fallback_results(row: dict[str, str], *, reason: str) -> list[dict[str, str]]:
    return [
        {
            "dataset": row["dataset"],
            "raw_index": row["raw_index"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "label": "original",
            "status": reason,
            "before_nodes": row["orig_nodes"],
            "before_params": row["orig_params"],
            "after_nodes": row["simpl_nodes"],
            "after_params": row["simpl_params"],
            "runtime_ms": "na",
            "rendered_haskell": "",
            "rendered_python": "",
        },
        {
            "dataset": row["dataset"],
            "raw_index": row["raw_index"],
            "algorithm": row["algorithm"],
            "algo_row": row["algo_row"],
            "label": "sympy",
            "status": reason,
            "before_nodes": row["orig_nodes_sympy"],
            "before_params": row["orig_params_sympy"],
            "after_nodes": row["simpl_nodes_sympy"],
            "after_params": row["simpl_params_sympy"],
            "runtime_ms": "na",
            "rendered_haskell": "",
            "rendered_python": "",
        },
    ]


def _run_haskell_rows_serial(rows: list[dict[str, str]], *, batch_size: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        try:
            batch_results = _run_haskell_chunk(batch)
        except subprocess.CalledProcessError:
            if len(batch) == 1:
                row = batch[0]
                print(
                    f"live Haskell failed on {row['dataset']} {row['algorithm']}#{row['algo_row']}",
                    flush=True,
                )
                batch_results = _archived_fallback_results(row, reason="archived_fallback_stack_overflow")
                results.extend(batch_results)
                end = start + len(batch)
                print(f"[{end}/{total}] live Haskell rows complete", flush=True)
                continue
            print(
                f"live Haskell batch overflow; splitting {batch[0]['dataset']} {batch[0]['algorithm']}#{batch[0]['algo_row']} .. {batch[-1]['dataset']} {batch[-1]['algorithm']}#{batch[-1]['algo_row']}",
                flush=True,
            )
            mid = len(batch) // 2
            batch_results = _run_haskell_rows_serial(batch[:mid], batch_size=max(1, mid))
            batch_results.extend(_run_haskell_rows_serial(batch[mid:], batch_size=max(1, len(batch) - mid)))
        results.extend(batch_results)
        end = start + len(batch)
        print(f"[{end}/{total}] live Haskell rows complete", flush=True)
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
        task = progress.add_task("live Haskell rows", total=len(rows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_haskell_rows_serial, batch, batch_size=batch_size): len(batch) for batch in batches
            }
            for future in as_completed(futures):
                results.extend(future.result())
                progress.advance(task, futures[future])
    return results


def _write_live_rows(rows: list[dict[str, str]], live_results: list[dict[str, str]], output_path: Path) -> None:
    indexed = {
        (row["dataset"], row["raw_index"], row["algorithm"], row["algo_row"], row["label"]): row for row in live_results
    }
    output_rows: list[dict[str, str]] = []
    for row in rows:
        original = indexed[(row["dataset"], row["raw_index"], row["algorithm"], row["algo_row"], "original")]
        sympy = indexed[(row["dataset"], row["raw_index"], row["algorithm"], row["algo_row"], "sympy")]
        output_row = dict(row)
        output_row.update(
            {
                "orig_nodes": original["before_nodes"],
                "orig_params": original["before_params"],
                "simpl_nodes": original["after_nodes"],
                "simpl_params": original["after_params"],
                "orig_live_status": original["status"],
                "orig_nodes_sympy": sympy["before_nodes"],
                "orig_params_sympy": sympy["before_params"],
                "simpl_nodes_sympy": sympy["after_nodes"],
                "simpl_params_sympy": sympy["after_params"],
                "sympy_live_status": sympy["status"],
                "orig_runtime_ms": original["runtime_ms"],
                "sympy_runtime_ms": sympy["runtime_ms"],
                "orig_rendered_haskell": original["rendered_haskell"],
                "orig_rendered_python": original["rendered_python"],
                "sympy_rendered_haskell": sympy["rendered_haskell"],
                "sympy_rendered_python": sympy["rendered_python"],
                "baseline_source": (
                    "live_haskell"
                    if original["status"] == "saturated" and sympy["status"] == "saturated"
                    else "archived_fallback"
                ),
            }
        )
        output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LIVE_ROWS_PATH,
        help="Path to write the live Haskell corpus artifact.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to evaluate per temporary Haskell runner invocation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of Haskell batches to evaluate in parallel.",
    )
    args = parser.parse_args()

    rows = _load_rows()
    live_results = _run_haskell_rows(rows, batch_size=args.batch_size, workers=args.workers)
    _write_live_rows(rows, live_results, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
