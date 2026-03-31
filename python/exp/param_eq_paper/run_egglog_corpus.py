from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_SEC = 5.0

OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
HASKELL_ROWS_PATH = OUTPUT_DIR / "haskell_paper_rows.csv"
EGGLOG_ROWS_PATH = OUTPUT_DIR / "egglog_paper_rows.csv"

MODES = ("egglog-baseline", "egglog-height-guard")


def _load_rows() -> list[dict[str, str]]:
    with HASKELL_ROWS_PATH.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["is_paper_row"] == "1"]


def _run_one(source: str, mode: str) -> dict[str, str]:
    start = time.perf_counter()
    proc = None
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "egglog.exp.param_eq_hegg",
                "--mode",
                mode,
                f"--expr={source}",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
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
        message = proc.stderr.strip() or proc.stdout.strip() or "subprocess failure"
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
    payload = json.loads(proc.stdout)
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


def main() -> None:
    rows = _load_rows()
    output_rows: list[dict[str, str]] = []
    total = len(rows) * len(MODES)
    counter = 0
    for source_row in rows:
        for mode in MODES:
            counter += 1
            if counter == 1 or counter % 25 == 0:
                print(f"[{counter}/{total}] {source_row['dataset']} {source_row['algorithm']}#{source_row['algo_row']} {mode}", flush=True)
            original = _run_one(source_row["original_expr"], mode)
            sympy = _run_one(source_row["sympy_expr"], mode)
            output_rows.append(
                {
                    "dataset": source_row["dataset"],
                    "raw_index": source_row["raw_index"],
                    "algorithm_raw": source_row["algorithm_raw"],
                    "algorithm": source_row["algorithm"],
                    "algo_row": source_row["algo_row"],
                    "is_paper_row": source_row["is_paper_row"],
                    "drop_reason": source_row["drop_reason"],
                    "n_params": source_row["n_params"],
                    "n_rank": source_row["n_rank"],
                    "mode": mode,
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
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with EGGLOG_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
