from __future__ import annotations

import ast
import csv
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from egglog.exp.srtree_eqsat import load_example_hl_rows, parse_hl_expr, run_baseline_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
SRTREE_EQSAT_ROOT = REPO_ROOT.parent / "srtree-eqsat"
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "corpus_baseline_rows.csv"
SUMMARY_PATH = OUTPUT_DIR / "corpus_baseline_summary.md"
STACK_EXE = shutil.which("stack") or "stack"


@dataclass(frozen=True)
class HaskellRow:
    row: int
    status: str
    runtime_sec: float
    before_params: int | None
    after_params: int | None
    before_nodes: int | None
    after_nodes: int | None
    simplified_output: str | None
    error: str | None = None


@dataclass(frozen=True)
class EgglogRow:
    row: int
    status: str
    numeric_status: str
    runtime_sec: float
    before_params: int
    after_params: int
    optimal_params: int | None
    gap_to_optimal: int | None
    total_size: int
    node_count: int
    eclass_count: int
    rendered_output: str
    input_expr: str


def _name_to_haskell(name: str) -> str:
    if name == "alpha":
        return "alpha_"
    if name == "beta":
        return "beta_"
    if name == "theta":
        return "theta_"
    return name


def _const_to_haskell(value: int | float) -> str:  # noqa: PYI041
    if isinstance(value, bool):
        msg = "unexpected bool literal"
        raise TypeError(msg)
    if isinstance(value, int):
        return str(value)
    return repr(value)


def _ast_to_haskell(node: ast.AST) -> str:  # noqa: C901, PLR0912
    if isinstance(node, ast.Expression):
        return _ast_to_haskell(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return _const_to_haskell(node.value)
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        return _name_to_haskell(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{_ast_to_haskell(node.operand)})"
    if isinstance(node, ast.BinOp):
        lhs = _ast_to_haskell(node.left)
        rhs = _ast_to_haskell(node.right)
        if isinstance(node.op, ast.Add):
            op = "+"
        elif isinstance(node.op, ast.Sub):
            op = "-"
        elif isinstance(node.op, ast.Mult):
            op = "*"
        elif isinstance(node.op, ast.Div):
            op = "/"
        elif isinstance(node.op, ast.Pow):
            op = "**"
        else:
            msg = f"unsupported binop: {ast.dump(node.op)}"
            raise TypeError(msg)
        return f"({lhs} {op} {rhs})"
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            msg = f"unsupported call target: {ast.dump(node.func)}"
            raise TypeError(msg)
        func = _name_to_haskell(node.func.id)
        args = ", ".join(_ast_to_haskell(arg) for arg in node.args)
        return f"{func}({args})"
    raise ValueError(f"unsupported AST: {ast.dump(node)}")


def _to_haskell_expr(source: str) -> str:
    return _ast_to_haskell(ast.parse(source, mode="eval"))


def _haskell_driver_source(source: str, row: int) -> str:
    return f"""{{-# LANGUAGE ImportQualifiedPost #-}}
module Main where

import Data.SRTree
import Data.SRTree.EqSat (simplifyEqSat)
import Data.SRTree.Print (showPython)
import System.CPUTime (getCPUTime)

main :: IO ()
main = runRow ({row}, {_to_haskell_expr(source)})

runRow :: (Int, Fix SRTree) -> IO ()
runRow (row, tree) = do
  start <- getCPUTime
  let simplified = simplifyEqSat tree
      beforeParams = countParams . fst $ floatConstsToParam tree
      afterParams = countParams . fst $ floatConstsToParam simplified
      beforeNodes = countNodes tree
      afterNodes = countNodes simplified
      rendered = showPython simplified
  rendered `seq` pure ()
  end <- getCPUTime
  let elapsed = fromIntegral (end - start) / 1e12 :: Double
  putStrLn $ show row <> "\\t" <> show beforeParams <> "\\t" <> show afterParams <> "\\t" <> show beforeNodes <> "\\t" <> show afterNodes <> "\\t" <> show elapsed <> "\\t" <> rendered

alpha_, beta_, theta_ :: Fix SRTree
alpha_ = var 0
beta_ = var 1
theta_ = var 2

sqr :: Fix SRTree -> Fix SRTree
sqr x = x ** 2

cube :: Fix SRTree -> Fix SRTree
cube x = x ** 3

cbrt :: Fix SRTree -> Fix SRTree
cbrt x = x ** (1 / 3)
"""


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "unknown failure"


def run_haskell_row(row: int, source: str) -> HaskellRow:
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".hs", delete=False) as handle:
        handle.write(_haskell_driver_source(source, row))
        temp_path = Path(handle.name)
    proc = subprocess.run(
        [STACK_EXE, "exec", "--", "runghc", str(temp_path), "+RTS", "-K256M", "-RTS"],
        cwd=SRTREE_EQSAT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        error = _first_nonempty_line(proc.stderr) if proc.stderr else _first_nonempty_line(proc.stdout)
        return HaskellRow(
            row=row,
            status="failed",
            runtime_sec=elapsed,
            before_params=None,
            after_params=None,
            before_nodes=None,
            after_nodes=None,
            simplified_output=None,
            error=error,
        )
    row_s, before_s, after_s, before_nodes_s, after_nodes_s, runtime_s, rendered_s = proc.stdout.strip().split("\t", 6)
    return HaskellRow(
        row=int(row_s),
        status="ok",
        runtime_sec=float(runtime_s),
        before_params=int(before_s),
        after_params=int(after_s),
        before_nodes=int(before_nodes_s),
        after_nodes=int(after_nodes_s),
        simplified_output=rendered_s,
    )


def run_haskell_all(rows: Sequence[str]) -> dict[int, HaskellRow]:
    results: dict[int, HaskellRow] = {}
    for row, source in enumerate(rows, start=1):
        if row == 1 or row % 25 == 0:
            print(f"[haskell] row {row}/{len(rows)}", file=sys.stderr, flush=True)
        results[row] = run_haskell_row(row, source)
    return results


def run_egglog_all(rows: Sequence[str]) -> dict[int, EgglogRow]:
    results: dict[int, EgglogRow] = {}
    for row, source in enumerate(rows, start=1):
        if row == 1 or row % 50 == 0:
            print(f"[egglog] row {row}/{len(rows)}", file=sys.stderr, flush=True)
        num = parse_hl_expr(source)
        report = run_baseline_pipeline(num, node_cutoff=50_000, iteration_limit=12)
        results[row] = EgglogRow(
            row=row,
            status=report.stop_reason,
            numeric_status=report.numeric_status,
            runtime_sec=report.total_sec,
            before_params=report.metric_report.before_parameter_count,
            after_params=report.metric_report.after_parameter_count,
            optimal_params=report.metric_report.optimal_parameter_count,
            gap_to_optimal=report.metric_report.gap_to_optimal,
            total_size=report.total_size,
            node_count=report.node_count,
            eclass_count=report.eclass_count,
            rendered_output=report.rendered,
            input_expr=source,
        )
    return results


def _summary_stats(values: Sequence[float]) -> tuple[float, float, float, float]:
    return (
        sum(values),
        statistics.mean(values),
        statistics.median(values),
        max(values),
    )


def _as_text(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(haskell: Mapping[int, HaskellRow], egglog: Mapping[int, EgglogRow]) -> list[dict[str, str]]:
    fieldnames = [
        "row",
        "egglog_status",
        "egglog_numeric_status",
        "haskell_status",
        "egglog_runtime_s",
        "haskell_runtime_s",
        "egglog_before_params",
        "egglog_after_params",
        "haskell_before_params",
        "haskell_after_params",
        "optimal_params",
        "egglog_gap_to_optimal",
        "haskell_gap_to_optimal",
        "egglog_total_size",
        "egglog_node_count",
        "egglog_eclass_count",
        "haskell_after_nodes",
        "input_expr",
        "egglog_output",
        "haskell_output",
    ]
    rows: list[dict[str, str]] = []
    for row in sorted(egglog):
        egg = egglog[row]
        hs = haskell[row]
        optimal = egg.optimal_params
        haskell_gap = hs.after_params - optimal if hs.after_params is not None and optimal is not None else None
        rows.append({
            "row": str(row),
            "egglog_status": egg.status,
            "egglog_numeric_status": egg.numeric_status,
            "haskell_status": hs.status,
            "egglog_runtime_s": _as_text(egg.runtime_sec),
            "haskell_runtime_s": _as_text(hs.runtime_sec),
            "egglog_before_params": str(egg.before_params),
            "egglog_after_params": str(egg.after_params),
            "haskell_before_params": _as_text(hs.before_params),
            "haskell_after_params": _as_text(hs.after_params),
            "optimal_params": _as_text(optimal),
            "egglog_gap_to_optimal": _as_text(egg.gap_to_optimal),
            "haskell_gap_to_optimal": _as_text(haskell_gap),
            "egglog_total_size": str(egg.total_size),
            "egglog_node_count": str(egg.node_count),
            "egglog_eclass_count": str(egg.eclass_count),
            "haskell_after_nodes": _as_text(hs.after_nodes),
            "input_expr": egg.input_expr,
            "egglog_output": egg.rendered_output,
            "haskell_output": hs.simplified_output or "na",
        })
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_summary(haskell: Mapping[int, HaskellRow], egglog: Mapping[int, EgglogRow]) -> str:
    comparable_rows = [row for row, hs in haskell.items() if hs.status == "ok"]
    mismatches = [
        row
        for row in comparable_rows
        if egglog[row].before_params != cast("int", haskell[row].before_params)
        or egglog[row].after_params != cast("int", haskell[row].after_params)
    ]
    egg_times = [row.runtime_sec for row in egglog.values()]
    haskell_times = [haskell[row].runtime_sec for row in comparable_rows]
    egg_total, egg_mean, egg_median, egg_max = _summary_stats(egg_times)
    hs_total, hs_mean, hs_median, hs_max = _summary_stats(haskell_times)
    non_saturated = [row for row, egg in egglog.items() if egg.status != "saturated"]
    domain_limited = [row for row, egg in egglog.items() if egg.numeric_status != "ok"]
    haskell_failures = [row for row, hs in haskell.items() if hs.status != "ok"]

    lines = [
        "# Srtree-EqSat Baseline Corpus Summary",
        "",
        f"- Total rows: {len(egglog)}",
        f"- Comparable Haskell rows: {len(comparable_rows)}",
        f"- Haskell failures: {len(haskell_failures)} ({', '.join(str(row) for row in haskell_failures) or 'none'})",
        f"- Egglog non-saturated rows: {len(non_saturated)} ({', '.join(str(row) for row in non_saturated) or 'none'})",
        f"- Egglog domain-limited numeric rows: {len(domain_limited)} ({', '.join(str(row) for row in domain_limited) or 'none'})",
        f"- Parameter mismatches on comparable rows: {len(mismatches)}",
        "",
        "## Timing",
        "",
        "| impl | total_s | mean_s | median_s | max_s |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| egglog | {egg_total:.6f} | {egg_mean:.6f} | {egg_median:.6f} | {egg_max:.6f} |",
        f"| haskell | {hs_total:.6f} | {hs_mean:.6f} | {hs_median:.6f} | {hs_max:.6f} |",
        "",
        "## Mismatch Rows",
        "",
        ",".join(str(row) for row in mismatches) or "none",
        "",
    ]
    summary = "\n".join(lines)
    SUMMARY_PATH.write_text(summary)
    return summary


def main() -> None:
    rows = list(load_example_hl_rows())
    haskell = run_haskell_all(rows)
    egglog = run_egglog_all(rows)
    csv_rows = _write_csv(haskell, egglog)
    summary = _write_summary(haskell, egglog)

    mismatches = [
        row["row"]
        for row in csv_rows
        if row["haskell_status"] == "ok"
        and (
            row["egglog_before_params"] != row["haskell_before_params"]
            or row["egglog_after_params"] != row["haskell_after_params"]
        )
    ]
    print(f"csv_path\t{CSV_PATH}")
    print(f"summary_path\t{SUMMARY_PATH}")
    print(f"total_rows\t{len(csv_rows)}")
    print(f"mismatches\t{len(mismatches)}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
