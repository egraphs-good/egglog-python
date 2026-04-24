# # Param-Eq Container Candidate Probe
#
# This notebook-style script is meant to be converted with Jupytext or run
# directly as Python, following the same pattern as `replication.py`.
#
# It loads the current Egglog rank-miss artifacts, chooses rows that are likely
# to be useful container candidates, and can rerun one selected row with the
# current binary pipeline while reporting:
# - top rewrite-rule matches,
# - rule search/apply timing,
# - per-pass e-graph size,
# - extraction cost and rendered output.
#
# Containers are not run by default. The current public container path in this
# experiment is diagnostic only; set `PARAM_EQ_RUN_TRUE_CONTAINER_LOWERING=1` to
# try it and report its failure/success separately.
#
# Offline commands from the repository root:
# - `uv run --group dev python python/egglog/exp/param_eq/container_candidate_probe.py`
# - `PARAM_EQ_RUN_FRESH_PIPELINE=0 uv run --group dev python python/egglog/exp/param_eq/container_candidate_probe.py`
# - `uv run --with jupytext jupytext --to ipynb python/egglog/exp/param_eq/container_candidate_probe.py`

# +
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from egglog import EGraph, run
from egglog.exp.param_eq.domain import Num, ParamCost, param_cost_model, parse_expression, render_num
from egglog.exp.param_eq.egglog_results import load_egglog_results
from egglog.exp.param_eq.paths import ARTIFACT_DIR
from egglog.exp.param_eq.pipeline import (
    HASKELL_INNER_ITERATION_LIMIT,
    MAX_PASSES,
    _graph_size,
    _new_rewrite_scheduler,
    binary_analysis_schedule,
    binary_rewrite_ruleset,
    parse_expression_container,
    run_paper_pipeline_container,
)

KEY_COLUMNS = ["dataset", "raw_index", "algorithm_raw", "algo_row", "input_kind"]
ARTIFACT_ROOT = Path(os.environ.get("EGGLOG_PARAM_EQ_ARTIFACT_DIR", str(ARTIFACT_DIR)))
DATA_FILE = Path(os.environ.get("PARAM_EQ_CANDIDATE_DATA_FILE", str(ARTIFACT_ROOT / "egglog_rank_misses.csv")))
REVIEW_FILE = Path(
    os.environ.get(
        "PARAM_EQ_REVIEW_FILE",
        str(ARTIFACT_ROOT / "egglog_rank_miss_agent_review_current.csv"),
    )
)

RUN_FRESH_PIPELINE = os.environ.get("PARAM_EQ_RUN_FRESH_PIPELINE", "1") == "1"
RUN_TRUE_CONTAINER_LOWERING = os.environ.get("PARAM_EQ_RUN_TRUE_CONTAINER_LOWERING", "0") == "1"
SELECTED_DATASET = os.environ.get("PARAM_EQ_SELECTED_DATASET", "kotanchek")
SELECTED_ALGORITHM_RAW = os.environ.get("PARAM_EQ_SELECTED_ALGORITHM_RAW", "SRjl")
SELECTED_RAW_INDEX = int(os.environ.get("PARAM_EQ_SELECTED_RAW_INDEX", "203"))
SELECTED_ALGO_ROW = int(os.environ.get("PARAM_EQ_SELECTED_ALGO_ROW", "24"))
SELECTED_INPUT_KIND = os.environ.get("PARAM_EQ_SELECTED_INPUT_KIND", "sympy")
TOP_N = int(os.environ.get("PARAM_EQ_TOP_N", "12"))

RULE_NAME_REWRITES = {
    "egglog_exp_param_eq_domain_Num___init__": "Num",
    "egglog_exp_param_eq_domain_Num___add__": "+",
    "egglog_exp_param_eq_domain_Num___sub__": "-",
    "egglog_exp_param_eq_domain_Num___mul__": "*",
    "egglog_exp_param_eq_domain_Num___truediv__": "/",
    "egglog_exp_param_eq_domain_Num___pow__": "**",
    "egglog_exp_param_eq_domain_exp": "exp",
    "egglog_exp_param_eq_domain_log": "log",
    "egglog_exp_param_eq_domain_abs": "abs",
    "egglog.exp.param_eq.pipeline.": "",
}


@dataclass(frozen=True)
class ProbeResult:
    summary: dict[str, Any]
    pass_trace: pd.DataFrame
    rule_stats: pd.DataFrame
    function_sizes: pd.DataFrame


def _require_file(path: Path) -> None:
    if not path.exists():
        msg = f"Missing required artifact: {path}"
        raise FileNotFoundError(msg)


def _rank_gap(frame: pd.DataFrame) -> pd.Series:
    return frame["after_params"] - frame["n_rank"]


def _sort_smallest_first(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "source_orig_parsed_n_params",
        "source_before_nodes",
        "dataset",
        "algorithm_raw",
        "raw_index",
        "input_kind",
    ]
    present = [column for column in sort_columns if column in frame.columns]
    return frame.sort_values(present).reset_index(drop=True)


def _load_rank_misses() -> pd.DataFrame:
    _require_file(DATA_FILE)
    frame = pd.read_csv(DATA_FILE, na_values=["na"], keep_default_na=True)
    if "rank_gap" not in frame.columns:
        frame["rank_gap"] = _rank_gap(frame)
    return _sort_smallest_first(frame)


def _load_review() -> pd.DataFrame:
    if not REVIEW_FILE.exists():
        return pd.DataFrame(columns=KEY_COLUMNS)
    return pd.read_csv(REVIEW_FILE, na_values=["na"], keep_default_na=True)


def _load_joined_results() -> pd.DataFrame:
    frame = load_egglog_results(variant="baseline")
    if "rank_gap" not in frame.columns:
        frame["rank_gap"] = _rank_gap(frame)
    return frame


def _merge_review(frame: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    if review.empty:
        return frame.copy()
    review_columns = [
        *KEY_COLUMNS,
        "conclusion_kind",
        "new_rule_family",
        "proposed_rule",
        "proof_summary",
        "remaining_gap_notes",
    ]
    review_columns = [column for column in review_columns if column in review.columns]
    return frame.merge(review.loc[:, review_columns], on=KEY_COLUMNS, how="left")


def _candidate_key_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["dataset"] == SELECTED_DATASET)
        & (frame["algorithm_raw"] == SELECTED_ALGORITHM_RAW)
        & (frame["raw_index"] == SELECTED_RAW_INDEX)
        & (frame["algo_row"] == SELECTED_ALGO_ROW)
        & (frame["input_kind"] == SELECTED_INPUT_KIND)
    )


def _select_candidate(frame: pd.DataFrame) -> pd.Series:
    selected = frame[_candidate_key_mask(frame)]
    if not selected.empty:
        return selected.iloc[0]
    slow = frame[(frame["status"] == "saturated") & (frame["rank_gap"] > 0)].sort_values(
        ["runtime_ms", "egraph_total_size"],
        ascending=False,
    )
    if slow.empty:
        msg = "No saturated rank-miss row is available to probe."
        raise ValueError(msg)
    return slow.iloc[0]


def _show_table(title: str, frame: pd.DataFrame, columns: list[str], n: int = TOP_N) -> None:
    columns = [column for column in columns if column in frame.columns]
    print(f"\n## {title}")
    if frame.empty:
        print("(empty)")
        return
    print(frame.loc[:, columns].head(n).to_string(index=False, max_colwidth=100))


def _rule_ruleset(rule: str) -> str:
    match = re.search(r":ruleset ([^)]+)\)", rule)
    if match is None:
        return "unknown"
    return match.group(1).removeprefix("egglog.exp.param_eq.pipeline.")


def _compact_rule(rule: str, *, max_len: int = 170) -> str:
    text = rule
    for old, new in RULE_NAME_REWRITES.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _duration_ms(value: Any) -> float:
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds() * 1000.0)
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _report_rule_rows(report: Any, pass_index: int, iteration: int) -> list[dict[str, Any]]:
    match_counts = getattr(report, "num_matches_per_rule", {}) or {}
    timings = getattr(report, "search_and_apply_time_per_rule", {}) or {}
    rules = set(match_counts) | set(timings)
    return [
        {
            "pass": pass_index,
            "iteration": iteration,
            "ruleset": _rule_ruleset(rule_name),
            "matches": int(match_counts.get(rule_name, 0) or 0),
            "search_apply_ms": _duration_ms(timings.get(rule_name, 0.0)),
            "rule": _compact_rule(rule_name),
        }
        for rule_name in rules
    ]


def _function_size_rows(egraph: EGraph, pass_index: int, iteration: int) -> list[dict[str, Any]]:
    rows = []
    for function_name, size in egraph.all_function_sizes():
        rows.append(
            {
                "pass": pass_index,
                "iteration": iteration,
                "function": str(function_name),
                "size": int(size),
            }
        )
    return rows


def _run_instrumented(source: str) -> ProbeResult:
    initial = parse_expression(source)
    _, before_cost = EGraph().extract(initial, include_cost=True, cost_model=param_cost_model)

    current = initial
    current_rendered = render_num(current)
    trace_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    function_size_rows: list[dict[str, Any]] = []

    total_size = 0
    total_runtime_sec = 0.0
    last_cost = ParamCost()
    pass_index = 0
    converged = False

    for pass_index in range(1, MAX_PASSES + 1):
        egraph = EGraph()
        egraph.register(current, Num(0.0))

        pass_start = time.perf_counter()
        previous_size = _graph_size(egraph)
        egraph.run(binary_analysis_schedule)
        scheduler = _new_rewrite_scheduler()
        iteration = 0

        for iteration in range(1, HASKELL_INNER_ITERATION_LIMIT + 1):
            report = egraph.run(run(binary_rewrite_ruleset, scheduler=scheduler))
            egraph.run(binary_analysis_schedule)
            current_size = _graph_size(egraph)
            total_size = current_size
            rule_rows.extend(_report_rule_rows(report, pass_index, iteration))
            function_size_rows.extend(_function_size_rows(egraph, pass_index, iteration))
            trace_rows.append(
                {
                    "pass": pass_index,
                    "iteration": iteration,
                    "egraph_total_size": current_size,
                    "size_delta": current_size - previous_size,
                    "updated": bool(getattr(report, "updated", False)),
                    "can_stop": bool(getattr(report, "can_stop", False)),
                }
            )
            if current_size == previous_size:
                break
            previous_size = current_size

        extracted, last_cost = egraph.extract(current, include_cost=True, cost_model=param_cost_model)
        elapsed = time.perf_counter() - pass_start
        total_runtime_sec += elapsed
        extracted_rendered = render_num(extracted)
        trace_rows.append(
            {
                "pass": pass_index,
                "iteration": iteration,
                "egraph_total_size": total_size,
                "size_delta": 0,
                "updated": False,
                "can_stop": True,
                "pass_runtime_ms": elapsed * 1000.0,
                "extracted_nodes": last_cost.node_count,
                "extracted_params": last_cost.floats,
                "extracted": extracted_rendered,
            }
        )
        converged = extracted_rendered == current_rendered
        current = extracted
        current_rendered = extracted_rendered
        if converged:
            break

    summary = {
        "passes": pass_index,
        "runtime_ms": total_runtime_sec * 1000.0,
        "before_nodes": before_cost.node_count,
        "before_params": before_cost.floats,
        "after_nodes": last_cost.node_count,
        "after_params": last_cost.floats,
        "egraph_total_size": total_size,
        "converged": converged,
        "rendered": current_rendered,
    }
    return ProbeResult(
        summary=summary,
        pass_trace=pd.DataFrame(trace_rows),
        rule_stats=pd.DataFrame(rule_rows),
        function_sizes=pd.DataFrame(function_size_rows),
    )


def _summarize_rule_stats(rule_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rule_stats.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = (
        rule_stats.groupby(["ruleset", "rule"], dropna=False)
        .agg(
            matches=("matches", "sum"),
            search_apply_ms=("search_apply_ms", "sum"),
            active_iterations=("matches", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    top_matches = grouped.sort_values(["matches", "search_apply_ms"], ascending=False).reset_index(drop=True)
    top_time = grouped.sort_values(["search_apply_ms", "matches"], ascending=False).reset_index(drop=True)
    return top_matches, top_time


def _summarize_function_sizes(function_sizes: pd.DataFrame) -> pd.DataFrame:
    if function_sizes.empty:
        return function_sizes
    return (
        function_sizes.groupby(["pass", "iteration", "function"], dropna=False)
        .agg(size=("size", "max"))
        .reset_index()
        .sort_values(["size", "pass", "iteration"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _run_true_container_lowering(source: str) -> None:
    print("\n## True Container Lowering Diagnostic")
    if not RUN_TRUE_CONTAINER_LOWERING:
        print("Skipped. Set PARAM_EQ_RUN_TRUE_CONTAINER_LOWERING=1 to try this path.")
        return

    start = time.perf_counter()
    try:
        report = run_paper_pipeline_container(parse_expression_container(source))
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"failed_after_ms={elapsed * 1000.0:.3f}")
        print(f"exception={type(exc).__name__}: {exc}")
        return
    elapsed = time.perf_counter() - start
    print(
        pd.DataFrame(
            [
                {
                    "runtime_ms": elapsed * 1000.0,
                    "passes": report.passes,
                    "before_nodes": report.before_nodes,
                    "before_params": report.before_params,
                    "after_nodes": report.extracted_nodes,
                    "after_params": report.extracted_params,
                    "egraph_total_size": report.total_size,
                    "rendered": report.extracted,
                }
            ]
        ).to_string(index=False, max_colwidth=100)
    )


def main() -> None:
    rank_misses = _load_rank_misses()
    review = _load_review()
    joined = _merge_review(_load_joined_results(), review)
    missed = joined[(joined["status"] == "saturated") & joined["n_rank"].notna() & (joined["rank_gap"] > 0)].copy()

    print(f"artifact_root={ARTIFACT_ROOT}")
    print(f"rank_miss_file={DATA_FILE}")
    print(f"current_rank_misses_csv_rows={len(rank_misses)}")
    print(f"joined_saturated_rank_misses={len(missed)}")
    print(f"max_passes={MAX_PASSES}")
    print(f"haskell_inner_iteration_limit={HASKELL_INNER_ITERATION_LIMIT}")

    status_counts = joined["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="rows")
    _show_table("Baseline Status Counts", status_counts, ["status", "rows"], n=20)

    reviewed_misses = _merge_review(rank_misses, review)
    if "new_rule_family" in reviewed_misses.columns:
        family_counts = (
            reviewed_misses["new_rule_family"]
            .fillna("unreviewed")
            .value_counts()
            .rename_axis("new_rule_family")
            .reset_index(name="rows")
        )
        _show_table("Current Rank Misses By Review Family", family_counts, ["new_rule_family", "rows"], n=30)

    _show_table(
        "Slow Saturated Rank Misses",
        missed.sort_values(["runtime_ms", "egraph_total_size"], ascending=False),
        [
            "dataset",
            "algorithm_raw",
            "raw_index",
            "algo_row",
            "input_kind",
            "n_rank",
            "after_params",
            "rank_gap",
            "runtime_ms",
            "egraph_total_size",
            "source_orig_parsed_n_params",
            "source_before_nodes",
            "new_rule_family",
        ],
    )
    _show_table(
        "Largest Saturated Rank Misses",
        missed.sort_values(["egraph_total_size", "runtime_ms"], ascending=False),
        [
            "dataset",
            "algorithm_raw",
            "raw_index",
            "algo_row",
            "input_kind",
            "n_rank",
            "after_params",
            "rank_gap",
            "runtime_ms",
            "egraph_total_size",
            "source_orig_parsed_n_params",
            "source_before_nodes",
            "new_rule_family",
        ],
    )

    candidate = _select_candidate(missed)
    candidate_frame = pd.DataFrame([candidate])
    _show_table(
        "Selected Probe Row",
        candidate_frame,
        [
            "dataset",
            "algorithm_raw",
            "raw_index",
            "algo_row",
            "input_kind",
            "n_rank",
            "after_params",
            "rank_gap",
            "runtime_ms",
            "egraph_total_size",
            "new_rule_family",
            "proposed_rule",
        ],
        n=1,
    )
    print("\n## Selected Source Expression")
    print(candidate["source_orig_parsed_expr"])
    print("\n## Current Artifact Rendered Output")
    print(candidate["rendered"])

    if RUN_FRESH_PIPELINE:
        print("\n## Fresh Binary Pipeline Probe")
        probe = _run_instrumented(str(candidate["source_orig_parsed_expr"]))
        print(pd.DataFrame([probe.summary]).to_string(index=False, max_colwidth=120))

        trace_columns = [
            "pass",
            "iteration",
            "egraph_total_size",
            "size_delta",
            "updated",
            "can_stop",
            "pass_runtime_ms",
            "extracted_nodes",
            "extracted_params",
            "extracted",
        ]
        _show_table("Per-Pass / Per-Iteration Trace", probe.pass_trace, trace_columns, n=80)

        top_matches, top_time = _summarize_rule_stats(probe.rule_stats)
        _show_table(
            "Top Rules By Match Count",
            top_matches,
            ["ruleset", "matches", "active_iterations", "search_apply_ms", "rule"],
            n=25,
        )
        _show_table(
            "Top Rules By Search/Apply Time",
            top_time,
            ["ruleset", "search_apply_ms", "matches", "active_iterations", "rule"],
            n=25,
        )

        function_sizes = _summarize_function_sizes(probe.function_sizes)
        _show_table(
            "Largest Function Tables During Probe",
            function_sizes,
            ["pass", "iteration", "function", "size"],
            n=40,
        )
    else:
        print("\n## Fresh Binary Pipeline Probe")
        print("Skipped. Set PARAM_EQ_RUN_FRESH_PIPELINE=1 to rerun the selected row.")

    _run_true_container_lowering(str(candidate["source_orig_parsed_expr"]))


if __name__ == "__main__":
    main()
