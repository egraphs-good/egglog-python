"""Focused tests for the local param-eq trace and comparison harness."""

from __future__ import annotations

import pytest

from egglog import EGraph
from egglog.exp.param_eq.compare_trace import (
    REDUCED_TOY_CASE_ID,
    REDUCED_TOY_SOURCE,
    _compare_snapshots,
    _known_case_sources,
    compare_case_traces,
    trace_and_compare_case,
)
from egglog.exp.param_eq.pipeline import analysis_schedule, parse_expression
from egglog.exp.param_eq.trace_egglog import trace_egglog_case
from egglog.exp.param_eq.trace_haskell import haskell_trace_available, trace_haskell_case
from egglog.exp.param_eq.trace_tables import SnapshotTables, build_egglog_snapshot, read_snapshot


def test_snapshot_adapter_is_deterministic_and_round_trips_to_pandas(tmp_path) -> None:
    expr = parse_expression("x0 + 1")
    first = EGraph(expr)
    second = EGraph(expr)
    first.run(analysis_schedule.saturate())
    second.run(analysis_schedule.saturate())

    first_snapshot = build_egglog_snapshot(
        first,
        root=expr,
        metadata={"system": "egglog", "case_id": "unit", "step": "outer_1_pass_start"},
    )
    second_snapshot = build_egglog_snapshot(
        second,
        root=expr,
        metadata={"system": "egglog", "case_id": "unit", "step": "outer_1_pass_start"},
    )

    assert first_snapshot.to_jsonable() == second_snapshot.to_jsonable()
    path = tmp_path / "snapshot.json"
    first_snapshot.write_json(path)
    round_tripped = read_snapshot(path)
    assert round_tripped.to_jsonable() == first_snapshot.to_jsonable()
    frames = first_snapshot.to_pandas()
    assert {"functions", "rows", "classes", "nodes", "root"} <= set(frames)
    assert sum(len(frame.index) for frame in frames.values()) > 0


def test_snapshot_adapter_records_num_class_analysis_and_best_expr() -> None:
    expr = parse_expression("2 - 2")
    egraph = EGraph(expr)
    egraph.run(analysis_schedule.saturate())

    snapshot = build_egglog_snapshot(
        egraph,
        root=expr,
        metadata={"system": "egglog", "case_id": "unit", "step": "outer_1_pass_start"},
    )

    class_rows = snapshot.tables["classes"]
    assert class_rows
    assert any(row["analysis"] == {"kind": "some", "value": 0.0} for row in class_rows)
    assert all("best_expr" in row for row in class_rows)
    assert all("signature" in row for row in class_rows)


def test_snapshot_adapter_prefers_concrete_optional_analysis_over_join_wrappers() -> None:
    expr = parse_expression("((0 * x0) * ((x0 * x0) + 1))")
    egraph = EGraph()
    egraph.register(expr)
    scheduler = egraph._add_backoff_scheduler(
        match_limit=2500,
        ban_length=30,
        egg_like=True,
    )
    egraph.run(analysis_schedule.saturate())
    from egglog.exp.param_eq.pipeline import literal_rewrite_ruleset

    egraph._run_ruleset_with_scheduler(literal_rewrite_ruleset, scheduler)
    egraph.run(analysis_schedule.saturate())

    snapshot = build_egglog_snapshot(
        egraph,
        root=expr,
        metadata={"system": "egglog", "case_id": "unit", "step": "outer_1_inner_1_after_analysis"},
    )

    assert all(
        isinstance(row["analysis"], dict) and row["analysis"].get("kind") != "raw"
        for row in snapshot.tables["classes"]
    )


def test_egglog_step_tracer_emits_expected_checkpoints(tmp_path) -> None:
    result = trace_egglog_case(case_id="unit_trace_case", source="x0 + 1", output_root=tmp_path)

    step_names = [path.stem for path in result.step_paths]
    assert step_names[0] == "outer_1_pass_start"
    assert any(name.startswith("outer_1_inner_1_after_rewrite") for name in step_names)
    assert any(name.startswith("outer_1_inner_1_after_analysis") for name in step_names)
    assert "outer_1_extract" in step_names
    assert "final_simplify_e" in step_names


@pytest.mark.skipif(not haskell_trace_available(), reason="requires local param-eq-haskell checkout and stack")
def test_haskell_trace_driver_emits_expected_checkpoints(tmp_path) -> None:
    result = trace_haskell_case(case_id="unit_trace_case", source="x0 + 1", output_root=tmp_path)

    step_names = [path.stem for path in result.step_paths]
    assert "outer_1_pass_start" in step_names
    assert any(name.startswith("outer_1_inner_1_after_rebuild") for name in step_names)
    assert "outer_1_extract" in step_names
    assert "final_simplify_e" in step_names


def test_comparator_reports_no_difference_for_identical_snapshots() -> None:
    snapshot = SnapshotTables(
        metadata={
            "system": "egglog",
            "case_id": "unit",
            "step": "outer_1_extract",
            "root_analysis": {"kind": "none", "value": None},
            "class_count": 1,
            "node_count": 1,
            "memo_size": 0,
        },
        tables={
            "functions": [],
            "rows": [],
            "classes": [
                {
                    "class_id": "c0",
                    "type": "egglog.exp.param_eq.pipeline.Num",
                    "node_count": 1,
                    "analysis": {"kind": "none", "value": None},
                    "best_expr": "1.0",
                    "best_cost": 1,
                    "signature": "none|best:1.0|Const(1.0)",
                }
            ],
            "nodes": [
                {"node_id": "n0", "class_id": "c0", "op": "Num", "children": ["p0"], "cost": 1.0, "subsumed": False},
                {"node_id": "p0", "class_id": "f64-0", "op": "1.0", "children": [], "cost": 1.0, "subsumed": False},
            ],
            "root": [{"extracted_expr": "1.0", "root_analysis": {"kind": "none", "value": None}}],
        },
    )
    identical = SnapshotTables(
        metadata={
            "system": "egglog",
            "case_id": "unit",
            "step": "outer_1_extract",
            "root_analysis": {"kind": "none", "value": None},
            "class_count": 1,
            "node_count": 1,
            "memo_size": 0,
        },
        tables=snapshot.tables,
    )

    assert _compare_snapshots(snapshot, identical) is None


@pytest.mark.skipif(not haskell_trace_available(), reason="requires local param-eq-haskell checkout and stack")
def test_reduced_toy_comparison_reports_a_first_differing_step(tmp_path) -> None:
    summary = trace_and_compare_case(
        case_id=REDUCED_TOY_CASE_ID,
        source=REDUCED_TOY_SOURCE,
        trace_root=tmp_path,
    )

    assert summary.first_differing_step is not None
    assert summary.first_differing_step.startswith("outer_1_")
    assert summary.first_differing_aspect is not None
    assert summary.likely_phase in {"rewrite", "analysis"}


@pytest.mark.skipif(not haskell_trace_available(), reason="requires local param-eq-haskell checkout and stack")
def test_reduced_toy_comparison_can_ignore_extraction_differences(tmp_path) -> None:
    trace_and_compare_case(
        case_id=REDUCED_TOY_CASE_ID,
        source=REDUCED_TOY_SOURCE,
        trace_root=tmp_path,
    )
    summary = compare_case_traces(
        REDUCED_TOY_CASE_ID,
        trace_root=tmp_path,
        ignored_aspects=frozenset({"root_extracted_expr", "class_best_expr"}),
    )

    assert summary.first_differing_step is not None
    assert summary.first_differing_aspect not in {"root_extracted_expr", "class_best_expr"}
    assert summary.likely_phase in {"rewrite", "analysis"}

