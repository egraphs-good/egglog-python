"""Stepwise Egglog tracing for the retained param-eq Haskell-literal schedule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from egglog import EGraph
from egglog.exp.param_eq.paths import PARAM_EQ_DIR
from egglog.exp.param_eq.pipeline import (
    BACKOFF_BAN_LENGTH,
    BACKOFF_MATCH_LIMIT,
    HASKELL_INNER_ITERATION_LIMIT,
    MAX_PASSES,
    _serialized_counts,
    analysis_schedule,
    literal_rewrite_ruleset,
    parse_expression,
    render_num,
)
from egglog.exp.param_eq.trace_tables import SnapshotTables, build_egglog_snapshot

TRACE_ROOT = PARAM_EQ_DIR / "trace"


@dataclass(frozen=True)
class TraceResult:
    """Filesystem outputs for one traced system and case."""

    system: str
    case_id: str
    output_dir: Path
    step_paths: list[Path]
    final_rendered: str


def trace_egglog_case(
    *,
    case_id: str,
    source: str,
    output_root: Path = TRACE_ROOT,
) -> TraceResult:
    """Trace the literal Haskell-style Egglog schedule step by step."""
    current = parse_expression(source)
    output_dir = output_root / case_id / "egglog"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_paths: list[Path] = []
    for outer_pass in range(1, MAX_PASSES + 1):
        egraph = EGraph()
        egraph.register(current)
        scheduler = egraph._add_backoff_scheduler(
            match_limit=BACKOFF_MATCH_LIMIT,
            ban_length=BACKOFF_BAN_LENGTH,
            egg_like=True,
            haskell_backoff=True,
        )
        start_path = output_dir / f"outer_{outer_pass}_pass_start.json"
        build_egglog_snapshot(
            egraph,
            root=current,
            metadata={
                "system": "egglog",
                "case_id": case_id,
                "step": f"outer_{outer_pass}_pass_start",
                "outer_pass": outer_pass,
                "phase": "start",
                "source": source,
                "pass_input_expr": render_num(current),
            },
        ).write_json(start_path)
        step_paths.append(start_path)

        previous_counts = _serialized_counts(egraph)
        for inner_iteration in range(1, HASKELL_INNER_ITERATION_LIMIT + 1):
            rewrite_report = egraph._run_ruleset_with_scheduler(literal_rewrite_ruleset, scheduler)
            rewrite_path = output_dir / f"outer_{outer_pass}_inner_{inner_iteration}_after_rewrite.json"
            build_egglog_snapshot(
                egraph,
                root=current,
                metadata={
                    "system": "egglog",
                    "case_id": case_id,
                    "step": f"outer_{outer_pass}_inner_{inner_iteration}_after_rewrite",
                    "outer_pass": outer_pass,
                    "inner_iteration": inner_iteration,
                    "phase": "rewrite",
                    "source": source,
                    "pass_input_expr": render_num(current),
                    "rewrite_updated": rewrite_report.updated,
                    "rewrite_can_stop": rewrite_report.can_stop,
                },
            ).write_json(rewrite_path)
            step_paths.append(rewrite_path)

            analysis_report = egraph.run(analysis_schedule.saturate())
            analysis_snapshot = build_egglog_snapshot(
                egraph,
                root=current,
                metadata={},
            )
            node_count = analysis_snapshot.metadata["node_count"]
            class_count = analysis_snapshot.metadata["class_count"]
            assert isinstance(node_count, int)
            assert isinstance(class_count, int)
            current_counts = (
                node_count,
                class_count,
            )
            size_stable = current_counts == previous_counts
            analysis_path = output_dir / f"outer_{outer_pass}_inner_{inner_iteration}_after_analysis.json"
            analysis_snapshot.metadata.update(
                {
                    "system": "egglog",
                    "case_id": case_id,
                    "step": f"outer_{outer_pass}_inner_{inner_iteration}_after_analysis",
                    "outer_pass": outer_pass,
                    "inner_iteration": inner_iteration,
                    "phase": "analysis",
                    "source": source,
                    "pass_input_expr": render_num(current),
                    "rewrite_updated": rewrite_report.updated,
                    "analysis_updated": analysis_report.updated,
                    "size_stable": size_stable,
                    "rewrite_can_stop": rewrite_report.can_stop,
                    "analysis_can_stop": analysis_report.can_stop,
                }
            )
            analysis_snapshot.write_json(analysis_path)
            step_paths.append(analysis_path)
            if size_stable:
                break
            previous_counts = current_counts

        extract_path = output_dir / f"outer_{outer_pass}_extract.json"
        snapshot = build_egglog_snapshot(
            egraph,
            root=current,
            metadata={
                "system": "egglog",
                "case_id": case_id,
                "step": f"outer_{outer_pass}_extract",
                "outer_pass": outer_pass,
                "phase": "extraction",
                "source": source,
                "pass_input_expr": render_num(current),
            },
        )
        snapshot.write_json(extract_path)
        step_paths.append(extract_path)
        extracted_expr = str(snapshot.metadata["root_extracted_expr"])
        if extracted_expr == render_num(current):
            current = parse_expression(extracted_expr)
            break
        current = parse_expression(extracted_expr)

    final_snapshot = SnapshotTables(
        metadata={
            "system": "egglog",
            "case_id": case_id,
            "step": "final_simplify_e",
            "phase": "extraction",
            "source": source,
            "final_rendered": render_num(current),
        },
        tables={
            "root": [
                {
                    "extracted_expr": render_num(current),
                }
            ],
        },
    )
    final_path = output_dir / "final_simplify_e.json"
    final_snapshot.write_json(final_path)
    step_paths.append(final_path)
    return TraceResult(
        system="egglog",
        case_id=case_id,
        output_dir=output_dir,
        step_paths=step_paths,
        final_rendered=render_num(current),
    )
