"""Compare stepwise Haskell and Egglog param-eq traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egglog.exp.param_eq.paths import GOLDEN_PATH
from egglog.exp.param_eq.trace_egglog import TRACE_ROOT, trace_egglog_case
from egglog.exp.param_eq.trace_haskell import haskell_trace_available, trace_haskell_case
from egglog.exp.param_eq.trace_tables import JsonValue, SnapshotTables, read_snapshot

REDUCED_TOY_SOURCE = "(-2.2516087483e-06) + (-0.009788252341175882 * ((x0 * x0) + 1))"
REDUCED_TOY_CASE_ID = "reduced_pagie_second_pass_toy"


@dataclass(frozen=True)
class StepDifference:
    """The first detected difference for one comparison boundary."""

    step: str
    aspect: str
    detail: str


@dataclass(frozen=True)
class ComparisonSummary:
    """User-facing summary of the first detected divergence."""

    case_id: str
    first_differing_step: str | None
    first_differing_aspect: str | None
    likely_phase: str | None
    detail: str | None
    ignored_aspects: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "first_differing_step": self.first_differing_step,
            "first_differing_aspect": self.first_differing_aspect,
            "likely_phase": self.likely_phase,
            "detail": self.detail,
            "ignored_aspects": list(self.ignored_aspects),
        }


def _known_case_sources() -> dict[str, str]:
    payload = json.loads(GOLDEN_PATH.read_text())
    result = {case["case_id"]: case["source"] for case in payload["cases"]}
    result[REDUCED_TOY_CASE_ID] = REDUCED_TOY_SOURCE
    return result


def _step_sort_key(step: str) -> tuple[int, int, int, int]:
    if step == "final_simplify_e":
        return (99, 99, 99, 99)
    parts = step.split("_")
    outer_pass = int(parts[1])
    if parts[2] == "pass":
        return (outer_pass, 0, 0, 0)
    if parts[2] == "extract":
        return (outer_pass, 98, 0, 0)
    inner_iteration = int(parts[3])
    phase = parts[-1]
    phase_rank = 0 if phase == "rewrite" else 1
    return (outer_pass, inner_iteration, phase_rank, 0)


def _load_system_steps(case_id: str, system: str, *, trace_root: Path) -> dict[str, SnapshotTables]:
    directory = trace_root / case_id / system
    return {path.stem: read_snapshot(path) for path in sorted(directory.glob("*.json"))}


def _analysis_key(value: object) -> str:
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "none":
            return "none"
        if kind == "some":
            return f"some:{float(value['value']):.12g}"
    return json.dumps(value, sort_keys=True)


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _canonical_numeric(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    try:
        return str(float(stripped))
    except ValueError:
        return stripped


def _normalize_egglog_signature(snapshot: SnapshotTables, *, include_best_expr: bool) -> Counter[str]:
    if include_best_expr:
        return Counter(str(row["signature"]) for row in snapshot.tables["classes"])
    return Counter(str(row["semantic_signature"]) for row in snapshot.tables["classes"])


def _normalize_haskell_signature(snapshot: SnapshotTables, *, include_best_expr: bool) -> Counter[str]:  # noqa: C901
    class_rows = {str(row["class_id"]): row for row in snapshot.tables["classes"]}
    nodes_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in snapshot.tables["nodes"]:
        nodes_by_class[str(node["class_id"])].append(node)

    cache: dict[str, str] = {}
    active: set[str] = set()

    def node_signature(node: dict[str, Any]) -> str:  # noqa: C901, PLR0911
        op = str(node["op"])
        children = _string_list(node.get("children"))
        if op.startswith("VarF "):
            return f"Var(x{op.removeprefix('VarF ').strip()})"
        if op.startswith("ConstF "):
            return f"Const({_canonical_numeric(op.removeprefix('ConstF ').strip())})"
        if op.startswith("AddF"):
            return f"Add({','.join(class_signature(child) for child in children)})"
        if op.startswith("SubF"):
            return f"Sub({','.join(class_signature(child) for child in children)})"
        if op.startswith("MulF"):
            return f"Mul({','.join(class_signature(child) for child in children)})"
        if op.startswith("DivF"):
            return f"Div({','.join(class_signature(child) for child in children)})"
        if op.startswith("PowerF"):
            return f"Pow({','.join(class_signature(child) for child in children)})"
        if op.startswith("FunF Exp"):
            return f"Exp({class_signature(children[0])})"
        if op.startswith("FunF Log"):
            return f"Log({class_signature(children[0])})"
        if op.startswith("FunF Sqrt"):
            return f"Sqrt({class_signature(children[0])})"
        if op.startswith("FunF Abs"):
            return f"Abs({class_signature(children[0])})"
        return f"{op}({','.join(class_signature(child) for child in children)})"

    def class_signature(class_id: str) -> str:
        if class_id not in class_rows:
            return f"External({class_id})"
        if class_id in cache:
            return cache[class_id]
        if class_id in active:
            return f"Cycle({class_id})"
        active.add(class_id)
        row = class_rows[class_id]
        members = sorted(node_signature(node) for node in nodes_by_class[class_id])
        prefix = f"{_analysis_key(row['analysis'])}|"
        if include_best_expr:
            best_expr = str(row.get("best_expr", "<missing>"))
            prefix = f"{prefix}best:{best_expr}|"
        signature = f"{prefix}{'||'.join(members)}"
        cache[class_id] = signature
        active.remove(class_id)
        return signature

    return Counter(class_signature(class_id) for class_id in sorted(class_rows))


def _compare_snapshots(
    left: SnapshotTables,
    right: SnapshotTables,
    *,
    ignored_aspects: frozenset[str] = frozenset(),
) -> StepDifference | None:
    left_root = left.tables["root"][0] if left.tables["root"] else {}
    right_root = right.tables["root"][0] if right.tables["root"] else {}
    if "root_extracted_expr" not in ignored_aspects and left_root.get("extracted_expr") != right_root.get("extracted_expr"):
        return StepDifference(
            str(right.metadata["step"]),
            "root_extracted_expr",
            f"{left_root.get('extracted_expr')} != {right_root.get('extracted_expr')}",
        )
    if "root_analysis" not in ignored_aspects and _analysis_key(left.metadata.get("root_analysis")) != _analysis_key(
        right.metadata.get("root_analysis")
    ):
        return StepDifference(
            str(right.metadata["step"]),
            "root_analysis",
            f"{left.metadata.get('root_analysis')} != {right.metadata.get('root_analysis')}",
        )
    count_keys = ("class_count", "node_count", "memo_size")
    for key in count_keys:
        if key in ignored_aspects:
            continue
        left_value = left.metadata.get(key)
        right_value = right.metadata.get(key)
        if left_value is None or right_value is None:
            continue
        if left_value != right_value:
            return StepDifference(
                str(right.metadata["step"]), key, f"{left.metadata.get(key)} != {right.metadata.get(key)}"
            )
    include_best_expr = "class_best_expr" not in ignored_aspects
    left_classes = (
        _normalize_egglog_signature(left, include_best_expr=include_best_expr)
        if left.metadata["system"] == "egglog"
        else _normalize_haskell_signature(left, include_best_expr=include_best_expr)
    )
    right_classes = (
        _normalize_egglog_signature(right, include_best_expr=include_best_expr)
        if right.metadata["system"] == "egglog"
        else _normalize_haskell_signature(right, include_best_expr=include_best_expr)
    )
    if "normalized_classes" not in ignored_aspects and left_classes != right_classes:
        return StepDifference(str(right.metadata["step"]), "normalized_classes", "class signature multiset differs")
    return None


def compare_case_traces(
    case_id: str,
    *,
    trace_root: Path = TRACE_ROOT,
    ignored_aspects: frozenset[str] = frozenset(),
) -> ComparisonSummary:
    """Compare one traced case and return the first differing phase."""
    egg_steps = _load_system_steps(case_id, "egglog", trace_root=trace_root)
    hs_steps = _load_system_steps(case_id, "haskell", trace_root=trace_root)

    for step in sorted(hs_steps, key=_step_sort_key):
        if step.endswith("_pass_start"):
            if step not in egg_steps:
                # Later outer-pass checkpoints can disappear when the systems
                # only differ by which equal-cost representative was extracted.
                if ignored_aspects & {"root_extracted_expr", "class_best_expr"}:
                    continue
                return ComparisonSummary(
                    case_id,
                    step,
                    "missing_step",
                    "analysis",
                    f"Egglog trace did not emit step {step!r}",
                    tuple(sorted(ignored_aspects)),
                )
            diff = _compare_snapshots(egg_steps[step], hs_steps[step], ignored_aspects=ignored_aspects)
            if diff is not None:
                return ComparisonSummary(case_id, diff.step, diff.aspect, "analysis", diff.detail, tuple(sorted(ignored_aspects)))
            continue
        if step.endswith("_after_rebuild"):
            rewrite_step = step.replace("_after_rebuild", "_after_rewrite")
            analysis_step = step.replace("_after_rebuild", "_after_analysis")
            if rewrite_step not in egg_steps or analysis_step not in egg_steps:
                return ComparisonSummary(
                    case_id,
                    step,
                    "missing_step",
                    "rewrite",
                    f"Egglog trace did not emit rewrite/analysis checkpoints for {step!r}",
                    tuple(sorted(ignored_aspects)),
                )
            rewrite_diff = _compare_snapshots(egg_steps[rewrite_step], hs_steps[step], ignored_aspects=ignored_aspects)
            analysis_diff = _compare_snapshots(egg_steps[analysis_step], hs_steps[step], ignored_aspects=ignored_aspects)
            if analysis_diff is None:
                continue
            if rewrite_diff is None:
                return ComparisonSummary(
                    case_id,
                    analysis_diff.step,
                    analysis_diff.aspect,
                    "analysis",
                    analysis_diff.detail,
                    tuple(sorted(ignored_aspects)),
                )
            return ComparisonSummary(
                case_id,
                rewrite_diff.step,
                rewrite_diff.aspect,
                "rewrite",
                rewrite_diff.detail,
                tuple(sorted(ignored_aspects)),
            )
        if step not in egg_steps:
            return ComparisonSummary(
                case_id,
                step,
                "missing_step",
                "extraction",
                f"Egglog trace did not emit step {step!r}",
                tuple(sorted(ignored_aspects)),
            )
        diff = _compare_snapshots(egg_steps[step], hs_steps[step], ignored_aspects=ignored_aspects)
        if diff is not None:
            return ComparisonSummary(case_id, diff.step, diff.aspect, "extraction", diff.detail, tuple(sorted(ignored_aspects)))

    return ComparisonSummary(case_id, None, None, None, None, tuple(sorted(ignored_aspects)))


def write_comparison_report(summary: ComparisonSummary, *, trace_root: Path = TRACE_ROOT) -> tuple[Path, Path]:
    """Write JSON and Markdown comparison summaries beside the trace output."""
    case_dir = trace_root / summary.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    json_path = case_dir / "comparison_summary.json"
    md_path = case_dir / "comparison_summary.md"
    json_path.write_text(json.dumps(summary.to_jsonable(), indent=2, sort_keys=False) + "\n")
    if summary.first_differing_step is None:
        markdown = f"# {summary.case_id}\n\nNo divergence was detected in the traced checkpoints.\n"
    else:
        markdown = "\n".join([
            f"# {summary.case_id}",
            "",
            f"- First differing step: `{summary.first_differing_step}`",
            f"- First differing aspect: `{summary.first_differing_aspect}`",
            f"- Likely phase: `{summary.likely_phase}`",
            f"- Detail: {summary.detail}",
            (
                f"- Ignored aspects: `{', '.join(summary.ignored_aspects)}`"
                if summary.ignored_aspects
                else "- Ignored aspects: none"
            ),
            "",
        ])
    md_path.write_text(markdown)
    return json_path, md_path


def trace_and_compare_case(
    *,
    case_id: str,
    source: str,
    trace_root: Path = TRACE_ROOT,
    ignored_aspects: frozenset[str] = frozenset(),
) -> ComparisonSummary:
    """Run both traces, compare them, and persist the summary."""
    trace_egglog_case(case_id=case_id, source=source, output_root=trace_root)
    trace_haskell_case(case_id=case_id, source=source, output_root=trace_root)
    summary = compare_case_traces(case_id, trace_root=trace_root, ignored_aspects=ignored_aspects)
    write_comparison_report(summary, trace_root=trace_root)
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source")
    parser.add_argument("--ignore-aspect", action="append", default=[])
    args = parser.parse_args()

    if not haskell_trace_available():
        msg = "Haskell trace prerequisites are not available"
        raise SystemExit(msg)

    known_sources = _known_case_sources()
    source = args.source or known_sources.get(args.case_id)
    if source is None:
        msg = f"Unknown case id {args.case_id!r}; pass --source explicitly"
        raise SystemExit(msg)
    summary = trace_and_compare_case(
        case_id=args.case_id,
        source=source,
        ignored_aspects=frozenset(args.ignore_aspect),
    )
    print(json.dumps(summary.to_jsonable(), indent=2, sort_keys=False))


if __name__ == "__main__":
    _cli()
