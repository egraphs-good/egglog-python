"""Compare Egglog baseline reruns before and after rewrite-rule changes."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from egglog.exp.param_eq.egglog_results import default_egglog_results_path, load_egglog_results
from egglog.exp.param_eq.original_results import NA_REPR, RESULT_KEY_COLUMNS, load_original_results_for_join
from egglog.exp.param_eq.paths import artifact_dir

SUCCESS_STATUSES = frozenset({"saturated", "ok"})
ADDED_RULE_FAMILIES = frozenset({
    "exp_additive_constant_extraction",
    "quotient_coefficient_exposure",
    "affine_constant_collection",
    "log_abs_scale_extraction",
    "exp_product_or_quotient_normalization",
    "square_quotient_normalization",
})
ARTIFACT_OR_NO_RULE_FAMILIES = frozenset({"no_rank_reducing_rule", "shared_parameter_or_rank_artifact"})
REVIEW_COLUMNS = [
    "agent_status",
    "conclusion_kind",
    "new_rule_family",
    "proposed_rule",
    "proof_summary",
    "remaining_gap_notes",
]
SOURCE_COLUMNS = [
    *RESULT_KEY_COLUMNS,
    "algorithm",
    "n_rank",
    "source_orig_parsed_expr",
    "source_orig_parsed_n_params",
    "source_before_nodes",
]
RESULT_VALUE_COLUMNS = [
    "status",
    "after_params",
    "after_rank_difference",
    "after_parsed_rank_difference",
    "simpl_parsed_n_params",
    "runtime_ms",
    "peak_rss_mb",
    "passes",
    "egraph_total_size",
    "rendered",
]


def before_results_path() -> Path:
    return artifact_dir() / "egglog_results_before_extra_rules.csv"


def before_rank_misses_path() -> Path:
    return artifact_dir() / "egglog_rank_misses_before_extra_rules.csv"


def after_rank_misses_path() -> Path:
    return artifact_dir() / "egglog_rank_misses_after_extra_rules.csv"


def longer_probe_path() -> Path:
    return artifact_dir() / "egglog_rank_miss_longer_runs_after_extra_rules.csv"


def review_path() -> Path:
    return artifact_dir() / "egglog_rank_miss_agent_review.csv"


def delta_output_path() -> Path:
    return artifact_dir() / "egglog_rerun_delta.csv"


def remaining_output_path() -> Path:
    return artifact_dir() / "egglog_rank_misses_remaining.csv"


def report_output_path() -> Path:
    return artifact_dir() / "egglog_rerun_report.md"


def _read_csv_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, na_values=[NA_REPR], keep_default_na=True)


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    if frame.empty or not set(RESULT_KEY_COLUMNS).issubset(frame.columns):
        return set()
    return set(frame.loc[:, RESULT_KEY_COLUMNS].itertuples(index=False, name=None))


def _is_true(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _is_positive(value: object) -> bool:
    return not pd.isna(value) and float(value) > 0


def _is_nonpositive(value: object) -> bool:
    return not pd.isna(value) and float(value) <= 0


def _is_negative(value: object) -> bool:
    return not pd.isna(value) and float(value) < 0


def _select_source_columns(source: pd.DataFrame) -> pd.DataFrame:
    result = source.copy()
    for column in SOURCE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, SOURCE_COLUMNS]


def _select_result_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    for column in RESULT_VALUE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result.loc[:, [*RESULT_KEY_COLUMNS, *RESULT_VALUE_COLUMNS]]
    return result.rename(
        columns={
            "status": f"{prefix}_status",
            "after_params": f"{prefix}_after_params",
            "after_rank_difference": f"{prefix}_rank_gap",
            "after_parsed_rank_difference": f"{prefix}_parsed_rank_gap",
            "simpl_parsed_n_params": f"{prefix}_parsed_after_params",
            "runtime_ms": f"{prefix}_runtime_ms",
            "peak_rss_mb": f"{prefix}_peak_rss_mb",
            "passes": f"{prefix}_passes",
            "egraph_total_size": f"{prefix}_egraph_total_size",
            "rendered": f"{prefix}_rendered",
        }
    )


def _select_review_columns(review: pd.DataFrame | None) -> pd.DataFrame:
    if review is None or review.empty:
        return pd.DataFrame(columns=[*RESULT_KEY_COLUMNS, *REVIEW_COLUMNS])
    result = review.copy()
    for column in REVIEW_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, [*RESULT_KEY_COLUMNS, *REVIEW_COLUMNS]].drop_duplicates(RESULT_KEY_COLUMNS)


def _select_probe_columns(probe: pd.DataFrame | None) -> pd.DataFrame:
    if probe is None or probe.empty:
        return pd.DataFrame(
            columns=[
                *RESULT_KEY_COLUMNS,
                "longer_probe_status",
                "longer_probe_after_params",
                "longer_probe_rank_gap",
                "longer_probe_improved",
                "longer_probe_reached_rank",
                "longer_probe_runtime_ms",
                "longer_probe_peak_rss_mb",
            ]
        )
    result = probe.copy()
    for column in [
        "probe_status",
        "probe_after_params",
        "probe_rank_gap",
        "improved",
        "reached_rank",
        "probe_runtime_ms",
        "probe_peak_rss_mb",
    ]:
        if column not in result.columns:
            result[column] = pd.NA
    result["_probe_reached"] = result["reached_rank"].map(_is_true)
    result["_probe_improved"] = result["improved"].map(_is_true)
    result["_probe_ok"] = result["probe_status"] == "ok"
    result["_param_delta"] = pd.to_numeric(result.get("param_delta"), errors="coerce")
    result = result.sort_values(
        [*RESULT_KEY_COLUMNS, "_probe_reached", "_probe_improved", "_probe_ok", "_param_delta"],
        ascending=[True, True, True, True, True, False, False, False, False],
    )
    result = result.drop_duplicates(RESULT_KEY_COLUMNS, keep="first")
    result = result.rename(
        columns={
            "probe_status": "longer_probe_status",
            "probe_after_params": "longer_probe_after_params",
            "probe_rank_gap": "longer_probe_rank_gap",
            "improved": "longer_probe_improved",
            "reached_rank": "longer_probe_reached_rank",
            "probe_runtime_ms": "longer_probe_runtime_ms",
            "probe_peak_rss_mb": "longer_probe_peak_rss_mb",
        }
    )
    return result.loc[
        :,
        [
            *RESULT_KEY_COLUMNS,
            "longer_probe_status",
            "longer_probe_after_params",
            "longer_probe_rank_gap",
            "longer_probe_improved",
            "longer_probe_reached_rank",
            "longer_probe_runtime_ms",
            "longer_probe_peak_rss_mb",
        ],
    ]


def _row_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in RESULT_KEY_COLUMNS)


def _primary_outcome(row: pd.Series) -> str:
    for column, outcome in [
        ("missing_rank", "missing_rank"),
        ("execution_issue", "execution_issue"),
        ("under_rank", "under_rank"),
        ("fixed_by_extra_rules", "fixed_by_extra_rules"),
        ("still_missed", "still_missed"),
        ("newly_missed", "newly_missed"),
        ("regressed_params", "regressed_params"),
    ]:
        if _is_true(row[column]):
            return outcome
    return "unchanged_ranked"


def _remaining_issue(row: pd.Series) -> str | None:
    if not _is_true(row["still_missed"]):
        return None
    if _is_true(row["longer_probe_improved"]) or _is_true(row["longer_probe_reached_rank"]):
        return "existing_rules_more_iterations"
    agent_status = row.get("agent_status")
    conclusion_kind = row.get("conclusion_kind")
    family = row.get("new_rule_family")
    if pd.isna(agent_status) or agent_status != "done" or conclusion_kind in {"pending", "inconclusive"}:
        return "unreviewed_remaining"
    if conclusion_kind == "needs_many_params_or_rank_artifact" or family in ARTIFACT_OR_NO_RULE_FAMILIES:
        return "needs_many_params_or_rank_artifact"
    if family in ADDED_RULE_FAMILIES:
        return "rule_added_but_not_matching"
    return "rule_not_added"


def build_rerun_delta_frame(
    source: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    before_misses: pd.DataFrame | None = None,
    after_misses: pd.DataFrame | None = None,
    review: pd.DataFrame | None = None,
    longer_probe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    before_miss_keys = _key_set(before_misses if before_misses is not None else pd.DataFrame())
    after_miss_keys = _key_set(after_misses if after_misses is not None else pd.DataFrame())
    frame = _select_source_columns(source)
    frame = frame.merge(
        _select_result_columns(before, "before"), on=RESULT_KEY_COLUMNS, how="left", validate="one_to_one"
    )
    frame = frame.merge(
        _select_result_columns(after, "after"), on=RESULT_KEY_COLUMNS, how="left", validate="one_to_one"
    )
    frame = frame.merge(_select_review_columns(review), on=RESULT_KEY_COLUMNS, how="left", validate="one_to_one")
    frame = frame.merge(_select_probe_columns(longer_probe), on=RESULT_KEY_COLUMNS, how="left", validate="one_to_one")

    frame["before_rank_miss"] = [
        _row_key(row) in before_miss_keys or _is_positive(row["before_rank_gap"]) for _, row in frame.iterrows()
    ]
    frame["after_rank_miss"] = [
        _row_key(row) in after_miss_keys or _is_positive(row["after_rank_gap"]) for _, row in frame.iterrows()
    ]
    frame["param_delta"] = pd.to_numeric(frame["before_after_params"], errors="coerce") - pd.to_numeric(
        frame["after_after_params"], errors="coerce"
    )
    frame["rank_gap_delta"] = pd.to_numeric(frame["before_rank_gap"], errors="coerce") - pd.to_numeric(
        frame["after_rank_gap"], errors="coerce"
    )
    after_status = frame["after_status"].fillna("missing")
    frame["missing_rank"] = frame["n_rank"].isna()
    frame["execution_issue"] = ~after_status.isin(SUCCESS_STATUSES) | frame["after_after_params"].isna()
    frame["fixed_by_extra_rules"] = [
        bool(not missing_rank and not execution_issue and before_miss and _is_nonpositive(after_gap))
        for missing_rank, execution_issue, before_miss, after_gap in zip(
            frame["missing_rank"],
            frame["execution_issue"],
            frame["before_rank_miss"],
            frame["after_rank_gap"],
            strict=True,
        )
    ]
    frame["still_missed"] = [
        bool(not missing_rank and not execution_issue and before_miss and _is_positive(after_gap))
        for missing_rank, execution_issue, before_miss, after_gap in zip(
            frame["missing_rank"],
            frame["execution_issue"],
            frame["before_rank_miss"],
            frame["after_rank_gap"],
            strict=True,
        )
    ]
    frame["newly_missed"] = [
        bool(not missing_rank and not execution_issue and not before_miss and _is_positive(after_gap))
        for missing_rank, execution_issue, before_miss, after_gap in zip(
            frame["missing_rank"],
            frame["execution_issue"],
            frame["before_rank_miss"],
            frame["after_rank_gap"],
            strict=True,
        )
    ]
    frame["under_rank"] = [
        bool(not missing_rank and not execution_issue and _is_negative(after_gap))
        for missing_rank, execution_issue, after_gap in zip(
            frame["missing_rank"], frame["execution_issue"], frame["after_rank_gap"], strict=True
        )
    ]
    frame["regressed_params"] = [
        bool(
            not missing_rank
            and not execution_issue
            and not pd.isna(before_params)
            and not pd.isna(after_params)
            and float(after_params) > float(before_params)
        )
        for missing_rank, execution_issue, before_params, after_params in zip(
            frame["missing_rank"],
            frame["execution_issue"],
            frame["before_after_params"],
            frame["after_after_params"],
            strict=True,
        )
    ]
    frame["outcome"] = frame.apply(_primary_outcome, axis=1)
    frame["remaining_issue"] = frame.apply(_remaining_issue, axis=1)
    return _sort_delta(frame)


def _sort_delta(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_expr_len"] = result["source_orig_parsed_expr"].fillna("").map(len)
    result = result.sort_values(
        [
            "source_orig_parsed_n_params",
            "source_before_nodes",
            "_expr_len",
            "dataset",
            "raw_index",
            "algo_row",
            "input_kind",
        ],
        na_position="last",
    )
    return result.drop(columns="_expr_len").reset_index(drop=True)


def remaining_rank_misses_frame(delta: pd.DataFrame) -> pd.DataFrame:
    remaining = delta[delta["still_missed"] == True].copy()  # noqa: E712 - pandas boolean mask.
    if remaining.empty:
        return remaining
    return _sort_delta(remaining)


def _counts(series: pd.Series) -> Counter[str]:
    return Counter(str(value) for value in series.fillna("na"))


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_format_cell(value) for value in row) + " |" for row in rows)
    return lines


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return "na"
    text = str(value).replace("\n", " ")
    if len(text) > 120:
        return text[:117] + "..."
    return text


def render_markdown_report(delta: pd.DataFrame, remaining: pd.DataFrame) -> str:
    lines = ["# Egglog Baseline Rerun Report", ""]
    ranked = delta[delta["n_rank"].notna()]
    lines.extend([
        "## Overview",
        "",
        f"- Total rows: `{len(delta)}`",
        f"- Rows with rank: `{len(ranked)}`",
        f"- Before rank misses: `{int(delta['before_rank_miss'].sum())}`",
        f"- After rank misses: `{int(delta['after_rank_miss'].sum())}`",
        f"- Fixed by extra rules: `{int(delta['fixed_by_extra_rules'].sum())}`",
        f"- Still missed: `{int(delta['still_missed'].sum())}`",
        f"- Newly missed: `{int(delta['newly_missed'].sum())}`",
        f"- Execution issues: `{int(delta['execution_issue'].sum())}`",
        f"- Under-rank rows: `{int(delta['under_rank'].sum())}`",
        f"- Param regressions: `{int(delta['regressed_params'].sum())}`",
        "",
        "## Outcomes",
        "",
    ])
    lines.extend(_table(["Outcome", "Rows"], sorted(_counts(delta["outcome"]).items())))
    lines.extend(["", "## Remaining Misses By Cause", ""])
    if remaining.empty:
        lines.append("No remaining misses.")
    else:
        lines.extend(_table(["Remaining issue", "Rows"], sorted(_counts(remaining["remaining_issue"]).items())))
        lines.extend(["", "### Longer-Run Probe Status", ""])
        lines.extend(_table(["Probe status", "Rows"], sorted(_counts(remaining["longer_probe_status"]).items())))
        lines.extend(["", "### Remaining Misses By Review Family", ""])
        lines.extend(_table(["Rule family", "Rows"], sorted(_counts(remaining["new_rule_family"]).items())))
    lines.extend(["", "## Smallest Fixed Previous Misses", ""])
    fixed = delta[delta["fixed_by_extra_rules"] == True].head(20)  # noqa: E712 - pandas boolean mask.
    if fixed.empty:
        lines.append("No previous misses were fixed.")
    else:
        lines.extend(
            _table(
                ["Key", "Before gap", "After gap", "Family", "Expression"],
                (
                    (
                        _key_label(row),
                        row.before_rank_gap,
                        row.after_rank_gap,
                        row.new_rule_family,
                        row.source_orig_parsed_expr,
                    )
                    for row in fixed.itertuples(index=False)
                ),
            )
        )
    lines.extend(["", "## Smallest Remaining Misses", ""])
    if remaining.empty:
        lines.append("No remaining misses.")
    else:
        lines.extend(
            _table(
                ["Key", "Gap", "Cause", "Family", "Expression"],
                (
                    (
                        _key_label(row),
                        row.after_rank_gap,
                        row.remaining_issue,
                        row.new_rule_family,
                        row.source_orig_parsed_expr,
                    )
                    for row in remaining.head(30).itertuples(index=False)
                ),
            )
        )
    lines.extend(["", "## Cause Interpretation", ""])
    lines.extend([
        "- `existing_rules_more_iterations`: the longer-run probe improved or reached rank; this is budget/scheduler-sensitive.",
        "- `rule_added_but_not_matching`: the reviewed family is in the new minimal rule set, but the rerun still missed; inspect orientation, guards, or needed composition.",
        "- `rule_not_added`: the reviewed family was intentionally not part of this minimal rule set.",
        "- `needs_many_params_or_rank_artifact`: reviewed as irreducible under tree occurrence counting or likely rank/counting artifact.",
        "- `unreviewed_remaining`: no completed usable review taxonomy exists for the row.",
    ])
    return "\n".join(lines) + "\n"


def _key_label(row: object) -> str:
    return f"{row.dataset}/{row.algorithm}/raw {row.raw_index}/{row.input_kind}"


def write_comparison_outputs(
    delta: pd.DataFrame,
    *,
    delta_path: Path = delta_output_path(),
    remaining_path: Path = remaining_output_path(),
    report_path: Path = report_output_path(),
) -> tuple[pd.DataFrame, str]:
    remaining = remaining_rank_misses_frame(delta)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(delta_path, index=False, na_rep=NA_REPR)
    remaining.to_csv(remaining_path, index=False, na_rep=NA_REPR)
    report = render_markdown_report(delta, remaining)
    report_path.write_text(report, encoding="utf-8")
    return remaining, report


def load_rerun_delta_frame(
    *,
    before_results: Path = before_results_path(),
    after_results: Path = default_egglog_results_path(),
    before_misses: Path = before_rank_misses_path(),
    after_misses: Path = after_rank_misses_path(),
    review: Path = review_path(),
    longer_probe: Path = longer_probe_path(),
) -> pd.DataFrame:
    source = load_original_results_for_join()
    return build_rerun_delta_frame(
        source,
        load_egglog_results(before_results, variant="baseline"),
        load_egglog_results(after_results, variant="baseline"),
        before_misses=_read_csv_if_exists(before_misses),
        after_misses=_read_csv_if_exists(after_misses),
        review=_read_csv_if_exists(review),
        longer_probe=_read_csv_if_exists(longer_probe),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-results", type=Path, default=before_results_path())
    parser.add_argument("--after-results", type=Path, default=default_egglog_results_path())
    parser.add_argument("--before-misses", type=Path, default=before_rank_misses_path())
    parser.add_argument("--after-misses", type=Path, default=after_rank_misses_path())
    parser.add_argument("--review", type=Path, default=review_path())
    parser.add_argument("--longer-probe", type=Path, default=longer_probe_path())
    parser.add_argument("--delta-output", type=Path, default=delta_output_path())
    parser.add_argument("--remaining-output", type=Path, default=remaining_output_path())
    parser.add_argument("--report-output", type=Path, default=report_output_path())
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    delta = load_rerun_delta_frame(
        before_results=args.before_results,
        after_results=args.after_results,
        before_misses=args.before_misses,
        after_misses=args.after_misses,
        review=args.review,
        longer_probe=args.longer_probe,
    )
    remaining, _ = write_comparison_outputs(
        delta,
        delta_path=args.delta_output,
        remaining_path=args.remaining_output,
        report_path=args.report_output,
    )
    print(f"{args.delta_output} ({len(delta)} rows)")
    print(f"{args.remaining_output} ({len(remaining)} rows)")
    print(args.report_output)


if __name__ == "__main__":
    main()
