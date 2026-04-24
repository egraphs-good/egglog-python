"""Print a Rich summary comparing two canonical param-eq corpus CSV runs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from egglog.exp.param_eq.egglog_results import EGGLOG_RESULTS_SCHEMA, load_egglog_results
from egglog.exp.param_eq.live_results import LIVE_RESULTS_SCHEMA, load_live_results
from egglog.exp.param_eq.original_results import RESULT_KEY_COLUMNS

DISPLAY_NAMES = {
    ("haskell", "paper"): "Archived Haskell",
    ("haskell", "live"): "Live Haskell",
    ("egglog", "baseline"): "Egglog",
    ("egglog", "container"): "Egglog Container",
}


def _display_name(implementation: str, variant: str) -> str:
    return DISPLAY_NAMES.get((implementation, variant), f"{implementation} {variant}")


def _load_frame(path: Path, input_kind: str, *, variant: str | None) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    if header == list(EGGLOG_RESULTS_SCHEMA.columns):
        frame = load_egglog_results(path, variant=variant)
        if variant is None and frame["variant"].nunique(dropna=True) > 1:
            msg = f"{path} contains multiple Egglog variants; pass --old-variant/--new-variant"
            raise ValueError(msg)
    elif header == list(LIVE_RESULTS_SCHEMA.columns):
        frame = load_live_results(path)
    else:
        msg = f"Unrecognized result schema for {path}"
        raise ValueError(msg)
    if input_kind != "both":
        frame = frame[frame["input_kind"] == input_kind].copy()
    return frame


def _median(series: pd.Series) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.median())


def _fmt_number(value: float | None, *, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "[dim]na[/dim]"
    if value == int(value):
        return str(int(value))
    return f"{value:.{digits}f}"


def _change_style(value: float | None, *, lower_is_better: bool) -> str:
    if value is None or value == 0:
        return "dim"
    return "green" if (value < 0) == lower_is_better else "red"


def _fmt_compared_number(
    value: float | None,
    delta: float | None,
    *,
    lower_is_better: bool,
    digits: int = 1,
) -> str:
    return f"[{_change_style(delta, lower_is_better=lower_is_better)}]{_fmt_number(value, digits=digits)}[/]"


def _fmt_delta(value: float | None, *, lower_is_better: bool, digits: int = 1) -> str:
    if value is None:
        return "[dim]na[/dim]"
    if value == 0:
        return "0"
    return f"[{_change_style(value, lower_is_better=lower_is_better)}]{value:+.{digits}f}[/]"


def _fmt_pct(value: float | None, *, lower_is_better: bool) -> str:
    if value is None:
        return "[dim]na[/dim]"
    if value == 0:
        return "0.0%"
    return f"[{_change_style(value, lower_is_better=lower_is_better)}]{value:+.1f}%[/]"


def _fmt_outcome(value: float | None, *, lower_is_better: bool) -> str:
    if value is None:
        return "[dim]unknown[/dim]"
    if value == 0:
        return "[dim]same[/dim]"
    if (value < 0) == lower_is_better:
        return "[green]better[/green]"
    return "[red]worse[/red]"


def _fmt_changed(changed: bool) -> str:
    return "[yellow]changed[/yellow]" if changed else "[green]same[/green]"


def _comparison_frame(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = old.merge(new, on=[*RESULT_KEY_COLUMNS, "algorithm"], suffixes=("_old", "_new"))
    merged["row_id"] = (
        merged["dataset"].astype(str)
        + "/"
        + merged["algorithm"].astype(str)
        + " row="
        + merged["algo_row"].astype(str)
        + " raw="
        + merged["raw_index"].astype(str)
        + " "
        + merged["input_kind"].astype(str)
    )
    merged["runtime_delta_ms"] = merged["runtime_ms_new"] - merged["runtime_ms_old"]
    merged["runtime_delta_pct"] = (merged["runtime_delta_ms"] / merged["runtime_ms_old"]) * 100.0
    merged.loc[merged["runtime_ms_old"] == 0, "runtime_delta_pct"] = pd.NA
    merged["egraph_total_size_delta"] = merged["egraph_total_size_new"] - merged["egraph_total_size_old"]
    merged["param_reduction_old"] = merged["before_params_old"] - merged["after_params_old"]
    merged["param_reduction_new"] = merged["before_params_new"] - merged["after_params_new"]
    merged["param_reduction_delta"] = merged["param_reduction_new"] - merged["param_reduction_old"]
    merged["simplified_rank_old"] = merged["after_parsed_rank_difference_old"]
    merged["simplified_rank_new"] = merged["after_parsed_rank_difference_new"]
    merged["simplified_rank_delta"] = merged["simplified_rank_new"] - merged["simplified_rank_old"]
    merged["rendered_changed"] = (
        merged["simpl_parsed_expr_old"].fillna("<na>") != merged["simpl_parsed_expr_new"].fillna("<na>")
    )
    merged["status_changed"] = merged["status_old"] != merged["status_new"]
    return merged


def _summarize(frame: pd.DataFrame) -> dict[str, float | int | None]:
    runtime_deltas = frame["runtime_delta_ms"].dropna()
    size_deltas = frame["egraph_total_size_delta"].dropna()
    param_deltas = frame["param_reduction_delta"].dropna()
    rank_deltas = frame["simplified_rank_delta"].dropna()
    return {
        "examples": int(len(frame)),
        "status_changed": int(frame["status_changed"].sum()),
        "old_non_success": int((frame["status_old"] != "saturated").sum()),
        "new_non_success": int((frame["status_new"] != "saturated").sum()),
        "rendered_changed": int(frame["rendered_changed"].sum()),
        "faster": int((runtime_deltas < 0).sum()),
        "slower": int((runtime_deltas > 0).sum()),
        "same_runtime": int((runtime_deltas == 0).sum()),
        "smaller_egraph": int((size_deltas < 0).sum()),
        "larger_egraph": int((size_deltas > 0).sum()),
        "same_egraph": int((size_deltas == 0).sum()),
        "better_param_reduction": int((param_deltas > 0).sum()),
        "worse_param_reduction": int((param_deltas < 0).sum()),
        "same_param_reduction": int((param_deltas == 0).sum()),
        "better_rank": int((rank_deltas < 0).sum()),
        "worse_rank": int((rank_deltas > 0).sum()),
        "same_rank": int((rank_deltas == 0).sum()),
        "old_median_runtime_ms": _median(frame.loc[frame["runtime_ms_old"].notna() & frame["runtime_ms_new"].notna(), "runtime_ms_old"]),
        "new_median_runtime_ms": _median(frame.loc[frame["runtime_ms_old"].notna() & frame["runtime_ms_new"].notna(), "runtime_ms_new"]),
        "median_runtime_delta_ms": _median(runtime_deltas),
        "old_median_egraph_total_size": _median(
            frame.loc[frame["egraph_total_size_old"].notna() & frame["egraph_total_size_new"].notna(), "egraph_total_size_old"]
        ),
        "new_median_egraph_total_size": _median(
            frame.loc[frame["egraph_total_size_old"].notna() & frame["egraph_total_size_new"].notna(), "egraph_total_size_new"]
        ),
        "median_egraph_total_size_delta": _median(size_deltas),
        "old_median_param_reduction": _median(
            frame.loc[frame["param_reduction_old"].notna() & frame["param_reduction_new"].notna(), "param_reduction_old"]
        ),
        "new_median_param_reduction": _median(
            frame.loc[frame["param_reduction_old"].notna() & frame["param_reduction_new"].notna(), "param_reduction_new"]
        ),
        "median_param_reduction_delta": _median(param_deltas),
        "old_median_simplified_rank": _median(
            frame.loc[frame["simplified_rank_old"].notna() & frame["simplified_rank_new"].notna(), "simplified_rank_old"]
        ),
        "new_median_simplified_rank": _median(
            frame.loc[frame["simplified_rank_old"].notna() & frame["simplified_rank_new"].notna(), "simplified_rank_new"]
        ),
        "median_simplified_rank_delta": _median(rank_deltas),
    }


def _summary_table(summary: dict[str, float | int | None], old_label: str, new_label: str) -> Table:
    table = Table(title="High-Level Comparison")
    table.add_column("Metric")
    table.add_column(old_label, justify="right")
    table.add_column(new_label, justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Result")
    table.add_row(
        "Median runtime ms (lower is better)",
        _fmt_number(summary["old_median_runtime_ms"]),
        _fmt_compared_number(summary["new_median_runtime_ms"], summary["median_runtime_delta_ms"], lower_is_better=True),
        _fmt_delta(summary["median_runtime_delta_ms"], lower_is_better=True),
        _fmt_outcome(summary["median_runtime_delta_ms"], lower_is_better=True),
    )
    table.add_row(
        "Median e-graph size (lower is better)",
        _fmt_number(summary["old_median_egraph_total_size"]),
        _fmt_compared_number(
            summary["new_median_egraph_total_size"],
            summary["median_egraph_total_size_delta"],
            lower_is_better=True,
        ),
        _fmt_delta(summary["median_egraph_total_size_delta"], lower_is_better=True),
        _fmt_outcome(summary["median_egraph_total_size_delta"], lower_is_better=True),
    )
    table.add_row(
        "Median param reduction (higher is better)",
        _fmt_number(summary["old_median_param_reduction"]),
        _fmt_compared_number(
            summary["new_median_param_reduction"],
            summary["median_param_reduction_delta"],
            lower_is_better=False,
        ),
        _fmt_delta(summary["median_param_reduction_delta"], lower_is_better=False),
        _fmt_outcome(summary["median_param_reduction_delta"], lower_is_better=False),
    )
    table.add_row(
        "Median simplified rank (lower is better)",
        _fmt_number(summary["old_median_simplified_rank"]),
        _fmt_compared_number(
            summary["new_median_simplified_rank"],
            summary["median_simplified_rank_delta"],
            lower_is_better=True,
        ),
        _fmt_delta(summary["median_simplified_rank_delta"], lower_is_better=True),
        _fmt_outcome(summary["median_simplified_rank_delta"], lower_is_better=True),
    )
    return table


def _outcome_table(summary: dict[str, float | int | None]) -> Table:
    table = Table(title="Outcome Counts")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_row("Examples compared", str(summary["examples"]))
    table.add_row(
        "Runtime better / worse / same",
        f"[green]{summary['faster']}[/green] / [red]{summary['slower']}[/red] / [dim]{summary['same_runtime']}[/dim]",
    )
    table.add_row(
        "Total e-graph size better / worse / same",
        f"[green]{summary['smaller_egraph']}[/green] / [red]{summary['larger_egraph']}[/red] / [dim]{summary['same_egraph']}[/dim]",
    )
    table.add_row(
        "Param reduction better / worse / same",
        f"[green]{summary['better_param_reduction']}[/green] / [red]{summary['worse_param_reduction']}[/red] / [dim]{summary['same_param_reduction']}[/dim]",
    )
    table.add_row(
        "Simplified rank better / worse / same",
        f"[green]{summary['better_rank']}[/green] / [red]{summary['worse_rank']}[/red] / [dim]{summary['same_rank']}[/dim]",
    )
    table.add_row("Rendered output changed (inspect)", f"[yellow]{summary['rendered_changed']}[/yellow]")
    table.add_row("Status changed (inspect)", f"[yellow]{summary['status_changed']}[/yellow]")
    table.add_row("Non-saturated status rows", f"{summary['old_non_success']} old / {summary['new_non_success']} new")
    return table


def _status_counts_table(frame: pd.DataFrame, old_label: str, new_label: str) -> Table:
    old_counts = Counter(frame["status_old"])
    new_counts = Counter(frame["status_new"])
    table = Table(title="Status Counts")
    table.add_column("Status")
    table.add_column(old_label, justify="right")
    table.add_column(new_label, justify="right")
    for status in sorted(set(old_counts) | set(new_counts)):
        style = "green" if status == "saturated" else "yellow"
        table.add_row(f"[{style}]{status}[/{style}]", str(old_counts[status]), str(new_counts[status]))
    return table


def _gap_label(value: float | None) -> str:
    if value is None:
        return "[dim]na[/dim]"
    if value == int(value):
        value = int(value)
    if value == 0:
        return "0 (target)"
    return f"{abs(value):g} {'over target' if value > 0 else 'under target'}"


def _gap_counts_table(frame: pd.DataFrame, old_label: str, new_label: str) -> Table:
    old_values = [None if pd.isna(value) else float(value) for value in frame["simplified_rank_old"]]
    new_values = [None if pd.isna(value) else float(value) for value in frame["simplified_rank_new"]]
    old_counts = Counter(old_values)
    new_counts = Counter(new_values)
    keys = sorted(key for key in set(old_counts) | set(new_counts) if key is not None)
    if None in old_counts or None in new_counts:
        keys.append(None)
    table = Table(title="Simplified Parsed Param Gap Counts")
    table.add_column("after_parsed_n_params - target rank")
    table.add_column(old_label, justify="right")
    table.add_column(new_label, justify="right")
    table.add_column("count Δ", justify="right")
    for key in keys:
        delta = new_counts[key] - old_counts[key]
        style = "dim" if delta == 0 else ("green" if delta < 0 else "red")
        table.add_row(_gap_label(key), str(old_counts[key]), str(new_counts[key]), f"[{style}]{delta:+d}[/]")
    return table


def _sort_frame(frame: pd.DataFrame, sort: str) -> pd.DataFrame:
    sortable = frame.copy()
    sort_columns = {
        "row": ["row_id"],
        "runtime-delta": ["runtime_delta_ms", "row_id"],
        "abs-runtime-delta": ["abs_runtime_delta_ms", "row_id"],
        "size-delta": ["egraph_total_size_delta", "row_id"],
        "param-delta": ["param_reduction_delta", "row_id"],
        "rank-delta": ["simplified_rank_delta", "row_id"],
        "render-changed": ["rendered_changed", "row_id"],
    }
    sortable["abs_runtime_delta_ms"] = sortable["runtime_delta_ms"].abs()
    ascending = sort in {"row", "render-changed"}
    if sort == "render-changed":
        sortable["rendered_changed"] = ~sortable["rendered_changed"]
    return sortable.sort_values(sort_columns[sort], ascending=ascending, na_position="last")


def _details_table(frame: pd.DataFrame, old_label: str, new_label: str) -> Table:
    table = Table(title="Per-Example Comparison")
    table.add_column("Example", overflow="fold")
    table.add_column(f"{old_label} ms", justify="right")
    table.add_column(f"{new_label} ms", justify="right")
    table.add_column("time", justify="right")
    table.add_column("ms Δ", justify="right")
    table.add_column("% Δ", justify="right")
    table.add_column(f"{old_label} params red.", justify="right")
    table.add_column(f"{new_label} params red.", justify="right")
    table.add_column("params", justify="right")
    table.add_column("params Δ", justify="right")
    table.add_column(f"{old_label} rank", justify="right")
    table.add_column(f"{new_label} rank", justify="right")
    table.add_column("rank", justify="right")
    table.add_column("rank Δ", justify="right")
    table.add_column(f"{old_label} size", justify="right")
    table.add_column(f"{new_label} size", justify="right")
    table.add_column("size", justify="right")
    table.add_column("size Δ", justify="right")
    table.add_column("render/status")
    for row in frame.itertuples(index=False):
        render_status = _fmt_changed(bool(row.rendered_changed))
        if row.status_changed:
            render_status = f"{render_status} / [yellow]status[/yellow]"
        table.add_row(
            row.row_id,
            _fmt_number(row.runtime_ms_old),
            _fmt_compared_number(row.runtime_ms_new, row.runtime_delta_ms, lower_is_better=True),
            _fmt_outcome(row.runtime_delta_ms, lower_is_better=True),
            _fmt_delta(row.runtime_delta_ms, lower_is_better=True),
            _fmt_pct(row.runtime_delta_pct, lower_is_better=True),
            _fmt_number(row.param_reduction_old),
            _fmt_compared_number(row.param_reduction_new, row.param_reduction_delta, lower_is_better=False),
            _fmt_outcome(row.param_reduction_delta, lower_is_better=False),
            _fmt_delta(row.param_reduction_delta, lower_is_better=False),
            _fmt_number(row.simplified_rank_old),
            _fmt_compared_number(row.simplified_rank_new, row.simplified_rank_delta, lower_is_better=True),
            _fmt_outcome(row.simplified_rank_delta, lower_is_better=True),
            _fmt_delta(row.simplified_rank_delta, lower_is_better=True),
            _fmt_number(row.egraph_total_size_old),
            _fmt_compared_number(row.egraph_total_size_new, row.egraph_total_size_delta, lower_is_better=True),
            _fmt_outcome(row.egraph_total_size_delta, lower_is_better=True),
            _fmt_delta(row.egraph_total_size_delta, lower_is_better=True),
            render_status,
        )
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True, help="Old canonical corpus CSV.")
    parser.add_argument("--new", type=Path, required=True, help="New canonical corpus CSV.")
    parser.add_argument("--old-label", default=None)
    parser.add_argument("--new-label", default=None)
    parser.add_argument("--old-variant", choices=("baseline", "container"), default=None)
    parser.add_argument("--new-variant", choices=("baseline", "container"), default=None)
    parser.add_argument("--input-kind", choices=("both", "original", "sympy"), default="both")
    parser.add_argument("--limit", type=int, default=40, help="Maximum per-example rows to print. Use 0 for all.")
    parser.add_argument(
        "--sort",
        choices=("row", "runtime-delta", "abs-runtime-delta", "size-delta", "param-delta", "rank-delta", "render-changed"),
        default="row",
    )
    parser.add_argument("--width", type=int, default=180, help="Rich console width for wide comparison tables.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    old = _load_frame(args.old, args.input_kind, variant=args.old_variant)
    new = _load_frame(args.new, args.input_kind, variant=args.new_variant)
    old_label = args.old_label or _display_name(str(old.iloc[0]["implementation"]), str(old.iloc[0]["variant"]))
    new_label = args.new_label or _display_name(str(new.iloc[0]["implementation"]), str(new.iloc[0]["variant"]))
    compared = _comparison_frame(old, new)
    sorted_frame = _sort_frame(compared, args.sort)
    displayed = sorted_frame if args.limit == 0 else sorted_frame.head(args.limit)
    summary = _summarize(compared)

    console = Console(width=args.width)
    console.print(
        Panel.fit(
            "\n".join(
                (
                    f"[bold]old:[/bold] {args.old}",
                    f"[bold]new:[/bold] {args.new}",
                    f"[bold]rows:[/bold] {len(old)} old, {len(new)} new, {len(compared)} shared",
                    "Param reduction means before_params minus after_params; higher is better.",
                    "Simplified rank means after_parsed_n_params minus target rank; lower is better.",
                )
            ),
            title="param_eq Corpus Comparison",
        )
    )
    console.print(_summary_table(summary, old_label, new_label))
    console.print(_outcome_table(summary))
    console.print(_gap_counts_table(compared, old_label, new_label))
    console.print(_status_counts_table(compared, old_label, new_label))
    console.print(_details_table(displayed, old_label, new_label))
    if args.limit and len(sorted_frame) > args.limit:
        console.print(f"[dim]Showing {args.limit} of {len(sorted_frame)} examples. Pass --limit 0 to show all.[/dim]")


if __name__ == "__main__":
    main()
