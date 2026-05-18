import warnings

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()
# alt.renderers.enable("default")

warnings.filterwarnings(
    "ignore",
    message="Automatically deduplicated selection parameter",
    category=UserWarning,
)

OUTCOME_DOMAIN = ["container better", "same", "binary better"]
OUTCOME_RANGE = ["#1f77b4", "#9e9e9e", "#d62728"]
OUTCOME_ORDER = {outcome: i for i, outcome in enumerate(OUTCOME_DOMAIN)}

VARIANT_DOMAIN = ["binary", "container"]
VARIANT_RANGE = ["#c73a35", "#2f78b7"]

RUNTIME_ABS_TOL_SEC = 0.005
RUNTIME_REL_TOL = 0.05
MEANINGFUL_RATIO_BAND = 1.25
MAX_DELTA_BARS = None
TOP_OUTLIERS = 10
USE_LOG_SCATTER_SCALES = False
TEXT_TRUNCATE = 180


def classify_delta(delta, *, lower_is_better=True, tolerance=0.0):
    if pd.isna(delta):
        return np.nan
    tol = 0.0 if pd.isna(tolerance) else float(tolerance)
    if abs(float(delta)) <= tol:
        return "same"
    container_wins = delta < 0 if lower_is_better else delta > 0
    return "container better" if container_wins else "binary better"


def safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where((denominator > 0) & denominator.notna() & numerator.notna(), numerator / denominator, np.nan)


def geo_mean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    values = values[values > 0]
    return np.exp(np.log(values).mean()) if len(values) else np.nan


def fmt_num(value, digits=3):
    if pd.isna(value):
        return "na"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def fmt_delta(value, digits=3):
    if pd.isna(value):
        return "na"
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer():
        return f"{int(value):+d}"
    return f"{value:+.{digits}f}"


def fmt_ratio(value):
    if pd.isna(value):
        return "na"
    return f"{float(value):.3f}x"


def truncate_text(value, limit=TEXT_TRUNCATE):
    if pd.isna(value):
        return "na"
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " ..."

def plot_dashboard(df):

    # ---------------------------------------------------------------------
    # Pair alternating binary/container rows.
    # ---------------------------------------------------------------------

    work_df = df.copy()
    work_df["variant"] = work_df["variant"].replace({"baseline": "binary"})

    value_columns = [
        "expr",
        "extracted",
        "passes",
        "total_sec",
        "total_size",
        "before_nodes",
        "before_params",
        "after_nodes",
        "after_params",
        "extracted_cost",
        "cost",
    ]
    value_columns = [col for col in value_columns if col in work_df.columns]

    paired = (
        work_df
        .reset_index(drop=True)
        .assign(pair_id=lambda d: d.index // 2)
        .pivot_table(index="pair_id", columns="variant", values=value_columns, aggfunc="first")
    )

    paired.columns = [f"{col}_{variant}" for col, variant in paired.columns.to_flat_index()]
    paired = paired.reset_index(drop=True)

    required_columns = [
        "expr_binary",
        "expr_container",
        "total_sec_binary",
        "total_sec_container",
        "total_size_binary",
        "total_size_container",
        "after_params_binary",
        "after_params_container",
    ]
    missing = [col for col in required_columns if col not in paired.columns]
    if missing:
        raise ValueError(f"Missing paired columns: {missing}")

    paired["expr"] = paired["expr_binary"].fillna(paired["expr_container"])
    paired["expr_id"] = paired.index.astype(int)
    paired["expr_label"] = [f"expr {i}" for i in paired["expr_id"]]
    paired["expr_short"] = paired["expr"].map(lambda value: truncate_text(value, 80))
    paired["before_nodes"] = paired["before_nodes_binary"].fillna(paired["before_nodes_container"])

    paired["runtime_delta_sec"] = paired["total_sec_container"] - paired["total_sec_binary"]
    paired["runtime_ratio"] = safe_ratio(paired["total_sec_container"], paired["total_sec_binary"])
    paired["runtime_log10_ratio"] = np.log10(pd.to_numeric(paired["runtime_ratio"], errors="coerce"))
    paired["runtime_tolerance_sec"] = np.maximum(
        RUNTIME_ABS_TOL_SEC,
        RUNTIME_REL_TOL * paired["total_sec_binary"].abs(),
    )
    paired["runtime_outcome"] = [
        classify_delta(delta, lower_is_better=True, tolerance=tol)
        for delta, tol in zip(paired["runtime_delta_sec"], paired["runtime_tolerance_sec"], strict=True)
    ]

    paired["egraph_size_delta"] = paired["total_size_container"] - paired["total_size_binary"]
    paired["egraph_size_ratio"] = safe_ratio(paired["total_size_container"], paired["total_size_binary"])
    paired["egraph_size_log10_ratio"] = np.log10(pd.to_numeric(paired["egraph_size_ratio"], errors="coerce"))
    paired["egraph_size_outcome"] = [classify_delta(delta, lower_is_better=True) for delta in paired["egraph_size_delta"]]

    paired["after_params_delta"] = paired["after_params_container"] - paired["after_params_binary"]
    paired["after_params_ratio"] = safe_ratio(paired["after_params_container"], paired["after_params_binary"])
    paired["after_params_outcome"] = [classify_delta(delta, lower_is_better=True) for delta in paired["after_params_delta"]]

    cost_base = None
    if {"extracted_cost_binary", "extracted_cost_container"}.issubset(paired.columns):
        cost_base = "extracted_cost"
    elif {"cost_binary", "cost_container"}.issubset(paired.columns):
        cost_base = "cost"

    if cost_base is not None:
        paired["cost_delta"] = paired[f"{cost_base}_container"] - paired[f"{cost_base}_binary"]
        paired["cost_ratio"] = safe_ratio(paired[f"{cost_base}_container"], paired[f"{cost_base}_binary"])
        paired["cost_log10_ratio"] = np.log10(pd.to_numeric(paired["cost_ratio"], errors="coerce"))
        paired["cost_outcome"] = [classify_delta(delta, lower_is_better=True) for delta in paired["cost_delta"]]

    metric_specs = [
        {
            "key": "runtime",
            "label": "Runtime sec",
            "binary": "total_sec_binary",
            "container": "total_sec_container",
            "delta": "runtime_delta_sec",
            "ratio": "runtime_ratio",
            "outcome": "runtime_outcome",
            "digits": 3,
            "lower_is_better": True,
        },
        {
            "key": "egraph_size",
            "label": "E-graph size",
            "binary": "total_size_binary",
            "container": "total_size_container",
            "delta": "egraph_size_delta",
            "ratio": "egraph_size_ratio",
            "outcome": "egraph_size_outcome",
            "digits": 1,
            "lower_is_better": True,
        },
        {
            "key": "after_params",
            "label": "After params",
            "binary": "after_params_binary",
            "container": "after_params_container",
            "delta": "after_params_delta",
            "ratio": "after_params_ratio",
            "outcome": "after_params_outcome",
            "digits": 1,
            "lower_is_better": True,
        },
    ]

    if cost_base is not None:
        metric_specs.append({
            "key": "cost",
            "label": "Cost",
            "binary": f"{cost_base}_binary",
            "container": f"{cost_base}_container",
            "delta": "cost_delta",
            "ratio": "cost_ratio",
            "outcome": "cost_outcome",
            "digits": 1,
            "lower_is_better": True,
        })


    # ---------------------------------------------------------------------
    # Summary frames.
    # ---------------------------------------------------------------------

    summary_rows = []
    outcome_rows = []
    ratio_rows = []
    distribution_rows = []

    for order, spec in enumerate(metric_specs):
        valid = paired.dropna(subset=[spec["binary"], spec["container"]]).copy()
        counts = valid[spec["outcome"]].value_counts().reindex(OUTCOME_DOMAIN, fill_value=0)

        ratio_values = pd.to_numeric(valid[spec["ratio"]], errors="coerce").dropna()
        ratio_values = ratio_values[ratio_values > 0]

        summary_rows.append({
            "metric_order": order,
            "metric": spec["label"],
            "n": len(valid),
            "binary_median": fmt_num(valid[spec["binary"]].median(), spec["digits"]),
            "container_median": fmt_num(valid[spec["container"]].median(), spec["digits"]),
            "median_delta": fmt_delta(valid[spec["delta"]].median(), spec["digits"]),
            "median_ratio": fmt_ratio(ratio_values.median()) if not ratio_values.empty else "na",
            "geo_mean_ratio": fmt_ratio(geo_mean(ratio_values)),
            "c_same_b": f"{counts['container better']} / {counts['same']} / {counts['binary better']}",
        })

        for outcome in OUTCOME_DOMAIN:
            outcome_rows.append({
                "metric_order": order,
                "metric": spec["label"],
                "outcome": outcome,
                "outcome_order": OUTCOME_ORDER[outcome],
                "count": int(counts[outcome]),
            })

        if not ratio_values.empty:
            ratio_rows.append({
                "metric_order": order,
                "metric": spec["label"],
                "p10": ratio_values.quantile(0.10),
                "p25": ratio_values.quantile(0.25),
                "p50": ratio_values.quantile(0.50),
                "p75": ratio_values.quantile(0.75),
                "p90": ratio_values.quantile(0.90),
                "geo_mean": geo_mean(ratio_values),
                "container_25pct_faster": int((ratio_values < 1 / MEANINGFUL_RATIO_BAND).sum()),
                "near_same": int(
                    ((ratio_values >= 1 / MEANINGFUL_RATIO_BAND) & (ratio_values <= MEANINGFUL_RATIO_BAND)).sum()
                ),
                "binary_25pct_faster": int((ratio_values > MEANINGFUL_RATIO_BAND).sum()),
            })

        for variant, col in [("binary", spec["binary"]), ("container", spec["container"])]:
            values = pd.to_numeric(valid[col], errors="coerce").dropna()
            if values.empty:
                continue
            distribution_rows.append({
                "metric_order": order,
                "metric": spec["label"],
                "variant": variant,
                "variant_order": VARIANT_DOMAIN.index(variant),
                "n": len(values),
                "median": fmt_num(values.quantile(0.50), spec["digits"]),
                "p75": fmt_num(values.quantile(0.75), spec["digits"]),
                "p90": fmt_num(values.quantile(0.90), spec["digits"]),
                "p95": fmt_num(values.quantile(0.95), spec["digits"]),
                "max": fmt_num(values.max(), spec["digits"]),
                "total": fmt_num(values.sum(), spec["digits"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    outcome_counts = pd.DataFrame(outcome_rows)
    ratio_stats = pd.DataFrame(ratio_rows)
    distribution_stats = pd.DataFrame(distribution_rows)

    runtime_distribution = pd.concat(
        [
            paired.assign(variant="binary", variant_order=0, runtime_sec=paired["total_sec_binary"]),
            paired.assign(variant="container", variant_order=1, runtime_sec=paired["total_sec_container"]),
        ],
        ignore_index=True,
    ).dropna(subset=["runtime_sec"])
    runtime_distribution = runtime_distribution[runtime_distribution["runtime_sec"] > 0].copy()

    runtime_ecdf_parts = []
    for variant, group in runtime_distribution.groupby("variant", sort=False):
        group = group.sort_values("runtime_sec").copy()
        group["cdf"] = np.arange(1, len(group) + 1) / len(group)
        runtime_ecdf_parts.append(group)
    runtime_ecdf = pd.concat(runtime_ecdf_parts, ignore_index=True) if runtime_ecdf_parts else pd.DataFrame()

    runtime_pair_long = pd.concat(
        [
            paired.assign(variant="binary", variant_order=0, runtime_sec=paired["total_sec_binary"]),
            paired.assign(variant="container", variant_order=1, runtime_sec=paired["total_sec_container"]),
        ],
        ignore_index=True,
    ).dropna(subset=["runtime_sec"])
    runtime_pair_long = runtime_pair_long[runtime_pair_long["runtime_sec"] > 0].copy()


    # ---------------------------------------------------------------------
    # Shared selections and chart helpers.
    # ---------------------------------------------------------------------

    hover = alt.selection_point(fields=["expr_id"], on="mouseover", clear="mouseout", empty=False, name="hover_expr")
    click = alt.selection_point(fields=["expr_id"], on="click", clear="dblclick", empty=False, name="click_expr")


    def text_table(table_df, columns, *, title, width=1180, row_height=24, font_size=12):
        table_df = table_df.reset_index(drop=True).copy()
        x_positions = np.linspace(0, width - 160, len(columns))
        body_records = []
        header_records = []

        for x, (column, label) in zip(x_positions, columns, strict=True):
            header_records.append({"row": -1, "x": x, "text": label})
            if column not in table_df.columns:
                table_df[column] = "na"
            for row_idx, value in enumerate(table_df[column].astype(str)):
                body_records.append({"row": row_idx, "x": x, "text": value})

        y_sort = [-1, *range(len(table_df))]
        header = (
            alt
            .Chart(pd.DataFrame(header_records))
            .mark_text(align="left", baseline="middle", fontSize=font_size, fontWeight="bold")
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, width])),
                y=alt.Y("row:O", axis=None, sort=y_sort),
                text="text:N",
            )
        )
        body = (
            alt
            .Chart(pd.DataFrame(body_records))
            .mark_text(align="left", baseline="middle", fontSize=font_size)
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, width])),
                y=alt.Y("row:O", axis=None, sort=y_sort),
                text="text:N",
            )
        )
        return (header + body).properties(width=width, height=row_height * (len(table_df) + 1), title=title)


    def scatter_with_identity(frame, x_col, y_col, outcome_col, x_title, y_title, title):
        plot_df = frame.dropna(subset=[x_col, y_col]).copy()
        if plot_df.empty:
            return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point().properties(width=560, height=320, title=title)

        lo = min(plot_df[x_col].min(), plot_df[y_col].min())
        hi = max(plot_df[x_col].max(), plot_df[y_col].max())
        if lo == hi:
            hi = lo + 1.0

        scale_type = "log" if USE_LOG_SCATTER_SCALES and lo > 0 else "linear"
        scale = alt.Scale(type=scale_type)

        identity = (
            alt
            .Chart(pd.DataFrame({"v": [lo, hi]}))
            .mark_line(strokeDash=[6, 4], color="black", opacity=0.65)
            .encode(x=alt.X("v:Q", scale=scale), y=alt.Y("v:Q", scale=scale))
        )

        tooltip = [
            alt.Tooltip("expr_label:N", title="id"),
            alt.Tooltip("expr:N", title="expression"),
            alt.Tooltip(f"{outcome_col}:N", title="metric result"),
            alt.Tooltip(f"{x_col}:Q", title=x_title, format=".4f"),
            alt.Tooltip(f"{y_col}:Q", title=y_title, format=".4f"),
            alt.Tooltip("after_params_delta:Q", title="after params delta"),
            alt.Tooltip("runtime_ratio:Q", title="runtime ratio", format=".3f"),
            alt.Tooltip("egraph_size_ratio:Q", title="egraph size ratio", format=".3f"),
        ]

        base = (
            alt
            .Chart(plot_df)
            .mark_circle(size=65, opacity=0.28)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_title, scale=scale),
                y=alt.Y(f"{y_col}:Q", title=y_title, scale=scale),
                color=alt.Color(
                    f"{outcome_col}:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), legend=None
                ),
                tooltip=tooltip,
            )
        )

        hover_focus = (
            alt
            .Chart(plot_df)
            .transform_filter(hover)
            .mark_circle(size=170, opacity=0.95, stroke="black", strokeWidth=1.5)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=scale),
                y=alt.Y(f"{y_col}:Q", scale=scale),
                color=alt.Color(
                    f"{outcome_col}:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), legend=None
                ),
            )
        )

        click_focus = (
            alt
            .Chart(plot_df)
            .transform_filter(click)
            .mark_circle(size=240, opacity=1.0, stroke="black", strokeWidth=2.5)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=scale),
                y=alt.Y(f"{y_col}:Q", scale=scale),
                color=alt.Color(
                    f"{outcome_col}:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), legend=None
                ),
            )
        )

        return (identity + base + hover_focus + click_focus).properties(width=560, height=320, title=title)


    # ---------------------------------------------------------------------
    # Summary charts.
    # ---------------------------------------------------------------------

    scoreboard = text_table(
        summary_df,
        [
            ("metric", "metric"),
            ("n", "n"),
            ("binary_median", "binary median"),
            ("container_median", "container median"),
            ("median_delta", "median delta"),
            ("median_ratio", "median ratio"),
            ("geo_mean_ratio", "geo mean ratio"),
            ("c_same_b", "container / same / binary"),
        ],
        title="Paired Scoreboard",
    )

    tail_scoreboard = text_table(
        distribution_stats.sort_values(["metric_order", "variant_order"]),
        [
            ("metric", "metric"),
            ("variant", "variant"),
            ("median", "median"),
            ("p75", "p75"),
            ("p90", "p90"),
            ("p95", "p95"),
            ("max", "max"),
            ("total", "total"),
        ],
        title="Distribution Tail Stats by Variant",
    )

    ratio_stats_display = ratio_stats.copy()
    if not ratio_stats_display.empty:
        for col in ["p10", "p25", "p50", "p75", "p90", "geo_mean"]:
            ratio_stats_display[col] = ratio_stats_display[col].map(fmt_ratio)
        ratio_stats_display["threshold_counts"] = (
            ratio_stats["container_25pct_faster"].astype(str)
            + " / "
            + ratio_stats["near_same"].astype(str)
            + " / "
            + ratio_stats["binary_25pct_faster"].astype(str)
        )

    ratio_scoreboard = text_table(
        ratio_stats_display.sort_values("metric_order") if not ratio_stats_display.empty else pd.DataFrame(),
        [
            ("metric", "metric"),
            ("p10", "p10"),
            ("p25", "p25"),
            ("p50", "median"),
            ("p75", "p75"),
            ("p90", "p90"),
            ("geo_mean", "geo mean"),
            ("threshold_counts", ">1.25x C / near / >1.25x B"),
        ],
        title="Paired Ratio Stats (container / binary)",
    )

    outcome_bar = (
        alt
        .Chart(outcome_counts)
        .mark_bar()
        .encode(
            y=alt.Y("metric:N", sort=list(summary_df["metric"]), title=None),
            x=alt.X("count:Q", stack="zero", title="rows"),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), title="result"),
            order=alt.Order("outcome_order:Q"),
            tooltip=["metric:N", "outcome:N", "count:Q"],
        )
        .properties(width=560, height=150, title="Outcome Counts by Metric")
    )

    outcome_text = (
        alt
        .Chart(outcome_counts[outcome_counts["count"] > 0])
        .mark_text(color="white", fontSize=12, fontWeight="bold")
        .encode(
            y=alt.Y("metric:N", sort=list(summary_df["metric"]), title=None),
            x=alt.X("count:Q", stack="zero"),
            detail="outcome:N",
            text="count:Q",
            order=alt.Order("outcome_order:Q"),
        )
    )

    outcome_chart = outcome_bar + outcome_text


    # ---------------------------------------------------------------------
    # Runtime distribution charts.
    # ---------------------------------------------------------------------

    runtime_ecdf_chart = (
        alt
        .Chart(runtime_ecdf)
        .mark_line(size=2)
        .encode(
            x=alt.X("runtime_sec:Q", title="runtime sec (log)", scale=alt.Scale(type="log")),
            y=alt.Y("cdf:Q", title="fraction of rows <= runtime", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("variant:N", scale=alt.Scale(domain=VARIANT_DOMAIN, range=VARIANT_RANGE), title="variant"),
            tooltip=[
                "variant:N",
                alt.Tooltip("runtime_sec:Q", title="runtime sec", format=".4f"),
                alt.Tooltip("cdf:Q", title="fraction", format=".3f"),
            ],
        )
        .properties(width=380, height=280, title="Runtime ECDF (left/up is faster)")
    )

    runtime_slope_base = (
        alt
        .Chart(runtime_pair_long)
        .mark_line(opacity=0.08, color="#555")
        .encode(
            x=alt.X("variant:N", sort=VARIANT_DOMAIN, title=None),
            y=alt.Y("runtime_sec:Q", title="runtime sec (log)", scale=alt.Scale(type="log")),
            detail="expr_id:N",
        )
    )

    runtime_slope_points = (
        alt
        .Chart(runtime_pair_long)
        .mark_circle(size=25, opacity=0.25)
        .encode(
            x=alt.X("variant:N", sort=VARIANT_DOMAIN, title=None),
            y=alt.Y("runtime_sec:Q", scale=alt.Scale(type="log")),
            color=alt.Color("runtime_outcome:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), legend=None),
            tooltip=[
                alt.Tooltip("expr_label:N", title="id"),
                alt.Tooltip("expr:N", title="expression"),
                "variant:N",
                alt.Tooltip("runtime_sec:Q", title="runtime sec", format=".4f"),
                alt.Tooltip("runtime_ratio:Q", title="container / binary", format=".3f"),
                alt.Tooltip("runtime_outcome:N", title="runtime result"),
            ],
        )
    )

    runtime_slope_hover = (
        alt
        .Chart(runtime_pair_long)
        .transform_filter(hover)
        .mark_line(size=3, opacity=1, color="black")
        .encode(
            x=alt.X("variant:N", sort=VARIANT_DOMAIN, title=None),
            y=alt.Y("runtime_sec:Q", scale=alt.Scale(type="log")),
            detail="expr_id:N",
        )
    )

    runtime_slope_click = (
        alt
        .Chart(runtime_pair_long)
        .transform_filter(click)
        .mark_line(size=5, opacity=1, color="black")
        .encode(
            x=alt.X("variant:N", sort=VARIANT_DOMAIN, title=None),
            y=alt.Y("runtime_sec:Q", scale=alt.Scale(type="log")),
            detail="expr_id:N",
        )
    )

    runtime_slope_chart = (
        runtime_slope_base + runtime_slope_points + runtime_slope_hover + runtime_slope_click
    ).properties(width=380, height=280, title="Paired Runtime Lines")

    ratio_hist_df = paired.dropna(subset=["runtime_log10_ratio"]).copy()
    runtime_ratio_hist = (
        alt
        .Chart(ratio_hist_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("runtime_log10_ratio:Q", bin=alt.Bin(maxbins=36), title="log10(container / binary runtime)"),
            y=alt.Y("count():Q", title="rows"),
            color=alt.Color(
                "runtime_outcome:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), title="result"
            ),
            tooltip=[
                alt.Tooltip("count():Q", title="rows"),
                alt.Tooltip("runtime_outcome:N", title="result"),
            ],
        )
        .properties(width=380, height=280, title="Runtime Ratio Histogram")
    )

    ratio_zero_line = (
        alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[6, 4], color="black", opacity=0.7).encode(x="x:Q")
    )

    runtime_ratio_hist = runtime_ratio_hist + ratio_zero_line

    runtime_distribution_panel = alt.hconcat(
        runtime_ecdf_chart,
        runtime_slope_chart,
        runtime_ratio_hist,
    ).resolve_scale(color="independent")

    runtime_by_input_base = (
        alt
        .Chart(runtime_pair_long)
        .mark_line(opacity=0.45, strokeWidth=2)
        .encode(
            x=alt.X("before_nodes:Q", title="input expression size (before nodes)"),
            y=alt.Y("runtime_sec:Q", title="runtime sec"),
            color=alt.Color(
                "variant:N",
                scale=alt.Scale(domain=VARIANT_DOMAIN, range=VARIANT_RANGE),
                title="variant",
            ),
            detail="variant:N",
            tooltip=[
                alt.Tooltip("variant:N", title="variant"),
                alt.Tooltip("expr_label:N", title="id"),
                alt.Tooltip("before_nodes:Q", title="before nodes"),
                alt.Tooltip("runtime_sec:Q", title="runtime sec", format=".4f"),
                alt.Tooltip("runtime_ratio:Q", title="container / binary", format=".3f"),
                alt.Tooltip("after_params_delta:Q", title="after params delta"),
                alt.Tooltip("expr:N", title="expression"),
            ],
        )
    )

    runtime_by_input_points = (
        alt
        .Chart(runtime_pair_long)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("before_nodes:Q", title="input expression size (before nodes)"),
            y=alt.Y("runtime_sec:Q", title="runtime sec"),
            color=alt.Color(
                "variant:N",
                scale=alt.Scale(domain=VARIANT_DOMAIN, range=VARIANT_RANGE),
                title="variant",
            ),
            tooltip=[
                alt.Tooltip("variant:N", title="variant"),
                alt.Tooltip("expr_label:N", title="id"),
                alt.Tooltip("before_nodes:Q", title="before nodes"),
                alt.Tooltip("runtime_sec:Q", title="runtime sec", format=".4f"),
                alt.Tooltip("runtime_ratio:Q", title="container / binary", format=".3f"),
                alt.Tooltip("after_params_delta:Q", title="after params delta"),
                alt.Tooltip("expr:N", title="expression"),
            ],
        )
    )

    runtime_by_input_hover = (
        alt
        .Chart(runtime_pair_long)
        .transform_filter(hover)
        .mark_circle(size=170, opacity=0.95, stroke="black", strokeWidth=1.5)
        .encode(
            x="before_nodes:Q",
            y="runtime_sec:Q",
            color=alt.Color(
                "variant:N",
                scale=alt.Scale(domain=VARIANT_DOMAIN, range=VARIANT_RANGE),
                legend=None,
            ),
        )
    )

    runtime_by_input_click = (
        alt
        .Chart(runtime_pair_long)
        .transform_filter(click)
        .mark_circle(size=240, opacity=1.0, stroke="black", strokeWidth=2.5)
        .encode(
            x="before_nodes:Q",
            y="runtime_sec:Q",
            color=alt.Color(
                "variant:N",
                scale=alt.Scale(domain=VARIANT_DOMAIN, range=VARIANT_RANGE),
                legend=None,
            ),
        )
    )

    runtime_by_input_chart = (
        runtime_by_input_base
        + runtime_by_input_points
        + runtime_by_input_hover
        + runtime_by_input_click
    ).properties(
        width=1180,
        height=340,
        title="Runtime vs Input Size (Before Nodes)",
    )


    # ---------------------------------------------------------------------
    # Ratio percentile chart.
    # ---------------------------------------------------------------------

    if ratio_stats.empty:
        ratio_chart = (
            alt
            .Chart(pd.DataFrame({"x": [], "y": []}))
            .mark_point()
            .properties(width=560, height=150, title="Ratio Percentiles")
        )
    else:
        ratio_scale = alt.Scale(type="log")
        ratio_sort = list(ratio_stats.sort_values("metric_order")["metric"])

        ratio_whisker = (
            alt
            .Chart(ratio_stats)
            .mark_rule(size=2)
            .encode(
                y=alt.Y("metric:N", sort=ratio_sort, title=None),
                x=alt.X("p10:Q", title="container / binary ratio", scale=ratio_scale),
                x2="p90:Q",
                color=alt.value("#666"),
                tooltip=[
                    "metric:N",
                    alt.Tooltip("p10:Q", format=".3f"),
                    alt.Tooltip("p50:Q", format=".3f"),
                    alt.Tooltip("p90:Q", format=".3f"),
                ],
            )
        )

        ratio_iqr = (
            alt
            .Chart(ratio_stats)
            .mark_bar(height=14, opacity=0.55, color="#6baed6")
            .encode(
                y=alt.Y("metric:N", sort=ratio_sort, title=None),
                x=alt.X("p25:Q", scale=ratio_scale),
                x2="p75:Q",
            )
        )

        ratio_median = (
            alt
            .Chart(ratio_stats)
            .mark_tick(thickness=3, size=26, color="black")
            .encode(
                y=alt.Y("metric:N", sort=ratio_sort, title=None),
                x=alt.X("p50:Q", scale=ratio_scale),
                tooltip=["metric:N", alt.Tooltip("p50:Q", title="median ratio", format=".3f")],
            )
        )

        ratio_one_line = (
            alt
            .Chart(pd.DataFrame({"ratio": [1.0]}))
            .mark_rule(strokeDash=[6, 4], color="black", opacity=0.65)
            .encode(x=alt.X("ratio:Q", scale=ratio_scale))
        )

        ratio_chart = (ratio_whisker + ratio_iqr + ratio_median + ratio_one_line).properties(
            width=560,
            height=150,
            title="Ratio Percentiles (p10-p90, box=p25-p75, tick=median)",
        )


    # ---------------------------------------------------------------------
    # After-param delta bars and interactive scatters.
    # ---------------------------------------------------------------------

    params_bar_df = (
        paired
        .dropna(subset=["after_params_binary", "after_params_container"])
        .sort_values(["after_params_delta", "expr_id"], ascending=[True, True])
        .reset_index(drop=True)
    )

    if MAX_DELTA_BARS is not None and len(params_bar_df) > MAX_DELTA_BARS:
        left = params_bar_df.head(MAX_DELTA_BARS // 2)
        right = params_bar_df.tail(MAX_DELTA_BARS - len(left))
        params_bar_df = pd.concat([left, right], ignore_index=True)

    params_bar_df["delta_rank"] = np.arange(1, len(params_bar_df) + 1)

    bar_tooltip = [
        alt.Tooltip("delta_rank:O", title="sorted rank"),
        alt.Tooltip("expr_label:N", title="id"),
        alt.Tooltip("expr:N", title="expression"),
        alt.Tooltip("after_params_binary:Q", title="binary after params"),
        alt.Tooltip("after_params_container:Q", title="container after params"),
        alt.Tooltip("after_params_delta:Q", title="delta"),
        alt.Tooltip("runtime_ratio:Q", title="runtime ratio", format=".3f"),
        alt.Tooltip("egraph_size_ratio:Q", title="size ratio", format=".3f"),
    ]

    bars_base = (
        alt
        .Chart(params_bar_df)
        .mark_bar(opacity=0.42)
        .encode(
            x=alt.X(
                "delta_rank:O", title="expressions sorted by after-param delta", axis=alt.Axis(labels=False, ticks=False)
            ),
            y=alt.Y("after_params_delta:Q", title="container after params - binary after params"),
            color=alt.Color(
                "after_params_outcome:N",
                scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE),
                legend=alt.Legend(title="result"),
            ),
            tooltip=bar_tooltip,
        )
    )

    bars_hover = (
        alt
        .Chart(params_bar_df)
        .transform_filter(hover)
        .mark_bar(opacity=0.95, stroke="black", strokeWidth=1.5)
        .encode(
            x="delta_rank:O",
            y="after_params_delta:Q",
            color=alt.Color(
                "after_params_outcome:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), legend=None
            ),
        )
    )

    bars_click = (
        alt
        .Chart(params_bar_df)
        .transform_filter(click)
        .mark_bar(opacity=1.0, stroke="black", strokeWidth=2.5)
        .encode(
            x="delta_rank:O",
            y="after_params_delta:Q",
            color=alt.Color(
                "after_params_outcome:N", scale=alt.Scale(domain=OUTCOME_DOMAIN, range=OUTCOME_RANGE), legend=None
            ),
        )
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black", strokeDash=[6, 4], opacity=0.7).encode(y="y:Q")

    params_chart = (bars_base + bars_hover + bars_click + zero_line).properties(
        width=1180,
        height=300,
        title="After Params Delta by Expression (negative favors containers)",
    )

    runtime_scatter = scatter_with_identity(
        paired,
        "total_sec_binary",
        "total_sec_container",
        "runtime_outcome",
        "Binary runtime (sec)",
        "Container runtime (sec)",
        "Runtime: Binary vs Container",
    )

    size_scatter = scatter_with_identity(
        paired,
        "total_size_binary",
        "total_size_container",
        "egraph_size_outcome",
        "Binary e-graph size",
        "Container e-graph size",
        "E-Graph Size: Binary vs Container",
    )


    # ---------------------------------------------------------------------
    # Details and outliers.
    # ---------------------------------------------------------------------

    available_detail_fields = [
        ("expr", "expression"),
        ("before_nodes", "input before nodes"),
        ("extracted_binary", "binary extracted"),
        ("extracted_container", "container extracted"),
        ("after_params_binary", "binary after params"),
        ("after_params_container", "container after params"),
        ("after_params_delta", "delta after params"),
        ("total_sec_binary", "binary runtime sec"),
        ("total_sec_container", "container runtime sec"),
        ("runtime_delta_sec", "runtime delta sec"),
        ("runtime_ratio", "runtime ratio"),
        ("total_size_binary", "binary egraph size"),
        ("total_size_container", "container egraph size"),
        ("egraph_size_delta", "egraph size delta"),
        ("egraph_size_ratio", "egraph size ratio"),
        ("passes_binary", "binary passes"),
        ("passes_container", "container passes"),
    ]

    if cost_base is not None:
        available_detail_fields.extend([
            (f"{cost_base}_binary", "binary cost"),
            (f"{cost_base}_container", "container cost"),
            ("cost_delta", "cost delta"),
            ("cost_ratio", "cost ratio"),
        ])

    available_detail_fields = [(field, label) for field, label in available_detail_fields if field in paired.columns]

    detail_long = (
        paired
        .loc[:, ["expr_id", "expr_label"] + [field for field, _ in available_detail_fields]]
        .copy()
        .melt(id_vars=["expr_id", "expr_label"], var_name="field", value_name="value")
    )

    detail_long["field_label"] = detail_long["field"].map(dict(available_detail_fields))
    detail_long["field_order"] = detail_long["field"].map({
        field: i for i, (field, _) in enumerate(available_detail_fields)
    })
    detail_long["value_label"] = [truncate_text(value) for value in detail_long["value"]]

    instruction = (
        alt
        .Chart(
            pd.DataFrame({"text": ["Hover to preview across charts. Click to pin one row. Double-click a chart to clear."]})
        )
        .mark_text(align="left", fontSize=13, color="#444")
        .encode(text="text:N")
        .properties(width=1180, height=24)
    )

    detail_base = alt.Chart(detail_long).transform_filter(click)

    detail_keys = detail_base.mark_text(align="left", fontWeight="bold", fontSize=12).encode(
        y=alt.Y("field_order:O", axis=None),
        x=alt.value(0),
        text="field_label:N",
    )

    detail_vals = detail_base.mark_text(align="left", fontSize=12).encode(
        y=alt.Y("field_order:O", axis=None),
        x=alt.value(250),
        text="value_label:N",
    )

    details_chart = (detail_keys + detail_vals).properties(
        width=1180,
        height=22 * max(1, len(available_detail_fields)),
        title="Selected Details",
    )

    runtime_bad = (
        paired
        .dropna(subset=["runtime_ratio"])
        .sort_values(["runtime_ratio", "expr_id"], ascending=[False, True])
        .head(TOP_OUTLIERS)
        .assign(reason="runtime slowdown")
    )

    runtime_good = (
        paired
        .dropna(subset=["runtime_ratio"])
        .sort_values(["runtime_ratio", "expr_id"], ascending=[True, True])
        .head(TOP_OUTLIERS)
        .assign(reason="runtime speedup")
    )

    params_bad = (
        paired
        .dropna(subset=["after_params_delta"])
        .query("after_params_delta > 0")
        .sort_values(["after_params_delta", "expr_id"], ascending=[False, True])
        .head(TOP_OUTLIERS)
        .assign(reason="after-param regression")
    )

    params_good = (
        paired
        .dropna(subset=["after_params_delta"])
        .query("after_params_delta < 0")
        .sort_values(["after_params_delta", "expr_id"], ascending=[True, True])
        .head(TOP_OUTLIERS)
        .assign(reason="after-param improvement")
    )

    outliers = (
        pd
        .concat([runtime_bad, runtime_good, params_bad, params_good], ignore_index=True)
        .drop_duplicates("expr_id")
        .head(TOP_OUTLIERS)
        .copy()
    )

    outliers["runtime_ratio_label"] = outliers["runtime_ratio"].map(fmt_ratio)
    outliers["size_ratio_label"] = outliers["egraph_size_ratio"].map(fmt_ratio)
    outliers["after_params_delta_label"] = outliers["after_params_delta"].map(lambda value: fmt_delta(value, 1))
    outliers["expr_short"] = outliers["expr"].map(lambda value: truncate_text(value, 70))

    outlier_table = text_table(
        outliers.reset_index(drop=True),
        [
            ("reason", "reason"),
            ("expr_label", "id"),
            ("runtime_ratio_label", "runtime ratio"),
            ("size_ratio_label", "size ratio"),
            ("after_params_delta_label", "params delta"),
            ("expr_short", "expression"),
        ],
        title="Top Outliers to Inspect",
        width=1180,
        row_height=24,
        font_size=11,
    )


    # ---------------------------------------------------------------------
    # Dashboard.
    # ---------------------------------------------------------------------

    return (
        alt
        .vconcat(
            scoreboard,
            tail_scoreboard,
            ratio_scoreboard,
            alt.hconcat(outcome_chart, ratio_chart).resolve_scale(x="independent", y="independent", color="independent"),
            runtime_distribution_panel,
            runtime_by_input_chart,
            params_chart,
            alt.hconcat(runtime_scatter, size_scatter).resolve_scale(x="independent", y="independent", color="independent"),
            instruction,
            details_chart,
            outlier_table,
        )
        .add_params(hover, click)
        .resolve_scale(color="independent")
    )
