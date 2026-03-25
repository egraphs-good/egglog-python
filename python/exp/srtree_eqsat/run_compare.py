from __future__ import annotations

from egglog.exp.srtree_eqsat import core_examples, run_baseline_pipeline, run_multiset_pipeline


def _table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0])
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    header = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = "\n".join(" | ".join(row[header].ljust(widths[header]) for header in headers) for row in rows)
    return f"{header}\n{separator}\n{body}"


def main() -> None:
    rows: list[dict[str, str]] = []
    for example in core_examples():
        baseline = run_baseline_pipeline(example.expr, node_cutoff=50_000, iteration_limit=12)
        multiset = run_multiset_pipeline(
            example.expr, saturate_without_limits=False, node_cutoff=50_000, iteration_limit=2
        )
        rows.append({
            "example": example.name,
            "baseline_stop": baseline.stop_reason,
            "baseline_params": f"{baseline.metric_report.before_parameter_count}->{baseline.metric_report.after_parameter_count}",
            "baseline_size": str(baseline.total_size),
            "baseline_time": f"{baseline.total_sec:.4f}",
            "multiset_stop": multiset.stop_reason,
            "multiset_params": f"{multiset.metric_report.before_parameter_count}->{multiset.metric_report.after_parameter_count}",
            "multiset_size": str(multiset.total_size),
            "multiset_time": f"{multiset.total_sec:.4f}",
        })
    print(_table(rows))


if __name__ == "__main__":
    main()
