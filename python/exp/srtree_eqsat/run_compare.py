from __future__ import annotations

from egglog.exp.srtree_eqsat import HASKELL_REFERENCE_ROWS, compare_to_haskell, core_examples, run_baseline_pipeline


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
        baseline = run_baseline_pipeline(
            example.expr,
            node_cutoff=50_000,
            iteration_limit=12,
            input_names=example.input_names,
            sample_points=example.sample_points,
        )
        comparison = compare_to_haskell(example.name, baseline, HASKELL_REFERENCE_ROWS[example.row])
        rows.append({
            "example": example.name,
            "egglog_stop": baseline.stop_reason,
            "egglog_params": f"{baseline.metric_report.before_parameter_count}->{baseline.metric_report.after_parameter_count}",
            "haskell_params": f"{HASKELL_REFERENCE_ROWS[example.row].before_parameter_count}->{HASKELL_REFERENCE_ROWS[example.row].after_parameter_count}",
            "optimal_params": str(baseline.metric_report.optimal_parameter_count),
            "egglog_size": str(baseline.total_size),
            "egglog_time": f"{baseline.total_sec:.4f}",
            "haskell_time": f"{HASKELL_REFERENCE_ROWS[example.row].runtime_sec:.4f}",
            "notes": "; ".join(comparison.notes) or "-",
        })
    print(_table(rows))


if __name__ == "__main__":
    main()
