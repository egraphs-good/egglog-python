from __future__ import annotations

import argparse
from collections.abc import Iterable

from egglog.exp.pandoc_symreg import (
    PipelineReport,
    Witness,
    build_sanity_witnesses,
    count_float_params,
    run_binary_pipeline,
    run_multiset_pipeline,
    selected_witnesses,
)


def witness_catalog() -> dict[str, Witness]:
    sanity_1, sanity_2 = build_sanity_witnesses()
    readable, dramatic, pysr_stress = selected_witnesses()
    return {
        sanity_1.name: sanity_1,
        sanity_2.name: sanity_2,
        readable.name: readable,
        dramatic.name: dramatic,
        pysr_stress.name: pysr_stress,
    }


def _format_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0])
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = "\n".join(" | ".join(str(row[header]).ljust(widths[header]) for header in headers) for row in rows)
    return f"{header_line}\n{separator}\n{body}"


def _report_rows(witness: Witness, reports: Iterable[PipelineReport]) -> list[dict[str, str]]:
    before_params = count_float_params(witness.expr)
    rows: list[dict[str, str]] = []
    for report in reports:
        metrics = report.metric_report
        rows.append({
            "mode": report.mode,
            "before_params": str(before_params),
            "after_params": str(metrics.parameter_count),
            "ratio": f"{metrics.parameter_reduction_ratio:.3f}",
            "rank_gap": str(metrics.parameter_count - metrics.jacobian_rank),
            "cost": str(report.cost),
            "total_size": str(report.total_size),
            "time_sec": f"{report.total_sec:.4f}",
            "max_abs_error": f"{report.numeric_max_abs_error:.3g}",
        })
    return rows


def _print_report(witness: Witness, reports: list[PipelineReport]) -> None:
    print(f"\n== {witness.name} ==")
    print(f"Source: {witness.source_path}:{witness.row}")
    print(f"Why this witness: {witness.description}")
    print(_format_table(_report_rows(witness, reports)))
    for report in reports:
        print(f"{report.mode} extracted: {report.python_source}")
        if report.notes:
            for note in report.notes:
                print(f"{report.mode} note: {note}")


def _core_witnesses() -> list[str]:
    return ["erro-1", "readable", "dramatic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the pandoc-symreg binary EqSat pipeline against the multiset variant."
    )
    parser.add_argument(
        "--witness",
        choices=["all", "erro-1", "erro-2", "readable", "dramatic", "pysr-stress"],
        default="all",
        help="Which witness to run. 'all' runs the core set used in the tutorial.",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "multiset", "both"],
        default="both",
        help="Which pipeline(s) to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = witness_catalog()
    witness_names = _core_witnesses() if args.witness == "all" else [args.witness]

    print("Pandoc-symreg EqSat replication in egglog")
    print("Binary mode reproduces the copied pandoc-symreg rule pipeline.")
    print("Multiset mode is a hypothesis test: it ports A/C structure to containers, then reruns the binary rules.")

    for name in witness_names:
        witness = catalog[name]
        reports: list[PipelineReport] = []
        if args.mode in {"binary", "both"}:
            reports.append(run_binary_pipeline(witness))
        if args.mode in {"multiset", "both"}:
            reports.append(run_multiset_pipeline(witness))
        _print_report(witness, reports)


if __name__ == "__main__":
    main()
