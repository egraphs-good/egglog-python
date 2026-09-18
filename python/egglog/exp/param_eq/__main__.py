"""Simplify one expression with the experimental Param-Eq pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .domain import binary_to_containers, parse_expression
from .pipeline import run_paper_pipeline, run_paper_pipeline_container


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single-expression JSON CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expr", required=True, help="Python-like symbolic expression")
    parser.add_argument("--variant", choices=("binary", "container"), default="binary")
    args = parser.parse_args(argv)

    report = (
        run_paper_pipeline(parse_expression(args.expr))
        if args.variant == "binary"
        else run_paper_pipeline_container(binary_to_containers(parse_expression(args.expr)))
    )
    print(json.dumps({"variant": args.variant, **asdict(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
