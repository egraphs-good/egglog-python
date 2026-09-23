# mypy: disable-error-code="empty-body"
"""
Export a Python-authored program for Rust or another frontend.
============================================================

Run ``python -m egglog.examples.shared_program [output.json]``. Without an
output path, the versioned JSON is printed to standard output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from egglog import EGraph, Expr, SharedProgram, eq, i64, i64Like, rule, ruleset, run, union, vars_


class Math(Expr):
    def __init__(self, value: i64Like) -> None: ...

    def __add__(self, other: Math) -> Math: ...


def build_program() -> SharedProgram:
    """Construct, execute, and export an addition rule using the public Python API."""
    x, y = vars_("x y", i64)
    (result,) = vars_("result", Math)
    addition = ruleset(
        rule(eq(Math(x) + Math(y)).to(result), name="add-numbers").then(union(result).with_(Math(x + y))),
        name="arithmetic",
    )
    graph = EGraph(record_program=True)
    total = graph.let("total", Math(2) + Math(3))
    graph.run(run(addition).saturate())
    graph.check(eq(total).to(Math(5)))
    return graph.recorded_program


# The documentation gallery executes examples as notebook cells, without __file__.
if __name__ == "__main__" and "__file__" in globals():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    serialized = build_program().to_json() + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)
