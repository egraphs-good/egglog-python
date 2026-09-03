"""
Expression hoisting must be deterministic across processes.

`_exprs_multiple_parents` used to traverse the expression DAG with a set of
``TypedExprDecl`` objects and ``.pop()`` from it, so the traversal order --
and with it the order in which shared subterms were hoisted into
``$__expr_N`` let-bindings -- depended on id()-based hashing (memory
addresses).  Identical programs were therefore lowered into different
(equivalent) e-graphs in different processes, making ``serialize()`` output
nondeterministic.
"""

import subprocess
import sys

SCRIPT = """
from __future__ import annotations

import hashlib

from egglog import EGraph, Expr, StringLike


class B(Expr):
    @classmethod
    def var(cls, name: StringLike) -> B: ...

    def __and__(self, o: B) -> B: ...

    def __or__(self, o: B) -> B: ...

    def __invert__(self) -> B: ...


eg = EGraph()
x, y = B.var("x"), B.var("y")
shared = x & y
expr = shared | y
for _ in range(24):
    expr = (shared & expr) | (expr & (shared | x))
eg.let("$e", expr)
print(hashlib.sha256(eg._serialize().to_json().encode()).hexdigest())
"""


def test_serialization_is_deterministic_across_processes() -> None:
    hashes = []
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            capture_output=True,
            text=True,
            check=True,
        )
        hashes.append(proc.stdout.strip())
    assert len(set(hashes)) == 1, f"nondeterministic serialization: {hashes}"
