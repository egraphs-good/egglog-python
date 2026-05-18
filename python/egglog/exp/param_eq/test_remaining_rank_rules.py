from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from egglog import *

from .domain import *
from .pipeline import *

KEY_COLUMNS = ["dataset", "raw_index", "algorithm_raw", "algo_row", "input_kind"]

IMPLEMENTED_CANDIDATES = {
    ("kotanchek", 98, "GOMEA", 9, "original"),
    ("pagie", 127, "Operon", 8, "original"),
    ("pagie", 30, "EPLEX", 1, "original"),
    ("pagie", 30, "EPLEX", 1, "sympy"),
    ("kotanchek", 40, "EPLEX", 11, "original"),
    ("kotanchek", 190, "SRjl", 11, "original"),
    ("pagie", 92, "GOMEA", 3, "original"),
    ("pagie", 96, "GOMEA", 7, "original"),
    ("pagie", 100, "GOMEA", 11, "original"),
    ("pagie", 109, "GOMEA", 20, "original"),
    ("pagie", 111, "GOMEA", 22, "original"),
    ("kotanchek", 188, "SRjl", 9, "sympy"),
    ("pagie", 42, "EPLEX", 13, "original"),
    ("kotanchek", 29, "Bingo", 29, "sympy"),
    ("pagie", 110, "GOMEA", 21, "original"),
    ("pagie", 203, "SRjl", 24, "sympy"),
    ("kotanchek", 41, "EPLEX", 12, "original"),
    ("kotanchek", 125, "Operon", 6, "sympy"),
    ("kotanchek", 195, "SRjl", 16, "sympy"),
    ("pagie", 39, "EPLEX", 10, "sympy"),
    ("kotanchek", 206, "SRjl", 27, "original"),
}

DEFERRED_CANDIDATES = {
    ("kotanchek", 124, "Operon", 5, "sympy"),
    ("kotanchek", 166, "SBP", 17, "sympy"),
    ("kotanchek", 193, "SRjl", 14, "original"),
    ("kotanchek", 203, "SRjl", 24, "sympy"),
    ("kotanchek", 205, "SRjl", 26, "sympy"),
    ("kotanchek", 208, "SRjl", 29, "sympy"),
    ("pagie", 51, "EPLEX", 22, "sympy"),
    ("pagie", 163, "SBP", 13, "sympy"),
    ("kotanchek", 184, "SRjl", 5, "original"),
    ("kotanchek", 21, "Bingo", 21, "original"),
    ("kotanchek", 21, "Bingo", 21, "sympy"),
    ("pagie", 55, "EPLEX", 26, "original"),
    ("pagie", 5, "Bingo", 6, "sympy"),
    ("pagie", 120, "Operon", 1, "sympy"),
}


def _artifact_dir() -> Path:
    return Path(__file__).parent / "artifacts"


def _current_rank_reducing_rows() -> pd.DataFrame:
    review = pd.read_csv(_artifact_dir() / "egglog_rank_miss_agent_review_current.csv")
    rows = review[review["conclusion_kind"].eq("new_rule_reduces_to_rank")].copy()
    candidate_keys = IMPLEMENTED_CANDIDATES | DEFERRED_CANDIDATES
    return rows[rows.apply(lambda row: _key_tuple(row) in candidate_keys, axis=1)].copy()


def _key_tuple(row: pd.Series) -> tuple[str, int, str, int, str]:
    return (
        str(row["dataset"]),
        int(row["raw_index"]),
        str(row["algorithm_raw"]),
        int(row["algo_row"]),
        str(row["input_kind"]),
    )


def _assert_rewrite_equivalent(source: str, target: str) -> None:
    egraph = EGraph()
    source_expr = parse_expression(source)
    target_expr = parse_expression(target)
    egraph.register(source_expr, target_expr, Num(0.0))
    egraph.run(binary_analysis_schedule)
    for _ in range(8):
        egraph.run(run(binary_rewrite_ruleset))
        egraph.run(binary_analysis_schedule)
    egraph.check(eq(source_expr).to(target_expr))


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            "1.7 * (3.1 - 4.2 / exp(x0) - 5.3 / exp(x1))",
            "(1.7 * 4.2) * (3.1 / 4.2 - 1.0 / exp(x0) - (5.3 / 4.2) / exp(x1))",
        ),
        ("2.7 / (3.4 / x0 + 5.6)", "(2.7 / 3.4) / (1.0 / x0 + 5.6 / 3.4)"),
        ("((x0 + 1.2) * (x0 + 3.4)) + 5.6", "x0 * (x0 + 4.6) + 9.68"),
        ("(x0 * (x1 * x2 - x3)) / (x2 * x4)", "x0 * (x1 - x3 / x2) / x4"),
        (
            "2.0 + log(abs(3.0 / x0 + (x0 / 4.0) ** 3.0))",
            "log(abs((exp(2.0) * 3.0) / x0 + (exp(2.0) / (4.0 ** 3.0)) * (x0 ** 3.0)))",
        ),
        ("x0 ** 4.0 / ((x1 * x0 ** 2.0 + x2) ** 2.0)", "1.0 / ((x1 + x2 / x0 ** 2.0) ** 2.0)"),
        ("x0 * -0.9999999", "-x0"),
    ],
)
def test_remaining_rule_schema_proofs(source: str, target: str) -> None:
    _assert_rewrite_equivalent(source, target)


def test_current_rank_reducing_candidate_partition_is_explicit() -> None:
    current_keys = {_key_tuple(row) for _, row in _current_rank_reducing_rows().iterrows()}
    assert current_keys == IMPLEMENTED_CANDIDATES | DEFERRED_CANDIDATES
    assert IMPLEMENTED_CANDIDATES.isdisjoint(DEFERRED_CANDIDATES)
    assert len(IMPLEMENTED_CANDIDATES) == 21
    assert len(DEFERRED_CANDIDATES) == 14


@pytest.mark.skipif(
    os.environ.get("PARAM_EQ_SLOW_RULE_TESTS") != "1",
    reason="set PARAM_EQ_SLOW_RULE_TESTS=1 to run current corpus-row rank canaries",
)
@pytest.mark.parametrize("candidate_key", sorted(IMPLEMENTED_CANDIDATES))
def test_implemented_current_rank_reducing_rows_reach_rank(candidate_key: tuple[str, int, str, int, str]) -> None:
    rows = _current_rank_reducing_rows()
    row = rows[
        (rows["dataset"] == candidate_key[0])
        & (rows["raw_index"] == candidate_key[1])
        & (rows["algorithm_raw"] == candidate_key[2])
        & (rows["algo_row"] == candidate_key[3])
        & (rows["input_kind"] == candidate_key[4])
    ].iloc[0]
    result = run_paper_pipeline(parse_expression(str(row["source_orig_parsed_expr"])))
    assert result.extracted_params <= int(row["n_rank"]), result.extracted
