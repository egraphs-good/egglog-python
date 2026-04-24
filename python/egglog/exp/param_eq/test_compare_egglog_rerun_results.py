from __future__ import annotations

import pandas as pd

from egglog.exp.param_eq import compare_egglog_rerun_results as compare
from egglog.exp.param_eq.original_results import RESULT_KEY_COLUMNS


def _row(raw_index: int, *, n_rank: float | None = 2.0) -> dict[str, object]:
    return {
        "dataset": "pagie",
        "raw_index": raw_index,
        "algorithm_raw": "Bingo",
        "algo_row": raw_index + 1,
        "input_kind": "original" if raw_index % 2 == 0 else "sympy",
        "algorithm": "Bingo",
        "n_rank": n_rank,
        "source_orig_parsed_expr": f"x0 + {raw_index}.5",
        "source_orig_parsed_n_params": float(raw_index + 1),
        "source_before_nodes": float(raw_index + 3),
    }


def _result(
    source: pd.DataFrame, raw_index: int, *, status: str = "saturated", after_params: float | None
) -> dict[str, object]:
    row = source[source["raw_index"] == raw_index].iloc[0]
    n_rank = row["n_rank"]
    return {
        **{key: row[key] for key in RESULT_KEY_COLUMNS},
        "status": status,
        "after_params": after_params,
        "after_rank_difference": None if after_params is None or pd.isna(n_rank) else after_params - float(n_rank),
        "after_parsed_rank_difference": None
        if after_params is None or pd.isna(n_rank)
        else after_params - float(n_rank),
        "simpl_parsed_n_params": after_params,
        "runtime_ms": 1.0,
        "peak_rss_mb": 2.0,
        "passes": 1.0,
        "egraph_total_size": 10.0,
        "rendered": f"x0 + {raw_index}.5",
    }


def _misses(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["status"] == "saturated") & (frame["after_rank_difference"] > 0)].copy()


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.DataFrame.from_records([_row(i, n_rank=None if i == 4 else 2.0) for i in range(8)])
    before = pd.DataFrame.from_records([
        _result(source, 0, after_params=3.0),
        _result(source, 1, after_params=3.0),
        _result(source, 2, after_params=2.0),
        _result(source, 3, after_params=2.0),
        _result(source, 4, after_params=3.0),
        _result(source, 5, after_params=2.0),
        _result(source, 6, after_params=3.0),
        _result(source, 7, after_params=4.0),
    ])
    after = pd.DataFrame.from_records([
        _result(source, 0, after_params=2.0),
        _result(source, 1, after_params=3.0),
        _result(source, 2, after_params=3.0),
        _result(source, 3, status="timeout", after_params=None),
        _result(source, 4, after_params=3.0),
        _result(source, 5, after_params=3.0),
        _result(source, 6, after_params=1.0),
        _result(source, 7, after_params=3.0),
    ])
    review = pd.DataFrame.from_records([
        {
            **{key: source[source["raw_index"] == 1].iloc[0][key] for key in RESULT_KEY_COLUMNS},
            "agent_status": "done",
            "conclusion_kind": "new_rule_reduces_to_rank",
            "new_rule_family": "exp_additive_constant_extraction",
            "proposed_rule": "exp(c + x) -> exp(c) * exp(x)",
            "proof_summary": "requires a rule family that was added",
            "remaining_gap_notes": "still missed after rerun",
        },
        {
            **{key: source[source["raw_index"] == 7].iloc[0][key] for key in RESULT_KEY_COLUMNS},
            "agent_status": "done",
            "conclusion_kind": "new_rule_reduces_to_rank",
            "new_rule_family": "coefficient_lattice_factoring",
            "proposed_rule": "factor related coefficients",
            "proof_summary": "longer run improves this row",
            "remaining_gap_notes": "probe should override taxonomy",
        },
    ])
    probe = pd.DataFrame.from_records([
        {
            **{key: source[source["raw_index"] == 7].iloc[0][key] for key in RESULT_KEY_COLUMNS},
            "probe_status": "ok",
            "probe_after_params": 2.5,
            "probe_rank_gap": 0.5,
            "improved": True,
            "reached_rank": False,
            "probe_runtime_ms": 12.0,
            "probe_peak_rss_mb": 50.0,
            "param_delta": 0.5,
        }
    ])
    return source, before, after, review, probe


def test_build_rerun_delta_classifies_core_outcomes() -> None:
    source, before, after, review, probe = _fixture_frames()

    delta = compare.build_rerun_delta_frame(
        source,
        before,
        after,
        before_misses=_misses(before),
        after_misses=_misses(after),
        review=review,
        longer_probe=probe,
    ).set_index("raw_index")

    assert bool(delta.loc[0, "fixed_by_extra_rules"]) is True
    assert delta.loc[0, "outcome"] == "fixed_by_extra_rules"
    assert bool(delta.loc[1, "still_missed"]) is True
    assert delta.loc[1, "remaining_issue"] == "rule_added_but_not_matching"
    assert bool(delta.loc[2, "newly_missed"]) is True
    assert delta.loc[2, "outcome"] == "newly_missed"
    assert bool(delta.loc[3, "execution_issue"]) is True
    assert delta.loc[3, "outcome"] == "execution_issue"
    assert bool(delta.loc[4, "missing_rank"]) is True
    assert delta.loc[4, "outcome"] == "missing_rank"
    assert bool(delta.loc[5, "regressed_params"]) is True
    assert bool(delta.loc[6, "under_rank"]) is True
    assert bool(delta.loc[6, "fixed_by_extra_rules"]) is True
    assert delta.loc[6, "outcome"] == "under_rank"
    assert delta.loc[7, "remaining_issue"] == "existing_rules_more_iterations"


def test_remaining_rank_misses_are_sorted_by_smallest_expression() -> None:
    source, before, after, review, probe = _fixture_frames()

    delta = compare.build_rerun_delta_frame(
        source,
        before,
        after,
        before_misses=_misses(before),
        after_misses=_misses(after),
        review=review,
        longer_probe=probe,
    )
    remaining = compare.remaining_rank_misses_frame(delta)

    assert list(remaining["raw_index"]) == [1, 7]


def test_write_comparison_outputs_creates_csvs_and_report(tmp_path) -> None:
    source, before, after, review, probe = _fixture_frames()
    delta = compare.build_rerun_delta_frame(
        source,
        before,
        after,
        before_misses=_misses(before),
        after_misses=_misses(after),
        review=review,
        longer_probe=probe,
    )

    remaining, report = compare.write_comparison_outputs(
        delta,
        delta_path=tmp_path / "delta.csv",
        remaining_path=tmp_path / "remaining.csv",
        report_path=tmp_path / "report.md",
    )

    assert len(pd.read_csv(tmp_path / "delta.csv")) == len(delta)
    assert list(pd.read_csv(tmp_path / "remaining.csv")["raw_index"]) == list(remaining["raw_index"])
    assert "Fixed by extra rules: `2`" in report
    assert "Still missed: `2`" in report
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == report
