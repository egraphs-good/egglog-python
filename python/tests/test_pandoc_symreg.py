from __future__ import annotations

from pathlib import Path

import pytest

from egglog.exp.pandoc_symreg import (
    build_sanity_witnesses,
    run_binary_pipeline,
    run_multiset_pipeline,
    selected_witnesses,
)


def test_erro_1_replicates_parameter_reduction() -> None:
    erro_1, _ = build_sanity_witnesses()
    binary = run_binary_pipeline(erro_1)
    multiset = run_multiset_pipeline(erro_1)

    assert binary.metric_report.parameter_count == 2
    assert binary.metric_report.parameter_reduction_ratio == pytest.approx(0.5)
    assert binary.numeric_max_abs_error == pytest.approx(0.0)

    assert multiset.metric_report.parameter_count == 2
    assert multiset.metric_report.parameter_reduction_ratio == pytest.approx(0.5)
    assert multiset.total_size < binary.total_size
    assert multiset.numeric_max_abs_error == pytest.approx(0.0)


def test_readable_and_dramatic_multisets_reduce_size() -> None:
    readable, dramatic, _ = selected_witnesses()
    for witness in (readable, dramatic):
        binary = run_binary_pipeline(witness)
        multiset = run_multiset_pipeline(witness)

        assert multiset.total_size < binary.total_size
        assert multiset.cost == binary.cost
        assert multiset.numeric_max_abs_error == pytest.approx(0.0)


def test_no_direct_egraph_saturate_calls() -> None:
    source = Path("/Users/saul/p/egg-smol-python/python/egglog/exp/pandoc_symreg.py").read_text()
    assert "egraph.saturate(" not in source
